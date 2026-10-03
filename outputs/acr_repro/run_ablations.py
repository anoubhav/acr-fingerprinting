"""Frozen ACR ablations on a public source-disjoint protocol.

Caches extraction independently of PCA. Linear mel work is shared for window
and transform variants, and only one complete variant index is held at a time.
Run with the same protocol/cache as run_acr_protocol.py. The output JSON uses
the primary runner's prediction schema; ablation names are recorded explicitly.
"""
from __future__ import annotations
import argparse
from collections import Counter
import concurrent.futures
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from acr_fp import FingerprintConfig, Fingerprinter, ExactIndex, load_audio, ablation_configs, audio_decoder_info
from run_acr_protocol import fingerprint_key, query_crop, extract_ref

VARIANT_NAMES = ("no_standardization", "no_deltas", "fp16_roundtrip", "no_pca",
                 "pca16", "window16", "window48", "stride2")
REEXTRACT = ("no_standardization", "window16", "window48")


def initialize_worker():
    # macOS multiprocessing uses spawn; BLAS limits set only in the parent
    # would not reliably bound extraction workers' independent thread pools.
    from threadpoolctl import threadpool_limits
    threadpool_limits(1, user_api="blas")


def source_key(path):
    source = Path(path)
    return hashlib.sha256((str(source.resolve()) + str(source.stat().st_size)).encode()).hexdigest()[:20]


def ref_path(cache, reference, config):
    return Path(cache) / f"{fingerprint_key(config)}_{source_key(reference['path'])}.npz"


def query_path(cache, query, config):
    key = hashlib.sha256(json.dumps({k: query[k] for k in ("path", "start_s", "duration_s")},
                                   sort_keys=True).encode()).hexdigest()[:24]
    return Path(cache) / f"query_{fingerprint_key(config)}_{key}.npz"


def atomic_npz(path, **arrays):
    temp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def bundle_reference(task):
    ref, cache, names = task
    variants = ablation_configs()
    needed = [name for name in names if not ref_path(cache, ref, variants[name]).exists()]
    if not needed:
        return ref["reference_id"], False
    started = time.perf_counter()
    audio, sr = load_audio(ref["path"], 8000)
    decode_s = time.perf_counter() - started
    default = Fingerprinter()
    started = time.perf_counter()
    mel = default.mel_spectrogram(audio, sr)
    spectral_s = time.perf_counter() - started
    for name in needed:
        model = Fingerprinter(variants[name])
        started = time.perf_counter()
        x, t = model.raw_features_from_mels(mel)
        extraction_s = spectral_s + time.perf_counter() - started
        atomic_npz(ref_path(cache, ref, model.config), features=x, times=t,
                   duration_s=len(audio)/sr, decode_s=decode_s, extraction_s=extraction_s)
    return ref["reference_id"], True


def transform_default_raw(x, t, name):
    """Only transformations whose required information exists in the base cache."""
    if name == "no_deltas":
        return x[:, :32].copy(), t.copy()
    if name == "fp16_roundtrip":
        return x.astype(np.float16).astype(np.float32), t.copy()
    if name == "stride2":
        frames = np.rint((t - FingerprintConfig().first_center_seconds) / (186/8000)).astype(np.int64)
        keep = frames % 2 == 0
        return x[keep], t[keep]
    if name in ("pca16", "no_pca", "default"):
        return x.copy(), t.copy()
    raise ValueError(f"Cannot derive {name} from standardized default rows")


def read_reference(reference, name, base_cache, ablation_cache, config):
    if name in REEXTRACT:
        p = ref_path(ablation_cache, reference, config)
    else:
        p = ref_path(base_cache, reference, FingerprintConfig())
    with np.load(p, allow_pickle=False) as z:
        x, t = z["features"].copy(), z["times"].copy()
        duration, extraction_s = float(z["duration_s"]), float(z["extraction_s"])
    if name not in REEXTRACT:
        x, t = transform_default_raw(x, t, name)
    return x, t, duration, extraction_s


def ensure_query_bundle(query, base_cache, ablation_cache):
    configs = ablation_configs()
    base_config = FingerprintConfig()
    needed = [name for name in ("default", *REEXTRACT)
              if not query_path(base_cache if name == "default" else ablation_cache,
                                query, configs[name]).exists()]
    if not needed:
        return
    audio, sr = load_audio(query["path"], 8000)
    start = round(query["start_s"] * sr)
    y = audio[start:start + round(query["duration_s"] * sr)]
    if len(y) < round(query["duration_s"] * sr) - 5:
        raise ValueError(f"Short decoded query {query['query_id']}")
    default = Fingerprinter(base_config)
    started = time.perf_counter()
    mel = default.mel_spectrogram(y, sr)
    spectral_s = time.perf_counter() - started
    for name in needed:
        model = Fingerprinter(configs[name])
        started = time.perf_counter()
        x, t = model.raw_features_from_mels(mel)
        extraction_s = spectral_s + time.perf_counter() - started
        p = query_path(base_cache if name == "default" else ablation_cache, query, model.config)
        atomic_npz(p, features=x, times=t, extraction_s=extraction_s)


def read_query(query, name, model, base_cache, ablation_cache):
    cache = ablation_cache if name in REEXTRACT else base_cache
    config = model.config if name in REEXTRACT else FingerprintConfig()
    with np.load(query_path(cache, query, config), allow_pickle=False) as z:
        x, t, extraction_s = z["features"].copy(), z["times"].copy(), float(z["extraction_s"])
    if name not in REEXTRACT:
        x, t = transform_default_raw(x, t, name)
    return model.transform(x), t, extraction_s


def summarize_integrity(protocol, references, queries, model, predictions):
    ids = [p["query_id"] for p in predictions]
    if len(ids) != len(set(ids)) or len(ids) != len(queries):
        raise ValueError("Prediction completeness/uniqueness failed")
    fit = set(model.fit_content_ids)
    gallery = {r["reference_id"] for r in references if r["role"] != "pca_fit"}
    unknown = {q["reference_id"] for q in queries if q["role"].endswith("unknown")}
    if fit & gallery or gallery & unknown:
        raise ValueError("Source split or unknown-gallery integrity failed")
    return {"expected_queries": len(queries), "predictions": len(predictions),
            "unique_query_ids": len(set(ids)), "fit_gallery_overlap": len(fit & gallery),
            "unknown_gallery_overlap": len(gallery & unknown),
            "empty_query_features": sum(p["query_fingerprints"] == 0 for p in predictions),
            "query_role_counts": dict(Counter(p["role"] for p in predictions))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANT_NAMES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--factors", default="1,8")
    parser.add_argument("--dense-test-known-limit", type=int)
    parser.add_argument("--dense-known-only", action="store_true")
    a = parser.parse_args()
    from threadpoolctl import threadpool_limits
    import faiss
    threadpool_limits(1, user_api="blas")
    faiss.omp_set_num_threads(a.threads)
    protocol = json.loads(a.protocol.read_text())
    protocol_hash = hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    configs = ablation_configs()
    variants = a.variants.split(",")
    if any(name not in configs for name in variants):
        raise ValueError("Unknown prespecified variant")
    for path in (a.base_cache, a.cache, a.output):
        path.mkdir(parents=True, exist_ok=True)
    references = [r for r in protocol["references"] if r["role"] in
                  ("pca_fit", "calibration_known", "test_known")]
    fit_refs = [r for r in references if r["role"] == "pca_fit"]
    gallery = [r for r in references if r["role"] != "pca_fit"]
    queries = [query_crop(q, a.duration) for q in protocol["queries"] if q["evaluate"]]
    started = time.perf_counter()
    # Produce independent variant caches first so a simultaneous primary run
    # can populate the shared default cache without duplicated extraction.
    needed_names = [name for name in REEXTRACT if name in variants]
    if needed_names:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, initializer=initialize_worker) as pool:
            tasks = [(r, str(a.cache), needed_names) for r in references]
            for i, _ in enumerate(pool.map(bundle_reference, tasks)):
                if (i+1) % 100 == 0:
                    print(f"Shared mel cache {i+1}/{len(references)}", flush=True)
    # Missing default rows can be filled independently. Atomic ablation writes
    # avoid reading incomplete files if the run is interrupted and resumed.
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, initializer=initialize_worker) as pool:
        for i, _ in enumerate(pool.map(extract_ref,
                    [(r, asdict(FingerprintConfig()), str(a.base_cache)) for r in references])):
            if (i+1) % 100 == 0:
                print(f"Default cache verified {i+1}/{len(references)}", flush=True)
    reference_cache_wall_s = time.perf_counter() - started
    for i, query in enumerate(queries):
        ensure_query_bundle(query, a.base_cache, a.cache)
        if (i+1) % 100 == 0:
            print(f"Query cache {i+1}/{len(queries)}", flush=True)

    factors = [int(x) for x in a.factors.split(",")]
    dense_ids = None
    if a.dense_test_known_limit is not None:
        known = [q for q in queries if q["role"] == "test_known"]
        known.sort(key=lambda q: hashlib.sha256(("acr-dense-ablation-v1:" +
                  q["query_id"].rsplit("@", 1)[0]).encode()).hexdigest())
        dense_ids = {q["query_id"] for q in known[:a.dense_test_known_limit]}
    for name in variants:
        if all((a.output / f"{name}_d{factor}_{a.duration:g}s.json").exists() for factor in factors):
            print(f"Results complete for {name}, skipping", flush=True)
            continue
        config = configs[name]
        model = Fingerprinter(config)
        rng = np.random.default_rng(20261002)
        def fit_batches():
            for reference in fit_refs:
                x, _, _, _ = read_reference(reference, name, a.base_cache, a.cache, config)
                if len(x) > 2000:
                    x = x[np.sort(rng.choice(len(x), 2000, replace=False))]
                yield x
        model.fit(fit_batches(), content_ids=[r["reference_id"] for r in fit_refs])
        model.assert_disjoint([r["reference_id"] for r in gallery], [q["reference_id"] for q in queries])
        model.save(a.output / f"{name}_pca_model.npz")
        entries, duration_total, extraction_total = [], 0., 0.
        for reference in gallery:
            x, t, duration, extraction_s = read_reference(reference, name, a.base_cache, a.cache, config)
            entries.append((reference["reference_id"], model.transform(x), t))
            duration_total += duration
            extraction_total += extraction_s
        query_features = [read_query(q, name, model, a.base_cache, a.cache) for q in queries]
        for factor in factors:
            out = a.output / f"{name}_d{factor}_{a.duration:g}s.json"
            if out.exists():
                continue
            started = time.perf_counter()
            index = ExactIndex.from_entries(entries, factor=factor,
                grid_step_sec=config.hop_length*config.stride_frames/config.sample_rate,
                grid_origin_sec=config.first_center_seconds)
            stats = index.bytes()
            stats.update(index_build_s=time.perf_counter()-started,
                reference_audio_duration_s=duration_total, reference_count=len(gallery),
                reference_extraction_work_s=extraction_total,
                extraction_wall_s=reference_cache_wall_s,
                pca_fitted_fingerprints=model.fit_fingerprint_count,
                pca_explained_variance=float(model.explained_variance_ratio.sum())
                    if model.explained_variance_ratio is not None else None,
                nominal_bytes_per_hour=config.byte_budget(factor),
                measured_payload_bytes_per_hour=stats["payload_bytes"]/duration_total*3600)
            predictions = []
            evaluation_rows = [(q, features) for q, features in zip(queries, query_features)
                if factor != 1 or dense_ids is None or
                ((q["role"] == "test_known" and q["query_id"] in dense_ids) or
                 (q["role"] != "test_known" and not a.dense_known_only))]
            for i, (query, (x, t, extraction_s)) in enumerate(evaluation_rows):
                started = time.perf_counter()
                hits = index.match(x, t, top_k=5, tolerance_sec=.12, limit=1)
                search_s = time.perf_counter() - started
                hit = hits[0] if hits else {}
                rid, offset = hit.get("content_id"), hit.get("offset_sec")
                correct = rid == query["reference_id"]
                predictions.append({"query_id": query["query_id"],
                    "base_query_id": query["query_id"].rsplit("@", 1)[0],
                    "source_id": query["source_id"], "source_query_id": query["source_query_id"],
                    "role": query["role"], "duration_s": query["duration_s"],
                    "target_reference_id": query["reference_id"], "reference_id": rid,
                    "score": hit.get("score", 0.), "offset_s": offset,
                    "expected_reference_start_s": query["expected_reference_start_s"],
                    "correct_reference": correct,
                    "localized_correct": correct and abs(offset-query["expected_reference_start_s"]) <= 2,
                    "query_fingerprints": len(x), "vote_fraction": hit.get("vote_fraction", 0.),
                    "mean_squared_l2": hit.get("mean_distance"),
                    "query_extraction_s": extraction_s, "query_search_s": search_s,
                    "latency_s": extraction_s+search_s, "annotation": query["annotation"],
                    "overlap": bool(query["overlapping_other_annotations"])})
                if (i+1) % 100 == 0:
                    print(f"{name} d{factor} queried {i+1}/{len(evaluation_rows)}", flush=True)
            integrity = summarize_integrity(protocol, references, [q for q, _ in evaluation_rows], model, predictions)
            result = {"method": "ACR-PCA", "ablation": name, "factor": factor,
                "config": asdict(config), "matcher": {"top_k": 5, "tolerance_sec": .12,
                "exact": True, "score": "vote_fraction/(1+mean_squared_L2)", "threads": a.threads},
                "stats": stats, "predictions": predictions, "integrity": integrity,
                "environment": {"platform": platform.platform(), "numpy": np.__version__,
                                "faiss": faiss.__version__, "audio_decoder": audio_decoder_info()}, "protocol_sha256": protocol_hash,
                "evaluation_cohort": {"name": "sha200_dense_control" if factor == 1 and dense_ids is not None else "complete835",
                    "dense_test_known_limit": a.dense_test_known_limit if factor == 1 else None,
                    "dense_known_only": a.dense_known_only if factor == 1 else False,
                    "selection": "SHA256('acr-dense-ablation-v1:'+base_query_id), smallest hashes" if factor == 1 and dense_ids is not None else "all canonical eligible queries"},
                "timing_caveat": "Ablation cache timing is shared/reused work; use warm extraction benchmark for device-independent timings."}
            temp = out.with_suffix(".partial.json")
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(out)
            print(f"Saved {out}", flush=True)
            del index
            gc.collect()
        del entries, query_features, model
        gc.collect()


if __name__ == "__main__":
    main()
