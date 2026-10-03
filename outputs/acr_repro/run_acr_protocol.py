"""Reproduce the non-neural ACR evaluation from a frozen source split."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import platform
import time
import numpy as np
from acr_fp import FingerprintConfig, Fingerprinter, ExactIndex, load_audio, AUDIO_DECODER_REVISION, audio_decoder_info

def fingerprint_key(config):
    # Decoding is part of feature provenance. Cached soundfile rows cannot be
    # reused after a switch to the common FFmpeg benchmark backend.
    return hashlib.sha256(json.dumps({"config": asdict(config),
        "audio_decoder": audio_decoder_info()}, sort_keys=True).encode()).hexdigest()[:16]

def extract_ref(task):
    ref, config_dict, cache_dir = task
    config = FingerprintConfig(**config_dict)
    source = Path(ref["path"])
    key = hashlib.sha256((str(source.resolve()) + str(source.stat().st_size)).encode()).hexdigest()[:20]
    out = Path(cache_dir) / f"{fingerprint_key(config)}_{key}.npz"
    if out.exists():
        return ref["reference_id"], str(out), True
    started = time.perf_counter()
    audio, sr = load_audio(source, config.sample_rate)
    decode_s = time.perf_counter() - started
    started = time.perf_counter()
    x, t = Fingerprinter(config).raw_features(audio, sr)
    extract_s = time.perf_counter() - started
    np.savez_compressed(out, features=x, times=t, duration_s=len(audio)/sr,
                        decode_s=decode_s, extraction_s=extract_s)
    return ref["reference_id"], str(out), False

def query_crop(query, duration):
    q = dict(query)
    # Canonical numeric type keeps semantically identical 5/5.0 crop cache
    # keys identical across command-line and programmatic callers.
    duration = min(float(duration), float(q["duration_s"]))
    shift = (q["duration_s"] - duration) / 2
    q["start_s"] += shift
    q["duration_s"] = duration
    q["expected_reference_start_s"] += shift * q["expected_time_scale"]
    q["query_id"] += f"@{duration:g}s"
    return q

def extract_query(query, model, cache_dir):
    key = hashlib.sha256(json.dumps({k:query[k] for k in
        ("path","start_s","duration_s")},sort_keys=True).encode()).hexdigest()[:24]
    out = Path(cache_dir) / f"query_{fingerprint_key(model.config)}_{key}.npz"
    if out.exists():
        with np.load(out) as z:
            started=time.perf_counter()
            projected=model.transform(z["features"])
            projection_s=time.perf_counter()-started
            return projected, z["times"].copy(), float(z["extraction_s"])+projection_s
    started = time.perf_counter()
    audio, sr = load_audio(query["path"], model.config.sample_rate)
    start = round(query["start_s"] * sr)
    y = audio[start:start + round(query["duration_s"] * sr)]
    if len(y) < round(query["duration_s"] * sr) - 5:
        raise ValueError(f"Short decoded query: {query['query_id']}")
    # Decoding is excluded from extraction timing and cached independently.
    started = time.perf_counter()
    features, t = model.raw_features(y, sr)
    extraction_s = time.perf_counter() - started
    np.savez_compressed(out, features=features, times=t, extraction_s=extraction_s)
    started=time.perf_counter()
    projected=model.transform(features)
    return projected, t, extraction_s+time.perf_counter()-started

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--durations", default="5,1,2,3,10")
    p.add_argument("--factors", default="1,2,4,6,8")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=0.12)
    p.add_argument("--calibration-only", action="store_true")
    a = p.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)
    m = json.loads(a.protocol.read_text())
    config = FingerprintConfig(**(json.loads(a.config.read_text()) if a.config else {}))
    a.cache.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    reference_rows = [r for r in m["references"] if r["role"] in
        ("pca_fit","calibration_known","test_known")]
    paths = {}
    started = time.perf_counter()
    tasks = [(r,asdict(config),str(a.cache)) for r in reference_rows]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(rid,path,cached) in enumerate(pool.map(extract_ref,tasks)):
            paths[rid] = path
            if (i+1)%50 == 0:
                print(f"ACR extracted {i+1}/{len(tasks)} references",flush=True)
    extraction_wall_s = time.perf_counter()-started
    fit_refs = [r for r in reference_rows if r["role"]=="pca_fit"]
    rng = np.random.default_rng(20261002)
    def fit_batches():
        for r in fit_refs:
            with np.load(paths[r["reference_id"]]) as z:
                x = z["features"]
                if len(x)>2000:
                    x=x[np.sort(rng.choice(len(x),2000,replace=False))]
                yield x
    model = Fingerprinter(config).fit(fit_batches(),content_ids=[r["reference_id"] for r in fit_refs])
    gallery = [r for r in reference_rows if r["role"] in ("calibration_known","test_known")]
    model.assert_disjoint([r["reference_id"] for r in gallery],
        [q["reference_id"] for q in m["queries"] if q["evaluate"]])
    model.save(a.output/"pca_model.npz")
    entries=[]
    duration_total=extract_work=0.0
    for r in gallery:
        with np.load(paths[r["reference_id"]]) as z:
            entries.append((r["reference_id"],model.transform(z["features"]),z["times"].copy()))
            duration_total += float(z["duration_s"])
            extract_work += float(z["extraction_s"])
    queries = [q for q in m["queries"] if q["evaluate"] and
        (not a.calibration_only or q["role"].startswith("calibration"))]
    durations=[float(s) for s in a.durations.split(",")]
    factors=[int(s) for s in a.factors.split(",")]
    all_features={}
    for duration in durations:
        records=[query_crop(q,duration) for q in queries]
        for i,q in enumerate(records):
            all_features[q["query_id"]]=extract_query(q,model,a.cache)
        print(f"ACR cached {len(records)} queries at {duration:g}s",flush=True)
    for factor in factors:
        started=time.perf_counter()
        index=ExactIndex.from_entries(entries,factor=factor,
            grid_step_sec=config.hop_length/config.sample_rate,
            grid_origin_sec=config.first_center_seconds)
        build_s=time.perf_counter()-started
        stats=index.bytes()
        stats.update(reference_audio_duration_s=duration_total,
            reference_count=len(gallery),reference_extraction_work_s=extract_work,
            extraction_wall_s=extraction_wall_s,index_build_s=build_s,
            pca_fitted_fingerprints=model.fit_fingerprint_count,
            pca_explained_variance=float(model.explained_variance_ratio.sum()) if model.explained_variance_ratio is not None else None,
            nominal_bytes_per_hour=config.byte_budget(factor),
            measured_payload_bytes_per_hour=stats["payload_bytes"]/duration_total*3600)
        for duration in durations:
            out=a.output/f"acr_d{factor}_{duration:g}s.json"
            if out.exists():
                print(f"ACR result exists {out.name}",flush=True)
                continue
            predictions=[]
            for i,original in enumerate(queries):
                q=query_crop(original,duration)
                x,t,extract_s=all_features[q["query_id"]]
                started=time.perf_counter()
                hits=index.match(x,t,top_k=a.top_k,tolerance_sec=a.tolerance,limit=1)
                search_s=time.perf_counter()-started
                hit=hits[0] if hits else {}
                rid=hit.get("content_id")
                offset=hit.get("offset_sec")
                correct=rid==q["reference_id"]
                predictions.append({"query_id":q["query_id"],"base_query_id":original["query_id"],
                    "source_id":q["source_id"],"source_query_id":q.get("source_query_id",q["source_id"]),
                    "role":q["role"],"duration_s":q["duration_s"],
                    "target_reference_id":q["reference_id"],"reference_id":rid,
                    "score":hit.get("score",0.0),"offset_s":offset,
                    "expected_reference_start_s":q["expected_reference_start_s"],
                    "correct_reference":correct,
                    "localized_correct":correct and abs(offset-q["expected_reference_start_s"])<=2,
                    "query_fingerprints":len(x),"vote_fraction":hit.get("vote_fraction",0.0),
                    "mean_squared_l2":hit.get("mean_distance"),
                    "query_extraction_s":extract_s,"query_search_s":search_s,
                    "latency_s":extract_s+search_s,"annotation":q["annotation"],
                    "overlap":bool(q["overlapping_other_annotations"])})
                if (i+1)%100==0:
                    print(f"ACR d{factor} {duration:g}s queried {i+1}/{len(queries)}",flush=True)
            result={"method":"ACR-PCA", "factor":factor,"duration_s_requested":duration,"config":asdict(config),
                "matcher":{"top_k":a.top_k,"tolerance_sec":a.tolerance,
                    "exact":True,"score":"vote_fraction/(1+mean_squared_L2)",
                    "threads":a.threads},"stats":stats,"predictions":predictions,
                "environment":{"platform":platform.platform(),"numpy":np.__version__,"faiss":faiss.__version__,
                    "audio_decoder":audio_decoder_info()},
                "protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest()}
            out.write_text(json.dumps(result,indent=2)+"\n")
            print(f"Saved {out}",flush=True)
        del index

if __name__=="__main__":
    main()
