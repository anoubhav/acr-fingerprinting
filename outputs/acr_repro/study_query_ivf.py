"""Select an ACR query-stride/IVF operating point using calibration only.

All settings keep the recovered representation and factor-eight reference
grid fixed. This evaluates server-side query/index choices, not new fingerprint
quality. Selection maximizes calibrated accepted-correct calibration-known
rate, breaking exact ties by fewer query rows and then lower nprobe.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from acr_fp import Fingerprinter, ExactIndex, audio_decoder_info
from run_acr_protocol import query_crop
from run_ablations import ref_path, query_path


QUERY_STRIDES = (1, 2, 4, 8, 16)
NPROBES = (1, 4, 16, 64)


def threshold_from_unknown(scores, target=.01):
    x = np.sort(np.asarray(scores, dtype=np.float64))
    if not len(x):
        raise ValueError("Unknown calibration set is empty")
    allowed = int(np.floor(target*len(x)))
    # Accept score >= threshold; strictly exceed the relevant negative order
    # statistic. Ties are conservatively excluded rather than exceeding target.
    rank = len(x)-allowed-1
    return float(np.nextafter(x[max(0, rank)], np.inf))


def thin_query(x, t, stride, config):
    frames = np.rint((t-config.first_center_seconds)/(config.hop_length/config.sample_rate)).astype(np.int64)
    selected = frames % stride == 0
    return x[selected], t[selected]


def prediction(query, x, times, index, stride, nprobe, config, extraction_s):
    x, times = thin_query(x, times, stride, config)
    started = time.perf_counter()
    hits = index.match(x, times, top_k=5, tolerance_sec=.12, limit=1)
    search_s = time.perf_counter()-started
    hit = hits[0] if hits else {}
    rid, offset = hit.get("content_id"), hit.get("offset_sec")
    correct = rid == query["reference_id"]
    return {"query_id": query["query_id"], "base_query_id": query["query_id"].rsplit("@", 1)[0],
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
        "overlap": bool(query["overlapping_other_annotations"]),
        "query_stride": stride, "nprobe": nprobe}


def calibration_summary(rows):
    negative = [r for r in rows if r["role"] == "calibration_unknown"]
    known = [r for r in rows if r["role"] == "calibration_known"]
    threshold = threshold_from_unknown([r["score"] for r in negative])
    return {"threshold": threshold, "calibration_known": len(known),
        "calibration_unknown": len(negative),
        "calibration_accepted_correct": sum(r["correct_reference"] and r["score"] >= threshold for r in known),
        "calibration_accepted_correct_rate": sum(r["correct_reference"] and r["score"] >= threshold for r in known)/len(known),
        "calibration_false_accepts": sum(r["score"] >= threshold for r in negative),
        "calibration_top1": sum(r["correct_reference"] for r in known)/len(known),
        "median_query_search_s": float(np.median([r["query_search_s"] for r in rows])),
        "p95_query_search_s": float(np.percentile([r["query_search_s"] for r in rows],95)),
        "median_query_fingerprints": float(np.median([r["query_fingerprints"] for r in rows]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--reuse-calibration", type=Path)
    p.add_argument("--max-calibration-loss-pp", type=float, default=0.)
    p.add_argument("--index-only", action="store_true")
    a = p.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)
    m = json.loads(a.protocol.read_text())
    protocol_hash = hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    model = Fingerprinter.load(a.model)
    c = model.config
    a.output.mkdir(parents=True, exist_ok=True)
    plan = {"protocol_sha256": protocol_hash, "query_strides": QUERY_STRIDES,
        "nprobes": NPROBES, "reference_factor": 8, "nlist": 512,
        "training_source_roles": ["pca_fit"], "training_samples_cap": 50000,
        "training_seed": 20261002, "niter": 20, "query_duration_s": 5,
        "selection": "maximize calibration-known accepted-correct rate at empirically <=1% calibration-unknown false accepts; tie: fewer query rows then lower nprobe" if not a.max_calibration_loss_pp else f"calibration accepted-correct within {a.max_calibration_loss_pp:g} percentage points of maximum; favor fewer query rows then lower nprobe",
        "selection_uses_test_outcomes": False,
        "interpretation": "Server query/index configuration study. ANN approximation must not be attributed to fingerprint representation."}
    (a.output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    gallery = [r for r in m["references"] if r["role"] in ("calibration_known", "test_known")]
    fit_refs = [r for r in m["references"] if r["role"] == "pca_fit"]
    model.assert_disjoint(reference_ids=[r["reference_id"] for r in gallery])
    entries, duration_total = [], 0.
    for reference in gallery:
        with np.load(ref_path(a.cache, reference, c)) as z:
            entries.append((reference["reference_id"], model.transform(z["features"]), z["times"].copy()))
            duration_total += float(z["duration_s"])
    index = ExactIndex.from_entries(entries, factor=8,
        grid_step_sec=c.hop_length/c.sample_rate, grid_origin_sec=c.first_center_seconds)
    del entries
    gc.collect()
    fit_arrays = []
    rng = np.random.default_rng(20261002)
    for reference in fit_refs:
        with np.load(ref_path(a.cache,reference,c)) as z:
            raw = z["features"]
            if len(raw) > 2000:
                raw = raw[np.sort(rng.choice(len(raw),2000,replace=False))]
            fit_arrays.append(model.transform(raw))
    train = np.concatenate(fit_arrays)
    del fit_arrays
    if len(train)>50000:
        train = train[np.sort(rng.choice(len(train),50000,replace=False))]
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(c.output_dim), c.output_dim, 512, faiss.METRIC_L2)
    ivf.cp.seed, ivf.cp.niter = 20261002, 20
    started = time.perf_counter()
    ivf.train(np.ascontiguousarray(train,dtype=np.float32))
    training_s = time.perf_counter()-started
    started = time.perf_counter()
    ivf.add(index._search_vectors)
    add_s = time.perf_counter()-started
    index._faiss = ivf
    train_count = len(train)
    faiss.write_index(ivf,str(a.output/"ivf_factor8.index"))
    del train
    gc.collect()
    stats = index.bytes()
    stats.update(reference_count=len(gallery), reference_audio_duration_s=duration_total,
        index_train_s=training_s, index_add_s=add_s, ivf_train_samples=train_count,
        index_serialized_bytes=int(faiss.serialize_index(ivf).nbytes),
        measured_payload_bytes_per_hour=stats["payload_bytes"]/duration_total*3600,
        ivf_nlist=512, ivf_training_source_ids=[r["reference_id"] for r in fit_refs])
    (a.output/"trained_index_metadata.json").write_text(json.dumps({"stats":stats,
        "protocol_sha256":protocol_hash,"pca_model_sha256":hashlib.sha256(a.model.read_bytes()).hexdigest(),
        "config":asdict(c),"reference_factor":8,"index_file":"ivf_factor8.index",
        "instruction":"Set nprobe from the frozen selected_setting.json before retrieval; query stride does not change reference centroids or postings."},indent=2)+"\n")
    if a.index_only:
        print("Saved trained factor-eight IVF512 index and provenance",flush=True)
        return
    queries = [query_crop(q,5) for q in m["queries"] if q["evaluate"]]
    query_features = {}
    for q in queries:
        with np.load(query_path(a.cache,q,c)) as z:
            query_features[q["query_id"]] = (model.transform(z["features"]), z["times"].copy(), float(z["extraction_s"]))
    calibration = [q for q in queries if q["role"].startswith("calibration")]
    settings = []
    for stride in QUERY_STRIDES:
        for nprobe in NPROBES:
            old_path = a.reuse_calibration / f"cal_qs{stride}_np{nprobe}.json" if a.reuse_calibration else None
            if old_path is not None:
                old = json.loads(old_path.read_text())
                if old["protocol_sha256"] != protocol_hash:
                    raise ValueError("Reused calibration protocol mismatch")
                settings.append(old["setting"])
                (a.output/f"cal_qs{stride}_np{nprobe}.json").write_text(json.dumps(old,indent=2)+"\n")
                continue
            ivf.nprobe = nprobe
            rows = []
            for q in calibration:
                x,t,extract_s = query_features[q["query_id"]]
                rows.append(prediction(q,x,t,index,stride,nprobe,c,extract_s))
            summary = {"query_stride":stride,"nprobe":nprobe,**calibration_summary(rows)}
            settings.append(summary)
            out = {"phase":"calibration_only","setting":summary,"predictions":rows,
                "stats":stats,"plan":plan,"protocol_sha256":protocol_hash}
            (a.output/f"cal_qs{stride}_np{nprobe}.json").write_text(json.dumps(out,indent=2)+"\n")
            print(f"Calibrated qstride{stride} nprobe{nprobe}: accepted-correct {summary['calibration_accepted_correct']}/{summary['calibration_known']}; negative {summary['calibration_false_accepts']}/{summary['calibration_unknown']}",flush=True)
    if a.max_calibration_loss_pp:
        best_rate = max(s["calibration_accepted_correct_rate"] for s in settings)
        eligible = [s for s in settings if s["calibration_accepted_correct_rate"] >= best_rate-a.max_calibration_loss_pp/100-1e-12]
        selected = max(eligible,key=lambda s:(s["query_stride"],-s["nprobe"],s["calibration_accepted_correct_rate"]))
    else:
        selected = max(settings,key=lambda s:(s["calibration_accepted_correct_rate"],s["query_stride"],-s["nprobe"]))
    # Write immutable selection before looking at any held-out test prediction.
    selection = {"selected":selected,"all_settings":settings,"plan":plan,
                 "protocol_sha256":protocol_hash,"selection_uses_test_outcomes":False}
    (a.output/"selected_setting.json").write_text(json.dumps(selection,indent=2)+"\n")
    stride,nprobe = selected["query_stride"],selected["nprobe"]
    ivf.nprobe=nprobe
    rows=[]
    for i,q in enumerate(queries):
        x,t,extract_s = query_features[q["query_id"]]
        row=prediction(q,x,t,index,stride,nprobe,c,extract_s)
        row["accepted"]=row["score"]>=selected["threshold"]
        rows.append(row)
        if (i+1)%100==0:
            print(f"Selected setting queried {i+1}/{len(queries)}",flush=True)
    result={"method":"ACR-PCA (calibration-selected IVF/query stride)","factor":8,
        "config":asdict(c),"matcher":{"top_k":5,"tolerance_sec":.12,"exact":False,
            "threads":a.threads,"query_stride":stride,"ivf_nlist":512,"nprobe":nprobe,
            "score":"vote_fraction/(1+mean_squared_L2)"},"stats":stats,"predictions":rows,
        "frozen_threshold":selected["threshold"],"selection":selection,
        "environment":{"platform":platform.platform(),"numpy":np.__version__,"faiss":faiss.__version__,
            "audio_decoder":audio_decoder_info()},"protocol_sha256":protocol_hash,
        "timing_caveat":"Concurrent experimental jobs can contend; timings include matching but exclude decode."}
    (a.output/"acr_ivf_selected_5s.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"Selected qstride{stride}/nprobe{nprobe}; complete {len(rows)}-query test evaluation saved",flush=True)


if __name__=="__main__":
    main()
