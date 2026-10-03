"""Quiet common-cohort CPU profiling of actual sparse NMFP window inference.

Only retained1s windows enter the official mel frontend and frozen network.
The original segmentation helper disallows gaps(H>L); a direct selector is
therefore used for1s/2s hops, preserving exactly its samples and float64 buffer
dtype. This is not dense encoding followed by discarded neural outputs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

from neural_baseline import NMFPExtractor, load_cached_embeddings
from neural_density_controls import check_grid, density_index, sha


def sparse_encode(extractor, wave, factor):
    hop = round(0.5*factor*extractor.fs)
    starts = np.arange(0, len(wave)-extractor.segment_samples+1, hop)
    # Official segment_audio allocates float64 and copies float32 source PCM.
    windows = np.stack([wave[s:s+extractor.segment_samples] for s in starts]).astype(np.float64)
    tick = time.perf_counter()
    inputs = extractor.frontend.compute_batch(windows)[:, :, :, None].astype(np.float32)
    frontend = time.perf_counter()-tick
    tick = time.perf_counter()
    embedding = np.asarray(extractor.forward(inputs), np.float32)
    forward = time.perf_counter()-tick
    return embedding, starts.astype(float)/extractor.fs, {"frontend_s": frontend, "forward_s": forward}


def distribution_per_query(rows, key):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["query_id"]].append(r[key])
    medians = np.asarray([np.median(v) for v in grouped.values()])
    return {"median": float(np.median(medians)), "p95": float(np.quantile(medians,.95)),
            "p99": float(np.quantile(medians,.99)), "unit": "seconds",
            "aggregation": "quantiles of50 query medians over5 repeats"}


def profile(args):
    manifest = json.loads(args.inputs.read_text())
    audio = np.load(manifest["audio_path"], allow_pickle=False)
    if audio.shape != (50,40000):
        raise ValueError("Expected frozen common50x5s8k cohort")
    queries = json.loads(args.query_cache.read_text())["queries"]
    references = json.loads(args.reference_cache.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    extractor = NMFPExtractor(batch_size=256, threads=4)
    setup = time.perf_counter()-tick
    sys.path.append(str(Path(__file__).resolve().parents[2]/"work/venv/lib/python3.12/site-packages"))
    import faiss
    faiss.omp_set_num_threads(4)
    result = {"method": "NMFP-Triplet actual sparse-window encoder", "inputs_sha256": sha(args.inputs),
              "audio_sha256": sha(manifest["audio_path"]), "runner_sha256": sha(__file__),
              "reference_cache_sha256": sha(args.reference_cache), "query_cache_sha256": sha(args.query_cache),
              "cohort": manifest, "provenance": extractor.provenance, "model_setup_s": setup,
              "faiss_version": faiss.__version__, "faiss_compile_options": faiss.get_compile_options(),
              "threads": 4, "repeats": 5, "warmups": 2, "explicit_gpu_index": False,
              "selection": "select only retained1s windows before the official mel/network; source buffer dtype matches official segment_audio",
              "setup_and_cold_traces_excluded_from_warm_clock": True, "variants": []}
    for factor in (2,4):
        tick = time.perf_counter()
        embedding,times,_ = sparse_encode(extractor,audio[0],factor)
        cold = time.perf_counter()-tick
        # Audit all50 inputs versus frozen dense cache BEFORE clocks; this also
        # warms the actual5-/3-frame model shapes without synthetic dummy input.
        max_delta = 0.0
        for i,wave in enumerate(audio):
            embedding,times,_ = sparse_encode(extractor,wave,factor)
            dense,dt,_ = load_cached_embeddings(queries[manifest["queries"][i]["query_id"]]["path"])
            check_grid(embedding,times,.5*factor)
            if not np.array_equal(times,dt[::factor]):
                raise AssertionError("Sparse physical window times changed")
            delta = float(np.max(np.abs(embedding-dense[::factor])))
            max_delta = max(max_delta,delta)
            if delta > 5e-5:
                raise AssertionError(f"Sparse inference changed cached embedding: {delta}")
        index = density_index({rid:e["path"] for rid,e in references["references"].items()},factor)
        build = index.build_index()
        for _ in range(2):
            e,_,_=sparse_encode(extractor,audio[0],factor)
            index.search(e,20)
        rows=[]
        for repeat in range(5):
            for i,wave in enumerate(audio):
                pipeline=time.perf_counter();tick=time.perf_counter()
                e,t,parts=sparse_encode(extractor,wave,factor)
                encoder=time.perf_counter()-tick;tick=time.perf_counter()
                hit=index.search(e,20);match=time.perf_counter()-tick
                total=time.perf_counter()-pipeline
                rows.append({"query_id":manifest["queries"][i]["query_id"],"repeat":repeat,
                             "encoder_s":encoder,"matching_s":match,"pipeline_s":total,
                             "fingerprints":len(e),"frame_times_s":t.tolist(),
                             "reference_id":hit["reference_id"],"score":hit["score"],**parts})
        result["variants"].append({"reference_factor":factor,"query_factor":factor,
                                   "reference_hop_s":.5*factor,"query_hop_s":.5*factor,
                                   "query_frames":5 if factor==2 else 3,
                                   "cold_first_actual_shape_s":cold,"index_build_s":build,
                                   "sparse_inference_dense_cache_max_abs_delta":max_delta,
                                   "encoder_s":distribution_per_query(rows,"encoder_s"),
                                   "matching_s":distribution_per_query(rows,"matching_s"),
                                   "pipeline_s":distribution_per_query(rows,"pipeline_s"),
                                   "timing":rows,"candidate_index":"exact IndexFlatIP, top20"})
        (args.output/"profile.json").write_text(json.dumps(result,indent=2)+"\n")
        print({k:v for k,v in result["variants"][-1].items() if k!="timing"},flush=True)
        del index


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,default=Path("work/profiling/common_inputs/inputs.json"))
    p.add_argument("--reference-cache",type=Path,default=Path("work/benchmarks/nmfp_cache/cache_index.json"))
    p.add_argument("--query-cache",type=Path,default=Path("work/benchmarks/nmfp_cache/query_cache_index.json"))
    p.add_argument("--output",type=Path,default=Path("work/profiling/nmfp_density"))
    profile(p.parse_args())
