"""Evaluate both frozen IVF operating points on an extended unknown protocol.

Reference centroids, PCA, factor8, nprobe16 and query strides1/4 are frozen from
the official calibration study. Only the rejection threshold is recalibrated
using the declared calibration-unknown population at an empirical0.1% target.
Original835-query outputs are preserved in their separate directories.
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
from run_acr_protocol import query_crop,extract_query
from run_ablations import ref_path, query_path
from study_query_ivf import prediction, threshold_from_unknown


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--original-protocol",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)
    extended=json.loads(a.protocol.read_text())
    original=json.loads(a.original_protocol.read_text())
    model=Fingerprinter.load(a.model)
    c=model.config
    roles=("calibration_known","test_known")
    gallery=[r for r in extended["references"] if r["role"] in roles]
    original_gallery=[r for r in original["references"] if r["role"] in roles]
    if [(r["reference_id"],r["path"],r["role"]) for r in gallery] != [(r["reference_id"],r["path"],r["role"]) for r in original_gallery]:
        raise ValueError("Extended study changed reference membership/order")
    queries=[query_crop(q,5) for q in extended["queries"] if q["evaluate"]]
    unknown={q["reference_id"] for q in queries if q["role"].endswith("unknown")}
    if unknown & {r["reference_id"] for r in gallery}:
        raise ValueError("Extended unknown sources overlap gallery")
    model.assert_disjoint(reference_ids=[r["reference_id"] for r in gallery],query_ids=[q["reference_id"] for q in queries])
    a.output.mkdir(parents=True,exist_ok=True)
    protocol_hash=hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    plan={"protocol_sha256":protocol_hash,"original_protocol_sha256":hashlib.sha256(a.original_protocol.read_bytes()).hexdigest(),
        "frozen_index_sha256":hashlib.sha256(a.index.read_bytes()).hexdigest(),
        "frozen_pca_sha256":hashlib.sha256(a.model.read_bytes()).hexdigest(),
        "reference_factor":8,"ivf_nlist":512,"nprobe":16,"query_strides":[1,4],
        "threshold_calibration_role":"calibration_unknown","empirical_calibration_far_target":.001,
        "configuration_selected_from_extended_test_outcomes":False,
        "populations":"Additional clean FMA and official noisy PEX negatives reported separately"}
    (a.output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    entries=[];duration_total=0.
    for r in gallery:
        with np.load(ref_path(a.cache,r,c)) as z:
            entries.append((r["reference_id"],model.transform(z["features"]),z["times"].copy()))
            duration_total+=float(z["duration_s"])
    index=ExactIndex.from_entries(entries,factor=8,
        grid_step_sec=c.hop_length/c.sample_rate,grid_origin_sec=c.first_center_seconds)
    del entries
    gc.collect()
    ivf=faiss.read_index(str(a.index))
    if ivf.ntotal!=len(index.vectors) or ivf.d!=c.output_dim or ivf.nlist!=512:
        raise ValueError("Frozen IVF index shape does not match gallery")
    ivf.make_direct_map()
    check=np.random.default_rng(20261002).choice(len(index.vectors),100,replace=False)
    for row in check:
        np.testing.assert_array_equal(ivf.reconstruct(int(row)),index.vectors[row])
    ivf.nprobe=16
    index._faiss=ivf
    stats=index.bytes()
    stats.update(reference_count=len(gallery),reference_audio_duration_s=duration_total,
        measured_payload_bytes_per_hour=stats["payload_bytes"]/duration_total*3600,
        index_serialized_bytes=int(faiss.serialize_index(ivf).nbytes),
        frozen_reference_vectors_checked=100,frozen_index_vector_discrepancies=0)
    calibration=[q for q in queries if q["role"]=="calibration_unknown"]
    for stride in (1,4):
        out=a.output/f"acr_ivf_qs{stride}_extended_5s.json"
        if out.exists() and json.loads(out.read_text())["protocol_sha256"]==protocol_hash:
            print(f"Complete extended stride{stride} result exists",flush=True)
            continue
        def evaluate(q):
            path=query_path(a.cache,q,c)
            if path.exists():
                with np.load(path) as z:
                    x,t,extract_s=model.transform(z["features"]),z["times"].copy(),float(z["extraction_s"])
            else:
                # Fresh artifact users need not run an expensive exact study
                # solely to prepare new unknown clips for this frozen ANN run.
                x,t,extract_s=extract_query(q,model,a.cache)
            row=prediction(q,x,t,index,stride,16,c,extract_s)
            row["negative_population"]=q.get("negative_population","PEX_official_noisy" if q["role"].endswith("unknown") else "PEX_official_known")
            row["artist_id"]=q.get("artist_id")
            return row
        calibration_rows=[]
        for i,q in enumerate(calibration):
            calibration_rows.append(evaluate(q))
            if (i+1)%250==0:
                print(f"Stride{stride} calibrated {i+1}/{len(calibration)} unknowns",flush=True)
        threshold=threshold_from_unknown([r["score"] for r in calibration_rows],target=.001)
        frozen={"query_stride":stride,"nprobe":16,"threshold":threshold,
            "calibration_unknown_count":len(calibration_rows),
            "calibration_false_accepts":sum(r["score"]>=threshold for r in calibration_rows),
            "empirical_far_target":.001,"uses_extended_test_outcomes":False}
        (a.output/f"threshold_qs{stride}.json").write_text(json.dumps(frozen,indent=2)+"\n")
        cached={r["query_id"]:r for r in calibration_rows}
        rows=[]
        for i,q in enumerate(queries):
            row=cached.get(q["query_id"]) or evaluate(q)
            row["accepted"]=row["score"]>=threshold
            rows.append(row)
            if (i+1)%500==0:
                print(f"Stride{stride} extended queried {i+1}/{len(queries)}",flush=True)
        if len({r["query_id"] for r in rows})!=len(queries):
            raise ValueError("Extended prediction completeness failed")
        result={"method":f"ACR-PCA IVF512 query-stride{stride}","factor":8,
            "config":asdict(c),"matcher":{"top_k":5,"tolerance_sec":.12,"exact":False,
                "threads":a.threads,"query_stride":stride,"ivf_nlist":512,"nprobe":16,
                "score":"vote_fraction/(1+mean_squared_L2)"},
            "stats":stats,"predictions":rows,"frozen_threshold":threshold,
            "threshold_calibration":frozen,"plan":plan,"protocol_sha256":protocol_hash,
            "environment":{"platform":platform.platform(),"numpy":np.__version__,"faiss":faiss.__version__,"audio_decoder":audio_decoder_info()},
            "timing_caveat":"Concurrent experiments can contend; source features are cached and decode time is excluded."}
        temp=out.with_suffix(".partial.json")
        temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(out)
        print(f"Saved complete {len(rows)}-query extended stride{stride} result",flush=True)


if __name__=="__main__":
    main()
