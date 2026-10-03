"""Calibration-only sensitivity of NN-estimated affine time alignment.

This script deliberately never evaluates held-out query roles. It compares the
fixed unit matcher and a prespecified affine prototype on the same calibration
sources and frozen representation/index. No annotated tempo enters inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from acr_fp import Fingerprinter, ExactIndex
from run_acr_protocol import query_crop
from run_ablations import query_path,ref_path
from study_query_ivf import thin_query,threshold_from_unknown
from affine_match import match_affine


def load_frozen_index(protocol,cache,model,index_path):
    import faiss
    gallery=[r for r in protocol["references"] if r["role"] in ("calibration_known","test_known")]
    codes,times,names=[],[],[]
    c=model.config
    for i,r in enumerate(gallery):
        with np.load(ref_path(cache,r,c)) as z:
            t=z["times"]
        frames=np.rint((t-c.first_center_seconds)/(c.hop_length/c.sample_rate)).astype(np.int64)
        selected=t[frames%8==0]
        codes.append(np.full(len(selected),i,np.int32));times.append(selected);names.append(r["reference_id"])
    ivf=faiss.read_index(str(index_path));ivf.make_direct_map();ivf.nprobe=16
    vectors=ivf.reconstruct_n(0,ivf.ntotal)
    index=ExactIndex(vectors,np.concatenate(codes),np.concatenate(times),names,factor=8)
    if len(index.vectors)!=ivf.ntotal:
        raise ValueError("Frozen sparse metadata does not match index")
    index._faiss=ivf
    return index


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)
    protocol=json.loads(a.protocol.read_text());model=Fingerprinter.load(a.model)
    queries=[query_crop(q,5.) for q in protocol["queries"] if q["evaluate"] and q["role"].startswith("calibration")]
    if any(q["role"].startswith("test") for q in queries):
        raise ValueError("Held-out queries cannot enter this sensitivity study")
    a.output.mkdir(parents=True,exist_ok=True)
    plan={"roles":["calibration_known","calibration_unknown"],"reference_factor":8,
        "ivf_nlist":512,"nprobe":16,"query_stride":4,"top_k":5,"time_tolerance_sec":.12,
        "affine_scale_range":[.5,2.],"affine_proposals":64,"minimum_pair_query_span_sec":1.,
        "affine_seed":20261002,"annotated_tempo_used_in_inference":False,
        "empirical_calibration_far_target":.001,"held_out_queries_evaluated":False,
        "protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest()}
    (a.output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    index=load_frozen_index(protocol,a.cache,model,a.index)
    results={"unit":[],"affine":[]}
    for i,q in enumerate(queries):
        with np.load(query_path(a.cache,q,model.config)) as z:
            x,t=model.transform(z["features"]),z["times"].copy()
        x,t=thin_query(x,t,4,model.config)
        for name in ("unit","affine"):
            started=time.perf_counter()
            hit=(index.match(x,t,limit=1) if name=="unit" else match_affine(index,x,t,limit=1))
            elapsed=time.perf_counter()-started;hit=hit[0] if hit else {}
            correct=hit.get("content_id")==q["reference_id"]
            results[name].append({"query_id":q["query_id"],"role":q["role"],
                "source_id":q["source_id"],"source_query_id":q["source_query_id"],
                "target_reference_id":q["reference_id"],"reference_id":hit.get("content_id"),
                "score":hit.get("score",0.),"correct_reference":correct,
                "offset_s":hit.get("offset_sec"),"estimated_time_scale":hit.get("time_scale",1.),
                "matching_s":elapsed,"query_fingerprints":len(x),
                "annotation":q["annotation"],"negative_population":q.get("negative_population")})
        if (i+1)%200==0:
            print(f"Affine sensitivity calibrated {i+1}/{len(queries)}",flush=True)
    summaries=[]
    for name,rows in results.items():
        known=[r for r in rows if r["role"]=="calibration_known"]
        unknown=[r for r in rows if r["role"]=="calibration_unknown"]
        threshold=threshold_from_unknown([r["score"] for r in unknown],target=.001)
        summaries.append({"matcher":name,"threshold":threshold,"calibration_known":len(known),
            "calibration_unknown":len(unknown),"top1_correct":sum(r["correct_reference"] for r in known),
            "accepted_correct":sum(r["correct_reference"] and r["score"]>=threshold for r in known),
            "false_accepts":sum(r["score"]>=threshold for r in unknown),
            "median_matching_ms":float(np.median([r["matching_s"] for r in rows])*1000),
            "p95_matching_ms":float(np.percentile([r["matching_s"] for r in rows],95)*1000)})
        (a.output/f"{name}_calibration.json").write_text(json.dumps({"plan":plan,"summary":summaries[-1],"predictions":rows},indent=2)+"\n")
    out={"plan":plan,"summaries":summaries,"interpretation":"Calibration-only prototype sensitivity. Unit baseline remains frozen; any held-out affine evaluation requires a separately frozen configuration, without using test outcomes."}
    (a.output/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2),flush=True)


if __name__=="__main__":
    main()
