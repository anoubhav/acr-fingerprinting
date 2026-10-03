"""Matched source-unseen sensitivity using unchanged global thresholds.

The exact 10k NMFP training list is unavailable; exclude its entire documented
FMA-medium superset, as in the verified neural sensitivity analysis. This does
not imply artist/composition separation. Cross-method pairs use base query IDs
and verify target, actual duration and expected source position.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from summarize_results import summarize,cluster_interval,crossed_interval,wilson


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--exact",type=Path,required=True)
    p.add_argument("--ivf",type=Path,required=True)
    p.add_argument("--nmfp",type=Path,required=True)
    p.add_argument("--peaknet",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());audit=json.loads(a.audit.read_text())
    unseen=set(audit["conservative_unseen_reference_ids"])
    results={name:json.loads(path.read_text()) for name,path in
             (("ACR exact d8",a.exact),("ACR IVF qstride4",a.ivf),("NMFP native",a.nmfp),("PeakNet native",a.peaknet))}
    summaries=[];pairs={};thresholds={}
    for name,result in results.items():
        # Reproduce the global threshold on the full calibration cohort before
        # filtering the sensitivity slice. No fit/gallery memberships change.
        global_summary=summarize(result,protocol)
        threshold=global_summary["threshold"];thresholds[name]=threshold
        known=[r for r in result["predictions"] if r["role"]=="test_known" and r["source_id"] in unseen]
        unknown=[r for r in result["predictions"] if r["role"]=="test_unknown" and r["source_id"] in unseen]
        if len(known)!=389:
            raise ValueError(f"Expected389 matched source-unseen known queries, found{len(known)} for{name}")
        pairs[name]={r["base_query_id"]:r for r in known}
        if len(pairs[name])!=len(known):
            raise ValueError("Duplicate base query IDs in matched cohort")
        false_accepts=sum(r["accepted"] for r in unknown)
        summaries.append({"method":name,"threshold":threshold,"threshold_held_fixed":True,
            "test_known":{key:cluster_interval(known,key) for key in
                ("correct_reference","localized_correct","accepted_correct","accepted_localized","accepted_wrong")},
            "test_known_crossed_ci":{key:crossed_interval(known,key) for key in
                ("correct_reference","accepted_correct")},
            "test_unknown":{"n":len(unknown),"false_accepts":false_accepts,"far_wilson95":wilson(false_accepts,len(unknown))}})
    comparisons=[]
    for ours in ("ACR exact d8","ACR IVF qstride4"):
        for other in ("NMFP native","PeakNet native"):
            if set(pairs[ours])!=set(pairs[other]):
                raise ValueError("Matched source-unseen query sets differ")
            for metric in ("correct_reference","accepted_correct"):
                rows=[];wins=losses=0
                for qid,left in pairs[ours].items():
                    right=pairs[other][qid]
                    for key in ("target_reference_id","duration_s","expected_reference_start_s"):
                        if left[key]!=right[key]:
                            raise ValueError(f"Cross-method query crop/target differs:{qid}/{key}")
                    delta=int(left[metric])-int(right[metric])
                    wins+=delta==1;losses+=delta==-1
                    rows.append({**left,"paired_delta":delta})
                ci=crossed_interval(rows,"paired_delta")
                ci["net_success_difference"]=ci.pop("successes")
                source_ci=cluster_interval(rows,"paired_delta")
                source_ci["net_success_difference"]=source_ci.pop("successes")
                comparisons.append({"difference":f"{ours} minus {other}","metric":metric,
                    "paired_wins":wins,"paired_losses":losses,"crossed_cluster_ci":ci,
                    "source_cluster_ci":source_ci})
    out={"interpretation":audit["interpretation"],"gallery_unchanged":True,
        "thresholds_fit_on_full_original_calibration_then_held_fixed":True,
        "known_source_unseen_queries":389,"rows":summaries,"paired_comparisons":comparisons,
        "protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
        "warning":"Confidence intervals describe this adapted sample. Exclusion from FMA_medium does not prove artist/composition independence."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    for row in summaries:
        print(row["method"],row["test_known"]["correct_reference"]["successes"],
              row["test_known"]["accepted_correct"]["successes"],row["threshold"])


if __name__=="__main__":
    main()
