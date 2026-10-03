"""Held-out ROC description; these thresholds are not deployment calibration.

Each model and search configuration must already be frozen. The held-out clean
negative score distribution determines descriptive thresholds only; none are
fed back into the model, operating-point selection, or calibrated result files.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from summarize_results import wilson

def describe(result,protocol):
    lookup={q["query_id"]:q for q in protocol["queries"]}
    positive=[p for p in result["predictions"] if p["role"]=="test_known"]
    negative=[p for p in result["predictions"] if p["role"]=="test_unknown" and
              lookup[p["query_id"].split("@")[0]].get("negative_population")=="additional_clean_FMA"]
    if len(negative)!=4000:raise ValueError("Expected exactly 4,000 fixed clean unknowns")
    pscore=np.array([p["score"] for p in positive])
    correct=np.array([p["correct_reference"] and p["reference_id"] is not None for p in positive])
    nscore=np.array([p["score"] for p in negative])
    nreturned=np.array([p["reference_id"] is not None for p in negative])
    rows=[]
    for fpr in (0,.00025,.0005,.001,.0025,.005,.01,.025,.05,.1):
        allowed=int(np.floor(fpr*len(negative)))
        threshold=float(np.nextafter(np.sort(nscore)[::-1][allowed],np.inf))
        fa=int(((nscore>=threshold)&nreturned).sum())
        tp=int(((pscore>=threshold)&correct).sum())
        rows.append({"requested_test_fpr":fpr,"descriptive_threshold":threshold,
                     "test_false_accepts":fa,"test_negative_queries":len(negative),
                     "observed_test_fpr":fa/len(negative),"fpr_wilson_ci95":wilson(fa,len(negative)),
                     "test_accepted_correct":tp,"test_known_queries":len(positive),
                     "accepted_correct_rate":tp/len(positive)})
    return {"method":result["method"],"factor":result.get("factor",1),"points":rows,
        "interpretation":"Retrospective held-out discrimination curve at the fixed distorted-positive/clean-negative population. Thresholds are test-derived and are not validated service operating points."}

def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--results",required=True,nargs="+",type=Path);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();m=json.loads(a.protocol.read_text())
    rows=[describe(json.loads(x.read_text()),m) for x in a.results]
    a.output.write_text(json.dumps(rows,indent=2)+"\n")
    for r in rows:
        at=next(x for x in r["points"] if x["requested_test_fpr"]==.001)
        print(f"{r['method']}: descriptive TPR {at['test_accepted_correct']}/{at['test_known_queries']} at observed clean FPR {at['test_false_accepts']}/4000")

if __name__=="__main__":main()
