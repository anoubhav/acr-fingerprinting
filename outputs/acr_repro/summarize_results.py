"""Compute calibrated operating points and source-cluster confidence intervals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def cluster_interval(records, key, seed=20261002, replicates=2000, cluster_key="source_id"):
    if not records:
        return None
    groups={}
    for r in records:
        groups.setdefault(r[cluster_key],[]).append(float(r[key]))
    a=np.array([[sum(x),len(x)] for x in groups.values()])
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(replicates):
        draw=a[rng.integers(0,len(a),len(a))].sum(axis=0)
        values.append(draw[0]/draw[1])
    lo,hi=np.quantile(values,[.025,.975])
    return {"estimate":sum(r[key] for r in records)/len(records),
        "ci95":[float(lo),float(hi)],"n_queries":len(records),
        "n_source_clusters":len(groups),"successes":sum(int(r[key]) for r in records)}

def crossed_interval(records,key,seed=20261002,replicates=2000):
    """Pigeonhole bootstrap: independently resample source and montage IDs.

    The two-way resampling addresses shared target sources and shared query
    montage conditions; it does not prove independent FMA compositions.
    """
    if not records:
        return None
    sources={v:i for i,v in enumerate(sorted({r["source_id"] for r in records}))}
    montages={v:i for i,v in enumerate(sorted({r["source_query_id"] for r in records}))}
    si=np.array([sources[r["source_id"]] for r in records])
    mi=np.array([montages[r["source_query_id"]] for r in records])
    y=np.array([float(r[key]) for r in records])
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(replicates):
        ws=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)))
        wm=rng.multinomial(len(montages),np.full(len(montages),1/len(montages)))
        w=ws[si]*wm[mi]
        if w.sum():
            draws.append(float((w*y).sum()/w.sum()))
    lo,hi=np.quantile(draws,[.025,.975])
    return {"estimate":float(y.mean()),"ci95":[float(lo),float(hi)],
        "n_queries":len(records),"n_source_clusters":len(sources),
        "n_montage_clusters":len(montages),"successes":int(y.sum()),
        "resampling":"source IDs and montage IDs independently; product weights"}

def wilson(successes, n, z=1.959963984540054):
    if not n:
        return None
    p=successes/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(max(0,center-half)),float(min(1,center+half))]

def select_threshold(unknown_records, target_fpr=.01):
    if not unknown_records:
        raise ValueError("No unknown calibration queries; threshold cannot be calibrated")
    scores=np.array([r["score"] for r in unknown_records],dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite calibration score")
    allowed=int(np.floor(target_fpr*len(scores)))
    threshold=float(np.nextafter(np.sort(scores)[::-1][allowed],np.inf))
    return threshold

def summarize(result, protocol):
    predictions=result["predictions"]
    ids=[r["query_id"] for r in predictions]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate query predictions")
    lookup={q["query_id"]:q for q in protocol["queries"]}
    for r in predictions:
        q=lookup[r["query_id"].split("@")[0]]
        if r["role"]!=q["role"] or not q["evaluate"]:
            raise ValueError("Prediction not in frozen source split")
        r["annotation"]=q["annotation"]
        r["overlap"]=bool(q["overlapping_other_annotations"])
        r["source_query_id"]=q.get("source_query_id",q["source_id"])
        r["negative_population"]=q.get("negative_population","official_distorted")
    cal_unknown=[r for r in predictions if r["role"]=="calibration_unknown"]
    target_fpr=float(protocol.get("protocol",{}).get("calibration_fpr_target",.01))
    threshold=select_threshold(cal_unknown,target_fpr)
    for r in predictions:
        r["accepted"]=r["reference_id"] is not None and r["score"]>=threshold
        r["accepted_correct"]=r["accepted"] and r["correct_reference"]
        r["accepted_localized"]=r["accepted"] and r["localized_correct"]
        r["accepted_wrong"]=r["accepted"] and not r["correct_reference"]
    known=[r for r in predictions if r["role"]=="test_known"]
    unknown=[r for r in predictions if r["role"]=="test_unknown"]
    cal_known=[r for r in predictions if r["role"]=="calibration_known"]
    subsets={"all":known,
        "no_pitch_or_tempo":[r for r in known if float(r["annotation"]["pitch"] or 0)==0
                              and float(r["annotation"]["tempo"] or 100)==100],
        "pitch_or_tempo":[r for r in known if float(r["annotation"]["pitch"] or 0)!=0
                              or float(r["annotation"]["tempo"] or 100)!=100],
        "no_overlap":[r for r in known if not r["overlap"]],
        "overlap":[r for r in known if r["overlap"]]}
    values={name:{key:cluster_interval(rows,key) for key in
        ("correct_reference","localized_correct","accepted_correct","accepted_localized","accepted_wrong")}
        for name,rows in subsets.items() if rows}
    fn=sum(r["accepted"] for r in unknown)
    latency=np.array([r["latency_s"] for r in known])
    return {"method":result["method"],"factor":result.get("factor",1),
        "duration_s_requested":result.get("duration_s_requested",known[0]["duration_s"]),
        "threshold":threshold,"threshold_target_calibration_fpr":target_fpr,
        "calibration_unknown":{"accepted":sum(r["accepted"] for r in cal_unknown),"n":len(cal_unknown)},
        "calibration_known":{"correct":sum(r["correct_reference"] for r in cal_known),"n":len(cal_known)},
        "unknown_test":{"false_accepts":fn,"n":len(unknown),"fpr":fn/len(unknown),
            "wilson_ci95":wilson(fn,len(unknown)),
            "source_cluster_interval":cluster_interval(unknown,"accepted"),
            "n_source_clusters":len({r["source_id"] for r in unknown})},
        "unknown_test_populations":{population:{"false_accepts":sum(r["accepted"] for r in unknown if r["negative_population"]==population),
            "n":sum(r["negative_population"]==population for r in unknown),
            "wilson_ci95":wilson(sum(r["accepted"] for r in unknown if r["negative_population"]==population),
                sum(r["negative_population"]==population for r in unknown))}
            for population in sorted({r["negative_population"] for r in unknown})},
        "test_known":values,
        "test_known_crossed_ci":{key:crossed_interval(known,key) for key in
            ("correct_reference","localized_correct","accepted_correct","accepted_localized","accepted_wrong")},
        "latency_s":{"median":float(np.median(latency)),"p95":float(np.quantile(latency,.95)),
            "p99":float(np.quantile(latency,.99)),"note":"Observed pipeline latency; concurrent experimental jobs may contend"},
        "stats":result["stats"],"config":result.get("config"),"matcher":result.get("matcher")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--results",required=True,nargs="+",type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_text())
    rows=[summarize(json.loads(path.read_text()),protocol) for path in a.results]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,indent=2)+"\n")
    for row in rows:
        k=row["test_known"]["all"]["correct_reference"]
        t=row["test_known"]["all"]["accepted_correct"]
        u=row["unknown_test"]
        print(f"{row['method']} d{row['factor']} {row['duration_s_requested']}s: "
              f"{k['successes']}/{k['n_queries']} raw; {t['successes']} accepted correct; "
              f"FPR {u['false_accepts']}/{u['n']}; threshold {row['threshold']:.5g}")

if __name__=="__main__":
    main()
