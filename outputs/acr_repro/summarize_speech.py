"""Speech-specific summaries: one pooled calibration rule, speaker uncertainty."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from summarize_results import select_threshold,cluster_interval,wilson

def summarize_speech(result,protocol):
    lookup={q["query_id"]:q for q in protocol["queries"]}
    rows=[]
    for p in result["predictions"]:
        q=lookup[p["query_id"].split("@")[0]]
        if not q["evaluate"] or p["role"]!=q["role"]:
            raise ValueError("Speech prediction not in source/speaker split")
        rows.append(dict(p,condition=q["condition"],speaker_id=q["speaker_id"],source_id=q["source_id"]))
    if len(rows)!=len({p["query_id"] for p in rows}):
        raise ValueError("Duplicate speech predictions")
    calibration=[p for p in rows if p["role"]=="calibration_unknown"]
    threshold=select_threshold(calibration,.01)
    for p in rows:
        p["accepted"]=p["reference_id"] is not None and p["score"]>=threshold
        p["accepted_correct"]=p["accepted"] and p["correct_reference"]
        p["accepted_localized"]=p["accepted"] and p["localized_correct"]
        p["accepted_wrong"]=p["accepted"] and not p["correct_reference"]
    known=[p for p in rows if p["role"]=="test_known"]
    unknown=[p for p in rows if p["role"]=="test_unknown"]
    def rates(k,u):
        return {"known_queries":len(k),"known_source_utterances":len({p["source_id"] for p in k}),
            "known_speakers":len({p["speaker_id"] for p in k}),
            "raw_correct":sum(p["correct_reference"] for p in k),
            "accepted_correct":sum(p["accepted_correct"] for p in k),
            "accepted_localized":sum(p["accepted_localized"] for p in k),
            "accepted_wrong_known":sum(p["accepted_wrong"] for p in k),
            "accepted_correct_utterance_ci":cluster_interval(k,"accepted_correct"),
            "accepted_correct_speaker_ci":cluster_interval(k,"accepted_correct",cluster_key="speaker_id"),
            "unknown_queries":len(u),"unknown_source_utterances":len({p["source_id"] for p in u}),
            "unknown_speakers":len({p["speaker_id"] for p in u}),
            "unknown_false_accepts":sum(p["accepted"] for p in u),
            "unknown_query_wilson_ci":wilson(sum(p["accepted"] for p in u),len(u)),
            "unknown_utterance_cluster_ci":cluster_interval(u,"accepted"),
            "unknown_speaker_cluster_ci":cluster_interval(u,"accepted",cluster_key="speaker_id")}
    conditions={c:rates([p for p in known if p["condition"]==c],
                        [p for p in unknown if p["condition"]==c])
                for c in sorted({p["condition"] for p in known})}
    return {"method":result["method"],"factor":result.get("factor",1),
            "calibration_threshold":threshold,"calibration_fpr_target":.01,
            "calibration_false_accepts":sum(p["accepted"] for p in calibration),
            "calibration_negative_queries":len(calibration),
            "calibration_note":"One pooled threshold across ten perturbations; never fitted on held-out labels",
            "pooled":rates(known,unknown),"conditions":conditions,
            "stats":result["stats"],"config":result.get("config"),
            "uncertainty_note":"Ten conditions reuse utterances; four unknown test speakers. Query-level Wilson bounds are conditional descriptive bounds, not independent-speaker population guarantees."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--results",nargs="+",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_text())
    out=[summarize_speech(json.loads(r.read_text()),protocol) for r in a.results]
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    for r in out:
        t=r["pooled"]
        print(f"{r['method']} d{r['factor']}: {t['raw_correct']}/{t['known_queries']} raw; "
              f"{t['accepted_correct']} accepted-correct; {t['unknown_false_accepts']}/{t['unknown_queries']} unknown accepts")

if __name__=="__main__":main()
