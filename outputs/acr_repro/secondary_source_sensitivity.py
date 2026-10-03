"""Frozen target-vs-secondary content sensitivity, with actual query crop bounds.

The primary labels/predictions/thresholds are unchanged. An annotated secondary
source is admissible only if its effective interval physically overlaps the
actual evaluated crop, not merely the longest 10 s parent crop.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(path, protocol, alpha, population):
    result = json.loads(Path(path).read_text())
    predictions = result["predictions"]
    query_map = {q["query_id"]: q for q in protocol["queries"]}
    montages = {}
    for q in protocol["queries"]:
        if "query_begin" in q.get("annotation", {}):
            montages.setdefault(q["source_query_id"], []).append(q)
    cal = [p for p in predictions if p["role"] == "calibration_unknown"]
    if not cal:
        raise ValueError(f"No calibration unknowns: {path}")
    threshold = float(np.nextafter(sorted([p["score"] for p in cal], reverse=True)
                     [int(np.floor(alpha * len(cal)))], np.inf))
    known = [p for p in predictions if p["role"] == "test_known"]
    counts = {"raw_target_correct": 0, "raw_annotated_source_admissible": 0,
              "accepted_target_correct": 0, "accepted_annotated_source_admissible": 0,
              "accepted_target_mismatch": 0, "accepted_mismatch_secondary_valid": 0,
              "accepted_no_annotated_source": 0}
    cases = []
    for p in known:
        base_id = p.get("base_query_id", p["query_id"].split("@")[0])
        q = query_map[base_id]
        duration = p["duration_s"]
        begin = q["start_s"] + (q["duration_s"] - duration) / 2
        end = begin + duration
        admissible = {q["reference_id"]}
        overlap = []
        for b in montages.get(q["source_query_id"], []):
            if b["reference_id"] == q["reference_id"]:
                continue
            a = b["annotation"]
            segment_end = b.get("effective_annotation_query_end_s", float(a["query_end"]))
            length = min(end, segment_end) - max(begin, float(a["query_begin"]))
            if length > 1e-6:
                admissible.add(b["reference_id"])
                overlap.append({"reference_id": b["reference_id"], "overlap_duration_s": length,
                                "annotation_query_begin_s": float(a["query_begin"]),
                                "effective_annotation_query_end_s": segment_end})
        target = p["reference_id"] == q["reference_id"]
        any_source = p["reference_id"] in admissible
        accepted = p["reference_id"] is not None and p["score"] >= threshold
        counts["raw_target_correct"] += target
        counts["raw_annotated_source_admissible"] += any_source
        counts["accepted_target_correct"] += accepted and target
        counts["accepted_annotated_source_admissible"] += accepted and any_source
        counts["accepted_target_mismatch"] += accepted and not target
        counts["accepted_mismatch_secondary_valid"] += accepted and not target and any_source
        counts["accepted_no_annotated_source"] += accepted and not any_source
        if accepted and not target:
            cases.append({"query_id": p["query_id"], "target_reference_id": q["reference_id"],
                          "predicted_reference_id": p["reference_id"], "score": p["score"],
                          "actual_crop_begin_s": begin, "actual_crop_end_s": end,
                          "annotated_secondary_overlap": overlap, "secondary_source_valid": any_source})
    return {"result_path": str(Path(path)), "method": result["method"],
            "factor": result.get("factor", 1), "population": population,
            "calibration_fpr_target": alpha, "threshold_unchanged": threshold,
            "calibration_unknown_n": len(cal), "known_n": len(known), **counts,
            "cases": cases}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary-protocol", required=True, type=Path)
    p.add_argument("--extended-protocol", required=True, type=Path)
    p.add_argument("--primary-results", required=True, nargs="+", type=Path)
    p.add_argument("--extended-results", required=True, nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    primary = json.loads(args.primary_protocol.read_text())
    extended = json.loads(args.extended_protocol.read_text())
    rows = [analyze(path, primary, .01, "official_original") for path in args.primary_results]
    rows += [analyze(path, extended, .001, "expanded_unknown") for path in args.extended_results]
    out = {"primary_protocol_unchanged": True, "prediction_ids_unchanged": True,
           "thresholds_unchanged": True,
           "sensitivity": "Identification of any source whose integer annotated interval overlaps the actual 5 s crop; localization is not relabeled",
           "annotation_precision_note": "Interval annotations are integer seconds; overlap sensitivity retains their uncertainty",
           "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print(f"{r['population']} {r['method']}: target {r['accepted_target_correct']}/{r['known_n']}, "
              f"admissible {r['accepted_annotated_source_admissible']}, "
              f"mismatch {r['accepted_target_mismatch']}, "
              f"secondary-valid {r['accepted_mismatch_secondary_valid']}, "
              f"neither {r['accepted_no_annotated_source']}")


if __name__ == "__main__":
    main()
