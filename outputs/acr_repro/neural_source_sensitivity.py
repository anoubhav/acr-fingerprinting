"""Report a conservative unseen-source slice at the unchanged global threshold."""
import argparse
import json
from pathlib import Path
from summarize_results import summarize, cluster_interval, wilson


def sensitivity(result, protocol, audit):
    global_summary = summarize(result, protocol)
    unseen = set(audit["conservative_unseen_reference_ids"])
    known = [r for r in result["predictions"] if r["role"] == "test_known" and r["source_id"] in unseen]
    unknown = [r for r in result["predictions"] if r["role"] == "test_unknown" and r["source_id"] in unseen]
    false_accepts = sum(r["accepted"] for r in unknown)
    interpretation = audit["interpretation"]
    exposure_source = audit["training_source_statement"]
    if result["method"].startswith("PeakNetFP"):
        interpretation = interpretation.replace("NMFP", "PeakNetFP")
        exposure_source = "https://github.com/guillemcortes/peaknetfp#train-and-validation"
    return {
        "method": result["method"], "duration_s_requested": result["duration_s_requested"],
        "threshold": global_summary["threshold"], "threshold_held_fixed": True,
        "interpretation": interpretation, "training_source_statement": exposure_source,
        "gallery_unchanged": True,
        "test_known": {key: cluster_interval(known, key) for key in
                       ("correct_reference", "localized_correct", "accepted_correct", "accepted_localized", "accepted_wrong")},
        "test_unknown": {"false_accepts": false_accepts, "n": len(unknown),
                         "wilson_ci95": wilson(false_accepts, len(unknown)),
                         "source_cluster_interval": cluster_interval(unknown, "accepted")},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = sensitivity(json.loads(a.result.read_text()), json.loads(a.protocol.read_text()), json.loads(a.audit.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
