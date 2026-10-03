"""Paired exact-ACR versus frozen neural density controls, with exposure slice."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from neural_density_controls import sha
from summarize_results import crossed_interval, summarize


def compare(args):
    protocol = json.loads(args.protocol.read_text())
    eligible = {q["query_id"] for q in protocol["queries"] if q.get("evaluate", True)}
    audit = json.loads(args.overlap_audit.read_text())
    unseen = set(audit["conservative_unseen_reference_ids"])
    acr = json.loads(args.acr_result.read_text())
    if acr["protocol_sha256"] != sha(args.protocol):
        raise ValueError("ACR result protocol differs")
    acr_summary = summarize(acr, protocol)
    A = {q["query_id"].split("@")[0]: q for q in acr["predictions"] if q["role"] == "test_known"}
    output = {"comparison": "ACR exact factor8 minus neural sparse controls",
              "protocol_sha256": sha(args.protocol), "acr_result_sha256": sha(args.acr_result),
              "overlap_audit_sha256": sha(args.overlap_audit),
              "note": "Paired crossed source/montage bootstrap with2000 fixed-seed replicates. Raw/localization are unthresholded; acceptance uses each original74-negative calibration gate. Held-out FPRs need not match. The conservative source slice changes neither galleries nor global thresholds.",
              "acr_original_unknown_test": acr_summary["unknown_test"], "rows": []}
    for model in ("nmfp", "peaknet"):
        for factor in (2, 4):
            qfactor = factor if model == "nmfp" else 1
            path = args.density_root / model / f"{model}_density_r{factor}_q{qfactor}_5s.json"
            result = json.loads(path.read_text())
            result["predictions"] = [q for q in result["predictions"] if q["base_query_id"] in eligible]
            summary = summarize(result, protocol)
            B = {q["query_id"].split("@")[0]: q for q in result["predictions"] if q["role"] == "test_known"}
            if A.keys() != B.keys():
                raise ValueError("Known-query pairing differs")
            for key in A:
                for field in ("source_id", "role", "target_reference_id", "duration_s", "expected_reference_start_s"):
                    if A[key][field] != B[key][field]:
                        raise ValueError(f"Paired query source/role/crop/target changed: {key} {field}")
            row = {"method": result["method"], "results_sha256": sha(path),
                   "original_threshold": summary["threshold"], "original_unknown_test": summary["unknown_test"],
                   "differences": {}, "conservative_unseen": {"global_threshold_unchanged": True, "gallery_unchanged": True, "differences": {}}}
            for metric in ("correct_reference", "localized_correct", "accepted_correct", "accepted_localized"):
                records = [{"source_id": A[k]["source_id"], "source_query_id": A[k]["source_query_id"],
                            "difference": int(A[k][metric])-int(B[k][metric])} for k in A]
                paired = crossed_interval(records, "difference")
                paired["net_query_difference"] = paired.pop("successes")
                row["differences"][metric] = paired
                records = [r for r in records if r["source_id"] in unseen]
                paired = crossed_interval(records, "difference")
                paired["net_query_difference"] = paired.pop("successes")
                row["conservative_unseen"]["differences"][metric] = paired
            row["conservative_unseen"]["acr_counts"] = {m: sum(A[k][m] for k in A if A[k]["source_id"] in unseen)
                for m in ("correct_reference", "localized_correct", "accepted_correct", "accepted_localized")}
            row["conservative_unseen"]["neural_counts"] = {m: sum(B[k][m] for k in B if B[k]["source_id"] in unseen)
                for m in ("correct_reference", "localized_correct", "accepted_correct", "accepted_localized")}
            output["rows"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    for row in output["rows"]:
        print(row["method"], "unseen", row["conservative_unseen"]["neural_counts"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=Path("work/benchmarks/hard_medium_protocol.json"))
    p.add_argument("--acr-result", type=Path, default=Path("work/benchmarks/acr_medium/acr_d8_5s.json"))
    p.add_argument("--density-root", type=Path, default=Path("work/benchmarks/neural_density_controls"))
    p.add_argument("--overlap-audit", type=Path, default=Path("work/benchmarks/nmfp_training_overlap_audit.json"))
    p.add_argument("--output", type=Path, default=Path("work/benchmarks/neural_density_controls/paired_acr_neural_primary.json"))
    compare(p.parse_args())
