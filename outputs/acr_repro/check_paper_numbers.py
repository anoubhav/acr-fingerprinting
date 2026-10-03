"""Recompute manuscript operating points from frozen predictions and protocols."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from summarize_results import summarize

FILES = {
    "small": [
        ("ACR exact k8", "acr_default_official_from_extended/acr_d8_5s.json"),
        ("ACR IVF k8 q4", "acr_ivf_efficiency_study/acr_ivf_selected_5s.json"),
        ("Audfprint native", "audfprint_medium/audfprint_5s.json"),
        ("Audfprint candidate", "audfprint_calibrated_medium/audfprint_5s.json"),
        ("MinHash native", "soundfingerprinting_native_medium/soundfingerprinting_5s.json"),
        ("NMFP native", "nmfp_results/nmfp_native_5s.json"),
        ("PeakNet native", "peaknet_primary_results/peaknet_native_5s.json")],
    "extended": [
        ("ACR IVF k8 q1", "acr_ivf_extended/acr_ivf_qs1_extended_5s.json"),
        ("ACR IVF k8 q4", "acr_ivf_extended/acr_ivf_qs4_extended_5s.json"),
        ("Audfprint candidate", "audfprint_calibrated_extended/audfprint_5s.json"),
        ("MinHash native", "soundfingerprinting_native_extended/soundfingerprinting_5s.json"),
        ("NMFP native", "nmfp_extended_results/nmfp_native_5s.json"),
        ("NMFP IVF", "nmfp_ivf_extended_results/nmfp_native_ivf1024_p16_5s.json"),
        ("PeakNet native", "peaknet_extended_results/peaknet_native_5s.json")]
}
# Machine paths and concurrent-run timing are provenance; these fields contain
# the scientific operating points, uncertainty and frozen calibration decisions.
CHECK_KEYS = ("method", "factor", "duration_s_requested", "threshold",
              "threshold_target_calibration_fpr", "calibration_unknown",
              "calibration_known", "unknown_test", "unknown_test_populations",
              "test_known", "test_known_crossed_ci")


def derive(results: Path):
    output = {}
    for cohort, methods in FILES.items():
        protocol_path = results / ("hard_medium_protocol.json" if cohort == "small"
                                   else "extended_unknown_protocol.json")
        protocol = json.loads(protocol_path.read_text())
        output[cohort] = []
        for label, filename in methods:
            row = summarize(json.loads((results / filename).read_text()), protocol)
            output[cohort].append({"label": label, "prediction_path": filename, **row})
    return output


def equal(actual, expected, location):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError(f"{location}: keys differ")
        for key in actual:
            equal(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: lengths differ")
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f"{location}[{index}]")
    elif isinstance(actual, float):
        if not isinstance(expected, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{location}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"{location}: {actual!r} != {expected!r}")


def check(output, recorded):
    if set(output) != set(recorded):
        raise ValueError("Recorded cohort set differs")
    for cohort, rows in output.items():
        saved = {r["label"]: r for r in recorded[cohort]}
        if set(saved) != {r["label"] for r in rows}:
            raise ValueError(f"{cohort}: recorded method set differs")
        for row in rows:
            for key in CHECK_KEYS:
                equal(row[key], saved[row["label"]][key], f"{cohort}.{row['label']}.{key}")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=(root / "results" if (root / "results").is_dir()
                                 else root / "work/benchmarks"))
    parser.add_argument("--output", type=Path, default=root / "work/paper_numbers_recomputed.json")
    parser.add_argument("--check", type=Path, help="Compare operating points against saved paper_numbers.json")
    args = parser.parse_args()
    output = derive(args.results)
    if args.check:
        check(output, json.loads(args.check.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for cohort, rows in output.items():
        for row in rows:
            known = row["test_known"]["all"]
            print(cohort, row["label"], "raw", known["correct_reference"]["successes"],
                  "accepted", known["accepted_correct"]["successes"],
                  "localized", known["accepted_localized"]["successes"],
                  "target_mismatch", known["accepted_wrong"]["successes"],
                  "unknown", row["unknown_test"]["false_accepts"], "threshold", row["threshold"])
    if args.check:
        print("Recorded operating points match frozen predictions and calibration.")


if __name__ == "__main__":
    main()
