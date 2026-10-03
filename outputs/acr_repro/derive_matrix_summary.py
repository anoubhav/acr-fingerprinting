"""Recalculate all 25 exact duration/density cells without downloading audio."""
from pathlib import Path
import argparse
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def derive(results):
    summary = {}
    protocol_hash = None
    config = None
    for factor in (1, 2, 4, 6, 8):
        for duration in (1, 2, 3, 5, 10):
            path = results / "acr_medium" / f"acr_d{factor}_{duration}s.json"
            data = json.loads(path.read_text())
            if protocol_hash is None:
                protocol_hash, config = data["protocol_sha256"], data["config"]
            assert data["protocol_sha256"] == protocol_hash, path
            assert data["config"] == config and data["factor"] == factor, path
            # Earlier frozen runs record the request in the filename and actual
            # duration per prediction, without a duplicated top-level field.
            assert data.get("duration_s_requested", duration) == duration, path
            rows = data["predictions"]
            assert len(rows) == 835 and len({r["query_id"] for r in rows}) == 835, path
            assert all(math.isfinite(r["score"]) for r in rows), path
            assert all(0 < r["duration_s"] <= duration + 1e-9 for r in rows), path
            calibration = [r["score"] for r in rows if r["role"] == "calibration_unknown"]
            known = [r for r in rows if r["role"] == "test_known"]
            unknown = [r for r in rows if r["role"] == "test_unknown"]
            assert (len(calibration), len(known), len(unknown)) == (74, 543, 71), path
            # With 74 negatives, a 1% empirical target permits zero accepts.
            gate = float(np.nextafter(max(calibration), np.inf))
            accepted = lambda r: r["reference_id"] is not None and r["score"] >= gate
            summary[f"k{factor}_{duration}s"] = {
                "candidate": sum(r["correct_reference"] for r in known),
                "accepted": sum(r["correct_reference"] and accepted(r) for r in known),
                "unknown": sum(accepted(r) for r in unknown),
                "signal_mb_per_hour": data["stats"]["measured_payload_bytes_per_hour"] / 1e6,
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="Compare with the packaged paper_matrix_counts.json instead of writing.")
    args = parser.parse_args()
    summary = derive(args.results)
    output = args.output or args.results / "paper_matrix_counts.json"
    if args.check:
        expected = json.loads(output.read_text())
        assert expected.keys() == summary.keys()
        for key, row in summary.items():
            for field, value in row.items():
                assert math.isclose(value, expected[key][field], rel_tol=1e-12, abs_tol=1e-12), (key, field)
        print("All 25 exact duration/density cells reproduce the paper's counts and payloads.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
