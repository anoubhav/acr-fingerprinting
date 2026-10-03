"""Count paired candidate outcomes before duration-specific acceptance gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def compare(results_dir, durations=(1, 2, 3, 5, 10)):
    paired = {}
    protocol_sha = None
    for duration in durations:
        dense_path = results_dir / f"acr_d1_{duration:g}s.json"
        sparse_path = results_dir / f"acr_d8_{duration:g}s.json"
        if not dense_path.exists() or not sparse_path.exists():
            continue
        dense_data = json.loads(dense_path.read_text())
        sparse_data = json.loads(sparse_path.read_text())
        if dense_data["protocol_sha256"] != sparse_data["protocol_sha256"]:
            raise ValueError("Result protocols differ")
        if protocol_sha is not None and protocol_sha != dense_data["protocol_sha256"]:
            raise ValueError("Duration protocols differ")
        protocol_sha = dense_data["protocol_sha256"]
        if dense_data["factor"] != 1 or sparse_data["factor"] != 8:
            raise ValueError("Expected reference density factors 1 and 8")
        if dense_data["config"] != sparse_data["config"]:
            raise ValueError("Fingerprint configurations differ")
        # Early matrix JSONs encode the requested duration in the filename;
        # later runs also retain this explicit metadata field.
        if dense_data.get("duration_s_requested", duration) != duration or sparse_data.get("duration_s_requested", duration) != duration:
            raise ValueError("Requested durations differ")
        if not (dense_data["matcher"]["exact"] and sparse_data["matcher"]["exact"]):
            raise ValueError("Both results must use exact search")
        if any(dense_data["matcher"][key] != sparse_data["matcher"][key]
               for key in ("top_k", "tolerance_sec", "score")):
            raise ValueError("Matching rules differ")
        dense = {r["query_id"]: r for r in dense_data["predictions"]
                 if r["role"] == "test_known"}
        sparse = {r["query_id"]: r for r in sparse_data["predictions"]
                  if r["role"] == "test_known"}
        if (len(dense) != sum(r["role"] == "test_known" for r in dense_data["predictions"])
                or len(sparse) != sum(r["role"] == "test_known" for r in sparse_data["predictions"])):
            raise ValueError("Duplicate known-query IDs")
        if dense.keys() != sparse.keys() or not dense:
            raise ValueError("Paired known-query IDs differ or are empty")
        counts = {
            "known_queries": len(dense),
            "candidate_id_changes": sum(dense[k]["reference_id"] !=
                sparse[k]["reference_id"] for k in dense),
            "dense_only_correct": sum(dense[k]["correct_reference"] and
                not sparse[k]["correct_reference"] for k in dense),
            "sparse_only_correct": sum(sparse[k]["correct_reference"] and
                not dense[k]["correct_reference"] for k in dense),
            "both_correct": sum(dense[k]["correct_reference"] and
                sparse[k]["correct_reference"] for k in dense),
            "neither_correct": sum(not dense[k]["correct_reference"] and
                not sparse[k]["correct_reference"] for k in dense),
            "dense_candidate_correct": sum(r["correct_reference"] for r in dense.values()),
            "sparse_candidate_correct": sum(r["correct_reference"] for r in sparse.values()),
        }
        if sum(counts[k] for k in ("dense_only_correct", "sparse_only_correct",
                "both_correct", "neither_correct")) != len(dense):
            raise ValueError("Paired outcomes do not partition known queries")
        paired[f"{duration:g}s"] = counts
    if not paired:
        raise ValueError("No complete paired duration cells found")
    return {"schema_version": 1, "protocol_sha256": protocol_sha,
        "comparison": "Exact ACR reference density factor 1 versus factor 8",
        "population": "Frozen test_known queries; primary-source correctness",
        "note": "Unthresholded candidate statistics only. Separately calibrated acceptance gates are not applied.",
        "durations": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "paired_density_counts.json"
    output.write_text(json.dumps(compare(args.results), indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
