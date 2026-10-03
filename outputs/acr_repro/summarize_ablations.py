"""Ablation accuracy and paired uncertainty without tuning on test outcomes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def cluster_interval(rows, value_key, cluster_key, repeats=2000, seed=20261002):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row[cluster_key]].append(float(row[value_key]))
    if not clusters:
        return None
    totals = np.array([sum(values) for values in clusters.values()], dtype=float)
    counts = np.array([len(values) for values in clusters.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(totals), (repeats, len(totals)))
    samples = totals[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {"cluster_key": cluster_key, "clusters": len(clusters), "repeats": repeats,
            "estimate": float(totals.sum()/counts.sum()),
            "percentile95_ci": np.percentile(samples, [2.5, 97.5]).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--default-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summary = []
    for path in sorted(a.results.glob("*_d*_5s.json")):
        result = json.loads(path.read_text())
        predictions = result["predictions"]
        known = [r for r in predictions if r["role"] == "test_known"]
        ids = [r["query_id"] for r in predictions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate prediction IDs in {path}")
        row = {"ablation": result["ablation"], "factor": result["factor"],
               "test_known_queries": len(known), "test_known_top1": sum(r["correct_reference"] for r in known)/len(known),
               "test_known_localized": sum(r["localized_correct"] for r in known)/len(known),
               "top1_content_cluster_ci": cluster_interval(known, "correct_reference", "source_id"),
               "top1_queryfile_cluster_ci": cluster_interval(known, "correct_reference", "source_query_id"),
               "localized_queryfile_cluster_ci": cluster_interval(known, "localized_correct", "source_query_id"),
               "payload_bytes": result["stats"]["payload_bytes"],
               "measured_payload_bytes_per_hour": result["stats"]["measured_payload_bytes_per_hour"],
               "index_fingerprints": result["stats"]["fingerprints"],
               "source_result": str(path), "integrity": result["integrity"],
               "protocol_sha256": result["protocol_sha256"],
               "warning": "Closed-set top1 and localization only. No test-set tuning or calibrated rejection comparison."}
        default_path = a.default_results / f"acr_d{result['factor']}_5s.json"
        if default_path.exists():
            default = json.loads(default_path.read_text())
            if default["protocol_sha256"] != result["protocol_sha256"]:
                raise ValueError("Protocol hashes differ in paired ablation comparison")
            lookup = {r["query_id"]: r for r in default["predictions"] if r["role"] == "test_known"}
            if result.get("evaluation_cohort",{}).get("dense_test_known_limit"):
                # Dense compute controls use an immutable SHA-selected subset;
                # compare against exactly those same default predictions.
                wanted = {r["query_id"] for r in known}
                lookup = {qid:r for qid,r in lookup.items() if qid in wanted}
            if set(lookup) != set(r["query_id"] for r in known):
                raise ValueError("Paired ablation/default query sets differ")
            paired = []
            gained = lost = 0
            for pred in known:
                base = lookup[pred["query_id"]]
                delta = int(pred["correct_reference"]) - int(base["correct_reference"])
                gained += delta == 1
                lost += delta == -1
                paired.append({**pred, "accuracy_difference": delta})
            row.update(default_test_known_top1=sum(r["correct_reference"] for r in lookup.values())/len(lookup),
                paired_gained_queries=gained, paired_lost_queries=lost,
                paired_delta_content_cluster_ci=cluster_interval(paired, "accuracy_difference", "source_id"),
                paired_delta_queryfile_cluster_ci=cluster_interval(paired, "accuracy_difference", "source_query_id"))
        summary.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"rows": summary, "bootstrap_seed": 20261002,
        "inferential_scope": "Prespecified exploratory ablations on the frozen test split; intervals are descriptive and not multiplicity-adjusted. Query-file and content clusters both address dependence."}, indent=2)+"\n")
    print(f"Summarized {len(summary)} ablation result files to {a.output}")


if __name__ == "__main__":
    main()
