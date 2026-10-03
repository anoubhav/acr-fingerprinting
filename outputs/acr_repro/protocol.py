"""Frozen source-disjoint evaluation protocol; no test-label parameter tuning."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROLES = [(0.10, "pca_fit"), (0.15, "calibration_known"),
         (0.10, "calibration_unknown"), (0.55, "test_known"),
         (0.10, "test_unknown")]

def assign_roles(reference_ids, seed="acr-public-v1"):
    """Split by complete source track; stable under filesystem reordering."""
    ids = sorted(set(reference_ids), key=lambda x: hashlib.sha256(
        f"{seed}:{x}".encode()).digest())
    out, start = {}, 0
    cumulative = 0.0
    for proportion, role in ROLES:
        cumulative += proportion
        end = round(len(ids) * cumulative)
        for rid in ids[start:end]:
            out[rid] = role
        start = end
    assert len(out) == len(ids)
    return out

def split_manifest(manifest, seed="acr-public-v1", excluded_source_ids=()):
    excluded = set(excluded_source_ids)
    role = assign_roles([r["source_id"] for r in manifest["references"]
                         if r["source_id"] not in excluded], seed)
    for r in manifest["references"]:
        if r["source_id"] in excluded:
            role[r["source_id"]] = "development_overlap"
    refs = [dict(r, role=role[r["source_id"]]) for r in manifest["references"]]
    gallery_ids = {r["reference_id"] for r in refs
                   if r["role"] in ("calibration_known", "test_known")}
    queries = []
    for q in manifest["queries"]:
        q = dict(q, role=role[q["source_id"]])
        # A montage containing a gallery source is not a valid unknown query.
        contamination = any(o["reference_id"] in gallery_ids
                            for o in q["overlapping_other_annotations"])
        q["unknown_gallery_contamination"] = contamination
        q["evaluate"] = q["role"] not in ("pca_fit", "development_overlap") and not (
            q["role"].endswith("unknown") and contamination)
        queries.append(q)
    summary = dict(manifest["summary"], seed=seed,
                   reference_roles=dict(Counter(r["role"] for r in refs)),
                   query_roles=dict(Counter(q["role"] for q in queries if q["evaluate"])),
                   gallery_reference_count=len(gallery_ids),
                   excluded_unknown_contaminated=sum(q["role"].endswith("unknown")
                        and q["unknown_gallery_contamination"] for q in queries))
    return {"summary": summary, "references": refs, "queries": queries,
            "protocol": {"version": 1, "seed": seed,
                "split_unit": "FMA source track ID",
                "split_proportions": ROLES,
                "pca_fit_max_fingerprints_per_track": 2000,
                "pca_fit_sampling_seed": 20261002,
                "test_label_tuning": False,
                "excluded_development_source_ids": sorted(excluded.intersection(role)),
                "gallery_roles": ["calibration_known", "test_known"],
                "query_durations_s": [1, 2, 3, 5, 10],
                "decimation_factors": [1, 2, 4, 6, 8],
                "localization_tolerance_s": 2.0,
                "unknown_policy": "Exclude target from gallery and exclude crops overlapping another gallery source",
                "threshold_selection": "Calibration only; choose smallest score with empirical unknown FPR <= 0.01; show held-out FPR with numerator/denominator and uncertainty",
                "primary_task": "Adapted centered annotated-chunk content identification, not native PEX segment detection",
                "confidence_interval": "95% source-cluster percentile bootstrap; 2000 replicates, seed 20261002"}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", default="acr-public-v1")
    p.add_argument("--exclude-manifest", type=Path)
    a = p.parse_args()
    excluded = [] if not a.exclude_manifest else [r["source_id"] for r in
                     json.loads(a.exclude_manifest.read_text())["references"]]
    m = split_manifest(json.loads(a.manifest.read_text()), a.seed, excluded)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["summary"], indent=2))

if __name__ == "__main__":
    main()
