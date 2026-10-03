"""Measure native and externally calibrated landmark variants on shared hashes.

The candidate variant changes only upstream's internal minimum-count floor from
5 to1. External thresholds are selected later using calibration unknowns only.
Both variants are reported, with no selection using test outcomes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from audfprint_baseline import AudfprintBaseline
from run_audfprint_protocol import shorten_query


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--native-output-dir", required=True, type=Path)
    p.add_argument("--calibrated-output-dir", required=True, type=Path)
    p.add_argument("--durations", nargs="+", type=float, default=[5, 1, 2, 3, 10])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    manifest = json.loads(args.protocol.read_text())
    protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    gallery = [r for r in manifest["references"] if r["role"] in manifest["protocol"]["gallery_roles"]]
    queries = [q for q in manifest["queries"] if q["evaluate"]]
    for directory in (args.native_output_dir, args.calibrated_output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    baseline = AudfprintBaseline(args.upstream, args.cache, min_count=1)
    baseline.fit(gallery, args.workers)
    native_matcher = copy.copy(baseline.matcher)
    native_matcher.threshcount = 5
    for duration in args.durations:
        native, candidate = [], []
        for i, original in enumerate(queries):
            query = shorten_query(original, duration)
            hashes, extraction_stats = baseline.query_features(query)
            # Alternate order to avoid consistently favoring either variant's
            # data/cache locality when measuring search wall-clock time.
            calls = [(native_matcher, native), (baseline.matcher, candidate)]
            if i % 2:
                calls.reverse()
            for matcher, rows in calls:
                result = baseline.match_query_features(query, hashes, extraction_stats, matcher)
                result.update(role=query["role"], target_reference_id=query["reference_id"],
                              source_id=query["source_id"], duration_s=query["duration_s"],
                              expected_reference_start_s=query["expected_reference_start_s"],
                              correct_reference=result["reference_id"] == query["reference_id"])
                result["localized_correct"] = result["correct_reference"] and abs(
                    result["offset_s"] - query["expected_reference_start_s"]) <= 2
                if "condition" in query:
                    result.update(condition=query["condition"], speaker_id=query["speaker_id"])
                rows.append(result)
            if (i + 1) % 100 == 0:
                print(f"audfprint dual {duration:g}s queried {i + 1}/{len(queries)}", flush=True)
        for rows, floor, method, directory in (
                (native, 5, "audfprint", args.native_output_dir),
                (candidate, 1, "audfprint_calibrated", args.calibrated_output_dir)):
            config = dict(baseline.config, min_count=floor)
            result = {"method": method, "duration_s_requested": duration,
                      "config": config, "stats": baseline.stats,
                      "protocol": manifest["protocol"], "protocol_sha256": protocol_sha256,
                      "query_frontend_shared_across_variants": True,
                      "latency_note": "latency_s excludes separately timed native FFmpeg decode; feature/hash extraction plus matching, aligned with ACR evaluator",
                      "predictions": rows}
            path = directory / f"audfprint_{duration:g}s.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
