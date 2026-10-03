"""Verify density-matcher clock semantics and full-density native parity.

Real parity compares all original 835 queries against the current corrected
native adapter and records differences from historical exact-search output.
Synthetic clock tests use only
random vectors and declared physical times; no benchmark labels affect settings.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from neural_baseline import NMFPSequenceIndex, load_cached_embeddings
from neural_density_controls import PhysicalPeakNetIndex, check_grid, density_index, sha
from peaknet_baseline import PeakNetSequenceIndex


def verify(root, output, threads=4):
    import faiss
    faiss.omp_set_num_threads(threads)
    report = {}
    for model, oldclass, refs, queries, native in (
        ("nmfp", NMFPSequenceIndex, "nmfp_cache/cache_index.json", "nmfp_cache/query_cache_index.json", "nmfp_results/nmfp_native_5s.json"),
        ("peaknet", PeakNetSequenceIndex, "peaknet_cache/cache_index.json", "peaknet_cache/cache_index.json", "peaknet_primary_results/peaknet_native_5s.json"),
    ):
        refs, queries, native = [root / p for p in (refs, queries, native)]
        historical_path = native if model == "nmfp" else root.parent / "audit/peaknet_catalog_rounding/peaknet_primary_results/peaknet_native_5s.json"
        rc, qc = json.loads(refs.read_text()), json.loads(queries.read_text())
        rows = json.loads(native.read_text())["predictions"]
        old_rows = {p["query_id"]: p for p in json.loads(historical_path.read_text())["predictions"]}
        index = density_index({rid: e["path"] for rid, e in rc["references"].items()}, 1,
                              PhysicalPeakNetIndex if model == "peaknet" else NMFPSequenceIndex)
        index.build_index()
        old = oldclass(index.reference_ids, index.embeddings, index.starts, index.ends, 0.5)
        old._index = index._index
        mismatches, historical, max_error = [], [], 0.0
        for p in rows:
            e, times, _ = load_cached_embeddings(qc["queries"][p["query_id"]]["path"])
            new = index.search_times(e, times, 20) if model == "peaknet" else index.search(e, 20)
            unchanged = old.search(e, 20)
            error = abs(new["score"] - unchanged["score"])
            max_error = max(max_error, error)
            if new["reference_id"] != unchanged["reference_id"] or new["start_s"] != unchanged["start_s"] or error > 1e-7:
                mismatches.append(p["query_id"])
            p_old = old_rows[p["query_id"]]
            if new["reference_id"] != p_old["reference_id"] or new["start_s"] != p_old["offset_s"] or abs(new["score"] - p_old["score"]) > 1e-7:
                historical.append(p["query_id"])
        report[model] = {"rows": len(rows), "current_native_mismatches": mismatches,
                         "historical_result_mismatches": historical, "max_score_error": max_error,
                         "reference_cache_sha256": sha(refs), "query_cache_sha256": sha(queries),
                         "historical_result_sha256": sha(historical_path),
                         "current_result_sha256": sha(native),
                         "historical_deltas_expected_for_repaired_Peak_clock": model == "peaknet"}
        del index, old
    rng = np.random.default_rng(13)
    vectors = rng.normal(size=(80, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    clocks = []
    for reference_hop in (0.5, 1.0, 2.0):
        index = PhysicalPeakNetIndex(["synthetic"], vectors, np.array([0]), np.array([80]), reference_hop)
        index.audit_times = np.arange(80) * reference_hop
        index.build_index()
        for actual_scale in (0.75, 1.0, 1.5):
            query_times = np.arange(9) * 0.5
            origin = 20 * reference_hop
            frame_ids = np.rint((origin + query_times * actual_scale) / reference_hop).astype(int)
            hit = index.search_times(vectors[frame_ids], query_times, 1)
            if hit["reference_id"] != "synthetic" or hit["start_s"] != origin:
                raise AssertionError("Physical clock localization failed")
            clocks.append({"reference_hop_s": reference_hop, "query_hop_s": 0.5,
                           "physical_start_s": origin, "actual_reference_time_scale": actual_scale,
                           "estimated_reference_time_scale": hit["estimated_reference_time_scale"],
                           "matched_start_s": hit["start_s"], "note": "Nearest-grid aliasing need not recover exact scale; localization must use physical seconds."})
    try:
        check_grid(vectors[:5], np.arange(5) * 1.0, 0.5)
    except ValueError:
        wrong_hop_rejected = True
    else:
        wrong_hop_rejected = False
    report["clock_tests"] = clocks
    report["wrong_hop_rejected"] = wrong_hop_rejected
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    assert wrong_hop_rejected
    assert not any(report[m]["current_native_mismatches"] for m in ("nmfp", "peaknet"))
    assert not report["nmfp"]["historical_result_mismatches"]
    print(f"Verified 835 queries per model plus nine physical-clock cases; saved {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", type=Path, default=Path("work/benchmarks"))
    p.add_argument("--output", type=Path, default=Path("work/benchmarks/neural_density_parity.json"))
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    verify(args.benchmark_root, args.output, args.threads)
