"""Frozen official neural checkpoints on native and session-open SD-RR protocols.

Uses source-hash-verified native-protocol caches; the session protocol changes
only gallery membership and roles, never physical audio or model parameters.
Native 10s and centered 5s remain separate evaluations. Density variants use
the PEX-declared factors/rules and are not selected from acoustic test labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from neural_baseline import NMFPSequenceIndex, centered_query, load_cached_embeddings
from neural_density_controls import PhysicalPeakNetIndex, check_grid, compact_stats, density_index, sha
from summarize_acoustic_rr import summarize
from summarize_results import select_threshold


def run(args):
    import faiss
    faiss.omp_set_num_threads(args.threads)
    cache = json.loads(args.cache_index.read_text())
    native = json.loads(args.native_protocol.read_text())
    opened = json.loads(args.open_protocol.read_text())
    exposure = json.loads(args.exposure_audit.read_text())
    if cache["manifest_sha256"] != sha(args.native_protocol):
        raise ValueError("Cache belongs to a different physical native protocol")
    model_name = "NMFP-Triplet" if args.model == "nmfp" else "PeakNetFP"
    if cache["provenance"]["method"] != model_name or cache["failures"]:
        raise ValueError("Wrong model or cache failures")
    if len(cache["references"]) != len(native["references"]):
        raise ValueError("Incomplete native reference cache")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    declaration = {
        "model": model_name, "native_durations_s": [10, 5], "density_controls_durations_s": [5],
        "open_protocol_duration_s": 5, "factors": [1, 2, 4], "top_k": 20,
        "settings_from_previous_PEX_control": True, "model_refit": False,
        "test_selected_settings": False, "oracle_tempo": False, "exact_IndexFlatIP": True,
        "physical_native_protocol_sha256": sha(args.native_protocol),
        "session_open_protocol_sha256": sha(args.open_protocol), "cache_index_sha256": sha(args.cache_index),
        "exposure_audit_sha256": sha(args.exposure_audit),
        "runner_sha256": sha(__file__), "density_matcher_sha256": sha(Path(__file__).with_name("neural_density_controls.py")),
        "timing_claims": False, "declared_before_acoustic_evaluation": True,
    }
    declaration_path = args.output_dir / "declaration.json"
    declaration_path.write_text(json.dumps(declaration, indent=2) + "\n")
    summaries = []
    for study, protocol in (("native", native), ("session_open", opened)):
        gallery = {r["reference_id"] for r in protocol["references"]
                   if r["role"] in protocol["protocol"]["gallery_roles"]}
        if not gallery <= cache["references"].keys():
            raise ValueError("Missing gallery references")
        references = {rid: e for rid, e in cache["references"].items() if rid in gallery}
        paths = {rid: e["path"] for rid, e in references.items()}
        reference_cache_subset = {"references": references}
        closed = protocol["protocol"]["closed_set"]
        originals = [q for q in protocol["queries"] if q.get("evaluate", True)]
        if not closed:
            originals.sort(key=lambda q: (not q["role"].startswith("calibration"), q["query_id"]))
        for factor in (1, 2, 4):
            index = density_index(paths, factor, PhysicalPeakNetIndex if args.model == "peaknet" else NMFPSequenceIndex)
            index.build_index()
            stats = compact_stats(index, reference_cache_subset, factor)
            for duration in ([10, 5] if study == "native" and factor == 1 else [5]):
                qfactor = factor if args.model == "nmfp" else 1
                output = args.output_dir / f"{args.model}_{study}_r{factor}_q{qfactor}_{duration}s.json"
                if output.exists():
                    result = json.loads(output.read_text())
                    if result["declaration_sha256"] != sha(declaration_path):
                        raise ValueError("Existing result uses different settings or sources")
                else:
                    rows, threshold = [], None
                    for n, original in enumerate(originals):
                        # Commit calibration-only threshold before the first
                        # held-out query is searched in the session-open study.
                        if not closed and original["role"].startswith("test") and threshold is None:
                            calibration = [r for r in rows if r["role"] == "calibration_unknown"]
                            threshold = select_threshold(calibration, protocol["protocol"]["calibration_fpr_target"])
                            gate = {"threshold": threshold, "calibration_unknown_n": len(calibration),
                                    "target": protocol["protocol"]["calibration_fpr_target"],
                                    "heldout_queries_scored_before_gate": 0,
                                    "settings_selected_from_test": False}
                            (args.output_dir / f"{args.model}_gate_r{factor}_q{qfactor}_{duration}s.json").write_text(json.dumps(gate, indent=2) + "\n")
                        q = centered_query(original, duration)
                        entry = cache["queries"][q["query_id"]]
                        for field in ("start_s", "duration_s", "path", "source_id", "reference_id"):
                            if entry["item"][field] != q[field]:
                                raise ValueError(f"Physical crop/source changed: {q['query_id']} {field}")
                        emb, times, metadata = load_cached_embeddings(entry["path"])
                        if metadata["provenance"] != cache["provenance"]:
                            raise ValueError("Cached checkpoint provenance mismatch")
                        check_grid(emb, times, 0.5)
                        emb, times = np.ascontiguousarray(emb[::qfactor]), times[::qfactor]
                        check_grid(emb, times, 0.5*qfactor)
                        hit = index.search_times(emb, times, 20) if args.model == "peaknet" else index.search(emb, 20)
                        correct = hit["reference_id"] == q["reference_id"]
                        error = abs(hit["start_s"] - q["expected_reference_start_s"]) if correct else None
                        row = {
                            "query_id": q["query_id"], "base_query_id": original["query_id"],
                            "source_id": q["source_id"], "source_query_id": q["recording_id"],
                            "recording_id": q["recording_id"], "creator_group": q["creator_group"],
                            "role": q["role"], "duration_s": q["duration_s"],
                            "target_reference_id": q["reference_id"], "reference_id": hit["reference_id"],
                            "score": hit["score"] if np.isfinite(hit["score"]) else -1.0,
                            "offset_s": hit["start_s"], "expected_reference_start_s": q["expected_reference_start_s"],
                            "correct_reference": correct, "localized_correct": bool(correct and error <= 2),
                            "localized_correct_0p1s": bool(correct and error <= .1),
                            "localized_correct_2s": bool(correct and error <= 2),
                            "query_fingerprints": len(emb), "query_frame_times_s": times.tolist(),
                            "query_evidence_span_s": float(times[-1]+1),
                            "query_covered_audio_s": float(1.0 + np.minimum(np.diff(times), 1.0).sum()),
                            "unique_reference_evidence_frames": hit.get("unique_reference_evidence_frames", len(emb)),
                            "candidate_count": hit.get("candidate_count", 0),
                            "estimated_reference_time_scale": hit.get("estimated_reference_time_scale", 1.0),
                            "qc_expected_rank": q["qc_expected_rank"], "alignment_warning": q.get("alignment_warning", ""),
                            "annotation": q["annotation"], "source_query_cache_sha256": sha(entry["path"]),
                        }
                        rows.append(row)
                        if (n+1) % 300 == 0:
                            print(f"{args.model} {study} r{factor}/q{qfactor} {duration}s: {n+1}/{len(originals)}", flush=True)
                    if not closed:
                        for row in rows:
                            row["accepted"] = bool(row["reference_id"] is not None and row["score"] >= threshold)
                    result = {"method": f"{model_name} (native)" if factor == 1 else f"{model_name} (density control r{factor}/q{qfactor})",
                              "dataset": protocol["dataset"], "dataset_doi": protocol["dataset_doi"],
                              "closed_set": closed, "duration_s_requested": duration, "factor": factor,
                              "config": cache["provenance"], "stats": stats, "predictions": rows,
                              "protocol_sha256": sha(args.native_protocol if study == "native" else args.open_protocol),
                              "physical_cache_protocol_sha256": cache["manifest_sha256"],
                              "declaration_sha256": sha(declaration_path), "frozen_threshold": threshold,
                              "representation_refit_on_sdrr": False, "settings_selected_from_sdrr_test": False,
                              "matcher": {"top_k": 20, "exact": True, "candidate_index": "IndexFlatIP", "threads": args.threads,
                                          "reference_factor": factor, "query_factor": qfactor,
                                          "reference_hop_s": .5*factor, "query_hop_s": .5*qfactor,
                                          "score": "mean aligned cosine", "oracle_tempo": False,
                                          "time_scaling": "estimated physical scale in [0.5,2]" if args.model == "peaknet" else "unit only"},
                              "protocol": protocol["protocol"], "provenance": protocol["provenance"],
                              "timing_caveat": "No isolated inference, matching, or device-energy claim"}
                    output.write_text(json.dumps(result, indent=2) + "\n")
                summary = summarize(result, protocol, result["protocol_sha256"], exposure)
                summary["factor"] = factor
                summary["matcher"] = result["matcher"]
                summary["results_path"] = str(output.resolve())
                summary["results_sha256"] = sha(output)
                summaries.append(summary)
                known = summary["known"]
                print(f"Saved {output.name}: identity {known['correct_reference']['successes']}/{known['n']}, loc0.1 {known['localized_0_1_s']['successes']}, loc2 {known['localized_2_s']['successes']}, unknown {summary.get('unknown_test',{}).get('false_accepts')}", flush=True)
            del index
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("nmfp", "peaknet"), required=True)
    p.add_argument("--native-protocol", type=Path, required=True)
    p.add_argument("--open-protocol", type=Path, required=True)
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--exposure-audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    run(p.parse_args())
