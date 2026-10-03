"""Summarize frozen SD-RR predictions without altering scores or native rankings.

The native release is closed-set and receives no invented gate. The separately
adapted capture-session protocol calibrates rejection only on its declared
unknown calibration sources. All released QC strata are retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from summarize_results import cluster_interval, crossed_interval, select_threshold, wilson


def count_metrics(rows, gated=False):
    keys = ["correct_reference", "localized_0_1_s", "localized_2_s"]
    if gated:
        keys += ["accepted_correct", "accepted_localized_0_1_s", "accepted_localized_2_s", "accepted_target_mismatch"]
    return {"n": len(rows), "n_sources": len({r["source_id"] for r in rows}),
            "n_creators": len({r["creator_group"] for r in rows}),
            **{key: {"successes": sum(bool(r[key]) for r in rows),
                     "rate": sum(bool(r[key]) for r in rows) / len(rows) if rows else None}
               for key in keys}}


def exposure_subset(rows, excluded):
    return [r for r in rows if r["upstream_reference_id"] not in excluded]


def summarize(result, protocol, expected_hash, exposure):
    if result["protocol_sha256"] != expected_hash:
        raise ValueError("Results do not match the frozen protocol byte hash")
    source = {q["query_id"]: q for q in protocol["queries"] if q["evaluate"]}
    predictions = result["predictions"]
    ids = [r["query_id"].split("@")[0] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(source):
        raise ValueError("Missing/duplicate/extra acoustic predictions")
    closed = bool(protocol["protocol"]["closed_set"])
    gallery = {r["reference_id"] for r in protocol["references"] if r["role"] in protocol["protocol"]["gallery_roles"]}
    rows = []
    for prediction in predictions:
        row = dict(prediction)
        query = source[row["query_id"].split("@")[0]]
        if row["role"] != query["role"] or row["source_id"] != query["source_id"] or row["target_reference_id"] != query["reference_id"]:
            raise ValueError("Prediction source/role/target differs from the frozen protocol")
        if row["reference_id"] is not None and row["reference_id"] not in gallery:
            raise ValueError("Candidate is not in the declared reference gallery")
        length = row["duration_s"]
        shift = (query["duration_s"] - length) / 2
        expected = query["expected_reference_start_s"] + shift * query["expected_time_scale"]
        if abs(row["expected_reference_start_s"] - expected) > 1e-8:
            raise ValueError("Query crop/source offset was changed")
        if not np.isfinite(row["score"]):
            raise ValueError("Non-finite query score")
        correct = row["reference_id"] == query["reference_id"]
        if row["correct_reference"] != correct:
            raise ValueError("Stored correctness differs from native candidate identity")
        error = abs(row["offset_s"] - expected) if correct else None
        row.update(correct_reference=correct, localized_0_1_s=correct and error <= .1,
                   localized_2_s=correct and error <= 2., offset_error_s=error,
                   creator_group=query["creator_group"], source_query_id=query["recording_id"],
                   recording_id=query["recording_id"], upstream_reference_id=query["reference_id"].split(":")[-1],
                   qc_stratum="upstream_expected_rank_1" if query["qc_expected_rank"] == 1 else "upstream_expected_rank_not_1")
        if row.get("raw_score") is not None and result["method"].startswith("SoundFingerprinting"):
            if row["score"] != round(row["raw_score"], 12):
                raise ValueError("Public MinHash Confidence is not canonicalized to12 decimals")
        rows.append(row)
    threshold = None
    if not closed:
        calibration = [r for r in rows if r["role"] == "calibration_unknown"]
        threshold = select_threshold(calibration, protocol["protocol"]["calibration_fpr_target"])
        for row in rows:
            accepted = row["reference_id"] is not None and row["score"] >= threshold
            row.update(accepted=accepted, accepted_correct=accepted and row["correct_reference"],
                       accepted_localized_0_1_s=accepted and row["localized_0_1_s"],
                       accepted_localized_2_s=accepted and row["localized_2_s"],
                       accepted_target_mismatch=accepted and not row["correct_reference"])
    known = [r for r in rows if r["role"] == "test_known"]
    keys = ["correct_reference", "localized_0_1_s", "localized_2_s"]
    if not closed:
        keys += ["accepted_correct", "accepted_localized_0_1_s", "accepted_localized_2_s", "accepted_target_mismatch"]
    output = {"method": result["method"], "factor": result.get("factor", 1),
        "matcher": result.get("matcher"), "duration_s_requested": result["duration_s_requested"],
        "protocol_sha256": expected_hash, "closed_set": closed,
        "native_rankings_and_scores_unchanged": True,
        "known": count_metrics(known, not closed),
        "known_source_cluster_ci": {key: cluster_interval(known, key) for key in keys},
        "known_creator_cluster_ci": {key: cluster_interval(known, key, cluster_key="creator_group") for key in keys},
        "known_source_session_crossed_ci": {key: crossed_interval(known, key) for key in keys},
        "by_recording_session": {name: count_metrics([r for r in known if r["recording_id"] == name], not closed)
                                 for name in sorted({r["recording_id"] for r in known})},
        "by_creator": {name: count_metrics([r for r in known if r["creator_group"] == name], not closed)
                       for name in sorted({r["creator_group"] for r in known})},
        "by_source": {name: count_metrics([r for r in known if r["source_id"] == name], not closed)
                      for name in sorted({r["source_id"] for r in known})},
        "by_upstream_qc": {name: count_metrics([r for r in known if r["qc_stratum"] == name], not closed)
                           for name in sorted({r["qc_stratum"] for r in known})},
        "source_exposure_sensitivities": {
            "exclude_all_fma_creator_title_matches": count_metrics(exposure_subset(known, exposure["all_fma_title_matched_sdrr_ids"]), not closed),
            "exclude_possible_nmfp_training_source_matches": count_metrics(exposure_subset(known, exposure["possible_nmfp_training_superset_sdrr_ids"]), not closed)},
        "threshold": threshold, "config": result["config"], "stats": result["stats"],
        "offset_tolerances_s": [.1, 2.], "queries_excluded_by_qc": 0,
        "timing_note": "Concurrent experimental jobs; no isolated acoustic timing claim",
        "limited_capture_population": "One speaker laptop, one phone, four capture sessions; room labels unavailable"}
    if closed:
        output["gate_note"] = "Nativeclosedset has no unknown calibration or open-set false-accept estimate"
    else:
        cal = [r for r in rows if r["role"] == "calibration_unknown"]
        unknown = [r for r in rows if r["role"] == "test_unknown"]
        hits = sum(r["accepted"] for r in unknown)
        source_events = {}
        for row in unknown:
            source_events[row["source_id"]] = source_events.get(row["source_id"], False) or row["accepted"]
        event_count = sum(source_events.values())
        source_ci = cluster_interval(unknown, "accepted")
        creator_ci = cluster_interval(unknown, "accepted", cluster_key="creator_group")
        crossed_ci = crossed_interval(unknown, "accepted")
        if hits == 0:
            for interval in (source_ci, creator_ci, crossed_ci):
                interval.update(zero_event_bootstrap_degenerate=True,
                                interpretation="A zero-event empirical bootstrap is not an upper population FPR bound")
        output.update(calibration_fpr_target=protocol["protocol"]["calibration_fpr_target"],
            calibration_unknown={"n": len(cal), "false_accepts": sum(r["accepted"] for r in cal)},
            unknown_test={"n": len(unknown), "n_sources": len({r["source_id"] for r in unknown}),
                "n_creators": len({r["creator_group"] for r in unknown}), "false_accepts": hits,
                "rate": hits / len(unknown), "wilson_query_interval": wilson(hits, len(unknown)),
                "source_cluster_ci": source_ci, "creator_cluster_ci": creator_ci,
                "source_session_crossed_ci": crossed_ci,
                "source_event_sensitivity": {"sources_with_any_false_accept": event_count,
                    "n_sources": len(source_events), "event_definition": "At least one of a source's three query clips is accepted",
                    "wilson_source_event_interval": wilson(event_count, len(source_events)),
                    "zero_event_one_sided_95_iid_source_upper": 1 - .05 ** (1 / len(source_events)) if event_count == 0 else None,
                    "interpretation": "Source-event sensitivity differs from query FPR; binomial intervals additionally assume independent sources despite shared capture sessions, and are not a device/room population guarantee"}},
            unknown_by_recording_session={name: {"n": len([r for r in unknown if r["recording_id"] == name]),
                    "false_accepts": sum(r["accepted"] for r in unknown if r["recording_id"] == name)}
                    for name in sorted({r["recording_id"] for r in unknown})},
            gate_note="Newdomain transfercalibration only on627 declared acoustic unknowns; not a population/rare-FPR guarantee")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--exposure-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    exposure = json.loads(args.exposure_audit.read_text())
    expected = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    rows = [{"result_path": str(path), "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             **summarize(json.loads(path.read_text()), protocol, expected, exposure)} for path in args.results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        metrics = row["known"]
        print(row["method"], row["duration_s_requested"], "known", metrics["n"],
              "content", metrics["correct_reference"]["successes"], "offset0.1", metrics["localized_0_1_s"]["successes"],
              "offset2", metrics["localized_2_s"]["successes"], "threshold", row["threshold"],
              "unknown", row.get("unknown_test", {}).get("false_accepts"))


if __name__ == "__main__":
    main()
