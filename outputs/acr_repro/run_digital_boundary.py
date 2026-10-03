"""Frozen digital-capture boundary stress experiment; no timing claims.

Declare the four transforms before inference. Freshly decode original public
montage audio; reuse decoded bytes only after source-hash verification. Stop if
unchanged extraction and retrieval do not exactly reproduce the frozen result.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys


CONDITIONS = (
    {"name": "unchanged_5s", "lead_s": 0., "trim_s": 0., "output_s": 5.},
    {"name": "leading_0p5s_5s", "lead_s": .5, "trim_s": 0., "output_s": 5.},
    {"name": "leading_1s_5s", "lead_s": 1., "trim_s": 0., "output_s": 5.},
    {"name": "trim_1s_4s", "lead_s": 0., "trim_s": 1., "output_s": 4.},
)
SCIENCE_FIELDS = ("reference_id", "score", "offset_s", "correct_reference",
                  "localized_correct", "query_fingerprints", "vote_fraction",
                  "mean_squared_l2", "accepted")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--frozen-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    if not 1 <= a.threads <= 4:
        raise ValueError("Use one to four FAISS CPU threads")
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.source_dir.resolve()))
    import numpy as np
    import faiss
    from threadpoolctl import threadpool_limits
    from acr_fp import Fingerprinter, audio_decoder_info, load_audio
    from compact_streaming_index import CompactTemporalIndex
    from run_acr_protocol import query_crop
    from study_query_ivf import thin_query
    from summarize_results import cluster_interval, crossed_interval, wilson
    threadpool_limits(1, user_api="blas")
    faiss.omp_set_num_threads(a.threads)
    protocol = json.loads(a.protocol.read_text())
    frozen = json.loads(a.frozen_result.read_text())
    model = Fingerprinter.load(a.model)
    c = model.config
    queries = [query_crop(q, 5) for q in protocol["queries"] if q["evaluate"]]
    gallery = [r for r in protocol["references"] if r["role"] in ("calibration_known", "test_known")]
    previous = {r["query_id"]: r for r in frozen["predictions"]}
    if len(queries) != 835 or len(gallery) != 659 or set(previous) != {q["query_id"] for q in queries}:
        raise ValueError("Original cohort changed")
    if any(q["duration_s"] != 5 for q in queries):
        raise ValueError("Expected original full5s input for every query")
    if frozen["protocol_sha256"] != sha(a.protocol):
        raise ValueError("Frozen prediction protocol differs")
    if frozen["factor"] != 8 or frozen["matcher"]["query_stride"] != 4 or frozen["matcher"]["ivf_nlist"] != 512 or frozen["matcher"]["nprobe"] != 16:
        raise ValueError("Frozen matcher differs")
    gate = frozen["frozen_threshold"]
    source_paths = sorted({q["path"] for q in queries})
    plan = {
        "schema_version": 1, "experiment": "digital-input leading-silence and initial-truncation stress",
        "conditions": CONDITIONS, "original_input_s": 5., "query_count": 835,
        "query_role_counts": dict(Counter(q["role"] for q in queries)),
        "query_ids": [q["query_id"] for q in queries], "gallery_sources": 659,
        "gallery_reference_ids": [r["reference_id"] for r in gallery],
        "source_montage_count": len(source_paths),
        "source_audio_sha256": {path: sha(path) for path in source_paths},
        "protocol_sha256": sha(a.protocol), "pca_model_sha256": sha(a.model),
        "compact_index_sha256": sha(a.index), "compact_metadata_sha256": sha(a.metadata),
        "frozen_result_sha256": sha(a.frozen_result), "runner_sha256": sha(__file__),
        "source_sha256": {name: sha(a.source_dir / name) for name in (
            "acr_fp.py", "compact_streaming_index.py", "run_acr_protocol.py", "study_query_ivf.py", "summarize_results.py")},
        "pca_refit": False, "centroid_retraining": False, "gate_retuning": False,
        "condition_selection": None, "all_conditions_mandatory": True,
        "matcher": {"reference_factor": 8, "query_stride": 4, "ivf_nlist": 512,
            "nprobe": 16, "top_k": 5, "tolerance_sec": .12, "threads": a.threads},
        "frozen_original_5s_gate": gate,
        "query_grid": "Physical original input-frame residues modulo4, preserved across silence rejection; query times remain relative to transformed input.",
        "ground_truth": "Original annotation affine map R(t)=original5s_GT+t*expected_time_scale. Leading lead seconds makes GT intercept original5s_GT-lead*scale; trimming trim seconds makes it original5s_GT+trim*scale. Transform does not change scale.",
        "localization_primary_s": 2., "localization_diagnostic_s": .1,
        "diagnostic_caveat": "Publisher segment boundaries have integer-second annotations;0.1s is annotation-agreement sensitivity, not measured synchronization accuracy.",
        "uncertainty": "2000 paired bootstrap resamples with seed20261002; source-only and crossed source/montage IDs; conditional on frozen model/settings/gate. Report every condition, no multiplicity-based selection.",
        "unchanged_gate": "Before transformed inference, every one of835 unchanged top1 IDs/scores/offsets/evidence/accepted decisions must equal canonical frozen result exactly; mismatch stops execution.",
        "interpretation": "Post-hoc prespecified synthetic digital-input capture stress on public PEX audio. Not actual TV capture, deployment validation, production impact, or a new acoustic benchmark.",
        "timing_claim": False, "config": asdict(c),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "faiss": faiss.__version__, "audio_decoder": audio_decoder_info()},
    }
    plan_path = a.output / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        compare = dict(existing)
        compare.pop("created_utc")
        # JSON arrays replace tuple constants when reading.
        if compare != json.loads(json.dumps(plan)):
            raise ValueError("Frozen plan inputs or implementation changed; use a fresh output directory")
        plan = existing
    else:
        plan["created_utc"] = datetime.now(timezone.utc).isoformat()
        save(plan_path, plan)
    plan_hash = sha(plan_path)
    if a.plan_only:
        print(json.dumps({"plan": str(plan_path), "plan_sha256": plan_hash,
            "queries": len(queries), "montages": len(source_paths), "frozen_gate": gate}), flush=True)
        return
    compact = CompactTemporalIndex.load(a.index, a.metadata, nprobe=16)
    if compact.factor != 8 or compact.content_ids != [r["reference_id"] for r in gallery]:
        raise ValueError("Compact gallery differs from protocol")
    if compact._faiss.nlist != 512:
        raise ValueError("Expected frozen IVF512")
    model.assert_disjoint([r["reference_id"] for r in gallery], [q["reference_id"] for q in queries])
    decoded_ledger = {}
    crop_ledger = {}

    @lru_cache(maxsize=8)
    def decode(path):
        if sha(path) != plan["source_audio_sha256"][path]:
            raise ValueError("Audio source changed after plan freeze")
        audio, sr = load_audio(path, c.sample_rate)
        if sr != c.sample_rate or not np.isfinite(audio).all():
            raise ValueError("Unexpected decoder output")
        rec = {"file_sha256": plan["source_audio_sha256"][path], "sample_rate": sr,
            "frames": len(audio), "decoded_pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest()}
        if path in decoded_ledger and decoded_ledger[path] != rec:
            raise ValueError("Repeated source decoding differs")
        decoded_ledger[path] = rec
        return audio

    def crop(q):
        audio = decode(q["path"])
        start = round(q["start_s"] * c.sample_rate)
        end = start + round(q["duration_s"] * c.sample_rate)
        y = audio[start:end]
        if len(y) != 40000:
            raise ValueError(f"Short or wrong original crop: {q['query_id']}")
        rec = {"path": q["path"], "start_frame": start, "end_frame": end, "frames": len(y),
            "crop_pcm_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
        if q["query_id"] in crop_ledger and crop_ledger[q["query_id"]] != rec:
            raise ValueError("Repeated original crop differs")
        crop_ledger[q["query_id"]] = rec
        return y

    def infer(q, condition, y):
        lead = round(condition["lead_s"] * c.sample_rate)
        trim = round(condition["trim_s"] * c.sample_rate)
        wanted = round(condition["output_s"] * c.sample_rate)
        if lead:
            transformed = np.concatenate([np.zeros(lead, dtype=y.dtype), y[:wanted-lead]])
        else:
            transformed = y[trim:trim+wanted]
        if len(transformed) != wanted:
            raise ValueError("Wrong transformed geometry")
        raw, dense_t = model.raw_features(transformed, c.sample_rate)
        x = model.transform(raw)
        x, t = thin_query(x, dense_t, 4, c)
        original_frames = np.rint((t-c.first_center_seconds)/(c.hop_length/c.sample_rate)).astype(np.int64)
        if np.any(original_frames % 4) or (len(t) and np.max(np.abs(t-(original_frames*c.hop_length/c.sample_rate+c.first_center_seconds))) > 1e-8):
            raise ValueError("Query clock/residue changed by silence gating")
        hits = compact.match(x, t, top_k=5, tolerance_sec=.12, limit=1)
        hit = hits[0] if hits else {}
        rid, offset = hit.get("content_id"), hit.get("offset_sec")
        expected = q["expected_reference_start_s"] + (condition["trim_s"]-condition["lead_s"]) * q["expected_time_scale"]
        correct = rid == q["reference_id"]
        accepted = rid is not None and hit.get("score", 0.) >= gate
        return {"query_id": q["query_id"], "base_query_id": q["query_id"].rsplit("@", 1)[0],
            "condition": condition["name"], "source_id": q["source_id"], "source_query_id": q["source_query_id"],
            "role": q["role"], "duration_s": condition["output_s"], "target_reference_id": q["reference_id"],
            "reference_id": rid, "score": hit.get("score", 0.), "offset_s": offset,
            "original5s_expected_reference_start_s": q["expected_reference_start_s"],
            "expected_reference_start_s": expected, "expected_time_scale": q["expected_time_scale"],
            "correct_reference": correct, "localized_correct": bool(correct and abs(offset-expected) <= 2.),
            "localized_diagnostic_0p1s": bool(correct and abs(offset-expected) <= .1),
            "accepted": accepted, "accepted_correct": bool(accepted and correct),
            "accepted_localized": bool(accepted and correct and abs(offset-expected) <= 2.),
            "accepted_localized_diagnostic_0p1s": bool(accepted and correct and abs(offset-expected) <= .1),
            "accepted_wrong": bool(accepted and not correct), "query_fingerprints": len(x),
            "dense_usable_query_fingerprints": len(raw), "empty_features": len(x) == 0,
            "query_physical_frame_indices": original_frames.tolist(),
            "vote_fraction": hit.get("vote_fraction", 0.), "mean_squared_l2": hit.get("mean_distance"),
            "transformed_pcm_sha256": hashlib.sha256(transformed.tobytes()).hexdigest(),
            "input_frames": len(transformed), "leading_zero_frames": lead, "trim_frames": trim,
            "annotation": q["annotation"], "overlap": bool(q["overlapping_other_annotations"])}

    baseline_rows = []
    baseline_condition = CONDITIONS[0]
    for q in queries:
        row = infer(q, baseline_condition, crop(q))
        old = previous[q["query_id"]]
        for field in SCIENCE_FIELDS:
            if row[field] != old[field]:
                save(a.output / "unchanged_mismatch.json", {"query_id": q["query_id"], "field": field,
                    "fresh": row[field], "canonical": old[field], "plan_sha256": plan_hash})
                raise ValueError(f"Fresh unchanged result differs: {q['query_id']} {field}")
        baseline_rows.append(row)
        if len(baseline_rows) % 100 == 0:
            print(f"Fresh unchanged equivalence {len(baseline_rows)}/835", flush=True)
    save(a.output / "unchanged_parity.json", {"query_count": len(baseline_rows), "compared_fields": SCIENCE_FIELDS,
        "all835_exactly_equal": True, "frozen_result_sha256": plan["frozen_result_sha256"],
        "plan_sha256": plan_hash, "completed_utc": datetime.now(timezone.utc).isoformat()})
    print("All835 fresh unchanged results exactly equal; transformed inference now permitted", flush=True)
    all_rows = {baseline_condition["name"]: baseline_rows}
    for condition in CONDITIONS[1:]:
        rows = []
        for q in queries:
            rows.append(infer(q, condition, crop(q)))
            if len(rows) % 100 == 0:
                save(a.output / f"{condition['name']}_checkpoint.json", {"plan_sha256": plan_hash, "predictions": rows})
                print(f"{condition['name']}: {len(rows)}/835", flush=True)
        all_rows[condition["name"]] = rows
        save(a.output / f"{condition['name']}.json", {"method": "ACR-PCA compact frozen IVF/query stride digital-boundary stress",
            "condition": condition, "config": asdict(c), "matcher": plan["matcher"],
            "protocol_sha256": plan["protocol_sha256"], "plan_sha256": plan_hash,
            "frozen_threshold": gate, "gate_retuned": False, "timing_claim": False, "predictions": rows})
    save(a.output / "unchanged_5s.json", {"method": "ACR-PCA compact frozen IVF/query stride fresh unchanged conformance",
        "condition": baseline_condition, "config": asdict(c), "matcher": plan["matcher"],
        "protocol_sha256": plan["protocol_sha256"], "plan_sha256": plan_hash,
        "frozen_threshold": gate, "gate_retuned": False, "timing_claim": False, "predictions": baseline_rows})
    save(a.output / "decoded_input_ledger.json", {"plan_sha256": plan_hash, "source_audio": decoded_ledger, "original_crops": crop_ledger})
    baseline_map = {r["query_id"]: r for r in baseline_rows}
    summaries = []
    metrics = ("correct_reference", "localized_correct", "accepted_correct", "accepted_localized", "accepted_wrong")
    for condition in CONDITIONS:
        rows = all_rows[condition["name"]]
        by_role = {}
        for role in ("calibration_known", "calibration_unknown", "test_known", "test_unknown"):
            sub = [r for r in rows if r["role"] == role]
            by_role[role] = {"n": len(sub), "candidate_correct": sum(r["correct_reference"] for r in sub),
                "candidate_localized_2s": sum(r["localized_correct"] for r in sub),
                "accepted_correct": sum(r["accepted_correct"] for r in sub),
                "accepted_localized_2s": sum(r["accepted_localized"] for r in sub),
                "accepted_wrong_id": sum(r["accepted_wrong"] for r in sub), "accepted": sum(r["accepted"] for r in sub),
                "no_candidate": sum(r["reference_id"] is None for r in sub), "empty_features": sum(r["empty_features"] for r in sub),
                "diagnostic_0p1s_candidate": sum(r["localized_diagnostic_0p1s"] for r in sub),
                "diagnostic_0p1s_accepted": sum(r["accepted_localized_diagnostic_0p1s"] for r in sub)}
        known = [r for r in rows if r["role"] == "test_known"]
        unknown = [r for r in rows if r["role"] == "test_unknown"]
        paired = {}
        for metric in metrics:
            diffs = [{"source_id": r["source_id"], "source_query_id": r["source_query_id"],
                      "difference": int(r[metric])-int(baseline_map[r["query_id"]][metric])} for r in known]
            paired[metric] = {"source_cluster": cluster_interval(diffs, "difference"),
                "source_montage_crossed": crossed_interval(diffs, "difference")}
        unknown_diffs = [{"source_id": r["source_id"], "source_query_id": r["source_query_id"],
            "difference": int(r["accepted"])-int(baseline_map[r["query_id"]]["accepted"])} for r in unknown]
        summary = {"condition": condition, "frozen_threshold": gate, "roles": by_role,
            "test_known_uncertainty": {metric: {"source_cluster": cluster_interval(known, metric),
                "source_montage_crossed": crossed_interval(known, metric)} for metric in metrics},
            "paired_difference_vs_unchanged": paired,
            "unknown_false_accepts": {"n": len(unknown), "false_accepts": sum(r["accepted"] for r in unknown),
                "wilson_ci95": wilson(sum(r["accepted"] for r in unknown), len(unknown)),
                "source_cluster": cluster_interval(unknown, "accepted"), "source_montage_crossed": crossed_interval(unknown, "accepted"),
                "paired_source_montage_crossed": crossed_interval(unknown_diffs, "difference")},
            "candidate_ids_changed_all835": sum(r["reference_id"] != baseline_map[r["query_id"]]["reference_id"] for r in rows),
            "candidate_ids_changed_test_known": sum(r["reference_id"] != baseline_map[r["query_id"]]["reference_id"] for r in known),
            "query_fingerprints_test_known": {"min": min(r["query_fingerprints"] for r in known),
                "median": float(np.median([r["query_fingerprints"] for r in known])), "max": max(r["query_fingerprints"] for r in known)},
            "result_sha256": sha(a.output / f"{condition['name']}.json")}
        summaries.append(summary)
        print(f"{condition['name']}: raw{by_role['test_known']['candidate_correct']}/543; accepted{by_role['test_known']['accepted_correct']}; localized{by_role['test_known']['accepted_localized_2s']}; unknown{by_role['test_unknown']['accepted']}/71", flush=True)
    save(a.output / "summary.json", {"plan_sha256": plan_hash, "all4_conditions_complete": True,
        "all835_fresh_unchanged_exact_parity": True, "gate_retuned": False, "rows": summaries,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "timing_claim": False,
        "interpretation": plan["interpretation"], "diagnostic_caveat": plan["diagnostic_caveat"]})
    for f in a.output.glob("*_checkpoint.json"):
        f.unlink()


if __name__ == "__main__":
    main()
