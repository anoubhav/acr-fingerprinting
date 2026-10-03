"""Re-derive complete digital-boundary stress counts and verify input geometry.

This verifier does not rerun retrieval or retune any threshold. With --audio it
independently decodes the219 public source montages and confirms all3340 input
PCM hashes, their prescribed geometry, and the affine ground-truth mapping.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--frozen-result", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--audio", action="store_true")
    a = p.parse_args()
    sys.path.insert(0, str(a.source_dir.resolve()))
    import numpy as np
    from acr_fp import load_audio
    from run_acr_protocol import query_crop
    plan = json.loads((a.results / "plan.json").read_text())
    summary = json.loads((a.results / "summary.json").read_text())
    protocol = json.loads(a.protocol.read_text())
    frozen = json.loads(a.frozen_result.read_text())
    ledger = json.loads((a.results / "decoded_input_ledger.json").read_text())
    plan_hash = sha(a.results / "plan.json")
    assert sha(a.protocol) == plan["protocol_sha256"]
    assert sha(a.frozen_result) == plan["frozen_result_sha256"]
    assert summary["plan_sha256"] == ledger["plan_sha256"] == plan_hash
    gate = plan["frozen_original_5s_gate"]
    assert gate == frozen["frozen_threshold"]
    original = [query_crop(q, 5) for q in protocol["queries"] if q["evaluate"]]
    assert len(original) == 835
    qmap = {q["query_id"]: q for q in original}
    gallery = {r["reference_id"] for r in protocol["references"] if r["role"] in ("calibration_known", "test_known")}
    expected_roles = Counter(q["role"] for q in original)
    previous = {r["query_id"]: r for r in frozen["predictions"]}
    config = plan["config"]
    assert config["sample_rate"] == 8000
    assert set(ledger["original_crops"]) == set(qmap)
    assert len(ledger["source_audio"]) == 219

    @lru_cache(maxsize=8)
    def decode(path):
        assert sha(path) == plan["source_audio_sha256"][path]
        y, sr = load_audio(path, 8000)
        rec = ledger["source_audio"][path]
        assert sr == rec["sample_rate"] == 8000
        assert len(y) == rec["frames"]
        assert hashlib.sha256(y.tobytes()).hexdigest() == rec["decoded_pcm_sha256"]
        return y

    derived = []
    audio_hash_checks = 0
    for condition, reported in zip(plan["conditions"], summary["rows"], strict=True):
        path = a.results / f"{condition['name']}.json"
        assert sha(path) == reported["result_sha256"]
        result = json.loads(path.read_text())
        assert result["condition"] == condition
        assert result["plan_sha256"] == plan_hash
        assert result["protocol_sha256"] == plan["protocol_sha256"]
        assert result["frozen_threshold"] == gate and result["gate_retuned"] is False
        rows = result["predictions"]
        assert len(rows) == 835 and [r["query_id"] for r in rows] == plan["query_ids"]
        assert Counter(r["role"] for r in rows) == expected_roles
        for r in rows:
            q = qmap[r["query_id"]]
            assert r["source_id"] == q["source_id"] and r["source_query_id"] == q["source_query_id"]
            assert r["role"] == q["role"] and r["target_reference_id"] == q["reference_id"]
            assert r["reference_id"] is None or r["reference_id"] in gallery
            expected = q["expected_reference_start_s"] + (condition["trim_s"]-condition["lead_s"]) * q["expected_time_scale"]
            assert r["expected_reference_start_s"] == expected
            assert r["expected_time_scale"] == q["expected_time_scale"]
            assert r["original5s_expected_reference_start_s"] == q["expected_reference_start_s"]
            assert r["input_frames"] == round(condition["output_s"] * 8000)
            assert r["leading_zero_frames"] == round(condition["lead_s"] * 8000)
            assert r["trim_frames"] == round(condition["trim_s"] * 8000)
            assert r["duration_s"] == condition["output_s"]
            assert len(r["query_physical_frame_indices"]) == r["query_fingerprints"]
            assert all(i >= 0 and i % 4 == 0 for i in r["query_physical_frame_indices"])
            assert len(set(r["query_physical_frame_indices"])) == r["query_fingerprints"]
            assert r["query_physical_frame_indices"] == sorted(r["query_physical_frame_indices"])
            assert r["empty_features"] == (r["query_fingerprints"] == 0)
            assert np.isfinite(r["score"])
            correct = r["reference_id"] == q["reference_id"]
            accepted = r["reference_id"] is not None and r["score"] >= gate
            loc = correct and abs(r["offset_s"]-expected) <= 2.
            strict = correct and abs(r["offset_s"]-expected) <= .1
            assert r["correct_reference"] == correct and r["accepted"] == accepted
            assert r["localized_correct"] == loc and r["localized_diagnostic_0p1s"] == strict
            assert r["accepted_correct"] == bool(accepted and correct)
            assert r["accepted_localized"] == bool(accepted and loc)
            assert r["accepted_localized_diagnostic_0p1s"] == bool(accepted and strict)
            assert r["accepted_wrong"] == bool(accepted and not correct)
            if condition["name"] == "unchanged_5s":
                for field in ("reference_id", "score", "offset_s", "correct_reference", "localized_correct",
                              "query_fingerprints", "vote_fraction", "mean_squared_l2", "accepted"):
                    assert r[field] == previous[r["query_id"]][field], (r["query_id"], field)
            if a.audio:
                pcm = decode(q["path"])
                start = round(q["start_s"] * 8000)
                y = pcm[start:start+40000]
                assert len(y) == 40000
                crop = ledger["original_crops"][r["query_id"]]
                assert crop["start_frame"] == start and crop["end_frame"] == start+40000 and crop["frames"] == 40000
                assert hashlib.sha256(y.tobytes()).hexdigest() == crop["crop_pcm_sha256"]
                lead, trim, frames = r["leading_zero_frames"], r["trim_frames"], r["input_frames"]
                transformed = np.r_[np.zeros(lead, dtype=np.float32), y[:frames-lead]] if lead else y[trim:trim+frames]
                assert len(transformed) == frames
                assert hashlib.sha256(transformed.tobytes()).hexdigest() == r["transformed_pcm_sha256"]
                audio_hash_checks += 1
        roles = {}
        for role in expected_roles:
            sub = [r for r in rows if r["role"] == role]
            roles[role] = {"n": len(sub), "candidate_correct": sum(r["correct_reference"] for r in sub),
                "candidate_localized_2s": sum(r["localized_correct"] for r in sub),
                "accepted_correct": sum(r["accepted_correct"] for r in sub),
                "accepted_localized_2s": sum(r["accepted_localized"] for r in sub),
                "accepted_wrong_id": sum(r["accepted_wrong"] for r in sub),
                "accepted": sum(r["accepted"] for r in sub), "no_candidate": sum(r["reference_id"] is None for r in sub),
                "empty_features": sum(r["empty_features"] for r in sub),
                "diagnostic_0p1s_candidate": sum(r["localized_diagnostic_0p1s"] for r in sub),
                "diagnostic_0p1s_accepted": sum(r["accepted_localized_diagnostic_0p1s"] for r in sub)}
        assert roles == reported["roles"]
        known = [r for r in rows if r["role"] == "test_known"]
        for metric, intervals in reported["paired_difference_vs_unchanged"].items():
            total = sum(int(r[metric])-int(previous[r["query_id"]].get(metric,
                        previous[r["query_id"]]["accepted"] and previous[r["query_id"]]["correct_reference"] if metric == "accepted_correct" else
                        previous[r["query_id"]]["accepted"] and previous[r["query_id"]]["localized_correct"] if metric == "accepted_localized" else
                        previous[r["query_id"]]["accepted"] and not previous[r["query_id"]]["correct_reference"])) for r in known)
            for kind in ("source_cluster", "source_montage_crossed"):
                assert intervals[kind]["successes"] == total
                assert intervals[kind]["estimate"] == total/543
        derived.append({"condition": condition["name"], "roles": roles})
    report = {"verified": True, "plan_sha256": plan_hash, "conditions": len(derived),
        "queries_per_condition": 835, "total_predictions": 4*835,
        "fresh_unchanged_exact_parity_rederived": True, "frozen_gate_decisions_rederived": True,
        "all_role_denominators_ground_truth_geometry_and_counts_rederived": True,
        "paired_point_estimates_rederived": True,
        "independently_redecoded_input_pcm_hashes": audio_hash_checks,
        "audio_hash_verification_complete": audio_hash_checks == 4*835 if a.audio else None,
        "no_retrieval_or_threshold_retuning": True, "rows": derived}
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
