"""Small end-to-end oracle for gallery roles, crops, clocks, and acoustic gates."""
from __future__ import annotations
from argparse import Namespace
import json
from pathlib import Path
import tempfile

import numpy as np

from neural_baseline import centered_query
from neural_density_controls import sha
from run_neural_acoustic import run


def test():
    Path("work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acoustic_oracle_", dir="work") as directory:
        root = Path(directory)
        rng = np.random.default_rng(21)
        provenance = {"method": "NMFP-Triplet", "synthetic_test_only": True}
        cache = {"provenance": provenance, "references": {}, "queries": {}, "failures": []}
        references, queries = [], []
        for i in range(4):
            rid = f"sdrr:{i:03d}"
            source = f"synthetic:{i}"
            reference = {"reference_id": rid, "source_id": source, "path": f"synthetic-reference-{i}", "role": "test_known"}
            query = {"query_id": f"{rid}-q1", "reference_id": rid, "source_id": source,
                     "path": f"synthetic-query-{i}", "role": "test_known", "evaluate": True,
                     "start_s": 0.0, "duration_s": 10.0, "expected_reference_start_s": 10.0,
                     "expected_time_scale": 1.0, "recording_id": f"session:{i//2}",
                     "creator_group": f"creator:{i}", "qc_expected_rank": 1,
                     "overlapping_other_annotations": [], "annotation": {"pitch": "0", "tempo": "100"}}
            references.append(reference); queries.append(query)
            frames = rng.normal(size=(60, 128)).astype(np.float32)
            frames /= np.linalg.norm(frames, axis=1, keepdims=True)
            path = root / f"reference-{i}.npz"
            np.savez(path, embeddings=frames, times=np.arange(60)*.5,
                     metadata_json=json.dumps({"provenance": provenance, "timing": {"audio_duration_s": 30.0}}))
            cache["references"][rid] = {"path": str(path.resolve()), "item": reference, "timing": {"audio_duration_s": 30.0}}
            for duration, selected in ((10, frames[20:39]), (5, frames[25:34])):
                q = centered_query(query, duration)
                path = root / f"query-{i}-{duration}.npz"
                np.savez(path, embeddings=selected, times=np.arange(len(selected))*.5,
                         metadata_json=json.dumps({"provenance": provenance, "item": q}))
                cache["queries"][q["query_id"]] = {"path": str(path.resolve()), "item": q}
        native = {"references": references, "queries": queries, "dataset": "synthetic clock oracle",
                  "dataset_doi": None, "provenance": {"synthetic_test_only": True},
                  "protocol": {"closed_set": True, "gallery_roles": ["test_known"]}}
        native_path = root / "native.json"
        native_path.write_text(json.dumps(native))
        opened = json.loads(json.dumps(native))
        roles = ["calibration_known", "calibration_unknown", "test_known", "test_unknown"]
        for collection in (opened["references"], opened["queries"]):
            for i, item in enumerate(collection):
                item["role"] = roles[i]
        opened["protocol"] = {"closed_set": False, "gallery_roles": ["calibration_known", "test_known"], "calibration_fpr_target": .01}
        open_path = root / "open.json"
        open_path.write_text(json.dumps(opened))
        cache["manifest_sha256"] = sha(native_path)
        cache_path = root / "cache.json"
        cache_path.write_text(json.dumps(cache))
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps({"all_fma_title_matched_sdrr_ids": [], "possible_nmfp_training_superset_sdrr_ids": []}))
        output = root / "results"
        run(Namespace(model="nmfp", cache_index=cache_path, native_protocol=native_path,
                      open_protocol=open_path, exposure_audit=audit_path, output_dir=output, threads=1))
        summaries = json.loads((output / "summary.json").read_text())
        assert len(summaries) == 7
        for summary in summaries:
            if summary["closed_set"]:
                assert summary["known"]["n"] == 4 and summary["threshold"] is None
                if summary["factor"] == 1:
                    assert summary["known"]["correct_reference"]["successes"] == 4
                    assert summary["known"]["localized_0_1_s"]["successes"] == 4
            else:
                assert summary["known"]["n"] == 1
                assert summary["calibration_unknown"]["n"] == 1 and summary["unknown_test"]["n"] == 1
                assert summary["stats"]["reference_count"] == 2
                gate = json.loads((output / f"nmfp_gate_r{summary['factor']}_q{summary['factor']}_5s.json").read_text())
                assert gate["heldout_queries_scored_before_gate"] == 0
        for filename in ("nmfp_native_r1_q1_10s.json", "nmfp_native_r1_q1_5s.json"):
            result = json.loads((output / filename).read_text())
            for row in result["predictions"]:
                expected = 10.0 if result["duration_s_requested"] == 10 else 12.5
                assert row["offset_s"] == expected
        print("Acoustic oracle passed: native4/4 exact physical offsets, centered crops, disjoint open gallery, calibration-only gate, and complete denominators.")


if __name__ == "__main__":
    test()
