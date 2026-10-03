"""Map additional absent FMA sources to the shared 5-second NNFP crop schema."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source_manifest, destination):
    data = json.loads(Path(source_manifest).read_text())
    queries = []
    for row in data["tracks"]:
        duration = float(row["decoded_duration_s"])
        if duration < 5 or not row["valid"]:
            raise ValueError("Additional unknown source cannot yield a full 5s query")
        source_id = row["reference_id"]
        queries.append({
            "query_id": f"extra_unknown:{source_id}", "source_query_id": source_id,
            "reference_id": source_id, "source_id": source_id,
            "path": row["path"], "start_s": (duration - 5.0) / 2.0, "duration_s": 5.0,
            "role": row["role"], "evaluate": True, "annotation_index": len(queries),
            "expected_reference_start_s": 0.0, "expected_reference_end_s": 5.0,
            "expected_time_scale": 1.0, "localization_tolerance_s": 2.0,
            "overlapping_other_annotations": [], "annotation": {"tempo": "100", "pitch": ""},
            "source_attribution": row,
        })
    result = {"references": [], "queries": queries,
              "summary": {"dataset": "additional_gallery_absent_fma_medium", "query_count": len(queries)},
              "protocol": {"gallery_roles": ["test_known", "calibration_known"],
                           "task": "unknown queries against unchanged original gallery",
                           "source_manifest_sha256": hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest()},
              "source_manifest": data}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_manifest", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    result = prepare(a.source_manifest, a.destination)
    print(json.dumps(result["summary"], indent=2))
