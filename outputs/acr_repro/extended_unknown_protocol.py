"""Combine fixed distorted positives with a separately labeled clean OOV study."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True,type=Path)
    p.add_argument("--extra",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    base=json.loads(a.base.read_text())
    extra=json.loads(a.extra.read_text())
    allbaseids={r["reference_id"] for r in base["references"]}
    if allbaseids.intersection(t["reference_id"] for t in extra["tracks"]):
        raise ValueError("Extra negatives overlap gallery/fit/source roles")
    queries=[dict(q,negative_population="official_distorted" if q["role"].endswith("unknown") else None)
             for q in base["queries"]]
    for t in extra["tracks"]:
        rid=t["reference_id"]
        start=(t["decoded_duration_s"]-5)/2
        queries.append({"query_id":f"fma_medium_unknown:{rid}","source_query_id":f"fma_medium_unknown:{rid}",
            "reference_id":rid,"source_id":rid,"path":t["path"],"start_s":start,"duration_s":5.0,
            "expected_reference_start_s":start,"expected_reference_end_s":start+5,
            "expected_time_scale":1.0,"localization_tolerance_s":2.0,
            "role":t["role"],"evaluate":True,"overlapping_other_annotations":[],
            "annotation":{"tempo":"100","pitch":"","condition":"extra_clean_oov"},
            "negative_population":"additional_clean_FMA","artist_id":t["artist_id"],
            "audio_sha256":t["sha256"],"source_file_duration_s":t["decoded_duration_s"]})
    protocol=dict(base["protocol"],extended_unknown_study=True,
        calibration_fpr_target=.001,
        additional_unknown_source_count=extra["selected_count"],
        additional_unknown_calibration_count=extra["calibration_count"],
        additional_unknown_test_count=extra["test_count"],
        positive_population="Unchanged official distorted PEX crops",
        negative_population="Official distorted unknowns plus separately reported unmodified gallery-absent FMA medium crops",
        unknown_claim_scope="Source IDs absent from gallery; no guarantee of distinct compositions or remasters")
    summary=dict(base["summary"],query_roles=dict(Counter(q["role"] for q in queries if q["evaluate"])),
                 additional_unknown_study=extra["selected_count"])
    out={"references":base["references"],"queries":queries,"summary":summary,"protocol":protocol}
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
