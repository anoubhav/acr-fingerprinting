"""Conservative training-source audit for the official FMA-trained NMFP model.

The NMFP authors document a training set selected entirely from FMA_medium.
Without its exact 10k train list, exclusion of every medium/small track is a
conservative unseen-source sensitivity set. This script does not relabel every
FMA_medium track as a proven training example.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import pathlib
import zipfile
from collections import Counter

FMA_METADATA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"


def audit(manifest_path, metadata_zip, output_path):
    metadata_zip = pathlib.Path(metadata_zip)
    with metadata_zip.open("rb") as f:
        sha1 = hashlib.file_digest(f, "sha1").hexdigest()
    if sha1 != FMA_METADATA_SHA1:
        raise ValueError("FMA metadata archive differs from the official repository checksum")
    subsets = {}
    with zipfile.ZipFile(metadata_zip) as z, z.open("fma_metadata/tracks.csv") as raw:
        rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
        families, fields = next(rows), next(rows)
        subset_column = next(i for i, pair in enumerate(zip(families, fields)) if pair == ("set", "subset"))
        next(rows)  # index-name header
        for row in rows:
            subsets[str(int(row[0])).zfill(6)] = row[subset_column]
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    references = []
    for ref in manifest["references"]:
        reference_id = ref["reference_id"]
        subset = subsets.get(reference_id)
        if subset is None:
            raise ValueError(f"Reference {reference_id} missing in official FMA metadata")
        references.append({"reference_id": reference_id, "role": ref["role"], "fma_subset": subset,
                           "conservative_unseen_by_nmfp": subset not in ("small", "medium")})
    unseen = {r["reference_id"] for r in references if r["conservative_unseen_by_nmfp"]}
    used_queries = [q for q in manifest["queries"] if q.get("evaluate", True)]
    result = {
        "method": "NMFP-Triplet", "metadata_zip_sha1": sha1,
        "metadata_source": "https://github.com/mdeff/fma",
        "training_source_statement": "https://github.com/raraz15/neural-music-fp/blob/main/dataset_creation/README.md",
        "interpretation": "A sensitivity analysis excluding the entire FMA_medium superset of documented NMFP training audio. Membership in medium is possible exposure, not proof of exposure. FMA-large/full tracks outside medium are conservatively source-unseen under the documented training protocol.",
        "reference_subsets": dict(Counter(r["fma_subset"] for r in references)),
        "references": references,
        "conservative_unseen_reference_ids": sorted(unseen),
        "query_counts_by_role": dict(Counter(q["role"] for q in used_queries)),
        "conservative_unseen_query_counts_by_role": dict(Counter(q["role"] for q in used_queries if q["reference_id"] in unseen)),
        "conservative_unseen_query_ids": [q["query_id"] for q in used_queries if q["reference_id"] in unseen],
    }
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k not in ("references", "conservative_unseen_reference_ids", "conservative_unseen_query_ids")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=pathlib.Path)
    p.add_argument("metadata_zip", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    args = p.parse_args()
    print(json.dumps(audit(args.manifest, args.metadata_zip, args.output), indent=2))
