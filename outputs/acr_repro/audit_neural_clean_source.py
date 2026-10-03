"""Calibrated-cohort clean-source and heterogeneous-batch parity sanity audit.

This audits implementation correctness without adjusting any model/matcher
parameters. Query audio comes directly from declared reference positions for
20 hash-selected calibration-known clips. It is not an acoustic performance
row and does not replace or filter any released microphone query.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from neural_baseline import NMFPExtractor, centered_query, load_cached_embeddings
from neural_density_controls import PhysicalPeakNetIndex, density_index, sha


def audit(args):
    root = Path(__file__).resolve().parents[2]
    # Use the same FAISS build as the main matching environment after loading
    # this environment's NumPy/TensorFlow; no model conversion occurs.
    sys.path.append(str(root / "work/venv/lib/python3.12/site-packages"))
    import faiss
    faiss.omp_set_num_threads(2)
    cache = json.loads(args.cache_index.read_text())
    protocol = json.loads(args.protocol.read_text())
    refs = {r["reference_id"]: r for r in protocol["references"]}
    selected = sorted([q for q in protocol["queries"] if q["role"] == "calibration_known"],
                      key=lambda q: hashlib.sha256(("clean-neural-audit-v1\0"+q["query_id"]).encode()).digest())[:20]
    if args.model == "peaknet":
        from peaknet_baseline import PeakNetExtractor
        extractor = PeakNetExtractor(batch_size=125, threads=2)
    else:
        extractor = NMFPExtractor(batch_size=256, threads=2)
    if extractor.provenance != cache["provenance"]:
        raise ValueError("Audit extractor provenance differs from actual SD-RR caches")
    index = density_index({rid: e["path"] for rid, e in cache["references"].items()}, 1,
                          PhysicalPeakNetIndex if args.model == "peaknet" else __import__("neural_baseline").NMFPSequenceIndex)
    index.build_index()
    rows = []
    for original in selected:
        q = centered_query(original, 5)
        begin = q["expected_reference_start_s"]
        emb, times, _ = extractor.extract_file(refs[q["reference_id"]]["path"], begin, 5)
        reference, rt, _ = load_cached_embeddings(cache["references"][q["reference_id"]]["path"])
        positions = np.searchsorted(rt, begin+times)
        if not np.allclose(rt[positions], begin+times, atol=1e-9):
            raise ValueError("Clean oracle was not on the native physical reference grid")
        delta = float(np.max(np.abs(emb-reference[positions])))
        hit = index.search_times(emb, times, 20) if args.model == "peaknet" else index.search(emb, 20)
        if hit["reference_id"] != q["reference_id"] or abs(hit["start_s"]-begin) > 1e-8 or delta > 5e-5:
            raise AssertionError(f"Clean-source parity/retrieval failed: {q['query_id']}, delta={delta}, hit={hit}")
        microphone, mt, _ = load_cached_embeddings(cache["queries"][q["query_id"]]["path"])
        # Physical aligned similarity is diagnostic only; it never supplies an
        # oracle alignment to the benchmark matcher.
        aligned_cosine = np.sum(microphone*reference[positions], axis=1)
        rows.append({"query_id": q["query_id"], "reference_id": q["reference_id"],
                     "recording_id": q["recording_id"], "expected_start_s": begin,
                     "matched_clean_reference_id": hit["reference_id"], "matched_clean_start_s": hit["start_s"],
                     "max_abs_embedding_difference_different_batches": delta,
                     "matched_clean_score": hit["score"],
                     "actual_microphone_aligned_cosine_mean": float(aligned_cosine.mean()),
                     "actual_microphone_aligned_cosine_min": float(aligned_cosine.min()),
                     "actual_microphone_aligned_cosine_max": float(aligned_cosine.max())})
        print(rows[-1], flush=True)
    output = {"model": args.model, "source": "hash-selected20 calibration-known sources only",
              "protocol_sha256": sha(args.protocol), "cache_index_sha256": sha(args.cache_index),
              "clean_audio_oracle_only_not_acoustic_performance": True,
              "no_parameters_adjusted": True, "released_queries_filtered": 0,
              "rows": rows, "clean_identity_and_exact_offset_correct": len(rows),
              "max_abs_heterogeneous_batch_embedding_difference": max(r["max_abs_embedding_difference_different_batches"] for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(f"Clean-source audit passed {len(rows)}/{len(rows)}; max batch embedding delta {output['max_abs_heterogeneous_batch_embedding_difference']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("nmfp", "peaknet"), required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    audit(p.parse_args())
