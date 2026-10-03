"""Evaluate the official pretrained NMFP cache using native full-density search.

Exact frame-candidate retrieval and mean aligned-cosine sequence reranking follow
the upstream retrieval protocol. This is a PEX-derived content retrieval task,
not a reproduction of the authors' different published benchmark accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from neural_baseline import NMFPSequenceIndex, centered_query, load_cached_embeddings


def run(args):
    import faiss
    faiss.omp_set_num_threads(args.threads)
    manifest = json.loads(args.protocol.read_text())
    refs_cache = json.loads(args.reference_cache.read_text())
    queries_cache = json.loads(args.query_cache.read_text())
    protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    if queries_cache["manifest_sha256"] != protocol_hash:
        raise ValueError("Query cache was extracted from a different protocol")
    if refs_cache["provenance"] != queries_cache["provenance"]:
        raise ValueError("Reference/query extraction provenance differs")
    gallery_roles = set(manifest["protocol"]["gallery_roles"])
    gallery = [r for r in manifest["references"] if r["role"] in gallery_roles]
    if set(refs_cache["references"]) != {r["reference_id"] for r in gallery}:
        raise ValueError("Reference cache is incomplete or has extra gallery entries")
    paths = {rid: item["path"] for rid, item in refs_cache["references"].items()}
    index_class = NMFPSequenceIndex
    if args.index_type == "peaknet":
        from peaknet_baseline import PeakNetSequenceIndex
        index_class = PeakNetSequenceIndex
    expected_method = "PeakNetFP" if args.index_type == "peaknet" else "NMFP-Triplet"
    if refs_cache["provenance"]["method"] != expected_method:
        raise ValueError("Model cache does not match requested native matcher")
    index = index_class.from_files(paths)
    build_s = (index.build_cpu_ivf(args.nlist, args.nprobe) if args.candidate_index == "ivf"
               else index.build_index())
    duration_total = sum(x["timing"]["audio_duration_s"] for x in refs_cache["references"].values())
    # With a regular .5s grid, timestamps can be reconstructed from the hop
    # and per-track boundaries; no per-frame floating timestamp is needed here.
    metadata_bytes = index.starts.nbytes + index.ends.nbytes + sum(len(r.encode()) for r in index.reference_ids)
    logical_payload_bytes = index.embeddings.nbytes + metadata_bytes
    index_serialized_bytes = int(faiss.serialize_index(index._index).nbytes)
    stats = {
        "reference_count": len(index.reference_ids), "fingerprints": len(index.embeddings),
        "reference_audio_duration_s": duration_total,
        "embedding_bytes": index.embeddings.nbytes, "metadata_bytes": metadata_bytes,
        "payload_bytes": logical_payload_bytes,
        "measured_payload_bytes_per_hour": logical_payload_bytes / duration_total * 3600,
        "index_serialized_bytes": index_serialized_bytes,
        "resident_vector_bytes": index.embeddings.nbytes + index_serialized_bytes + metadata_bytes,
        "reference_extraction_work_s": sum(x["timing"]["frontend_s"] + x["timing"]["forward_s"] for x in refs_cache["references"].values()),
        "index_build_s": build_s,
        "storage_note": "Logical full-density FP32 embeddings plus reconstructible regular-grid track metadata. Resident vector bytes estimate the FAISS serialized search structures and sequence-rerank copy; not measured process RSS or client model weights.",
        "timing_note": "Extraction excludes shared decoding and warmed graph build; concurrent experimental jobs can contend. No smart-TV device claim.",
    }
    queries = [q for q in manifest["queries"] if q.get("evaluate", True)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for duration in args.durations:
        prefix = "peaknet_native" if args.index_type == "peaknet" else "nmfp_native"
        if args.candidate_index == "ivf":
            prefix += f"_ivf{args.nlist}_p{args.nprobe}"
        output = args.output_dir / f"{prefix}_{duration:g}s.json"
        if output.exists() and json.loads(output.read_text())["protocol_sha256"] == protocol_hash:
            print(f"Result already exists: {output}", flush=True)
            continue
        predictions = []
        for i, original in enumerate(queries):
            q = centered_query(original, duration)
            entry = queries_cache["queries"][q["query_id"]]
            if entry["item"]["start_s"] != q["start_s"] or entry["item"]["duration_s"] != q["duration_s"]:
                raise ValueError(f"Query bounds changed: {q['query_id']}")
            fingerprints, _, metadata = load_cached_embeddings(entry["path"])
            hit = index.search(fingerprints, args.top_k)
            correct = hit["reference_id"] == q["reference_id"]
            extraction_s = metadata["timing"]["frontend_s"] + metadata["timing"]["forward_s"]
            predictions.append({
                "query_id": q["query_id"], "base_query_id": original["query_id"],
                "source_id": q["source_id"], "source_query_id": q.get("source_query_id", q["query_id"]),
                "role": q["role"], "duration_s": q["duration_s"],
                "target_reference_id": q["reference_id"], "reference_id": hit["reference_id"],
                "score": hit["score"] if np.isfinite(hit["score"]) else -1.0,
                "offset_s": hit["start_s"], "expected_reference_start_s": q["expected_reference_start_s"],
                "correct_reference": correct,
                "localized_correct": correct and abs(hit["start_s"] - q["expected_reference_start_s"]) <= 2.0,
                "query_fingerprints": len(fingerprints), "query_extraction_s": extraction_s,
                "query_search_s": hit["search_s"], "latency_s": extraction_s + hit["search_s"],
                "candidate_count": hit.get("candidate_count", 0), "annotation": q["annotation"],
                "estimated_reference_time_scale": hit.get("estimated_reference_time_scale", 1.0),
                "negative_population": q.get("negative_population"),
                "overlap": bool(q["overlapping_other_annotations"]),
            })
            if (i + 1) % 100 == 0:
                print(f"{expected_method}-native {duration:g}s queried {i + 1}/{len(queries)}", flush=True)
        result = {
            "method": expected_method + (" (native-IVF)" if args.candidate_index == "ivf" else " (native)"), "factor": 1, "duration_s_requested": duration,
            "stats": stats, "predictions": predictions, "config": refs_cache["provenance"],
            "matcher": {"top_k": args.top_k, "exact": args.candidate_index == "exact", "threads": args.threads,
                        "candidate_index": args.candidate_index,
                        "ivf_nlist": args.nlist if args.candidate_index == "ivf" else None,
                        "ivf_nprobe": args.nprobe if args.candidate_index == "ivf" else None,
                        "index_training": "gallery-only clustering; no neural model training" if args.candidate_index == "ivf" else None,
                        "score": "mean aligned cosine", "time_scaling": "estimated candidate-specific" if args.index_type == "peaknet" else "unit only",
                        "boundary_fix": "every complete in-track sequence including final track allowed"},
            "protocol": manifest["protocol"], "protocol_sha256": protocol_hash,
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        known = [r for r in predictions if r["role"] == "test_known"]
        print(f"Saved {output}; raw correct {sum(r['correct_reference'] for r in known)}/{len(known)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--reference-cache", required=True, type=Path)
    p.add_argument("--query-cache", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--durations", type=float, nargs="+", default=[5, 1, 2, 3, 10])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--index-type", choices=("nmfp", "peaknet"), default="nmfp")
    p.add_argument("--candidate-index", choices=("exact", "ivf"), default="exact")
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", type=int, default=16)
    run(p.parse_args())
