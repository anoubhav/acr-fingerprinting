"""Frozen-checkpoint storage-density controls using physical-time-aware matching.

NMFP thins both regular reference and query grids, retaining native unit-scale
aligned-cosine reranking. PeakNetFP thins references while retaining all query
frames; its native pairwise stretch estimator and reranking are expressed in
seconds, so the physical [0.5, 2] stretch bounds survive unequal grid hops.
No model inference, retraining, oracle tempo, or test-selected settings occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from neural_baseline import NMFPSequenceIndex, centered_query, load_cached_embeddings
from summarize_results import summarize


DECLARATION = {
    "control": "uniform post-extraction density reduction of frozen public neural checkpoints",
    "declared_reference_factors": [2, 4],
    "native_hop_s": 0.5,
    "nmfp_query_policy": "same stride factor as references; unit-scale native aligned cosine",
    "peaknet_query_policy": "retain dense query; native estimated physical stretch and nearest reference-grid alignment",
    "candidate_retrieval": "exact CPU IndexFlatIP, top20 per query frame",
    "gate": "existing calibration-only order statistic; original and extended negative populations separately",
    "no_training": True,
    "no_test_selected_parameters": True,
    "no_oracle_tempo": True,
    "timing_claims": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_grid(embeddings, times, hop_s):
    """Validate that compact ordinal/hop metadata reproduces actual saved times."""
    if len(embeddings) != len(times) or not len(times):
        raise ValueError("Invalid or empty cached fingerprint grid")
    if not np.allclose(times, np.arange(len(times)) * hop_s, atol=1e-9, rtol=0):
        raise ValueError("Cached physical timestamps do not match the declared regular grid")
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite cached embedding")


def density_index(paths, factor, index_class=NMFPSequenceIndex):
    ids, arrays, starts, ends, positions, position = [], [], [], [], [], 0
    for rid, path in sorted(paths.items()):
        emb, times, _ = load_cached_embeddings(path)
        check_grid(emb, times, 0.5)
        emb, times = np.ascontiguousarray(emb[::factor]), times[::factor]
        check_grid(emb, times, 0.5 * factor)
        ids.append(rid); arrays.append(emb); starts.append(position)
        positions.append(times)
        position += len(emb); ends.append(position)
    index = index_class(ids, np.concatenate(arrays), np.asarray(starts), np.asarray(ends), 0.5 * factor)
    # Actual timestamps are retained for audits and are exactly reconstructible
    # from the per-track origin/hop; the logical compact payload omits this copy.
    index.audit_times = np.concatenate(positions)
    return index


class PhysicalPeakNetIndex(NMFPSequenceIndex):
    """Native PeakNetFP rules in seconds, including unequal reference/query hops."""

    def search_times(self, query, query_times, top_k=20):
        query = np.asarray(query, np.float32)
        query_times = np.asarray(query_times, np.float64)
        tick = time.perf_counter()
        if len(query) != len(query_times):
            raise ValueError("Query physical times do not match embeddings")
        if not len(query):
            return {"reference_id": None, "start_s": None, "score": -1.0, "search_s": 0.0}
        _, neighbors = self._index.search(query, min(top_k, len(self.embeddings)))
        track_matches = {}
        for row, values in enumerate(neighbors):
            for idx in values:
                track = int(np.searchsorted(self.starts, idx, side="right") - 1)
                track_matches.setdefault(track, {}).setdefault(row, int(idx))
        inverse_scales = {}
        for track, pairs in track_matches.items():
            qtimes = query_times[np.asarray(list(pairs))]
            # The saved reference timestamps, rather than a .5s assumption,
            # define the scale estimate. Constant concatenation origins cancel.
            rtimes = self.audit_times[np.asarray(list(pairs.values()))]
            qdiff, rdiff = qtimes[:, None] - qtimes, rtimes[:, None] - rtimes
            ratios = np.divide(qdiff, rdiff, out=np.zeros_like(qdiff), where=rdiff != 0)
            nonzero = ratios[ratios != 0]
            inverse_scale = float(np.mean(nonzero)) if len(nonzero) else 1.0
            inverse_scales[track] = inverse_scale if 0.5 <= inverse_scale <= 2 else 1.0
        candidates = {}
        for row, values in enumerate(neighbors):
            for idx in values:
                track = int(np.searchsorted(self.starts, idx, side="right") - 1)
                inverse_scale = inverse_scales[track]
                # Quantize on the actual track-local clock at EVERY density.
                # Global parity can flip even native source identity when
                # estimated inverse scale2 puts query frames at half-grid ties.
                local = int(idx) - int(self.starts[track])
                start = round(local - query_times[row] / (inverse_scale * self.hop_s))
                sequence = np.rint(start + query_times / (inverse_scale * self.hop_s)).astype(int) + self.starts[track]
                if sequence[0] < self.starts[track] or sequence[-1] >= self.ends[track]:
                    continue
                candidates[(track, tuple(sequence))] = inverse_scale
        if not candidates:
            return {"reference_id": None, "start_s": None, "score": -1.0,
                    "search_s": time.perf_counter() - tick, "candidate_count": 0}
        best, best_score = None, -np.inf
        for (track, sequence), inverse_scale in candidates.items():
            score = float(np.mean(np.sum(query * self.embeddings[list(sequence)], axis=1)))
            if score > best_score:
                best, best_score = (track, sequence, inverse_scale), score
        track, sequence, inverse_scale = best
        return {"reference_id": self.reference_ids[track],
                "start_s": float(self.audit_times[sequence[0]]), "score": best_score,
                "candidate_count": len(candidates), "estimated_reference_time_scale": 1 / inverse_scale,
                "unique_reference_evidence_frames": len(set(sequence)),
                "search_s": time.perf_counter() - tick}


def compact_stats(index, refs_cache, factor):
    import faiss
    audio_s = sum(e["timing"]["audio_duration_s"] for e in refs_cache["references"].values())
    # Int64 boundaries, UTF8 reference IDs, one shared float64 hop, and one
    # float64 origin per track. No per-frame identity or timestamp is needed.
    metadata = index.starts.nbytes + index.ends.nbytes + sum(len(r.encode()) for r in index.reference_ids)
    metadata += 8 + 8 * len(index.reference_ids)
    signal = int(index.embeddings.nbytes)
    serialized = int(faiss.serialize_index(index._index).nbytes)
    return {
        "reference_count": len(index.reference_ids), "fingerprints": len(index.embeddings),
        "reference_audio_duration_s": audio_s, "reference_factor": factor, "reference_hop_s": index.hop_s,
        "embedding_dimension": index.embeddings.shape[1], "embedding_dtype": "float32",
        "embedding_bytes": signal, "metadata_bytes": metadata, "payload_bytes": signal + metadata,
        "measured_signal_bytes_per_hour": signal / audio_s * 3600,
        "measured_payload_bytes_per_hour": (signal + metadata) / audio_s * 3600,
        "explicit_timestamp_bytes_if_stored": int(index.audit_times.nbytes),
        "index_serialized_bytes": serialized,
        "resident_vector_bytes": signal + serialized + metadata,
        "actual_run_keeps_timestamp_audit_copy_bytes": int(index.audit_times.nbytes),
        "storage_note": "Signal-only FP32 embeddings reported separately from compact logical track metadata; a FAISS search copy and sequence-reranking copy are both resident. Timestamp audit array is exactly reconstructible from uniform grids and separately counted.",
        "timing_note": "Cached matching control; no fresh model extraction or end-to-end latency claim.",
    }


def evidence_summary(predictions):
    known = [p for p in predictions if p["role"] == "test_known"]
    return {"query_fingerprints_test_known": dict(Counter(p["query_fingerprints"] for p in known)),
            "query_evidence_span_s_test_known": dict(Counter(p["query_evidence_span_s"] for p in known)),
            "query_covered_audio_s_test_known": dict(Counter(p["query_covered_audio_s"] for p in known)),
            "candidate_count_test_known": {"median": float(np.median([p["candidate_count"] for p in known])),
                                           "min": min(p["candidate_count"] for p in known),
                                           "max": max(p["candidate_count"] for p in known)},
            "candidate_reference_correct_test_known": sum(p["correct_reference"] for p in known),
            "localized_correct_test_known": sum(p["localized_correct"] for p in known)}


def evaluate(args):
    import faiss
    faiss.omp_set_num_threads(args.threads)
    protocol = json.loads(args.protocol.read_text())
    primary = json.loads(args.primary_protocol.read_text())
    ref_cache = json.loads(args.reference_cache.read_text())
    query_cache = json.loads(args.query_cache.read_text())
    protocol_hash = sha(args.protocol)
    if query_cache["manifest_sha256"] != protocol_hash:
        raise ValueError("Query-cache protocol SHA256 mismatch")
    if query_cache["provenance"] != ref_cache["provenance"]:
        raise ValueError("Reference/query checkpoint provenance differs")
    method = "NMFP-Triplet" if args.model == "nmfp" else "PeakNetFP"
    if ref_cache["provenance"]["method"] != method:
        raise ValueError("Incorrect requested checkpoint")
    gallery = {r["reference_id"] for r in protocol["references"] if r["role"] in protocol["protocol"]["gallery_roles"]}
    if gallery != set(ref_cache["references"]):
        raise ValueError("Gallery mismatch")
    primary_ids = {q["query_id"] for q in primary["queries"] if q.get("evaluate", True)}
    queries = [q for q in protocol["queries"] if q.get("evaluate", True)]
    if not primary_ids <= {q["query_id"] for q in queries}:
        raise ValueError("Primary protocol is not a subset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    declaration_path = args.output_dir / "declaration.json"
    declaration = {**DECLARATION, "protocol_sha256": protocol_hash,
                   "primary_protocol_sha256": sha(args.primary_protocol),
                   "query_cache_index_sha256": sha(args.query_cache),
                   "reference_cache_index_sha256": sha(args.reference_cache),
                   "model": method, "factors_in_this_run": args.factors, "declared_before_evaluation": True}
    declaration["code_sha256"] = {name: sha(Path(__file__).with_name(name))
        for name in ("neural_density_controls.py", "neural_baseline.py", "peaknet_baseline.py")}
    if args.model == "peaknet":
        declaration["catalog_rounding_invariance_repair"] = "round half to even in track-local physical grid at all densities; no unrelated track parity"
        declaration["repair_basis"] = "independent synthetic catalog-order invariance tests after initial scoring; all declared variants rerun without outcome-based selection"
    declaration_path.write_text(json.dumps(declaration, indent=2) + "\n")
    paths = {rid: entry["path"] for rid, entry in ref_cache["references"].items()}
    results = []
    for factor in args.factors:
        if factor not in (1, 2, 4):
            raise ValueError("Undeclared density factor")
        output = args.output_dir / f"{args.model}_density_r{factor}_q{factor if args.model == 'nmfp' else 1}_5s.json"
        if output.exists():
            result = json.loads(output.read_text())
            if result["protocol_sha256"] != protocol_hash or result["declaration_sha256"] != sha(declaration_path):
                raise ValueError("Existing result provenance mismatch")
        else:
            index = density_index(paths, factor, PhysicalPeakNetIndex if args.model == "peaknet" else NMFPSequenceIndex)
            index.build_index()
            stats = compact_stats(index, ref_cache, factor)
            predictions = []
            qfactor = factor if args.model == "nmfp" else 1
            for n, original in enumerate(queries):
                q = centered_query(original, 5)
                entry = query_cache["queries"][q["query_id"]]
                if entry["item"]["start_s"] != q["start_s"] or entry["item"]["duration_s"] != q["duration_s"]:
                    raise ValueError(f"Changed query crop {q['query_id']}")
                emb, times, _ = load_cached_embeddings(entry["path"])
                check_grid(emb, times, 0.5)
                original_count = len(emb)
                emb, times = np.ascontiguousarray(emb[::qfactor]), times[::qfactor]
                check_grid(emb, times, 0.5 * qfactor)
                hit = index.search_times(emb, times, args.top_k) if args.model == "peaknet" else index.search(emb, args.top_k)
                correct = hit["reference_id"] == q["reference_id"]
                predictions.append({
                    "query_id": q["query_id"], "base_query_id": original["query_id"],
                    "source_id": q["source_id"], "source_query_id": q.get("source_query_id", q["query_id"]),
                    "role": q["role"], "duration_s": q["duration_s"], "target_reference_id": q["reference_id"],
                    "reference_id": hit["reference_id"], "score": hit["score"] if np.isfinite(hit["score"]) else -1.0,
                    "offset_s": hit["start_s"], "expected_reference_start_s": q["expected_reference_start_s"],
                    "correct_reference": correct,
                    "localized_correct": bool(correct and abs(hit["start_s"] - q["expected_reference_start_s"]) <= 2.0),
                    "query_fingerprints": len(emb), "query_fingerprints_before_control": original_count,
                    "query_frame_times_s": times.tolist(), "query_hop_s": 0.5 * qfactor,
                    "query_evidence_span_s": float(times[-1] + 1.0),
                    "query_covered_audio_s": float(1.0 + np.minimum(np.diff(times), 1.0).sum()),
                    "unique_reference_evidence_frames": hit.get("unique_reference_evidence_frames", len(emb)),
                    "query_extraction_s": 0.0, "query_search_s": hit["search_s"], "latency_s": hit["search_s"],
                    "candidate_count": hit.get("candidate_count", 0),
                    "estimated_reference_time_scale": hit.get("estimated_reference_time_scale", 1.0),
                    "annotation": q["annotation"], "negative_population": q.get("negative_population"),
                    "overlap": bool(q["overlapping_other_annotations"]),
                })
                if (n + 1) % 500 == 0:
                    print(f"{args.model} r{factor}/q{qfactor}: {n+1}/{len(queries)}", flush=True)
            result = {"method": f"{method} (density control r{factor}/q{qfactor})", "factor": factor,
                      "duration_s_requested": 5, "stats": stats, "predictions": predictions,
                      "config": ref_cache["provenance"], "protocol": protocol["protocol"],
                      "protocol_sha256": protocol_hash, "declaration_sha256": sha(declaration_path),
                      "matcher": {"reference_factor": factor, "query_factor": qfactor,
                                  "reference_hop_s": 0.5*factor, "query_hop_s": 0.5*qfactor,
                                  "top_k": args.top_k, "exact": True, "candidate_index": "IndexFlatIP",
                                  "physical_timestamps_validated": True, "threads": args.threads,
                                  "score": "mean aligned cosine", "time_scaling": "estimated candidate-specific in seconds" if args.model == "peaknet" else "unit only",
                                  "physical_reference_time_scale_range": [0.5, 2.0] if args.model == "peaknet" else [1.0, 1.0],
                                  "boundary_fix": "all complete sequences bounded within one reference",
                                  "density_control": True, "frozen_checkpoint": True,
                                  "query_cost_note": "Post-extraction subsampling only; no sparse inference timing measured",
                                  "reference_grid_rounding": "track-local physical grid at all densities, independent of other track lengths" if args.model == "peaknet" else None,
                                  "estimated_scale_rerank_note": "Dense PeakNet query frames may map to repeated sparse reference frames; unique evidence is reported" if args.model == "peaknet" else None}}
            output.write_text(json.dumps(result, indent=2) + "\n")
            del index
        for study, study_protocol in (("primary", primary), ("extended", protocol)):
            study_result = {**result, "predictions": [dict(p) for p in result["predictions"]
                              if study == "extended" or p["base_query_id"] in primary_ids]}
            summary = summarize(study_result, study_protocol)
            summary.pop("latency_s", None)
            summary["timing_claims"] = False
            summary["study"] = study
            summary["query_evidence"] = evidence_summary(study_result["predictions"])
            summary["results_path"] = str(output.resolve())
            summary["results_sha256"] = sha(output)
            results.append(summary)
            known = summary["test_known"]["all"]
            unknown = summary["unknown_test"]
            print(f"{args.model} r{factor} {study}: raw {known['correct_reference']['successes']}/543, accepted {known['accepted_correct']['successes']}; FPR {unknown['false_accepts']}/{unknown['n']}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("nmfp", "peaknet"), required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--primary-protocol", type=Path, required=True)
    p.add_argument("--reference-cache", type=Path, required=True)
    p.add_argument("--query-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--factors", type=int, nargs="+", default=[2, 4])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=20)
    evaluate(p.parse_args())
