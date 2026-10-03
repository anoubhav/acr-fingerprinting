"""Pinned upstream audfprint landmark baseline for manifest retrieval tasks.

Upstream files are imported unmodified.  Defaults mirror its command-line
analyzer and matcher: 20 hashes/s, reference shifts=1, query shifts=4,
match window=2, min count=5, search depth=100.  Storage time bits are 18 to
avoid six-minute time aliasing. Threshold selection belongs to the evaluator.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time
import numpy as np

from benchmark_data import materialize_crop

UPSTREAM_REVISION = "cb03ba99feafd41b8874307f0f4e808a6ce34362"


def _extract_reference(args):
    upstream, ref, cache_dir, density = args
    sys.path.insert(0, str(upstream))
    import audfprint_analyze
    cache_key = hashlib.sha256((str(pathlib.Path(ref["path"]).resolve()) + ":" +
        str(pathlib.Path(ref["path"]).stat().st_size) + ":" + str(density) + ":" + UPSTREAM_REVISION).encode()).hexdigest()
    cache_path = pathlib.Path(cache_dir) / f"reference_{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path) as data:
            return ref["reference_id"], data["hashes"], float(data["duration_s"]), 0.0, True
    analyzer = audfprint_analyze.Analyzer(density=density)
    started = time.perf_counter()
    hashes = analyzer.wavfile2hashes(ref["path"])
    elapsed = time.perf_counter() - started
    np.savez_compressed(cache_path, hashes=hashes, duration_s=analyzer.soundfiledur)
    return ref["reference_id"], hashes, analyzer.soundfiledur, elapsed, False


class AudfprintBaseline:
    def __init__(self, upstream, cache_dir, density=20.0, seed=20261002,
                 match_window=2, min_count=5, search_depth=100, maxtimebits=18):
        self.upstream = pathlib.Path(upstream).resolve()
        revision = subprocess.check_output(["git", "-C", str(self.upstream), "rev-parse", "HEAD"], text=True).strip()
        if revision != UPSTREAM_REVISION:
            raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, found {revision}")
        self.cache_dir = pathlib.Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(self.upstream))
        import audfprint_analyze, audfprint_match, hash_table
        self.analyzer = audfprint_analyze.Analyzer(density=density)
        self.analyzer.shifts = 4
        self.matcher = audfprint_match.Matcher()
        self.matcher.window = match_window
        self.matcher.threshcount = min_count
        self.matcher.search_depth = search_depth
        self.matcher.max_returns = 1
        self.hash_table_class = hash_table.HashTable
        self.config = {"upstream_revision": revision, "density": density, "sample_rate": 11025,
                       "n_fft": 512, "n_hop": 256, "reference_shifts": 1, "query_shifts": 4,
                       "match_window_frames": match_window, "min_count": min_count,
                       "search_depth": search_depth, "exact_count": False,
                       "hashbits": 20, "bucketsize": 100, "maxtimebits": maxtimebits, "seed": seed}
        self.stats = {}

    def fit(self, references, workers=4):
        started = time.perf_counter()
        random.seed(self.config["seed"])
        self.table = self.hash_table_class(hashbits=20, depth=100,
                                         maxtime=2 ** self.config["maxtimebits"])
        args = [(self.upstream, r, self.cache_dir, self.config["density"]) for r in references]
        hashes_total = duration_total = extraction_work_s = cache_hits = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            # Ordered insertion makes bucket sampling independent of worker order.
            for i, (rid, hashes, duration_s, elapsed, cached) in enumerate(pool.map(_extract_reference, args)):
                if len(hashes) and int(hashes[:, 0].max()) >= 2 ** self.config["maxtimebits"]:
                    raise ValueError(f"Reference {rid} exceeds time storage capacity")
                self.table.store(rid, hashes)
                hashes_total += len(hashes)
                duration_total += duration_s
                extraction_work_s += elapsed
                cache_hits += int(cached)
                if (i + 1) % 100 == 0:
                    print(f"audfprint indexed {i + 1}/{len(references)} references", flush=True)
        self.stats = {"index_build_wall_s": time.perf_counter() - started,
                      "reference_extraction_work_s": extraction_work_s,
                      "reference_count": len(references), "reference_hash_count": hashes_total,
                      "reference_audio_duration_s": duration_total, "reference_cache_hits": cache_hits,
                      "allocated_index_bytes": self.table.table.nbytes + self.table.counts.nbytes + self.table.hashesperid.nbytes,
                      "bucket_overflow_count": int(np.maximum(self.table.counts - self.table.depth, 0).sum()),
                      "stored_posting_count": int(np.minimum(self.table.counts, self.table.depth).sum())}
        return self.stats

    def query_features(self, query):
        key = hashlib.sha256(json.dumps({k: query[k] for k in
                 ("query_id", "path", "start_s", "duration_s")}, sort_keys=True).encode()).hexdigest()
        crop_path = self.cache_dir / f"query_ffmpeg_bounded_v3_{key}.wav"
        started = time.perf_counter()
        source = pathlib.Path(query["path"])
        if source.suffix.lower() == ".wav" and query["start_s"] == 0:
            import soundfile as sf
            info = sf.info(source)
            if abs(info.duration - query["duration_s"]) < 1 / info.samplerate:
                crop_path = source  # Already-materialized synthetic speech crop.
        if not crop_path.exists():
            materialize_crop(query, crop_path)
        crop_decode_s = time.perf_counter() - started
        # Time the unchanged upstream reader separately so feature/search
        # latency excludes decoding in the same way as the ACR evaluator.
        import audio_read
        original_reader = audio_read.audio_read
        reader_elapsed_s = 0.0
        def timed_reader(*args, **kwargs):
            nonlocal reader_elapsed_s
            reader_started = time.perf_counter()
            result = original_reader(*args, **kwargs)
            reader_elapsed_s += time.perf_counter() - reader_started
            return result
        started = time.perf_counter()
        audio_read.audio_read = timed_reader
        try:
            hashes = self.analyzer.wavfile2hashes(str(crop_path))
        finally:
            audio_read.audio_read = original_reader
        extraction_total_s = time.perf_counter() - started
        extraction_s = max(0.0, extraction_total_s - reader_elapsed_s)
        return hashes, {"query_crop_decode_s": crop_decode_s,
                        "query_decode_s": reader_elapsed_s,
                        "query_extraction_s": extraction_s,
                        "query_decode_plus_extraction_s": extraction_total_s}

    def match_query_features(self, query, hashes, extraction_stats, matcher=None):
        matcher = self.matcher if matcher is None else matcher
        started = time.perf_counter()
        results = matcher.match_hashes(self.table, hashes)
        search_s = time.perf_counter() - started
        out = {"query_id": query["query_id"], "reference_id": None, "score": 0.0,
               "matched_hash_count": 0, "query_hash_count": len(hashes),
               "normalized_match_count": 0.0, "offset_s": None,
               **extraction_stats,
               "query_search_s": search_s,
               "latency_s": extraction_stats["query_extraction_s"] + search_s,
               "decode_plus_feature_search_latency_s": extraction_stats["query_decode_plus_extraction_s"] + search_s}
        if len(results):
            hit = results[0]
            out.update(reference_id=self.table.names[int(hit[0])], score=float(hit[1]),
                       matched_hash_count=int(hit[1]), raw_matched_hash_count=int(hit[3]),
                       normalized_match_count=float(hit[1]) / max(len(hashes), 1),
                       offset_s=float(hit[2]) * self.analyzer.n_hop / self.analyzer.target_sr)
        return out

    def query(self, query):
        hashes, stats = self.query_features(query)
        return self.match_query_features(query, hashes, stats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=pathlib.Path)
    p.add_argument("--upstream", required=True, type=pathlib.Path)
    p.add_argument("--cache", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gallery-ids", type=pathlib.Path,
                   help="JSON list of allowed reference IDs (same gallery as all compared methods)")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    refs = manifest["references"]
    if args.gallery_ids:
        allowed = set(json.loads(args.gallery_ids.read_text()))
        refs = [r for r in refs if r["reference_id"] in allowed]
        if {r["reference_id"] for r in refs} != allowed:
            raise ValueError("Gallery IDs not present in manifest")
    baseline = AudfprintBaseline(args.upstream, args.cache)
    baseline.fit(refs, args.workers)
    predictions = []
    for i, query in enumerate(manifest["queries"]):
        predictions.append(baseline.query(query))
        if (i + 1) % 100 == 0:
            print(f"audfprint queried {i + 1}/{len(manifest['queries'])}", flush=True)
    result = {"method": "audfprint", "config": baseline.config,
              "stats": baseline.stats, "predictions": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(baseline.stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
