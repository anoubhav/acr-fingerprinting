"""Official pretrained NMFP-Triplet adapter with auditable, cached inference.

Model and mel frontend are imported unchanged from raraz15/neural-music-fp.
Source decoding uses FFmpeg mono8k float32 PCM, shared with the other baselines;
this replaces upstream MonoLoader so source and query timing remain consistent.
TensorFlow is loaded lazily, so cached embeddings and native sequence retrieval
can be used in the main benchmark environment without installing TensorFlow.
There is no training or checkpoint conversion. CPU inference defaults to float32;
the supplied checkpoint used mixed-float16 training. This runtime choice is
recorded and can be changed with --precision mixed_float16 for parity checks.

Cache format: embeddings[n,128] float32, times[n] float64 window-start seconds,
metadata_json scalar JSON. Query times are relative to the extracted crop.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = WORKSPACE / "work/upstream/neural-music-fp"
DEFAULT_MODEL_DIR = WORKSPACE / "work/models/nmfp/nmfp-triplet"
UPSTREAM_URL = "https://github.com/raraz15/neural-music-fp"
UPSTREAM_COMMIT = "15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc"
MODEL_ARCHIVE_URL = "https://zenodo.org/records/15719945/files/nmfp-triplet.zip?download=1"
MODEL_ARCHIVE_MD5 = "ee8a3358fc5e5cdd09d6d2245d395021"


def file_hash(path, algorithm="sha256"):
    with pathlib.Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def package_versions():
    versions = {}
    for name in ("numpy", "tensorflow", "tf-keras", "essentia", "scipy", "soundfile", "pyyaml"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def centered_query(query, requested_duration_s):
    """Apply the canonical centered crop convention without changing annotations."""
    out = dict(query)
    duration = min(float(requested_duration_s), float(query["duration_s"]))
    crop_shift = (float(query["duration_s"]) - duration) / 2.0
    out["start_s"] = float(query["start_s"]) + crop_shift
    out["duration_s"] = duration
    out["query_id"] = f'{query["query_id"]}@{float(requested_duration_s):g}'
    out["expected_reference_start_s"] = float(query["expected_reference_start_s"]) + crop_shift * float(query["expected_time_scale"])
    out["expected_reference_end_s"] = out["expected_reference_start_s"] + duration * float(query["expected_time_scale"])
    return out


def load_cached_embeddings(path):
    with np.load(path, allow_pickle=False) as saved:
        return saved["embeddings"], saved["times"], json.loads(str(saved["metadata_json"]))


class NMFPExtractor:
    """Pretrained official model, restored strictly before processing any audio."""

    def __init__(self, upstream=DEFAULT_UPSTREAM, model_dir=DEFAULT_MODEL_DIR,
                 batch_size=256, threads=4, precision="float32", hop_s=0.5):
        upstream, model_dir = pathlib.Path(upstream).resolve(), pathlib.Path(model_dir).resolve()
        if not (upstream / "nmfp/model/nnfp.py").exists():
            raise FileNotFoundError(f"Official NMFP source missing at {upstream}")
        if not (model_dir / "config.yaml").exists():
            raise FileNotFoundError(f"Official NMFP model config missing at {model_dir}")
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if revision != UPSTREAM_COMMIT:
            raise ValueError(f"Expected NMFP revision {UPSTREAM_COMMIT}, received {revision}")
        dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty:
            raise ValueError("Official upstream source has tracked modifications")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        sys.path.insert(0, str(upstream))
        import yaml
        import tensorflow as tf
        import essentia.standard as es
        from nmfp.model.nnfp import FingerPrinter
        from nmfp.audio_processing.melspectrogram import Melspec_layer
        from nmfp.audio_processing.segmentation import segment_audio

        tf.config.threading.set_intra_op_parallelism_threads(int(threads))
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.mixed_precision.set_global_policy(precision)
        tf.random.set_seed(27)
        cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
        audio, inp, architecture = (cfg["MODEL"][key] for key in ("AUDIO", "INPUT", "ARCHITECTURE"))
        self.fs = int(audio["FS"])
        self.segment_s = float(audio["SEGMENT_DUR"])
        self.hop_s = float(hop_s)
        self.segment_samples = round(self.fs * self.segment_s)
        self.hop_samples = round(self.fs * self.hop_s)
        if not (0 < self.hop_samples <= self.segment_samples):
            raise ValueError("Hop must be positive and no longer than model window")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        self.es, self.segment_audio, self.tf = es, segment_audio, tf
        self.decoded_audio_cache = OrderedDict()
        self.frontend = Melspec_layer(
            segment_duration=self.segment_s, fs=self.fs, n_fft=int(inp["STFT_WIN"]),
            stft_hop=int(inp["STFT_HOP"]), n_mels=int(inp["N_MELS"]),
            f_min=float(inp["F_MIN"]), f_max=float(inp["F_MAX"]),
            dynamic_range=float(inp["DYNAMIC_RANGE"]), scale=bool(inp["SCALE"]))
        self.model = FingerPrinter(emb_sz=int(architecture["EMB_SZ"]), fc_unit_dim=[32, 1],
                                  norm=architecture["BN"], mixed_precision=precision == "mixed_float16")
        self.model.trainable = False
        # Building first makes assert_existing_objects_matched meaningful: lazy
        # checkpoint restores otherwise can conceal a randomly initialized layer.
        dummy = self.frontend.compute(np.zeros(self.segment_samples, dtype=np.float32))[None, :, :, None]
        self.model(dummy, training=False)
        checkpoint_path = tf.train.latest_checkpoint(str(model_dir))
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint in {model_dir}")
        checkpoint = tf.train.Checkpoint(model=self.model)
        status = checkpoint.restore(checkpoint_path)
        status.assert_existing_objects_matched()
        status.expect_partial()  # Optimizer state is deliberately not restored.
        self.forward = tf.function(self.model, reduce_retracing=True)
        # Compile/warm-up after restoring so measured extraction excludes graph build.
        warm = np.asarray(self.forward(dummy), dtype=np.float32)
        if not np.all(np.isfinite(warm)) or not np.allclose(np.linalg.norm(warm, axis=1), 1.0, atol=2e-5):
            raise ValueError("Restored model emitted nonfinite or non-unit embeddings")
        self.provenance = {
            "method": "NMFP-Triplet", "paper_url": "https://arxiv.org/abs/2506.22661",
            "upstream_url": UPSTREAM_URL, "upstream_commit": revision,
            "upstream_tracked_files_unmodified": True,
            "model_archive_url": MODEL_ARCHIVE_URL, "model_archive_md5_expected": MODEL_ARCHIVE_MD5,
            "model_archive_md5_verified": file_hash(model_dir.parent / "nmfp-triplet.zip", "md5") == MODEL_ARCHIVE_MD5,
            "checkpoint": pathlib.Path(checkpoint_path).name,
            "checkpoint_files_sha256": {p.name: file_hash(p) for p in sorted(model_dir.glob("ckpt-*")) if p.is_file()},
            "model_config_sha256": file_hash(model_dir / "config.yaml"), "official_model_config": cfg,
            "restoration_assert_existing_objects_matched": True,
            "inference_precision": precision, "published_training_mixed_precision": bool(cfg["TRAIN"]["MIXED_PRECISION"]),
            "inference_runtime_change": "TensorFlow2.16.2/legacy Keras2.16 and Essentia1177 for Python3.12 Apple arm64; official modules and weights unchanged",
            "decode_backend": "FFmpeg mono8k float32 PCM; shared benchmark decoding",
            "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "embedding_dimension": int(architecture["EMB_SZ"]), "sample_rate": self.fs,
            "segment_duration_s": self.segment_s, "hop_s": self.hop_s,
            "batch_size": self.batch_size, "intra_op_threads": int(threads), "inter_op_threads": 1,
            "packages": package_versions(), "platform": platform.platform(), "machine": platform.machine(),
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()

    def extract_array(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < self.segment_samples:
            return np.empty((0, self.provenance["embedding_dimension"]), np.float32), np.empty(0), {"frontend_s": 0.0, "forward_s": 0.0}
        segments, _ = self.segment_audio(audio, L=self.segment_samples, H=self.hop_samples, discard_remainder=True)
        output, frontend_s, forward_s = [], 0.0, 0.0
        for begin in range(0, len(segments), self.batch_size):
            tick = time.perf_counter()
            inputs = self.frontend.compute_batch(segments[begin:begin + self.batch_size])[:, :, :, None].astype(np.float32)
            frontend_s += time.perf_counter() - tick
            tick = time.perf_counter()
            output.append(np.asarray(self.forward(inputs), dtype=np.float32))
            forward_s += time.perf_counter() - tick
        embeddings = np.concatenate(output)
        if not np.all(np.isfinite(embeddings)) or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=5e-5):
            raise ValueError("Nonfinite or non-unit fingerprint")
        return embeddings, np.arange(len(embeddings), dtype=np.float64) * self.hop_samples / self.fs, {
            "frontend_s": frontend_s, "forward_s": forward_s}

    def extract_file(self, path, start_s=0.0, duration_s=None):
        """Use shared FFmpeg PCM decoding and official frontend; crop after decode."""
        tick = time.perf_counter()
        path = pathlib.Path(path).resolve()
        cache_key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        decoded_cache_hit = cache_key in self.decoded_audio_cache
        if decoded_cache_hit:
            audio = self.decoded_audio_cache.pop(cache_key)
        else:
            command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
                       "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(self.fs),
                       "-f", "f32le", "pipe:1"]
            decoded = subprocess.run(command, check=True, capture_output=True)
            audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
        self.decoded_audio_cache[cache_key] = audio
        while len(self.decoded_audio_cache) > 4:
            self.decoded_audio_cache.popitem(last=False)
        decode_s = time.perf_counter() - tick
        # Match the shared ACR runner's sample-accurate crop convention. The
        # physical protocol stores seconds; bounded fractional endpoints can
        # imply half-samples after centering a shorter query.
        start = round(float(start_s) * self.fs)
        end = len(audio) if duration_s is None else start + round(float(duration_s) * self.fs)
        if start < 0 or start >= len(audio) or end > len(audio) + int(0.01 * self.fs):
            raise ValueError(f"Invalid crop [{start_s}, {duration_s}] for {path}")
        audio = audio[start:min(end, len(audio))]
        embeddings, times, timing = self.extract_array(audio)
        timing.update({"decode_s": decode_s, "total_extraction_s": time.perf_counter() - tick,
                       "audio_duration_s": len(audio) / self.fs, "fingerprint_count": len(embeddings),
                       "decoded_audio_cache_hit": decoded_cache_hit,
                       "decoder_crop_shortfall_samples": max(0, end - (start + len(audio)))})
        return embeddings, times, timing

    def cache(self, item, cache_dir, kind="query"):
        item_id = item["query_id"] if kind == "query" else item["reference_id"]
        specification = {"kind": kind, "id": item_id, "path": str(pathlib.Path(item["path"]).resolve()),
                         "start_s": float(item.get("start_s", 0.0)),
                         "duration_s": item.get("duration_s") if kind == "query" else None,
                         "source_sha256": file_hash(item["path"]), "extractor_fingerprint": self.fingerprint}
        if kind == "query":
            specification["crop_sample_rounding"] = "round_half_to_even"
        key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        dest = pathlib.Path(cache_dir) / kind / f"{key}.npz"
        if dest.exists():
            emb, times, metadata = load_cached_embeddings(dest)
            if metadata["specification"] != specification:
                raise ValueError("Cached specification mismatch")
            return dest, metadata, True
        emb, times, timing = self.extract_file(item["path"], specification["start_s"], specification["duration_s"])
        metadata = {"specification": specification, "timing": timing, "provenance": self.provenance, "item": item}
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(".tmp.npz")
        np.savez(temporary, embeddings=emb, times=times, metadata_json=json.dumps(metadata, sort_keys=True))
        temporary.replace(dest)
        return dest, metadata, False


@dataclass
class NMFPSequenceIndex:
    """Native NMFP sequence reranking with exact candidate retrieval.

    Original rule in upstream retrieval.py: nearest frame matches generate
    candidate sequence offsets, then mean corresponding cosine ranks them.
    This helper fixes an upstream last-track boundary exclusion, so every
    complete in-track sequence remains eligible. It is for full-density,
    equal-hop references; a common sparse matcher is a separate experiment.
    """
    reference_ids: list[str]
    embeddings: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    hop_s: float = 0.5

    @classmethod
    def from_files(cls, paths, hop_s=0.5):
        ids, arrays, starts, ends, pos = [], [], [], [], 0
        for reference_id, path in sorted(paths.items()):
            emb, times, _ = load_cached_embeddings(path)
            if len(emb) == 0:
                continue
            if len(times) > 1 and not np.allclose(np.diff(times), hop_s):
                raise ValueError("Native NMFP matching requires equal full-density hop")
            ids.append(reference_id); arrays.append(emb); starts.append(pos)
            pos += len(emb); ends.append(pos)
        return cls(ids, np.concatenate(arrays), np.asarray(starts), np.asarray(ends), hop_s)

    def search(self, query, top_k=20):
        import faiss
        query = np.asarray(query, np.float32)
        tick = time.perf_counter()
        if not hasattr(self, "_index"):
            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(np.ascontiguousarray(self.embeddings))
        if not len(query):
            return {"reference_id": None, "start_s": None, "score": float("-inf"), "search_s": 0.0}
        _, neighbors = self._index.search(query, min(int(top_k), len(self.embeddings)))
        candidate_starts = np.unique((neighbors - np.arange(len(query))[:, None]).ravel())
        candidate_starts = candidate_starts[candidate_starts >= 0]
        tracks = np.searchsorted(self.starts, candidate_starts, side="right") - 1
        keep = candidate_starts + len(query) <= self.ends[tracks]
        candidate_starts, tracks = candidate_starts[keep], tracks[keep]
        if not len(candidate_starts):
            return {"reference_id": None, "start_s": None, "score": float("-inf"), "search_s": time.perf_counter() - tick}
        # Diagonal dot products avoid the QxQ matrix created upstream while
        # implementing exactly the same aligned mean-cosine score.
        scores = np.asarray([np.sum(query * self.embeddings[begin:begin + len(query)], axis=1).mean()
                             for begin in candidate_starts])
        winner = int(np.argmax(scores))
        return {"reference_id": self.reference_ids[int(tracks[winner])],
                "start_s": float(candidate_starts[winner] - self.starts[tracks[winner]]) * self.hop_s,
                "score": float(scores[winner]), "candidate_count": len(candidate_starts),
                "search_s": time.perf_counter() - tick}

    def build_index(self):
        """Build outside the query timer, as for the other benchmark methods."""
        import faiss
        tick = time.perf_counter()
        self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self._index.add(np.ascontiguousarray(self.embeddings))
        return time.perf_counter() - tick

    def build_cpu_ivf(self, nlist=1024, nprobe=16, seed=20261002):
        """Fit only a retrieval index on gallery embeddings, preserving the model."""
        import faiss
        tick = time.perf_counter()
        dimension = self.embeddings.shape[1]
        self._index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension,
                                        int(nlist), faiss.METRIC_INNER_PRODUCT)
        self._index.cp.seed = int(seed)
        self._index.cp.niter = 25
        self._index.train(np.ascontiguousarray(self.embeddings))
        self._index.add(np.ascontiguousarray(self.embeddings))
        self._index.nprobe = int(nprobe)
        return time.perf_counter() - tick


def extract_manifest(args, extractor_class=NMFPExtractor):
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    extractor = extractor_class(args.upstream, args.model_dir, args.batch_size, args.threads, args.precision, args.hop)
    cache_dir = pathlib.Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / args.cache_index_name
    result = {"manifest_path": str(pathlib.Path(args.manifest).resolve()),
              "manifest_sha256": file_hash(args.manifest), "provenance": extractor.provenance,
              "references": {}, "queries": {}, "failures": []}
    if index_path.exists():
        previous = json.loads(index_path.read_text())
        if previous["manifest_sha256"] == result["manifest_sha256"] and previous["provenance"] == result["provenance"]:
            result = previous
        elif previous["provenance"] == result["provenance"]:
            # A physical query-boundary correction does not change reference
            # features. Preserve only source-hash-verified current gallery refs.
            canonical = {r["reference_id"]: r for r in manifest["references"]
                         if r.get("role", "test_known") in ("test_known", "calibration_known")}
            for rid, entry in previous["references"].items():
                if rid not in canonical or not pathlib.Path(entry["path"]).exists():
                    continue
                _, _, saved = load_cached_embeddings(entry["path"])
                specification = saved["specification"]
                if specification["source_sha256"] == file_hash(canonical[rid]["path"]) and specification["path"] == str(pathlib.Path(canonical[rid]["path"]).resolve()):
                    result["references"][rid] = {**entry, "item": canonical[rid]}
    def persist():
        temporary = index_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(index_path)
    gallery_roles = {"test_known", "calibration_known"}
    refs = [r for r in manifest["references"] if r.get("role", "test_known") in gallery_roles]
    queries = [q for q in manifest["queries"] if q.get("evaluate", True)]
    if args.limit_references is not None:
        refs = refs[:args.limit_references]
    if args.limit_queries is not None:
        queries = queries[:args.limit_queries]
    tasks = []
    if not args.queries_only:
        tasks.extend(("reference", r) for r in refs)
    if not args.references_only:
        tasks.extend(("query", centered_query(q, duration))
                     for q in sorted(queries, key=lambda q: (q["path"], q.get("annotation_index", 0), q["query_id"]))
                     for duration in args.durations)
    for n, (kind, item) in enumerate(tasks):
        item_id = item["reference_id"] if kind == "reference" else item["query_id"]
        try:
            path, metadata, cached = extractor.cache(item, cache_dir, kind)
            collection = "references" if kind == "reference" else "queries"
            result[collection][item_id] = {"path": str(path), "timing": metadata["timing"], "item": item}
            print(json.dumps({"progress": f"{n + 1}/{len(tasks)}", "kind": kind, "id": item_id,
                              "cached": cached, **metadata["timing"]}), flush=True)
        except Exception as exc:
            result["failures"].append({"kind": kind, "id": item_id, "error": repr(exc)})
            persist()
            raise
        if n % 10 == 0:
            persist()
    persist()
    print(f"Cache manifest: {index_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=pathlib.Path)
    p.add_argument("--cache-dir", required=True, type=pathlib.Path)
    p.add_argument("--cache-index-name", default="cache_index.json",
                   help="Separate filenames permit independent reference/query phases")
    p.add_argument("--upstream", type=pathlib.Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--precision", choices=("float32", "mixed_float16"), default="float32")
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--durations", type=float, nargs="+", default=[5, 1, 2, 3, 10])
    p.add_argument("--limit-references", type=int)
    p.add_argument("--limit-queries", type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--queries-only", action="store_true")
    group.add_argument("--references-only", action="store_true")
    extract_manifest(p.parse_args())


if __name__ == "__main__":
    main()
