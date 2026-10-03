"""Published small PeakNetFP checkpoint with unchanged model/peak frontend.

Peak functions are loaded as exact AST function definitions from the official
trainer module to avoid importing its optimizer/training-only dependencies.
All model variables must match the official checkpoint before inference.
"""
from __future__ import annotations
import argparse
import ast
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from neural_baseline import NMFPExtractor, NMFPSequenceIndex, extract_manifest, file_hash

WORKSPACE = Path(__file__).resolve().parents[2]
PEAK_COMMIT = "d55071a644f20fd6448fb8ed33c0c115769fb7a5"
KAPRE_COMMIT = "ae8f5077ec9071e5529992702885de03c3321941"
ARCHIVE_MD5 = "df23e1556a206fd72bfa99d0724d241f"


class PeakNetExtractor(NMFPExtractor):
    def __init__(self, upstream=None, model_dir=None, batch_size=125,
                 threads=4, precision="float32", hop_s=0.5):
        upstream = Path(upstream or WORKSPACE / "work/upstream/peaknetfp").resolve()
        model_dir = Path(model_dir or WORKSPACE / "work/models/peaknetfp/peaknetfp_checkpoints/peaknetfp").resolve()
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if revision != PEAK_COMMIT:
            raise ValueError("Unexpected PeakNetFP revision")
        dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty:
            raise ValueError("Tracked PeakNetFP source was modified")
        if precision != "float32":
            raise ValueError("Published PeakNetFP config uses float32")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        import tensorflow as tf
        import yaml
        import kapre
        tf.config.threading.set_intra_op_parallelism_threads(int(threads))
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.random.set_seed(13)
        sys.path.insert(0, str(upstream))
        from model.fp.melspec.melspectrogram import get_melspec_layer
        from model.fp.nnfp import get_fingerprinter
        config_path = upstream / "config/peaknetfp.yaml"
        cfg = yaml.safe_load(config_path.read_text())
        self.fs = int(cfg["MODEL"]["FS"])
        self.segment_s = float(cfg["MODEL"]["DUR"])
        self.hop_s = float(hop_s)
        self.segment_samples = round(self.fs * self.segment_s)
        self.hop_samples = round(self.fs * self.hop_s)
        self.batch_size = int(batch_size)
        self.decoded_audio_cache = OrderedDict()
        self.frontend = get_melspec_layer(cfg, trainable=False)
        self.model = get_fingerprinter(cfg, trainable=False)
        self.tf, self.cfg = tf, cfg
        selected_names = {"extract_peak_tensors_from_mel", "sort_peaks", "pad_peaks",
                          "sample_peaks_matrix", "tiled_peaks_matrix", "normalize_peaks",
                          "extract_peaks", "get_peaks_from_mel"}
        trainer_path = upstream / "model/trainer.py"
        tree = ast.parse(trainer_path.read_text())
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected_names]
        if {node.name for node in definitions} != selected_names:
            raise ValueError("Official peak frontend definitions missing")
        namespace = {"tf": tf, "np": np, "sys": sys, "PADDING_VALUE": -1.0}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(trainer_path), "exec"), namespace)
        self.peak_function = namespace["get_peaks_from_mel"]
        tone = np.sin(2 * np.pi * 440 * np.arange(self.segment_samples, dtype=np.float32) / self.fs)[None, None, :]
        points = self.peak_function(self.frontend(tone), cfg)
        self.model(points)
        checkpoint_path = str(model_dir / "ckpt-100")
        checkpoint = tf.train.Checkpoint(model=self.model)
        status = checkpoint.restore(checkpoint_path)
        status.assert_existing_objects_matched()
        status.expect_partial()
        self.forward = tf.function(lambda wave: self.model(self.peak_function(self.frontend(wave), self.cfg)))
        # Fixed batch shapes are required by the published dynamic partition
        # implementation. Padding duplicates complete windows; no samples from
        # another source enter a window, and only real-window outputs are kept.
        warm_wave = np.repeat(tone, self.batch_size, axis=0)
        output = np.asarray(self.forward(warm_wave), dtype=np.float32)
        if not np.all(np.isfinite(output)) or not np.allclose(np.linalg.norm(output, axis=1), 1, atol=5e-5):
            raise ValueError("Invalid restored PeakNetFP output")
        query_warm = np.asarray(self.forward(np.repeat(tone, 9, axis=0)), dtype=np.float32)
        if not np.allclose(query_warm[0], output[0], atol=2e-5):
            raise ValueError("PeakNetFP embeddings depend materially on batch padding")
        self.provenance = {
            "method": "PeakNetFP", "paper_url": "https://arxiv.org/abs/2506.21086",
            "upstream_url": "https://github.com/guillemcortes/peaknetfp", "upstream_commit": revision,
            "upstream_tracked_files_unmodified": True,
            "official_config_source": "config/peaknetfp.yaml", "model_config_sha256": file_hash(config_path),
            "official_model_config": cfg, "peak_function_source_sha256": file_hash(trainer_path),
            "peak_functions_loaded_without_modification": True,
            "checkpoint": "ckpt-100", "checkpoint_files_sha256": {p.name: file_hash(p) for p in sorted(model_dir.glob("ckpt-*"))},
            "model_archive_url": "https://zenodo.org/records/15782389/files/peaknetfp_checkpoints.tar.gz?download=1",
            "model_archive_md5_expected": ARCHIVE_MD5,
            "model_archive_md5_verified": file_hash(model_dir.parents[1] / "peaknetfp_checkpoints.tar.gz", "md5") == ARCHIVE_MD5,
            "restoration_assert_existing_objects_matched": True,
            "model_parameter_count": self.model.count_params(), "embedding_dimension": int(cfg["MODEL"]["EMB_SZ"]),
            "sample_rate": self.fs, "segment_duration_s": self.segment_s, "hop_s": self.hop_s,
            "inference_precision": "float32", "batch_size": self.batch_size,
            "query_batch_policy": "actual query frame count up to reference batch size; fixed padded reference batches",
            "intra_op_threads": int(threads), "inter_op_threads": 1,
            "decode_backend": "FFmpeg mono8k float32 PCM; shared benchmark decoding",
            "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "kapre_source_commit": KAPRE_COMMIT, "kapre_reported_version": kapre.__version__,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "tensorflow", "tf-keras", "librosa", "kapre", "scipy", "pyyaml")},
            "platform": platform.platform(), "machine": platform.machine(),
            "inference_runtime_change": "TF2.16.2 legacy Keras on Apple CPU; original Kapre0.3.7 source tag pinned, original model and peak functions unchanged",
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()

    def extract_file(self, path, start_s=0.0, duration_s=None):
        self.query_mode = duration_s is not None
        try:
            return super().extract_file(path, start_s, duration_s)
        finally:
            self.query_mode = False

    def extract_array(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < self.segment_samples:
            return np.empty((0, self.provenance["embedding_dimension"]), np.float32), np.empty(0), {"frontend_s": 0.0, "forward_s": 0.0}
        windows = np.lib.stride_tricks.sliding_window_view(audio, self.segment_samples)[::self.hop_samples]
        batch_size = min(self.batch_size, len(windows)) if getattr(self, "query_mode", False) else self.batch_size
        output, elapsed = [], 0.0
        for begin in range(0, len(windows), batch_size):
            batch = np.array(windows[begin:begin + batch_size], dtype=np.float32)
            real_count = len(batch)
            if real_count < batch_size:
                batch = np.concatenate([batch, np.repeat(batch[-1:], batch_size - real_count, axis=0)])
            tick = time.perf_counter()
            output.append(np.asarray(self.forward(batch[:, None, :]), dtype=np.float32)[:real_count])
            elapsed += time.perf_counter() - tick
        embeddings = np.concatenate(output)
        if not np.all(np.isfinite(embeddings)) or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=5e-5):
            raise ValueError("Invalid PeakNetFP embeddings")
        # Frontend and network are one warmed compiled graph, unlike NMFP's
        # separate Essentia frontend. Never infer a zero frontend cost here.
        return embeddings, np.arange(len(embeddings), dtype=np.float64) * self.hop_s, {
            "frontend_s": 0.0, "forward_s": elapsed, "combined_frontend_forward_s": elapsed,
            "timing_components_separable": False}


class PeakNetSequenceIndex(NMFPSequenceIndex):
    """Published per-track estimated stretching, then mean-cosine reranking."""
    def search(self, query, top_k=20):
        query = np.asarray(query, np.float32)
        if not len(query):
            return {"reference_id": None, "start_s": None, "score": -1.0, "search_s": 0.0}
        tick = time.perf_counter()
        if not hasattr(self, "_index"):
            self.build_index()
        _, neighbors = self._index.search(query, min(top_k, len(self.embeddings)))
        track_matches = {}
        for row, values in enumerate(neighbors):
            for index in values:
                track = int(np.searchsorted(self.starts, index, side="right") - 1)
                track_matches.setdefault(track, {}).setdefault(row, int(index))
        factors = {}
        for track, pairs in track_matches.items():
            q = np.array(list(pairs))
            r = np.array(list(pairs.values()))
            qdiff, rdiff = q[:, None] - q, r[:, None] - r
            ratios = np.divide(qdiff, rdiff, out=np.zeros_like(qdiff, dtype=float), where=rdiff != 0)
            nonzero = ratios[ratios != 0]
            factor = float(np.mean(nonzero)) if len(nonzero) else 1.0
            factors[track] = factor if 0.5 <= factor <= 2 else 1.0
        candidates = {}
        for row, values in enumerate(neighbors):
            for index in values:
                track = int(np.searchsorted(self.starts, index, side="right") - 1)
                factor = factors[track]
                start = round(int(index) - row / factor)
                ids = np.rint(start + np.arange(len(query)) / factor).astype(int)
                # Bound every candidate to one real track. The original script
                # bounds only against the concatenated DB, allowing crossings.
                if ids[0] < self.starts[track] or ids[-1] >= self.ends[track]:
                    continue
                candidates[(track, tuple(ids))] = factor
        if not candidates:
            return {"reference_id": None, "start_s": None, "score": -1.0, "search_s": time.perf_counter() - tick}
        best_key, best_score = None, -np.inf
        for (track, ids), factor in candidates.items():
            score = float(np.mean(np.sum(query * self.embeddings[list(ids)], axis=1)))
            if score > best_score:
                best_key, best_score = (track, ids, factor), score
        track, ids, factor = best_key
        return {"reference_id": self.reference_ids[track],
                "start_s": float(ids[0] - self.starts[track]) * self.hop_s,
                "score": best_score, "candidate_count": len(candidates),
                "estimated_reference_time_scale": 1 / factor,
                "search_s": time.perf_counter() - tick}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--cache-index-name", default="cache_index.json")
    p.add_argument("--upstream", type=Path, default=WORKSPACE / "work/upstream/peaknetfp")
    p.add_argument("--model-dir", type=Path, default=WORKSPACE / "work/models/peaknetfp/peaknetfp_checkpoints/peaknetfp")
    p.add_argument("--batch-size", type=int, default=125)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--precision", default="float32")
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--durations", type=float, nargs="+", default=[5])
    p.add_argument("--limit-references", type=int)
    p.add_argument("--limit-queries", type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--queries-only", action="store_true")
    group.add_argument("--references-only", action="store_true")
    extract_manifest(p.parse_args(), extractor_class=PeakNetExtractor)
