"""Paper-derived ACR fingerprints with an auditable exact-search reference.

This is a reconstruction, not recovered production code. Defaults and their
provenance are documented in ALGORITHM.md. NumPy is sufficient at 8 kHz;
resampling requires scipy and file loading requires soundfile or ffmpeg.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Iterable
import json
import shutil
import subprocess
import numpy as np

DECIMATION_FACTORS = (1, 2, 4, 6, 8)
AUDIO_DECODER_REVISION = "ffmpeg_first_v1"


@dataclass(frozen=True)
class FingerprintConfig:
    sample_rate: int = 8000
    n_fft: int = 1024
    hop_length: int = 186
    n_mels: int = 32
    fmin: float = 315.0
    fmax: float = 1960.0
    window_frames: int = 32
    stride_frames: int = 1
    standardize: bool = True
    deltas: bool = True
    quantize_fp16: bool = False
    pca_dim: int | None = 32
    spectrum_power: float = 1.0
    epsilon: float = 1e-8
    pad_audio: bool = True
    silence_std_threshold: float = 1e-6
    silence_mean_threshold: float = 1e-3

    def __post_init__(self):
        for name in ("sample_rate", "n_fft", "hop_length", "n_mels", "window_frames", "stride_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_mels < 2:
            raise ValueError("at least two mel bands are required")
        if not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("frequency bounds must be within Nyquist")
        if self.pca_dim is not None and not 1 <= self.pca_dim <= self.raw_dim:
            raise ValueError("PCA dimension exceeds the feature dimension")
        if self.spectrum_power <= 0 or self.epsilon <= 0:
            raise ValueError("spectrum_power and epsilon must be positive")

    @property
    def raw_dim(self):
        return self.n_mels * 2 - 1 if self.deltas else self.n_mels

    @property
    def output_dim(self):
        return self.raw_dim if self.pca_dim is None else self.pca_dim

    @property
    def fingerprint_rate(self):
        return self.sample_rate / (self.hop_length * self.stride_frames)

    @property
    def support_seconds(self):
        return (self.n_fft + (self.window_frames - 1) * self.hop_length) / self.sample_rate

    @property
    def first_center_seconds(self):
        padding = self.n_fft // 2 if self.pad_audio else 0
        return self.support_seconds / 2 - padding / self.sample_rate

    def byte_budget(self, factor=1, storage_dtype="float32"):
        """Asymptotic bytes/hour; array metadata and file headers excluded."""
        if factor < 1:
            raise ValueError("factor must be positive")
        count = 3600 * self.fingerprint_rate / factor
        payload = self.output_dim * np.dtype(storage_dtype).itemsize
        return {"fingerprints_per_hour": count, "payload_bytes_per_fp": payload,
                "payload_bytes_per_hour": count * payload,
                "metadata_bytes_per_fp": 12,
                "array_bytes_per_hour": count * (payload + 12)}


def ablation_configs(base=None):
    """Prespecified one-change-at-a-time variants; none selected on test data."""
    b = base or FingerprintConfig()
    variants = {"default": b, "no_standardization": replace(b, standardize=False),
                "no_deltas": replace(b, deltas=False),
                "fp16_roundtrip": replace(b, quantize_fp16=True),
                "no_pca": replace(b, pca_dim=None),
                "window16": replace(b, window_frames=16),
                "window48": replace(b, window_frames=48),
                "stride2": replace(b, stride_frames=2)}
    for d in (16, 32, 64):
        if d <= b.raw_dim:
            variants[f"pca{d}"] = replace(b, pca_dim=d)
    # With 32 bands, 63 pre-PCA values cannot produce a 64-D PCA. A
    # 64-D ablation therefore requires the explicitly different 64-band input.
    if b.raw_dim < 64:
        variants["mel64_pca64"] = replace(b, n_mels=64, pca_dim=64)
    return variants


def _hz_to_mel(freq):
    # Slaney's linear scale below 1 kHz and logarithmic scale above it.
    freq = np.asarray(freq, dtype=np.float64)
    return np.where(freq >= 1000,
                    15 + np.log(np.maximum(freq, 1e-30) / 1000) / (np.log(6.4) / 27),
                    freq / (200 / 3))


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel >= 15, 1000 * np.exp((mel - 15) * (np.log(6.4) / 27)),
                    mel * (200 / 3))


def mel_filterbank(config):
    """Slaney triangular bands with area normalization, no logarithm."""
    edges = _mel_to_hz(np.linspace(_hz_to_mel(config.fmin), _hz_to_mel(config.fmax),
                                  config.n_mels + 2))
    freqs = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    lower = (freqs[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    upper = (edges[2:, None] - freqs[None, :]) / np.diff(edges)[1:, None]
    filt = np.maximum(0, np.minimum(lower, upper))
    filt *= (2 / (edges[2:] - edges[:-2]))[:, None]
    if np.any(filt.sum(axis=1) == 0):
        raise ValueError("empty mel band: increase FFT size or reduce mel count")
    return filt.astype(np.float32)


def prepare_audio(audio, sample_rate, target_rate):
    """Mean-downmix sample-major channels and anti-alias before resampling."""
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("audio must be finite samples or samples x channels")
    if sample_rate < 1:
        raise ValueError("sample rate must be positive")
    if sample_rate != target_rate:
        from scipy.signal import resample_poly
        divisor = gcd(int(sample_rate), int(target_rate))
        y = resample_poly(y, target_rate // divisor, sample_rate // divisor)
    return y.astype(np.float32)


def load_audio(path, sample_rate=8000):
    """Use the same FFmpeg decoder/resampler for every benchmark method.

    Soundfile is a fallback only when the FFmpeg executable is unavailable.
    Decoding errors are not silently routed to a different backend.
    """
    if shutil.which("ffmpeg"):
        data = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                               "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
                              check=True, capture_output=True).stdout
        return np.frombuffer(data, dtype="<f4").copy(), sample_rate
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    return prepare_audio(y, sr, sample_rate), sample_rate


@lru_cache(maxsize=1)
def audio_decoder_info():
    """Serializable backend/version information for manifests and cache keys."""
    executable = shutil.which("ffmpeg")
    if executable:
        version = subprocess.run([executable, "-version"], check=True,
                                 capture_output=True, text=True).stdout.splitlines()[0]
        return {"backend": "ffmpeg", "version": version, "revision": AUDIO_DECODER_REVISION}
    import soundfile as sf
    return {"backend": "soundfile", "version": sf.__version__,
            "libsndfile_version": sf.__libsndfile_version__, "revision": AUDIO_DECODER_REVISION}


def standardize_rows(x, epsilon=1e-8):
    """Per-vector population standard deviation; constant vectors become zero."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return np.divide(centered, sd, out=np.zeros_like(centered), where=sd > epsilon).astype(np.float32)


class Fingerprinter:
    def __init__(self, config=None):
        self.config = config or FingerprintConfig()
        self.filters = mel_filterbank(self.config)
        self.pca_mean = None
        self.pca_components = None
        self.explained_variance_ratio = None
        self.fit_content_ids = []
        self.fit_fingerprint_count = 0

    def mel_spectrogram(self, audio, sample_rate):
        c = self.config
        y = prepare_audio(audio, sample_rate, c.sample_rate)
        if c.pad_audio:
            y = np.pad(y, (c.n_fft // 2, c.n_fft // 2))
        if len(y) < c.n_fft:
            return np.empty((0, c.n_mels), dtype=np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(y, c.n_fft)[::c.hop_length]
        # Periodic Hann; center=False after explicit zero padding.
        hann = .5 - .5 * np.cos(2 * np.pi * np.arange(c.n_fft) / c.n_fft)
        # Bound FFT scratch space independently of clip length. Long public
        # references must not allocate a full frames x FFT float64 matrix.
        mel = np.empty((len(frames), c.n_mels), dtype=np.float32)
        for start in range(0, len(frames), 2048):
            frame_batch = frames[start:start + 2048]
            magnitude = np.abs(np.fft.rfft(frame_batch * hann, axis=1))
            mel[start:start + len(frame_batch)] = (magnitude ** c.spectrum_power) @ self.filters.T
        return mel

    def raw_features(self, audio, sample_rate):
        """Return pre-PCA rows and fingerprint center times relative to input."""
        mel = self.mel_spectrogram(audio, sample_rate)
        return self.raw_features_from_mels(mel)

    def raw_features_from_mels(self, mel):
        """Apply the recovered transformations to time-major linear mel rows.

        This supports shared STFT work across prespecified window/transform
        ablations. The supplied mel matrix must use this config's spectral
        parameters and explicit-padding convention.
        """
        c = self.config
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[1] != c.n_mels or not np.all(np.isfinite(mel)):
            raise ValueError("invalid time-major mel spectrogram")
        if len(mel) < c.window_frames:
            return np.empty((0, c.raw_dim), np.float32), np.empty(0, np.float64)
        starts = np.arange(0, len(mel) - c.window_frames + 1, c.stride_frames)
        cumulative = np.vstack([np.zeros((1, c.n_mels)), np.cumsum(mel, axis=0, dtype=np.float64)])
        average = ((cumulative[starts + c.window_frames] - cumulative[starts]) / c.window_frames).astype(np.float32)
        # Source applies skip on the grid before silence filtering. Preserve
        # original grid indices when rows are removed; never compact time.
        valid = ((np.std(average, axis=1) > c.silence_std_threshold)
                 & (np.mean(average, axis=1) > c.silence_mean_threshold))
        average, starts = average[valid], starts[valid]
        features = standardize_rows(average, c.epsilon) if c.standardize else average
        if c.deltas:
            delta = np.diff(average, axis=1)
            if c.standardize:
                delta = standardize_rows(delta, c.epsilon)
            features = np.concatenate([features, delta], axis=1)
        # Optional precision ablation of the complete pre-PCA feature rows.
        # The documented SDK mode sends 32 raw band means, before these
        # normalization/delta transforms; it does not send these 63-D rows.
        if c.quantize_fp16:
            if np.max(np.abs(features), initial=0) > np.finfo(np.float16).max:
                raise ValueError("pre-PCA features overflow float16")
            features = features.astype(np.float16).astype(np.float32)
        centers = starts * c.hop_length / c.sample_rate + c.first_center_seconds
        return features.astype(np.float32), centers.astype(np.float64)

    def fit(self, feature_batches: Iterable[np.ndarray], content_ids=None):
        """Fit PCA on disjoint calibration-content features using streaming moments.

        This method never extracts reference/test rows implicitly. The caller
        supplies calibration batches and identifiers for auditable split checks.
        """
        c = self.config
        count, total, cross = 0, np.zeros(c.raw_dim), np.zeros((c.raw_dim, c.raw_dim))
        for batch in feature_batches:
            x = np.asarray(batch, dtype=np.float64)
            if x.ndim != 2 or x.shape[1] != c.raw_dim or not np.all(np.isfinite(x)):
                raise ValueError("invalid calibration feature matrix")
            count += len(x)
            total += x.sum(axis=0)
            cross += x.T @ x
        self.fit_content_ids = [str(x) for x in (content_ids if content_ids is not None else [])]
        self.fit_fingerprint_count = count
        if c.pca_dim is None:
            return self
        if count < max(2, c.pca_dim + 1):
            raise ValueError("too few calibration fingerprints for requested PCA dimension")
        mean = total / count
        cov = (cross - np.outer(total, total) / count) / (count - 1)
        cov = (cov + cov.T) / 2
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0)
        components = vectors[:, order[:c.pca_dim]].T
        # Fix eigenvector signs for deterministic serialization and features.
        for row in components:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        self.pca_mean = mean.astype(np.float32)
        self.pca_components = components.astype(np.float32)
        self.explained_variance_ratio = (values[:c.pca_dim] / max(values.sum(), c.epsilon)).astype(np.float64)
        return self

    def fit_audio(self, entries):
        """Fit on (content_id, audio, sample_rate) calibration tuples."""
        entries = list(entries)
        return self.fit((self.raw_features(y, sr)[0] for _, y, sr in entries),
                        content_ids=[cid for cid, _, _ in entries])

    def transform(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.config.raw_dim or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        if self.config.pca_dim is None:
            return x.copy()
        if self.pca_components is None:
            raise RuntimeError("fit PCA on disjoint calibration content first")
        return ((x - self.pca_mean) @ self.pca_components.T).astype(np.float32)

    def extract(self, audio, sample_rate):
        x, times = self.raw_features(audio, sample_rate)
        return self.transform(x), times

    def assert_disjoint(self, reference_ids=(), query_ids=()):
        overlap = set(self.fit_content_ids) & (set(map(str, reference_ids)) | set(map(str, query_ids)))
        if overlap:
            raise ValueError(f"calibration content overlaps reference/query: {sorted(overlap)[:5]}")

    def save(self, path):
        metadata = {"config": asdict(self.config), "fit_content_ids": self.fit_content_ids,
                    "fit_fingerprint_count": self.fit_fingerprint_count, "format_version": 1}
        arrays = {"metadata": np.array(json.dumps(metadata))}
        if self.pca_components is not None:
            arrays.update(mean=self.pca_mean, components=self.pca_components,
                          explained=self.explained_variance_ratio)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta.get("format_version") != 1:
                raise ValueError("unsupported fingerprint model format")
            obj = cls(FingerprintConfig(**meta["config"]))
            obj.fit_content_ids = meta["fit_content_ids"]
            obj.fit_fingerprint_count = meta["fit_fingerprint_count"]
            if "mean" in z:
                obj.pca_mean = z["mean"].copy()
                obj.pca_components = z["components"].copy()
                obj.explained_variance_ratio = z["explained"].copy()
            return obj


class ExactIndex:
    """Exhaustive squared Euclidean search; no ANN/hardware confound."""
    def __init__(self, vectors, content_codes, times, content_ids, factor=1):
        self.vectors = np.ascontiguousarray(vectors)
        self.content_codes = np.asarray(content_codes, dtype=np.int32)
        self.times = np.asarray(times, dtype=np.float64)
        self.content_ids = list(map(str, content_ids))
        self.factor = int(factor)
        if self.vectors.ndim != 2 or not len(self.vectors):
            raise ValueError("index must contain a nonempty feature matrix")
        if len(self.content_codes) != len(self.vectors) or len(self.times) != len(self.vectors):
            raise ValueError("index metadata length mismatch")
        if not np.all(np.isfinite(self.vectors)) or not np.all(np.isfinite(self.times)):
            raise ValueError("index values must be finite")
        if np.any(self.content_codes < 0) or np.any(self.content_codes >= len(self.content_ids)):
            raise ValueError("invalid content code")
        self._search_vectors = np.asarray(self.vectors, dtype=np.float32)
        self._norms = np.sum(self._search_vectors * self._search_vectors, axis=1)
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss.IndexFlatL2(self.vectors.shape[1])
            self._faiss.add(self._search_vectors)
        except ImportError:
            pass

    @classmethod
    def from_entries(cls, entries, factor=1, storage_dtype="float32", phase=0,
                     grid_step_sec=None, grid_origin_sec=0.0):
        """Store one in factor grid frames (skip=factor-1) per content.

        Supply grid_step_sec=config.hop_length/config.sample_rate and
        grid_origin_sec=config.first_center_seconds to reproduce source skip
        before silence removal. Without these, decimation is over surviving
        rows, an explicit alternative for generic fingerprint representations.
        """
        if not isinstance(factor, int) or factor < 1 or not 0 <= phase < factor:
            raise ValueError("factor must be a positive integer and phase in [0,factor)")
        arrays, codes, times, names = [], [], [], []
        for cid, x, t in entries:
            x, t = np.asarray(x), np.asarray(t)
            if x.ndim != 2 or len(x) != len(t):
                raise ValueError("invalid entry")
            if np.any(np.diff(t) <= 0):
                raise ValueError("entry timestamps must increase strictly")
            if str(cid) in names:
                raise ValueError("content identifiers must be unique")
            if grid_step_sec is None:
                selection = np.arange(phase, len(x), factor)
            else:
                if grid_step_sec <= 0:
                    raise ValueError("grid step must be positive")
                frames = np.rint((t - grid_origin_sec) / grid_step_sec).astype(np.int64)
                selection = np.flatnonzero(frames % factor == phase)
            selected = x[selection]
            arrays.append(selected)
            codes.append(np.full(len(selected), len(names), dtype=np.int32))
            times.append(t[selection])
            names.append(str(cid))
        if not arrays:
            raise ValueError("no reference content supplied")
        return cls(np.concatenate(arrays).astype(storage_dtype), np.concatenate(codes),
                   np.concatenate(times), names, factor=factor)

    def search(self, queries, k=5, block_size=32768):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.vectors.shape[1] or not np.all(np.isfinite(q)):
            raise ValueError("invalid query matrix")
        if k < 1:
            raise ValueError("k must be positive")
        k = min(k, len(self.vectors))
        if self._faiss is not None:
            return self._faiss.search(q, k)
        # Chunked direct subtraction avoids a quadratic output allocation and
        # cancellation errors in the ||q||² + ||r||² - 2q.r identity.
        result_d = np.empty((len(q), k), np.float32)
        result_i = np.empty((len(q), k), np.int64)
        for row, query in enumerate(q):
            best_d, best_i = np.empty(0, np.float32), np.empty(0, np.int64)
            for start in range(0, len(self.vectors), block_size):
                refs = self._search_vectors[start:start + block_size]
                d = np.sum((refs - query) ** 2, axis=1)
                i = np.arange(start, start + len(refs))
                joined_d, joined_i = np.r_[best_d, d], np.r_[best_i, i]
                order = np.lexsort((joined_i, joined_d))[:k]
                best_d, best_i = joined_d[order], joined_i[order]
            result_d[row], result_i[row] = best_d, best_i
        return result_d, result_i

    def match(self, queries, query_times, top_k=5, tolerance_sec=.12,
              max_distance=np.inf, min_votes=1, limit=10):
        """Temporal offset consensus of top-k exact fingerprint neighbours.

        Each query row contributes at most one vote to a content. Inliers must
        fit reference_time = query_time + offset within tolerance_sec. The
        tolerance accommodates reference-grid decimation without requiring the
        exact stored frame. Score is vote_fraction/(1+mean_squared_L2), not a
        calibrated probability; calibrate acceptance thresholds on held-out data.
        """
        qt = np.asarray(query_times, dtype=np.float64)
        if len(queries) != len(qt) or not np.all(np.isfinite(qt)) or np.any(np.diff(qt) <= 0):
            raise ValueError("query times must match rows and increase strictly")
        if tolerance_sec < 0 or min_votes < 1:
            raise ValueError("invalid temporal matching parameters")
        if not len(queries):
            return []
        distances, ids = self.search(queries, top_k)
        qrows = np.repeat(np.arange(len(qt)), ids.shape[1])
        ri, ds = ids.ravel(), distances.ravel()
        # FAISS approximate indexes can return -1/maximum-distance padding
        # when a probed list has fewer than k vectors. Never map -1 to the
        # final reference row through NumPy's negative indexing semantics.
        accepted = (ri >= 0) & (ri < len(self.vectors)) & np.isfinite(ds) & (ds <= max_distance)
        qrows, ri, ds = qrows[accepted], ri[accepted], ds[accepted]
        offsets = self.times[ri] - qt[qrows]
        candidates = []
        for code in np.unique(self.content_codes[ri]):
            mask = self.content_codes[ri] == code
            oi, qi, di = offsets[mask], qrows[mask], ds[mask]
            sort = np.argsort(oi, kind="stable")
            oi, qi, di = oi[sort], qi[sort], di[sort]
            best = None
            right = 0
            # A sorted interval of width 2*tolerance has a feasible common
            # offset at its midpoint, including arbitrary decimation phases.
            for left in range(len(oi)):
                right = max(right, left)
                while right < len(oi) and oi[right] - oi[left] <= 2 * tolerance_sec + 1e-12:
                    right += 1
                chosen = {}
                for j in range(left, right):
                    old = chosen.get(int(qi[j]))
                    if old is None or di[j] < di[old]:
                        chosen[int(qi[j])] = j
                if not chosen:
                    continue
                selected = np.array(list(chosen.values()))
                votes = len(selected)
                mean_d = float(np.mean(di[selected]))
                rank = (votes, -mean_d)
                if best is None or rank > best[0]:
                    # Clip median to feasible interval; all chosen rows remain
                    # within tolerance of the returned offset.
                    lo, hi = oi[selected].max() - tolerance_sec, oi[selected].min() + tolerance_sec
                    offset = float(np.clip(np.median(oi[selected]), lo, hi))
                    residual = float(np.max(np.abs(oi[selected] - offset)))
                    best = (rank, votes, mean_d, offset, residual)
            if best is not None and best[1] >= min_votes:
                _, votes, mean_d, offset, residual = best
                ratio = votes / len(qt)
                candidates.append({"content_id": self.content_ids[int(code)],
                                   "votes": votes, "vote_fraction": ratio,
                                   "mean_distance": mean_d, "offset_sec": offset,
                                   "max_time_residual_sec": residual,
                                   "score": ratio / (1 + mean_d), "query_fingerprints": len(qt)})
        candidates.sort(key=lambda x: (-x["votes"], x["mean_distance"], x["content_id"]))
        return candidates[:limit]

    def bytes(self):
        payload = self.vectors.nbytes
        metadata = self.content_codes.nbytes + self.times.nbytes
        return {"fingerprints": len(self.vectors), "payload_bytes": payload,
                "metadata_bytes": metadata, "array_bytes": payload + metadata,
                "content_lookup_bytes": sum(len(x.encode("utf8")) for x in self.content_ids),
                "search_float32_copy_bytes": 0 if self.vectors.dtype == np.float32 else self._search_vectors.nbytes,
                "faiss_vector_copy_bytes": self._search_vectors.nbytes if self._faiss is not None else 0,
                "norm_cache_bytes": self._norms.nbytes}

    def save(self, path):
        np.savez_compressed(path, vectors=self.vectors, content_codes=self.content_codes,
                            times=self.times, content_ids=np.asarray(self.content_ids),
                            factor=np.array(self.factor))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["vectors"], z["content_codes"], z["times"], z["content_ids"], int(z["factor"]))
