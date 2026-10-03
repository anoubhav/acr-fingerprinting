"""Reproducible speaker-disjoint content-ID adaptation of LibriSpeech test-clean.

This creates synthetic content-identification queries from an ASR corpus; it is
not a standard LibriSpeech fingerprinting benchmark. Audio is kept in work/.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import scipy.signal
import soundfile as sf
from protocol import assign_roles

CONDITIONS = ["clean", "mp3_32kbps", "gain_plus6db", "gain_minus6db",
              "clipping_peak40pct", "white_noise_snr10db", "preemphasis_0p9",
              "eq_tilt_plus6db", "eq_tilt_minus6db", "packet_dropout10pct"]
SAMPLE_RATE = 16000
SEED = "acr-librispeech-speaker-v1"


def condition_waveform(clean, condition, source_id):
    """Deterministic amplitude/EQ/noise/dropout transforms on a 5 s crop."""
    y = clean.copy()
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{source_id}:{condition}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    if condition == "gain_plus6db":
        y *= 10 ** (6 / 20)
    elif condition == "gain_minus6db":
        y *= 10 ** (-6 / 20)
    elif condition == "clipping_peak40pct":
        threshold = 0.4 * np.max(np.abs(y))
        y = np.clip(y, -threshold, threshold)
    elif condition == "white_noise_snr10db":
        noise = rng.standard_normal(len(y))
        noise *= np.sqrt(np.mean(y.astype(float) ** 2) / (10 * np.mean(noise ** 2)))
        y = y + noise
    elif condition == "preemphasis_0p9":
        y = scipy.signal.lfilter([1.0, -0.9], [1.0], y)
    elif condition in ("eq_tilt_plus6db", "eq_tilt_minus6db"):
        sign = 1 if condition == "eq_tilt_plus6db" else -1
        # Log-amplitude tilt from -6 to +6 dB across DC..Nyquist (reverse for -).
        frequency_tilt_db = sign * 6 * np.linspace(-1, 1, len(y) // 2 + 1)
        y = np.fft.irfft(np.fft.rfft(y) * 10 ** (frequency_tilt_db / 20), n=len(y))
    elif condition == "packet_dropout10pct":
        packet_samples = SAMPLE_RATE // 50  # 20 ms; clock duration unchanged.
        packets = len(y) // packet_samples
        chosen = rng.choice(packets, size=round(0.1 * packets), replace=False)
        for packet in chosen:
            y[packet * packet_samples:(packet + 1) * packet_samples] = 0
    elif condition not in ("clean", "mp3_32kbps"):
        raise ValueError(condition)
    return np.asarray(y, dtype=np.float32)


def _prepare_utterance(args):
    source, role, cache = args
    source_id = source["source_id"]
    y, sr = sf.read(source["path"], dtype="float32")
    assert sr == SAMPLE_RATE and y.ndim == 1
    start_sample = (len(y) - 5 * SAMPLE_RATE) // 2
    clean = y[start_sample:start_sample + 5 * SAMPLE_RATE]
    # Common headroom avoids unintended int16 saturation in the gain/EQ/noise
    # conditions. It is applied to every condition and recorded explicitly.
    peak = float(np.max(np.abs(clean)))
    common_gain = min(1.0, 0.35 / max(peak, 1e-12))
    clean = clean * common_gain
    out = []
    for condition in CONDITIONS:
        path = Path(cache) / condition / f"{source_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if condition == "mp3_32kbps":
                clean_path = Path(cache) / "clean" / f"{source_id}.wav"
                mp3_path = path.with_suffix(".mp3")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clean_path),
                                "-c:a", "libmp3lame", "-b:a", "32k", str(mp3_path)], check=True)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3_path),
                                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(path)], check=True)
                mp3_path.unlink()
            else:
                wave = condition_waveform(clean, condition, source_id)
                if np.max(np.abs(wave)) >= 1:
                    raise ValueError(f"Unintended saturation for {source_id} {condition}")
                sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")
        info = sf.info(path)
        if info.frames != 5 * SAMPLE_RATE or info.samplerate != SAMPLE_RATE:
            raise ValueError(f"Incorrect crop length/sample rate: {path} {info}")
        out.append({"query_id": f"librispeech_test_clean:{source_id}:{condition}",
                    "source_query_id": source_id,
                    "path": str(path.resolve()), "start_s": 0.0, "duration_s": 5.0,
                    "reference_id": source_id, "source_id": source_id,
                    "speaker_id": source["speaker_id"], "role": role, "evaluate": True,
                    "expected_reference_start_s": start_sample / SAMPLE_RATE,
                    "expected_reference_end_s": start_sample / SAMPLE_RATE + 5,
                    "expected_time_scale": 1.0, "localization_tolerance_s": 2.0,
                    "condition": condition, "common_headroom_gain": common_gain,
                    "overlapping_other_annotations": [],
                    "annotation": {"tempo": "100", "pitch": "", "condition": condition}})
    return out


def prepare(root, cache, workers=4):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    references = []
    for path in sorted(root.rglob("*.flac")):
        info = sf.info(path)
        if info.duration < 6.0:
            continue
        sid = path.stem
        references.append({"reference_id": sid, "source_id": sid, "speaker_id": sid.split("-")[0],
                           "path": str(path.resolve()), "duration_s": info.duration,
                           "sample_rate": info.samplerate, "channels": info.channels,
                           "license_url": "https://creativecommons.org/licenses/by/4.0/",
                           "dataset_url": "https://www.openslr.org/12"})
    speakers = {r["speaker_id"] for r in references}
    roles = assign_roles(speakers, seed=SEED)
    for ref in references:
        ref["role"] = roles[ref["speaker_id"]]
    query_refs = [r for r in references if r["role"] != "pca_fit"]
    args = [(r, r["role"], str(cache)) for r in query_refs]
    queries = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for i, batch in enumerate(executor.map(_prepare_utterance, args)):
            queries.extend(batch)
            if (i + 1) % 100 == 0:
                print(f"prepared speech {i + 1}/{len(query_refs)} utterances", flush=True)
    summary = {"dataset": "librispeech_test_clean_content_id", "reference_count": len(references),
               "speaker_count": len(speakers), "total_reference_audio_duration_s": sum(r["duration_s"] for r in references),
               "speaker_roles": dict(Counter(roles.values())),
               "reference_roles": dict(Counter(r["role"] for r in references)),
               "query_roles_per_condition": dict(Counter(r["role"] for r in query_refs)),
               "query_count": len(queries), "conditions": CONDITIONS,
               "gallery_reference_count": sum(r["role"] in ("calibration_known", "test_known") for r in references)}
    protocol = {"version": 1, "seed": SEED, "split_unit": "LibriSpeech speaker ID",
                "primary_task": "Synthetic centered 5 s content-ID adaptation, not standard ASR or fingerprint evaluation",
                "gallery_roles": ["calibration_known", "test_known"], "query_durations_s": [5],
                "localization_tolerance_s": 2.0, "test_label_tuning": False,
                "conditions": CONDITIONS,
                "common_query_headroom": "If necessary scale every variant to source crop peak 0.35 before perturbation",
                "clipping_definition": "Symmetric clipping at 40% of headroom-scaled crop peak",
                "eq_definition": "FFT amplitude tilt linearly from -6 dB at DC to +6 dB at Nyquist, or reverse",
                "packet_loss_definition": "Exactly 25 of 250 nonoverlapping 20 ms packets zeroed at deterministic uniform positions; no time compression",
                "mp3_definition": "FFmpeg libmp3lame 32 kbps, 16 kHz mono, gapless file encode/decode, exact 80000 samples verified",
                "threshold_selection": "Calibration unknown scores only; target empirical FPR <=0.01, report test counts/uncertainty"}
    return {"summary": summary, "references": references, "queries": queries, "protocol": protocol}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    manifest = prepare(args.root, args.cache, args.workers)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
