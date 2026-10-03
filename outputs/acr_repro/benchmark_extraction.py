"""Warm CPU extraction timing on disjoint PCA-fit sources, excluding file I/O.

This is host-computer timing, not a TV/embedded-device measurement. It records
wall and process CPU time and reports across-source medians and p95 values.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
from acr_fp import Fingerprinter, load_audio, audio_decoder_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clips", type=int, default=100)
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--concurrency-note", default="Concurrent jobs may affect wall time; CPU process time is also reported.")
    a = p.parse_args()
    from threadpoolctl import threadpool_limits, threadpool_info
    threadpool_limits(1, user_api="blas")
    model = Fingerprinter.load(a.model)
    protocol = json.loads(a.protocol.read_text())
    references = sorted([r for r in protocol["references"] if r["role"] == "pca_fit"],
                        key=lambda r: r["reference_id"])[:a.clips]
    if not references:
        raise ValueError("No disjoint PCA-fit sources")
    if set(r["reference_id"] for r in references) - set(model.fit_content_ids):
        raise ValueError("Timing sources were not part of the model's disjoint fit split")
    records = []
    for i, reference in enumerate(references):
        audio, sr = load_audio(reference["path"], model.config.sample_rate)
        duration = min(a.duration, len(audio)/sr)
        start = max(0, (len(audio)/sr-duration)/2)
        y = audio[round(start*sr):round(start*sr)+round(duration*sr)]
        for _ in range(a.warmups):
            model.extract(y, sr)
        wall, cpu, raw_wall, pca_wall = [], [], [], []
        for _ in range(a.repeats):
            cpu_start = time.process_time_ns()
            start_ns = time.perf_counter_ns()
            x, times = model.raw_features(y, sr)
            split_ns = time.perf_counter_ns()
            features = model.transform(x)
            end_ns = time.perf_counter_ns()
            cpu_end = time.process_time_ns()
            if not np.all(np.isfinite(features)):
                raise ValueError("Nonfinite timed fingerprint")
            wall.append((end_ns-start_ns)/1e6)
            cpu.append((cpu_end-cpu_start)/1e6)
            raw_wall.append((split_ns-start_ns)/1e6)
            pca_wall.append((end_ns-split_ns)/1e6)
        actual_duration = len(y)/sr
        records.append({"reference_id": reference["reference_id"],
            "audio_duration_s": actual_duration, "crop_start_s": start,
            "fingerprints": len(features), "wall_ms": wall, "cpu_ms": cpu,
            "raw_wall_ms": raw_wall, "pca_wall_ms": pca_wall,
            "median_wall_ms_per_audio_second": float(np.median(wall)/actual_duration),
            "median_cpu_ms_per_audio_second": float(np.median(cpu)/actual_duration),
            "median_raw_wall_ms_per_audio_second": float(np.median(raw_wall)/actual_duration),
            "median_pca_wall_ms_per_audio_second": float(np.median(pca_wall)/actual_duration)})
        if (i+1) % 20 == 0:
            print(f"Warm extraction timed {i+1}/{len(references)}", flush=True)
    wall_rates = np.array([r["median_wall_ms_per_audio_second"] for r in records])
    cpu_rates = np.array([r["median_cpu_ms_per_audio_second"] for r in records])
    rng = np.random.default_rng(20261002)
    bootstrap = np.median(wall_rates[rng.integers(0, len(records), (2000, len(records)))], axis=1)
    result = {"method": "ACR-PCA32 warm extraction", "requested_clips": a.clips,
        "actual_disjoint_sources": len(references), "repeats_per_source": a.repeats,
        "warmups_per_source": a.warmups, "requested_crop_duration_s": a.duration,
        "includes": "STFT, mel filtering, mean/deltas/standardization, silence gating, PCA transform",
        "excludes": "file decoding/I/O, model construction, indexing, search, networking",
        "median_wall_ms_per_audio_second": float(np.median(wall_rates)),
        "p95_wall_ms_per_audio_second": float(np.percentile(wall_rates, 95)),
        "median_cpu_ms_per_audio_second": float(np.median(cpu_rates)),
        "p95_cpu_ms_per_audio_second": float(np.percentile(cpu_rates, 95)),
        "median_wall_bootstrap95_ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "peak_process_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_note": "Process high-water mark includes Python/libraries/full-file decoding; not extraction-only peak RAM.",
        "concurrency_note": a.concurrency_note,
        "deployment_claim": "Host CPU timing only; no claim about embedded TV/device latency or RAM.",
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "numpy": np.__version__, "audio_decoder": audio_decoder_info(),
                        "threadpools": threadpool_info()},
        "protocol_sha256": hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
        "pca_model_sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(), "records": records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "environment")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
