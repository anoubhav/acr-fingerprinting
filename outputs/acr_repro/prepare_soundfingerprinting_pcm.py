"""Prepare exactly bounded FFmpeg float5512 inputs for the public C# API.

Decoding is independent of labels/model predictions. Media caches remain under
work/; no padding or external query context is added. Native normalization and
fingerprinting are performed later by the unmodified upstream library.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
from run_audfprint_protocol import shorten_query


def prepare_one(task):
    kind, identity, path, start, duration, cache = task
    key = hashlib.sha256(json.dumps({"path": path, "start": start, "duration": duration,
                                    "sample_rate": 5512, "dtype": "f32le"}, sort_keys=True).encode()).hexdigest()
    destination = Path(cache) / f"{key}.f32"
    metadata = destination.with_suffix(".json")
    if destination.exists() and metadata.exists():
        return kind, identity, json.loads(metadata.read_text())
    args = ["ffmpeg", "-v", "error", "-i", path]
    if start is not None:
        args += ["-ss", str(start), "-t", str(duration)]
    args += ["-f", "f32le", "-ac", "1", "-ar", "5512", "pipe:1"]
    before = time.perf_counter()
    result = subprocess.run(args, check=True, capture_output=True)
    elapsed = time.perf_counter() - before
    if len(result.stdout) % 4:
        raise ValueError("Non-float-aligned decode")
    samples = len(result.stdout) // 4
    if duration is not None and abs(samples - round(duration * 5512)) > 2:
        raise ValueError(f"Physical crop length mismatch for {identity}")
    destination.write_bytes(result.stdout)
    entry = {"pcm_path": str(destination.resolve()), "decoded_samples": samples,
             "decoded_duration_s": samples / 5512, "source_path": path,
             "start_s": start, "duration_s": duration, "decode_s": elapsed,
             "warning": result.stderr.decode(errors="replace")[:2048],
             "sha256": hashlib.sha256(result.stdout).hexdigest()}
    metadata.write_text(json.dumps(entry, indent=2) + "\n")
    return kind, identity, entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--durations", default="5")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    refs = [r for r in protocol["references"] if r["role"] in protocol["protocol"]["gallery_roles"]]
    queries = [q for q in protocol["queries"] if q["evaluate"]]
    tasks = [("references", r["reference_id"], r["path"], None, None, str(args.cache)) for r in refs]
    for duration in map(float, args.durations.split(",")):
        for query in queries:
            q = shorten_query(query, duration)
            tasks.append(("queries", q["query_id"] + f"@{duration:g}s", q["path"],
                          q["start_s"], q["duration_s"], str(args.cache)))
    started = time.perf_counter()
    out = {"references": {}, "queries": {}, "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest()}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (kind, identity, entry) in enumerate(pool.map(prepare_one, tasks)):
            out[kind][identity] = entry
            if (i + 1) % 100 == 0:
                print(f"Prepared float5512 {i + 1}/{len(tasks)}", flush=True)
    out["prepare_wall_s"] = time.perf_counter() - started
    out["prepare_workers"] = args.workers
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
