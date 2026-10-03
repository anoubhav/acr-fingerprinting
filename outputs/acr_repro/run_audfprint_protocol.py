"""Run the pinned landmark baseline on the frozen protocol and query lengths."""
import argparse
import json
import hashlib
from pathlib import Path
from audfprint_baseline import AudfprintBaseline


def shorten_query(query, duration_s):
    q = dict(query)
    duration = min(duration_s, q["duration_s"])
    shift = (q["duration_s"] - duration) / 2
    q["start_s"] += shift
    q["duration_s"] = duration
    q["expected_reference_start_s"] += shift * q["expected_time_scale"]
    q["expected_reference_end_s"] = q["expected_reference_start_s"] + duration * q["expected_time_scale"]
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--durations", nargs="+", type=float, default=[5, 1, 2, 3, 10])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    manifest = json.loads(args.protocol.read_text())
    protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    gallery_roles = set(manifest["protocol"]["gallery_roles"])
    references = [r for r in manifest["references"] if r["role"] in gallery_roles]
    queries = [q for q in manifest["queries"] if q["evaluate"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = AudfprintBaseline(args.upstream, args.cache)
    baseline.fit(references, args.workers)
    for duration in args.durations:
        predictions = []
        for i, original in enumerate(queries):
            query = shorten_query(original, duration)
            result = baseline.query(query)
            result.update(role=query["role"], target_reference_id=query["reference_id"],
                          source_id=query["source_id"], duration_s=query["duration_s"],
                          expected_reference_start_s=query["expected_reference_start_s"],
                          correct_reference=result["reference_id"] == query["reference_id"])
            if "condition" in query:
                result["condition"] = query["condition"]
                result["speaker_id"] = query["speaker_id"]
            result["localized_correct"] = result["correct_reference"] and abs(
                result["offset_s"] - query["expected_reference_start_s"]) <= 2
            predictions.append(result)
            if (i + 1) % 100 == 0:
                print(f"audfprint duration={duration:g}s queried {i + 1}/{len(queries)}", flush=True)
        out = {"method": "audfprint", "duration_s_requested": duration,
               "config": baseline.config, "stats": baseline.stats,
               "protocol": manifest["protocol"], "protocol_sha256": protocol_sha256,
               "predictions": predictions}
        path = args.output_dir / f"audfprint_{duration:g}s.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
