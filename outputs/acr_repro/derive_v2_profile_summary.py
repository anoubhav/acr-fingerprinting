"""Derive current paper timings from recorded per-query repetitions, without audio."""
from pathlib import Path
import argparse
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def derive(profiles):
    profiles = Path(profiles)
    load = lambda name: json.loads((profiles / name / "profile.json").read_text())
    rows = []
    fields_ms = {"encoder": ("encoder_wall_ms", 1), "matching": ("matcher_wall_ms", 1),
                 "pipeline": ("end_to_end_wall_ms", 1)}
    fields_s = {"encoder": ("encoder_s", 1000), "matching": ("matching_s", 1000),
                "pipeline": ("pipeline_s", 1000)}

    def make(key, label, groups, fields, sha, frames=None, encoded_frames=None):
        if len(groups) != 50 or any(len(v) != 5 for v in groups.values()):
            raise ValueError(f"Expected 50 queries with five timed repeats: {key}")
        metrics = {}
        for destination, (source, scale) in fields.items():
            values = [float(np.median([r[source] for r in repeats])) * scale
                      for repeats in groups.values()]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"Invalid timing: {key}/{source}")
            metrics[destination] = {"p50": float(np.quantile(values, .5)),
                                    "p95": float(np.quantile(values, .95))}
        rows.append({"key": key, "label": label, "audio_sha256": sha,
                     "n_queries": 50, "repeats": 5,
                     "aggregation": "Quantiles over50 per-query raw-repeat medians.",
                     "query_frames": frames, "encoded_query_frames": encoded_frames,
                     **metrics})

    d = load("acr_compact")
    groups = {r["query_id"]: r["repetitions"] for r in d["records"]}
    encoded = [int(np.median([v["encoder_fingerprints"] for v in r["repetitions"]]))
               for r in d["records"]]
    searched = [int(np.median([v["searched_fingerprints"] for v in r["repetitions"]]))
                for r in d["records"]]
    make("acr_compact", r"ACR compact, IVF512/probe16, $k=8,q=4$", groups, fields_ms,
         d["common_input_sha256"], float(np.median(searched)), float(np.median(encoded)))
    rows[-1]["query_frames_range"] = [min(searched), max(searched)]
    rows[-1]["encoded_query_frames_range"] = [min(encoded), max(encoded)]

    for key, label in [("audfprint", "Audfprint candidate floor1"),
                       ("soundfingerprinting", "Public MinHash native votes4")]:
        d = load(key)
        groups = {r["query_id"]: [{k: r[k][i] for k in ("encoder_s", "matching_s", "pipeline_s")}
                                 for i in range(5)] for r in d["queries"]}
        make(key, label, groups, fields_s, d["common_input_sha256"])

    for directory, key, label in [
            ("nmfp", "nmfp_native", "NMFP native, IVF1024/probe16"),
            ("peaknet_corrected", "peak_native", "PeakNetFP native, IVF1024/probe16")]:
        d = load(directory)
        groups = {}
        for r in d["timing"]:
            if r["variant"] == "ivf":
                groups.setdefault(r["query_id"], []).append(r)
        make(key, label, groups, fields_s, d["inputs"]["audio_sha256"], 9, 9)

    d = load("nmfp_density")
    for variant in d["variants"]:
        groups = {}
        for r in variant["timing"]:
            groups.setdefault(r["query_id"], []).append(r)
        if any(r["fingerprints"] != variant["query_frames"]
               for repeats in groups.values() for r in repeats):
            raise ValueError("Sparse model executed the wrong frame count")
        factor = variant["query_factor"]
        make(f"nmfp_r{factor}", rf"NMFP $r=q={factor}$, exact FlatIP", groups, fields_s,
             d["audio_sha256"], variant["query_frames"], variant["query_frames"])
    if len({r["audio_sha256"] for r in rows}) != 1:
        raise ValueError("Timing cohorts have different PCM hashes")
    return rows


def check(rows, expected):
    if [r["key"] for r in rows] != [r["key"] for r in expected]:
        raise ValueError("Timing row order/keys differ")
    for actual, recorded in zip(rows, expected):
        for field in ("audio_sha256", "n_queries", "repeats", "query_frames", "encoded_query_frames"):
            if actual[field] != recorded[field]:
                raise ValueError((actual["key"], field))
        for metric in ("encoder", "matching", "pipeline"):
            for quantile in ("p50", "p95"):
                if not math.isclose(actual[metric][quantile], recorded[metric][quantile],
                                    abs_tol=1e-9, rel_tol=1e-12):
                    raise ValueError((actual["key"], metric, quantile))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=ROOT / "results/profiling")
    parser.add_argument("--output", type=Path, default=ROOT / "work/v2_timing_recomputed.json")
    parser.add_argument("--check", type=Path,
                        help="Verify cohort and timing cells against a recorded summary.")
    args = parser.parse_args()
    rows = derive(args.profiles)
    if args.check:
        check(rows, json.loads(args.check.read_text()))
        print("All seven current timing rows match raw-repeat derivation and shared PCM cohort.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        values = "; ".join(f"{m} {row[m]['p50']:.3f}/{row[m]['p95']:.3f} ms"
                           for m in ("encoder", "matching", "pipeline"))
        print(f"{row['key']}: {values}")


if __name__ == "__main__":
    main()
