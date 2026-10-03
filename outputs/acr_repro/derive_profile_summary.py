"""Recompute common-cohort timing from raw repeats in workspace or artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

METHODS = ("acr", "audfprint", "soundfingerprinting", "nmfp", "peaknet")
METRICS = ("encoder_ms", "matching_ms", "pipeline_ms")


def derive(profiles: Path, root: Path):
    rows = []
    for method in METHODS:
        path = profiles / method / "profile.json"
        data = json.loads(path.read_text())
        values = {}
        if method == "acr":
            records = data["records"]
            if data["repeats_per_query"] != 5 or data["warmups_per_query"] != 2:
                raise ValueError("ACR profile does not use the declared 2/5 warmup/repeat policy")
            for key, target in [("encoder_wall_ms", "encoder_ms"),
                                ("matcher_wall_ms", "matching_ms"),
                                ("end_to_end_wall_ms", "pipeline_ms")]:
                values[target] = [r["per_query_medians"][key] for r in records]
            digest = data["common_input_sha256"]
        elif method in ("audfprint", "soundfingerprinting"):
            if data["repeats"] != 5 or data["warmups"] != 2:
                raise ValueError(f"{method}: unexpected warmup/repeat policy")
            for key, target in [("encoder_s", "encoder_ms"),
                                ("matching_s", "matching_ms"),
                                ("pipeline_s", "pipeline_ms")]:
                if any(len(r[key]) != 5 for r in data["queries"]):
                    raise ValueError(f"{method}: a query lacks five measured repeats")
                values[target] = [1000 * np.median(r[key]) for r in data["queries"]]
            digest = data["common_input_sha256"]
        else:
            groups = {}
            for record in data["timing"]:
                if record["variant"] == "ivf":
                    groups.setdefault(record["query_id"], []).append(record)
            if any(len(records) != 5 for records in groups.values()):
                raise ValueError(f"{method}: a query lacks five measured repeats")
            for key, target in [("encoder_s", "encoder_ms"),
                                ("matching_s", "matching_ms"),
                                ("pipeline_s", "pipeline_ms")]:
                values[target] = [1000 * np.median([r[key] for r in records])
                                  for records in groups.values()]
            digest = data["inputs"]["audio_sha256"]
        if any(len(v) != 50 or not np.isfinite(v).all() for v in values.values()):
            raise ValueError(f"{method}: expected 50 finite per-query medians")
        summary = {key: {"p50": float(np.quantile(v, .5)),
                         "p95": float(np.quantile(v, .95))}
                   for key, v in values.items()}
        try:
            profile_path = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            profile_path = str(path.resolve())
        rows.append({"method": method, "input_sha256": digest,
                     "profile_path": profile_path, "n_queries": 50, "audio_s": 5,
                     "aggregation": "linear p50/p95 over 50 per-query medians of five timed repeats after two warmups",
                     **summary})
    if len({r["input_sha256"] for r in rows}) != 1:
        raise ValueError("Timing methods do not share the same input cohort hash")
    return rows


def check(rows, recorded):
    old = {r["method"]: r for r in recorded}
    if set(old) != set(METHODS):
        raise ValueError("Recorded timing method set differs")
    for row in rows:
        saved = old[row["method"]]
        if any(row[key] != saved[key] for key in ("input_sha256", "n_queries", "audio_s")):
            raise ValueError(f"{row['method']}: recorded cohort differs")
        for metric in METRICS:
            for quantile in ("p50", "p95"):
                if not np.isclose(row[metric][quantile], saved[metric][quantile], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"{row['method']}.{metric}.{quantile}: saved value differs")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path,
                        default=(root / "results/profiling" if (root / "results/profiling").is_dir()
                                 else root / "work/profiling"))
    parser.add_argument("--output", type=Path, default=root / "work/paper_timing_recomputed.json")
    parser.add_argument("--check", type=Path, help="Compare metrics/cohort against an existing paper_timing.json")
    args = parser.parse_args()
    rows = derive(args.profiles, root)
    if args.check:
        check(rows, json.loads(args.check.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row["method"], *(f'{row[k]["p50"]:.3f}/{row[k]["p95"]:.3f}' for k in METRICS))
    if args.check:
        print("Recorded timing values match the raw-repeat derivation.")


if __name__ == "__main__":
    main()
