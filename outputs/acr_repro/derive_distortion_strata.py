"""Read-only descriptive PEX strata from frozen five-second predictions.

No audio is decoded, no model is fitted, and no primary result is modified.
Metadata-only mutually exclusive groups are declared below. This diagnostic is
exploratory, after the primary analysis; it is not an isolated attack experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261003
REPLICATES = 2000
GROUPS = ("pitch_altered", "tempo_only", "neither")
METRICS = ("candidate_correct", "candidate_localized", "accepted_correct",
           "accepted_localized", "accepted_target_mismatch", "accepted_any")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def base_id(row):
    return row.get("base_query_id", row["query_id"].split("@")[0])


def classify(annotation):
    # PEX README schema: tempo is integer percent, pitch is integer cents.
    # Generator rounds encoded modification values. Grouping therefore uses
    # annotations, not an inference that underlying audio is exactly unmodified.
    pitch = float(annotation.get("pitch") or 0)
    tempo = float(annotation.get("tempo") or 100)
    if not np.isfinite([pitch, tempo]).all() or tempo <= 0:
        raise ValueError("Invalid PEX pitch/tempo annotation")
    if pitch != 0:
        return "pitch_altered"
    if tempo != 100:
        return "tempo_only"
    return "neither"


def select_original_gate(rows):
    # Original 1% policy with exactly 74 calibration unknowns permits zero
    # empirical events, so freeze above their largest score. It is global,
    # never a different gate per descriptive stratum.
    scores = np.array([float(r["score"]) for r in rows
                       if r["role"] == "calibration_unknown"])
    if len(scores) != 74 or not np.isfinite(scores).all():
        raise ValueError("Expected all 74 finite calibration-unknown scores")
    return float(np.nextafter(scores.max(), np.inf))


def bootstrap_weights(records):
    sources = sorted({r["source_id"] for r in records})
    montages = sorted({r["source_query_id"] for r in records})
    si = np.array([sources.index(r["source_id"]) for r in records])
    mi = np.array([montages.index(r["source_query_id"]) for r in records])
    rng = np.random.default_rng(SEED)
    ws = rng.multinomial(len(sources), np.full(len(sources), 1 / len(sources)), REPLICATES)[:, si]
    wm = rng.multinomial(len(montages), np.full(len(montages), 1 / len(montages)), REPLICATES)[:, mi]
    return {"source": ws, "montage": wm, "source_montage_crossed": ws * wm}


def metric_summaries(records, weights):
    y = np.asarray([[r[m] for m in METRICS] for r in records], dtype=float)
    n = len(records)
    summaries = {}
    for j, metric in enumerate(METRICS):
        item = {"count": int(y[:, j].sum()), "n": n,
                "rate": float(y[:, j].mean()), "bootstrap_ci95": {}}
        for name, w in weights.items():
            den = w.sum(axis=1)
            valid = den > 0
            values = (w[valid] @ y[:, j]) / den[valid]
            item["bootstrap_ci95"][name] = [float(x) for x in np.quantile(values, [.025, .975])]
        summaries[metric] = item
    return summaries


def paired(first, second, weights):
    # second-minus-first, with identical resampling weights for both methods.
    assert [r["query_id"] for r in first] == [r["query_id"] for r in second]
    out = {"direction": "sparse factor8 minus dense factor1",
           "n": len(first), "candidate_id_changes": sum(a["reference_id"] != b["reference_id"]
                                                       for a, b in zip(first, second)), "metrics": {}}
    for metric in METRICS:
        a = np.array([r[metric] for r in first], dtype=float)
        b = np.array([r[metric] for r in second], dtype=float)
        delta = b - a
        item = {"dense_count": int(a.sum()), "sparse_count": int(b.sum()),
                "dense_only": int(((a == 1) & (b == 0)).sum()),
                "sparse_only": int(((a == 0) & (b == 1)).sum()),
                "difference_pp": float(100 * delta.mean()), "bootstrap_ci95_pp": {}}
        for name, w in weights.items():
            den = w.sum(axis=1)
            valid = den > 0
            values = 100 * (w[valid] @ delta) / den[valid]
            item["bootstrap_ci95_pp"][name] = [float(x) for x in np.quantile(values, [.025, .975])]
        out["metrics"][metric] = item
    return out


def derive(args):
    protocol = read(args.protocol)
    lookup = {q["query_id"]: q for q in protocol["queries"] if q.get("evaluate", True)}
    known = sorted((q for q in lookup.values() if q["role"] == "test_known"), key=lambda q: q["query_id"])
    if len(known) != 543:
        raise ValueError("Full frozen test-known cohort must contain 543 queries")
    primary = read(args.paper_numbers)["small"]
    specs = {row["label"]: {"path": row["prediction_path"], "frozen_summary": row} for row in primary}
    specs["ACR exact k1"] = {"path": "acr_medium/acr_d1_5s.json"}
    if len(specs) != 8:
        raise ValueError("Expected seven main-table configurations plus exact dense ACR")
    paths = [args.protocol, args.paper_numbers, Path(__file__)]
    methods = {}
    query_records = []
    for q in known:
        annotation = q["annotation"]
        query_records.append({"query_id": q["query_id"], "source_id": q["source_id"],
                              "source_query_id": q["source_query_id"], "group": classify(annotation),
                              "pitch_cents": float(annotation.get("pitch") or 0),
                              "tempo_percent": float(annotation.get("tempo") or 100),
                              "noise_present": bool(annotation.get("noise_type")),
                              "noise_snr_db": float(annotation["noise_snr"]) if annotation.get("noise_snr") else None,
                              "echo_present": bool(annotation.get("echo_delay")),
                              "reverb_present": bool(annotation.get("reverb")),
                              "filter_present": bool(annotation.get("high_pass") or annotation.get("low_pass")),
                              "overlapping_other_annotation": bool(q.get("overlapping_other_annotations"))})
    qids = {q["query_id"] for q in known}
    for label, spec in specs.items():
        path = args.benchmarks / spec["path"]
        paths.append(path)
        result = read(path)
        predictions = result["predictions"]
        keyed = {base_id(r): r for r in predictions}
        if len(keyed) != len(predictions) or len(predictions) != 835:
            raise ValueError(f"{label}: expected complete unique 835-row original cohort")
        if {k for k, r in keyed.items() if r["role"] == "test_known"} != qids:
            raise ValueError(f"{label}: known cohort differs")
        for k, r in keyed.items():
            q = lookup[k]
            if r["role"] != q["role"] or r["source_id"] != q["source_id"] or r["target_reference_id"] != q["reference_id"]:
                raise ValueError(f"{label}: split or target mismatch")
            if not np.isfinite(float(r["score"])):
                raise ValueError(f"{label}: nonfinite score")
        gate = select_original_gate(predictions)
        if "frozen_summary" in spec and gate != spec["frozen_summary"]["threshold"]:
            raise ValueError(f"{label}: reconstructed gate differs from frozen primary gate")
        enriched = []
        for qrec in query_records:
            r = keyed[qrec["query_id"]]
            if abs(float(r["duration_s"]) - 5) > 1e-6:
                raise ValueError(f"{label}: five-second crop changed")
            correct = r["reference_id"] == r["target_reference_id"]
            if bool(r["correct_reference"]) != correct:
                raise ValueError(f"{label}: stored candidate correctness inconsistent with IDs")
            accepted = r["reference_id"] is not None and float(r["score"]) >= gate
            enriched.append({**qrec, "reference_id": r["reference_id"], "score": float(r["score"]),
                             "candidate_correct": bool(correct), "candidate_localized": bool(r["localized_correct"]),
                             "accepted_correct": bool(accepted and correct),
                             "accepted_localized": bool(accepted and r["localized_correct"]),
                             "accepted_target_mismatch": bool(accepted and not correct), "accepted_any": bool(accepted)})
        if "frozen_summary" in spec:
            expected = spec["frozen_summary"]["test_known"]["all"]
            for key, expected_key in (("candidate_correct", "correct_reference"), ("candidate_localized", "localized_correct"),
                                      ("accepted_correct", "accepted_correct"), ("accepted_localized", "accepted_localized"),
                                      ("accepted_target_mismatch", "accepted_wrong")):
                if sum(r[key] for r in enriched) != expected[expected_key]["successes"]:
                    raise ValueError(f"{label}: full-cohort count differs from primary summary: {key}")
        methods[label] = {"prediction_path": spec["path"], "frozen_global_gate": gate,
                          "calibration_unknown_n": 74, "rows": enriched}
    # Confirm the dense/sparse exact comparison uses identical encoded queries
    # and agreed original cohort. Canonical d8 must equal the density-matrix d8.
    matrix_sparse_path = args.benchmarks / "acr_medium/acr_d8_5s.json"
    paths.append(matrix_sparse_path)
    matrix_sparse = {base_id(r): r for r in read(matrix_sparse_path)["predictions"]}
    canonical_sparse = {base_id(r): r for r in read(args.benchmarks / specs["ACR exact k8"]["path"])["predictions"]}
    for k in matrix_sparse:
        for field in ("reference_id", "score", "offset_s", "correct_reference", "localized_correct", "duration_s"):
            if matrix_sparse[k][field] != canonical_sparse[k][field]:
                raise ValueError("Canonical exact sparse row differs from original density sweep")
    schema_files = {}
    if args.upstream:
        for name in ("README.md", "generate_query_audios.py", "util.py", "fields.py"):
            path = args.upstream / name
            schema_files[name] = sha256(path)
            paths.append(path)
        readme = (args.upstream / "README.md").read_text()
        if "Tempo in percent of the original" not in readme or "Pitch change in cents" not in readme:
            raise ValueError("Upstream annotation schema differs from reviewed definitions")
    group_results = {}
    for group in ("all", *GROUPS):
        records = [r for r in query_records if group == "all" or r["group"] == group]
        weights = bootstrap_weights(records)
        rows_by_method = {label: [r for r in d["rows"] if group == "all" or r["group"] == group]
                          for label, d in methods.items()}
        group_results[group] = {"n_queries": len(records), "n_source_ids": len({r["source_id"] for r in records}),
                               "n_montage_ids": len({r["source_query_id"] for r in records}),
                               "other_annotation_counts": {key: sum(r[key] for r in records) for key in
                                     ("noise_present", "echo_present", "reverb_present", "filter_present", "overlapping_other_annotation")},
                               "methods": {label: metric_summaries(rows, weights) for label, rows in rows_by_method.items()},
                               "dense_sparse_paired": paired(rows_by_method["ACR exact k1"], rows_by_method["ACR exact k8"], weights)}
    if sum(group_results[g]["n_queries"] for g in GROUPS) != 543:
        raise ValueError("Exclusive groups fail to partition full known cohort")
    for label in methods:
        for metric in METRICS:
            if sum(group_results[g]["methods"][label][metric]["count"] for g in GROUPS) != group_results["all"]["methods"][label][metric]["count"]:
                raise ValueError("Group counts fail to sum to canonical full cohort")
    return {"schema_version": 1,
            "scope": "Exploratory descriptive annotation strata after primary analysis; full 543-known cohort retained",
            "group_definitions": {"pitch_altered": "annotated pitch cents != 0, regardless of tempo",
                                  "tempo_only": "annotated pitch cents == 0 and tempo percent != 100",
                                  "neither": "annotated pitch cents == 0 and tempo percent == 100"},
            "interpretation": "Other degradations and mixed content may co-occur. Groups do not isolate causal distortion effects; no per-stratum tuning or exclusions.",
            "field_schema": {"primary_url": "https://github.com/Pexeso/audio-fingerprinting-benchmark-toolkit/blob/25ae1edd7e3252ef1b64fd0c90d5f50c45b5824b/README.md",
                             "revision": "25ae1edd7e3252ef1b64fd0c90d5f50c45b5824b", "pitch": "integer cents", "tempo": "integer percent",
                             "rounding": "generator rounds pitch_scale_to_cents and tempo_scale_to_per_cent; annotations do not prove perfectly unchanged audio",
                             "reviewed_upstream_file_sha256": schema_files},
            "calibration": "Original global empirical 1% gates reconstructed using 74 calibration unknowns, compared exactly with frozen primary gates; no model/configuration/gate changes",
            "uncertainty": {"seed": SEED, "replicates": REPLICATES, "interval": "percentile 95%",
                            "resampling": "source-only, montage-only, and independent source/montage multinomial product weights; identical weights across methods and paired dense/sparse differences",
                            "limit": "Finite clustered empirical sensitivities, not population guarantees; annotations and related sources may remain dependent."},
            "provenance_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in paths},
            "methods": {label: {k: v for k, v in d.items() if k != "rows"} for label, d in methods.items()},
            "groups": group_results, "query_assignments": query_records,
            "per_query_outcomes": {label: d["rows"] for label, d in methods.items()},
            "validation": "PASS: full known coverage, roles/targets/durations, all 7 canonical gates and full-cohort counts, matrix-sparse row parity, exclusive group completeness"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "work/benchmarks/hard_medium_protocol.json")
    p.add_argument("--benchmarks", type=Path, default=ROOT / "work/benchmarks")
    p.add_argument("--paper-numbers", type=Path, default=ROOT / "work/paper_numbers_v2.json")
    p.add_argument("--upstream", type=Path, default=ROOT / "work/benchmarks/pexeso")
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("distortion_strata.json"))
    args = p.parse_args()
    summary = derive(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary["validation"])
    for group in GROUPS:
        g = summary["groups"][group]
        print(group, g["n_queries"], "queries", g["n_source_ids"], "sources", g["n_montage_ids"], "montages")
        for label, outcomes in g["methods"].items():
            print(" ", label, "candidate", outcomes["candidate_correct"]["count"], "accepted-correct",
                  outcomes["accepted_correct"]["count"], "accepted-target-mismatch", outcomes["accepted_target_mismatch"]["count"])


if __name__ == "__main__":
    main()
