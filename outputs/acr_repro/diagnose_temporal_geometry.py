"""Descriptive continuity checks on fixed calibration sources, not test tuning.

Random cross-content distances are not nearest-competitor margins and do not
establish recognition specificity. They provide a clearly labeled scale only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from acr_fp import Fingerprinter
from run_ablations import ref_path


def quantiles(values):
    x = np.asarray(values, dtype=float)
    return {"pairs": len(x), "p10": float(np.percentile(x, 10)),
            "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sources", type=int, default=50)
    p.add_argument("--pairs-per-source", type=int, default=1024)
    a = p.parse_args()
    model = Fingerprinter.load(a.model)
    m = json.loads(a.protocol.read_text())
    refs = sorted([r for r in m["references"] if r["role"] == "calibration_known"],
                  key=lambda r: r["reference_id"])[:a.sources]
    model.assert_disjoint(reference_ids=[r["reference_id"] for r in refs])
    rng = np.random.default_rng(20261002)
    shifts = (1, 2, 4, 6, 8, 16, 32, 64)
    values = {str(r): [] for r in shifts}
    sampled, source_rows = [], []
    for reference in refs:
        with np.load(ref_path(a.cache, reference, model.config)) as z:
            raw, times = z["features"], z["times"]
            duration = float(z["duration_s"])
        fp = model.transform(raw)
        frames = np.rint((times-model.config.first_center_seconds)
                         / (model.config.hop_length/model.config.sample_rate)).astype(np.int64)
        lookup = {int(frame): i for i, frame in enumerate(frames)}
        eligible = np.flatnonzero((times > 1) & (times < duration-1))
        if not len(eligible):
            continue
        anchors = rng.choice(eligible, min(a.pairs_per_source, len(eligible)), replace=False)
        sampled.append(fp[anchors].copy())
        source_rows.append({"reference_id": reference["reference_id"], "anchor_frames": len(anchors)})
        for r in shifts:
            pairs = [(i, lookup.get(int(frames[i]+r))) for i in anchors]
            pairs = [(i, j) for i, j in pairs if j is not None and times[j] < duration-1]
            if pairs:
                left, right = np.asarray(pairs).T
                values[str(r)].extend(np.linalg.norm(fp[left]-fp[right], axis=1).tolist())
    if len(sampled) < 2:
        raise ValueError("Need at least two calibration sources")
    cross = []
    # Cyclic different-content pairing is fixed and samples no test labels.
    for i, fp in enumerate(sampled):
        other = sampled[(i+1) % len(sampled)]
        choices = rng.integers(0, len(other), len(fp))
        cross.extend(np.linalg.norm(fp-other[choices], axis=1).tolist())
    rows = [{"shift_frames": r, "shift_seconds": r*model.config.hop_length/model.config.sample_rate,
             **quantiles(values[str(r)])} for r in shifts if values[str(r)]]
    result = {"diagnostic": "ACR-PCA32 temporal continuity on held-out calibration-known sources",
        "sources": source_rows, "seed": 20261002, "distance": "Euclidean L2, not squared L2",
        "same_content_offsets": rows, "random_cross_content": quantiles(cross),
        "interpretation_limit": "Random cross-content distances are not closest-wrong-content margins. This descriptive diagnostic does not establish specificity or imply a recognition guarantee.",
        "uses_test_queries_or_test_outcomes": False,
        "protocol_sha256": hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
        "pca_model_sha256": hashlib.sha256(a.model.read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "sources"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
