"""Audit strong source-ID false accepts for actual aligned waveform overlap.

This is a label-quality sensitivity audit. It does not modify predictions,
thresholds, labels, or the primary protocol. High waveform correlation is
evidence for shared audio, not a general proof of composition identity.
"""
import argparse
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.signal import correlate
from acr_fp import load_audio


@lru_cache(maxsize=4)
def decoded(path):
    return load_audio(path, 8000)[0]


def aligned_correlation(query, reference, offset_s, radius_s=.25):
    sr = 8000
    source = decoded(query["path"])
    begin = round(query["start_s"] * sr)
    q = source[begin:begin + round(query["duration_s"] * sr)].astype(float)
    r = decoded(reference["path"])
    # Require full crop support; partial/negative-offset matches stay unverified.
    lower = max(0, round((offset_s - radius_s) * sr))
    upper = min(len(r), round((offset_s + radius_s) * sr) + len(q))
    window = r[lower:upper].astype(float)
    if len(window) < len(q) or len(q) < 8000:
        return {"waveform_audit_status": "insufficient_full_crop_support"}
    centered = q - q.mean()
    norm_q = np.linalg.norm(centered)
    if norm_q < 1e-9:
        return {"waveform_audit_status": "silent_query"}
    numerator = correlate(window, centered, mode="valid", method="fft")
    cs = np.r_[0, np.cumsum(window)]
    cs2 = np.r_[0, np.cumsum(window ** 2)]
    n = len(q)
    sums = cs[n:] - cs[:-n]
    sums2 = cs2[n:] - cs2[:-n]
    norm_r = np.sqrt(np.maximum(sums2 - sums ** 2 / n, 0))
    pearson = np.divide(numerator, norm_q * norm_r,
                        out=np.zeros_like(numerator), where=norm_r > 1e-9)
    peak = int(np.argmax(np.abs(pearson)))
    value = float(pearson[peak])
    return {"waveform_audit_status": "measured", "signed_waveform_correlation": value,
            "absolute_waveform_correlation": abs(value),
            "refined_reference_offset_s": (lower + peak) / sr,
            "audited_crop_duration_s": len(q) / sr,
            "strong_waveform_overlap_0p95": abs(value) >= .95}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    result = json.loads(args.results.read_text())
    refs = {r["reference_id"]: r for r in protocol["references"]}
    queries = {q["query_id"]: q for q in protocol["queries"]}
    candidates = [r for r in result["predictions"] if r["role"] == "test_unknown"
                  and r["reference_id"] is not None
                  and queries[r["query_id"]].get("negative_population") == "additional_clean_FMA"]
    candidates.sort(key=lambda x: -x["score"])
    audited = []
    for prediction in candidates[:args.limit]:
        query = dict(queries[prediction["query_id"]])
        if query["duration_s"] != prediction["duration_s"]:
            raise ValueError("Audit supports current 5 s extra-negative crops only")
        ref = refs[prediction["reference_id"]]
        row = {"query_id": prediction["query_id"], "unknown_source_id": query["source_id"],
               "predicted_reference_id": prediction["reference_id"], "score": prediction["score"],
               "landmark_offset_s": prediction["offset_s"],
               "unknown_artist_id": query.get("artist_id"),
               "reference_artist_name": ref.get("artist_name"), "reference_track_title": ref.get("track_title")}
        row.update(aligned_correlation(query, ref, prediction["offset_s"]))
        audited.append(row)
    out = {"method": result["method"], "candidate_count": len(candidates),
           "top_score_audited_count": len(audited), "offset_refinement_radius_s": .25,
           "waveform_overlap_rule": "Absolute Pearson >=0.95 for the complete 5s crop, allowing +/-0.25s offset refinement",
           "prediction_labels_unchanged": True, "audited": audited}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "audited"}, indent=2))
    print("strong waveform overlaps", sum(r.get("strong_waveform_overlap_0p95", False) for r in audited))


if __name__ == "__main__":
    main()
