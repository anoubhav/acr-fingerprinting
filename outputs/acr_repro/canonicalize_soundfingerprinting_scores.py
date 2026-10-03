"""Remove numerical tie splitting without changing native candidates or labels."""
import argparse
import json
from pathlib import Path


def canonicalize(path):
    result = json.loads(Path(path).read_text())
    if not result["method"].startswith("SoundFingerprinting"):
        raise ValueError("This precision rule applies to native library Confidence")
    for row in result["predictions"]:
        raw = row.get("raw_score", row["score"])
        row["raw_score"] = raw
        row["score"] = round(raw, 12)
    result["config"]["confidence_canonicalization_decimal_places"] = 12
    result["precision_provenance"] = {
        "operation": "Round native ResultEntry.Confidence to 12 decimal places for calibration and acceptance; preserve raw_score",
        "reason": "Mathematically identical one-fingerprint coverage can differ by about 1e-14 from floating-point subtraction; tied evidence must not be split by numerical jitter",
        "native_candidate_ranking_unchanged": True,
        "prediction_labels_unchanged": True,
    }
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    for path in parser.parse_args().results:
        canonicalize(path)
        print(path)
