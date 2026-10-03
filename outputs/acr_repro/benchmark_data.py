"""Parse the official Pexeso audio benchmark without changing its annotations.

The adapted retrieval task uses a centered crop from each annotated query chunk.
This is not the toolkit's native multiple-match segment-detection task.  Intervals
are integer annotations; expected source times are therefore approximate. Audio
stays in work/, and manifests preserve upstream track attribution metadata.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import pathlib
import stat
import subprocess
import zipfile
from collections import Counter
import soundfile as sf


def sha256(path):
    with pathlib.Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_extract(archive, destination):
    """Reject traversal, symlink and encrypted ZIP members before extraction."""
    destination = pathlib.Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = pathlib.PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                raise ValueError(f"Unsupported archive member: {member.filename}")
        z.extractall(destination)


def decoded_audio_info(path):
    # All evaluated methods use FFmpeg because the original toolkit generated
    # its annotations/perturbations with FFmpeg. Container metadata and MP3
    # SoundFile.info().frames can overestimate actually decoded audio.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                          "-ac", "1", "-ar", "8000", "pipe:1"],
                         check=True, capture_output=True).stdout
    return {"duration_s": len(raw) / 4 / 8000, "sample_rate": 8000,
            "channels": 1, "decoded_frames": len(raw) // 4,
            "duration_decoder": "ffmpeg_decoded_8000hz"}


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate,channels",
         "-of", "json", str(path)], check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    decoded = decoded_audio_info(path)
    return dict(decoded, container_duration_s=float(info["format"]["duration"]))


def load_pexeso(root, crop_duration_s=10.0, probe_audio=False):
    root = pathlib.Path(root).resolve()
    if not (root / "annotations.csv").exists():
        children = list(root.glob("*/annotations.csv"))
        if len(children) != 1:
            raise FileNotFoundError(f"Cannot find annotations.csv under {root}")
        root = children[0].parent
    with (root / "fma_tracks.csv").open(newline="", encoding="utf-8") as f:
        metadata = {r["track_id"]: r for r in csv.DictReader(f)}
    with (root / "annotations.csv").open(newline="", encoding="utf-8") as f:
        annotations = list(csv.DictReader(f))
    refs = {}
    for path in sorted((root / "references").rglob("*.mp3")):
        rid = path.stem
        if rid in refs:
            raise ValueError(f"Duplicate reference ID {rid}")
        m = metadata[rid]
        refs[rid] = {"reference_id": rid, "path": str(path), "source_id": rid,
                     "artist_name": m["artist_name"], "track_title": m["track_title"],
                     "track_url": m["track_url"], "license_url": m["license_url"],
                     "license_title": m["license_title"],
                     "license_image_file_large": m["license_image_file_large"],
                     "license_review_required": "Sound_Recording_Common_Law" in m["license_url"],
                     "metadata_duration_s": float(m["track_duration"])}
    query_paths = {p.stem: p for p in sorted((root / "queries").glob("*.mp3"))}
    if set(refs) != {a["reference_id"] for a in annotations}:
        raise ValueError("Gallery IDs do not equal annotation reference IDs")
    if set(query_paths) != {a["query_id"] for a in annotations}:
        raise ValueError("Query files do not equal annotation query IDs")
    if probe_audio:
        paths = [r["path"] for r in refs.values()] + list(query_paths.values())
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            probes = dict(zip(map(str, paths), executor.map(probe, paths)))
        for ref in refs.values():
            ref.update(probes[ref["path"]])
    else:
        # Crop safety is mandatory even when the optional reference audit is off.
        probes = {str(path): decoded_audio_info(path) for path in query_paths.values()}
    queries = []
    for i, a in enumerate(annotations):
        rb, re, qb, qe = [float(a[k]) for k in
                          ("reference_begin", "reference_end", "query_begin", "query_end")]
        annotated_qe = qe
        file_duration = probes[str(query_paths[a["query_id"]])]["duration_s"]
        qe = min(qe, file_duration)
        if min(rb, qb) < 0 or re <= rb or qe <= qb:
            raise ValueError(f"Invalid annotation at row {i + 2}")
        tempo = float(a["tempo"] or 100) / 100
        dur = min(float(crop_duration_s), qe - qb)
        start = qb + (qe - qb - dur) / 2
        reference_start = rb + (start - qb) * tempo
        other = []
        for j, b in enumerate(annotations):
            if i == j or b["query_id"] != a["query_id"]:
                continue
            overlap = min(start + dur, float(b["query_end"])) - max(start, float(b["query_begin"]))
            if overlap > 0:
                other.append({"reference_id": b["reference_id"], "overlap_duration_s": overlap})
        q = {"query_id": f"{root.name}:chunk{i:04d}", "source_query_id": a["query_id"],
             "annotation_index": i, "path": str(query_paths[a["query_id"]]),
             "start_s": start, "duration_s": dur,
             "reference_id": a["reference_id"], "source_id": a["reference_id"],
             "expected_reference_start_s": reference_start,
             "expected_reference_end_s": reference_start + dur * tempo,
             "expected_time_scale": tempo, "annotation": a,
             "overlapping_other_annotations": other,
             "localization_tolerance_s": 2.0,
             "source_file_duration_s": file_duration,
             "annotation_query_end_clamped": qe != annotated_qe,
             "annotation_query_end_clamp_s": annotated_qe - qe,
             "effective_annotation_query_end_s": qe}
        if probe_audio:
            if start + dur > file_duration + 1e-6:
                raise ValueError(f"Query annotation exceeds audio duration: {q['query_id']}")
            if re > refs[a["reference_id"]]["duration_s"] + 1e-6:
                raise ValueError(f"Reference annotation exceeds audio duration: {q['query_id']}")
        queries.append(q)
    summary = {"dataset": root.name, "root": str(root), "reference_count": len(refs),
               "query_file_count": len(query_paths), "annotated_chunk_count": len(queries),
               "crop_duration_s_requested": crop_duration_s,
               "crop_duration_s_min": min(q["duration_s"] for q in queries),
               "crops_with_other_annotated_sources": sum(bool(q["overlapping_other_annotations"]) for q in queries),
               "annotation_query_end_clamped_count": sum(q["annotation_query_end_clamped"] for q in queries),
               "annotation_query_end_clamp_max_s": max(q["annotation_query_end_clamp_s"] for q in queries),
               "reference_annotation_sha256": sha256(root / "annotations.csv"),
               "fma_metadata_sha256": sha256(root / "fma_tracks.csv"),
               "license_counts": dict(Counter(r["license_url"] for r in refs.values())),
               "license_review_reference_ids": [r["reference_id"] for r in refs.values() if r["license_review_required"]]}
    return {"summary": summary, "references": list(refs.values()), "queries": queries}


def materialize_crop(query, destination, sample_rate=11025):
    """FFmpeg accurate post-input-seek PCM crops; cache under caller's work dir."""
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", query["path"],
                    "-ss", str(query["start_s"]), "-t", str(query["duration_s"]),
                    "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
                    str(destination)], check=True)
    count = round(query["duration_s"] * sample_rate)
    if abs(sf.info(destination).frames - count) > 2:
        raise ValueError(f"Crop differs from expected decoded duration: {query['query_id']}")
    return destination


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--crop-duration", type=float, default=10.0)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    manifest = load_pexeso(args.root, args.crop_duration, args.probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
