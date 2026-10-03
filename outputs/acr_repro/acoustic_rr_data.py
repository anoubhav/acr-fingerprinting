"""Acquire and adapt the public SD-RR v1.0 real acoustic rerecording release.

Publisher metadata are retained with their CC BY-SA 4.0 provenance. Audio stays
under work/; each track has its own license. This adapter does not redistribute
media or change the original closed-set labels, alignment or quality-control
values. The separate open-set/session protocol is an explicitly new adaptation.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter
import numpy as np
from benchmark_data import decoded_audio_info

RECORD = "22169646"
DOI = "10.5281/zenodo.22169646"
UPSTREAM_REPOSITORY = "https://github.com/JihengLi/POLARIS"
UPSTREAM_REVISION = "fe8943897d3c723060d31485d83d042c9d6a3057"
FILES = {
    "CITATION.cff": "45e3f61ac8366ada7b133241b9762aab75b3eda0325038d206598c3d3e6c6940",
    "DATASET_CARD.md": "daf7831de1e351c74f73ad3d3e1614fc4ea9732696caaad2eb78d80b79b7d041",
    "LICENSES.md": "e29aae134ba2ce507a4602a8d642035d4f099e8b33cd0083f506a10454b17c95",
    "README.md": "d7074fcdca9edf59237d6e73f769ba0d6ffc9be0a2cd4e64ae3bfd1a7ca5c1ad",
    "SHA256SUMS": "4fe052f16e7e57855a62493362c0a14c5db4efb2517f5b774b085d5edf5b5fb1",
    "attribution.csv": "ff9fa4d0bc270589342ca1de0c10fc1fd4c9c62e7e4f224c31c3b71b11f4d5f5",
    "dataset_config.json": "aec6eb31a9f09212f024489d1953af722f4fb529ff293bcb64206a30c1a9dd2b",
    "dataset_summary.json": "6a6e84b1930043e3582d57282edcb240837ff8577a380f4f7ef1bd0db43ae6a9",
    "exclusions.csv": "665d6db51ce4407a660378d9169e30433635fd10e3e16fef80206433d08819a6",
    "manifest.csv": "5e6c66cdfcf17aa98e085a0de9a545096f44183c65787608270644f82556e0e5",
    "reference_manifest.csv": "1843cf0c1d9aa658b5b1009f2eb21107c08f6bfd8f4ac75c552162eab78c1f27",
    "sdrr-v1.0-queries.zip": "48c27ce1dbc0244d8461dd8ab17256d17afb3345041614c7f94a2e527776dbb5",
    "sdrr-v1.0-references.zip": "e3c33f1ab71080dfbac5f850324888fd4edba3d3de55cb223da8713233d7efdb",
}


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def safe_relative(raw):
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ValueError(f"Unsafe release-relative path: {raw}")
    return Path(*path.parts)


def _download(file, destination):
    """Resume an incomplete public asset, then check both publisher digests."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = FILES[file["key"]]
    if destination.exists() and digest(destination) == expected_sha:
        return {"file": file["key"], "sha256": expected_sha, "size_bytes": destination.stat().st_size,
                "cached": True}
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > file["size"]:
            raise ValueError(f"Oversized partial download: {temporary}")
        if offset == file["size"]:
            break
        headers = {"User-Agent": "ACR-acoustic-reproduction/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(file["links"]["self"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = bool(offset and response.status == 206)
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected resume range")
                mode = "ab" if resumed else "wb"
                written = offset if resumed else 0
                next_report = written + 200 * 1024 * 1024
                with temporary.open(mode) as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        written += len(block)
                        if written >= next_report:
                            print(f'{file["key"]}: {written / 1e6:.0f}/{file["size"] / 1e6:.0f} MB', flush=True)
                            next_report += 200 * 1024 * 1024
            if temporary.stat().st_size == file["size"]:
                break
        except (OSError, TimeoutError) as exc:
            if attempt == 4:
                raise
            print(f'Retry {file["key"]}: {type(exc).__name__}', flush=True)
            time.sleep(min(2 ** attempt, 16))
    if temporary.stat().st_size != file["size"] or digest(temporary) != expected_sha:
        raise ValueError(f"Incomplete or SHA256-mismatched asset: {file['key']}")
    algorithm, expected = file["checksum"].split(":", 1)
    if digest(temporary, algorithm) != expected:
        raise ValueError(f"Zenodo checksum mismatch: {file['key']}")
    os.replace(temporary, destination)
    print(f'Verified {file["key"]}', flush=True)
    return {"file": file["key"], "sha256": expected_sha, "zenodo_checksum": file["checksum"],
            "size_bytes": destination.stat().st_size, "cached": False}


def _extract_archive(archive, destination):
    with zipfile.ZipFile(archive) as opened:
        members = opened.infolist()
        total = sum(member.file_size for member in members)
        if total > 8 * 1024 ** 3:
            raise ValueError("Unexpected SD-RR archive expanded size")
        seen = set()
        for member in members:
            relative = safe_relative(member.filename)
            if relative in seen:
                raise ValueError(f"Duplicate ZIP member: {relative}")
            seen.add(relative)
            if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                raise ValueError(f"Unsupported ZIP member: {member.filename}")
            if not member.is_dir() and relative.parts[0] not in ("references", "queries"):
                raise ValueError(f"Unexpected media archive member: {relative}")
        opened.extractall(destination)
    return {"archive": archive.name, "members": len(members), "expanded_bytes": total,
            "traversal_symlinks_encryption_duplicate_paths": "none"}


def download_release(root, workers=2):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = json.loads(urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}", timeout=60).read())
    if record["metadata"]["version"] != "1.0.0" or {f["key"] for f in record["files"]} != set(FILES):
        raise ValueError("Zenodo release differs from the pinned SD-RR v1.0 file set")
    (root / "zenodo_record.json").write_text(json.dumps(record, indent=2) + "\n")
    data = root / "data"
    data.mkdir(exist_ok=True)
    reports = []
    small = [f for f in record["files"] if not f["key"].endswith(".zip")]
    archives = [f for f in record["files"] if f["key"].endswith(".zip")]
    for file in small:
        reports.append(_download(file, data / file["key"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        reports += list(pool.map(lambda file: _download(file, root / "archives" / file["key"]), archives))
    archive_reports = [_extract_archive(root / "archives" / file["key"], data) for file in archives]
    expected = {}
    for row in read_csv(data / "reference_manifest.csv"):
        expected["references/" + row["reference_file_name"]] = row["reference_sha256"]
    for row in read_csv(data / "manifest.csv"):
        expected[row["query_path"]] = row["query_sha256"]
    if len(expected) != 1984:
        raise ValueError("Expected exactly 496 references and 1488 query files")
    for index, (relative, checksum) in enumerate(sorted(expected.items()), 1):
        if digest(data / safe_relative(relative)) != checksum:
            raise ValueError(f"Audio checksum mismatch: {relative}")
        if index % 250 == 0:
            print(f"Verified individual audio {index}/{len(expected)}", flush=True)
    audit = {"dataset_doi": DOI, "version": "1.0.0", "repository": UPSTREAM_REPOSITORY,
             "repository_revision": UPSTREAM_REVISION, "assets": reports, "archives": archive_reports,
             "individual_audio_sha256_count": len(expected), "audio_redistributed": False,
             "metadata_license": "CC BY-SA 4.0; embedded source metadata retain their upstream terms",
             "audio_license": "Per-track attribution.csv; no blanket license"}
    (root / "acquisition_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return data


def creator_key(value):
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _metadata_root(root):
    data = Path(root) / "data"
    if (data / "manifest.csv").exists():
        return data
    return Path(root) / "release_metadata"


def build_protocols(root, model_path, fit_protocol_path, probe_workers=4, metadata_only=False):
    root = Path(root).resolve()
    metadata_root = _metadata_root(root)
    data_root = root / "data"
    for name in FILES:
        if not name.endswith(".zip") and digest(metadata_root / name) != FILES[name]:
            raise ValueError(f"Publisher metadata hash mismatch: {name}")
    rows = read_csv(metadata_root / "manifest.csv")
    reference_rows = read_csv(metadata_root / "reference_manifest.csv")
    attribution = {r["reference_id"]: r for r in read_csv(metadata_root / "attribution.csv")}
    if len(rows) != 1488 or len(reference_rows) != 496:
        raise ValueError("Not the complete SD-RR v1.0 release")
    if len({r["query_id"] for r in rows}) != 1488 or len({r["reference_id"] for r in reference_rows}) != 496:
        raise ValueError("Duplicate native query or reference labels")
    with np.load(model_path, allow_pickle=False) as saved:
        model_metadata = json.loads(str(saved["metadata"]))
    if model_metadata["config"]["pca_dim"] != 32 or model_metadata["config"]["sample_rate"] != 8000:
        raise ValueError("This frozen transfer protocol requires the original FMA PCA32/8k model")
    fit_protocol = json.loads(Path(fit_protocol_path).read_text())
    fit_ids = {r["reference_id"] for r in fit_protocol["references"] if r["role"] == "pca_fit"}
    if set(model_metadata["fit_content_ids"]) != fit_ids:
        raise ValueError("Frozen PCA fit IDs do not match the original source protocol")
    source_sessions = {}
    for row in rows:
        source_sessions.setdefault(row["reference_id"], set()).add(row["recording_id"])
    if any(len(sessions) != 1 for sessions in source_sessions.values()):
        raise ValueError("A source crosses capture sessions; declared source/session split is invalid")
    if Counter(r["reference_id"] for r in rows) != Counter({r["reference_id"]: 3 for r in reference_rows}):
        raise ValueError("Each native reference must have exactly three real recordings")
    paths = [data_root / "references" / r["reference_file_name"] for r in reference_rows]
    paths += [data_root / safe_relative(r["query_path"]) for r in rows]
    decoded = {}
    if not metadata_only:
        if not (root / "acquisition_audit.json").exists():
            raise ValueError("Finish verified acquisition before building executable protocols")
        with concurrent.futures.ThreadPoolExecutor(max_workers=probe_workers) as pool:
            for index, (path, info) in enumerate(zip(paths, pool.map(decoded_audio_info, paths)), 1):
                decoded[str(path)] = info
                if index % 250 == 0:
                    print(f"Decoded physical bounds {index}/{len(paths)}", flush=True)
    references = []
    for row in reference_rows:
        rid = row["reference_id"]
        path = data_root / "references" / row["reference_file_name"]
        meta = attribution[rid]
        key = creator_key(meta["creator"])
        enrolled = int.from_bytes(hashlib.sha256(("sdrr-acoustic-open-v1\0" + key).encode()).digest()[:8], "big") < 2 ** 63
        session = next(iter(source_sessions[rid]))
        scope = "calibration" if session == "batch-037-706" else "test"
        references.append({"reference_id": "sdrr:" + rid, "upstream_reference_id": rid,
            "source_id": "jamendo:" + row["source_track_id"], "path": str(path),
            "creator_group": key, "recording_id": session, "artist_name": meta["creator"],
            "track_title": meta["work_title"], "track_url": meta["source_url"],
            "license_url": meta["license_url"], "license_family": meta["license_family"],
            "upstream_attribution": meta, "upstream_reference_metadata": row,
            "metadata_duration_s": float(row["source_duration_seconds"]),
            "sha256": row["reference_sha256"], "role": "test_known",
            "open_session_role": scope + ("_known" if enrolled else "_unknown"),
            **decoded.get(str(path), {})})
    reference_map = {r["upstream_reference_id"]: r for r in references}
    queries = []
    for row in rows:
        ref = reference_map[row["reference_id"]]
        path = data_root / safe_relative(row["query_path"])
        duration = float(row["query_duration_seconds"])
        begin = float(row["reference_begin_seconds"])
        end = float(row["reference_end_seconds"])
        if duration != 10.0 or abs(end - begin - duration) > 1e-8 or begin < 0:
            raise ValueError(f"Invalid native query interval: {row['query_id']}")
        if not metadata_only:
            if abs(decoded[str(path)]["duration_s"] - duration) > 1 / 8000:
                raise ValueError(f"Decoded query duration differs: {row['query_id']}")
            if end > ref["duration_s"] + 1 / 8000:
                raise ValueError(f"Reference crop exceeds decoded endpoint: {row['query_id']}")
        queries.append({"query_id": "sdrr:" + row["query_id"], "upstream_query_id": row["query_id"],
            "source_query_id": row["recording_id"], "recording_id": row["recording_id"],
            "reference_id": ref["reference_id"], "source_id": ref["source_id"],
            "creator_group": ref["creator_group"], "path": str(path), "start_s": 0.0,
            "duration_s": duration, "source_file_duration_s": duration,
            "expected_reference_start_s": begin, "expected_reference_end_s": end,
            "expected_time_scale": 1.0, "localization_tolerance_s": 0.1,
            "secondary_localization_tolerance_s": 2.0, "overlapping_other_annotations": [],
            "annotation": {"reference_begin": begin, "reference_end": end,
                "query_begin": 0.0, "query_end": duration, "tempo": "100", "pitch": "0"},
            "upstream_metadata": row, "alignment_warning": row["alignment_warning"],
            "qc_expected_rank": int(row["qc_expected_rank"]), "role": "test_known", "evaluate": True,
            "open_session_role": ref["open_session_role"], **decoded.get(str(path), {})})
    shared = {"dataset": "SD-RR-v1.0", "dataset_doi": DOI, "root": str(data_root),
        "references": references, "queries": queries,
        "frozen_model": {"path": str(Path(model_path).resolve()), "sha256": digest(model_path),
            "fit_protocol_path": str(Path(fit_protocol_path).resolve()), "fit_protocol_sha256": digest(fit_protocol_path),
            "fit_content_ids": sorted(fit_ids), "config": model_metadata["config"],
            "refit_on_sdrr": False},
        "provenance": {"manifest_sha256": FILES["manifest.csv"], "reference_manifest_sha256": FILES["reference_manifest.csv"],
            "recording_session_query_counts": dict(Counter(r["recording_id"] for r in rows)),
            "capture_devices": "2023 14-inch MacBook Pro speakers; iPhone 14 Pro Max / Voice Memos",
            "distance_m": 1.6, "system_volume_percent": 40,
            "publisher_environment_reference_counts": {"apartment": 460, "outdoors": 36},
            "per_query_environment_available": False, "qc_non_top1_count": sum(q["qc_expected_rank"] != 1 for q in queries),
            "alignment_warning_count": sum(bool(q["alignment_warning"]) for q in queries),
            "qc_queries_excluded": 0, "physical_bounds_verified": not metadata_only,
            "audio_not_redistributed": True, "metadata_license": "CC BY-SA 4.0 with source attribution",
            "licensing_files": ["attribution.csv", "LICENSES.md"],
            "cross_corpus_does_not_prove_distinct_compositions": True}}
    native = dict(shared, protocol={"name": "sdrr_native_closed_set_frozen_fma_pca_v1",
        "gallery_roles": ["test_known"],
        "closed_set": True, "calibration_queries": 0, "unknown_queries": 0,
        "frozen_parameters_from_original_music_protocol": True,
        "native_query_duration_s": 10.0, "centered_5s_is_separate_adaptation": True,
        "localization_tolerances_s": [0.1, 2.0], "source_cluster_unit": "source_id",
        "recording_session_count": 4, "model_fitting_on_sdrr": False})
    opened = json.loads(json.dumps(shared))
    for ref in opened["references"]:
        ref["role"] = ref["open_session_role"]
    for query in opened["queries"]:
        query["role"] = query["open_session_role"]
        if query["role"].endswith("unknown"):
            query["negative_population"] = "real_phone_recording_absent_creator_source"
    gallery_creators = {r["creator_group"] for r in opened["references"] if r["role"].endswith("_known")}
    unknown_creators = {r["creator_group"] for r in opened["references"] if r["role"].endswith("unknown")}
    if gallery_creators & unknown_creators:
        raise ValueError("An unknown creator is represented in the adapted gallery")
    cal_creators = {r["creator_group"] for r in references if r["open_session_role"].startswith("calibration")}
    test_creators = {r["creator_group"] for r in references if r["open_session_role"].startswith("test")}
    opened["protocol"] = {"name": "sdrr_session_open_set_frozen_fma_pca_v1", "closed_set": False,
        "gallery_roles": ["calibration_known", "test_known"],
        "calibration_fpr_target": 0.01, "calibration_sessions": ["batch-037-706"],
        "test_sessions": sorted({next(iter(sessions)) for sessions in source_sessions.values()} - {"batch-037-706"}),
        "known_unknown_assignment": "SHA256('sdrr-acoustic-open-v1\\0'+NFKC(creator).strip().casefold()) first64bits <2^63 => known",
        "creator_group_known_fraction": 0.5, "source_session_calibration_test_disjoint": True,
        "unknown_creator_gallery_disjoint": True, "shared_calibration_test_creator_count": len(cal_creators & test_creators),
        "reference_role_counts": dict(Counter(r["role"] for r in opened["references"])),
        "query_role_counts": dict(Counter(q["role"] for q in opened["queries"])),
        "source_cluster_unit": "source_id", "additional_cluster_unit": "creator_group",
        "native_full_protocol_preserved": True, "model_fitting_on_sdrr": False,
        "room_device_population_claim": False, "rare_fpr_population_claim": False}
    suffix = "_metadata_only" if metadata_only else ""
    for label, protocol in (("native", native), ("session_open", opened)):
        output = root / f"sdrr_{label}_protocol{suffix}.json"
        output.write_text(json.dumps(protocol, indent=2) + "\n")
        print(output, flush=True)
    if not metadata_only:
        audit = {"dataset_doi": DOI, "decoder": "FFmpeg mono float32 8000Hz actual sample count",
            "reference_count": 496, "query_count": 1488, "crop_endpoint_violations": 0,
            "query_requested_duration_s": 10.0,
            "decoded_query_frame_counts": dict(Counter(q["decoded_frames"] for q in queries)),
            "decoded_reference_duration_s_min": min(r["duration_s"] for r in references),
            "decoded_reference_duration_s_max": max(r["duration_s"] for r in references),
            "original_reference_bounds_preserved": True, "annotations_clamped": 0}
        (root / "physical_bounds_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return native, opened


def audit_source_overlap(root, fma_metadata, fit_protocol, pex_sha_files, prior_extra_protocol=None):
    """Flag exposure without equating unrelated corpus ID namespaces."""
    root = Path(root).resolve()
    metadata_root = _metadata_root(root)
    attribution = read_csv(metadata_root / "attribution.csv")
    pairs = {}
    for row in attribution:
        pairs.setdefault((creator_key(row["creator"]), creator_key(row["work_title"])), []).append(row["reference_id"])
    creators = {creator_key(row["creator"]) for row in attribution}
    matches = []
    shared_creators = set()
    count = 0
    with zipfile.ZipFile(fma_metadata) as archive:
        filename = next(name for name in archive.namelist() if name.endswith("/tracks.csv"))
        with io.TextIOWrapper(archive.open(filename), encoding="utf-8") as stream:
            reader = csv.reader(stream)
            columns = list(zip(next(reader), next(reader)))
            next(reader)  # pandas's index-name row
            artist_column = columns.index(("artist", "name"))
            title_column = columns.index(("track", "title"))
            subset_column = columns.index(("set", "subset"))
            for row in reader:
                count += 1
                artist, title = creator_key(row[artist_column]), creator_key(row[title_column])
                if artist in creators:
                    shared_creators.add(artist)
                if (artist, title) in pairs:
                    matches.append({"fma_id": row[0].zfill(6), "fma_subset": row[subset_column],
                        "artist": row[artist_column], "title": row[title_column],
                        "sdrr_reference_ids": pairs[(artist, title)]})
    encoded_digests = {}
    pex_ids = set()
    for filename in pex_sha_files:
        for path, checksum in json.loads(Path(filename).read_text()).items():
            if "/references/" in path:
                encoded_digests.setdefault(checksum, []).append(Path(path).stem)
                pex_ids.add(Path(path).stem)
    encoded_matches = [{"sdrr_id": row["reference_id"], "fma_ids": encoded_digests[row["reference_sha256"]]}
                       for row in read_csv(metadata_root / "reference_manifest.csv")
                       if row["reference_sha256"] in encoded_digests]
    fitted = {r["reference_id"] for r in json.loads(Path(fit_protocol).read_text())["references"] if r["role"] == "pca_fit"}
    prior_extra = []
    if prior_extra_protocol and Path(prior_extra_protocol).exists():
        matched_ids = {m["fma_id"] for m in matches}
        prior_extra = [{"source_id": q["source_id"], "role": q["role"], "query_id": q["query_id"]}
                       for q in json.loads(Path(prior_extra_protocol).read_text())["queries"]
                       if q["source_id"] in matched_ids and q.get("negative_population", "").startswith("additional_clean")]
    out = {"sdrr_reference_count": 496, "fma_metadata_track_count": count,
        "fma_metadata_sha256": digest(fma_metadata), "exact_normalized_creator_title_matches": matches,
        "all_fma_title_matched_sdrr_ids": sorted({v for m in matches for v in m["sdrr_reference_ids"]}),
        "possible_nmfp_training_superset_sdrr_ids": sorted({v for m in matches if m["fma_subset"] in ("small", "medium") for v in m["sdrr_reference_ids"]}),
        "original_pca_fit_id_title_matches": [m for m in matches if m["fma_id"] in fitted],
        "prior_pex_id_title_matches": [m for m in matches if m["fma_id"] in pex_ids],
        "prior_extra_unknown_id_title_matches": prior_extra,
        "shared_normalized_creator_count": len(shared_creators), "shared_normalized_creators": sorted(shared_creators),
        "existing_pex_reference_encoded_sha256_matches": encoded_matches,
        "method": "NFKC+strip+casefold exact creator/title pair; separate publisher-verified byte SHA256 against PEX references",
        "caution": "Different source namespaces or absent encoded-byte/title equality do not prove distinct compositions or absence of transformed-recording/pretraining exposure."}
    path = root / "source_overlap_audit.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(path, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "protocol", "overlap"))
    parser.add_argument("--root", type=Path, default=Path("work/benchmarks/acoustic_rr"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", type=Path, default=Path("work/benchmarks/acr_medium/pca_model.npz"))
    parser.add_argument("--fit-protocol", type=Path, default=Path("work/benchmarks/hard_medium_protocol.json"))
    parser.add_argument("--metadata-only", action="store_true", help="Write visibly provisional protocols before media verification")
    parser.add_argument("--fma-metadata", type=Path, default=Path("work/benchmarks/fma_metadata.zip"))
    parser.add_argument("--pex-sha", nargs="+", type=Path, default=[Path("work/benchmarks/pexafb_hard_small_media_sha256.json"), Path("work/benchmarks/pexafb_hard_medium_media_sha256.json")])
    parser.add_argument("--prior-extra-protocol", type=Path, default=Path("work/benchmarks/extended_unknown_protocol.json"))
    args = parser.parse_args()
    if args.command == "download":
        print(download_release(args.root, args.workers))
    elif args.command == "protocol":
        build_protocols(args.root, args.model, args.fit_protocol, args.workers, args.metadata_only)
    else:
        audit_source_overlap(args.root, args.fma_metadata, args.fit_protocol, args.pex_sha, args.prior_extra_protocol)


if __name__ == "__main__":
    main()
