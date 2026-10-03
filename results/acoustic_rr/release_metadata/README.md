# SD-RR v1.0

SD-RR (Song Describer Real Re-recording) is a closed-set audio-identification
benchmark built from openly licensed music in the Song Describer corpus. It
contains 496 reference recordings and 1,488 ten-second smartphone re-recorded
queries (three per reference).

Upstream Song Describer dataset: <https://doi.org/10.5281/zenodo.10072001>

## Downloaded files

Keep every release file in one directory, then verify it before extraction:

```bash
shasum -a 256 -c SHA256SUMS
unzip sdrr-v1.0-references.zip
unzip sdrr-v1.0-queries.zip
```

The extracted layout is:

```text
SD-RR-v1.0/
├── references/                 # 496 source reference MP3 files
├── queries/                    # 1,488 mono 44.1-kHz PCM WAV files
├── manifest.csv                # one row per query; all paths are relative
├── reference_manifest.csv      # reference identity and SHA-256
├── attribution.csv             # per-track attribution and exact audio license
├── exclusions.csv              # all 210 tracks excluded from the 706-track source
├── dataset_config.json         # deterministic excerpt/preparation parameters
├── dataset_summary.json        # machine-readable release counts
├── DATASET_CARD.md
├── LICENSES.md
├── CITATION.cff
└── SHA256SUMS
```

`manifest.csv` is the authoritative evaluation manifest. `reference_id` is the
closed-set class label, `reference_begin_seconds` is the ground-truth reference
offset, and `query_path`/`reference_path` resolve relative to this directory.
The non-contiguous reference identifiers are intentional: they preserve the
fixed ordering assigned before license filtering.

## Evaluation protocol

- Index all 496 files in `references/`.
- Evaluate all 1,488 rows in `manifest.csv`.
- Count a query as track-correct when the predicted `reference_id` equals its
  manifest value; failures to return a candidate count as incorrect.
- For track-and-offset accuracy at tolerance 0.1 s, additionally require the
  predicted reference offset to differ from `reference_begin_seconds` by no
  more than 0.1 s.
- Aggregate over queries for the headline Top-1 metrics. For inferential
  comparisons, cluster resampling or paired tests by `reference_id`, because
  each reference contributes three queries.

The POLARIS reproduction repository will accept this extracted directory as
`data/sdrr/`; no absolute paths from the original workstation are retained.

## Rights and attribution

The audio files do not have one global license. Read `LICENSES.md` and preserve
`attribution.csv` whenever the audio is redistributed. Some tracks prohibit
commercial use, and many require ShareAlike distribution of adaptations.

## Version

This directory describes SD-RR version 1.0.0. Once a DOI is reserved, add it to
`CITATION.cff` and cite the archived version rather than a mutable download URL.
