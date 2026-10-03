# Dataset card: SD-RR v1.0

## Summary

SD-RR is a real acoustic re-recording benchmark for closed-set audio
identification. Its source music comes from Song Describer. The public release
contains 496 reference tracks and three ten-second smartphone re-recordings per
track, for 1,488 queries.

## Source selection

Song Describer contains 706 unique tracks in the local source manifest. Tracks
were placed in a fixed pseudorandom order with seed `20260819`; the resulting
three-digit identifiers were retained throughout data preparation. The public
release excludes 209 tracks whose source licenses include a NoDerivatives
restriction and one track from which three non-overlapping usable ten-second
excerpts could not be selected. `exclusions.csv` records every excluded item and
reason. The retained-license distribution is recorded in `attribution.csv` and
summarized in `dataset_summary.json`.

## Excerpt preparation

Three non-overlapping ten-second windows were selected deterministically from
each retained source track. Before playback, each excerpt was normalized toward
`-20 dBFS RMS`, with a `-1 dBFS` peak ceiling. Exact reference offsets,
normalization gains, peak-limiting status, and source-window levels appear in
`manifest.csv`; the generation constants appear in `dataset_config.json`.

Each 46-second per-track playback unit contained three music excerpts and four
one-second synchronization chirps. The nominal music starts were 3, 18, and 33
seconds within each unit. Units were concatenated into continuous recording
batches.

## Acoustic recording

Playback used the built-in loudspeakers of a 14-inch 2023 MacBook Pro at a fixed
40% system-volume setting. An iPhone 14 Pro Max approximately 1.6 metres away
recorded with Apple Voice Memos. In aggregate, 460 tracks were recorded in a
residential apartment and 36 outdoors.

The four final recording-session identifiers and their query counts are stored
in `dataset_summary.json` and each query's session is stored in
`manifest.csv:recording_id`. A per-query apartment/outdoor label was not
recorded, so the aggregate 460/36 environment counts must not be interpreted as
query-level metadata.

## Alignment and query files

For each continuous batch, the playback-to-recording intercept was estimated
from the synchronization chirps at the beginning of the batch. Clock scale was
fixed to 1.0; later query positions were derived from the known playback
schedule rather than fitting a separate clock scale or aligning every track.
The resulting intercept, fixed scale, recording start, and any alignment warning
are present in `manifest.csv`.

Queries are 10-second, mono, 16-bit PCM WAV files at 44.1 kHz. The reference
files retain their Song Describer MP3 encodings. A separate spectral audit is
represented by the `qc_*` columns in `manifest.csv`.

## Intended and out-of-scope uses

The intended use is reproducible evaluation of audio-fingerprinting and music
identification systems under a physical loudspeaker-room-smartphone channel.
SD-RR is a closed-set identification benchmark, not a training corpus and not
an open-set rejection benchmark. It contains only one playback laptop, one
recording phone, and a limited set of environments. It was not designed to
evaluate global tempo or pitch invariance.

## Integrity and provenance

`reference_manifest.csv` and `manifest.csv` provide per-file SHA-256 values.
`SHA256SUMS` covers every release-level file and archive. The source Song
Describer manifest SHA-256, fixed seed, normalization targets, chirp definitions,
and timing constants are frozen in `dataset_config.json`.

## Ethical and privacy notes

The benchmark was designed to record music playback rather than people. Outdoor
recordings can nevertheless contain incidental environmental sound. No speaker,
speech, demographic, or location annotations are provided. Users should not use
the corpus for person identification or infer sensitive properties from
incidental background audio.

## Licensing

Audio licensing is per track. Consult `LICENSES.md` and `attribution.csv` before
use or redistribution. In particular, the release includes NonCommercial and
ShareAlike works and must not be treated as uniformly CC BY.

