# Frozen real-acoustic SD-RR evaluation

SD-RR v1.0.0 is the public Song Describer Real Re-recording release by Jiheng Li:
<https://doi.org/10.5281/zenodo.22169646>. The primary release card, manifests and
attribution are authoritative. The accompanying POLARIS source was inspected at
`fe8943897d3c723060d31485d83d042c9d6a3057`:
<https://github.com/JihengLi/POLARIS>.

The full native task has 496 Song Describer/Jamendo references and 1,488 real
10-second phone recordings, three per source. A 2023 14-inch MacBook Pro at 40%
volume played through built-in speakers; an iPhone 14 Pro Max approximately 1.6 m
away captured Apple Voice Memos audio. The publisher reports 460 apartment and 36
outdoor sources, but no per-query room label exists. Four capture-session IDs are
available. This is a limited physical playback/recording channel, not evidence
from connected-TV hardware or a device/room population.

Audio licenses are per track, including NonCommercial/ShareAlike/FreeArt terms.
SD-RR-authored metadata/documentation are CC-BY-SA 4.0, with upstream embedded
metadata retaining its own terms. Preserve the downloaded `attribution.csv`,
`LICENSES.md` and citation. This artifact redistributes no audio and makes no
independent rights-clearance claim.

## Acquisition and fixed protocols

Run from the original artifact/package root, using the primary Python environment
and FFmpeg. The original FMA PCA32 model and its source protocol must already
exist; no transform is refitted on SD-RR.

```sh
work/venv/bin/python outputs/acr_repro/acoustic_rr_data.py download --root work/benchmarks/acoustic_rr --workers 2
work/venv/bin/python outputs/acr_repro/acoustic_rr_data.py protocol --root work/benchmarks/acoustic_rr --workers 4 --model work/benchmarks/acr_medium/pca_model.npz --fit-protocol work/benchmarks/hard_medium_protocol.json
work/venv/bin/python outputs/acr_repro/acoustic_rr_data.py overlap --root work/benchmarks/acoustic_rr
```

Download checks include independent pinned publisher SHA256, Zenodo MD5, safe ZIP
paths and all 1,984 per-file SHA256 values. Both archives total 3.41 GB. FFmpeg sample
counts check every query and reference endpoint; all 1,488 queries have 80,000
samples at 8 kHz, no clamps or crop overruns. Source-reference durations range 33.891375
to 119.994625 s. Protocol files are:

- `sdrr_native_protocol.json`: all 496 references and all 1,488 queries, closed-set.
  No native unknowns or rejection calibration are invented. Candidate identity and
  offset accuracy at 0.1 s are native outcomes; 2 s is a labelled sensitivity. A centered
  5s crop is a separate adaptation with source offset advanced by 2.5 s.
- `sdrr_session_open_protocol.json`: new, separately labelled open-set adaptation.
  The largest recording session supplies calibration; other three sessions are held
  out. Normalized creator-group hash assigns known/unknown enrollment before scoring,
  and unknown groups are absent from the 231-reference gallery. Calibration has 552
  known/627 unknown queries; test has 141 known/168 unknown. All three clips of a
  source remain within one calibration/test scope. Fifty-one creator strings are
  shared between scopes, so this is not creator-disjoint calibration.

Full native metrics only use the already frozen FMA system/parameters. A new
representation chosen using SD-RR calibration cannot claim the full native query
set as held out, since 1,179 of those queries belong to that calibration scope.
No query is removed using upstream quality control or our outcomes:243 released
QC expected-rank values are not 1 and are retained as a reported stratum. Clock
scale remains the publisher's fixed 1.0; no oracle alignment or query-specific
clock fitting uses evaluation ground truth.

`source_overlap_audit.json` conservatively flags 10 exact normalized creator/title
matches with full FMA metadata, including five within the documented possible
NMFP FMA-small/medium training superset. Summaries preserve the full task and add
unchanged-gallery/gate sensitivities excluding 30 or 15 queries respectively. There
is no byte/title equality overlap with earlier PEX references or original PCA-fit
sources. Two matching titles appeared among prior additional FMA negatives;
these are disclosed. Different corpus names/IDs or missing equality do not prove
composition or model-training disjointness.

## Audfprint and public MinHash

Audfprint remains the previously pinned upstream implementation/configuration,
with both native internal count floor 5 and exposed-candidate floor 1 reported.
No native closed-set rejection threshold is inferred. These are our fixed
configurations, not the POLARIS paper's separately payload-matched Aud profiles.

```sh
work/venv/bin/python outputs/acr_repro/run_audfprint_variants.py --protocol work/benchmarks/acoustic_rr/sdrr_native_protocol.json --upstream work/benchmarks/audfprint --cache work/benchmarks/acoustic_rr/audfprint_cache --native-output-dir work/benchmarks/acoustic_rr/audfprint_native --calibrated-output-dir work/benchmarks/acoustic_rr/audfprint_candidate --durations 10 5 --workers 2
work/venv/bin/python outputs/acr_repro/run_audfprint_variants.py --protocol work/benchmarks/acoustic_rr/sdrr_session_open_protocol.json --upstream work/benchmarks/audfprint --cache work/benchmarks/acoustic_rr/audfprint_cache --native-output-dir work/benchmarks/acoustic_rr/audfprint_session_native --calibrated-output-dir work/benchmarks/acoustic_rr/audfprint_session_candidate --durations 5 --workers 2
```

Build the bridge as documented in `SOUNDFINGERPRINTING_BASELINE.md`. The public
library remains NuGet 15.14.1, source `9f882368258c76ac1776e192817d8f60eaa6f22b`,
native four-vote descriptor-pair qualification, native AudioSamplesNormalizer,
FFT/Haar/MinHash/path reconstruction and seed 20261002. Confidence is canonicalized
to 12 decimals only for calibration/acceptance; raw confidence and native ranking
are preserved. The phone recordings do not change any parameters.

```sh
work/venv/bin/python outputs/acr_repro/prepare_soundfingerprinting_pcm.py --protocol work/benchmarks/acoustic_rr/sdrr_native_protocol.json --cache work/benchmarks/acoustic_rr/soundfingerprinting_pcm --output work/benchmarks/acoustic_rr/soundfingerprinting_native_pcm.json --durations 10,5 --workers 2
DOTNET_CLI_HOME="$PWD/work/dotnet_home" DOTNET_PROCESSOR_COUNT=2 work/dotnet/dotnet work/soundfingerprinting_build/SoundFingerprintingBridge.dll --protocol work/benchmarks/acoustic_rr/sdrr_native_protocol.json --pcm-manifest work/benchmarks/acoustic_rr/soundfingerprinting_native_pcm.json --cache work/benchmarks/acoustic_rr/soundfingerprinting_cache --native-output-dir work/benchmarks/acoustic_rr/soundfingerprinting_native --calibrated-output-dir work/benchmarks/acoustic_rr/soundfingerprinting_permissive_unused --durations 10,5 --votes 4
work/venv/bin/python outputs/acr_repro/prepare_soundfingerprinting_pcm.py --protocol work/benchmarks/acoustic_rr/sdrr_session_open_protocol.json --cache work/benchmarks/acoustic_rr/soundfingerprinting_pcm --output work/benchmarks/acoustic_rr/soundfingerprinting_session_pcm.json --durations 5 --workers 2
DOTNET_CLI_HOME="$PWD/work/dotnet_home" DOTNET_PROCESSOR_COUNT=2 work/dotnet/dotnet work/soundfingerprinting_build/SoundFingerprintingBridge.dll --protocol work/benchmarks/acoustic_rr/sdrr_session_open_protocol.json --pcm-manifest work/benchmarks/acoustic_rr/soundfingerprinting_session_pcm.json --cache work/benchmarks/acoustic_rr/soundfingerprinting_cache --native-output-dir work/benchmarks/acoustic_rr/soundfingerprinting_session_native --calibrated-output-dir work/benchmarks/acoustic_rr/soundfingerprinting_session_permissive_unused --durations 5 --votes 4
```

The four-vote setting returns few candidates on these particular real recordings,
despite supported input lengths and nonempty 127/50 descriptors at 10/5 s. A separate
fixed-position clean-source diagnostic identifies 20/20 selected reference sources
at both durations in the same 496-source gallery. It uses first 20 metadata source
IDs and a fixed 10 s source position, not phone benchmark ground-truth frames.
This diagnoses integration, not microphone robustness or a general MinHash limit.
The original native scores, exclusions and parameters remain unchanged.

## Metrics and uncertainty

```sh
work/venv/bin/python outputs/acr_repro/summarize_acoustic_rr.py --protocol work/benchmarks/acoustic_rr/sdrr_native_protocol.json --results work/benchmarks/acoustic_rr/audfprint_candidate/audfprint_10s.json work/benchmarks/acoustic_rr/soundfingerprinting_native/soundfingerprinting_10s.json --exposure-audit work/benchmarks/acoustic_rr/source_overlap_audit.json --output work/benchmarks/acoustic_rr/native_summary.json
work/venv/bin/python outputs/acr_repro/summarize_acoustic_rr.py --protocol work/benchmarks/acoustic_rr/sdrr_session_open_protocol.json --results work/benchmarks/acoustic_rr/audfprint_session_candidate/audfprint_5s.json work/benchmarks/acoustic_rr/soundfingerprinting_session_native/soundfingerprinting_5s.json --exposure-audit work/benchmarks/acoustic_rr/source_overlap_audit.json --output work/benchmarks/acoustic_rr/session_summary.json
```

The helper checks complete IDs, frozen hash, source/role/crop consistency and
native candidate identity; enriches source/creator/session/QC strata; and reports
0.1s/2s localization. The adapted gate uses only 627 calibration-unknown recordings
at a 1% empirical target. Its 168 held-out unknown clips represent 56 sources in
three sessions. Source/creator bootstrap and source/session crossed intervals are
reported with zero-event degeneracy flags. A bootstrap [0,0] from zero observed
false accepts is not an upper population bound. Source-event Wilson sensitivity
has a different unit (any of three clips accepted); even zero events permit about
5.2% at the one-sided 95% binomial-source bound, conditional on source independence
and without a room/device population claim. Concurrent jobs mean these new
pipeline times do not support isolated acoustic latency comparisons.
