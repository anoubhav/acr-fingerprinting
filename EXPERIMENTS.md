# V2 experiment and evidence guide

Keep the exported layout `outputs/acr_repro/`, `results/`, and a new `work/` directory. Commands run from this package root. Install the primary Python requirements and FFmpeg as described in the source README. TensorFlow/official NMFP weights, PyTorch/Essentia/official PeakNetFP weights, .NET/SoundFingerprinting and large public audio are separate reproduction dependencies, not prerequisites for inspecting retained numerical evidence.

## Verify without audio or external weights

```sh
work/venv/bin/python -m unittest discover -s outputs/acr_repro -p 'test_*.py'
work/venv/bin/python outputs/acr_repro/test_neural_acoustic.py
work/venv/bin/python outputs/acr_repro/verify_v2_evidence.py --check results/v2_evidence_summary.json
work/venv/bin/python outputs/acr_repro/check_paper_numbers.py --check results/paper_numbers_v2.json
work/venv/bin/python outputs/acr_repro/derive_matrix_summary.py --check
work/venv/bin/python outputs/acr_repro/derive_v2_profile_summary.py --check results/profiling/paper_timing_final.json
```

The V2 verifier derives accuracy, localization and calibration gates from current canonical individual predictions, including corrected Peak rows. It checks protocol identities, complete cohorts, dense/sparse reference controls, all24 fresh-process streaming-store replicas, full835 ranking/decision conformance, calibrated32-band gain scope and actual cache-creation chronology, and seven current common-cohort timing rows. It retains original execution hashes while separately checking path-sanitized export bytes through EXPORT_PROVENANCE.json and SHA256SUMS.json. A changed filesystem path does not silently replace the recorded original protocol identity.

## Frozen original music and speech studies

The main README reproduces public PEX/FMA acquisition, source-disjoint PCA fitting, all25 duration/reference-factor cells, calibrated IVF settings, the5000 additional absent-source pool and the speaker-disjoint LibriSpeech adaptation. Keep these frozen protocols and scripts. Original small-negative and expanded-negative gates use different declared calibration populations and must remain separate. A numeric equality in an aggregate table does not imply identical individual predictions.

## Matched reference budgets

`NEURAL_DENSITY_CONTROLS.md` documents frozen NMFP and corrected Peak sparse-reference/query controls. `SOUNDFINGERPRINTING_BASELINE.md` documents native four-vote MinHash reference factors1/2/4, preserving original sequence numbers, starts, signature values and full duration while keeping query extraction dense. No reference-density-specific native temporal setting is introduced. Source/protocol/config/hash/parity audits are retained in results/neural_density_controls and results/soundfingerprinting_density_*. Native and sparse points are explicit variants, not a claim that every library recommends decimation.

## Compact state and enrollment

`COMPACT_STORE.md` explains the stream-enrolled uint32 original-frame indices plus per-track boundaries/IDs, eliminating float64 per-row clocks and unchanged frozen IVF/voting system. Results/streaming_index includes24 fresh-process reports across141/256/512/659 source prefixes, publication_summary.json, full835_parity.json and a vector figure. All complete candidate dictionaries and descriptors agree across layouts; historical source hashes describe the measured prototype, while current integrated source hashes are separately recorded in the actual compact profile. Original plan/private-path summaries, raw metadata/index binaries and feature caches are excluded. Explicit state, process RSS, serialization and matching are different measures.

## Real acoustic recordings and supervised diagnostic

`ACOUSTIC_RR_BASELINE.md`, `SDRR.md` and `NEURAL_ACOUSTIC.md` reproduce SD-RR acquisition, frozen full496-source/1488-query transfer and separately labelled creator/session open adaptation. All243 publisher QC non-top1 queries remain in the primary cohort. The native task has no OOV population or invented gate. The open adaptation uses only627 calibration unknowns for its1% empirical gate and tests141 known/168 unknown clips; its56 unknown sources and three test sessions do not validate rare/device/room population FPR.

A separate post-hoc32-band query gain diagnostic fits only552 known calibration pairs from184 sources, preserves references/original FMA PCA/settings, freezes gains and a new gate before scoring309 held-out clips, and reports limited improvement rather than acoustic SOTA. The exported chronology ledger records source hashes, creation timestamps and the held-out cache audit without exporting raw caches. Filesystem chronology is an audit trail, not independently tamper-proof evidence.

The ten title-matched FMA-source/possible training exposure flags, physical endpoints, neural grid-origin audit and source/creator/session/QC strata remain available. Sparse neural localization is measured on actual physical clocks; unavailable offsets on a coarser query/reference grid are not evidence of weak embeddings. Zero-event empirical bootstrap[0,0] is not an upper population false-accept bound.

## Export scope

Only canonical predictions, declared gates/plans, small aggregate PCA/gain states, public attribution metadata, source code, tests and provenance are bundled. Media/PCM, raw feature or embedding caches, runtime binaries, index snapshots, private Drive notebooks/TV captures, execution logs and obsolete Peak compatibility files are excluded. THIRD_PARTY.md distinguishes new code, upstream implementations, weights and each dataset's terms. New timings come from actual warmed encoder-to-matcher calls on the same50 predecoded clips; no desktop number establishes TV hardware latency or fleet scale.
