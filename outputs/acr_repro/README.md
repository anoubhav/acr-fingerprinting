# Reproducible sparse ACR study

This artifact contains the recovered-parameter ACR reconstruction, exact and IVF retrieval, calibrated summaries, public-data adapters, and pinned baseline bridges. The research package includes a companion manuscript and individual-query results; the public code/results repository omits manuscript files. It does not redistribute proprietary notebooks, television recordings, or public music audio. The reconstruction is distinguished from the original production PCA transform and historical measurements throughout the paper.

Keep the package layout: `outputs/acr_repro/`, `results/`, and a new `work/` directory for downloaded assets and caches. All commands below run from the package root. Use Python 3.12 and FFmpeg/FFprobe. The recorded run used Apple M2 Max, 12 CPU cores, 32 GiB RAM, FFmpeg 6.0, and FAISS 1.15.1. Exact-search accuracy does not require a GPU. The CPU comparisons use explicit FAISS CPU indexes; no GPU clone is used for reported operating points.

## Inspect the evidence without downloading audio

The `results/` directory contains complete predictions, split/crop protocols, summaries, source-exposure and decoding audits, and calibration selections. Paths in exported JSON are relative to the package root. Models and search settings were selected from calibration data; held-out thresholds in `describe_roc.py` are labeled retrospective discrimination curves and are not validated deployment thresholds. Small-negative and expanded-negative studies remain separate. Speech intervals account for repeated utterances and the small number of held-out speakers.

`ALGORITHM.md` documents the extraction parameters and differences from the supplied notebooks. `NEURAL_BASELINE.md`, `PEAKNET_BASELINE.md`, and `SOUNDFINGERPRINTING_BASELINE.md` document original checkpoints, source revisions, precision, native matching and supported window lengths. The public MinHash library is not the proprietary 72-byte predecessor. Its native four-vote row is the serious baseline; permissive one-vote results are a sensitivity, not a substitute.

After installing the primary environment below, verify the manuscript operating-point summaries and common-cohort timing directly from the retained predictions and raw timing repeats. These checks need no audio or model weights:

```sh
work/venv/bin/python outputs/acr_repro/check_paper_numbers.py --check results/paper_numbers_v2.json
work/venv/bin/python outputs/acr_repro/verify_v2_evidence.py --check results/v2_evidence_summary.json
work/venv/bin/python outputs/acr_repro/derive_v2_profile_summary.py --check results/profiling/paper_timing_final.json
```

The scripts discover the packaged `results/` layout by default and write recomputed evidence under `work/`. Their `--results`/`--profiles` and `--output` options also support another layout. `results/secondary_source_sensitivity.json` reports an additional admissibility analysis using actual five-second crop overlap with other annotated gallery sources; it preserves primary target labels and frozen thresholds.

## Install the primary environment and validate the implementation

```sh
python3.12 -m venv work/venv
work/venv/bin/python -m pip install -r outputs/acr_repro/requirements.txt
work/venv/bin/python -m unittest discover -s outputs/acr_repro -p 'test_*.py'
```

The tests compare the frontend to an independent librosa/direct-window calculation, preserve physical clocks through silence, check original-grid thinning and exact distances, validate unique temporal votes and missing ANN slots, and check calibration ties, source exclusion, and rare-event intervals. They are implementation checks, not substitutes for benchmark evaluation.

## Recreate the music protocol and run the non-neural system

```sh
work/venv/bin/python outputs/acr_repro/fetch_assets.py --assets pex_small pex_medium fma_metadata --repos audfprint
work/venv/bin/python outputs/acr_repro/benchmark_data.py work/benchmarks/pexafb_hard_small --probe --output work/benchmarks/hard_small_manifest.json
work/venv/bin/python outputs/acr_repro/benchmark_data.py work/benchmarks/pexafb_hard_medium --probe --output work/benchmarks/hard_medium_manifest.json
work/venv/bin/python outputs/acr_repro/protocol.py work/benchmarks/hard_medium_manifest.json work/benchmarks/hard_medium_protocol.json --exclude-manifest work/benchmarks/hard_small_manifest.json
work/venv/bin/python outputs/acr_repro/run_acr_protocol.py --protocol work/benchmarks/hard_medium_protocol.json --cache work/benchmarks/acr_cache --output work/benchmarks/acr_exact --durations 5,1,2,3,10 --factors 1,2,4,6,8 --workers 3 --threads 4
```

The main source split has 94 PCA-fit sources, 659 gallery sources, 147 calibration-known, 74 calibration-unknown, 543 test-known, and 71 test-unknown query crops. Twelve development-overlap source IDs are excluded, and unknown crops overlapping a gallery source are excluded before predictions. Audio boundaries are based on actual decoded sample counts, not rounded container durations. The 1/2/3/5/10-second task is an adapted centered annotated-chunk identification task, not PEX's native montage segmentation task.

Fit the declared calibration-only IVF/query-stride grid, then evaluate both frozen selections:

```sh
work/venv/bin/python outputs/acr_repro/study_query_ivf.py --protocol work/benchmarks/hard_medium_protocol.json --model work/benchmarks/acr_exact/pca_model.npz --cache work/benchmarks/acr_cache --output work/benchmarks/acr_ivf --threads 4
work/venv/bin/python outputs/acr_repro/study_query_ivf.py --protocol work/benchmarks/hard_medium_protocol.json --model work/benchmarks/acr_exact/pca_model.npz --cache work/benchmarks/acr_cache --output work/benchmarks/acr_ivf_cost --threads 4 --reuse-calibration work/benchmarks/acr_ivf --max-calibration-loss-pp 2
```

## Baselines

Audfprint source is imported unchanged at the pinned revision. The extra candidate variant exposes low-count evidence for outer calibration while retaining the native row.

```sh
work/venv/bin/python outputs/acr_repro/run_audfprint_variants.py --protocol work/benchmarks/hard_medium_protocol.json --upstream work/benchmarks/audfprint --cache work/benchmarks/audfprint_cache --native-output-dir work/benchmarks/audfprint_native --calibrated-output-dir work/benchmarks/audfprint_candidate --durations 5 1 2 3 10 --workers 4
```

For NMFP and PeakNet, create separate environments from their requirements files, then fetch verified weights/source using `fetch_assets.py --assets nmfp peaknet --repos nmfp peaknet kapre`. Follow the two baseline documents for extraction/cache and native exact/IVF retrieval commands. They restore the authors' weights rather than train new networks. Neural source exposure is audited against the conservative FMA-medium/small superset. Pretrained-model accuracy on this task is measured anew; no published number is substituted.

The MinHash bridge uses NuGet 15.14.1 and the audited release commit, with a locked dependency file. Its build/cache paths can be set inside `work/`; the detailed bridge document includes commands. SoundFingerprinting's full physical support is 1.846 seconds, so its native released configuration is marked unavailable for fresh 1-second input. Native Confidence is canonicalized to 12 decimal places only for threshold calibration/acceptance, with raw scores preserved, to prevent timestamp-subtraction jitter from splitting identical one-fingerprint evidence.

## Expanded absent-source study

```sh
work/venv/bin/python outputs/acr_repro/extra_unknown_data.py --metadata work/benchmarks/fma_metadata.zip --exclude-protocols work/benchmarks/hard_medium_protocol.json work/benchmarks/hard_small_manifest.json --output work/benchmarks/extra_unknown --count 5000 --workers 8
work/venv/bin/python outputs/acr_repro/extended_unknown_protocol.py --base work/benchmarks/hard_medium_protocol.json --extra work/benchmarks/extra_unknown/manifest.json --output work/benchmarks/extended_unknown_protocol.json
work/venv/bin/python outputs/acr_repro/run_ivf_extended.py --protocol work/benchmarks/extended_unknown_protocol.json --original-protocol work/benchmarks/hard_medium_protocol.json --model work/benchmarks/acr_exact/pca_model.npz --cache work/benchmarks/acr_cache --index work/benchmarks/acr_ivf/ivf_factor8.index --output work/benchmarks/acr_ivf_extended --threads 4
```

The 5,000 added clips have absent source IDs: 1,000 calibrate rejection and 4,000 are held out. This is not proof of distinct compositions or transformed recordings. Acquisition uses HTTPS archive ranges, member CRC checks and file SHA256, not a whole-archive SHA1 claim. Decoding exclusions happen before prediction. At the empirical 0.1% calibration target, report actual held-out rates and intervals even when they exceed the target. Positive distorted PEX crops, additional clean unknowns and original distorted unknowns are separate populations.

## Speech and summaries

Fetch `librispeech` with `fetch_assets.py`, then generate the frozen speaker-disjoint 5-second conditions using `speech_benchmark.py`. Run the same methods against its protocol. One pooled calibration threshold across ten perturbations is declared, and `summarize_speech.py` reports utterance/speaker intervals. It is a synthetic content-identification adaptation of an ASR corpus; no standard LibriSpeech fingerprinting benchmark is claimed.

Use `summarize_results.py` with the matching protocol for calibrated music points, `summarize_speech.py` for speech, and `describe_roc.py` only for explicitly retrospective test-derived curves. `portable_manifest.py --materialize --workspace .` rebases exported relative paths if using the provided split/crop files directly. Recreating protocols from downloaded audio is preferred for a fresh run; the exported protocols remain an independent check of source membership and numerical crop bounds.

Extraction, decoding, matching, allocations, serialized storage and signal payload are distinct measurements. Cache-hit build time is not cold build time. The isolated common-input profiles record actual warmup and model setup separately. Desktop timing and model coefficient counts do not establish TV hardware latency, device RAM, energy or fleet throughput.

The full 25-cell duration/reference-density matrix can also be checked without audio:

```sh
work/venv/bin/python outputs/acr_repro/derive_matrix_summary.py --check
work/venv/bin/python outputs/acr_repro/compare_reference_density.py --results results/acr_medium
```

The latter reports unthresholded paired changes; equal aggregate candidate counts need not mean identical predictions.

## V2 evidence

The final V2 experiment/export guide is `EXPERIMENTS.md`; `THIRD_PARTY.md` records SD-RR metadata and audio license distinctions. New studies include neural/public-MinHash matched reference budgets, exact compact streaming-store conformance and fresh-process memory scaling, real speaker-to-phone transfer with all quality-control strata, an explicitly post-hoc calibration-only gain diagnostic, and current actual compact/sparse-model CPU profiles. The exported default checks use corrected canonical Peak records and V2 timings; old Peak compatibility outputs and old timing tables are not primary evidence. Audio-free verification distinguishes original execution SHA identities from portable export SHA identities.
