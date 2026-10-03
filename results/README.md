# Recorded public experiments

These files contain predictions, source/crop protocols, calibration decisions, derived summaries, numerical audits, and small fitted PCA models. Audio, neural weights, and large search indexes are not included. All source IDs refer to the declared public FMA/PEX or LibriSpeech populations.

Useful entry points:

- `hard_medium_protocol.json`: source-disjoint PEX/FMA split and physical crop definitions.
- `paper_numbers.json`: calibrated original and expanded music operating points.
- `paper_matrix_counts.json`: all 25 exact duration/reference-density cells.
- `paper_timing.json` and `profiling/`: common-input CPU summary and raw repeat measurements.
- `acr_ablations/`: the eight prespecified sparse ablations and their fitted PCA states.
- `acr_ivf_study/` and `acr_ivf_efficiency_study/`: calibration grids and frozen search settings.
- `acr_ivf_extended/`: both frozen settings on the expanded absent-source population.
- `acr_source_unseen_comparison.json`: paired conservative source-exposure sensitivity.
- `speech_protocol.json` and `speech_summary_final.json`: speaker-disjoint synthetic speech adaptation.

Keep candidate accuracy, accepted-correct rate, localization, and unknown false accepts separate. Original small-negative and expanded-negative populations use different calibration sample sizes and targets. A calibration target is not an observed held-out error rate. Report added clean sources separately from official distorted PEX negatives.

The protocol metadata retains per-recording attribution/license fields and rights-review flags. This repository's software license does not relicense the source audio or upstream annotations. No audio is bundled.

To verify the numbers without downloading datasets, follow the commands in the root README or use `check_paper_numbers.py`, `derive_matrix_summary.py`, and `derive_profile_summary.py` under `outputs/acr_repro/`.
