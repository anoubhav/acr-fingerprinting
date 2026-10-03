# ACR fingerprinting

A reproducible audio-fingerprinting implementation and experiment suite for content identification. The encoder averages overlapping mel-spectral windows, normalizes band amplitudes and frequency differences, and projects them to 32 dimensions with PCA. Reference and query sampling are explicit; matching returns a recording ID, score and physical time offset.

The repository includes public benchmark predictions, frozen calibration decisions, source splits, baseline adapters, tests and provenance. Audio, proprietary notebooks, television recordings, neural weights and manuscript files are acquired or maintained separately. The one exported PDF is a standalone streaming-store figure.

The method originates in the 2023 preprint by **Anoubhav Agarwaal, Prabhat Kanaujia, Sartaki Sinha Roy, and Susmita Ghose**, [Robust and lightweight audio fingerprint for Automatic Content Recognition](https://arxiv.org/abs/2305.09559). This implementation refits PCA on disjoint public calibration recordings and discloses new evaluation and matching choices. These experiments are separate from the preprint's historical industrial measurements. The recovered SDK's 32 raw mel-band means precede gating, normalization, frequency differences and PCA; they are distinct from the public encoder's 32-dimensional PCA fingerprints.

## Recorded findings

The adapted PEX/FMA music study uses 659 reference recordings and 543 known test crops. At five seconds, exact candidate identification is **446/543 (82.14%)** at both full reference density and eightfold decimation. Signal payload falls from **19.206 to 2.401 MB per audio-hour**. Candidate correctness and thresholded recognition are reported separately.

The compact streaming store retains the frozen IVF512/probe16, reference-factor8/query-factor4 matcher. Across the full 42.27-hour gallery, explicit reference state falls from **222.106 to 111.095 MB**. All 835 queries preserve their complete top-ten candidate dictionaries, scores, offsets and original rejection decisions. Enrollment memory and timing are reported separately from retained state. On the common 50-query CPU cohort, the warmed encoder takes **1.615 ms per five-second input**, and the actual compact encoder-to-matcher pipeline takes **5.787 ms**. These desktop host measurements do not establish TV hardware latency.

Matched density controls do not establish an ACR storage–quality frontier: NMFP with query/reference factors2 identifies **464/543** known crops using **1.840 MB per reference audio-hour**. Native four-vote MinHash with reference factor2 and dense queries uses **1.899 MB/hour**, with three observed false accepts among 4,000 clean unknown test recordings. Sparse and native variants retain their declared clocks and calibration populations. The canonical PeakNetFP adapter now rounds on each track's physical clock; regression tests cover independence from unrelated catalog prefixes.

Real SD-RR speaker-to-phone recordings expose a transfer boundary. Frozen ACR identifies **72/1,488** ten-second recordings; **39/1,488** also meet the 0.1-second localization tolerance. Its separately labelled five-second session-open test identifies **19/141** known clips. A post-hoc query gain diagnostic, fitted only on calibration pairs, raises that to **34/141** and accepted correct matches from **3 to 11/141**, while accepting **10/168** unknown clips. This limited improvement is not robust acoustic recognition. The native closed-set study has no unknown-content population or invented false-accept rate; the open test contains only three held-out recording sessions.

The added review diagnostics retain every prescribed condition and reference-grid phase. Synthetic leading silence or initial truncation keeps aggregate candidate counts close to the original result, while accepted wrong known IDs rise from 2 to 4–6 and observed unknown accepts can rise from 1 to 2/71. All eight factor8 reference residues produce 445–449 correct candidates and 400–410 accepted correct matches; no favorable phase is selected. A full 543-query annotation partition exposes only 5/78 candidate matches in the pitch-altered stratum, with other distortions allowed to co-occur. These are scoped diagnostics rather than replacement headline results or causal attack claims.

Every one of the 16 previously accepted clean source-ID negatives in the expanded q4 test also receives the existing fixed waveform-overlap audit. Fifteen have complete five-second support; none reaches the 0.95 absolute-Pearson flag threshold, with maximum 0.153449. One unsupported reference crop is retained. The original 16/4000 count and labels stay unchanged, and absence of strong waveform overlap cannot prove distinct recording or composition identity. See [the review protocol and commands](outputs/acr_repro/ADVERSARIAL_REVIEW.md).

Read [the experiment and evidence guide](EXPERIMENTS.md), [algorithm and parameter provenance](outputs/acr_repro/ALGORITHM.md), [compact store](outputs/acr_repro/COMPACT_STORE.md), and [SD-RR protocol](outputs/acr_repro/SDRR.md) for denominators, calibration scope, confidence intervals and limitations.

## Quick checks

Use Python 3.12. Install FFmpeg/FFprobe separately and make them available on `PATH` for decoding and dataset preparation.

```sh
python3.12 -m venv work/venv
work/venv/bin/python -m pip install -r requirements-core.txt
work/venv/bin/python -m unittest discover -s outputs/acr_repro -p 'test_*.py'
work/venv/bin/python outputs/acr_repro/test_neural_acoustic.py
```

The 27 unit tests and separate acoustic end-to-end oracle check frontend agreement, physical clocks, source isolation, exact distances, temporal votes, serialization, compact-store conformance, track-local Peak rounding, calibration gates, gain scope and correlated-query uncertainty. They download no benchmark media or neural models. Inspect the retained evidence without audio or external weights:

```sh
work/venv/bin/python outputs/acr_repro/verify_v2_evidence.py \
  --check results/v2_evidence_summary.json
work/venv/bin/python outputs/acr_repro/verify_v3_evidence.py
work/venv/bin/python outputs/acr_repro/check_paper_numbers.py \
  --check results/paper_numbers_v2.json
work/venv/bin/python outputs/acr_repro/derive_matrix_summary.py --check
work/venv/bin/python outputs/acr_repro/derive_v2_profile_summary.py \
  --check results/profiling/paper_timing_final.json
```

`SHA256SUMS.json` binds all released files except itself. `EXPORT_PROVENANCE.json` preserves original execution-byte hashes separately from hashes of path-sanitized exports. The numerical verifiers check both recorded protocol identities and exported bytes. The V3 checker rederives every stress/phase prediction, all strata intervals and resource-replica ranges, and binds the complete frozen 16 waveform roster to the original gate and outcomes. It inspects retained audit evidence without claiming to decode audio or reread feature caches.

## Use the encoder

Run this from the repository root. Fit PCA on calibration recordings that are disjoint from reference/query recordings.

```python
import sys
sys.path.insert(0, "outputs/acr_repro")
from acr_fp import Fingerprinter, ExactIndex, load_audio

fit_audio, sr = load_audio("path/to/calibration.wav")
model = Fingerprinter().fit_audio([("calibration", fit_audio, sr)])

reference, sr = load_audio("path/to/reference.wav")
vectors, times = model.extract(reference, sr)
model.assert_disjoint(reference_ids=["reference"], query_ids=["query"])

config = model.config
index = ExactIndex.from_entries(
    [("reference", vectors, times)], factor=8,
    grid_step_sec=config.hop_length * config.stride_frames / config.sample_rate,
    grid_origin_sec=config.first_center_seconds,
)
query, sr = load_audio("path/to/query.wav")
query_vectors, query_times = model.extract(query, sr)
hits = index.match(query_vectors, query_times)
```

Use sufficiently long calibration audio to fit the requested PCA dimension. A returned candidate is not an accepted recognition: choose a rejection threshold using a separate unknown-content calibration set. `model.save()` and `index.save()` preserve reusable state. The [compact-store guide](outputs/acr_repro/COMPACT_STORE.md) documents streaming enrollment for the frozen IVF configuration.

## Reproduce experiments

The [source experiment guide](outputs/acr_repro/README.md) covers public acquisition, source splitting, exact search, calibration-selected IVF/query sampling, the expanded absent-source study, speech perturbations and baseline execution. The [experiment guide](EXPERIMENTS.md) adds density controls, streaming enrollment, real acoustic recordings, the channel diagnostic and corrected clocks. Upstream revisions and checkpoint hashes are recorded. Keep the existing `outputs/acr_repro/` layout: script defaults locate the repository root from it.

```text
outputs/acr_repro/    Encoder, evaluation scripts, baseline adapters, tests, guides
results/             Predictions, protocols, summaries, audits, small fitted states
work/                Local downloads, environments and caches; ignored by Git
```

The full recorded primary environment is [pinned separately](outputs/acr_repro/requirements.txt). NMFP and PeakNet use separate requirements files and environments. Their source and weights are fetched from the original providers rather than redistributed here. Source and session splits are stated explicitly; they must not be interpreted as universally creator-disjoint.

## License

New code is released under the [MIT License](LICENSE), copyright 2026 Anoubhav Agarwaal. Upstream software, model weights and datasets retain their own licenses; see [third-party sources](THIRD_PARTY.md). Retained SD-RR metadata includes its attribution and license records.
