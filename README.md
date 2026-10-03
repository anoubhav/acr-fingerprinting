# ACR fingerprinting

A reproducible audio-fingerprinting implementation and experiment suite for content identification. The main encoder averages overlapping mel-spectral windows, normalizes band amplitudes and frequency differences, and projects them to 32 dimensions with PCA. Reference and query sampling are explicit, and matching returns a recording ID and time offset.

The repository includes public benchmark predictions, calibration decisions, source splits, and baseline adapters. It contains no audio datasets, proprietary notebooks, television recordings, neural weights, or manuscript files.

The method originates in the 2023 preprint by **Anoubhav Agarwaal, Prabhat Kanaujia, Sartaki Sinha Roy, and Susmita Ghose**, [Robust and lightweight audio fingerprint for Automatic Content Recognition](https://arxiv.org/abs/2305.09559). This implementation refits PCA on disjoint public calibration recordings and discloses new evaluation/matching choices; the public experiments are separate from that preprint's historical industrial measurements.

## Recorded findings

The adapted PEX/FMA music study uses 659 reference recordings and 543 known test crops. At 5 seconds, exact candidate accuracy is **446/543 (82.14%)** at both full reference density and eightfold decimation. Measured signal payload falls from **19.206 to 2.401 MB per audio-hour**.

On the common 50-query CPU cohort, the warmed encoder takes **1.628 ms per 5-second input**. The frozen IVF512/query-stride4 pipeline takes **5.931 ms** for encoding and matching. These are host measurements; reference allocation, setup, and decoding are separate quantities.

This is a resource trade-off. NMFP is more accurate and its server IVF matching is faster in these runs. The maintained public MinHash baseline has fewer observed false accepts in the added clean unknown pool. At an empirical 0.1% calibration target, the ACR query-stride4 setting accepts 361/543 correct noisy queries and falsely accepts 16/4,000 clean unknown recordings (0.40%). The held-out rate exceeds the calibration target and is reported as observed.

Read [the algorithm and parameter provenance](outputs/acr_repro/ALGORITHM.md) and [the experiment guide](outputs/acr_repro/README.md) for the full protocol and limitations.

## Quick checks

Use Python 3.12. Install FFmpeg/FFprobe separately and make them available on `PATH` for audio-file decoding and dataset preparation.

```sh
python3.12 -m venv work/venv
work/venv/bin/python -m pip install -r requirements-core.txt
work/venv/bin/python -m unittest discover -s outputs/acr_repro -p 'test_*.py'
```

The 15 tests check independent librosa frontend agreement, silence and gain behavior, physical clocks, source-grid decimation, exact distances, temporal votes, serialization, source isolation, affine time estimation, calibration ties, and rare-event uncertainty. They do not download benchmark media or neural models.

Once the public results are present, verify the recorded operating points directly from predictions:

```sh
work/venv/bin/python outputs/acr_repro/check_paper_numbers.py \
  --results results --check results/paper_numbers.json \
  --output work/operating_points_recomputed.json
```

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

Use sufficiently long calibration audio to fit the requested PCA dimension. A returned candidate is not an accepted recognition: choose a rejection threshold using a separate unknown-content calibration set. `model.save()` and `index.save()` preserve reusable state; the public experiment scripts record more complete provenance.

## Reproduce experiments

The [experiment guide](outputs/acr_repro/README.md) covers acquisition, source splitting, exact search, calibration-selected IVF/query sampling, the expanded absent-source study, speech perturbations, and baseline execution. It records upstream revisions and checkpoint hashes. Keep the existing `outputs/acr_repro/` layout: script defaults locate the repository root from it.

The layout is:

```text
outputs/acr_repro/    Encoder, evaluation scripts, baseline adapters, tests, guides
results/             Public predictions, protocols, summaries, audits, small PCA models
work/                Local downloads, environments and caches; ignored by Git
```

The full recorded primary environment is [pinned separately](outputs/acr_repro/requirements.txt). NMFP and PeakNet use separate requirements files and environments. Their source and weights are fetched from the original providers rather than redistributed here.

## License

New code is released under the [MIT License](LICENSE), copyright 2026 Anoubhav Agarwaal. Upstream software, model weights, and audio data retain their own licenses; see [third-party sources](THIRD_PARTY.md).
