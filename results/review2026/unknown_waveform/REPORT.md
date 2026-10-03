# Frozen ACR clean-unknown waveform audit

The audit pre-listed all 16 accepted clean unknowns under the existing expanded q=4/probe16 gate, including source IDs, predicted recording IDs, offsets, complete crop bounds and encoded-input hashes. It retained the existing rule: absolute Pearson correlation at least 0.95 over the complete five-second crop, with an independent sample-lag search within ±0.25 s of the frozen predicted offset. No gate, score, prediction, label or false-accept numerator changed.

All 16 cases were attempted. Fifteen had sufficient full-crop support and were measurable: absolute correlation maxima range 0.021097–0.153449, median 0.046497; none reached 0.95. One predicted offset lacked a full five-second reference crop and remains unverified. No constant query/reference cases occurred.

An independent verifier recomputed every full-lag maximum with NumPy FFT and FFT rolling moments, then checked direct centered Pearson correlation at each selected lag. Maxima and lags match the original SciPy-based audit; maximum direct-Pearson difference is 1.96e-15. It also rechecks original frozen cohort/gate bindings, encoded-source digests and decoded crop/window digests.

The diagnostic is limited to strong shared five-second waveforms near the predicted offset. Low correlation does not prove distinct compositions or whole-recording identity; the unsupported case is not treated as a negative result. The main source-ID false-accept count remains 16/4,000.

Authoritative records: `results/plan.json`, `results/summary.json`, `results/strict_validation.json`. Audio is not exported.
