# Adversarial-review evidence

These diagnostics retain the original models, primary predictions and protocol. They were added after the primary analysis to test particular weaknesses; they do not replace the original full-cohort results or establish production performance. All configurations and prescribed conditions are retained, including unfavorable outcomes.

## Inspect the evidence without audio

Run from the exported package or repository root after installing the primary requirements:

```sh
python outputs/acr_repro/verify_v2_evidence.py --check results/v2_evidence_summary.json
python outputs/acr_repro/verify_v3_evidence.py
```

The V3 checker independently rederives all 3,340 stress predictions, all 6,680 reference-phase predictions, every metadata stratum across eight configurations, dense/sparse paired intervals, and the ranges of 24 actual streaming-store process trials. It verifies original execution identities through `EXPORT_PROVENANCE.json`, whose separate export hashes bind path-sanitized bytes. Dictionary path keys are rebased with collision checks. No audio is decoded and no feature cache is reread by this portable check; the retained PCM/cache digests and independent execution audits are provenance rather than a new inference run.

It also binds the entire frozen16 clean-unknown waveform-audit roster to the original accepted predictions and verifies the unchanged criterion, crop/digest/status ledger and agreement recorded by the independent numerical audit. Repeating the waveform calculation itself requires downloaded audio, as described below.

## Digital-input leading silence and truncation

`results/review2026/digital_boundary/` contains the frozen plan, original-crop and source-PCM digest ledger, all four conditions, paired clustered intervals, decision switches, and the independent full-audio verification report. The unchanged 5-second condition exactly reproduces all 835 original compact IVF/query-stride results before transformed inference is permitted. Every condition uses the original factor8 reference store, query stride4, IVF512/probe16, fitted PCA and original empirical gate. Leading silence shifts the source-time intercept by `-lead * time_scale`; initial trimming shifts it by `+trim * time_scale`. Surviving query frames keep their original physical modulo4 positions.

| Condition | Correct candidate /543 | Accepted correct | Accepted wrong known ID | Unknown accepts /71 |
|---|---:|---:|---:|---:|
| Unchanged5s |446|401|2|1|
| Leading0.5s silence, still5s input |444|400|4|1|
| Leading1s silence, still5s input |442|396|6|2|
| Trim initial1s, retain4s input |444|403|5|2|

The small change in aggregate recognition hides increased wrong-ID/unknown acceptances and individual decision changes. This is a prescribed synthetic capture-boundary stress on public montage audio, not actual television capture or validation of deployment impact. The2s annotation tolerance is primary;0.1s is an integer-annotation sensitivity and cannot establish synchronization accuracy.

The byte-identical recorded runner uses explicit paths, so it has no package-dependent input defaults. To reproduce inference, first acquire the original PEX assets, fit the original disjoint PCA, and prepare the frozen compact IVF snapshot as documented in `README.md` and `COMPACT_STORE.md`. Then run with a fresh output directory:

```sh
python outputs/acr_repro/run_digital_boundary.py --source-dir outputs/acr_repro \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --model work/benchmarks/acr_medium/pca_model.npz \
  --index work/benchmarks/compact_store/compact_659_r0.index \
  --metadata work/benchmarks/compact_store/compact_659_metadata.npz \
  --frozen-result work/benchmarks/acr_ivf_efficiency_study/acr_ivf_selected_5s.json \
  --output work/digital_boundary_fresh
python outputs/acr_repro/verify_digital_boundary_execution.py --source-dir outputs/acr_repro \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --frozen-result work/benchmarks/acr_ivf_efficiency_study/acr_ivf_selected_5s.json \
  --results work/digital_boundary_fresh --output work/digital_boundary_audio_audit.json --audio
```

Replace the model/index/metadata arguments with the actual reproduced artifacts; the runner verifies their full hashes and cohort. Its plan-only mode freezes all inputs before inference. The exported original plan contains execution digests but is not a substitute for the actual model/index/audio required to reproduce inference.

## Every reference-grid residue

`results/review2026/reference_phase/` retains all eight possible original-frame residues modulo8, the complete frozen835-query/659-reference plan, each835-row result, summary, strict verification and the small per-source residue ledger. Dense query extraction, original source clocks, fixed PCA, exact matcher and five-second crops remain unchanged. Each phase has its own calibration-only gate under the original74-negative1% empirical policy. All phases are reported; no preferred phase is selected and no primary result is overwritten.

Across the eight phases, candidate correct counts range445–449/543, accepted correct400–410, accepted wrong known IDs2–5 and unknown accepts0–1/71. Signature payload ranges2.4000–2.4015MB per audio hour. These finite phase ranges measure sensitivity; they are not confidence intervals. Paired dense-baseline source/montage intervals remain separately available.

```sh
python outputs/acr_repro/run_reference_phase.py --source-dir outputs/acr_repro \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --model work/benchmarks/acr_medium/pca_model.npz \
  --cache work/benchmarks/acr_cache_ffmpeg \
  --baseline-phase0 work/benchmarks/acr_medium/acr_d8_5s.json \
  --output work/reference_phase_fresh
python outputs/acr_repro/verify_reference_phase_execution.py --source-dir outputs/acr_repro \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --model work/benchmarks/acr_medium/pca_model.npz \
  --baseline-phase0 work/benchmarks/acr_medium/acr_d8_5s.json \
  --dense-baseline work/benchmarks/acr_medium/acr_d1_5s.json \
  --results work/reference_phase_fresh --output work/reference_phase_fresh/strict_validation.json
```

The recorded runner and strict verifier are byte-identical copies of the executed scripts. The strict execution verifier rereads and hashes all actual reference/query caches, binds reference caches to source IDs, recovers each original physical frame index, and checks every residue total. The portable checker can verify the resulting ledger and all scientific calculations without the unbundled caches; it does not claim to repeat that stronger cache read.

## Metadata-defined distortion strata

`distortion_strata.json` partitions all543 test-known queries into78 annotated pitch-altered,35 tempo-only and430 neither groups using the pinned PEX annotation schema. Other noise, reverb, echo, filters and mixed content can co-occur. Original global gates are unchanged across groups. Every one of the eight configurations has individual outcomes, group counts and source/montage/crossed intervals; the headline denominator remains543. For exact sparse ACR, candidate correct counts are5/78,34/35 and407/430 respectively. These are descriptive associations, not isolated causal attack effects or a license to exclude the difficult stratum.

`derive_distortion_strata.py` is the byte-identical diagnostic used for execution, with explicit input paths. Its optional upstream schema check requires the pinned toolkit checkout; the portable verifier supplies the exported protocol/predictions and verifies all group assignments/outcomes/intervals without importing that external checkout.

## Uncertainty and resource ranges

`paired_dense_sparse_review.json` reports all four paired known-query differences using unchanged exact dense and sparse predictions. Three dense-only and three sparse-only source matches explain the equal446/543 candidate totals. The crossed95% interval around their zero difference is approximately−1.65 to+1.66 percentage points; equality of totals is not proven equivalence. `memory_replica_ranges.json` derives medians/minima/maxima from every one of the three fresh-process runs per layout and gallery size, rather than reporting only a single run. These replica ranges are descriptive observations, not confidence bounds or fleet-scale extrapolations.

## Every frozen clean-unknown false acceptance

`results/review2026/unknown_waveform/` retains all16 previously accepted clean source-ID negatives from the expanded q4 test, their predeclared roster/offsets/digests, complete result statuses and an independent NumPy-FFT/direct-Pearson audit. The existing rule searches a complete five-second query against each sample lag within±0.25s of the frozen predicted offset and flags absolute Pearson correlation≥0.95. It performs no scale/tempo search, changes no gate, removes no query and changes no primary16/4000 false-accept count. Fifteen pairs have complete crop support: none flags, and the maximum correlation is0.153449. One pair lacks enough reference support for a full query crop and remains an explicit unsupported case. Absence of strong waveform overlap cannot prove distinct composition or recording identity.

After acquiring the original public assets and reproducing the expanded protocol/predictions/gate, run the byte-identical scripts with explicit paths:

```sh
python outputs/acr_repro/audit_acr_unknown_waveform.py --source-dir outputs/acr_repro \
  --protocol work/benchmarks/extended_unknown_protocol.json \
  --results work/benchmarks/acr_ivf_extended/acr_ivf_qs4_extended_5s.json \
  --gate work/benchmarks/acr_ivf_extended/threshold_qs4.json \
  --output work/unknown_waveform_fresh
python outputs/acr_repro/verify_acr_unknown_waveform.py --source-dir outputs/acr_repro \
  --plan work/unknown_waveform_fresh/plan.json --summary work/unknown_waveform_fresh/summary.json \
  --protocol work/benchmarks/extended_unknown_protocol.json \
  --results work/benchmarks/acr_ivf_extended/acr_ivf_qs4_extended_5s.json \
  --gate work/benchmarks/acr_ivf_extended/threshold_qs4.json \
  --output work/unknown_waveform_fresh/strict_validation.json
```

The independent verifier confirms the full-lag maxima and selected sample offsets through another FFT engine and direct centered dot products; the largest direct-Pearson discrepancy is approximately1.96×10^-15. It validates all16 encoded sources and decoded crop/search-window hashes. These are label-quality diagnostics and input audits, not adjusted identification results.

Private presentations, Drive notebooks, business-impact numbers, television audio, embeddings, raw PCM, indexes and external model weights remain outside the public artifact. New code uses the repository's existing MIT license; public dataset and upstream implementation terms remain as recorded in `THIRD_PARTY.md`.
