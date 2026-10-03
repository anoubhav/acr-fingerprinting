# Frozen real-acoustic transfer

`run_sdrr_frozen.py` evaluates the existing FMA-fitted PCA32 fingerprint on the public SD-RR-v1.0 loudspeaker-to-phone recordings. It does not fit a new representation. The canonical protocols are prepared by the dataset adapter, with verified public-file hashes, FFmpeg physical endpoints, source identities, creator groups, recording sessions, and publisher QC fields.

The native protocol contains 496 reference excerpts and 1,488 ten-second phone recordings. It is closed-set: report raw content recognition and content-plus-offset localization at the official 0.1-second tolerance, with 2-second sensitivity. It contains no native unknown population or calibrated rejection gate. A centered five-second crop is an explicitly separate adaptation; its reference start is shifted by 2.5 seconds.

The session-separated open adaptation enrolls 231 references. Calibration comprises 552 known and 627 unknown queries from one session; held-out evaluation comprises 141 known and 168 unknown queries from three other sessions. Calibration/test sources and sessions are disjoint. Unknown creators are absent from the gallery. Some known creators appear in both calibration and test; this is recorded in the protocol. The rejection threshold uses only calibration-unknown scores, targeting an empirical 1% calibration false-accept fraction with conservative handling of ties. This target is not a guarantee about held-out or device/room populations.

Run the frozen native controls:

```sh
python outputs/acr_repro/run_sdrr_frozen.py \
  --protocol work/benchmarks/acoustic_rr/sdrr_native_protocol.json \
  --model work/benchmarks/acr_medium/pca_model.npz \
  --frozen-ivf work/benchmarks/acr_ivf_study/ivf_factor8.index \
  --output work/benchmarks/acoustic_rr/acr_native \
  --durations 10,5 \
  --settings exact:8:1,ivf:8:4,exact:1:1 \
  --workers 2 --threads 2
```

A setting is `index:reference_factor:query_stride`. The exact sparse controls search every usable query frame; the cost setting preserves IVF512, nprobe 16, reference factor 8 and query stride 4. Its centroids come from the original FMA calibration-fit sources, not SD-RR. The dense exact control runs at five seconds. All matching uses the original top 5 neighbor voting and ±0.12-second consensus tolerance; that consensus tolerance is separate from the ground-truth localization criterion.

Run the session adaptation with the same model/settings:

```sh
python outputs/acr_repro/run_sdrr_frozen.py \
  --protocol work/benchmarks/acoustic_rr/sdrr_session_open_protocol.json \
  --output work/benchmarks/acoustic_rr/acr_session_open \
  --durations 5 --settings exact:8:1,ivf:8:4,exact:1:1 \
  --workers 2 --threads 2
```

Each gate file is written after calibration scoring and before any held-out query is scored. The representation, feature settings, index settings, and time-consensus rule remain fixed. There is no parameter selection from SD-RR test outcomes.

The primary cohort retains all 243 queries whose publisher QC expected rank is not 1. Outputs include QC and recording-session strata, source-cluster bootstrap intervals, and a declared sensitivity excluding the 30 queries from 10 source titles matching FMA metadata. This sensitivity changes only the reported query subset; it does not change the gallery or re-fit a model. Five of those sources also match the conservative neural-training candidate superset. Absence of title or encoded-byte equality does not prove absence of composition or transformed-recording exposure.

These runners may share a machine with other experiments. Their outputs make no latency claim; use the separate controlled common-cohort profiles for cost comparisons. Audio, large raw-feature caches, and index binaries are reconstruction inputs rather than public repository contents. Source audio licences are independent of the new code's MIT licence.

## Post-hoc channel-response diagnostic

`calibrate_sdrr_channel.py` tests a specific explanation for the frozen phone-transfer failure. It uses only the 552 known calibration phone/reference five-second pairs to estimate a single 32-band correction. For each pair, compute the log ratio of whole-crop mean mel magnitudes, flooring each magnitude at 1e-6; remove the across-band mean. Take the coordinatewise median across pairs, remove the resulting across-band mean again, and exponentiate. This gives geometric-mean-one gain coefficients. There is no clipping, smoothing, weighting, or band selection.

The fitted coefficients multiply query mel amplitudes before the original window means, gates, normalization, deltas, and FMA-fitted PCA. References, PCA weights, clocks, and exact factor 8/full-query voting stay unchanged. This is supervised query-frontend calibration; it is not an unchanged frozen-transfer result or a new confirmatory benchmark claim.

The numerical and cohort plan is saved before fitting. Its conservative helpfulness rule requires both raw and accepted calibration-known counts to strictly improve over frozen exact factor 8, using a separate 1% gate from the 627 calibration unknowns. Gains and that gate are written before any corrected held-out query is scored. If the calibration rule fails, the script retains a calibration-only null result and skips held-out scoring. It never reports a fitted version of the full 1,488 native queries as independent evaluation.

```sh
python outputs/acr_repro/calibrate_sdrr_channel.py --prepare-only
python outputs/acr_repro/calibrate_sdrr_channel.py
```

`qa_sdrr_transfer.py` separately checks ten source IDs chosen by a fixed metadata hash. It tests clean self-identification/clocks, fresh-versus-cached raw features, the independent Librosa formula oracle, and paired-signal/descriptor distances. Ground truth enters only these diagnostics, never the frozen retrieval rule.
