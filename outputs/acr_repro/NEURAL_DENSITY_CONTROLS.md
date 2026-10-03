# Frozen neural density controls

The matched-storage comparison changes the interpretation of the ACR result. The official NMFP checkpoint retains higher primary-source identification accuracy at a smaller signal payload than ACR factor8. ACR's distinct evidence concerns its small encoder, inexpensive dense feature extraction, deployment experience, and the measured joint tradeoff between identity, localization, rejection, and resources. Reference sampling alone is not a unique advantage.

## Declared controls

The uniform factors 2 and 4 were fixed before these held-out results were evaluated. All official weights, frontends, cached embeddings, source bounds, split roles, top20 frame candidates, exact CPU FlatIP retrieval, and within-track boundaries are unchanged. No retraining or oracle tempo is used. The two original native full-density rows remain the primary checkpoint baselines; the sparse rows are explicitly postprocessing controls.

NMFP samples both references and queries from the existing 0.5 s grids. Factor2 uses a 1 s hop and five 1 s windows across each 5 s query; factor4 uses a 2 s hop and three windows. Factor2 covers the full 5 s of audio; factor4's three windows cover 3 s across a 5 s span, leaving two 1 s gaps. Its native unit-scale sequence alignment then operates on matching physical hops. PeakNetFP samples references only and retains all nine query windows. Its published pairwise stretching estimator is expressed using actual query/reference timestamps and its physical scale range [0.5,2] remains unchanged. Dense query windows can map to repeated sparse reference frames; these are counted, rather than treated as independent evidence. This is a clock-aware adaptation of the native matcher for unequal hops.

Independent synthetic review found that PeakNet rounding in concatenated row coordinates could depend on unrelated earlier track lengths, even at native density when inverse scale2 creates half-grid ties. Rounding now uses each track's physical grid zero at every density. This data-independent invariance repair changes neither model parameters nor matcher settings. Native, sparse, exact, IVF, original, expanded, and acoustic results and gates were rerun consistently. Four regression tests cover source-identity and offset invariance under unrelated prefixes. Historical compatibility outputs are archived under `work/audit/`, outside the canonical result directories.

All original 835 queries and all 5,000 additional OOV queries were scored, against the same 659-reference gallery. The original gate is calibrated only on 74 distorted unknowns at a nominal 1% target; the expanded gate uses 1,074 unknowns at a nominal 0.1% target. Each gate is recalibrated from the declared calibration sources. The identical original/expanded NMFP thresholds arose from those order statistics, not from copying a threshold.

## Results

Counts below use the same 543 held-out known queries. Localization requires correct identity and an offset error at most 2 s. Signal storage is decimal MB per hour of physically decoded reference audio; compact logical metadata adds approximately 0.000468 MB/hour to each sparse row.

| Method/reference-query factors | Signal MB/hour | Raw correct | Raw localized | Correct accepted, original gate | Correct accepted, expanded gate | Localized accepted, expanded gate | Clean OOV false accepts /4,000 | Distorted OOV false accepts /71 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NMFP native r1/q1 | 3.675 | 466 | 443 | 448 | 445 | 425 | 5 | 2 |
| NMFP control r2/q2 | 1.839737 | 464 | 414 | 440 | 440 | 395 | 9 | 2 |
| NMFP control r4/q4 | 0.921613 | 458 | 338 | 381 | 381 | 289 | 8 | 1 |
| PeakNetFP native r1/q1 | 3.675 | 457 | 416 | 371 | 344 | 323 | 20 | 1 |
| PeakNetFP control r2/q1 | 1.839737 | 448 | 383 | 285 | 208 | 184 | 15 | 1 |
| PeakNetFP control r4/q1 | 0.921613 | 430 | 338 | 169 | 98 | 79 | 14 | 1 |

The expanded clean-OOV rates exceed the nominal 0.1% calibration target for every row here. These estimates are conditional on this gallery and unknown population; they do not establish fleet false-accept behavior. The 71 distorted unknowns should be reported separately from the 4,000 clean unknown sources.

NMFP factor2 loses two net identity hits but 29 net localized hits relative to its native grid. Factor4 loses eight identity hits and 105 localized hits. Paired counts retain both directions: factor2 loses/gains 3/1 identity hits and 32/3 localized hits; factor4 loses/gains 12/4 identity hits and 109/4 localized hits. This is direct evidence that identity can survive thinning while offset quality deteriorates. It does not prove a representation-specific causal mechanism.

Against exact ACR factor8, NMFP factor2 has 18 more identity hits and 35 more accepted-correct hits at the original gate. The paired ACR-minus-NMFP differences are -3.31 percentage points (95% crossed source/montage interval [-6.61,-0.20]) for identity and -6.45 [-11.00,-2.31] for accepted correctness. ACR has 14 more raw localized hits and 2 more accepted-localized hits, but both corresponding intervals include zero. On the 389-query conservative source-unseen slice, NMFP factor2 identifies 336 and accepts 318, versus ACR 324/294; the accepted difference remains -6.17 points [-11.87,-0.82]. These are paired protocol comparisons with separately calibrated gates, not equal-population-FPR guarantees.

PeakNet factor2 usually maps nine query frames to five unique reference frames (390/543 known queries); factor4 maps to three frames on 511/543. Its lower accepted recall illustrates why repeated query/reference alignments and calibration must be measured in addition to raw identity. Native scale/score rules and their data-independent clock repairs are retained; no new confidence correction was tuned on test data.

## Verification and accounting

`verify_neural_density.py` compares all 835 original 5 s rows against the current corrected native adapters, with zero maximum score error, and records historical compatibility differences separately. Nine random-vector clock tests check localization at reference hops 0.5/1/2 s and physical stretch 0.75/1/1.5; a deliberately incorrect hop is rejected. `test_peaknet_sparse_clocks.py` verifies native and sparse catalog-order invariance independently of benchmark outcomes. The sparse runner validates every saved grid and query crop against the frozen protocol; declarations include runner and adapter digests so changed code cannot silently reuse old results.

Factor2 stores 151,870 fingerprints, 77,757,440 signal bytes and 19,778 compact metadata bytes. Factor4 stores 76,079 fingerprints, 38,952,448 signal bytes and the same metadata. Metadata consists of int64 track boundaries, UTF8 IDs, a shared float64 hop, and per-track float64 origins. An explicit timestamp audit copy is retained in the actual process and counted separately; it is exactly reconstructible from the regular grids. FAISS search vectors and the reranking copy are separately reported as resident vector bytes. These are logical and serialized measures, not a process-RSS measurement.

The cached-embedding prediction/density-control study performs no fresh neural inference and makes no latency claim. The separate final common-cohort profile below executes only retained windows before fresh official neural inference; its measured sparse encoder costs are 52.76 ms and 42.01 ms at the two reference/query factors. Cached dense inference timings are not current sparse costs. Neither study makes a TV-device or energy claim.

A separate quiet common-cohort profile subsequently measures actual sparse NMFP encoding. It selects only retained 1 s windows before the unchanged official mel frontend and network. On the same 50 calibration-known 5 s inputs, four CPU threads, two warmups and five repeats, the distribution of per-query repeat medians is: r2/q2 encoder 52.76/53.94 ms, exact matching 4.36/4.69 ms and directly timed pipeline 57.11/58.14 ms (p50/p95); r4/q4 encoder 42.01/42.82 ms, matching 1.41/1.50 ms and pipeline 43.37/44.25 ms. Dense-cache embedding parity over all 50 inputs has maximum absolute error 2.24e-7 and 1.81e-5 respectively. Startup and the first actual model shape are separate; the first five-window shape takes 20.02 s cold. This measures sparse inference rather than encoding nine windows and discarding outputs. It makes no TV-device or energy claim. Records and ambient CPU snapshots are under `work/profiling/nmfp_density/`.

## Reproduction

Run in the standard benchmark environment; TensorFlow is unnecessary for cached matching:

```sh
work/venv/bin/python outputs/acr_repro/verify_neural_density.py
work/venv/bin/python outputs/acr_repro/compare_neural_density.py
work/venv/bin/python outputs/acr_repro/neural_density_controls.py --model nmfp --protocol work/benchmarks/extended_unknown_protocol.json --primary-protocol work/benchmarks/hard_medium_protocol.json --reference-cache work/benchmarks/nmfp_cache/cache_index.json --query-cache work/benchmarks/nmfp_cache/extended_query_cache_index.json --output-dir work/benchmarks/neural_density_controls/nmfp
work/venv/bin/python outputs/acr_repro/neural_density_controls.py --model peaknet --protocol work/benchmarks/extended_unknown_protocol.json --primary-protocol work/benchmarks/hard_medium_protocol.json --reference-cache work/benchmarks/peaknet_cache/cache_index.json --query-cache work/benchmarks/peaknet_cache/extended_query_cache_index.json --output-dir work/benchmarks/neural_density_controls/peaknet
```

Full per-query results, original/expanded summaries, before-evaluation declarations, protocol/cache hashes, physical query evidence, paired outcomes, and a source/result SHA256 ledger are saved under `work/benchmarks/neural_density_controls/`. Official model provenance and checksums remain in each result's `config` field and in `NEURAL_BASELINE.md` and `PEAKNET_BASELINE.md`.

## Closest prior art

Plapous, Berrani, Besset and Rault's *A low-complexity audio fingerprinting technique for embedded applications* appeared online in 2017 and in *Multimedia Tools and Applications* 77:5929–5948 (2018), DOI [10.1007/s11042-017-4505-4](https://doi.org/10.1007/s11042-017-4505-4). Its [Springer-provided first-page preview](https://www.researchgate.net/publication/314088262_A_low-complexity_audio_fingerprinting_technique_for_embedded_applications) identifies the same TV synchronization/embedded-device use case, with music and smartphone evaluation. Published figure captions show reference factors2 and10. The primary [Now Playing paper](https://storage.googleapis.com/gweb-research2023-media/pubtools/4142.pdf) describes its optimized IIR filter bank replacing FFTs. The full Plapous text was reviewed from the user-provided copy for literature context. No private copy is redistributed, and its embedded IIR system was not reimplemented as a baseline or assigned directly comparable accuracies.

Schreiber and Müller, [*Accelerating Index-Based Audio Identification*](https://doi.org/10.1109/TMM.2014.2318517), IEEE TMM16(6):1654–1664 (2014), explicitly connects temporal hash correlation with reliability and describes reference subsampling for10× storage reduction in its [institutional abstract](https://publica.fraunhofer.de/entities/publication/b26d80fd-84ca-4b1e-b07d-7d70a124f3ce). Full publisher text remains inaccessible. Both works preclude broad novelty claims about low-cost embedded audio recognition or temporal redundancy enabling sparse references. Their actual algorithms differ from the continuous PCA representation, and their published accuracies are not directly comparable to this protocol.
