# Recovered ACR fingerprint and reproduction choices

This implementation reconstructs the algorithm from the supplied notebooks. It is not the production implementation, the original PCA matrix, or a reproduction of the proprietary accuracy tables. Parameters below come from `4.Obtain_mels.ipynb` (`MelFingerprintsFromAudioWav`, code cell 1) and its duplicate in `1_merged_v3_for_channel.ipynb` (code cell 2). The source files were inspected as data; their filesystem mutations and proprietary CLI calls were not run.

## Exact settings recovered from source

| Setting | Notebook value | Public reconstruction |
|---|---:|---|
| Mono sample rate | 8,000 Hz | 8,000 Hz |
| STFT FFT length | 1,024 samples | 1,024 |
| STFT hop | 186 samples | 186 |
| Mel bands | 32 | 32 |
| Lower / upper frequency | 315 / 1,960 Hz | 315 / 1,960 |
| Spectrogram exponent | 1, magnitude | 1 |
| STFT `center` | `False` | `False` |
| Explicit audio padding | 512 zeros at both ends | same |
| Fingerprint averaging window | 32 STFT frames | 32 |
| Dense stride | `skip + 1 = 1` frame | 1 |
| Non-silence condition | band std > 10⁻⁶ and band mean > 10⁻³ | same |
| Standardization | per vector, population standard deviation | same, with singular-vector guard |
| Frequency deltas | adjacent differences of the unstandardized band means | same |
| Delta standardization | separately from the band-mean vector | same |
| Pre-PCA dimension | 32 + 31 = 63 | 63 |
| PCA output dimension | 32; external FAISS transform loaded | new PCA32 fitted on disjoint calibration content |
| Explicit fp16 conversion | absent from recovered extraction class | disabled by default; separate ablation |

The original paper includes an fp32-to-fp16 stage. The recovered extraction class does not implement it. Both versions are available through `quantize_fp16=False/True`; the main notebook-faithful run uses `False`. This stage rounds pre-PCA features to fp16 and converts back to fp32 for PCA. It does not make the default L2 index use fp16. Optional fp16 index storage is a distinct precision choice.

The notebooks use librosa defaults for the STFT window and mel filterbank. We make these defaults explicit: a periodic Hann window, Slaney frequency mapping with area normalization, and linear magnitude amplitudes without a logarithm. The current implementation was compared numerically with an independent librosa calculation in the tests. Tiny differences arise from using float64 accumulation rather than historical float32 summation. FFT scratch is processed in blocks of 2,048 frames rather than allocating a whole long-track FFT matrix. Public benchmark files use a common FFmpeg decoder/downmixer/resampler, with the backend version and revision in the cache key and result manifest. Standalone array inputs at a different sample rate use anti-aliased `scipy.signal.resample_poly`; the historical librosa resampling backend/version is unknown. The decoder audit found one reference with a 1.738-second FFmpeg/soundfile duration difference, so the backend was standardized before matching outcomes were produced. Soundfile is a fallback only when the FFmpeg executable is unavailable. FFmpeg's default stereo downmix uses approximately 0.707 per channel, whereas the standalone array API and historical librosa loading use an arithmetic mean; this gain difference is removed by standardization except for the absolute silence gates. It is disclosed as a public decoder choice, not claimed as bitwise historical loading equivalence.

## Formula and physical clock

For mel amplitudes `M[t,b]`, window start `s`, and window length `W=32`, the band-mean vector is `a[b] = mean(M[s:s+W,b])`. Remove the row if its across-band mean or population standard deviation fails the thresholds above. Form `d[b] = a[b+1] - a[b]`; independently standardize `a` and `d` and concatenate them. No temporal derivative is used. PCA is ordinary centered, unwhitened projection. Eigenvector signs are fixed for deterministic serialization; the missing production PCA matrix is not inferred.

The dense nominal fingerprint rate is `8000/186 = 43.010752688` per second. The 32-frame physical support is `(1024 + 31×186)/8000 = 0.84875` seconds. After correcting for the explicit leading padding, the center of window `s` is `0.360375 + s×186/8000` seconds relative to the original audio. Returned times retain the original `s` even after silence removal. The notebooks' `8000 // 186 = 43` is a coarse rate estimate; using it to infer long-duration offsets would accumulate drift.

For source-faithful decimation, pass `grid_step_sec=config.hop_length*config.stride_frames/config.sample_rate` and `grid_origin_sec=config.first_center_seconds` to `ExactIndex.from_entries`. It selects grid indices whose remainder modulo the factor is the chosen phase, preserving the source's order of skipping before silence removal. Without these arguments, the generic index API decimates surviving rows. Both meanings are explicit; benchmark comparisons should use the grid version. Factors 1, 2, 4, 6, 8 correspond to the paper's skip rates 0, 1, 3, 5, 7 when the dense stride is one. Stride2 plus factor8 stores one in sixteen original grid frames.

## PCA calibration and matching are new evaluation choices

Call `fit` or `fit_audio` on calibration content whose identifiers are disjoint from all reference and evaluation content. The serialized model records its calibration identifiers and row count. `assert_disjoint` checks the boundary but relies on truthful identifiers supplied by the caller. No query normalization statistics are learned across the test set.

`ExactIndex` performs exhaustive squared Euclidean search with FAISS `IndexFlatL2` when available and a bounded-memory NumPy fallback otherwise. Exhaustive search removes ANN approximation and CPU/GPU differences from the representation comparison. It does not replicate the original paper's tuned GPU IVF index.

The original paper says to use a minimum vote count and approximately ordered reference times, without providing code or thresholds. The public matcher therefore uses a disclosed reconstruction: search the top 5 neighbours per query fingerprint; find a common content and a reference-minus-query offset interval of width `2×tolerance`; count each query row once; rank contents by vote count then mean squared L2. Its default tolerance is ±0.12 seconds, large enough for half of an 8× reference-grid step (`0.093` seconds). The returned offset is constrained so all selected votes satisfy that tolerance. `score = vote_fraction/(1+mean_squared_L2)` is a heuristic used for held-out threshold calibration, not a probability. No false-positive claim is implied by a high score. Thresholds, neighbourhood count, and temporal tolerance must be frozen before test evaluation.

## Storage accounting and ablations

Default PCA32 fingerprints take 128 bytes as fp32, or 64 bytes as fp16 storage. The array metadata is another 12 bytes per row: int32 content code and float64 center time. Nominal dense fp32 payload is 19,819,354.84 bytes/hour; including row metadata, 21,677,419.35 bytes/hour. Divide by the decimation factor for the asymptotic budget. Actual counts change with clip boundaries, silence removal, and decimation phase. `bytes()` reports payload, row metadata, content-name lookup, and separate search/FAISS cache copies; compressed file size is a different quantity.

The prespecified `ablation_configs()` includes no standardization, no deltas, fp16 roundtrip, no PCA, PCA16/32, window16/48, and stride2. A 64-dimensional PCA cannot be fitted to this source's 63-dimensional input. `mel64_pca64` is an explicitly different 64-band representation, not a one-change PCA ablation. The earlier exploratory plotting notebook did use 64 mel bands and PCA64; it is not the recovered final extraction class.

The unit tests cover pre-PCA agreement with librosa and notebook formulas, gain invariance, silence filtering without clock compression, source skip order, serialization and split checks, exact L2 distances, storage accounting, and arbitrary-offset temporal matching across all five decimation factors. Synthetic tests validate implementation behavior; they are not benchmark evidence.
