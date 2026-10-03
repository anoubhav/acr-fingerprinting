# Published small neural comparator: PeakNetFP

[PeakNetFP](https://arxiv.org/abs/2506.21086) (Cortès-Sebastià, Martin, Molina, Serra, Hennequin; ISMIR 2025) uses sparse spectral peaks with PointNet++ to produce 128-D unit-norm fingerprints. The loaded model has **169,296 parameters**. The published checkpoint is used without training or conversion.

Source is pinned to [the official repository](https://github.com/guillemcortes/peaknetfp) at `d55071a644f20fd6448fb8ed33c0c115769fb7a5`. Checkpoints come from [Zenodo record 15782389](https://zenodo.org/records/15782389), archive `peaknetfp_checkpoints.tar.gz`, 155,440,468 bytes, MD5 `df23e1556a206fd72bfa99d0724d241f`. This checksum was verified against the record API before extraction. The archive contains both a NeuralFP model and PeakNetFP; this adapter selects **peaknetfp/ckpt-100**.

The author-supplied `config/peaknetfp.yaml` sets mono 8 kHz, 1 s windows at 0.5 s hop, a 1024/256 STFT, 256 mel bands from 300 to 4000 Hz, and relative magnitude normalization. It selects 256 amplitude-ordered peaks, normalizes their frequency/time coordinates, tiles when fewer peaks are available, and uses the published top-n/query-ball PointNet++ architecture. These settings, config/source hashes, checkpoint hashes and package versions are recorded with each cache.

The model classes and peak extraction function bodies are used unchanged. Peak function definitions are loaded directly from `model/trainer.py` without importing its unrelated training/optimizer dependencies. The mel frontend uses the original Kapre 0.3.7 source tag (`ae8f5077ec9071e5529992702885de03c3321941`). Strict TensorFlow restoration asserts that every built model variable matches the checkpoint. Reference extraction uses fixed 125-window batches, with duplicate-window padding removed after inference. Queries use their actual window count; a batch-parity check verifies that their outputs match reference batch outputs within 2e-5. Source decoding and crop rounding follow the common FFmpeg benchmark convention.

Its native matcher estimates each candidate track's time-scaling factor from query/reference nearest-frame correspondences, then reranks aligned sequences using their mean cosine. This follows `eval/eval_faiss.py`, including its mean pairwise-ratio estimate and 0.5–2 allowed factor range. The adapter restricts complete sequences to one reference track; the original script bounds against the concatenated database. Ground-truth tempo labels are not passed to matching.

## Reproduction

Use a separate Python 3.12 environment with `requirements-peaknet.txt`. Clone the official source at the pinned commit and extract the verified checkpoint archive under `work/models/peaknetfp/`. The default model directory is `work/models/peaknetfp/peaknetfp_checkpoints/peaknetfp`.

```sh
work/venv_peaknet/bin/python outputs/acr_repro/peaknet_baseline.py \
  --manifest work/benchmarks/hard_medium_protocol.json \
  --cache-dir work/benchmarks/peaknet_cache \
  --batch-size 125 --threads 4 --durations 5

work/venv/bin/python outputs/acr_repro/run_nmfp_protocol.py \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --reference-cache work/benchmarks/peaknet_cache/cache_index.json \
  --query-cache work/benchmarks/peaknet_cache/cache_index.json \
  --output-dir work/benchmarks/peaknet_primary_results \
  --index-type peaknet --durations 5 --threads 4
```

The shared extraction and result schemas are documented in `NEURAL_BASELINE.md`. PeakNetFP's warmed mel/peak/model computation is one compiled graph, so its recorded `combined_frontend_forward_s` must be reported as a combined cost; a zero placeholder in `frontend_s` does not mean that frontend computation is free. These desktop measurements do not establish TV-device suitability or energy consumption. The model was trained on FMA-derived music with a particular augmentation objective; retain the conservative training-source sensitivity slice and avoid substituting its published accuracy on a different dataset.
