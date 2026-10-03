# Official NMFP-Triplet baseline

This adapter uses the authors' pretrained NMFP-Triplet model at epoch 100. It does not train, convert the checkpoint, reduce its dimensions, or change the learned network or mel frontend. NMFP is the ISMIR 2025 method [Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for Music Identification](https://arxiv.org/abs/2506.22661).

The source revision is `15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc` from [the official repository](https://github.com/raraz15/neural-music-fp). The [official Zenodo checkpoint archive](https://zenodo.org/records/15719945) is 172,872,470 bytes, with MD5 `ee8a3358fc5e5cdd09d6d2245d395021` verified against Zenodo's record API. Every cache records the config and checkpoint SHA-256 values. Before inference, all model variables are built and checkpoint restoration must pass TensorFlow's `assert_existing_objects_matched` check.

The model takes 1 s segments at 8 kHz, with a 0.5 s hop, 256 mel bands from 160 to 4000 Hz,1024/256 STFT window/hop, and an 80 dB dynamic range. It emits 128-D unit-norm float32 fingerprints. The actual packaged config is used, rather than the repository's example training config.

The tested runtime is Python 3.12 on Apple arm64, with TensorFlow 2.16.2, legacy Keras 2.16, and Essentia 2.1b6.dev1177. These versions differ from the original training environment, but the official model and mel code are imported unchanged. Inference uses float32; the checkpoint used mixed-float16 training. The precision is configurable and recorded. FFmpeg mono 8 kHz float32 decoding is shared with other benchmark methods for consistent physical source timing. This intentionally replaces the upstream Essentia MonoLoader decoder; it does not replace Essentia's mel frontend.

## Reproduction

Install the requirements in a separate environment to keep TensorFlow's NumPy 1.26 requirement from changing the primary benchmark environment:

```sh
python3.12 -m venv work/venv_nmfp
work/venv_nmfp/bin/python -m pip install -r outputs/acr_repro/requirements-neural.txt
git clone https://github.com/raraz15/neural-music-fp.git work/upstream/neural-music-fp
git -C work/upstream/neural-music-fp checkout 15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc
mkdir -p work/models/nmfp
curl -L --fail 'https://zenodo.org/records/15719945/files/nmfp-triplet.zip?download=1' -o work/models/nmfp/nmfp-triplet.zip
```

Verify the archive MD5 before unpacking, then unpack it inside `work/models/nmfp/`. The resulting model directory should contain `config.yaml`, `checkpoint`, `ckpt-100.index`, and `ckpt-100.data-00000-of-00001`.

```sh
work/venv_nmfp/bin/python outputs/acr_repro/neural_baseline.py \
  --manifest work/benchmarks/hard_medium_protocol.json \
  --cache-dir work/benchmarks/nmfp_cache \
  --batch-size 256 --threads 4 --durations 5 1 2 3 10
```

Reference and query phases default to one `cache_index.json`. For parallel extraction, use `--references-only` with that default and `--queries-only --cache-index-name query_cache_index.json` to prevent two processes from writing the same index. The evaluation script accepts those two index files separately. Per-file NPZ records can be read by `load_cached_embeddings` without importing TensorFlow. The index maps each reference ID and `query_id@duration` to its cache path and timing record. Caches include the source hash, crop bounds, decoder, model, precision, batch and thread settings, and reject incompatible reuse. Source PCM is cached briefly to avoid decoding the same query montage repeatedly; timing records distinguish these hits. Initialization warms the initial one-window graph, while the first use of a different batch shape can still incur a trace; the first actual nine-window query incurred a separately recorded cold trace. The common-input steady profile warms the actual nine-window shape before measurement. These desktop extraction timings are not smart-TV measurements.

`NMFPSequenceIndex` implements the native equal-hop full-density retrieval protocol: top 20 frame neighbors with exact inner-product search, candidate offsets implied by their position in the query, followed by mean corresponding cosine reranking. It corrects the upstream retrieval script's last-track boundary exclusion. Sparse reference matching should be evaluated separately using the shared matcher; full-density native retrieval prevents attributing a weaker generic matcher to the pretrained network.

## Exposure audit

NMFP was trained on FMA audio, and PEX is derived from FMA. The authors document training audio drawn entirely from FMA_medium, but this checkout does not include the exact train 10k IDs. `nmfp_overlap_audit.py` uses checksum-verified official FMA metadata to define a conservative sensitivity set excluding **every** FMA_small/medium source. Membership in that superset means possible exposure, not proven exposure. The remaining FMA-large/full-only source IDs are unseen under the documented training protocol. Report this slice alongside the complete declared test set.

Successful checkpoint restoration and full-set execution establish a valid pretrained baseline. They do not mean that the original published benchmark accuracy has been replicated on a different PEX task. No published accuracy should be substituted for the results of this run. `run_nmfp_protocol.py` writes predictions compatible with the shared summary script, and its `--reference-cache` / `--query-cache` arguments identify the two completed cache indexes. Upstream code, weights, and dataset licenses remain governed by their respective releases; this adapter does not bundle those assets.
