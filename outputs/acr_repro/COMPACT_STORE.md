# Compact ACR reference storage

`compact_streaming_index.py` is an optional server-store implementation. It preserves the existing fingerprint, query policy, nearest-neighbor settings and temporal-vote rule. It does not change `Fingerprinter` or `ExactIndex` defaults.

The store keeps reference vectors once, inside FAISS. Each row needs only a uint32 original physical frame index. Int64 recording boundaries and the content-ID lookup recover row identity; frame indices recover the original float64 clock, including gaps from silence filtering. The matcher delegates to the existing `ExactIndex.match` implementation. It requires sequential FAISS row IDs, as returned by standard `add` on an initially empty L2 Flat/IVFFlat index. Inner-product indices are outside this matcher’s contract.

Enrollment reads one cached recording at a time and selects the original reference grid before row-wise PCA projection. This eliminates materialized dense-gallery projection workspaces, duplicate Python vectors, the unused norm cache, and per-row content codes/float64 timestamps. It does not compress vectors or train different centroids.

```python
import json
from pathlib import Path
import faiss
from compact_streaming_index import CompactTemporalIndex, streaming_enroll
from acr_fp import Fingerprinter

model = Fingerprinter.load("work/benchmarks/acr_medium/pca_model.npz")
protocol = json.loads(Path("work/benchmarks/hard_medium_protocol.json").read_text())
gallery = [r for r in protocol["references"]
           if r["role"] in ("calibration_known", "test_known")]
cache = Path("work/benchmarks/acr_cache_ffmpeg")
centroids = faiss.read_index("work/benchmarks/acr_ivf_study/ivf_factor8.index")
centroids.reset()
centroids.set_direct_map_type(faiss.DirectMap.NoMap)
centroids.nprobe = 16
index, descriptor_sha256 = streaming_enroll(gallery, model, cache, centroids, factor=8)
faiss.write_index(centroids, "compact.index")
index.save_metadata("compact_clock.npz")
loaded = CompactTemporalIndex.load("compact.index", "compact_clock.npz", nprobe=16)
```

The reusable module is colocated with the existing library. Scripts resolve their workspace/repository root from `outputs/acr_repro/` and accept input/output paths as CLI arguments. The repository preserves that directory layout. Source audio, feature caches, PCM profiling arrays, and large FAISS indices are reconstruction inputs and should not be committed.

Run the four conformance tests and frozen experiment:

```sh
python -m unittest discover -s outputs/acr_repro -p test_compact_streaming.py -v
python outputs/acr_repro/run_streaming_index_study.py \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --model work/benchmarks/acr_medium/pca_model.npz \
  --cache work/benchmarks/acr_cache_ffmpeg \
  --frozen-index work/benchmarks/acr_ivf_study/ivf_factor8.index \
  --inputs work/profiling/common_inputs/inputs.json \
  --output work/benchmarks/compact_store
python outputs/acr_repro/check_compact_full_parity.py
python outputs/acr_repro/summarize_streaming_study.py
```

The study uses three counterbalanced fresh processes for each of two layouts and four nested 141/256/512/659-source galleries. Every gallery retains all 141 calibration-known sources; the additional sources are fixed by ID hashes. Reference/query settings and FMA PCA/IVF centroids remain frozen. Fifty calibration-known queries are common to every gallery, with two matcher warmups and five repeats. No test outcome chooses a setting. Input/model/index and source-code hashes prevent stale result reuse; choose a fresh output directory after changing the experiment. The separate 835-query check compares full top-ten candidates and historical scores/offsets/rejection decisions; all match exactly in the measured study.

At 659 sources, the experiment observes 222.106→111.095 MB logical explicit reference state and 1,486.8→419.0 MB median post-enrollment RSS. Enrollment scope is cached-feature reads, PCA projection, metadata and index population; centroid fitting and audio/frontend work are excluded. RSS includes allocator-retained enrollment workspace and runtime overhead and is not a portable per-entry memory constant. Logical state counts serialized IVF plus explicit metadata/arrays, not allocated inverted-list capacity. This is a within-implementation comparison, not a memory frontier against optimized neural or hash stores. Matching did not improve: medians were 4.120→4.218 ms on the controlled query cohort.

`profile_common_compact_acr.py` separately measures the **actual complete public encoder→compact matcher** on the fixed 50 five-second PCM inputs. Model/index loading, decoding, sample-rate conversion, and I/O are outside the timer. Query thinning, search and vote aggregation are inside the matcher timer. The pipeline is measured directly for every repeat; it is not a sum of separately aggregated stage medians. Use only a coordinated quiet slot, four FAISS threads, two warmups and five repeats.

```sh
python outputs/acr_repro/profile_common_compact_acr.py \
  --inputs work/profiling/common_inputs/inputs.json \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --model work/benchmarks/acr_medium/pca_model.npz \
  --index work/benchmarks/compact_store/compact_659_r0.index \
  --metadata work/benchmarks/compact_store/compact_659_metadata.npz \
  --output work/profiling/acr_compact/profile.json \
  --threads 4 --warmups 2 --repeats 5
```

The old materialized-store pipeline measurement remains an old-layout observation. It cannot be relabeled as a compact-store measurement. These public full-encoder profiles also do not measure the deployed SDK's device budget: the documented SDK payload is 32 raw band means, while the public complete encoder additionally performs normalization, deltas and PCA.
