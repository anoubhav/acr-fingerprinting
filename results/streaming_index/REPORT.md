# Compact reference storage and streaming enrollment

The frozen ACR system can halve its explicit reference state while preserving every identification decision. The system change selects the reference grid before the row-wise PCA projection, enrolls one recording at a time, and retains one vector copy inside FAISS. An unsigned 32-bit original frame index and per-recording boundaries replace per-row float64 timestamps and content codes. Search and temporal voting are unchanged.

This is a within-implementation system result. It does not establish a compactness frontier against optimized neural or hash baselines.

## Frozen experiment

The original 659-source music gallery is unchanged. Nested 141/256/512/659-source galleries retain all 141 calibration-known sources; additional original known-test references are included by SHA256(`acr-memory-scaling-v1:` + source ID), retaining the canonical source order. No unknown source is enrolled. The existing FMA PCA32 model, IVF512 centroids, reference factor 8, query stride 4, nprobe 16, top 5 neighbors, and 0.12 s temporal tolerance remain fixed.

Fifty existing calibration-known five-second PCM queries are present in every gallery. Each layout/gallery uses three fresh processes, counterbalanced baseline→compact / compact→baseline / baseline→compact. FAISS uses four CPU threads. Each query has two matching warmups and five repeats. Per-query repeat medians form a cohort median and95th percentile; the table reports the median of those statistics over the three processes.

Enrollment time includes cached raw-feature reads, PCA projection, metadata, and index population. Audio decoding, frontend generation, frozen-centroid training, and serialization are excluded from the reported enrollment time. Both layouts use the same cached files. This is not a claim about end-to-end cold-media enrollment throughput.

All foreground benchmark jobs were held during the 24 workers. Background desktop activity was recorded per process. The screen-recording service used approximately 91% of one core consistently; the same host conditions apply to both counterbalanced layouts. Host: Apple M2 Max, 12 cores, 32 GB; macOS; Python 3.12; FAISS 1.15.1; NumPy 2.5.3.

## Measurements

MB below means decimal million bytes.

| Gallery | Hours | Entries | Materialized RSS MB | Compact RSS MB | Materialized explicit MB | Compact explicit MB | Enrollment s (materialized / compact) | Match median ms (materialized / compact) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|141|9.32|178,193|432.8|218.5|49.965|25.019|1.275 /1.150|2.787 /2.784|
|256|15.84|302,050|680.9|252.0|84.645|42.360|2.149 /1.937|3.071 /3.120|
|512|32.15|599,949|1,150.1|318.5|168.059|84.070|4.218 /3.839|3.646 /3.816|
|659|42.27|792,973|1,486.8|419.0|222.106|111.095|5.190 /5.067|4.120 /4.218|

At the full gallery, explicit reference state decreases 50.0%; post-enrollment process RSS decreases 71.8%. Enrollment peak RSS medians are 1,571.9 MB and 419.0 MB, a 73.3% reduction. Full-gallery matching95th percentiles are 4.954 ms and 5.114 ms. Matching latency does not improve; the compact metadata lookup adds a small observed cost.

“Explicit state” is serialized IVF size plus retained reference arrays, not total allocated FAISS capacity. Both layouts serialize the same 107,914,099-byte IVF index. Materialized state additionally retains 101,500,544 vector bytes, 9,515,676 code/clock bytes, 3,171,892 norm-cache bytes, and 3,954 UTF-8 content-ID bytes. Compact state retains 3,171,892 frame-index bytes, 5,272 boundary bytes, and 3,954 UTF-8 content ID bytes. Python object/container overhead is not included in this logical accounting. Actual process RSS includes runtime/allocator overhead and freed but allocator-retained enrollment workspace. On macOS, `getrusage().ru_maxrss` is bytes, not kilobytes. RSS is measured after Python garbage collection and is specific to this process/platform; it is not a portable per-entry memory constant.

## Exact conformance

Reference descriptor SHA256 hashes match byte-for-byte at all four gallery sizes and across all three repetitions. Every timed calibration query has identical candidate ID, votes, score, and offset.

The full original 835-query cohort was additionally checked with independently saved materialized codes/float64 clocks. Every top-ten candidate dictionary is exactly identical between layouts. All previously frozen top-one IDs, scores, offsets, and rejection decisions also match exactly. This uses the historic 0.05657886749843364 threshold only to check conformance; there is no recalibration or test-based setting selection. Preservation of identical scores also preserves decisions at the previously frozen expanded-negative threshold.

Four synthetic conformance tests pass: track boundaries including empty tracks, bit-exact physical clocks through silence gaps, neighbor/vote equivalence, and invalid metadata rejection. The unchanged main library’s 15 tests remain independently relevant; this experiment does not alter that library.

## Reproduce and inspect

Run from the original workspace, with the existing verified raw-feature cache and frozen PCA/IVF artifacts:

```sh
work/venv/bin/python -m unittest discover -s work/revision2026/outputs/acr_repro -p test_compact_streaming.py -v
work/venv/bin/python work/revision2026/outputs/acr_repro/run_streaming_index_study.py
work/venv/bin/python work/revision2026/outputs/acr_repro/check_compact_full_parity.py
work/venv/bin/python work/revision2026/outputs/acr_repro/summarize_streaming_study.py
```

`publication_summary.json` is a path-free summary with frozen protocol/model/index/code SHA256 values and all gallery IDs. `full835_parity.json` contains individual conformance records. Each `*_r*.json` contains fresh-process timing/RSS measurements, all raw query repeats, and background CPU activity. `streaming_store.pdf` is a vector scaling figure. The `.index` files are reconstruction scratch artifacts and should not be distributed with the small public result package.
