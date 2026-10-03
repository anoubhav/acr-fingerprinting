# Frozen neural fingerprints on actual smartphone re-recordings

The public checkpoints behave differently on this capture set. NMFP is substantially stronger than PeakNetFP, and its advantage over frozen ACR also survives real microphone capture. These findings restrict any broad robustness claim for the inexpensive PCA encoder. They do not establish performance on a population of rooms, phones, or connected-TV deployments.

## Protocols and unchanged models

The native SD-RR release has 496 clean reference recordings and 1,488 actual microphone queries, three 10 s excerpts per source. All released queries remain included, including the 243 whose upstream quality check did not rank the expected query first. The capture population uses one laptop speaker, one phone, and four sessions; room labels are unavailable. Audio is decoded through the same FFmpeg mono8k float32 path used by the other methods. Physical reference endpoints and native 19-frame sequence support are valid for every query.

Native 10 s recognition is closed-set and receives no invented rejection gate. Centered 5 s crops are a separate adaptation. The session-open adaptation enrolls 231 references, calibrates on 552 known and 627 unknown clips from one capture session, and tests 141 known and 168 unknown clips from the other three sessions. Test sources are disjoint from calibration sources, and unknown creators are absent from the gallery. Fifty-one creator groups occur across calibration and test, so this is not a creator-disjoint evaluation.

The official NMFP-Triplet and PeakNetFP checkpoints, mel frontends, preprocessing configurations, and 1 s embedding windows are unchanged. Both use exact FlatIP frame candidates, top20, and mean aligned cosine. NMFP retains its unit time scale; PeakNet estimates scale from candidate correspondences without annotated tempo. Its per-track boundary and rounding repairs prevent crossing recordings or depending on unrelated concatenated row parity. Original compatibility outputs remain under `work/audit/`.

Reference density factors 2 and 4 carry the same choices declared for PEX. NMFP also thins the query grid; PeakNet retains its dense query and uses actual physical timestamps against sparse references. These are disclosed clock-aware postprocessing controls, not newly trained models or settings selected from acoustic test labels.

## Complete results

Each native count uses all 1,488 known queries. Each session-open count uses the same 141 held-out known and 168 held-out unknown queries. Correct localization also requires the returned source identity. Acceptance uses only the 627 declared unknown calibration scores, with an empirical 1% target and six allowed calibration errors.

| Method | Study/input | Ref/query factor | Correct identity | Correct within0.1s | Correct within2s | Accepted correct | Accepted within2s | Accepted wrong known | Unknown accepts |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| NMFP | Native10s | r1/q1 | 1,073 | 1,026 | 1,027 | n/a | n/a | n/a | n/a |
| NMFP | Centered5s | r1/q1 | 921 | 852 | 855 | n/a | n/a | n/a | n/a |
| NMFP | Centered5s | r2/q2 | 671 | 0 | 486 | n/a | n/a | n/a | n/a |
| NMFP | Centered5s | r4/q4 | 585 | 0 | 408 | n/a | n/a | n/a | n/a |
| NMFP | Session-open5s | r1/q1 | 105 | 100 | 100 | 69 | 66 | 1 | 2/168 |
| NMFP | Session-open5s | r2/q2 | 79 | 0 | 62 | 35 | 25 | 0 | 1/168 |
| NMFP | Session-open5s | r4/q4 | 79 | 0 | 61 | 33 | 24 | 0 | 1/168 |
| PeakNetFP | Native10s | r1/q1 | 102 | 83 | 89 | n/a | n/a | n/a | n/a |
| PeakNetFP | Centered5s | r1/q1 | 61 | 45 | 50 | n/a | n/a | n/a | n/a |
| PeakNetFP | Centered5s | r2/q1 | 29 | 0 | 16 | n/a | n/a | n/a | n/a |
| PeakNetFP | Centered5s | r4/q1 | 18 | 0 | 10 | n/a | n/a | n/a | n/a |
| PeakNetFP | Session-open5s | r1/q1 | 28 | 22 | 24 | 0 | 0 | 2 | 0/168 |
| PeakNetFP | Session-open5s | r2/q1 | 16 | 0 | 10 | 0 | 0 | 0 | 0/168 |
| PeakNetFP | Session-open5s | r4/q1 | 9 | 0 | 7 | 0 | 0 | 0 | 0/168 |

The native signal payloads are 3.655832 MB/hour at r1, 1.828121 at r2 and 0.921572 at r4. Track boundaries and grid origins are reported separately. No isolated inference or matching latency is claimed by these acoustic runs.

There is a structural grid limit on the strict metric for sparse 5 s controls. The native reference annotations are integer-second origins; centering shifts every expected origin by 2.5 s. A 1 s or 2 s grid-only returned origin is therefore at least 0.5 s away for all 1,488 clips. Strict 0.1 s is unattainable for these sparse offset rules before any prediction is made. The 2 s sensitivity remains reported, alongside this limitation. No phase adjustment was selected from the annotations or test results.

PeakNet's zero unknown accepts do not constitute successful recognition: all three session-open configurations accept zero correct known queries. The native configuration also accepts two wrong identities. NMFP's native 2/168 unknown rate is about 1.19%, exceeding its nominal 1% calibration target. Per-query Wilson, source-cluster, creator-cluster, source/session crossed intervals, and source-event sensitivities are retained in the full summary. Zero-event empirical bootstrap intervals are degenerate and are not an upper population error bound.

## Implementation audit and evidence limits

Twenty hash-selected calibration-known clips per model were re-extracted from their clean reference positions, using actual query batches rather than the reference extraction batch size. Both models returned the correct source and exact offset on 20/20. The largest embedding difference from the stored reference windows was 7.43e-6 for NMFP and 3.87e-7 for PeakNet. All cache entries have finite unit-norm fingerprints, zero decoder crop shortfalls, and verified checkpoint/source provenance. These checks exclude the audited batching, crop, clock, and restoration failure modes; they do not prove that every capture or implementation issue is absent.

FMA creator/title matching identified ten possible shared sources, five in the conservative potential neural-training superset. The full summary reports both exclusion sensitivities without changing the gallery or gate. This is source-level caution; matching titles do not prove identical recordings, and undisclosed training content cannot be audited completely. Session and upstream-QC strata remain available rather than being filtered into a cleaner benchmark.

The observed gap is conditional on this released capture set and frozen configurations. Architecture, training loss, augmentations, reference corpus, and capture conditions differ together. These runs cannot attribute the gap solely to neural versus non-neural representation, or to one degradation mechanism. They also cannot justify a fleet-wide error guarantee from 168 correlated unknown clips.

## Reproduction and records

Extraction uses the existing isolated neural environments and verified public weights:

```sh
work/venv_nmfp/bin/python outputs/acr_repro/neural_baseline.py --manifest work/benchmarks/acoustic_rr/sdrr_native_protocol.json --cache-dir work/benchmarks/acoustic_rr/nmfp_cache --durations 10 5 --threads 2
work/venv_peaknet/bin/python outputs/acr_repro/peaknet_baseline.py --manifest work/benchmarks/acoustic_rr/sdrr_native_protocol.json --cache-dir work/benchmarks/acoustic_rr/peaknet_cache --durations 10 5 --threads 2
```

`run_neural_acoustic.py` runs the full native/density/session matrix from those caches. `summarize_acoustic_rr.py` verifies query membership, gallery roles, source identity, centered bounds, and finite scores, and computes all reported strata and intervals. `audit_neural_clean_source.py` reproduces the clean-source/batch audit; `test_neural_acoustic.py` verifies denominators, gallery separation and calibration before the first held-out query; `test_peaknet_sparse_clocks.py` verifies catalog-order invariance with synthetic vectors.

Canonical summaries and all per-query predictions are under `work/benchmarks/acoustic_rr/{nmfp_results,peaknet_results}/`. The same directory contains `neural_origin_grid_audit.json`, `neural_reference_bounds_audit.json`, both clean-source audits, full extraction logs, release protocol hashes, and a SHA256 provenance ledger. The released benchmark is [SD-RR, DOI10.5281/zenodo.22169646](https://doi.org/10.5281/zenodo.22169646), associated with the [POLARIS repository](https://github.com/JihengLi/POLARIS). Audio remains governed by its per-file licenses.
