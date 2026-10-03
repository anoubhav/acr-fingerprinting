# Full reference-phase sensitivity

The frozen plan declares all eight physical frame residues before inference. All original 835 five-second queries were evaluated with the same 659-reference gallery, dense usable query grid, FMA-fitted PCA, exact top-five neighbors, 0.24 s offset consensus and 2 s localization criterion. Each residue uses its own gate from the same 74 calibration unknowns at the original empirical 1% target. No phase was selected; phase 0 remains primary.

| Phase | Retained vectors | Candidate | Candidate+position | Accepted | Accepted+position | Target mismatch | Unknown accepts | Changed known IDs vs phase 0 | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 792,973 | 446 | 428 | 405 | 397 | 2 | 0 | 0 | 0.05309281 |
| 1 | 792,904 | 447 | 427 | 403 | 392 | 2 | 1 | 19 | 0.05340662 |
| 2 | 792,838 | 446 | 428 | 400 | 390 | 2 | 0 | 35 | 0.05720945 |
| 3 | 792,776 | 446 | 427 | 406 | 395 | 4 | 0 | 35 | 0.05065587 |
| 4 | 792,686 | 447 | 424 | 410 | 395 | 4 | 0 | 39 | 0.05027767 |
| 5 | 792,604 | 449 | 428 | 403 | 392 | 5 | 1 | 35 | 0.05096676 |
| 6 | 792,543 | 448 | 429 | 403 | 395 | 5 | 1 | 31 | 0.05139912 |
| 7 | 792,492 | 445 | 426 | 405 | 394 | 5 | 1 | 21 | 0.05246893 |

Known-count denominator is 543; unknown denominator is 71. Candidate counts range 445–449 (81.95–82.69%); accepted correctness ranges 400–410 (73.66–75.51%). Signal payload ranges 2.40004–2.40150 decimal MB/audio-hour. Phase 0 exactly reproduces every canonical original ID, score, offset, usable query count, vote fraction and mean squared-L2 value across all 835 queries.

The strict verifier independently rechecks all input hashes, binds each cache to its source ID, validates finite/monotonic physical timestamps and nonnegative/unique/monotonic original indices, and checks reconstruction residual at most 1e-8 s. Actual maximum reconstruction residual is 0.0. It derives residue counts, score arithmetic, role/crop/target scope, calibration gates and all counts. The small 659-source residue ledger and SHA-bound dense comparison support portable evidence checks without exporting caches.

These are descriptive all-phase sensitivities. They do not select a replacement configuration, establish formal noninferiority, give simultaneous inference over phases, or include fitting/gate-selection uncertainty in the fixed paired intervals.

Authoritative records: `results/plan.json`, eight `results/phase_*_5s.json`, `results/summary.json`, `results/strict_validation.json` and `results/reference_residue_counts.json`.
