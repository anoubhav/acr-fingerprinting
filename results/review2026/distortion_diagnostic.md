# Frozen descriptive PEX failure diagnostic

Derived with `work/venv/bin/python work/review2026/derive_distortion_strata.py`. Outputs: `distortion_strata.json`, containing source digests, exact original calibration gates, all per-query assignments/outcomes, eight configurations, three mutually exclusive strata, cluster sensitivities and dense/sparse paired results. No primary result or model was edited.

The schema was checked against the local pinned upstream PEX revision `25ae1edd7e3252ef1b64fd0c90d5f50c45b5824b`: README lines 115–116 defines tempo as percent and pitch as cents; `generate_query_audios.py` lines 441–450 rounds the annotation values; `util.py` lines 64–75 gives the conversions. Defaults are zero annotated pitch and 100% annotated tempo. These groups therefore describe annotations; they do not prove exact acoustic equality to an unmodified waveform.

The groups depend solely on these named fields, partition all 543 known queries, and retain their original global 74-negative gates. Other distortions co-occur. Within pitch altered, 32 clips have noise and 27 have overlapping other annotations; within tempo-only, 16 have noise and 18 have overlaps; within neither, 218 have noise, 33 echo, 31 reverberation, 84 filtering and 146 overlap. No group is a clean controlled attack experiment.

| Configuration | Pitch altered, n=78 | Tempo-only, n=35 | Neither, n=430 |
|---|---:|---:|---:|
| ACR exact dense | 6 / 3 / 1 | 33 / 29 / 0 | 407 / 373 / 2 |
| ACR exact factor8 | 5 / 4 / 1 | 34 / 29 / 0 | 407 / 372 / 1 |
| ACR IVF factor8 query4 | 5 / 3 / 1 | 33 / 28 / 0 | 408 / 370 / 1 |
| Audfprint native | 0 / 0 / 0 | 1 / 1 / 0 | 322 / 322 / 3 |
| Audfprint candidate | 2 / 0 / 1 | 6 / 1 / 0 | 379 / 327 / 3 |
| Public MinHash native | 3 / 3 / 1 | 17 / 14 / 0 | 374 / 353 / 1 |
| NMFP native | 6 / 1 / 0 | 35 / 33 / 0 | 425 / 414 / 1 |
| PeakNet native | 20 / 9 / 0 | 33 / 25 / 0 | 404 / 337 / 2 |

Cells are **candidate correct / accepted primary correct / accepted target mismatch**, among known queries. A target mismatch may be a secondary annotated source. These cells are not unknown false-accept rates. Every cell uses the original global gate and all eight rows reconcile exactly with the full-cohort summaries.

There are 78 sources/68 montages in pitch altered, 35/32 in tempo-only, and 412/201 in neither. Two thousand fixed-seed source-only, montage-only and crossed source/montage bootstrap draws use the same weights across methods. The full intervals and paired estimates are in JSON. For exact factor8 ACR, candidate-correct 95% crossed sensitivities are 0–18.46%, 84.21–100% and 90.69–97.81%, respectively. Accepted-correct sensitivities are 0–16.22%, 58.53–100% and 80.87–91.71%. Wide intervals and boundary-degenerate empirical intervals should not be interpreted as guarantees about new sources or recordings.

Dense/sparse candidate counts differ by minus1 / plus1 / zero across the three groups, with 33 / 2 / 14 changed candidate IDs. On the largest group both have 407 candidates; their accepted counts are 373 and 372. The corresponding sparse-minus-dense accepted difference is −0.233 percentage points, with crossed sensitivity [−2.169,1.559]. These are finite-cohort descriptive results, not equivalence tests.

## Recommended paper use

Prefer a compact count table to a multi-color 48-point chart. Counts show the severe pitch boundary and separate calibrated acceptance and target mismatches without hiding small denominators. The table can fit in one column using `tabularx` and short headers; it may also serve as an appendix table with a two-sentence main-paper description. Keep the original full-cohort table as the primary result.

Suggested main prose:

> An exploratory diagnostic partitions the frozen music tests by annotated pitch and tempo, without changing models or gates. Of the 97 factor-8 candidate errors, 73 occur among the 78 pitch-modified clips; on the 430 clips with neither annotated pitch nor tempo modification, dense and factor-8 enrollment each find 407 candidates and accept 373 and 372. Other degradations and mixtures co-occur, so these strata describe the failure boundary rather than isolate a causal pitch effect.

If source/montage uncertainty is shown in the main paper, use one interval for the largest stratum or paired density difference, and point to the artifact for all intervals. Do not present the 35-case tempo-only group as precise evidence of general tempo robustness. The proposed diagnostic also should not replace 82.14% full-cohort identification with a 94.65% subgroup headline.
