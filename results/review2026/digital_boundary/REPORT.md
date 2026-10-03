# Digital-input boundary stress experiment

This post-hoc experiment tests two concrete capture conditions described in the
ACR design record: leading silence and missing initial content. It modifies
public benchmark PCM under a fixed protocol. It is not actual TV capture or a
production impact measurement.

## Protocol and conformance

The immutable plan (`results/plan.json`, SHA256
`85187040d2780b4a324cfe7bd706bc6b8e3d85bd8589934751cc0cc25c365cc1`)
was saved before inference. It hashes all 219 original public PEX montage MP3s,
the original 835-query protocol, the disjoint 94-source FMA PCA, the compact 659
gallery IVF512 index and physical-frame metadata, the canonical original result,
and the implementation. Gallery reference factor 8, query stride 4, probe 16,
top 5 matching, vote tolerance 0.12 s, and the original 5 s gate
`0.05657886749843364` stay frozen. There is no refit, index retraining,
condition selection, gate retuning, or outcome-based exclusion.

Every original 5 s crop is freshly FFmpeg-decoded from its original montage.
Before transformed inference, all 835 unchanged crops exactly reproduced the
canonical top1 IDs, scores, offsets, correctness/localization, query-frame
counts, vote fractions, mean squared distances, and acceptance decisions. The
fresh decoder and matcher therefore reproduce the baseline without relying on
cached query descriptors. `results/unchanged_parity.json` records this gate.

All four conditions contain the same 543 test-known,71 test-unknown,147
calibration-known, and74 calibration-unknown queries, including any empty
feature outputs. No condition produced empty features in this cohort. At 8 kHz:

- Unchanged: the original 40,000 samples.
- Lead 0.5 s:4,000 zeros followed by the first36,000 original samples;5s total.
- Lead 1 s:8,000 zeros followed by the first32,000 original samples;5s total.
- Trim 1 s: original samples8,000 through39,999;4s total.

The affine annotation map is `reference_time = original5s_GT + query_time ×
expected_time_scale`. Leading silence changes its intercept by
`−lead × expected_time_scale`; trimming changes it by
`+trim × expected_time_scale`. This matters for the116 tempo-transformed queries.
The transform leaves scale unchanged. Query frame residues are recovered from
original physical timestamps and thinned modulo4, independently of silence
rejection; silence never compresses the clock. Each row retains the physical
frame indices, PCM hash, original/adjusted truth, scale, and complete prediction.

## Held-out results

Counts use 543 known and71 unknown test queries in every row. Localization is
agreement with the adjusted public annotation within2s, as in the primary
experiment. The gate remains the original 5 s gate even for the4s input.

| Input | Candidate correct | Candidate localized2s | Accepted correct | Accepted localized2s | Accepted wrong known ID | Unknown accepts | Median queried frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unchanged5s |446|429|401|392|2|1|47|
| Lead 0.5 s,5s total |444|423|400|386|4|1|47|
| Lead 1 s,5s total |442|424|396|388|6|2|44|
| Trim 1 s,4s total |444|423|403|392|5|2|36|

Wrong test-known candidate IDs number 97,99,101, and99, respectively. Small
aggregate differences do not imply identical decisions: transformed
conditions change 41,52, and38 test-known candidate IDs, respectively, and
108,126, and109 IDs over all 835 queries. Correctly accepted calibration-known
counts remain 109/147 in all four conditions. Calibration-unknown accepts drift
from 0/74 to 1/74,2/74, and3/74 at the frozen gate. The original calibration target
does not transfer as a guarantee under these perturbations.

Relative to unchanged input, the three perturbations lose/gain 5/3,7/3, and6/4
correct candidates; 2/1,7/2, and4/6 correctly accepted matches; and 7/1,9/5, and6/6
accepted-localized matches. These complete paired transitions are saved in
`results/decision_switches.json`.

Paired 95% crossed source/montage bootstrap intervals use 2,000 resamples,
seed 20261002,518 known source IDs, and207 test-known montage IDs. They condition
on the fixed model, gallery, operating point, and gate. Differences are in
percentage points relative to unchanged input:

| Input | Candidate-correct difference [95% CI] | Accepted-correct difference [95% CI] | Accepted-localized2s difference [95% CI] |
|---|---:|---:|---:|
| Lead 0.5 s |−0.37[−2.30,1.39]|−0.18[−1.44,0.92]|−1.10[−3.02,0.39]|
| Lead 1 s |−0.74[−3.02,1.28]|−0.92[−3.00,0.74]|−0.74[−3.28,1.50]|
| Trim 1 s |−0.37[−2.52,1.62]|0.37[−1.60,2.43]|0.00[−2.17,2.26]|

All crossed intervals in the table include zero. Source-only bootstrap
intervals are also retained; the lead 0.5 s accepted-localized interval excludes
zero (−2.21 to−0.18pp), while its crossed interval includes zero. The crossed
analysis incorporates dependence within shared query montages. This does not establish equivalence, absence of
harm, or a benefit from trimming. With only 71 unknowns, 1/71 accepts has a
Wilson95% interval of 0.25–7.56%;2/71 has 0.78–9.70%. The result supports a
limited statement: these prescribed digital-input perturbations produced small
aggregate recognition changes in this fixed public cohort, while changing
individual decisions and increasing some false acceptances.

## Annotation-sensitive diagnostic and scope

The 0.1 s annotation-agreement diagnostic is retained in per-query JSON and the
summary. Candidate/accepted counts are 74/70,75/71,75/71, and65/60 for the four
conditions. Publisher annotation boundaries are integer-second segment labels;
these numbers must not be interpreted as measured subsecond synchronization
accuracy. The 2 s endpoint remains primary. No timing values or latency claims are
reported. This experiment supplies a targeted digital boundary check, not a
general acoustic robustness result or production deployment validation.

## Reproduction and independent verification

`run_digital_boundary.py` has an explicit `--plan-only` stage and refuses changed
plan inputs. Its unchanged conformance gate precedes all transformed inference.
`verify_digital_boundary.py --audio` independently re-decodes all 219 source
montages, checks all 3340 original/transformed PCM hashes and sample geometry,
re-derives source/role/cohort membership, affine truth, frozen-gate decisions,
counts and paired point estimates, and rechecks the canonical 835 predictions.
It does not rerun retrieval or retune a gate. Full output is in
`verification.json`; `results/summary.json` records uncertainty and each result
hash. Audio stays in the licensed benchmark workspace and is not redistributed.

Example execution (paths are workspace-relative):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 work/venv/bin/python \
  work/review2026/digital_boundary/run_digital_boundary.py \
  --source-dir outputs/acr_repro \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --model work/benchmarks/acr_medium/pca_model.npz \
  --index work/revision2026/streaming_index/compact_659_r0.index \
  --metadata work/revision2026/streaming_index/compact_659_metadata.npz \
  --frozen-result work/benchmarks/acr_ivf_efficiency_study/acr_ivf_selected_5s.json \
  --output work/review2026/digital_boundary/results --threads 3
```

Save the plan first by adding `--plan-only`, then repeat without it. The experiment does not mutate
primary results, the paper, or the public repository.
