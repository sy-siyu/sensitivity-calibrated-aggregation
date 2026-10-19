# Reproduction

Run commands from an installed checkout with `python -m sca_reproduction` (or `sca-reproduce`). The workflow generates inputs and results inside the selected workspace using the tracked configurations and downloaded public data.

`prepare-data` is the only network step. Other runners use local generated artifacts and the two checksum-verified public CSVs. Each workspace records its exact configuration files, including descriptive metadata. Continue an existing run with those files unchanged; use a new workspace after configuration updates. This also prevents a smoke workspace from being reinterpreted as a full-paper run.

## Current dependency sequence

All individual commands below default to `--workspace outputs/paper --config configs/paper.json`.

1. `run assignment` generates 4,800 datasets over 48 fixed conditions and 100 replications. It computes the principal full-sample SCA procedure, family restrictions, confidence-construction controls, population-oracle diagnostics, and the `1/5` and `1/2` pilot splits used by later studies.
2. `run release` generates 100 independent blocks for each of two K64 controlled laws and the K16 IHDP-derived reference population: **300 independent calibration/outcome blocks total**. Each block supplies five capacities, so later capacity comparisons contain 1,500 dependent capacity cases, not 1,500 independent datasets.
3. `run certification` computes positive-margin oracles in each assignment dataset's realized candidate family and sparse-mixture lower bounds.
4. `run relevance` consumes assignment and release artifacts plus the verified RHC CSV. It completes the five-fraction LTT development comparison, selects `1/3`, evaluates A–F on 4,320 held-out datasets, certifies globally optimal one-dimensional propensity-logit clustering proposals on the 300 release blocks, and runs 21 RHC splits.
5. `run recovery` generates 300 independent nested paths at the 12 configured sample sizes. `run efficiency` combines assignment, release, and recovery evidence. Run both before `report efficiency`.
6. `run precision` reuses release observations and confidence regions at caps `1,11/10,5/4,3/2,2`. Repeated capacities, profiles, methods, and cohort prefixes do not create new independent blocks.
7. `run outcome` generates the controlled independent binary-outcome panel and raw-data references.
8. `run computation` performs the fresh sequential scaling study. It uses 200 primary calibration datasets, 30 capacity-provider cases derived from matched inputs, and 20 dependent repeated-cohort cases.

Prepare the public inputs first:

```sh
python -m sca_reproduction prepare-data --directory data --dataset all
```

Then either run the sequence explicitly or use:

```sh
python -m sca_reproduction run all --workspace outputs/paper
python -m sca_reproduction report all --workspace outputs/paper
```

The `all` sequence is `assignment`, `release`, `certification`, `relevance`, `recovery`, `efficiency`, `precision`, `outcome`, `computation`. The separate `split` runner reproduces the earlier two-fraction comparison. Its versioned configurations are:

- `configs/jmlr_statistical_baselines_v1_legacy_20261019.json`
- `configs/jmlr_split_ablation_v1_legacy_20261019.json`

They are not used for the current Table 4 or Figure 3.

## Finite-sample certification outputs

`configs/jmlr_certification_bottleneck_v1.json` fixes caps `{5/4,3/2,2}` and positive log-margin offsets `{0,1/40,1/20,1/10,1/5}`. For every fresh assignment dataset, the oracle is recomputed in the **same realized union** of the two fixed-tree pruning families and the estimated-order contiguous family used by the operational procedure. Population truth enters only this post-selection diagnostic.

The lower-bound component compares two-point and sparse-mixture alternatives using the identical `b` in each row:

| Alternative `b` | Exact distortion | Two-point necessary ceiling | Sparse-mixture necessary ceiling |
|---|---:|---:|---:|
| `4/13` | `827/372` | 1,900 | 2,705 |
| `199/500` | `685807/456307` | 7,156 | 9,606 |

These are same-alternative lower-bound ceilings. They are not the same object as either:

- 256,000, the first **prespecified empirical recovery-grid** size with complete recovery; or
- 2,852,384, the first integer satisfying the analytical Bonferroni CP-diameter sufficient criterion.

The last quantity is an analytical sufficiency reference, not an executed sample size or an optimal threshold. Full-precision records are written to `certification/tables/oracle_margin_exact.json` and `certification/tables/lower_bound_exact.json`; `oracle_margin_summary.csv` is the rounded display table.

## Empirical-relevance outputs

### LTT development and coherent evaluation

The development panel contains replications 0–9 in each of 48 conditions: 480 datasets. It compares LTT pilot fractions `{1/10,1/5,1/3,1/2,2/3}` at cap `3/2`, weights the fixed conditions equally, and breaks exact ties toward the smaller fraction. The selected fraction is `1/3`.

Replications 10–99 form the 4,320-dataset evaluation panel. A–F are recomputed coherently using the selected fraction:

| Label | Definition |
|---|---|
| A | Split LTT baseline with its unadjusted tests and original first-failure rule. |
| B | Held-out Coupled CP on saved proposals, first failure. |
| C | Held-out Coupled CP over every saved proposal. |
| D | Full-sample Coupled CP over that proposal list. |
| E | Full-sample Coupled CP over the complete pilot-order family. |
| F | SCA over the complete full-data-order combined family. |

The output `relevance/tables/ltt_exact.json` preserves exact rows. `ltt_development.csv`, `ltt_methods.csv`, and `ltt_contrasts.csv` contain publication-oriented numerical summaries. The contrast sequence is signed and descriptive; it is not a unique causal decomposition.

### Propensity-logit clustering

For every release block and reporting capacity, the baseline globally minimizes calibration-count-weighted within-cluster squared error of smoothed propensity logits over exactly `B` nonempty contiguous whole-state groups. A dynamic program searches all ordered cuts; immutable state IDs resolve ties. The smoothed propensity is `(2T_s+1)/(2N_s+2)`, matching the SCA ordering score. Clustering only proposes the partition: the unchanged Coupled CP region and exact cell certificate determine admission.

The resulting 1,500 rows derive from 300 independent blocks. The current release-capacity figure plots SCA and the equal-mass baseline, not this clustering comparison; clustering results support the accompanying text and are stored in `relevance/tables/clustering_exact.json` and `clustering_capacity_cases.csv`.

### RHC illustration

The frozen RHC protocol uses all 5,735 records, eight prespecified demographic states, a 4,014-record assignment/calibration cohort, and a disjoint 1,721-record outcome cohort. Split 0 is the primary illustration. Splits 1–20 reuse the same observed records under different deterministic permutations and are dependent stability checks—not independent datasets, coverage replications, or a clinical validation study.

At the primary capacity-four setting, all compared methods fail the declared admission ceiling. The operational result is therefore abstention with **no released aggregate interval**. Any finite inadmissible calculation is retained only inside an explicitly labeled diagnostic field. Exact and display outputs are `relevance/tables/rhc_exact.json` and `rhc_observed.csv`.

## Outcome-inference conventions

`SCA(alpha=...)` controls assignment calibration. `infer_outcomes` and `release_outcomes` separately accept `overall_alpha` and optional component errors. The default overall allocation at `1/20` is `1/40` calibration plus `1/80` for each raw and aggregate outcome family.

The binary branch preserves exact CP outcome intervals and sharp binary sensitivity formulas. The bounded branch normalizes using prespecified bounds, constructs Hoeffding mean intervals, intersects all components on the common normalized scale, and only then applies the exact positive affine map. Its population step is a valid mean-based outer enclosure; it need not equal a sharp distribution-aware bounded-outcome region.

A supplied drift bound changes state masses but assumes unchanged statewise propensities. Target mass boxes are intersected with the simplex, target group propensities are reprojected, and nonsingleton certificates are multiplied by `(1+zeta)/(1-zeta)`. Export refuses a partition that exceeds its reporting capacity or whose corrected transported budget exceeds the sensitivity ceiling.

Aggregate releases deliberately omit raw states, memberships, and the calibration region. The analyst can validate the strict schema, capacity, corrected admission arithmetic, and reconstructed intervals; validity still requires provider trust in outcome-blind selection and faithful construction. The coverage statement is unconditional for prespecified analysis/sample sizes. It does not claim nominal conditional coverage after admission.

## Exact values and display values

JSON artifacts encode rational values exactly. The certification and relevance reporters additionally write explicitly named `*_exact.json` files. CSV files are rounded or floating display summaries and are never used as replacements for exact evidence. Plotters read completed reporter tables and save a coordinate receipt next to every figure.

`paper-tables` requires all three complete current empirical table inputs. Current table/figure numbers and stable labels are mapped in [PAPER_RESULTS.md](PAPER_RESULTS.md).

## Randomness and numerical conventions

Assignment generation uses PCG64 `SeedSequence([20261008,300,K,mass_index,overlap_index,geometry_index,size_index,replication])`. The second fixed tree uses `[20261008,101,K]`; the permuted-law geometry uses `[20261008,102,K]`. Exact nominal laws retain the first `K-1` stored binary masses and use the exact final residual.

The five LTT fractions share nested counts through deterministic multivariate-hypergeometric substreams: `1/10` is nested inside `1/5`, `1/3` lies between `1/5` and `1/2`, and `2/3` extends `1/2`, preserving the two base splits. Pilot scores rank proposals only; verified certification does not use floating pilot bounds.

Release and precision streams use master 2026101502 with a separate development master. Controlled draws use exact 128-bit dyadic thresholds; IHDP resamples reference-record IDs. RHC split `j` uses `PCG64(SeedSequence([20261019,501,j]))`. Recovery uses master 2026101603 and computation uses 2026101701 except for its documented K64 release bridge.

Verified roots use a 48-bit outward endpoint grid, initial precision 128 bits through 1,024 bits, at most 192 evaluations, and a five-second endpoint limit unless a frozen study config says otherwise. Threshold rejection is not a completed upper certificate. A safe identity fallback does not establish count optimality. Numerical infinity and unresolved work remain separate statuses.

## Restart, cost, and bounded checks

Completed compatible records are reused. Restart the identical command to continue. `run STUDY --limit N` is an explicit incomplete prefix; finish with the unbounded command. Full reporters reject incomplete denominators.

The smoke configuration uses small support and bounded prefixes to exercise generation, certification, API reconstruction, reporting, and CLI wiring:

```sh
python -m sca_reproduction smoke --workspace outputs/smoke \
  --config configs/smoke.json --data data/ihdp_npci_1.csv --rhc-data data/rhc.csv
```

Smoke output is labeled `Execution checks only`; it cannot support manuscript estimates or uncertainty intervals. Full reproduction can require many CPU-hours and tens of GiB. The K256 capacity study can take many minutes and several GiB per process. Run one study command at a time in each workspace.

## Optional exact five-state check

`python -m experiments.jmlr_five_state_counterexample_v1` verifies a rational five-state counterexample by enumerating all 52 unrestricted set partitions and 16 contiguous cut partitions. It uses masses

`(13,610,21,89,377)/1110`, propensities `(545,723,764,836,874)/1000`, and cap `19/10`.

The unrestricted best two-cell distortion is `25244179/13307679 < 19/10`, while the best contiguous two-cell distortion is `12892811/6780186 > 19/10`; a contiguous three-cell partition has distortion `435789/341539 < 19/10`. Thus the unrestricted and contiguous optima are two and three groups, respectively.
