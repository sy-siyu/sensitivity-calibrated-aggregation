# Paper results and reproduction map

This map targets **Sensitivity Calibration Under State Aggregation**, the 49-page PDF created and modified **2026-10-19 10:23:18 AEDT** (pdfTeX 1.40.29). Its SHA-256 is `8024b6083b10bb6401325d0292bbc89a3d4b10e029bc18f8ea2d50473a70ea91`. The numbering below was checked against that PDF and the active `main.tex` experiment inputs. The manuscript/PDF are not distributed in this code repository. The corresponding code release is `paper-reproduction-2026-10-19`.

SCA means full-sample Coupled CP certification and complete search within the combined two-tree/estimated-order family. Certificate constructions and restricted searches are ablations; population oracles are diagnostics. Fixed-cap count minimization and fixed-capacity certificate minimization are distinct objectives, even though their completed feasibility decisions agree at matching budgets.

## Complete execution

From the installed public checkout, use a fresh workspace. These commands regenerate inputs and numerical evidence; they do not import published answers:

```sh
python -m pip install -e '.[plots]'
python -m sca_reproduction prepare-data --directory data
for study in assignment release recovery efficiency split precision computation; do
  python -m sca_reproduction run "$study" --workspace outputs/paper --config configs/paper.json
done
for study in assignment release recovery efficiency split precision computation; do
  python -m sca_reproduction report "$study" --workspace outputs/paper --config configs/paper.json
done
python -m sca_reproduction paper-tables --workspace outputs/paper
python -m sca_reproduction plot all --workspace outputs/paper
```

The dependency order is assignment → release → recovery → efficiency → split → precision → computation. These are the studies needed for the listed main results; the separate legacy outcome panel is not required. Recovery is an independently generated diagnostic, not another assignment dataset. Precision reuses release blocks. Computation regenerates its K64 count bridge directly, then runs four separate preflights before freezing resource limits and executing its panel. Full execution can take many CPU-hours and substantial storage; see [reproduction details](REPRODUCTION.md). Smoke outputs and `--limit` prefixes do not reproduce the paper panels.

All paths below are relative to `outputs/paper/`. Individual commands are `python -m sca_reproduction run STUDY` and `python -m sca_reproduction report STUDY`, using that workspace and `configs/paper.json` by default. Figures are generated as both PDF and PNG in `paper_figures/`, with `*_coordinates.csv` receipts identifying the saved statistics drawn. The table helper assembles all three tables together, so it requires all their listed reporter inputs; it writes numerical CSVs and a `manifest.json` in `paper_tables/`. Generated artifacts remain untracked. The helpers calculate from completed generated records/tables, without a hardcoded table of published estimates.

## Main tables

| Paper item and labels | Configuration and execution | Generated output and exact extraction |
|---|---|---|
| **Table 1 — Search and certification ablations.** `tab:larger-support` **and** `tab:efficiency-procedures` label the **same table**. | `configs/jmlr_verified_pipeline_v1.json`, `configs/jmlr_efficiency_gap_v1.json`. Run/report assignment; run release, recovery and efficiency; report efficiency. Its full reporter validates all three efficiency components. | `paper_tables/table1_assignment_search.csv`. First three rows: `assignment/tables/assignment_rows.csv`, filter `mode=unknown_mass`, `cap=3/2`, families `fixed`, `contiguous`, `combined`. Use `method=coupled` for compression/oracle gap/recoveries; pair `independent` with `coupled` for coupling gains. Last two rows: `efficiency/tables/historical_primary_equal_weight_summary.csv`, procedures `allocation_control`, `pooled_alternative`. |
| **Table 2 — Data allocation and search ablations at cap 3/2.** `tab:statistical-baseline`. | `configs/jmlr_statistical_baselines_v1.json`, `configs/jmlr_split_ablation_v1.json`. Run assignment, then `python -m sca_reproduction run split`; `python -m sca_reproduction report split`. | `paper_tables/table2_split_procedures.csv`, from `split/tables/methods_grid.csv`: `scope=all`, `cap=3/2`, procedures A–F, field `compression_pp_mean`. `paper_tables/table2_split_contrasts.csv` supplies its paired comparisons. |
| **Table 3 — Five-stage global width decomposition.** `tab:precision-stages`. | `configs/jmlr_release_capacity_v1.json`, `configs/jmlr_precision_frontier_v1.json`. Run release, then `python -m sca_reproduction run precision`; `python -m sca_reproduction report precision`. | `paper_tables/table3_precision_stages.csv`, from `precision/tables/decomposition_summary.csv`: balanced law (`law_index=0`), `cap=3/2`, `m_ratio=256`, `transport=global`, profiles `aligned` and `alternating`; fields `I0_width_mean` through `I4_width_mean`. |

For Table 1, compression is `100*(K-count)/K` (or `100*compression`); coupling gain is `100*(count_independent-count_coupled)/K` **within the same dataset and family**; oracle gap is `100*(count-oracle_count)/K`. The assignment `oracle_gap` field is a raw count difference, not already normalized. Recoveries count equality with that family's population-oracle count. For the confidence controls, use `100*mean` for metrics `compression` and `normalized_count_gap`, and count `count_recovered` in the primary historical rows. The primary filter is unknown masses and cap `3/2`; do not mix secondary caps. All point estimates weight the 48 conditions equally, each with 100 datasets (4,800 total). Efficiency contrast uncertainty uses 100 replication-index averages of the paired within-condition differences and normal 95% intervals.

Table 2 uses **evaluation replications 10–99 only**: 90 per condition, 4,320 datasets. Separate development replications 0–9 select the pilot fraction from `1/5` and `1/2`; evaluation uses the frozen development choice. A is Split LTT; B is held-out coupled/first failure; C inspects all held-out pilot proposals; D uses the full sample on that list; E searches the complete pilot-order family; F is SCA with full-data order. Compression already uses percentage points. Equal-condition aggregate normal intervals use variance `48^-2 * sum_h(s_h^2/90)`; condition intervals use paired Student95. Table 1 includes all 100 replications, so its SCA mean need not equal Table 2's 90-replication mean.

Table 3 uses all 100 balanced-law blocks, without admission filtering. I0 is the raw population interval at G; I1 uses true aggregate assignment/outcomes at true transported budgets; I2 uses certified budgets; I3 uses true assignment singleton boxes with **the operational outcome CP boxes and certified budgets**; I4 is the aggregate-only analyst enclosure. `paper_tables/table3_precision_increments.csv` retains the signed, sequence-dependent increments and negative-increment counts; I3 is not operational inference. `G*U`, not the ceiling `G*c`, determines inference.

## Main figures

### Figure 1 — Oracle-count inflation and homogeneous recovery (`fig:efficiency`)

Configuration: `configs/jmlr_efficiency_gap_v1.json`, with assignment and release prerequisites. Execute `python -m sca_reproduction run recovery`, `python -m sca_reproduction run efficiency`, then `python -m sca_reproduction report efficiency`. Generate with:

```sh
python -m sca_reproduction plot efficiency --workspace outputs/paper
```

Output: `paper_figures/statistical_efficiency.{pdf,png}`. Left: `efficiency/tables/historical_rows.csv`, filter `procedure=original`, `mass_mode=unknown_mass`, `cap=3/2`; x=`oracle_margin_log`, y=`minimax_log_inflation`, all 4,800 datasets. The latter is the log inflation of the **capacity-minimax certificate at the population oracle count**, not `threshold_log_inflation` or the threshold-selected map's `certificate_inflation_log`. Right: `efficiency/tables/recovery_condition_summary.csv`, `law_id=homogeneous`, `mass_mode=unknown_mass`, all three procedures and all 12 sizes; plot `certified_mean` and `certified_cp95_lower/upper`. Each size has the same 300 independent nested paths. The 90% line and connected grid points do not define an exact sample-complexity threshold. `recovery_grid_crossings.csv` records the empirical crossings; the 1,900 necessary and three-million sufficient analytical references have different scopes.

### Figure 2 — Paired compression contrasts (`fig:split-search-contrasts`)

Configuration/prerequisite: Table 2. Execute `python -m sca_reproduction run split` and `python -m sca_reproduction report split`, then:

```sh
python -m sca_reproduction plot split --workspace outputs/paper
```

Output: `paper_figures/split_search_contrasts.{pdf,png}`. Input: `split/tables/paired_grid.csv`, `scope=all`, `cap=3/2`; contrasts `C-B`, `D-C`, `E-D`, `F-E`, `C-A`, `F-C`, `F-A`. Plot `difference_pp_mean` with `difference_pp_analytic_Normal95_lower/upper`. Positive is left minus right. The first four follow the specified sequence, not unique causal component effects. The denominator remains 90 paired evaluation datasets in each of 48 equally weighted conditions. Preserve negative condition/dataset contrasts and stopping diagnostics in `paired_conditions.csv`, `paired_dataset_rows.csv`, `stopping_grid.csv` and `stopping_rows.csv`.

### Figure 3 — Reporting capacity, certificates and true distortion (`fig:release-capacity`)

Configuration: `configs/jmlr_release_capacity_v1.json`; prerequisite: checksum-verified public IHDP covariates. Execute `python -m sca_reproduction run release` and `python -m sca_reproduction report release`, then:

```sh
python -m sca_reproduction plot release --workspace outputs/paper
```

Output: `paper_figures/release_capacity.{pdf,png}`. Input: `release/tables/calibration_summary.csv`, all three laws and all capacities; methods `verified_capacity` (SCA) and `equal_calibration_mass` (same-capacity baseline). Plot `certificate_mean`, `true_distortion_mean`, `oracle_distortion_mean` and their `*_ci95_lower/upper`; annotate `n_admitted` out of `admitted_n`. Use the calibration table to avoid duplicating assignment results across profiles/cohorts. Each law/capacity has 100 independent blocks; continuous bars are normal95 and individual admission rates use CP95. The budget is G=2, aggregate ceiling 3, hence certificate threshold 1.5. Capacity oracles are diagnostics, and identity is unavailable as a smaller-capacity fallback.

### Figure 4 — Empirical minimum-count operating curves (`fig:precision-operating-curves`)

Configuration/prerequisite: Table 3. Execute `python -m sca_reproduction run precision` and `python -m sca_reproduction report precision`, then:

```sh
python -m sca_reproduction plot precision --workspace outputs/paper
```

Output: `paper_figures/precision_operating_curves.{pdf,png}`. Compression: `precision/tables/assignment_summary.csv`, all laws and caps, `compression_pp_mean` and its normal95 bounds. Widths: `method_width_target_summary.csv`; balanced/aligned, weak/aligned and IHDP/high_heterogeneity; `z_global` at `m_ratio=16,256`, `raw_stratified` and `raw_pooled` at 256; use `width_mean` and `width_ci95_lower/upper`. Each point uses **all 100 blocks**, not `admitted_width_mean`; retain caps 1 and 1.1 and the fixed target .20. Caps/profiles/cohorts/methods repeat those blocks. `paired_cap_contrasts.csv` records paired changes with `left_cap=2`, `right_cap=3/2`, `metric=z_global_width`; interpret the saved left-minus-right direction.

### Figure 5 — SCA computation from sufficient counts (`fig:computational-cost`)

Configuration: `configs/jmlr_computation_v1.json`; inputs are generated from source, including the K64 release-count bridge. Execute `python -m sca_reproduction run computation` and `python -m sca_reproduction report computation`, then:

```sh
python -m sca_reproduction plot computation --workspace outputs/paper
```

Output: `paper_figures/computational_cost.{pdf,png}`. Inputs: `computation/tables/primary_overview.csv` (both laws, K=16/32/64/128/256, 20 cases each) and `capacity_overview.csv` (K=64/128/256, five matched cases each). Plot `total_seconds_median` with `total_seconds_q25/q75`; these are **IQRs, not confidence intervals**. Primary totals run from loaded sufficient counts through aggregate inference; capacity totals cover the provider across its five reporting limits, without outcomes. Exclude startup/shutdown, serialization, subsequent raw-reference and repeated-cohort work. Whole-process lifetime memory includes those later tasks; it is not isolated analyst memory. Inspect `cases.csv`, `stage_summary.csv`, `timing_accounting.csv` and `status_denominators.csv` for actual completion, censored cases and additive accounting. Additional cohorts and capacity runs reuse inputs and do not add independent calibration datasets. Published seconds/bytes are hardware-dependent measurements, not equality targets for another machine.

## Headline checks and interpretation

- **Compression:** Table 1 uses the full assignment panel; Table 2 uses the held-out evaluation replications; Figure 4 uses the independent release panel. Do not interchange these denominators. Read SCA as combined-family Coupled CP (`coupled`, `original`, or F in the corresponding records). The search-gain headline pairs `100*(count_fixed-count_combined)/K` within each assignment dataset at unknown masses and cap `3/2`; use the fixed-grid paired variance, rather than subtracting the two compression intervals.
- **Admission:** Figure 3's balanced B=8 and IHDP B=4 comparisons use `verified_capacity` versus `equal_calibration_mass`. The separate certificate-construction comparison (75 versus 67/67 on IHDP) is in `efficiency/tables/release_condition_summary.csv`, not the equal-mass baseline comparison (75 versus 7).
- **Recovery:** Figure 1's recovery counts/rates use 300 paths per condition; repeated sizes and mass modes do not enlarge that denominator. The heterogeneous-law `unsafe_certification_*` fields retain all adverse safety checks.
- **Decomposition and precision:** Table 3 and `decomposition_summary.csv` preserve signed stages. `method_width_target_summary.csv` retains the negative .20-attainment result for every method condition, including pooled raw; `admitted_target_*` and `admitted_and_miss_*` use the unconditional 100-block denominator. Admission-conditioned widths are descriptive. Assignment validity is simultaneous over candidate selections, but outcome guarantees remain pointwise for the prespecified analysis.

## Verification status

Paper numbering, labels, filters and the PDF identifier were checked against the active manuscript. The unchanged numerical implementation retains the earlier release's 39-test validation. This update passed 21 targeted plotting/table/CLI tests. Temporary historical-table checks reproduced the three tables' means and relevant intervals, all five figures' coordinates, all 30 admission counts, all 144 recovery conditions (300 paths each), and exact accounting/nesting for Table 3's 200 profile-block records. Imported split completions retain their original status labels. Rightmost admission labels are kept inside the figure canvas. These establish execution and agreement with historical records, **not fresh regeneration of complete reported results**; no calibration, selection, outcome inference or timing experiment was rerun. A clean-checkout check of the new helpers is pending before publication. No unresolved numerical discrepancy currently requires a research rerun.
