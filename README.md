# Sensitivity Calibration Under State Aggregation

Code for **sensitivity-calibrated aggregation (SCA)**. SCA selects an outcome-blind grouping of a declared finite state space and certifies the resulting sensitivity distortion. Its principal workflow uses the **full assignment calibration sample**, a **Coupled Clopper–Pearson (CP)** distortion certificate, and complete search within a combined family of two fixed-tree pruning families and estimated-propensity contiguous partitions.

Baselines and component ablations are available through the experiment runners.

## Install

Python **3.13** is the tested environment; macOS and Linux are the supported execution targets (the process runners use POSIX locks). No GPU is needed. Use a checkout for reproduction so that the versioned configurations and examples are available:

```sh
git clone https://github.com/sy-siyu/sensitivity-calibrated-aggregation.git
cd sensitivity-calibrated-aggregation
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The numerical dependencies are pinned: NumPy, SciPy and python-flint (FLINT/Arb interval arithmetic). Main-paper plotting is optional: install it with `python -m pip install -e '.[plots]'`.

## Use SCA

Supply integer calibration counts `N[s]` (state size) and `T[s]` (treated count), including declared states with zero observations:

```python
from sca import SCA

N = [4000, 4000, 4000, 4000]
T = [1200, 1600, 2400, 2800]
fit = SCA().fit(N, T, cap="3/2")
print(fit.partition, fit.certificate, fit.status)
print(fit.optimization_completed)
```

The default calibration error is `1/40`, compatible with the outcome allocation below. Set `alpha` explicitly for another allocation, for example `SCA(alpha="1/20")` for assignment-only analysis.

Run the short synthetic example, including an independent binary-outcome cohort and aggregate-only analysis:

```sh
python -m sca_reproduction example
```

SCA supports two **different operating objectives** over the same realized combined family \(\mathcal F_n\):

| Objective | Call | Optimization |
|---|---|---|
| Fixed-cap count minimization | `SCA().fit(N, T, cap="3/2")` | Minimize \(|\pi|\) subject to \(U_n^v(\pi)\le c\). |
| Fixed-capacity certificate minimization | `SCA().fit(N, T, capacity=2)` | Minimize \(U_n^v(\pi)\) subject to \(|\pi|\le B\). |

They can return different partitions. A partition belongs to one constituent family; tree branches are never mixed across families. Complete search means optimization within this stated family, not all partitions. Ties use deterministic local recurrence rules.

Check the returned status and completion metadata. A completed infinite envelope is different from unresolved numerical work. Threshold rejection bounds are not completed upper certificates. Safe identity fallback does not prove count optimality, and identity is unavailable as a capacity fallback when `B < K`. At fixed capacity, `fit.admissible(raw_budget="2", aggregate_budget="3")` additionally checks the completed certificate and count against the release constraint. Choose the raw sensitivity budget and aggregate ceiling before examining outcomes.

An independent binary-outcome cohort can be exported using `fit.release(out_N, out_T, treated_successes)`. `sca.analyze_release(artifact)` uses only anonymous aggregate counts, projected bounds, completed cell distortions and confidence metadata. Every selected group is included, including empty treated groups. The standard allocation is calibration `1/40`, raw outcomes `1/80`, aggregate outcomes `1/80`. Assignment validity holds simultaneously across candidate partitions/capacities on its common event. Outcome guarantees retain their pointwise scope for a prespecified capacity, profile and cohort size; admission does not imply nominal conditional coverage.

The calibration and outcome cohorts must be independent samples sharing the same population assignment law \(P_{S,A}\), including both state frequencies and conditional treatment probabilities. Their observed counts need not match. Sharing only \(P(A\mid S)\) is insufficient because changes in state frequencies can change aggregate propensities.

## Reproduce the experiments

The runners generate counts, splits, pilot proposals, confidence regions, certificates and outcome cohorts in an ignored output workspace.

First prepare the public covariates needed for the capacity and precision studies:

```sh
python -m sca_reproduction prepare-data --directory data
```

This downloads the CEVAE IHDP CSV and checks its SHA-256 before use. The original treatments and outcomes are ignored. See [data construction](docs/DATA.md) for variables, exact ties and the finite benchmark population.

A bounded execution check covers generation, certification, both search objectives, aggregate-only outcomes, the comparison adapters and numerical table generation:

```sh
python -m unittest discover -s tests -v
python -m sca_reproduction smoke --workspace outputs/smoke
```

`configs/smoke.json` runs a small execution check. Use `configs/paper.json` for the full experimental panels.

| Paper experiment / purpose | Configuration | Full run command (after prerequisites) |
|---|---|---|
| Assignment calibration, search-family ablations, same-family oracle diagnostics; split-LTT baseline and development fraction choice | `configs/jmlr_verified_pipeline_v1.json`, `configs/jmlr_statistical_baselines_v1.json` | `python -m sca_reproduction run assignment` |
| Capacity-constrained SCA and equal-mass propensity baseline; independent outcome cohorts | `configs/jmlr_release_capacity_v1.json` | `python -m sca_reproduction run release` |
| Confidence allocation / certificate-construction ablations, occupancy and oracle-gap diagnostics | `configs/jmlr_efficiency_gap_v1.json` | `python -m sca_reproduction run efficiency` (requires assignment and release) |
| Independent nested recovery paths | same efficiency configuration | `python -m sca_reproduction run recovery` |
| Held-out certification, stopping and pilot-proposal/order ablations; split-LTT baseline | `configs/jmlr_split_ablation_v1.json` | `python -m sca_reproduction run split` (requires assignment) |
| Compression–distortion–precision operating curves | `configs/jmlr_precision_frontier_v1.json` | `python -m sca_reproduction run precision` (requires release) |
| Outcome inference, raw stratification and pooled-raw references | `configs/jmlr_outcome_inference_v1.json` | `python -m sca_reproduction run outcome` |
| Fresh sequential computational scaling and repeated-cohort costs | `configs/jmlr_computation_v1.json` | `python -m sca_reproduction run computation` |

All commands default to `--workspace outputs/paper --config configs/paper.json`. To execute the dependency sequence and generate numerical summaries/tables:

```sh
python -m sca_reproduction run all --workspace outputs/paper
python -m sca_reproduction report all --workspace outputs/paper
```

For a single study, use `report STUDY` with the same workspace. Full reporters require complete panels; bounded runs produce partial diagnostics. Outputs include input/protocol files, per-dataset records, `tables/*.csv`, and JSON summaries.

The [paper-to-code map](docs/PAPER_RESULTS.md) identifies the current three main tables and five main figures by number and stable LaTeX label. After the required complete reports exist, generate them with:

```sh
python -m pip install -e '.[plots]'
python -m sca_reproduction paper-tables --workspace outputs/paper
python -m sca_reproduction plot all --workspace outputs/paper
```

The commands read completed reports and write `paper_tables/` and `paper_figures/` inside the workspace. Plot one study with `plot efficiency`, `plot split`, `plot release`, `plot precision`, or `plot computation`.

See [reproduction details](docs/REPRODUCTION.md) for seed schedules, panel sizes, restart behavior, numerical settings, resource requirements and validation scope.

## Baselines, ablations and references

| Role | Implementations retained for the experiments |
|---|---|
| Proposed procedure | **SCA**: full sample, principal Coupled CP certificate, complete combined-family search. |
| Baselines | Split LTT (with its original stopping rule); whole-state equal-calibration-mass grouping along estimated propensity order. |
| Ablations | Held-out coupled certification; first-failure stopping; saved pilot-proposal restriction; pilot-order family; restricted tree/contiguous families; independent-CP and allocation/pooled-assignment certificate constructions. |
| Outcome references | Raw stratification and pooled raw, which retain stronger raw-state information. Pooled raw can reuse aggregation/pooling and is not an aggregate-only competitor. |
| Oracle diagnostics | Same-realized-family choices using true population distortion/counts. Truth is evaluated after operational selection and never enters SCA certification or selection. |

Baselines and ablations are configured through the experiment runners. The [procedure mapping](docs/REPRODUCTION.md#procedure-definitions) connects their output identifiers to the paper labels.

## Repository scope and attribution

`sca/` provides the public interface, `experiments/` implements the numerical routines and experiments, and `sca_reproduction/` manages data generation, execution and reporting. Configurations, tests, examples and documentation are in the corresponding directories.

Generated data, results and figures remain outside version control. The project is licensed under the [MIT License](LICENSE); see [licensing details](LICENSING.md) and [third-party attribution](THIRD_PARTY_NOTICES.md).
