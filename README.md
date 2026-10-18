# Sensitivity Calibration Under State Aggregation

Code for **sensitivity-calibrated aggregation (SCA)**. SCA selects an outcome-blind grouping of a declared finite state space and certifies the resulting sensitivity distortion. Its principal workflow uses the **full assignment calibration sample**, a **Coupled Clopper–Pearson (CP)** distortion certificate, and complete search within a combined family of two fixed-tree pruning families and estimated-propensity contiguous partitions.

Coupled CP is the certificate construction inside SCA. It is not a second proposed method. The repository also implements baselines and component ablations needed to reproduce the paper. This is research code for the paper; no journal acceptance is claimed.

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

The numerical dependencies are pinned: NumPy, SciPy and python-flint (FLINT/Arb interval arithmetic). Plotting code and plotting dependencies are omitted.

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

This uses every supplied assignment count. Population propensities, population masses, outcomes, and pilot proposals are not inputs. The default calibration error is `1/40`, compatible with the outcome allocation below. For an assignment-only analysis, `SCA(alpha="1/20")` changes the specified calibration error, not the method.

Run the short synthetic example, including an independent binary-outcome cohort and aggregate-only analysis:

```sh
python -m sca_reproduction example
```

SCA supports two **different operating objectives** over the same realized combined family \(\mathcal F_n\):

| Objective | Call | Optimization |
|---|---|---|
| Fixed-cap count minimization | `SCA().fit(N, T, cap="3/2")` | Minimize \(|\pi|\) subject to \(U_n^v(\pi)\le c\). |
| Fixed-capacity certificate minimization | `SCA().fit(N, T, capacity=2)` | Minimize \(U_n^v(\pi)\) subject to \(|\pi|\le B\). |

They can return different partitions. A partition belongs to one constituent family; tree branches are never mixed across families. Complete search means optimization within this stated family, not all partitions. Inherited local recurrence ties are deterministic; no global secondary optimum among all minimum-count partitions is asserted.

Check the returned status and completion metadata. A completed infinite envelope is different from unresolved numerical work. Threshold rejection bounds are not completed upper certificates. Safe identity fallback does not prove count optimality, and identity is unavailable as a capacity fallback when `B < K`. At fixed capacity, `fit.admissible(raw_budget="2", aggregate_budget="3")` additionally checks the completed certificate and count against the release constraint. These budgets are experimental choices, not a privacy guarantee or an institutional reporting rule.

An independent binary-outcome cohort can be exported using `fit.release(out_N, out_T, treated_successes)`. `sca.analyze_release(artifact)` uses only anonymous aggregate counts, projected bounds, completed cell distortions and confidence metadata. Every selected group is included, including empty treated groups. The standard allocation is calibration `1/40`, raw outcomes `1/80`, aggregate outcomes `1/80`. Assignment validity holds simultaneously across candidate partitions/capacities on its common event. Outcome guarantees retain their pointwise scope for a prespecified capacity, profile and cohort size; admission does not imply nominal conditional coverage. The analyst cannot reconstruct the provider's detailed raw-data certification evidence from this release.

## Reproduce the experiments

No saved results or private archives are distributed or required. The runners generate counts, splits, pilot proposals, confidence regions, certificates and outcome cohorts from source. Newly generated inputs and evidence are written to an ignored workspace. Full reproduction recomputes prerequisites rather than importing published answers.

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

`configs/smoke.json` explicitly reduces support/replication counts or selects bounded prefixes. Its results are execution diagnostics, not the full paper panel. `configs/paper.json` uses the original retained study configurations without those changes.

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

For a single study, use `report STUDY` with the same workspace. Full reporters check complete panel sizes; explicitly bounded runs use partial diagnostics. Outputs include generated input/protocol files, per-dataset records, `tables/*.csv`, and JSON summaries. No generated outputs belong in Git. There are no plotting commands.

See [reproduction details](docs/REPRODUCTION.md) for seed schedules, full panel sizes, restart behavior, numerical policies, cost limits and exact-regeneration scope. Full studies can take many CPU-hours and produce substantial exact-number evidence; they were **not rerun for this code release**. A smoke check establishes execution, not the paper's statistical conclusions.

## Baselines, ablations and references

| Role | Implementations retained for the experiments |
|---|---|
| Proposed procedure | **SCA**: full sample, principal Coupled CP certificate, complete combined-family search. |
| Baselines | Split LTT (with its original stopping rule); whole-state equal-calibration-mass grouping along estimated propensity order. |
| Ablations | Held-out coupled certification; first-failure stopping; saved pilot-proposal restriction; pilot-order family; restricted tree/contiguous families; independent-CP and allocation/pooled-assignment certificate constructions. |
| Outcome references | Raw stratification and pooled raw, which retain stronger raw-state information. Pooled raw can reuse aggregation/pooling and is not an aggregate-only competitor. |
| Oracle diagnostics | Same-realized-family choices using true population distortion/counts. Truth is evaluated after operational selection and never enters SCA certification or selection. |

The public default has no switch to these comparisons. They live in experiment configurations and runners. Legacy internal module names and raw record IDs are retained to preserve numerical provenance; display labels and [the procedure mapping](docs/REPRODUCTION.md) explain their roles.

## Repository scope and attribution

`sca/` is the small public interface. `experiments/` holds the shared verified arithmetic and retained experiment kernels; `sca_reproduction/` regenerates their input dependencies and orchestrates fresh runs. `configs/`, `tests/`, `examples/` and `docs/` complete the release. No neural-training or unrelated release machinery is included.

This repository contains code, configurations, tests and instructions only: no saved experimental evidence, generated tables/figures, manuscript, or original repository history. See [licensing status](LICENSING.md) and [third-party attribution](THIRD_PARTY_NOTICES.md). A project license has not yet been selected; this release does not silently relicense the research code or dependencies.
