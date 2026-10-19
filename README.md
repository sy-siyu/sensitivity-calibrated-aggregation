# Sensitivity Calibration Under State Aggregation

Code for **sensitivity-calibrated aggregation (SCA)** and the paper *Sensitivity Calibration Under State Aggregation*. SCA selects an outcome-blind grouping of a declared finite state space and certifies its sensitivity distortion. The principal procedure uses the full assignment-calibration sample, a Coupled Clopper–Pearson (CP) certificate, and complete search over two fixed-tree pruning families plus estimated-propensity contiguous partitions.

The repository includes finite-sample certification studies, five-fraction LTT comparisons, propensity-logit clustering, an observed RHC illustration, and aggregate-only binary or bounded-outcome inference.

## Install

Python **3.13** is the tested environment; macOS and Linux are supported. No GPU is needed.

```sh
git clone https://github.com/sy-siyu/sensitivity-calibrated-aggregation.git
cd sensitivity-calibrated-aggregation
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

NumPy, SciPy and python-flint are pinned. Main-paper plotting is optional: `python -m pip install -e '.[plots]'`.

## Select an aggregation

Supply integer state sizes `N[s]` and treated counts `T[s]`, including declared states with zero observations:

```python
from sca import SCA

N = [4000, 4000, 4000, 4000]
T = [1200, 1600, 2400, 2800]
fit = SCA().fit(N, T, cap="3/2")
print(fit.partition, fit.certificate, fit.status)
print(fit.optimization_completed)
```

The default **assignment-calibration error** is `1/40`; use `SCA(alpha="1/20")` for a different allocation. Outcome inference takes a separate `overall_alpha`, described below.

SCA supports two objectives over the same realized family:

| Objective | Call | Optimization |
|---|---|---|
| Fixed-cap count minimization | `SCA().fit(N, T, cap="3/2")` | Minimize the number of groups subject to certificate at most the cap. |
| Fixed-capacity certificate minimization | `SCA().fit(N, T, capacity=2)` | Minimize the certificate over partitions with at most two groups. |

The objectives can return different partitions. A partition belongs to one constituent family; tree branches are never mixed across families. “Complete search” means complete within this declared family, not over every set partition. Check `status` and `optimization_completed`: a mathematical infinite envelope, an unresolved numerical calculation, and a completed finite result are different outcomes.

## Infer outcomes and release aggregates

For binary outcomes, use treated-success counts:

```python
from sca import analyze_release

artifact = fit.release(outcome_N, outcome_T, treated_successes)
analysis = analyze_release(artifact)
```

This API exports every selected group, including groups with no treated observations, and reconstructs from anonymous aggregate counts and certified summaries.

For general confidence allocation, bounded outcomes, mass drift, and analyst reconstruction, supply treated counts and treated-outcome sums:

```python
from fractions import Fraction
from sca import SCA, analyze_outcome_release

fit = SCA().fit((80, 80), (24, 56), cap="10")
artifact = fit.release_outcomes(
    outcome_treated=(30, 40),
    treated_outcome_sums=(Fraction(25), Fraction(65)),
    branch="bounded",
    outcome_bounds=("-2", "3"),
    overall_alpha="1/20",
    reporting_capacity=1,
    sensitivity_ceiling="100",
    decimal_precision=40,
)
analysis = analyze_outcome_release(artifact)
```

`confidence_allocation(...)` exposes the allocation rule. With no component overrides, `overall_alpha` is split as one half for assignment calibration and one quarter for each raw and aggregate outcome family. An explicit component allocation must agree with the SCA calibration error used by `fit` and fit within the declared overall error.

The `binary` branch uses exact CP intervals and the sharp binary population formulas. The `bounded` branch requires prespecified finite outcome bounds, uses simultaneous Hoeffding mean intervals, and returns valid mean-based outer enclosures that need not be sharp for a nonbinary distribution. Calculations are intersected on the normalized `[0,1]` scale and then mapped exactly to the requested outcome scale. The release records the outward decimal precision; reconstruction reuses it and rejects an incompatible override.

`mass_drift=zeta` permits supplied state-mass drift satisfying `1-zeta <= p'_s/p_s <= 1+zeta` while statewise propensities remain unchanged. The implementation projects the corrected target mass and propensity ranges and inflates nonsingleton certificates before checking admission. `release_outcomes` enforces both the declared reporting capacity and corrected sensitivity ceiling before returning an artifact.

The artifact contains anonymous group aggregates and projected ranges, not state memberships or the provider’s raw calibration region. Analyst reconstruction therefore verifies internal arithmetic and admission from the release, but still trusts the provider to have used the promised records, outcome-blind selection, projections, and certificates. Coverage is an unconditional fixed-analysis repeated-sampling guarantee; with admission it controls the joint event “admitted and missed,” not coverage conditional on admission.

Calibration and outcome cohorts must be independent. With `mass_drift="0"`, they share the complete assignment law `P(S,A)`, including state frequencies and conditional treatment propensities; matching only `P(A|S)` is insufficient. A positive supplied drift bound relaxes the state-frequency equality under its documented coordinatewise ratio constraint while keeping statewise propensities fixed.

Run the three small user-facing examples:

```sh
python -m sca_reproduction example
python -m sca_reproduction example-bounded
python -m sca_reproduction example-drift
```

## Reproduce the studies

Prepare both public inputs. Downloads are checksum-verified and ignored by Git:

```sh
python -m sca_reproduction prepare-data --directory data
```

IHDP is used only as a covariate reference population for a semi-synthetic benchmark. RHC is an observed treatment/outcome illustration with a frozen split and preprocessing rule. See [data sources, scopes, and terms](docs/DATA.md).

Start with the bounded execution check. It exercises every current runner but is not scientific evidence for the full panels:

```sh
python -m unittest discover -s tests -v
python -m sca_reproduction smoke --workspace outputs/smoke \
  --data data/ihdp_npci_1.csv --rhc-data data/rhc.csv
```

The current full workflow is ordered as follows:

| Study | Purpose and prerequisite | Command |
|---|---|---|
| `assignment` | Fresh 4,800-dataset assignment panel | `python -m sca_reproduction run assignment` |
| `release` | Capacity-constrained releases; requires IHDP | `python -m sca_reproduction run release` |
| `certification` | Positive-margin same-family oracles and sparse-mixture lower bound; requires assignment | `python -m sca_reproduction run certification` |
| `relevance` | Five LTT fractions, coherent A–F evaluation, global logit clustering, RHC; requires assignment, release, and RHC | `python -m sca_reproduction run relevance` |
| `recovery`, `efficiency` | Recovery and certificate diagnostics; require assignment/release as documented | `python -m sca_reproduction run recovery`, then `run efficiency` |
| `precision` | Compression–distortion–precision curves; requires release | `python -m sca_reproduction run precision` |
| `outcome` | Independent binary outcome panel and raw references | `python -m sca_reproduction run outcome` |
| `computation` | Fresh sequential scaling diagnostics | `python -m sca_reproduction run computation` |

All commands default to `--workspace outputs/paper --config configs/paper.json`. `run all` and `report all` follow the current sequence:

```sh
python -m sca_reproduction run all --workspace outputs/paper
python -m sca_reproduction report all --workspace outputs/paper
python -m sca_reproduction paper-tables --workspace outputs/paper
```

Generate the current empirical Figures 2–5 individually:

```sh
python -m sca_reproduction plot efficiency --workspace outputs/paper
python -m sca_reproduction plot split --workspace outputs/paper
python -m sca_reproduction plot release --workspace outputs/paper
python -m sca_reproduction plot precision --workspace outputs/paper
```

`plot split` reads the current `relevance` report when present. `plot computation` generates an additional computational diagnostic. The separate `run split` command reproduces the earlier two-fraction comparison; `run all` uses the five-fraction `relevance` study.

Exact rational rows are stored in JSON outputs; CSV summaries are rounded/display-oriented. Reporters refuse to treat a bounded prefix as a completed full panel. See [reproduction details](docs/REPRODUCTION.md) and the [paper-to-code map](docs/PAPER_RESULTS.md).

## Current comparisons and scope

| Role | Implementation |
|---|---|
| Proposed procedure | Full-sample SCA, principal Coupled CP certificate, complete combined-family search. |
| LTT comparison | Five development fractions `{1/10,1/5,1/3,1/2,2/3}`; the globally selected `1/3` fraction is frozen before the coherent 4,320-dataset A–F evaluation. |
| Capacity baselines | Whole-state equal-calibration-mass grouping and globally optimal one-dimensional clustering of smoothed propensity logits; certification remains Coupled CP. |
| Observed illustration | Frozen RHC preprocessing and 4,014/1,721 calibration/outcome split; the primary capacity-four result abstains rather than releasing an inadmissible interval. Twenty additional splits are dependent stability checks. |
| Oracle diagnostics | Population truth is evaluated only after operational selection and never enters SCA certification or selection. |

The optional command below verifies a five-state counterexample at cap `19/10` by enumerating all 52 set partitions and all 16 contiguous cuts. The unrestricted optimum is two groups; the contiguous optimum is three. See the [paper-to-code map](docs/PAPER_RESULTS.md) for its scope.

```sh
python -m experiments.jmlr_five_state_counterexample_v1
```

## Repository scope and attribution

`sca/` is the public interface, `experiments/` contains numerical kernels and study code, and `sca_reproduction/` manages public data, execution, and reporting. Source, versioned configurations, tests, examples, and documentation are tracked. Downloaded data and generated outputs are excluded from version control.

The project is licensed under the [MIT License](LICENSE). Downloaded data and dependencies retain their own terms; see [licensing details](LICENSING.md) and [third-party attribution](THIRD_PARTY_NOTICES.md).
