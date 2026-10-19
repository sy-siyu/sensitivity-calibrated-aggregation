# Public data inputs

The current reproduction uses two different public inputs for different scientific roles:

- **IHDP/CEVAE:** covariates only, to define a finite semi-synthetic reference population. Original treatment and outcomes are ignored.
- **RHC:** observed treatment and 30-day survival, for one explicitly limited observed-data illustration.

Both files are downloaded on demand, verified by SHA-256, and excluded from version control. The repository does not redistribute either dataset.

```sh
python -m sca_reproduction prepare-data --directory data --dataset all
```

Use `--dataset ihdp` or `--dataset rhc` to prepare only one input. For offline setup, place the exact files at `data/ihdp_npci_1.csv` and `data/rhc.csv`; the same validation runs before use.

## IHDP covariate reference population

Source: [CEVAE IHDP replication-1 CSV](https://github.com/AMLab-Amsterdam/CEVAE/blob/master/datasets/IHDP/csv/ihdp_npci_1.csv)

```text
SHA-256: 2bdc8b94fdf2f669b1d3d49325fd9359d88418566545003867841312a1653411
Rows:    747
Columns: 30
```

The [CEVAE loader](https://github.com/AMLab-Amsterdam/CEVAE/blob/master/datasets.py) identifies the first five columns as treatment/outcome quantities and the remaining columns as covariates. The [NPCI definitions](https://github.com/vdorie/npci/blob/master/examples/ihdp_sim/data.R) provide the covariate order. This benchmark uses only zero-based CSV columns 5, 10, and 11: birth weight, maternal age, and infant sex. It uses the supplied coordinates and does not reinterpret standardized values as physical units.

The complete 747-record reference population is fixed before sampling. States are birth-weight quartile × maternal age strictly above its median × binary sex. Quantiles use exact decimal fractions, linear interpolation at index `(747-1)q`, and equality belongs to the lower bin. Occupied tuples are sorted lexicographically into immutable IDs, giving 16 supported states with mass `reference count / 747`.

Assignments and outcomes are newly generated:

- assignment propensity: `(8 + 3*birthweight_bin + 2*age_bit + sex)/32`;
- low-heterogeneity treated mean: `(12 + birthweight_bin + age_bit + sex)/32`;
- high-heterogeneity treated mean: `1/8` or `7/8` by tuple parity;
- control mean: `1/2`.

Calibration and outcome cohorts independently resample reference-record IDs. The experiment is therefore covariate-derived and semi-synthetic; it is not an analysis of the original IHDP treatment or outcome columns and does not estimate an IHDP treatment effect.

The source repositories should be consulted for their current terms and citations. The CEVAE repository does not supply a separate license statement for this downloaded CSV in the public path used here. The public command downloads the input directly rather than relicensing or bundling it; users remain responsible for applicable upstream terms. NPCI code and definitions retain their upstream terms.

## RHC observed-data illustration

Sources:

- [Vanderbilt Biostatistics dataset catalogue and permission/attribution statement](https://hbiostat.org/data/)
- [RHC dataset description and variable definitions](https://hbiostat.org/data/repo/rhc.html)
- [CSV](https://hbiostat.org/data/repo/rhc.csv)

```text
SHA-256: 9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8
Rows:    5,735
```

The catalogue grants permission to use the hosted datasets and requests the original-paper citation plus the attribution:

> Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.

Original study citation: Connors AF Jr et al., “The effectiveness of right heart catheterization in the initial care of critically ill patients,” *JAMA* 276(11):889–897, 1996, [doi:10.1001/jama.276.11.889](https://doi.org/10.1001/jama.276.11.889), [PMID 8782638](https://pubmed.ncbi.nlm.nih.gov/8782638/).

The frozen public preprocessing requires nonmissing valid values for `age`, `sex`, `ca`, `swang1`, `dth30`, and `ptid`; all 5,735 source rows pass. It defines:

- treatment `A=1`: `swang1 == "RHC"`, indicating RHC within the first 24 hours after study entry;
- outcome `Y=1`: `dth30 == "No"`, interpreted as 30-day survival;
- eight declared states: age at least 65 × male sex × any baseline cancer (`ca` is `Yes` or `Metastatic`), ordered lexicographically.

No cutpoint is fitted and no outcome is used to design the state catalogue. Every one of the `2^3` tuples remains declared even if absent from a split.

For split index `j`, `PCG64(SeedSequence([20261019,501,j]))` generates one uniform permutation. The first 4,014 records form the assignment/calibration cohort; the remaining 1,721 form the disjoint outcome cohort. Split 0 is primary. Splits 1–20 are dependent stability checks over the same 5,735 people, not independent datasets or repeated coverage trials.

This observational illustration targets `E[Y(1)]` under the stated sensitivity model. It is not a randomized comparison, a clinical recommendation, a validation of the state design, or evidence that admissions/abstentions have nominal conditional coverage. The primary capacity-four comparison abstains because the sensitivity ceiling is not met; no inadmissible interval is presented as operational.

## Controlled inputs

The remaining controlled experiments require no external download. Their laws, exact stored-binary conventions, fixed trees, and random streams are generated by the tracked source and versioned configurations described in [REPRODUCTION.md](REPRODUCTION.md).
