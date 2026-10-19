# Attribution and external dependencies

This repository contains the authors' implementation of SCA and the experiments for *Sensitivity Calibration Under State Aggregation*.

Installed dependencies retain their own notices and terms:

| Dependency | Upstream source and licensing information |
|---|---|
| NumPy | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt); BSD-3-Clause, with additional licenses for bundled components. |
| SciPy | [SciPy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt); BSD-3-Clause and bundled-component notices. |
| Matplotlib (optional plots) | [Matplotlib license](https://matplotlib.org/stable/users/project/license.html); Matplotlib's PSF-based license and bundled-component notices. |
| python-flint | [python-flint license](https://github.com/flintlib/python-flint/blob/master/LICENSE); MIT. Its linked FLINT/Arb libraries retain their own terms. |
| FLINT | [FLINT source/license](https://github.com/flintlib/flint); see the installed distribution and upstream licenses for FLINT and linked arithmetic libraries. |

The IHDP covariate input is downloaded on demand from [AMLab Amsterdam's CEVAE repository](https://github.com/AMLab-Amsterdam/CEVAE), with variable definitions traced to [Vincent Dorie's NPCI repository](https://github.com/vdorie/npci). Cite and respect the upstream data/source terms. Neither the dataset nor upstream implementation is copied into this release. The finite benchmark uses only covariates and generates new assignment/outcome laws; it does not reproduce original IHDP treatment effects.

The split-LTT comparator is the paper's finite-state specialization of learn-then-test, not a general implementation of every LTT variant. Its proposal scoring and original first-failure rule are retained. Clopper–Pearson refers to exact binomial confidence limits; the coupled construction combines assignment summaries in the distortion envelope.
