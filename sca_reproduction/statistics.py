"""Monte Carlo summaries for the submission studies."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy.stats import beta, norm, t


def paired(values: Any) -> dict[str, Any]:
    """Pointwise Student interval over independent paired original datasets."""
    x = np.asarray(values, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Paired Monte Carlo observations must be finite")
    n = len(x)
    if not n:
        return {"n": 0, "mean": None, "sample_sd": None, "MCSE": None, "Student95_lower": None,
                "Student95_upper": None, "zero_sample_variance": None, "minimum": None, "maximum": None}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    radius = float(t.ppf(.975, n - 1)) * se if se is not None else None
    return {"n": n, "mean": mean, "sample_sd": sd, "MCSE": se,
            "Student95_lower": mean - radius if radius is not None else None,
            "Student95_upper": mean + radius if radius is not None else None,
            "zero_sample_variance": sd == 0 if sd is not None else None,
            "minimum": float(x.min()), "maximum": float(x.max())}


def rate(values: Any) -> dict[str, Any]:
    """Per-condition CP Monte Carlo interval; never applied to pooled tasks."""
    x = np.asarray(values, bool)
    n, successes = len(x), int(x.sum())
    if not n:
        return {"n": 0, "successes": 0, "rate": None, "CP95_lower": None, "CP95_upper": None}
    return {"n": n, "successes": successes, "rate": successes / n,
            "CP95_lower": 0. if successes == 0 else float(beta.ppf(.025, successes, n - successes + 1)),
            "CP95_upper": 1. if successes == n else float(beta.ppf(.975, successes + 1, n - successes))}


def grid_interval(condition_rows: list[dict[str, Any]], prefix: str, expected_conditions: int = 48) -> dict[str, Any]:
    """Analytic stratified MC variance, fixed equal weights over conditions.

    Variance of H^-1 sum_h mean_h is H^-2 sum_h s_h²/n_h. The normal
    interval is an approximate simulation-error summary, with no task sampling.
    """
    rows = [row for row in condition_rows if row[prefix + "_n"]]
    if not rows:
        return {"conditions": 0, "expected_conditions": expected_conditions, "paired_datasets": 0,
                "mean": None, "MCSE": None, "analytic_Normal95_lower": None, "analytic_Normal95_upper": None}
    h = len(rows)
    mean = sum(row[prefix + "_mean"] for row in rows) / h
    variance = sum(row[prefix + "_sample_sd"] ** 2 / row[prefix + "_n"] for row in rows) / h ** 2 if all(row[prefix + "_sample_sd"] is not None for row in rows) else None
    se = math.sqrt(variance) if variance is not None else None
    radius = float(norm.ppf(.975)) * se if se is not None else None
    return {"conditions": h, "expected_conditions": expected_conditions, "paired_datasets": sum(row[prefix + "_n"] for row in rows),
            "all_conditions_present": h == expected_conditions,
            "mean": mean, "MCSE": se, "analytic_Normal95_lower": mean - radius if radius is not None else None,
            "analytic_Normal95_upper": mean + radius if radius is not None else None,
            "scope": "equal-weight fixed-condition mean; analytic simulation MC variance; not variation over a population of tasks"}
