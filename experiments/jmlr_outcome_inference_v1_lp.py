"""Exact outward certificates for the pooled raw outcome LP relaxation.

HiGHS proposes dual multipliers only.  For a maximization objective c, every
rational y >= 0 and unrestricted z proves

    c'x <= y'b + z'h + sum_j max(0, c_j - (A'y + E'z)_j)

on A x <= b, E x = h and 0 <= x <= 1.  In particular, numerical dual
feasibility and the solver's reported objective are never proof premises.
The negative objective gives the lower endpoint.  All input coefficients,
multiplier nonzeros, signed residual nonzeros and resulting bounds are saved
as rational strings; replay does not invoke an optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
import math
import time
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import csr_matrix


VERSION = "jmlr_outcome_inference_v1_lp"
ZERO, ONE = Q(0), Q(1)
SparseRow = tuple[tuple[int, Q], ...]
Box = tuple[tuple[Q, Q], ...]


def _q(value: Any) -> Q:
    """Keep binary floating inputs exact, and accept the study JSON wrappers."""
    if isinstance(value, bool):
        raise ValueError("A boolean is not an exact LP coefficient")
    if isinstance(value, dict):
        if set(value) == {"__fraction__"}:
            return Q(value["__fraction__"])
        if set(value) == {"numerator", "denominator"}:
            return Q(int(value["numerator"]), int(value["denominator"]))
        raise ValueError("Unrecognized rational encoding")
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("LP coefficients must be finite")
        return Q.from_float(float(value))
    return Q(value)


def _box(values: Iterable[Sequence[Any]], name: str) -> Box:
    result = []
    for pair in values:
        if len(pair) != 2:
            raise ValueError(f"{name} entries must be lower/upper pairs")
        lo, hi = map(_q, pair)
        if not ZERO <= lo <= hi <= ONE:
            raise ValueError(f"{name} must be ordered subintervals of [0,1]")
        result.append((lo, hi))
    return tuple(result)


def _row(terms: Iterable[tuple[int, Q]]) -> SparseRow:
    merged: dict[int, Q] = {}
    for index, coefficient in terms:
        merged[index] = merged.get(index, ZERO) + coefficient
    return tuple((index, value) for index, value in sorted(merged.items()) if value)


@lru_cache(maxsize=16)
def _static_rows(pbox: Box, ebox: Box, G: Q):
    """Reuse calibration/sensitivity rows across all outcomes for a selection."""
    k = len(pbox)
    rows, rhs, names, boundaries = [], [], [], []

    def add(name, terms, bound=ZERO):
        rows.append(_row(terms))
        rhs.append(bound)
        names.append(name)

    for i, ((pl, pu), (el, eu)) in enumerate(zip(pbox, ebox)):
        p, t, v, w = i, k + i, 2 * k + i, 3 * k + i
        add(f"mass_upper[{i}]", ((p, ONE),), pu)
        add(f"mass_lower[{i}]", ((p, -ONE),), -pl)
        add(f"propensity_lower[{i}]", ((p, el), (t, -ONE)))
        add(f"propensity_upper[{i}]", ((t, ONE), (p, -eu)))
        add(f"treated_mass_upper[{i}]", ((t, ONE), (p, -ONE)))
        add(f"treated_success_upper[{i}]", ((v, ONE), (t, -ONE)))
        add(f"control_mass_upper[{i}]", ((w, ONE), (t, ONE), (p, -ONE)))
        if eu > ZERO:
            a = (ONE - eu) / (eu * G)
            add(f"odds_lower_success[{i}]", ((v, a), (w, -ONE)))
            add(f"odds_lower_failure[{i}]",
                ((w, ONE), (p, -ONE), (t, ONE + a), (v, -a)))
        else:
            # Calibration already forces t = v = 0.  There is no treated
            # conditional outcome law at e = 0, so w can range over [0,p].
            # Multiplying an infinite odds factor by the forced zero mass
            # would spuriously constrain this unidentified outcome.
            boundaries.append((i, "upper_zero: omit undefined lower odds; t=v=0"))
        if el > ZERO:
            b = G * (ONE - el) / el
            add(f"odds_upper_success[{i}]", ((w, ONE), (v, -b)))
            add(f"odds_upper_failure[{i}]",
                ((p, ONE), (t, -(ONE + b)), (v, b), (w, -ONE)))
        else:
            boundaries.append((i, "lower_zero: omit infinite upper odds"))
        if el == ONE:
            boundaries.append((i, "lower_one: t=p and w=0"))
    return tuple(rows), tuple(rhs), tuple(names), tuple(boundaries)


@dataclass(frozen=True)
class LPRegion:
    K: int
    pbox: Box
    ebox: Box
    rbox: Box
    cells: tuple[tuple[int, ...], ...]
    pooled: Box
    G: Q
    rows: tuple[SparseRow, ...]
    rhs: tuple[Q, ...]
    row_names: tuple[str, ...]
    equalities: tuple[SparseRow, ...]
    targets: tuple[Q, ...]
    objective: tuple[Q, ...]
    boundaries: tuple[tuple[int, str], ...]

    @property
    def nvars(self) -> int:
        return 4 * self.K

    def evidence(self) -> dict[str, Any]:
        def boxes(values):
            return [[str(lo), str(hi)] for lo, hi in values]

        def rows(values):
            return [[[index, str(value)] for index, value in row] for row in values]

        return {
            "version": VERSION,
            "K": self.K,
            "nvars": self.nvars,
            "variable_order": ["p", "t", "v", "w"],
            "variable_bounds": ["0", "1"],
            "inputs": {"pbox": boxes(self.pbox), "ebox": boxes(self.ebox),
                       "rbox": boxes(self.rbox), "cells": [list(c) for c in self.cells],
                       "pooled": boxes(self.pooled), "G": str(self.G)},
            "rows": rows(self.rows),
            "rhs": [str(value) for value in self.rhs],
            "row_names": list(self.row_names),
            "equalities": rows(self.equalities),
            "targets": [str(value) for value in self.targets],
            "objective": _sparse(self.objective),
            "structural_boundaries": [[i, reason] for i, reason in self.boundaries],
        }


def build_region(pbox, ebox, rbox, cells, pooled, G) -> LPRegion:
    """Build the exact outer region, with no within-cell mean homogeneity.

    Each box is a sequence of (lower, upper) pairs.  ``cells`` is a partition
    of range(K), and ``pooled`` contains one treated-mean interval per cell.
    Ordered boxes can be jointly infeasible; that is left to the explicitly
    unresolved solver path, rather than mistaken for malformed input.
    """
    pbox, ebox, rbox = (_box(values, name)
                         for values, name in ((pbox, "pbox"), (ebox, "ebox"),
                                              (rbox, "rbox")))
    k = len(pbox)
    if not k or len(ebox) != k or len(rbox) != k:
        raise ValueError("The three raw boxes must have the same positive support")
    G = _q(G)
    if G < ONE:
        raise ValueError("The raw sensitivity budget must be at least one")
    partition = []
    for cell in cells:
        cell = tuple(cell)
        if not cell or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                           or not 0 <= i < k for i in cell):
            raise ValueError("Cells must contain valid raw-state coordinates")
        partition.append(tuple(sorted(map(int, cell))))
    flattened = [i for cell in partition for i in cell]
    if sorted(flattened) != list(range(k)):
        raise ValueError("Cells must partition the raw support exactly once")
    cells = tuple(partition)
    pooled = _box(pooled, "pooled")
    if len(pooled) != len(cells):
        raise ValueError("There must be one pooled mean interval per cell")

    static_rows, static_rhs, static_names, boundaries = _static_rows(pbox, ebox, G)
    rows, rhs, names = list(static_rows), list(static_rhs), list(static_names)

    def add(name, terms):
        rows.append(_row(terms))
        rhs.append(ZERO)
        names.append(name)

    for i, (rl, ru) in enumerate(rbox):
        t, v = k + i, 2 * k + i
        add(f"raw_outcome_lower[{i}]", ((t, rl), (v, -ONE)))
        add(f"raw_outcome_upper[{i}]", ((v, ONE), (t, -ru)))
    for z, (cell, (rl, ru)) in enumerate(zip(cells, pooled)):
        add(f"pooled_outcome_lower[{z}]",
            [(k + i, rl) for i in cell] + [(2 * k + i, -ONE) for i in cell])
        add(f"pooled_outcome_upper[{z}]",
            [(2 * k + i, ONE) for i in cell] + [(k + i, -ru) for i in cell])
    return LPRegion(k, pbox, ebox, rbox, cells, pooled, G,
                    tuple(rows), tuple(rhs), tuple(names),
                    (tuple((i, ONE) for i in range(k)),), (ONE,),
                    (ZERO,) * (2 * k) + (ONE,) * (2 * k), boundaries)


def _sparse(values: Sequence[Q]) -> list[list[Any]]:
    return [[i, str(value)] for i, value in enumerate(values) if value]


def _dense(values, length: int, name: str) -> tuple[Q, ...]:
    """Read a canonical sparse vector; duplicate/zero indices are rejected."""
    result = [ZERO] * length
    previous = -1
    for item in values:
        if len(item) != 2:
            raise ValueError(f"Malformed sparse {name}")
        i, encoded = item
        if isinstance(i, bool) or not isinstance(i, int) or not previous < i < length:
            raise ValueError(f"Noncanonical sparse {name} index")
        value = _q(encoded)
        if not value:
            raise ValueError(f"Sparse {name} must omit zero entries")
        result[i] = value
        previous = i
    return tuple(result)


def _upper_terms(region: LPRegion, objective, y, z):
    c, y, z = (tuple(map(_q, values)) for values in (objective, y, z))
    if len(c) != region.nvars or len(y) != len(region.rows) or len(z) != len(region.equalities):
        raise ValueError("Objective or dual multiplier dimension mismatch")
    if any(value < ZERO for value in y):
        raise ValueError("Max-form inequality multipliers must be nonnegative")
    residual = list(c)
    inequality_term, equality_term = ZERO, ZERO
    for row, bound, multiplier in zip(region.rows, region.rhs, y):
        if multiplier:
            inequality_term += multiplier * bound
            for j, coefficient in row:
                residual[j] -= coefficient * multiplier
    for row, bound, multiplier in zip(region.equalities, region.targets, z):
        if multiplier:
            equality_term += multiplier * bound
            for j, coefficient in row:
                residual[j] -= coefficient * multiplier
    positive = sum((value for value in residual if value > ZERO), ZERO)
    return c, y, z, tuple(residual), inequality_term, equality_term, positive


def certify_upper(region: LPRegion, objective, y, z) -> dict[str, Any]:
    """Certify arbitrary rational proposals, including perturbed duals.

    Negative inequality multipliers are rejected here.  ``solve_enclosure``
    clips negative numerical proposals to zero before calling this function.
    """
    c, y, z, residual, iy, ez, positive = _upper_terms(region, objective, y, z)
    return {"version": VERSION, "method": "rational_box_residual_upper",
            "objective": _sparse(c), "y": _sparse(y), "z": _sparse(z),
            "residual": _sparse(residual),
            "inequality_term": str(iy), "equality_term": str(ez),
            "positive_residual_sum": str(positive), "bound": str(iy + ez + positive)}


def verify_certificate(region: LPRegion, certificate, objective=None) -> Q:
    """Recompute every exact certificate term without numerical optimization."""
    if certificate.get("version") != VERSION or certificate.get("method") != "rational_box_residual_upper":
        raise ValueError("Unrecognized exact upper certificate")
    c = _dense(certificate["objective"], region.nvars, "objective")
    if objective is not None and c != tuple(map(_q, objective)):
        raise ValueError("The certificate proves a different objective")
    y = _dense(certificate["y"], len(region.rows), "y")
    z = _dense(certificate["z"], len(region.equalities), "z")
    _, _, _, residual, iy, ez, positive = _upper_terms(region, c, y, z)
    saved_residual = _dense(certificate["residual"], region.nvars, "residual")
    if saved_residual != residual:
        raise ArithmeticError("Saved signed residuals do not match the exact duals")
    for name, expected in (("inequality_term", iy), ("equality_term", ez),
                           ("positive_residual_sum", positive), ("bound", iy + ez + positive)):
        if _q(certificate[name]) != expected:
            raise ArithmeticError(f"Saved {name} does not match its exact expansion")
    return iy + ez + positive


def _matrix(rows, nvars):
    data, indices, indptr = [], [], [0]
    for row in rows:
        for index, value in row:
            data.append(float(value))
            indices.append(index)
        indptr.append(len(data))
    data = np.asarray(data, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Exact LP coefficients cannot be represented as finite solver inputs")
    return csr_matrix((data, np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int32)), shape=(len(rows), nvars))


def _proposal(region, c, matrices, time_limit):
    diagnostic: dict[str, Any] = {"method": "scipy_highs_dual_simplex", "native_threads": 1,
                                  "parallel": False, "feasibility_certified": False}
    options = {"threads": 1, "parallel": False}
    if time_limit is not None:
        options["time_limit"] = time_limit
    try:
        # SciPy deliberately forwards these two native HiGHS options while
        # issuing a generic unknown-option warning.  They are checked by the
        # bundled HiGHS wrapper, and they keep native solver work single-threaded.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected:.*",
                                    category=OptimizeWarning)
            fitted = linprog(-np.asarray(c, dtype=float),
                             A_ub=matrices[0], b_ub=matrices[1],
                             A_eq=matrices[2], b_eq=matrices[3],
                             bounds=(0, 1), method="highs-ds", options=options)
        diagnostic.update(success=bool(fitted.success), status=int(fitted.status),
                          message=str(fitted.message), nit=int(getattr(fitted, "nit", 0) or 0))
        fun = getattr(fitted, "fun", None)
        diagnostic["numerical_minimum"] = float(fun) if fun is not None and math.isfinite(fun) else None
        if not fitted.success:
            diagnostic["resolution"] = "unresolved_solver_status"
            return None, diagnostic
    except Exception as error:
        diagnostic.update(success=False, status=None,
                          message=f"{type(error).__name__}: {error}",
                          resolution="unresolved_solver_error")
        return None, diagnostic
    try:
        proposed_y = tuple(_q(-float(value)) for value in fitted.ineqlin.marginals)
        z = tuple(_q(-float(value)) for value in fitted.eqlin.marginals)
        y = tuple(max(ZERO, value) for value in proposed_y)
        diagnostic["negative_proposed_y_clipped"] = sum(value < ZERO for value in proposed_y)
        certificate = certify_upper(region, c, y, z)
    except Exception as error:
        diagnostic.update(resolution="unresolved_certificate",
                          certificate_error=f"{type(error).__name__}: {error}")
        return None, diagnostic
    diagnostic["resolution"] = "certified"
    return certificate, diagnostic


def _interval(bounds):
    # v <= t and w <= p-t imply 0 <= theta <= sum p = 1.
    return max(ZERO, -bounds["negated_lower"]), min(ONE, bounds["upper"])


def solve_enclosure(region: LPRegion, *, time_limit=None) -> dict[str, Any]:
    """Propose and rationally certify both endpoints, or deliver [0,1].

    Numerical infeasibility is explicitly unresolved.  No solver flag is a
    certificate that the statistical region is inconsistent.
    """
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("A solver time limit must be finite and positive")
    tick = time.perf_counter()
    certificates, solver, bounds = {}, {}, {}
    try:
        matrices = (_matrix(region.rows, region.nvars), np.asarray(region.rhs, dtype=float),
                    _matrix(region.equalities, region.nvars), np.asarray(region.targets, dtype=float))
        if not np.isfinite(matrices[1]).all() or not np.isfinite(matrices[3]).all():
            raise ValueError("Nonfinite floating solver right-hand side")
    except Exception as error:
        matrices = None
        matrix_error = f"{type(error).__name__}: {error}"
    for name, sign in (("upper", ONE), ("negated_lower", -ONE)):
        objective = tuple(sign * value for value in region.objective)
        if matrices is None:
            certificate = None
            diagnostic = {"success": False, "status": None, "resolution": "unresolved_solver_input",
                          "message": matrix_error, "feasibility_certified": False,
                          "native_threads": 1, "parallel": False}
        else:
            certificate, diagnostic = _proposal(region, objective, matrices, time_limit)
        certificates[name], solver[name] = certificate, diagnostic
        bounds[name] = None if certificate is None else _q(certificate["bound"])
    unresolved = [value["resolution"] for value in solver.values() if value["resolution"] != "certified"]
    if unresolved:
        interval, fallback, status = (ZERO, ONE), True, unresolved[0]
    else:
        interval = _interval(bounds)
        fallback = interval[0] > interval[1]
        status = "unresolved_empty_verified_interval" if fallback else "certified"
        if fallback:
            interval = (ZERO, ONE)
    return {"version": VERSION, "interval": interval, "status": status, "fallback": fallback,
            "bounds": bounds, "certificates": certificates, "solver": solver,
            "region": region.evidence(), "elapsed_seconds": time.perf_counter() - tick}


def verify_saved(record, region: LPRegion | None = None) -> dict[str, Any]:
    """Verify saved exact input rows, both bounds and the delivered interval.

    An optional independently rebuilt region connects saved evidence to the
    caller's original intervals and selected map.  Without one, the region
    is reconstructed from its saved exact inputs.  This checks an enclosure,
    not primal feasibility or whether a numerical solver really ran.
    """
    if record.get("version") != VERSION:
        raise ValueError("Unrecognized saved pooled-raw LP record")
    evidence = record["region"]
    if region is None:
        inputs = evidence["inputs"]
        region = build_region(inputs["pbox"], inputs["ebox"], inputs["rbox"],
                              inputs["cells"], inputs["pooled"], inputs["G"])
    if evidence != region.evidence():
        raise ArithmeticError("Saved exact LP rows differ from the rebuilt input region")
    bounds = {}
    for name, sign in (("upper", ONE), ("negated_lower", -ONE)):
        certificate = record["certificates"][name]
        bound = (None if certificate is None else verify_certificate(
            region, certificate, tuple(sign * value for value in region.objective)))
        saved = record["bounds"][name]
        if (saved is None) != (bound is None) or (bound is not None and _q(saved) != bound):
            raise ArithmeticError("Saved endpoint bound differs from its certificate")
        bounds[name] = bound
    interval = tuple(map(_q, record["interval"]))
    if len(interval) != 2:
        raise ValueError("A saved enclosure must have two endpoints")
    fallback = record["fallback"]
    if not isinstance(fallback, bool):
        raise ValueError("A saved fallback indicator must be boolean")
    if fallback:
        if interval != (ZERO, ONE) or not str(record["status"]).startswith("unresolved_"):
            raise ArithmeticError("An unresolved LP must retain the conservative [0,1] fallback")
        if record["status"] == "unresolved_empty_verified_interval":
            if any(value is None for value in bounds.values()) or _interval(bounds)[0] <= _interval(bounds)[1]:
                raise ArithmeticError("Saved empty-interval event lacks the exact crossing bounds")
    else:
        if record["status"] != "certified" or any(value is None for value in bounds.values()):
            raise ArithmeticError("A delivered nonfallback LP needs both exact certificates")
        if interval != _interval(bounds) or not ZERO <= interval[0] <= interval[1] <= ONE:
            raise ArithmeticError("Saved interval differs from its two exact endpoint bounds")
    return {"status": "verified", "interval": interval, "fallback": fallback, "bounds": bounds}
