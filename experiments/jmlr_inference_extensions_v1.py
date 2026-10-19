"""General-confidence, bounded-outcome, and mass-drift inference extensions.

This module is a backward-compatible layer over the frozen binary outcome and
verified LP implementations.  Historical experiments import neither this
module nor its release schema.  Exact rational inputs are required at the
scientific boundary; bounded-mean radii use a recorded high-precision Decimal
upper rounding before conversion to an exact rational endpoint.  The analyst
reuses that recorded precision and rejects an incompatible override.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Context, Decimal, ROUND_CEILING
from fractions import Fraction as F
import json
import math
import numbers

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_outcome_inference_v1_core import (
    confidence_enclosure,
    intersect_enclosures,
    outcome_family,
    project_assignment,
)
from experiments.jmlr_outcome_inference_v1_lp import build_region, solve_enclosure
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import exact_order, select_families


VERSION = "jmlr_inference_extensions_v1"
DEFAULT_ALPHA = F(1, 20)
DEFAULT_G = F(2)
DEFAULT_CAP = F(3, 2)
BRANCHES = ("binary", "bounded")


def _q(value, name="value"):
    """Accept exact finite scalars and reject silent floating approximation."""
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{name} must be supplied as an exact scalar")
    try:
        answer = F(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a finite exact rational") from error
    return answer


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _box(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a lower/upper pair")
    lo, hi = (_q(item, name) for item in value)
    if lo >= hi:
        raise ValueError(f"{name} must have strictly ordered endpoints")
    return lo, hi


@dataclass(frozen=True)
class ConfidenceAllocation:
    alpha: F
    alpha_cal: F
    alpha_s: F
    alpha_z: F

    def evidence(self):
        return {"alpha": str(self.alpha), "alpha_cal": str(self.alpha_cal),
                "alpha_s": str(self.alpha_s), "alpha_z": str(self.alpha_z),
                "sum": str(self.alpha_cal + self.alpha_s + self.alpha_z),
                "scope": "pointwise fixed analysis; union bound without family independence"}


def confidence_allocation(alpha=DEFAULT_ALPHA, *, alpha_cal=None, alpha_s=None,
                          alpha_z=None):
    """Validate a general allocation, using alpha/2, alpha/4, alpha/4 by default."""
    alpha = _q(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    supplied = (alpha_cal, alpha_s, alpha_z)
    if all(value is None for value in supplied):
        alpha_cal, alpha_s, alpha_z = alpha / 2, alpha / 4, alpha / 4
    elif any(value is None for value in supplied):
        raise ValueError("supply all three component allocations or none")
    alpha_cal, alpha_s, alpha_z = (
        _q(value, name) for value, name in
        zip((alpha_cal, alpha_s, alpha_z), ("alpha_cal", "alpha_s", "alpha_z")))
    if any(not 0 < value < 1 for value in (alpha_cal, alpha_s, alpha_z)):
        raise ValueError("component allocations must lie strictly between zero and one")
    if alpha_cal + alpha_s + alpha_z > alpha:
        raise ValueError("component allocations exceed alpha")
    return ConfidenceAllocation(alpha, alpha_cal, alpha_s, alpha_z)


def psi_bounds(mean, budget):
    """Mean-only density-ratio outer bounds on E[q W]."""
    mean, budget = _q(mean, "mean"), _q(budget, "budget")
    if not 0 <= mean <= 1 or budget < 1:
        raise ValueError("require a normalized mean and budget at least one")
    return (max(mean / budget, 1 - budget * (1 - mean)),
            min(budget * mean, 1 - (1 - mean) / budget))


def sharp_finite_support_density_mean(values, probabilities, budget):
    """Exact sharp reference for a known finite treated-outcome distribution.

    This diagnostic is deliberately separate from the release procedure: it
    uses the full distribution to solve the density-box linear program.
    """
    values = tuple(_q(value, "support value") for value in values)
    probabilities = tuple(_q(value, "support probability") for value in probabilities)
    budget = _q(budget, "budget")
    if not values or len(values) != len(probabilities):
        raise ValueError("support and probability vectors must align")
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("normalized support values must lie in [0,1]")
    if any(value < 0 for value in probabilities) or sum(probabilities) != 1 or budget < 1:
        raise ValueError("probabilities must sum to one and budget must be at least one")

    def endpoint(maximize):
        lower, upper = 1 / budget, budget
        q = [lower] * len(values)
        remaining = 1 - lower
        order = sorted(range(len(values)), key=lambda i: (values[i], i), reverse=maximize)
        for i in order:
            capacity = probabilities[i] * (upper - lower)
            take = min(remaining, capacity)
            if probabilities[i]:
                q[i] += take / probabilities[i]
            remaining -= take
        if remaining:
            raise ArithmeticError("density-box normalization could not be filled")
        if sum(probabilities[i] * q[i] for i in range(len(values))) != 1:
            raise ArithmeticError("density-box witness is not normalized")
        return {"value": sum(probabilities[i] * q[i] * values[i]
                             for i in range(len(values))),
                "density": tuple(q), "maximize": maximize}

    lower, upper = endpoint(False), endpoint(True)
    return {"interval": (lower["value"], upper["value"]),
            "lower_witness": lower, "upper_witness": upper}


def _decimal_upper_radius(treated, delta, precision=80):
    """Return a rational strictly above the computed Hoeffding radius.

    Decimal transcendental/square-root results are advanced by one unit in the
    final high-precision context.  Subsequent clipping and affine maps are
    exact rational operations.
    """
    treated = _count(treated, "treated")
    delta = _q(delta, "delta")
    if treated <= 0 or not 0 < delta < 1:
        raise ValueError("a positive treated count and 0 < delta < 1 are required")
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 32:
        raise ValueError("Decimal precision must be an integer of at least 32 digits")
    context = Context(prec=precision, rounding=ROUND_CEILING)
    ratio = context.divide(Decimal(2 * delta.denominator), Decimal(delta.numerator))
    logarithm = context.next_plus(context.ln(ratio))
    scaled = context.divide(logarithm, Decimal(2 * treated))
    radius = context.next_plus(context.sqrt(scaled))
    return F(radius)


def hoeffding_mean_interval(outcome_sum, treated, delta, *, precision=80):
    """Two-sided normalized mean interval with exact outward endpoints."""
    treated = _count(treated, "treated")
    delta = _q(delta, "delta")
    outcome_sum = _q(outcome_sum, "normalized treated-outcome sum")
    if not 0 < delta < 1 or not 0 <= outcome_sum <= treated:
        raise ValueError("normalized sum must lie in [0,M] and 0 < delta < 1")
    if treated == 0:
        return {"interval": (F(0), F(1)), "mean": None, "radius": None,
                "status": "structural", "method": "hoeffding", "delta": delta,
                "treated": 0, "outward_decimal_precision": precision}
    mean = outcome_sum / treated
    radius = _decimal_upper_radius(treated, delta, precision)
    return {"interval": (max(F(0), mean - radius), min(F(1), mean + radius)),
            "mean": mean, "radius": radius, "status": "verified",
            "method": "hoeffding", "delta": delta, "treated": treated,
            "outward_decimal_precision": precision}


def bounded_mean_family(outcome_sums, treated, alpha, *, precision=80):
    outcome_sums, treated = tuple(outcome_sums), tuple(treated)
    alpha = _q(alpha, "outcome-family allocation")
    if not outcome_sums or len(outcome_sums) != len(treated) or not 0 < alpha < 1:
        raise ValueError("bounded mean family requires aligned nonempty vectors")
    delta = alpha / len(treated)
    evidence = tuple(hoeffding_mean_interval(value, count, delta, precision=precision)
                     for value, count in zip(outcome_sums, treated))
    return tuple(record["interval"] for record in evidence), evidence


def _mean_family(outcome_sums, treated, alpha, branch, verifier, precision):
    if branch == "binary":
        successes = tuple(_count(value, "treated successes") for value in outcome_sums)
        return outcome_family(successes, treated, alpha, verifier)
    if branch == "bounded":
        return bounded_mean_family(outcome_sums, treated, alpha, precision=precision)
    raise ValueError("unknown outcome branch")


def normalize_outcome_sums(outcome_sums, treated, outcome_bounds, branch):
    treated = tuple(_count(value, "treated") for value in treated)
    if not treated or len(outcome_sums) != len(treated):
        raise ValueError("outcome sums and treated counts must align")
    ymin, ymax = _box(outcome_bounds, "outcome_bounds")
    scale = ymax - ymin
    if branch == "binary" and (ymin, ymax) != (F(0), F(1)):
        raise ValueError("the binary branch requires outcome bounds [0,1]")
    normalized = []
    original = []
    for value, count in zip(outcome_sums, treated):
        if branch == "binary":
            value = _count(value, "treated successes")
        else:
            value = _q(value, "treated-outcome sum")
        if not count * ymin <= value <= count * ymax:
            raise ValueError("treated-outcome sum falls outside the prespecified bounds")
        original.append(value)
        normalized.append(value if branch == "binary"
                          else (F(value) - count * ymin) / scale)
    return tuple(normalized), treated, tuple(original), (ymin, ymax)


def affine_interval(interval, outcome_bounds):
    ymin, ymax = _box(outcome_bounds, "outcome_bounds")
    lo, hi = (_q(value, "normalized endpoint") for value in interval)
    if not 0 <= lo <= hi <= 1:
        raise ValueError("normalized interval must lie in [0,1]")
    scale = ymax - ymin
    return ymin + scale * lo, ymin + scale * hi


def drift_factor(zeta):
    zeta = _q(zeta, "zeta")
    if not 0 <= zeta < 1:
        raise ValueError("zeta must lie in [0,1)")
    return (1 + zeta) / (1 - zeta)


def drifted_assignment_region(region, zeta):
    """Enlarge calibration mass bounds for a target population and intersect the simplex."""
    if not region.get("valid"):
        raise ValueError("a valid calibration assignment region is required")
    zeta = _q(zeta, "zeta")
    drift_factor(zeta)
    lower = tuple((1 - zeta) * _q(value) for value in region["mass_lower"])
    upper = tuple(min(F(1), (1 + zeta) * _q(value)) for value in region["mass_upper"])
    target = exact.region_from_bounds(lower, upper, region["propensity_lower"],
                                      region["propensity_upper"])
    if not target["valid"]:
        raise ValueError("the drift-enlarged mass box has no admissible target law")
    target.update(source="calibration mass box enlarged under supplied drift",
                  zeta=zeta, drift_multiplier=drift_factor(zeta))
    return target


def corrected_cell_certificates(partition, selected_cells, zeta):
    factor = drift_factor(zeta)
    selected_cells = tuple(selected_cells)
    if len(partition) != len(selected_cells):
        raise ValueError("one selected-cell certificate is required per partition cell")
    corrected = []
    for cell, record in zip(partition, selected_cells):
        if tuple(record.get("cell", cell)) != tuple(cell) or record.get("status") != "full":
            raise ValueError("selected-cell certificate contract is incomplete")
        value = _q(record.get("coupled"), "cell certificate")
        if value < 1:
            raise ValueError("cell certificates must be at least one")
        corrected.append(F(1) if len(cell) == 1 else factor * value)
    return tuple(corrected)


def release_decision(raw_budget, corrected_certificates, sensitivity_ceiling):
    raw_budget = _q(raw_budget, "raw_budget")
    sensitivity_ceiling = _q(sensitivity_ceiling, "sensitivity_ceiling")
    corrected_certificates = tuple(_q(value, "corrected certificate")
                                   for value in corrected_certificates)
    if raw_budget < 1 or sensitivity_ceiling < raw_budget or not corrected_certificates:
        raise ValueError("invalid sensitivity budget or ceiling")
    certificate = max(corrected_certificates)
    transported = raw_budget * certificate
    return {"corrected_certificate": certificate, "transported_budget": transported,
            "sensitivity_ceiling": sensitivity_ceiling,
            "admissible": transported <= sensitivity_ceiling}


def partition_distortion(masses, propensities, partition):
    """Exact normalized propensity distortion for a positive finite law."""
    masses = tuple(_q(value, "mass") for value in masses)
    propensities = tuple(_q(value, "propensity") for value in propensities)
    partition = exact.validate_partition(partition, len(masses))
    if any(value <= 0 for value in masses) or sum(masses) != 1:
        raise ValueError("masses must be positive and sum to one")
    if any(not 0 < value < 1 for value in propensities):
        raise ValueError("propensities must be interior")
    cells = []
    for cell in partition:
        mass = sum(masses[i] for i in cell)
        pooled = sum(masses[i] * propensities[i] for i in cell) / mass
        pooled_odds = pooled / (1 - pooled)
        value = max(max((propensities[i] / (1 - propensities[i])) / pooled_odds,
                        pooled_odds / (propensities[i] / (1 - propensities[i])))
                    for i in cell)
        cells.append(value)
    return {"value": max(cells, default=F(1)), "cell_values": tuple(cells)}


def choose_calibration_general(N, T, tree_orders, immutable_ids, *,
                               alpha=DEFAULT_ALPHA, alpha_cal=None,
                               alpha_s=None, alpha_z=None, cap=DEFAULT_CAP,
                               verifier=None):
    """The frozen assignment-only selector with a general confidence allocation."""
    allocation = confidence_allocation(alpha, alpha_cal=alpha_cal,
                                       alpha_s=alpha_s, alpha_z=alpha_z)
    verifier = verifier or RootVerifier()
    cap = _q(cap, "cap")
    if cap < 1:
        raise ValueError("cap must be at least one")
    tree_orders = tuple(tuple(value) for value in tree_orders)
    region = exact.confidence_region(N, T, allocation.alpha_cal, verifier=verifier)
    order = exact_order(N, T, immutable_ids)
    trees = tuple(exact.build_tree(ordering) for ordering in tree_orders)
    cells = tuple(sorted(set(exact.interval_cells(order)) |
                         set().union(*(set(tree) for tree in trees))))
    records = {cell: cell_certificate(region, cell, max_cap=cap) for cell in cells}
    values = {cell: (record["coupled"] if record["status"] == "full" else None)
              for cell, record in records.items()}
    choices = select_families(order, trees, values, cap)
    choice = choices["combined"]
    partition = exact.validate_partition(choice["partition"], len(N))
    selected = tuple(records[cell] for cell in partition)
    if any(record["status"] != "full" or record["coupled"] is None
           or record["coupled"] > cap for record in selected):
        raise ArithmeticError("selection lacks completed within-cap certificates")
    certificate = max(record["coupled"] for record in selected)
    return {"N": tuple(map(int, N)), "T": tuple(map(int, T)), "region": region,
            "order": order, "tree_orders": tuple(tuple(value) for value in tree_orders),
            "immutable_ids": tuple(immutable_ids), "family_cells": cells,
            "cell_records": tuple(records[cell] for cell in cells), "choices": choices,
            "partition": partition, "selected_cells": selected,
            "certificate": certificate, "count": len(partition), "cap": cap,
            "allocation": allocation, "alpha": allocation.alpha_cal}


def build_bounded_moment_region(pbox, ebox, raw_mean_box, cells, pooled_mean_box, raw_budget):
    """Build the inherited LP under normalized first-moment semantics."""
    return build_region(pbox, ebox, raw_mean_box, cells, pooled_mean_box, raw_budget)


def _scaled_method(record, outcome_bounds):
    result = dict(record)
    result["normalized_interval"] = tuple(record["interval"])
    result["interval"] = affine_interval(record["interval"], outcome_bounds)
    result["outcome_bounds"] = tuple(outcome_bounds)
    return result


def infer_shared(calibration, outcome_treated, treated_outcome_sums, *,
                 alpha=DEFAULT_ALPHA, alpha_cal=None, alpha_s=None, alpha_z=None,
                 outcome_bounds=(F(0), F(1)), branch="binary", raw_budget=DEFAULT_G,
                 zeta=F(0), sensitivity_ceiling=None, verifier=None,
                 decimal_precision=80, lp_time_limit=None):
    """Raw, transported, cellwise, and pooled inference on one shared event."""
    allocation = confidence_allocation(alpha, alpha_cal=alpha_cal,
                                       alpha_s=alpha_s, alpha_z=alpha_z)
    if branch not in BRANCHES:
        raise ValueError("branch must be binary or bounded")
    verifier = verifier or RootVerifier()
    raw_budget = _q(raw_budget, "raw_budget")
    if raw_budget < 1:
        raise ValueError("raw_budget must be at least one")
    region = calibration["region"]
    if region.get("alpha") is not None and region["alpha"] != allocation.alpha_cal:
        raise ValueError("calibration evidence uses a different alpha allocation")
    partition = exact.validate_partition(calibration["partition"], region["K"])
    normalized_sums, treated, original_sums, outcome_bounds = normalize_outcome_sums(
        treated_outcome_sums, outcome_treated, outcome_bounds, branch)
    if len(treated) != region["K"]:
        raise ValueError("outcome vectors must cover every declared raw state")
    target_region = drifted_assignment_region(region, zeta)
    raw_pbox = tuple(zip(target_region["mass_lower"], target_region["mass_upper"]))
    raw_ebox = tuple(zip(target_region["propensity_lower"], target_region["propensity_upper"]))
    raw_r, raw_evidence = _mean_family(normalized_sums, treated, allocation.alpha_s,
                                       branch, verifier, decimal_precision)
    cell_treated = tuple(sum(treated[i] for i in cell) for cell in partition)
    cell_sums = tuple(sum(normalized_sums[i] for i in cell) for cell in partition)
    cell_r, cell_evidence = _mean_family(cell_sums, cell_treated, allocation.alpha_z,
                                         branch, verifier, decimal_precision)
    cell_pbox, cell_ebox, projection = project_assignment(target_region, partition)
    corrected = corrected_cell_certificates(partition, calibration["selected_cells"], zeta)
    certificate = max(corrected)
    normalized_methods = {
        "raw_stratified": confidence_enclosure(raw_pbox, raw_ebox, raw_r, raw_budget),
        "z_global": confidence_enclosure(cell_pbox, cell_ebox, cell_r,
                                         raw_budget * certificate),
        "z_cellwise": confidence_enclosure(cell_pbox, cell_ebox, cell_r,
                                           tuple(raw_budget * value for value in corrected)),
    }
    global_interval = normalized_methods["z_global"]["interval"]
    cell_interval = normalized_methods["z_cellwise"]["interval"]
    if not global_interval[0] <= cell_interval[0] <= cell_interval[1] <= global_interval[1]:
        raise ArithmeticError("cellwise bounded-outcome enclosure escaped the global enclosure")
    identity = partition == tuple((i,) for i in range(region["K"]))
    if identity and allocation.alpha_s == allocation.alpha_z:
        if not (normalized_methods["raw_stratified"]["interval"] == global_interval == cell_interval):
            raise ArithmeticError("identity equality failed under matching allocations")

    lp_region = build_bounded_moment_region(raw_pbox, raw_ebox, raw_r, partition,
                                            cell_r, raw_budget)
    lp = solve_enclosure(lp_region, time_limit=lp_time_limit)
    normalized_methods["raw_pooled"] = intersect_enclosures(
        tuple(lp["interval"]), normalized_methods["raw_stratified"]["interval"], cell_interval)
    normalized_methods["raw_pooled"]["lp_status"] = lp["status"]
    methods = {name: _scaled_method(record, outcome_bounds)
               for name, record in normalized_methods.items()}
    decision = None if sensitivity_ceiling is None else release_decision(
        raw_budget, corrected, sensitivity_ceiling)
    return {
        "version": VERSION, "branch": branch, "allocation": allocation,
        "outcome_bounds": outcome_bounds, "outcome_treated": treated,
        "treated_outcome_sums": original_sums, "normalized_sums": normalized_sums,
        "raw_outcome_bounds": raw_r, "cell_outcome_bounds": cell_r,
        "raw_outcome_evidence": raw_evidence, "cell_outcome_evidence": cell_evidence,
        "partition": partition, "cell_treated": cell_treated, "cell_sums": cell_sums,
        "target_region": target_region, "projection": projection,
        "corrected_cell_certificates": corrected, "corrected_certificate": certificate,
        "raw_budget": raw_budget, "cell_budgets": tuple(raw_budget * value for value in corrected),
        "methods_normalized": normalized_methods, "methods": methods,
        "LP": lp, "release_decision": decision, "zeta": _q(zeta),
        "identity_matching_allocation_equality": identity and allocation.alpha_s == allocation.alpha_z,
    }


_TOP_KEYS = {"version", "branch", "outcome_bounds", "confidence", "sensitivity",
             "selection", "root_policy", "outcome_mean_policy", "groups"}
_GROUP_KEYS = {"label", "raw_state_count", "treated_count", "treated_outcome_sum", "mass_bounds",
               "propensity_bounds", "calibration_distortion_bound",
               "target_distortion_bound"}


def _allocation_from_evidence(evidence):
    if not isinstance(evidence, dict):
        raise ValueError("malformed confidence allocation")
    expected = {"alpha", "alpha_cal", "alpha_s", "alpha_z", "sum", "scope"}
    if set(evidence) != expected:
        raise ValueError("confidence allocation fields differ")
    allocation = confidence_allocation(evidence["alpha"], alpha_cal=evidence["alpha_cal"],
                                       alpha_s=evidence["alpha_s"], alpha_z=evidence["alpha_z"])
    if allocation.evidence() != evidence:
        raise ValueError("confidence allocation evidence is inconsistent")
    return allocation


def _policy(value):
    default = asdict(RootPolicy())
    if not isinstance(value, dict) or set(value) != set(default):
        raise ValueError("unrecognized root policy metadata")
    normalized = {}
    for key, item in value.items():
        if key == "endpoint_seconds":
            if (isinstance(item, bool) or not isinstance(item, numbers.Real)
                    or not math.isfinite(item)):
                raise ValueError("invalid root time budget")
            normalized[key] = float(item)
        else:
            if isinstance(item, bool) or not isinstance(item, numbers.Integral):
                raise ValueError("root precision and resource limits must be integers")
            normalized[key] = int(item)
    return RootPolicy(**normalized)


def _outcome_mean_policy(value, branch):
    if not isinstance(value, dict) or set(value) != {
            "method", "outward_decimal_precision"}:
        raise ValueError("unrecognized outcome-mean policy metadata")
    expected_method = "clopper_pearson" if branch == "binary" else "hoeffding"
    if value["method"] != expected_method:
        raise ValueError("outcome-mean policy does not match the release branch")
    precision = value["outward_decimal_precision"]
    if branch == "binary":
        if precision is not None:
            raise ValueError("binary releases do not use decimal Hoeffding precision")
        return {"method": expected_method, "outward_decimal_precision": None}
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 32:
        raise ValueError("bounded releases require decimal precision of at least 32 digits")
    return {"method": expected_method, "outward_decimal_precision": precision}


def validate_release(artifact):
    if not isinstance(artifact, dict) or set(artifact) != _TOP_KEYS:
        raise ValueError("release does not match the strict top-level schema")
    if artifact["version"] != VERSION or artifact["branch"] not in BRANCHES:
        raise ValueError("unknown release version or branch")
    branch = artifact["branch"]
    outcome_bounds = _box(artifact["outcome_bounds"], "outcome_bounds")
    if branch == "binary" and outcome_bounds != (F(0), F(1)):
        raise ValueError("binary releases require bounds [0,1]")
    allocation = _allocation_from_evidence(artifact["confidence"])
    policy = _policy(artifact["root_policy"])
    outcome_mean_policy = _outcome_mean_policy(artifact["outcome_mean_policy"], branch)
    sensitivity = artifact["sensitivity"]
    if not isinstance(sensitivity, dict) or set(sensitivity) != {
            "raw_budget", "zeta", "drift_multiplier", "sensitivity_ceiling"}:
        raise ValueError("malformed sensitivity metadata")
    raw_budget = _q(sensitivity["raw_budget"], "raw_budget")
    zeta = _q(sensitivity["zeta"], "zeta")
    ceiling = _q(sensitivity["sensitivity_ceiling"], "sensitivity_ceiling")
    if _q(sensitivity["drift_multiplier"], "drift_multiplier") != drift_factor(zeta):
        raise ValueError("drift multiplier does not match zeta")
    groups = artifact["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("release requires at least one selected group")
    pbox, ebox, original_u, corrected_u, treated, sums = [], [], [], [], [], []
    ymin, ymax = outcome_bounds
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or set(group) != _GROUP_KEYS:
            raise ValueError("release group fields differ")
        if group["label"] != f"group_{index:04d}":
            raise ValueError("release labels must use canonical anonymous order")
        raw_state_count = _count(group["raw_state_count"], "raw_state_count")
        if raw_state_count < 1:
            raise ValueError("each released group must contain a declared raw state")
        count = _count(group["treated_count"], "treated_count")
        value = (_count(group["treated_outcome_sum"], "treated successes")
                 if branch == "binary" else _q(group["treated_outcome_sum"], "treated-outcome sum"))
        if not count * ymin <= value <= count * ymax:
            raise ValueError("released outcome sum falls outside prespecified bounds")
        pbox.append(tuple(_q(item, "mass_bounds") for item in group["mass_bounds"]))
        ebox.append(tuple(_q(item, "propensity_bounds") for item in group["propensity_bounds"]))
        if any(len(box) != 2 or not 0 <= box[0] <= box[1] <= 1
               for box in (pbox[-1], ebox[-1])):
            raise ValueError("released assignment ranges are malformed")
        old = _q(group["calibration_distortion_bound"], "calibration certificate")
        new = _q(group["target_distortion_bound"], "target certificate")
        expected = F(1) if raw_state_count == 1 else drift_factor(zeta) * old
        if old < 1 or new != expected:
            raise ValueError("corrected certificate is not justified by the drift bound")
        original_u.append(old); corrected_u.append(new); treated.append(count); sums.append(value)
    if sum(lo for lo, _ in pbox) > 1 or sum(hi for _, hi in pbox) < 1:
        raise ValueError("released target mass bounds have empty simplex intersection")
    selection = artifact["selection"]
    if not isinstance(selection, dict) or set(selection) != {
            "corrected_certificate", "transported_budget", "reporting_capacity",
            "admissible"}:
        raise ValueError("malformed release decision")
    capacity = selection["reporting_capacity"]
    if (isinstance(capacity, bool) or not isinstance(capacity, numbers.Integral)
            or not 1 <= capacity or len(groups) > capacity):
        raise ValueError("release exceeds its reporting capacity")
    decision = release_decision(raw_budget, corrected_u, ceiling)
    if not decision["admissible"] or selection != {
            "corrected_certificate": str(decision["corrected_certificate"]),
            "transported_budget": str(decision["transported_budget"]),
            "reporting_capacity": int(capacity),
            "admissible": True}:
        raise ValueError("release does not satisfy its corrected sensitivity ceiling")
    return {"branch": branch, "outcome_bounds": outcome_bounds, "allocation": allocation,
            "policy": policy, "outcome_mean_policy": outcome_mean_policy,
            "raw_budget": raw_budget, "zeta": zeta,
            "sensitivity_ceiling": ceiling, "pbox": tuple(pbox), "ebox": tuple(ebox),
            "reporting_capacity": int(capacity),
            "original_certificates": tuple(original_u),
            "corrected_certificates": tuple(corrected_u),
            "treated": tuple(treated), "outcome_sums": tuple(sums), "decision": decision}


def release_bytes(artifact):
    validate_release(artifact)
    return json.dumps(artifact, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def export_release(calibration, outcome_treated, treated_outcome_sums, *,
                   alpha=DEFAULT_ALPHA, alpha_cal=None, alpha_s=None, alpha_z=None,
                   outcome_bounds=(F(0), F(1)), branch="binary", raw_budget=DEFAULT_G,
                   zeta=F(0), sensitivity_ceiling=F(3), verifier=None,
                   decimal_precision=80, reporting_capacity=None):
    """Create an admitted release, recording the bounded-mean precision policy."""
    allocation = confidence_allocation(alpha, alpha_cal=alpha_cal,
                                       alpha_s=alpha_s, alpha_z=alpha_z)
    if branch not in BRANCHES:
        raise ValueError("branch must be binary or bounded")
    region = calibration["region"]
    if region.get("alpha") is not None and region["alpha"] != allocation.alpha_cal:
        raise ValueError("calibration evidence uses a different alpha allocation")
    partition = exact.validate_partition(calibration["partition"], region["K"])
    if reporting_capacity is None:
        reporting_capacity = len(partition)
    if (isinstance(reporting_capacity, bool)
            or not isinstance(reporting_capacity, numbers.Integral)
            or not 1 <= reporting_capacity <= region["K"]):
        raise ValueError("reporting_capacity must be an integer between one and K")
    if len(partition) > reporting_capacity:
        raise ValueError("selected partition exceeds the reporting capacity")
    normalized, treated, original_sums, outcome_bounds = normalize_outcome_sums(
        treated_outcome_sums, outcome_treated, outcome_bounds, branch)
    if len(treated) != region["K"]:
        raise ValueError("outcome vectors must cover every declared raw state")
    target_region = drifted_assignment_region(region, zeta)
    pbox, ebox, _ = project_assignment(target_region, partition)
    original_u = tuple(_q(record["coupled"], "calibration certificate")
                       for record in calibration["selected_cells"])
    corrected_u = corrected_cell_certificates(partition, calibration["selected_cells"], zeta)
    decision = release_decision(raw_budget, corrected_u, sensitivity_ceiling)
    if not decision["admissible"]:
        raise ValueError("drift-corrected transported budget exceeds the sensitivity ceiling")
    cell_treated = tuple(sum(treated[i] for i in cell) for cell in partition)
    cell_sums = tuple(sum(original_sums[i] for i in cell) for cell in partition)
    policy = verifier.policy if verifier is not None else RootPolicy(**region.get(
        "root_policy", asdict(RootPolicy())))
    artifact = {
        "version": VERSION, "branch": branch,
        "outcome_bounds": [str(value) for value in outcome_bounds],
        "confidence": allocation.evidence(),
        "sensitivity": {"raw_budget": str(_q(raw_budget)), "zeta": str(_q(zeta)),
                        "drift_multiplier": str(drift_factor(zeta)),
                        "sensitivity_ceiling": str(_q(sensitivity_ceiling))},
        "selection": {"corrected_certificate": str(decision["corrected_certificate"]),
                      "transported_budget": str(decision["transported_budget"]),
                      "reporting_capacity": int(reporting_capacity),
                      "admissible": True},
        "root_policy": asdict(policy),
        "outcome_mean_policy": {
            "method": "clopper_pearson" if branch == "binary" else "hoeffding",
            "outward_decimal_precision": None if branch == "binary" else decimal_precision,
        },
        "groups": [
            {"label": f"group_{index:04d}", "raw_state_count": len(partition[index]),
             "treated_count": cell_treated[index],
             "treated_outcome_sum": (int(cell_sums[index]) if branch == "binary"
                                      else str(cell_sums[index])),
             "mass_bounds": [str(value) for value in pbox[index]],
             "propensity_bounds": [str(value) for value in ebox[index]],
             "calibration_distortion_bound": str(original_u[index]),
             "target_distortion_bound": str(corrected_u[index])}
            for index in range(len(partition))],
    }
    validate_release(artifact)
    return artifact


def analyze_release(artifact, verifier=None, *, decimal_precision=None):
    """Reproduce aggregate inference with the release's recorded numerical policy."""
    release = validate_release(artifact)
    if verifier is None:
        verifier = RootVerifier(release["policy"])
    elif verifier.policy != release["policy"]:
        raise ValueError("analyst verifier does not match released root policy")
    normalized, treated, _, outcome_bounds = normalize_outcome_sums(
        release["outcome_sums"], release["treated"], release["outcome_bounds"],
        release["branch"])
    recorded_precision = release["outcome_mean_policy"]["outward_decimal_precision"]
    if release["branch"] == "bounded":
        if decimal_precision is not None and decimal_precision != recorded_precision:
            raise ValueError("analyst precision does not match the bounded release")
        decimal_precision = recorded_precision
    bounds, evidence = _mean_family(normalized, treated, release["allocation"].alpha_z,
                                    release["branch"], verifier, decimal_precision)
    corrected = release["corrected_certificates"]
    normalized_methods = {
        "z_global": confidence_enclosure(release["pbox"], release["ebox"], bounds,
                                         release["raw_budget"] * max(corrected)),
        "z_cellwise": confidence_enclosure(
            release["pbox"], release["ebox"], bounds,
            tuple(release["raw_budget"] * value for value in corrected)),
    }
    a, b = normalized_methods["z_global"]["interval"], normalized_methods["z_cellwise"]["interval"]
    if not a[0] <= b[0] <= b[1] <= a[1]:
        raise ArithmeticError("released cellwise enclosure escaped its global enclosure")
    return {"version": VERSION, "branch": release["branch"],
            "outcome_bounds": outcome_bounds, "allocation": release["allocation"],
            "outcome_mean_bounds": bounds, "outcome_mean_evidence": evidence,
            "corrected_certificate": max(corrected),
            "transported_budget": release["decision"]["transported_budget"],
            "reporting_capacity": release["reporting_capacity"],
            "admission_verified": True,
            "methods_normalized": normalized_methods,
            "methods": {name: _scaled_method(record, outcome_bounds)
                        for name, record in normalized_methods.items()},
            "payload_bytes": len(release_bytes(artifact)),
            "provider_trust_required": True}
