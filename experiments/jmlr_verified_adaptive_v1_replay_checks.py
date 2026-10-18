"""Pure exact linkage checks for saved verified-calibration evidence.

This module neither evaluates special functions nor repeats mass optimization.
It checks that the saved sign evidence and exact arithmetic belong to the saved
counts/region, and that the actual label return path uses those same values.
"""
from fractions import Fraction as Q

ALPHA = Q(1, 20)
CAPS = (Q(5, 4), Q(3, 2), Q(2))
METHODS = ("coupled", "independent")
FAMILIES = ("fixed", "contiguous", "combined")


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _rational(value, name):
    _need(isinstance(value, Q), name+" is not an authoritative Fraction")
    return value


def _integer(value, name):
    _need(isinstance(value, int) and not isinstance(value, bool), name+" is not an integer")
    return value


def _cell(cell, K):
    cell = tuple(cell)
    _need(bool(cell) and all(isinstance(i, int) and not isinstance(i, bool) for i in cell), "Malformed cell")
    _need(cell == tuple(sorted(set(cell))) and 0 <= cell[0] <= cell[-1] < K, "Noncanonical or out-of-range cell")
    return cell


def _partition(partition, K):
    partition = tuple(_cell(cell, K) for cell in partition)
    _need(tuple(sorted(partition)) == partition, "Noncanonical partition")
    _need(sorted(i for cell in partition for i in cell) == list(range(K)), "Partition does not cover the known support exactly once")
    return partition


def _odds(value):
    value = _rational(value, "Odds argument")
    _need(0 < value < 1, "Noninterior finite odds argument")
    return value/(1-value)


def _bracket(bracket, expected_structural, direction, gamma, grid_bits):
    low, high = (_rational(bracket[key], "Root bracket "+key) for key in ("low", "high"))
    width = _rational(bracket["width"], "Root bracket width")
    _need(0 <= low <= high <= 1 and width == high-low, "Malformed dyadic root bracket")
    D = 1 << grid_bits
    _need(D % low.denominator == 0 and D % high.denominator == 0, "Root endpoints do not lie on the declared dyadic grid")
    status = bracket["status"]
    if status == "fallback":
        _need(low == 0 and high == 1 and width == 1 and bool(bracket["reason"]), "Malformed fallback bracket")
        return
    if expected_structural is not None:
        _need(status == "structural" and low == high == expected_structural and width == 0,
              "Structural root endpoint differs from its count case")
        _need(bracket["sign_low"] == bracket["sign_high"] == 0, "Structural root has nonzero sign tag")
        return
    _need(status == "verified" and 0 < width <= Q(2, D), "Root bracket lacks verified finite-grid width")
    expected_signs = (-1, 1) if direction == "lower" else (1, -1)
    _need((bracket["sign_low"], bracket["sign_high"]) == expected_signs, "Wrong root-bracket sign directions")
    for key, sign, endpoint in zip(("low", "high"), expected_signs, (low, high)):
        evidence = bracket["evidence"][key]
        lo = _rational(evidence["difference_lower"], "Tail difference lower")
        hi = _rational(evidence["difference_upper"], "Tail difference upper")
        _need(lo <= hi and evidence["sign"] == sign, "Malformed tail-sign enclosure")
        _need(hi < 0 if sign == -1 else lo > 0, "Tail enclosure does not prove its strict sign")
        if endpoint in (0, 1):
            z = endpoint if direction == "lower" else 1-endpoint
            expected = z-gamma/2
            _need(lo == hi == expected, "Structural tail-sign evidence differs from exact endpoint value")


def _root(interval, x, m, gamma, grid_bits):
    _need(_integer(interval["x"], "Root successes") == x and
          _integer(interval["m"], "Root trials") == m, "Root counts do not match saved N/T")
    _need(_rational(interval["gamma"], "Root gamma") == gamma, "Root error allocation differs")
    lower, upper = (_rational(interval[key], "Returned CP "+key) for key in ("lower", "upper"))
    _need(0 <= lower <= upper <= 1, "Malformed returned interval")
    lb, ub = interval["lower_bracket"], interval["upper_bracket"]
    _bracket(lb, Q(0) if x == 0 else None, "lower", gamma, grid_bits)
    _bracket(ub, Q(1) if x == m else None, "upper", gamma, grid_bits)
    reflected = ub["reflected_bracket"]
    _need(ub["low"] == 1-reflected["high"] and ub["high"] == 1-reflected["low"] and
          ub["width"] == reflected["width"] and ub["status"] == reflected["status"], "Upper bracket is not the reflected lower-root bracket")
    _need(ub["sign_low"] == reflected["sign_high"] and ub["sign_high"] == reflected["sign_low"], "Reflected upper signs differ")
    if interval["status"] == "fallback":
        _need(lower == 0 and upper == 1 and "fallback" in (lb["status"], ub["status"]), "Unjustified interval fallback")
        _need(interval["lower_expansion_bound"] == interval["upper_expansion_bound"] == 1, "Fallback expansion bound differs")
    else:
        _need(interval["status"] == ("structural" if m == 0 else "verified"), "Wrong interval status for counts")
        _need("fallback" not in (lb["status"], ub["status"]), "Unpropagated endpoint fallback")
        _need(lower == lb["low"] and upper == ub["high"], "Returned endpoint is inward of its saved bracket")
        _need(interval["lower_expansion_bound"] == lb["width"] and
              interval["upper_expansion_bound"] == ub["width"], "Root expansion bound differs from proved width")


def validate_records(cells, region):
    """Check saved exact odds aggregations, including full anchor completeness."""
    K = region["K"]
    lower, upper = region["propensity_lower"], region["propensity_upper"]
    records = {}
    for record in cells:
        cell = _cell(record["cell"], K)
        _need(cell not in records, "Duplicate saved cell record")
        records[cell] = record
        status = record["status"]
        _need(status in ("full", "boundary", "failure", "rejected"), "Unknown cell status")
        coupled, independent = record["coupled"], record["independent"]
        if len(cell) == 1:
            _rational(coupled, "Singleton coupled certificate")
            _rational(independent, "Singleton independent certificate")
            _need(status == "full" and coupled == independent == Q(1) and not record["anchors"], "Singleton certificate is not structural one")
            continue
        if status == "boundary":
            _need(region["valid"] and coupled is independent is None and not record["anchors"], "Boundary record contains a finite or partial certificate")
            _need(any(lower[i] == 0 or upper[i] == 1 for i in cell), "Spurious boundary infinity")
            continue
        if status == "failure":
            _need(coupled is None and bool(record["reason"]), "Failure record is favorable or untagged")
        else:
            _need(region["valid"] and all(0 < lower[i] <= upper[i] < 1 for i in cell), "Finite calculation has inadmissible region")
        if independent is not None:
            _rational(independent, "Independent certificate")
            qlo, qhi = record["independent_q_lower"], record["independent_q_upper"]
            _need(min(lower[i] for i in cell) <= qlo <= max(lower[i] for i in cell) and
                  min(upper[i] for i in cell) <= qhi <= max(upper[i] for i in cell) and qlo <= qhi,
                  "Independent mixtures lie outside coefficient spans")
            expected = max(Q(1), _odds(max(upper[i] for i in cell))/_odds(qlo),
                           _odds(qhi)/_odds(min(lower[i] for i in cell)))
            _need(independent == expected, "Independent exact odds aggregation differs")
        else:
            _need(status == "failure", "Finite/rejected record lacks full independent envelope")
        anchors = record["anchors"]
        states = tuple(_integer(a["state"], "Anchor state") for a in anchors)
        _need(len(set(states)) == len(states) and set(states) <= set(cell), "Duplicate or out-of-cell distortion anchor")
        partial = Q(1)
        for anchor in anchors:
            i = anchor["state"]
            qplus, qminus = anchor["q_plus"], anchor["q_minus"]
            vplus = [upper[i] if j == i else lower[j] for j in cell]
            vminus = [lower[i] if j == i else upper[j] for j in cell]
            _need(min(vplus) <= qplus <= max(vplus) and min(vminus) <= qminus <= max(vminus), "Anchor mixture lies outside its coefficient span")
            plus, minus = _odds(upper[i])/_odds(qplus), _odds(qminus)/_odds(lower[i])
            _rational(anchor["plus"], "Saved plus odds product")
            _rational(anchor["minus"], "Saved minus odds product")
            _need(anchor["plus"] == plus and anchor["minus"] == minus, "Anchor exact odds product differs")
            partial = max(partial, plus, minus)
        if status == "full":
            _need(sorted(states) == list(cell), "Full certificate does not cover every anchor exactly once")
            _need(_rational(coupled, "Coupled certificate") == partial <= independent, "Full coupled aggregation differs")
        elif status == "rejected":
            _rational(record["rejection_bound"], "Rejection lower predicate")
            _need(coupled is None and bool(anchors) and record["rejection_bound"] == min(partial, independent) > max(CAPS), "Invalid partial rejection predicate")
    return records


def _certificate_display(record, value):
    _need(record["status"] == "full" and isinstance(record["infinite"], bool), "Serialized certificate status differs")
    _need(record["value"] == value and record["infinite"] == (value is None), "Serialized certificate/value status differs")
    _need(isinstance(record["upper_decimal"], str), "Advertised upper display is not a decimal string")
    if value is None:
        _need(record["upper_decimal"] == "infinity", "Infinite certificate has finite display")
    else:
        _rational(value, "Decision certificate")
        _rational(record["value"], "Serialized exact certificate")
        _need(Q(record["upper_decimal"]) >= value, "Advertised upper decimal is inward")


def validate_chain(audit):
    """Validate a decoded selected-audit object using only integer/rational work."""
    _need(audit["version"] == "jmlr_verified_adaptive_v1" and audit["kind"] == "selected-partition-audit", "Wrong evidence schema")
    region, metadata = audit["region"], audit["metadata"]
    K = _integer(metadata["K"], "Known support")
    _need(K > 0 and region["K"] == K, "Known support differs")
    N, T = tuple(audit["N"]), tuple(audit["T"])
    _need(len(N) == len(T) == K and all(_integer(n, "N") >= 0 and 0 <= _integer(t, "T") <= n
                                        for n, t in zip(N, T)), "Malformed saved counts")
    n = sum(N)
    _need(n > 0 and metadata["n"] == region["n"] == n and tuple(region["N"]) == N and tuple(region["T"]) == T, "Region counts/total differ from saved data")
    _need(sorted(audit["saved_order"]) == list(range(K)), "Saved propensity order is not a support permutation")
    _need(_rational(region["alpha"], "Region alpha") == ALPHA, "Region alpha differs from the frozen exact constant")
    a, b, lower, upper = (tuple(region[key]) for key in ("mass_lower", "mass_upper", "propensity_lower", "propensity_upper"))
    _need(all(len(vector) == K for vector in (a, b, lower, upper)), "Region vector support differs")
    for vector in (a, b, lower, upper):
        for value in vector:
            _rational(value, "Region endpoint")
    _need(all(0 <= a[i] <= b[i] <= 1 and 0 <= lower[i] <= upper[i] <= 1 for i in range(K)), "Malformed region box")
    valid = (all(x > 0 for x in b) and sum(a) <= 1 <= sum(b) and
             (sum(a) != 1 or all(x > 0 for x in a)) and all(u > 0 and l < 1 for l, u in zip(lower, upper)))
    _need(region["valid"] == valid, "Saved region validity differs from exact positive-feasibility test")
    known = region["known_masses"]
    gamma = ALPHA/(K if known is not None else 2*K)
    _need(region["mode"] == ("known_mass" if known is not None else "unknown_mass") and
          _rational(region["gamma"], "Region gamma") == gamma, "Region mode/error allocation differs")
    propensity, mass = region["roots"]["propensity"], region["roots"]["mass"]
    _need(len(propensity) == K, "Missing propensity root")
    grid_bits = _integer(region["root_policy"]["grid_bits"], "Root grid bits")
    _need(4 <= grid_bits <= 256, "Invalid saved root grid")
    for i, interval in enumerate(propensity):
        _root(interval, T[i], N[i], gamma, grid_bits)
        _need((interval["lower"], interval["upper"]) == (lower[i], upper[i]), "Propensity region endpoints are not the returned root endpoints")
    if known is None:
        _need(mass is not None and len(mass) == K, "Missing unknown-mass root")
        for i, interval in enumerate(mass):
            _root(interval, N[i], n, gamma, grid_bits)
            _need((interval["lower"], interval["upper"]) == (a[i], b[i]), "Mass region endpoints are not the returned root endpoints")
    else:
        known = tuple(known)
        _need(mass is None and a == b == known and all(_rational(p, "Known mass") > 0 for p in known) and sum(known) == 1, "Invalid declared exact known masses")
    roots = list(propensity)+list(mass or [])
    fallbacks = sum(interval["status"] == "fallback" for interval in roots)
    _need(region["root_fallbacks"] == audit["root_diagnostics"]["fallback_intervals"] == fallbacks, "Fallback count differs")
    records = validate_records(audit["cells"], region)
    expected_grid = {(method, family, cap) for method in METHODS for family in FAMILIES for cap in CAPS}
    seen = set()
    for label in audit["labels"]:
        key = (label["method"], label["family"], _rational(label["cap"], "Label cap"))
        _need(key in expected_grid and key not in seen, "Duplicate or out-of-scope selection label")
        seen.add(key)
        method, _, cap = key
        proposed = _partition(label["proposed_partition"], K)
        _need(label["proposed_count"] == len(proposed), "Proposed count differs")
        _need(all(cell in records and records[cell]["status"] != "rejected" for cell in proposed), "Selected proposal lacks a full cell record")
        values = [(Q(1) if len(cell) == 1 else records[cell][method]) for cell in proposed]
        failed = tuple(cell for cell, value in zip(proposed, values) if value is None or value > cap)
        accepted = not failed
        value = None if any(x is None for x in values) else max(values)
        _certificate_display(label["new_certificate"], value)
        _need(isinstance(label["accepted"], bool) and isinstance(label["identity_fallback"], bool) and
              label["accepted"] == accepted and label["identity_fallback"] == (not accepted) and
              tuple(tuple(c) for c in label["failed_cells"]) == failed, "Exact acceptance/failed-cell decision differs")
        expected_output = proposed if accepted else tuple((i,) for i in range(K))
        _need(_partition(label["output_partition"], K) == expected_output and label["output_count"] == len(expected_output), "Actual safe return partition differs")
        _certificate_display(label["output_certificate"], value if accepted else Q(1))
        if accepted:
            decision, reason = "pass", "recertified"
        elif any(records[cell]["status"] == "failure" for cell in failed):
            decision, reason = "unresolved", "backend-failure"
        else:
            decision = "not_certified"
            reason = "boundary-infinity" if any(records[cell][method] is None for cell in failed) else "finite-exceeds-cap"
        _need((label["decision"], label["reason"]) == (decision, reason), "Reported recertification classification differs")
    _need(seen == expected_grid, "Selection label grid is incomplete")
    return {"status": "passed", "roots": len(roots), "cells": len(records), "labels": len(seen)}
