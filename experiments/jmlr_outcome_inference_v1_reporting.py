"""Saved-evidence reporting for the fixed outcome-inference study.

This module does not sample observations, select partitions, compute roots,
solve LPs, replay evidence, or choose scientific settings. Exact endpoints and
comparisons come from the completed blocks. Floating arithmetic is confined
to descriptive Monte Carlo summaries and scientific plot coordinates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as Q
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import beta, t

from experiments.jmlr_verified_adaptive_v1_math import decode, encode
from experiments.jmlr_outcome_inference_v1_core import METHODS, PROFILES, SHARED_METHODS

sys.set_int_max_str_digits(0)


VERSION = "jmlr_outcome_inference_v1"
LAW_IDS = ("K16_balanced_moderate_ordered", "K64_balanced_moderate_ordered",
           "K64_skewed_weak_ordered")
LAW_K = dict(zip(LAW_IDS, (16, 64, 64)))
RATIOS = (4, 16, 64, 256)
REPLICATIONS = 100
USEFUL_REFERENCE = Q(1, 50)
METHOD_NAMES = {
    "raw_stratified": "Raw stratified (shared event)",
    "z_global": "Z global transport (shared event)",
    "z_cellwise": "Z cellwise transport (shared event)",
    "raw_pooled": "Raw with pooled constraints (shared event)",
    "raw_standalone": "Stronger standalone raw (individual 95%)",
    "z_untransported": "Untransported Z (no raw-target guarantee)",
}
COLORS = {"raw_stratified": "#0072B2", "z_global": "#D55E00",
          "z_cellwise": "#009E73", "raw_pooled": "#CC79A7",
          "raw_standalone": "#2B2B2B", "z_untransported": "#7A7A7A"}
POPULATIONS = ("raw", "z_global_true", "z_cellwise_true", "z_global_certificate",
               "z_cellwise_certificate", "z_untransported")
DECOMPOSITION_TERMS = ("population_loss", "certificate_extra", "statistical_extra", "total_excess")
# Every sign is the exact sign of lhs width minus rhs width. A positive value
# means that rhs is narrower; "ties" are exact equalities, not equivalence.
COMPARISONS = {
    "raw_stratified_minus_z_global": ("raw_stratified", "z_global", "primary"),
    "raw_standalone_minus_z_global": ("raw_standalone", "z_global", "stronger standalone check"),
    "z_global_minus_raw_pooled": ("z_global", "raw_pooled", "release precision cost to pooled raw"),
    "z_global_minus_z_cellwise": ("z_global", "z_cellwise", "predeclared cellwise improvement"),
    "z_cellwise_minus_raw_pooled": ("z_cellwise", "raw_pooled", "cellwise release precision cost"),
    "raw_stratified_minus_raw_pooled": ("raw_stratified", "raw_pooled", "combined reference tightening over raw stratification"),
    "raw_stratified_minus_z_cellwise": ("raw_stratified", "z_cellwise", "cellwise versus stratified raw"),
    "raw_standalone_minus_z_cellwise": ("raw_standalone", "z_cellwise", "cellwise versus stronger standalone raw"),
}
EVENTS = ("calibration_covered", "raw_outcomes_covered", "z_outcomes_covered", "shared_event",
          "joint_raw_set_containment", "standalone_assignment_covered",
          "standalone_outcomes_covered", "standalone_event")
CONDITION_FIELDS = ("law_id", "K", "profile", "m_ratio", "m")


def _q(value: Any) -> Q:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("Saved scientific quantities must be exact rationals")
    return Q(value)


def _interval(value: Any) -> tuple[Q, Q]:
    if len(value) != 2:
        raise ValueError("An interval must have exactly two endpoints")
    lo, hi = map(_q, value)
    if not 0 <= lo <= hi <= 1:
        raise ValueError("A reported interval must be an ordered subset of [0,1]")
    return lo, hi


def _width(value: Any) -> Q:
    lo, hi = _interval(value)
    return hi - lo


def _load(path: Path, default: Any = None) -> Any:
    return decode(json.loads(path.read_text())) if path.exists() else default


def _dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(encode(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value.rstrip() + "\n")
    temporary.replace(path)


def _csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Exact fractions are rational strings; nested metrics use explicit JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(value) if isinstance(value, Q) else
                             json.dumps(encode(value), separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def _mean(values: Any) -> Q | None:
    values = list(values)
    return sum((_q(value) for value in values), Q()) / len(values) if values else None


def student(values: Any) -> dict[str, Any]:
    """Pointwise approximate MC95 interval over independent replication blocks.

    The supplied values must already be one scalar per block. In particular,
    profiles and nested prefixes must never be concatenated as replications.
    The interval is deliberately not clipped to a scientific parameter range.
    """
    values = list(values)
    x = np.asarray([float(value) for value in values], dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Monte Carlo observations must be finite scalars")
    n = len(x)
    if not n:
        return dict(n=0, mean=None, exact_mean=None, sample_sd=None, MCSE=None,
                    MC95_lower=None, MC95_upper=None, minimum=None, maximum=None,
                    zero_sample_variance=None)
    exact_mean = _mean(values) if all(not isinstance(v, (float, np.floating)) for v in values) else None
    mean = float(exact_mean) if exact_mean is not None else float(x.mean())
    sd = (0. if all(value == values[0] for value in values) else float(x.std(ddof=1))) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    radius = float(t.ppf(.975, n - 1)) * se if se is not None else None
    return dict(n=n, mean=mean, exact_mean=exact_mean, sample_sd=sd, MCSE=se,
                MC95_lower=mean-radius if radius is not None else None,
                MC95_upper=mean+radius if radius is not None else None,
                minimum=float(x.min()), maximum=float(x.max()),
                zero_sample_variance=sd == 0 if sd is not None else None)


def binomial_mc(values: Any) -> dict[str, Any]:
    """Descriptive CP95 interval for a per-condition replication frequency."""
    values = list(values)
    if any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("A Monte Carlo frequency requires boolean block indicators")
    n, count = len(values), sum(values)
    return dict(n=n, count=int(count), rate=count/n if n else None,
                CP95_lower=(0. if count == 0 else float(beta.ppf(.025, count, n-count+1))) if n else None,
                CP95_upper=(1. if count == n else float(beta.ppf(.975, count+1, n-count))) if n else None)


def _prefix(name: str, value: dict[str, Any]) -> dict[str, Any]:
    return {name + "_" + key: item for key, item in value.items()}


def paired(values: Any) -> dict[str, Any]:
    """Exact wins/ties/losses plus paired pointwise Student Monte Carlo error."""
    values = list(map(_q, values))
    return {**student(values), "positive_count": sum(value > 0 for value in values),
            "zero_count": sum(value == 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "rhs_narrower_win_count": sum(value > 0 for value in values),
            "exact_tie_count": sum(value == 0 for value in values),
            "rhs_wider_loss_count": sum(value < 0 for value in values),
            "at_least_0_02_count": sum(value >= USEFUL_REFERENCE for value in values),
            "at_most_minus_0_02_count": sum(value <= -USEFUL_REFERENCE for value in values)}


def _signed_summary(values: Any) -> dict[str, Any]:
    values = list(map(_q, values))
    return {**student(values), "positive_count": sum(value > 0 for value in values),
            "zero_count": sum(value == 0 for value in values),
            "negative_count": sum(value < 0 for value in values)}


def _descriptive(values: Any) -> dict[str, Any]:
    values = sorted(map(_q, values))
    if not values:
        return {"median": None, "q05": None, "q95": None, "distribution": {}}
    # Inverse empirical CDF: quantiles retain exact observed values.
    return {"median": values[math.ceil(.5*len(values))-1],
            "q05": values[math.ceil(.05*len(values))-1],
            "q95": values[math.ceil(.95*len(values))-1],
            "distribution": dict(sorted(Counter(str(v) for v in values).items()))}


def _statuses(records: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(record["status"]) for record in records).items()))


def _roots(region: dict[str, Any]) -> list[dict[str, Any]]:
    # The historical exact-region API keeps the two families separately.
    result = []
    if "roots" in region:
        for value in region["roots"].values():
            result.extend(value or ())
        return result
    for key in ("mass_roots", "propensity_roots"):
        value = region.get(key, ())
        result.extend(value or ())
    if not result and "root_evidence" in region:
        result.extend(region["root_evidence"])
    return result


def _base(law: str, profile: str, ratio: int) -> dict[str, Any]:
    return dict(law_id=law, K=LAW_K[law], profile=profile, m_ratio=ratio, m=LAW_K[law]*ratio)


def _condition(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[key] for key in CONDITION_FIELDS)


def collect(output_root: Path) -> dict[str, Any]:
    """Load each completed block once and check saved-report consistency.

    These are schema/arithmetic checks, not a substitute for the independent
    replay. A missing/corrupt block prevents an apparently complete report.
    """
    output_root = Path(output_root)
    manifest = _load(output_root / "manifest.json", {})
    replay = _load(output_root / "replay_checks.json", {})
    if manifest.get("status") not in ("replay_complete", "complete"):
        raise ValueError("Final reports require manifest status replay_complete or complete")
    if replay.get("status") != "passed" or replay.get("selection_replays") != 300 or replay.get("outcome_replays") != 3600:
        raise ValueError("Final reports require the saved completed 300-selection/3,600-outcome replay receipt")
    selections, evaluations, methods, comparisons, decompositions, plugins, populations = [], [], [], [], [], [], []
    seen = set()
    raw_references = {}
    for path in sorted((output_root / "blocks").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            block = decode(json.load(stream))
        law, K, rep = block["law_id"], block["K"], block["replication"]
        if block.get("preflight", False):
            raise ValueError("Implementation-only preflight blocks are not scientific evidence")
        if law not in LAW_IDS or K != LAW_K[law] or not isinstance(rep, int) or not 0 <= rep < REPLICATIONS:
            raise ValueError(f"Unexpected block identity: {path.name}")
        if path.name != f"{LAW_IDS.index(law)}_{rep:03d}.json.gz":
            raise ValueError(f"Block filename disagrees with its identity: {path.name}")
        if (law, rep) in seen:
            raise ValueError(f"Duplicate independent replication block: {law}, {rep}")
        seen.add((law, rep))
        cal = block["calibration"]
        partition = cal["partition"]
        U = _q(cal["certificate"])
        if cal["count"] != len(partition) or sorted(i for c in partition for i in c) != list(range(K)):
            raise ValueError("Selected map is not the stated partition")
        cell_U = tuple(_q(c["coupled"]) for c in cal["selected_cells"])
        if (len(cell_U) != len(partition) or max(cell_U) != U or not 1 <= U <= Q(3, 2)
                or any(c.get("status", "full") != "full" for c in cal["selected_cells"])):
            raise ValueError("Selected partition lacks its completed within-cap certificate")
        relative = str(path.relative_to(output_root))
        identity = cal["count"] == K
        selection = dict(law_id=law, K=K, replication=rep, evidence=relative,
                         selected_cells=cal["count"], certificate=U,
                         compression=Q(K-cal["count"], K), identity=identity,
                         selected_cell_certificates=cell_U,
                         selected_cell_sizes=tuple(len(c) for c in partition),
                         calibration_root_fallbacks=int(cal["region"].get("root_fallbacks", 0)),
                         calibration_root_statuses=_statuses(_roots(cal["region"])),
                         calibration_seconds=cal.get("compute_seconds"))
        for key, value in block.get("timings", {}).items():
            selection["block_"+key] = value
        if identity and (U != 1 or any(u != 1 for u in cell_U)):
            raise ValueError("An identity release must have singleton certificates one")
        block_grid = set()
        for evaluation in block["evaluations"]:
            profile, ratio = evaluation["profile"], evaluation["m_ratio"]
            if profile not in PROFILES or ratio not in RATIOS or evaluation["m"] != K*ratio:
                raise ValueError("Unexpected outcome operating point")
            if (profile, ratio) in block_grid:
                raise ValueError("Repeated profile/prefix in an independent block")
            block_grid.add((profile, ratio))
            inference, diag = evaluation["inference"], evaluation["diagnostics"]
            if set(inference["methods"]) != set(METHODS) or set(diag["methods"]) != set(METHODS):
                raise ValueError("Saved method grid differs from the six fixed procedures")
            if set(inference["plugins"]) != set(METHODS):
                raise ValueError("Saved plug-in grid must include all six fixed procedures")
            raw = _interval(diag["population"]["raw"])
            reference_key = law, profile
            if reference_key in raw_references and raw_references[reference_key] != raw:
                raise ValueError("Exact raw population reference changed within a law/profile")
            raw_references[reference_key] = raw
            base = dict(_base(law, profile, ratio), replication=rep, evidence=relative)
            event_values = {key: diag[key] for key in EVENTS}
            if any(not isinstance(value, bool) for value in event_values.values()):
                raise ValueError("Saved coverage events must be boolean")
            shared = diag["calibration_covered"] and diag["raw_outcomes_covered"] and diag["z_outcomes_covered"]
            if diag["shared_event"] != shared:
                raise ValueError("Shared-event indicator disagrees with its three components")
            if diag["standalone_event"] != (diag["standalone_assignment_covered"] and diag["standalone_outcomes_covered"]):
                raise ValueError("Standalone event disagrees with its two components")
            event = dict(base, selected_cells=cal["count"], certificate=U, compression=selection["compression"],
                         identity=identity, raw_population_lower=raw[0], raw_population_upper=raw[1],
                         raw_population_width=raw[1]-raw[0], **event_values)
            for key in ("zero_raw_treated", "zero_cell_treated"):
                event[key] = int(diag[key])
                event["any_"+key] = bool(diag[key])
            if sum(inference["N"]) != K*ratio or diag["zero_raw_treated"] != sum(x == 0 for x in inference["T"]) or diag["zero_cell_treated"] != sum(x == 0 for x in inference["Tz"]):
                raise ValueError("Saved label/zero-treated counts disagree")
            for family, key in (("raw", "raw_outcome_roots"), ("z", "z_outcome_roots"),
                                ("standalone", "standalone_outcome_roots")):
                statuses = _statuses(inference[key])
                event[family+"_outcome_root_statuses"] = statuses
                event[family+"_outcome_root_fallbacks"] = statuses.get("fallback", 0)
                event[family+"_outcome_root_intervals"] = len(inference[key])
            standalone_roots = _roots(inference["standalone_region"])
            event["standalone_assignment_root_statuses"] = _statuses(standalone_roots)
            event["standalone_assignment_root_fallbacks"] = int(inference["standalone_region"].get("root_fallbacks", 0))
            lp = inference["LP"]
            event["lp_status"] = str(lp["status"])
            event["lp_fallback"] = bool(lp["fallback"])
            event["empty_pooled_intersection"] = bool(inference["methods"]["raw_pooled"]["empty_intersection"])
            for endpoint, solver in lp.get("solver", {}).items():
                for key in ("status", "success", "resolution", "feasibility_certified"):
                    event["lp_"+endpoint+"_solver_"+key] = solver.get(key)
            for key, value in inference.get("timings", {}).items():
                if value is not None and (not math.isfinite(float(value)) or value < 0):
                    raise ValueError("Saved timing must be finite and nonnegative")
                event[key] = value
            intervals = {}
            for name in METHODS:
                md, saved = diag["methods"][name], inference["methods"][name]
                interval = _interval(md["interval"])
                if interval != _interval(saved["interval"]) or _q(md["width"]) != interval[1]-interval[0]:
                    raise ValueError("Saved method endpoints/width disagree")
                actual = dict(contains_raw_set=interval[0] <= raw[0] and interval[1] >= raw[1],
                              lower_miss=interval[0] > raw[0], upper_miss=interval[1] < raw[1],
                              vacuous=interval == (Q(0), Q(1)), fallback=bool(saved["fallback"]))
                if any(md[key] != value for key, value in actual.items()):
                    raise ValueError("Saved exact-set containment/miss/fallback predicate disagrees")
                intervals[name] = interval
                methods.append(dict(base, method=name, shared_event_method=name in SHARED_METHODS,
                                    raw_target_guarantee=name != "z_untransported",
                                    lower=interval[0], upper=interval[1], width=interval[1]-interval[0],
                                    width_minus_raw_population=interval[1]-interval[0]-(raw[1]-raw[0]),
                                    status=str(saved["status"]), lp_status=saved.get("lp_status"), **actual))
            joint = all(diag["methods"][name]["contains_raw_set"] for name in SHARED_METHODS)
            if diag["joint_raw_set_containment"] != joint or shared and not joint:
                raise ValueError("Saved shared-event containment implication disagrees")
            if diag["standalone_event"] and not diag["methods"]["raw_standalone"]["contains_raw_set"]:
                raise ValueError("Saved standalone coverage implication disagrees")
            global_, cellwise, pooled = (intervals[name] for name in ("z_global", "z_cellwise", "raw_pooled"))
            if not global_[0] <= cellwise[0] <= cellwise[1] <= global_[1]:
                raise ValueError("Saved cellwise interval is not nested in global Z")
            if identity and not intervals["raw_stratified"] == global_ == cellwise:
                raise ValueError("Saved identity release differs from raw stratification")
            if not event["empty_pooled_intersection"] and not all(a[0] <= pooled[0] <= pooled[1] <= a[1] for a in (global_, cellwise, intervals["raw_stratified"])):
                raise ValueError("Nonempty pooled-raw intersection violates structural nesting")
            differences = dict(base)
            for name, (lhs, rhs, role) in COMPARISONS.items():
                differences[name] = _width(intervals[lhs])-_width(intervals[rhs])
            comparisons.append(differences)
            for name in ("z_global", "z_cellwise"):
                terms = {key: _q(diag["decompositions"][name][key]) for key in DECOMPOSITION_TERMS}
                true_width = _width(diag["population"][name+"_true"])
                certified_width = _width(diag["population"][name+"_certificate"])
                expected = dict(population_loss=true_width-(raw[1]-raw[0]),
                                certificate_extra=certified_width-true_width,
                                statistical_extra=_width(intervals[name])-certified_width,
                                total_excess=_width(intervals[name])-(raw[1]-raw[0]))
                if terms != expected or sum(terms[key] for key in DECOMPOSITION_TERMS[:3]) != terms["total_excess"]:
                    raise ValueError("Saved signed width decomposition does not sum exactly")
                decompositions.append(dict(base, method=name, **terms))
            for name in POPULATIONS:
                interval = _interval(diag["population"][name])
                populations.append(dict(base, reference=name, lower=interval[0], upper=interval[1], width=interval[1]-interval[0]))
            for name in METHODS:
                interval = _interval(inference["plugins"][name])
                plugins.append(dict(base, method=name, lower=interval[0], upper=interval[1], width=interval[1]-interval[0],
                                    contains_raw_set=interval[0] <= raw[0] and interval[1] >= raw[1],
                                    lower_miss=interval[0] > raw[0], upper_miss=interval[1] < raw[1],
                                    plugin_alias_of="raw_stratified" if name == "raw_pooled" else None,
                                    descriptive_only=True, finite_sample_guarantee=False))
            event["true_distortion"] = _q(diag["population"]["true_distortion"])
            event["certificate_covers_true_distortion"] = U >= event["true_distortion"]
            evaluations.append(event)
        if block_grid != {(profile, ratio) for profile in PROFILES for ratio in RATIOS}:
            raise ValueError(f"Incomplete profile/prefix grid in {path.name}")
        calibration_events = {row["calibration_covered"] for row in evaluations if row["law_id"] == law and row["replication"] == rep}
        if len(calibration_events) != 1:
            raise ValueError("Calibration coverage changed across a block's profiles/prefixes")
        selection["calibration_covered"] = calibration_events.pop()
        selections.append(selection)
    missing = [{"law_id": law, "missing_replications": [rep for rep in range(REPLICATIONS) if (law, rep) not in seen]}
               for law in LAW_IDS]
    if any(row["missing_replications"] for row in missing):
        raise ValueError("Reporting requires the completed fixed grid; missing independent blocks: " +
                         "; ".join(f"{row['law_id']}: {len(row['missing_replications'])}" for row in missing))
    return dict(selections=selections, evaluations=evaluations, methods=methods, comparisons=comparisons,
                decompositions=decompositions, plugins=plugins, populations=populations,
                raw_references=raw_references, manifest=manifest,
                configuration=_load(output_root / "configuration.json", {}),
                frozen_protocol=_load(output_root / "frozen_protocol.json", {}),
                panel=_load(output_root / "panel_checks.json", {}),
                preflight=_load(output_root / "preflight.json", {}),
                replay=replay)


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    conditions, method_conditions, paired_conditions = [], [], []
    decomposition_conditions, plugin_conditions, population_conditions, selection_conditions = [], [], [], []
    grouped = {}
    for kind in ("evaluations", "methods", "comparisons", "decompositions", "plugins", "populations"):
        groups = defaultdict(list)
        for row in data[kind]:
            groups[_condition(row)].append(row)
        grouped[kind] = groups
    for law in LAW_IDS:
        rows = [row for row in data["selections"] if row["law_id"] == law]
        selection = dict(law_id=law, K=LAW_K[law], n=len(rows),
                         calibration_root_fallbacks=sum(row["calibration_root_fallbacks"] for row in rows),
                         calibration_root_statuses=dict(sum((Counter(row["calibration_root_statuses"]) for row in rows), Counter())))
        for key in ("selected_cells", "certificate", "compression"):
            selection.update(_prefix(key, student(row[key] for row in rows)))
            selection.update(_prefix(key, _descriptive(row[key] for row in rows)))
        for key in ("identity", "calibration_covered"):
            selection.update(_prefix(key, binomial_mc(row[key] for row in rows)))
        times = [row["calibration_seconds"] for row in rows if row["calibration_seconds"] is not None]
        selection.update(_prefix("calibration_seconds", student(times)))
        selection_conditions.append(selection)
        for profile in PROFILES:
            for ratio in RATIOS:
                base = _base(law, profile, ratio)
                key = _condition(base)
                rows = grouped["evaluations"][key]
                if len(rows) != REPLICATIONS or {row["replication"] for row in rows} != set(range(REPLICATIONS)):
                    raise ValueError("A condition does not contain exactly 100 independent blocks")
                condition = dict(base, n=len(rows), raw_population_lower=data["raw_references"][law, profile][0],
                                 raw_population_upper=data["raw_references"][law, profile][1],
                                 raw_population_width=_width(data["raw_references"][law, profile]))
                for field in ("selected_cells", "certificate", "compression", "true_distortion", "zero_raw_treated", "zero_cell_treated"):
                    condition.update(_prefix(field, student(row[field] for row in rows)))
                for field in EVENTS + ("identity", "certificate_covers_true_distortion", "any_zero_raw_treated",
                                       "any_zero_cell_treated", "lp_fallback", "empty_pooled_intersection"):
                    condition.update(_prefix(field, binomial_mc(row[field] for row in rows)))
                condition["lp_status_counts"] = dict(sorted(Counter(row["lp_status"] for row in rows).items()))
                for field in ("raw_outcome", "z_outcome", "standalone_outcome", "standalone_assignment"):
                    condition[field+"_root_fallbacks"] = sum(row[field+"_root_fallbacks"] for row in rows)
                    condition[field+"_root_status_counts"] = dict(sorted(sum((Counter(row[field+"_root_statuses"]) for row in rows), Counter()).items()))
                for field in rows[0]:
                    if field.startswith("lp_") and "_solver_" in field:
                        condition[field+"_counts"] = dict(sorted(Counter(str(row.get(field)) for row in rows).items()))
                    if field.endswith("seconds"):
                        times = [row[field] for row in rows if row.get(field) is not None]
                        condition.update(_prefix(field, student(times)))
                for name, (lhs, rhs, role) in COMPARISONS.items():
                    values = [row[name] for row in grouped["comparisons"][key]]
                    stats = paired(values)
                    condition.update(_prefix(name, stats))
                    paired_conditions.append(dict(base, comparison=name, lhs=lhs, rhs=rhs, role=role,
                                                  positive_means="rhs narrower", **stats))
                conditions.append(condition)
                for name in METHODS:
                    selected = [row for row in grouped["methods"][key] if row["method"] == name]
                    result = dict(base, method=name, shared_event_method=name in SHARED_METHODS,
                                  raw_target_guarantee=name != "z_untransported", n=len(selected))
                    for field in ("lower", "upper", "width", "width_minus_raw_population"):
                        result.update(_prefix(field, student(row[field] for row in selected)))
                    for field in ("contains_raw_set", "lower_miss", "upper_miss", "vacuous", "fallback"):
                        result.update(_prefix(field, binomial_mc(row[field] for row in selected)))
                    result["status_counts"] = dict(sorted(Counter(row["status"] for row in selected).items()))
                    method_conditions.append(result)
                    selected = [row for row in grouped["plugins"][key] if row["method"] == name]
                    result = dict(base, method=name, n=len(selected), descriptive_only=True, finite_sample_guarantee=False,
                                  plugin_alias_of="raw_stratified" if name == "raw_pooled" else None)
                    for field in ("lower", "upper", "width"):
                        result.update(_prefix(field, student(row[field] for row in selected)))
                    for field in ("contains_raw_set", "lower_miss", "upper_miss"):
                        result.update(_prefix(field, binomial_mc(row[field] for row in selected)))
                    plugin_conditions.append(result)
                for name in ("z_global", "z_cellwise"):
                    selected = [row for row in grouped["decompositions"][key] if row["method"] == name]
                    result = dict(base, method=name, n=len(selected), exact_sum_checked=True)
                    for field in DECOMPOSITION_TERMS:
                        result.update(_prefix(field, _signed_summary(row[field] for row in selected)))
                    if sum(result[field+"_exact_mean"] for field in DECOMPOSITION_TERMS[:3]) != result["total_excess_exact_mean"]:
                        raise ArithmeticError("A mean signed decomposition failed exact addition")
                    decomposition_conditions.append(result)
                for name in POPULATIONS:
                    selected = [row for row in grouped["populations"][key] if row["reference"] == name]
                    result = dict(base, reference=name, n=len(selected))
                    for field in ("lower", "upper", "width"):
                        result.update(_prefix(field, student(row[field] for row in selected)))
                    population_conditions.append(result)
    calibration_times = [row["calibration_seconds"] for row in data["selections"] if row["calibration_seconds"] is not None]
    evaluation_times = [row["compute_seconds"] for row in data["evaluations"] if row.get("compute_seconds") is not None]
    runtime = dict(calibration_blocks=len(data["selections"]), outcome_evaluations=len(data["evaluations"]),
                   calibration_worker_seconds=sum(calibration_times), calibration_timed_blocks=len(calibration_times),
                   outcome_worker_seconds=sum(evaluation_times), outcome_timed_evaluations=len(evaluation_times),
                   saved_manifest_compute_seconds=data["manifest"].get("compute_seconds"),
                   saved_manifest_stages=data["manifest"].get("stages", {}),
                   scope="Calibration counted once per block; outcome timings once per profile/prefix. Worker seconds are not wall time.")
    for field in ("block_sampling_seconds", "block_calibration_seconds", "block_outcome_seconds"):
        values = [row[field] for row in data["selections"] if row.get(field) is not None]
        runtime[field+"_sum"] = sum(values)
        runtime[field+"_recorded_blocks"] = len(values)
    panel_receipts = data.get("panel", {}).get("blocks", [])
    runtime["panel_receipt_blocks"] = len(panel_receipts)
    for field in ("sampling_seconds", "calibration_seconds", "outcome_seconds", "serialization_seconds", "compute_seconds", "bytes"):
        values = [row[field] for row in panel_receipts if row.get(field) is not None]
        runtime["panel_"+field+"_sum"] = sum(values) if values else None
        runtime["panel_"+field+"_recorded_blocks"] = len(values)
    runtime["preflight_status"] = data.get("preflight", {}).get("status")
    runtime["saved_forecast"] = data["manifest"].get("forecast")
    return dict(version=VERSION, conditions=conditions, method_conditions=method_conditions,
                paired_conditions=paired_conditions, decomposition_conditions=decomposition_conditions,
                plugin_conditions=plugin_conditions, population_conditions=population_conditions,
                selection_conditions=selection_conditions, runtime=runtime,
                interval_convention="Pointwise approximate paired Student MC95 over 100 independent blocks; no simultaneous or task-population claim.",
                frequency_convention="Per-condition empirical frequencies with descriptive Clopper-Pearson MC95 intervals.",
                useful_reference=USEFUL_REFERENCE,
                useful_reference_scope="Descriptive only; no stopping, publication, retuning, or success criterion.")




def _fmt(value: Any, places: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{places}f}"


def _ci(row: dict[str, Any], prefix: str = "", places: int = 3) -> str:
    prefix = prefix+"_" if prefix else ""
    return f"{_fmt(row.get(prefix+'mean'), places)} [{_fmt(row.get(prefix+'MC95_lower'), places)}, {_fmt(row.get(prefix+'MC95_upper'), places)}]"


def _count_rate(row: dict[str, Any], prefix: str) -> str:
    return f"{row[prefix+'_count']}/{row[prefix+'_n']} [{_fmt(row[prefix+'_CP95_lower'])}, {_fmt(row[prefix+'_CP95_upper'])}]"


def _markdown_table(headers: tuple[str, ...], rows: Any) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |" for row in rows])


def overview(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact counts describe the fixed grid; they are not an averaged effect."""
    conditions = summary["conditions"]
    result = {}
    for name in COMPARISONS:
        rows = [row for row in summary["paired_conditions"] if row["comparison"] == name]
        result[name] = dict(conditions=len(rows), positive_exact_means=sum(row["exact_mean"] > 0 for row in rows),
                           zero_exact_means=sum(row["exact_mean"] == 0 for row in rows),
                           negative_exact_means=sum(row["exact_mean"] < 0 for row in rows),
                           pointwise_MC95_above_zero=sum(row["MC95_lower"] > 0 for row in rows),
                           pointwise_MC95_below_zero=sum(row["MC95_upper"] < 0 for row in rows),
                           means_at_least_0_02=sum(row["exact_mean"] >= USEFUL_REFERENCE for row in rows),
                           exact_positive_block_comparisons=sum(row["positive_count"] for row in rows),
                           exact_equal_block_comparisons=sum(row["zero_count"] for row in rows),
                           exact_negative_block_comparisons=sum(row["negative_count"] for row in rows),
                           scope="Counts over a fixed dependent grid, not independent task trials or an average effect.")
    result["adverse"] = dict(
        unique_identity_releases=sum(row["identity_count"] for row in summary["selection_conditions"]),
        shared_event_failures=sum(row["n"]-row["shared_event_count"] for row in conditions),
        joint_raw_set_noncontainments=sum(row["n"]-row["joint_raw_set_containment_count"] for row in conditions),
        lp_fallback_evaluations=sum(row["lp_fallback_count"] for row in conditions),
        empty_pooled_intersections=sum(row["empty_pooled_intersection_count"] for row in conditions),
        diagnostic_raw_set_noncontainments=sum(row["n"]-row["contains_raw_set_count"] for row in summary["method_conditions"] if row["method"] == "z_untransported"),
        scope="Evaluation counts retain within-block dependence; identity releases count unique calibration blocks.")
    return result








def generate_reports(output_root: str | Path) -> dict[str, Any]:
    """Generate every table/figure/document solely from completed saved blocks."""
    start = time.perf_counter()
    output_root = Path(output_root).resolve()
    data = collect(output_root)
    summary = summarize(data)
    paths, width_data, pair_data = [], [], []
    rows = {
        "condition_summary": summary["conditions"], "method_conditions": summary["method_conditions"],
        "paired_conditions": summary["paired_conditions"], "decomposition_conditions": summary["decomposition_conditions"],
        "plugin_conditions": summary["plugin_conditions"], "population_conditions": summary["population_conditions"],
        "selection_conditions": summary["selection_conditions"], "evaluation_replications": data["evaluations"],
        "method_replications": data["methods"], "paired_replications": data["comparisons"],
        "decomposition_replications": data["decompositions"], "plugin_replications": data["plugins"],
        "population_replications": data["populations"], "selection_replications": data["selections"],
        
    }
    table_paths = {}
    for name, values in rows.items():
        path = output_root / "tables" / (name + ".csv")
        _csv(values, path)
        table_paths[name] = str(path.relative_to(output_root))
    compact = overview(summary)
    summary["counts"] = dict(blocks=len(data["selections"]), conditions=len(summary["conditions"]),
                             evaluations=len(data["evaluations"]), method_intervals=len(data["methods"]),
                             plugin_estimates=len(data["plugins"]))
    summary["manifest_status_at_reporting"] = data["manifest"].get("status")
    summary["saved_replay_checks"] = data["replay"]
    summary["overview"] = compact
    summary["tables"] = table_paths
    summary["figures"] = paths
    summary["report_seconds"] = time.perf_counter()-start
    _dump(summary, output_root / "reporting_summary.json")
    receipt = dict(status="complete", **summary["counts"], tables=table_paths, figures=paths,
                   overview=compact, report_seconds=summary["report_seconds"],
                   summary="reporting_summary.json")
    print(f"Outcome inference reporting: {receipt['blocks']} blocks, {receipt['conditions']} conditions, "
          f"{receipt['evaluations']} evaluations, {receipt['method_intervals']} delivered intervals.")
    for law in LAW_IDS:
        for profile in PROFILES:
            row = next(row for row in summary["conditions"] if row["law_id"] == law and row["profile"] == profile and row["m_ratio"] == 256)
            print(f"  {law} / {profile} / m/K=256: primary {_ci(row, 'raw_stratified_minus_z_global')}; "
                  f"standalone {_ci(row, 'raw_standalone_minus_z_global')}; release cost {_ci(row, 'z_global_minus_raw_pooled')}.")
    print("Adverse saved-evidence counts: " + json.dumps(compact["adverse"], sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results") / VERSION)
    args = parser.parse_args()
    generate_reports(args.output_root)


if __name__ == "__main__":
    main()
