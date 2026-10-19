"""Core calculations for the bounded JMLR empirical-relevance study.

The module only composes frozen, already verified SCA/certificate and outcome
inference components.  In particular, smoothed propensities are proposal data;
all admissions use the unchanged coupled CP certificate.
"""
from __future__ import annotations

import csv
from fractions import Fraction as F
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np

from experiments import jmlr_statistical_baselines_v1_core as ltt_core
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_outcome_inference_v1_core import (
    ALPHA_RAW,
    ALPHA_Z,
    confidence_enclosure,
    intersect_enclosures,
    outcome_family,
    project_assignment,
)
from experiments.jmlr_outcome_inference_v1_lp import build_region, solve_enclosure
from experiments.jmlr_outcome_inference_v1_sampling import counts
from experiments.jmlr_release_capacity_v1 import equal_mass_partition, select_provider
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator


G = F(2)
CERTIFICATE_CAP = F(3, 2)
RHC_SHA256 = "9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8"
RHC_STATES = tuple(itertools.product((0, 1), repeat=3))
LTT_FRACTIONS = ("1/10", "1/5", "1/3", "1/2", "2/3")
LTT_NEW_FRACTIONS = ("1/10", "1/3", "2/3")
LTT_CAPS = (F(5, 4), F(3, 2), F(2))


def _width(interval):
    return float(interval[1] - interval[0])


def choose_ltt_fraction(mean_compression):
    """Global development selection with the frozen smallest-fraction tie rule."""
    if set(mean_compression) != set(LTT_FRACTIONS):
        raise ValueError("selection requires the complete five-fraction grid")
    values = {label: F(value) for label, value in mean_compression.items()}
    best = max(values.values())
    return min((label for label, value in values.items() if value == best), key=F)


def paired_differences(left_rows, right_rows, identity_keys, metric):
    """Pair a metric without silently changing the compared dataset population."""
    def index(rows):
        answer = {}
        for row in rows:
            key = tuple(row[name] for name in identity_keys)
            if key in answer:
                raise ValueError("duplicate paired identity")
            answer[key] = row
        return answer
    left, right = index(left_rows), index(right_rows)
    if set(left) != set(right):
        raise ValueError("paired methods do not have identical dataset identities")
    return tuple((key, float(left[key][metric]) - float(right[key][metric]))
                 for key in sorted(left))


def _interval_fields(prefix, result):
    interval = tuple(result["interval"])
    return {
        prefix + "_lower": interval[0],
        prefix + "_upper": interval[1],
        prefix + "_width": _width(interval),
        prefix + "_status": result.get("status"),
        prefix + "_fallback": bool(result.get("fallback", False)),
    }


def load_rhc(path, expected_sha256=RHC_SHA256):
    """Read the frozen public RHC file and build the declared eight states.

    The state definition is intentionally fixed and contains no fitted
    preprocessing: age >= 65, male sex, and any baseline cancer.  The outcome
    is 30-day survival so the preserved causal target is E[Y(1)].
    """
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("RHC source bytes differ from the frozen public file")
    source = tuple(csv.DictReader(payload.decode("utf-8").splitlines()))
    if len(source) != 5735:
        raise ValueError("RHC file must contain the documented 5,735 records")
    state_to_index = {state: index for index, state in enumerate(RHC_STATES)}
    S, A, Y, ids = [], [], [], []
    missing = []
    for index, row in enumerate(source):
        required = {key: row.get(key, "").strip()
                    for key in ("age", "sex", "ca", "swang1", "dth30", "ptid")}
        if (not required["age"] or required["sex"] not in ("Female", "Male")
                or required["ca"] not in ("No", "Yes", "Metastatic")
                or required["swang1"] not in ("No RHC", "RHC")
                or required["dth30"] not in ("No", "Yes")
                or not required["ptid"]):
            missing.append(index)
            continue
        state = (int(float(required["age"]) >= 65),
                 int(required["sex"] == "Male"),
                 int(required["ca"] != "No"))
        S.append(state_to_index[state])
        A.append(int(required["swang1"] == "RHC"))
        Y.append(int(required["dth30"] == "No"))
        ids.append(required["ptid"])
    if missing:
        raise ValueError(f"Frozen complete-case rule unexpectedly excludes {len(missing)} rows")
    if len(set(ids)) != len(ids):
        raise ValueError("RHC patient identifiers must be unique")
    S, A, Y = (np.asarray(value, dtype=np.int64) for value in (S, A, Y))
    return {
        "S": S,
        "A": A,
        "Y": Y,
        "ptid": tuple(ids),
        "sha256": digest,
        "rows": len(S),
        "treated": int(A.sum()),
        "survived_30d": int(Y.sum()),
        "state_counts": tuple(int(np.sum(S == i)) for i in range(len(RHC_STATES))),
        "state_treated": tuple(int(np.sum(A[S == i])) for i in range(len(RHC_STATES))),
        "state_survived": tuple(int(np.sum(Y[S == i])) for i in range(len(RHC_STATES))),
    }


def rhc_split(data, split_index, calibration_size=4014):
    if not isinstance(split_index, int) or split_index < 0:
        raise ValueError("split_index must be a nonnegative integer")
    if not 0 < calibration_size < data["rows"]:
        raise ValueError("calibration size must create two nonempty cohorts")
    seed = [20261019, 501, split_index]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    permutation = rng.permutation(data["rows"])
    calibration = permutation[:calibration_size]
    outcome = permutation[calibration_size:]
    if len(set(map(int, calibration)).intersection(map(int, outcome))):
        raise ArithmeticError("assignment and outcome cohorts overlap")
    return {
        "split_index": split_index,
        "seed": seed,
        "calibration_indices": tuple(map(int, calibration)),
        "outcome_indices": tuple(map(int, outcome)),
    }


def _segment_sse(prefix_w, prefix_wx, prefix_wx2, left, right):
    weight = prefix_w[right] - prefix_w[left]
    if weight == 0:
        return 0.0
    weighted = prefix_wx[right] - prefix_wx[left]
    square = prefix_wx2[right] - prefix_wx2[left]
    value = square - weighted * weighted / weight
    # Roundoff can create a tiny negative value for identical logits.
    return max(0.0, value)


def propensity_logit_clustering(N, T, order, groups):
    """Global one-dimensional weighted squared-error partition by DP.

    All states remain in the ordered sequence.  Zero-weight states have zero
    objective contribution but still occupy a nonempty whole-state cell.
    Exact objective ties keep the lexicographically earliest cut tuple.
    """
    N, T, order = tuple(map(int, N)), tuple(map(int, T)), tuple(map(int, order))
    K = len(N)
    if len(T) != K or sorted(order) != list(range(K)):
        raise ValueError("counts and order must define one complete support")
    if any(n < 0 or not 0 <= t <= n for n, t in zip(N, T)):
        raise ValueError("require 0 <= T_i <= N_i")
    if not isinstance(groups, int) or not 1 <= groups <= K:
        raise ValueError("groups must lie between one and K")
    smoothed = tuple(F(2 * T[i] + 1, 2 * N[i] + 2) for i in order)
    logits = tuple(math.log(float(value / (1 - value))) for value in smoothed)
    weights = tuple(N[i] for i in order)
    prefix_w = [0.0]
    prefix_wx = [0.0]
    prefix_wx2 = [0.0]
    for weight, value in zip(weights, logits):
        prefix_w.append(prefix_w[-1] + weight)
        prefix_wx.append(prefix_wx[-1] + weight * value)
        prefix_wx2.append(prefix_wx2[-1] + weight * value * value)
    # Each reachable entry stores (objective, cut tuple).  Python tuple order is
    # the frozen deterministic tie rule.
    dp = {(0, 0): (0.0, ())}
    for right in range(1, K + 1):
        for count in range(1, min(groups, right) + 1):
            candidates = []
            for left in range(count - 1, right):
                if (left, count - 1) not in dp:
                    continue
                old_cost, old_cuts = dp[(left, count - 1)]
                cost = old_cost + _segment_sse(prefix_w, prefix_wx, prefix_wx2, left, right)
                cuts = old_cuts + ((left,) if count > 1 else ())
                candidates.append((cost, cuts))
            dp[(right, count)] = min(candidates)
    objective, internal = dp[(K, groups)]
    cuts = (0,) + internal + (K,)
    partition = exact.canonical_partition(tuple(order[a:b]) for a, b in zip(cuts, cuts[1:]))
    exact.validate_partition(partition, K)
    return {
        "partition": partition,
        "groups": groups,
        "objective": objective,
        "ordered_cuts": cuts,
        "ordered_states": order,
        "smoothed_propensities": smoothed,
        "logits": logits,
        "weights": weights,
    }


def _choice_from_partition(region, records, partition, capacity):
    partition = exact.validate_partition(partition, region["K"])
    for cell in partition:
        if cell not in records or records[cell]["status"] != "full":
            records[cell] = cell_certificate(region, cell)
    selected = tuple(records[cell] for cell in partition)
    if any(record["status"] == "failure" for record in selected):
        status, certificate = "unresolved", None
    elif any(record["status"] == "boundary" for record in selected):
        status, certificate = "infinite", None
    else:
        status = "finite"
        certificate = max(record["coupled"] for record in selected)
    return {
        "status": status,
        "completed": status != "unresolved",
        "certificate": certificate,
        "partition": partition,
        "cells": len(partition),
        "capacity": int(capacity),
        "family": "contiguous-propensity-logit-clustering",
    }


def clustering_choices(N, T, order, capacities, region, records):
    answer = {}
    details = {}
    for capacity in capacities:
        result = propensity_logit_clustering(N, T, order, min(int(capacity), len(N)))
        choice = _choice_from_partition(region, records, result["partition"], capacity)
        answer[int(capacity)] = choice
        details[int(capacity)] = result
    return answer, details


def infer_partition(region, records, partition, certificate, outcome_counts,
                    verifier, lp_seconds=10.0):
    """Apply the preserved raw/aggregate outcome construction to one map."""
    on, ot, ov = (tuple(map(int, value)) for value in outcome_counts)
    pbox = tuple(zip(region["mass_lower"], region["mass_upper"]))
    ebox = tuple(zip(region["propensity_lower"], region["propensity_upper"]))
    raw_r, raw_roots = outcome_family(ov, ot, ALPHA_RAW, verifier)
    raw = confidence_enclosure(pbox, ebox, raw_r, G)
    zp, ze, projection = project_assignment(region, partition)
    tz = tuple(sum(ot[i] for i in cell) for cell in partition)
    vz = tuple(sum(ov[i] for i in cell) for cell in partition)
    zr, z_roots = outcome_family(vz, tz, ALPHA_Z, verifier)
    cell_budgets = tuple(G * records[cell]["coupled"] for cell in partition)
    global_result = confidence_enclosure(zp, ze, zr, G * certificate)
    cellwise_result = confidence_enclosure(zp, ze, zr, cell_budgets)
    if not (global_result["interval"][0] <= cellwise_result["interval"][0]
            <= cellwise_result["interval"][1] <= global_result["interval"][1]):
        raise ArithmeticError("cellwise interval escaped global interval")
    lp = solve_enclosure(build_region(pbox, ebox, raw_r, partition, zr, G),
                         time_limit=lp_seconds)
    pooled = intersect_enclosures(tuple(lp["interval"]), tuple(raw["interval"]),
                                  tuple(cellwise_result["interval"]))
    row = {}
    row.update(_interval_fields("global", global_result))
    row.update(_interval_fields("cellwise", cellwise_result))
    row.update(_interval_fields("raw_stratified", raw))
    row.update(_interval_fields("pooled_raw", pooled))
    row.update({
        "zero_treated_groups": sum(value == 0 for value in tz),
        "LP_status": lp.get("status"),
        "LP_fallback": bool(lp.get("fallback", False)),
        "empty_pooled_intersection": bool(pooled.get("empty_intersection", False)),
        "raw_roots": raw_roots,
        "z_roots": z_roots,
        "projection": projection,
        "z_outcome_bounds": zr,
    })
    return row


def observed_split(data, split_index, config):
    split = rhc_split(data, split_index,
                      calibration_size=config["observed_case"]["split"]["calibration_size"])
    calibration = np.asarray(split["calibration_indices"], dtype=np.int64)
    outcome = np.asarray(split["outcome_indices"], dtype=np.int64)
    K = config["observed_case"]["state_catalogue"]["K"]
    N, T = counts(data["S"][calibration], data["A"][calibration], K=K)
    on, ot, ov = counts(data["S"][outcome], data["A"][outcome], data["Y"][outcome], K=K)
    verifier = RootVerifier(RootPolicy(**config["root_policy"]))
    provider = select_provider(N, T, config["observed_case"]["tree_orders"],
                               RHC_STATES, config["observed_case"]["capacities"], verifier)
    records = provider["records"]
    cluster, details = clustering_choices(N, T, provider["order"],
                                          config["observed_case"]["capacities"],
                                          provider["region"], records)
    rows = []
    for capacity in config["observed_case"]["capacities"]:
        choices = {
            "sca": provider["choices"][capacity]["verified_capacity"],
            "equal_calibration_mass": provider["choices"][capacity]["equal_calibration_mass"],
            "propensity_logit_clustering": cluster[capacity],
        }
        for method, choice in choices.items():
            admitted = bool(provider["region"]["valid"] and choice["status"] == "finite"
                            and choice["certificate"] <= CERTIFICATE_CAP)
            row = {
                "study": "rhc_observed",
                "split_index": split_index,
                "primary_split": split_index == config["observed_case"]["split"]["primary_index"],
                "capacity": capacity,
                "method": method,
                "groups": choice["cells"],
                "status": choice["status"],
                "certificate": choice["certificate"],
                "transported_budget": None if choice["certificate"] is None else G * choice["certificate"],
                "admitted": admitted,
                "abstained": not admitted,
                "partition": choice["partition"],
            }
            interval_keys = (
                "global_lower", "global_upper", "global_width",
                "cellwise_lower", "cellwise_upper", "cellwise_width",
                "pooled_raw_lower", "pooled_raw_upper", "pooled_raw_width")
            row["operational_interval_released"] = admitted
            if choice["status"] == "finite":
                inference = infer_partition(
                    provider["region"], records, choice["partition"], choice["certificate"],
                    (on, ot, ov), verifier, config["execution"]["LP_seconds_per_endpoint"])
                if admitted:
                    row.update(inference)
                    row["inadmissible_diagnostic"] = None
                else:
                    row.update({key: None for key in interval_keys})
                    row["inadmissible_diagnostic"] = inference
            else:
                row.update({key: None for key in interval_keys})
                row["inadmissible_diagnostic"] = None
            rows.append(row)
    return {
        "split": split,
        "N": N,
        "T": T,
        "outcome_N": on,
        "outcome_T": ot,
        "outcome_successes": ov,
        "region": provider["region"],
        "order": provider["order"],
        "cell_records": tuple(records[cell] for cell in sorted(records)),
        "cluster_details": details,
        "rows": rows,
        "cohort_disjoint": True,
    }


def release_clustering_block(block, protocol, config):
    """Add the frozen clustering baseline to one saved release-study block."""
    N, T = tuple(block["N"]), tuple(block["T"])
    saved_region = block["region"]
    region = exact.region_from_bounds(
        saved_region["mass_lower"], saved_region["mass_upper"],
        saved_region["propensity_lower"], saved_region["propensity_upper"],
        saved_region.get("known_masses"))
    region.update({key: value for key, value in saved_region.items()
                   if not key.startswith("_")})
    records = {tuple(record["cell"]): record for record in block["cells"]}
    capacities = tuple(map(int, block["choices"].keys()))
    choices, details = clustering_choices(N, T, block["order"], capacities, region, records)
    law = protocol["laws"][block["law_index"]]
    evaluator = TruthEvaluator(law)
    verifier = RootVerifier(RootPolicy(**config["root_policy"]))
    rows = []
    for capacity in capacities:
        choice = choices[capacity]
        admitted = bool(region["valid"] and choice["status"] == "finite"
                        and choice["certificate"] <= CERTIFICATE_CAP)
        truth = None
        if choice["partition"]:
            truth = evaluator.partition(choice["partition"])["value"]
        for outcome_index, outcome in enumerate(block["outcomes"]):
            row = {
                "study": "release_clustering",
                "law_id": block["law_id"],
                "law_index": block["law_index"],
                "replication": block["replication"],
                "profile": outcome["profile"],
                "m_ratio": outcome["m_ratio"],
                "capacity": capacity,
                "method": "propensity_logit_clustering",
                "groups": choice["cells"],
                "status": choice["status"],
                "certificate": choice["certificate"],
                "transported_budget": None if choice["certificate"] is None else G * choice["certificate"],
                "admitted": admitted,
                "abstained": not admitted,
                "partition": choice["partition"],
                "true_distortion": truth,
                "outcome_index": outcome_index,
            }
            if choice["status"] == "finite":
                row.update(infer_partition(region, records, choice["partition"],
                                           choice["certificate"],
                                           (outcome["N"], outcome["T"], outcome["V"]),
                                           verifier, config["execution"]["LP_seconds_per_endpoint"]))
            rows.append(row)
    return {
        "law_id": block["law_id"],
        "law_index": block["law_index"],
        "replication": block["replication"],
        "choices": choices,
        "details": details,
        "rows": rows,
        "completed_cell_records": tuple(records[cell] for cell in sorted(records)),
    }


def _joint_to_counts(joint, state_order, K):
    matrix = np.empty((K, 2), dtype=np.int64)
    matrix[list(state_order)] = np.asarray(joint, dtype=np.int64).reshape(-1, 2)
    return (tuple(int(value) for value in matrix.sum(axis=1)),
            tuple(int(value) for value in matrix[:, 1]))


def extended_ltt_splits(N, T, population_index, r_index, replication, immutable_ids):
    """Five nested uniform fixed-size splits preserving historical endpoints."""
    historical = ltt_core.nested_splits(N, T, population_index, r_index, replication,
                                        immutable_ids=immutable_ids)
    K = len(N)
    state_order = tuple(sorted(range(K), key=lambda index: immutable_ids[index]))
    joint = np.asarray([value for i in state_order for value in (N[i] - T[i], T[i])],
                       dtype=np.int64)
    def ordered_joint(split):
        return np.asarray([value for i in state_order
                           for value in (split["pilot_N"][i] - split["pilot_T"][i],
                                         split["pilot_T"][i])], dtype=np.int64)
    p20 = ordered_joint(historical["1/5"])
    p50 = ordered_joint(historical["1/2"])
    n = int(joint.sum())
    sizes = {"1/10": n // 10, "1/5": n // 5, "1/3": n // 3,
             "1/2": n // 2, "2/3": (2 * n) // 3}
    def draw(population, size, component):
        seed = [20261019, 502, population_index, r_index, replication, component]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        return rng.multivariate_hypergeometric(population, size), seed
    p10, seed10 = draw(p20, sizes["1/10"], 0)
    add33, seed33 = draw(p50 - p20, sizes["1/3"] - sizes["1/5"], 1)
    p33 = p20 + add33
    add67, seed67 = draw(joint - p50, sizes["2/3"] - sizes["1/2"], 2)
    p67 = p50 + add67
    pilots = {"1/10": p10, "1/5": p20, "1/3": p33, "1/2": p50, "2/3": p67}
    seeds = {"1/10": seed10, "1/5": historical["1/5"]["seed"],
             "1/3": seed33, "1/2": historical["1/2"]["seed"], "2/3": seed67}
    answer = {}
    previous = np.zeros_like(joint)
    for label in LTT_FRACTIONS:
        pilot = pilots[label]
        if np.any(pilot < previous) or np.any(pilot > joint):
            raise ArithmeticError("extended LTT split coupling is not nested")
        previous = pilot
        heldout = joint - pilot
        pilot_N, pilot_T = _joint_to_counts(pilot, state_order, K)
        heldout_N, heldout_T = _joint_to_counts(heldout, state_order, K)
        answer[label] = {
            "pilot_N": pilot_N, "pilot_T": pilot_T,
            "heldout_N": heldout_N, "heldout_T": heldout_T,
            "pilot_size": int(pilot.sum()), "heldout_size": int(heldout.sum()),
            "fraction": F(label), "seed": seeds[label],
            "generator": "PCG64 sequential multivariate_hypergeometric coupling",
            "joint_order": "immutable state, action 0 then action 1",
            "immutable_ids": tuple(immutable_ids),
        }
    for label in ("1/5", "1/2"):
        for key in ("pilot_N", "pilot_T", "heldout_N", "heldout_T",
                    "pilot_size", "heldout_size"):
            left, right = answer[label][key], historical[label][key]
            if isinstance(left, tuple):
                equal = tuple(left) == tuple(right)
            else:
                equal = left == right
            if not equal:
                raise AssertionError("historical LTT split changed")
    return answer


def evaluate_new_ltt_fractions(N, T, population_index, r_index, replication,
                               immutable_ids, tree_orders, root_policy):
    splits = extended_ltt_splits(N, T, population_index, r_index, replication,
                                 immutable_ids)
    answer = []
    for label in LTT_NEW_FRACTIONS:
        split = splits[label]
        evaluated = ltt_core.evaluate_split(
            split["pilot_N"], split["pilot_T"], split["heldout_N"], split["heldout_T"],
            tree_orders, caps=LTT_CAPS, root_policy=root_policy,
            immutable_ids=immutable_ids)
        rows = [dict(row, procedure="split_ltt" if row["method"] == "ltt"
                     else "split_coupled", fraction=label,
                     compression=F(len(N) - row["cells"], len(N)))
                for row in evaluated["selections"]]
        answer.append({"fraction": label, "split": split, "core": evaluated, "rows": rows})
    return answer


def evaluate_selected_ltt_with_ablation(N, T, population_index, r_index, replication,
                                        immutable_ids, tree_orders, root_policy,
                                        full_artifact, law, selected_fraction="1/3"):
    """Evaluate A--F at one selected fraction without mixing decompositions."""
    from experiments.jmlr_split_ablation_v1_core import (
        CellEngine, all_proposals, complete_pilot_search, validate_proposals)

    if selected_fraction != "1/3":
        raise ValueError("this frozen evaluation branch is for the selected 1/3 split")
    splits = extended_ltt_splits(N, T, population_index, r_index, replication,
                                 immutable_ids)
    split = splits[selected_fraction]
    base = ltt_core.evaluate_split(
        split["pilot_N"], split["pilot_T"], split["heldout_N"], split["heldout_T"],
        tree_orders, caps=LTT_CAPS, root_policy=root_policy,
        immutable_ids=immutable_ids)
    candidates = base["candidates"]
    pilot_order = base["pilot_order"]
    validate_proposals(candidates, pilot_order, tree_orders)
    held_records = base["control_evidence"]["cells"]
    if isinstance(held_records, dict):
        held_records = tuple(held_records.values())
    held = CellEngine(base["control_evidence"]["region"], held_records)
    modes = [mode for mode in full_artifact["core"]["modes"]
             if mode["mode"] == "unknown_mass"]
    if len(modes) != 1:
        raise ValueError("authoritative full-data unknown-mass mode differs")
    full_mode = modes[0]
    if (tuple(full_artifact["core"]["N"]), tuple(full_artifact["core"]["T"])) != (tuple(N), tuple(T)):
        raise ValueError("full-data evidence counts differ")
    full = CellEngine(full_mode["region"], full_mode["cells"])
    C = all_proposals(candidates, held, LTT_CAPS)
    D = all_proposals(candidates, full, LTT_CAPS)
    E = complete_pilot_search(pilot_order, tree_orders, full, LTT_CAPS)

    rows = []
    for selection in base["selections"]:
        procedure = "A" if selection["method"] == "ltt" else "B"
        rows.append({
            "procedure": procedure,
            "cap": selection["cap"],
            "partition": selection["partition"],
            "cells": selection["cells"],
            "value": selection["value"],
            "value_kind": "ltt_distortion_statistic" if procedure == "A" else "coupled_certificate",
            "status": selection["status"],
            "selection_scope": "saved accepted prefix",
            "stop_index": selection["stop_index"],
            "stop_reason": selection["stop_reason"],
            "accepted_prefix_indices": selection["accepted_prefix_indices"],
            "tested": selection["tested"],
        })
    for procedure, result in (("C", C), ("D", D), ("E", E)):
        for cap, choice in result["by_cap"].items():
            rows.append({
                **choice,
                "procedure": procedure,
                "cap": cap,
                "value": choice["certificate"],
                "value_kind": "coupled_certificate",
            })
    labels = [row for row in full_mode["labels"]
              if row["method"] == "coupled" and row["family"] == "combined"]
    if len(labels) != len(LTT_CAPS):
        raise ValueError("full-data SCA reference grid differs")
    for label in labels:
        rows.append({
            "procedure": "F", "cap": label["cap"],
            "partition": label["partition"], "cells": label["count"],
            "value": label["value"], "value_kind": "coupled_certificate",
            "status": "imported-preserved-selection",
            "selection_scope": "full-order combined family",
        })
    if len(rows) != 18 or {(row["procedure"], row["cap"]) for row in rows} != {
            (procedure, cap) for procedure in "ABCDEF" for cap in LTT_CAPS}:
        raise ArithmeticError("six-procedure evaluation grid differs")
    evaluator = TruthEvaluator(law)
    K = len(N)
    for row in rows:
        partition = exact.validate_partition(row["partition"], K)
        if row["cells"] != len(partition) or row["value"] is None or row["value"] > row["cap"]:
            raise ArithmeticError("procedure returned no completed passing selection")
        truth = evaluator.partition(partition)["value"]
        row.update(
            partition=partition,
            compression=F(K - len(partition), K),
            compression_pp=F(100 * (K - len(partition)), K),
            identity=len(partition) == K,
            true_distortion=truth,
            strict_cap_violation=truth > row["cap"],
        )
    checks = []
    for cap in LTT_CAPS:
        methods = {row["procedure"]: row for row in rows if row["cap"] == cap}
        if methods["C"]["cells"] > methods["B"]["cells"]:
            raise AssertionError("C must search a superset of B's accepted prefix")
        if methods["E"]["cells"] > methods["D"]["cells"]:
            raise AssertionError("E must search a superset of D's proposal list")
        values = {name: methods[name]["compression_pp"] for name in methods}
        decomposition = sum((values[right] - values[left] for left, right in
                             (("A", "B"), ("B", "C"), ("C", "D"),
                              ("D", "E"), ("E", "F"))), F(0))
        if decomposition != values["F"] - values["A"]:
            raise AssertionError("A--F paired decomposition identity failed")
        checks.append({"cap": cap, "C_cells_le_B": True, "E_cells_le_D": True,
                       "decomposition_identity": True})
    return {
        "selected_fraction": selected_fraction,
        "split": split,
        "base": base,
        "rows": rows,
        "checks": checks,
        "cache": {"heldout": held.evidence(), "full": full.evidence()},
        "full_source_version": full_artifact["version"],
    }
