"""Provider export and aggregate-only binary-outcome analyst.

The release contains certified summaries, not their raw-data proof evidence.
The analyst only needs this JSON object: neither memberships nor raw-state
counts, propensities, confidence regions, or population laws are accepted.
Outcome coverage is pointwise for a prespecified capacity/profile/cohort size;
the calibration confidence event can cover all assignment-based selections.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction as F
import json
import math
import numbers
from pathlib import Path
import resource
import sys
import time

from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier, rational
from experiments.jmlr_outcome_inference_v1_core import (
    ALPHA_CAL, ALPHA_RAW, ALPHA_Z, confidence_enclosure, outcome_family,
)

VERSION = "jmlr_release_capacity_v1"
CONFIDENCE = {
    "alpha_cal": str(ALPHA_CAL),
    "alpha_raw": str(ALPHA_RAW),
    "alpha_z": str(ALPHA_Z),
    "outcome_allocation": "alpha_z / number_of_released_groups",
    "outcome_scope": "pointwise_prespecified_capacity_profile_cohort_size",
    "assignment_scope": "common_calibration_event_all_capacities",
}
_TOP_KEYS = {"version", "groups", "raw_budget", "confidence", "root_policy"}
_GROUP_KEYS = {"label", "counts", "mass_bounds", "propensity_bounds", "distortion_bound"}
_COUNT_KEYS = {"size", "treated", "treated_successes"}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _q(value, name):
    try:
        return rational(value)
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a finite exact rational") from error


def _box(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a lower/upper pair")
    lo, hi = (_q(x, name) for x in value)
    if not 0 <= lo <= hi <= 1:
        raise ValueError(f"{name} must be an ordered subinterval of [0,1]")
    return lo, hi


def _policy(value):
    if not isinstance(value, dict) or set(value) != set(asdict(RootPolicy())):
        raise ValueError("Unrecognized root policy metadata")
    for key, item in value.items():
        if key == "endpoint_seconds":
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError("Invalid root time budget")
        elif isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("Root precision and resource counts must be integers")
    return RootPolicy(**value)


def _validate_release(artifact):
    """Reject extra fields so this interface cannot silently import raw data."""
    if not isinstance(artifact, dict) or set(artifact) != _TOP_KEYS:
        raise ValueError("Release does not match the aggregate-only top-level allowlist")
    if artifact["version"] != VERSION or artifact["confidence"] != CONFIDENCE:
        raise ValueError("Unknown release version or confidence allocation/scope")
    G = _q(artifact["raw_budget"], "raw_budget")
    if G < 1:
        raise ValueError("The raw sensitivity budget must be at least one")
    policy = _policy(artifact["root_policy"])
    groups = artifact["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("The release must include every selected group")
    pbox, ebox, U, N, T, V = [], [], [], [], [], []
    for j, group in enumerate(groups):
        if not isinstance(group, dict) or set(group) != _GROUP_KEYS:
            raise ValueError("Release group does not match the aggregate-only allowlist")
        if group["label"] != f"group_{j:04d}":
            raise ValueError("Anonymous labels must follow canonical release order")
        counts = group["counts"]
        if not isinstance(counts, dict) or set(counts) != _COUNT_KEYS:
            raise ValueError("Malformed aggregate counts")
        n, t, v = (_integer(counts[key], key) for key in ("size", "treated", "treated_successes"))
        if not v <= t <= n:
            raise ValueError("Aggregate counts require 0 <= successes <= treated <= size")
        pbox.append(_box(group["mass_bounds"], "mass_bounds"))
        ebox.append(_box(group["propensity_bounds"], "propensity_bounds"))
        u = _q(group["distortion_bound"], "distortion_bound")
        if u < 1:
            raise ValueError("Completed distortion bounds must be at least one")
        U.append(u); N.append(n); T.append(t); V.append(v)
    if sum(lo for lo, _ in pbox) > 1 or sum(hi for _, hi in pbox) < 1:
        raise ValueError("Released mass bounds define an empty simplex")
    return tuple(pbox), tuple(ebox), tuple(U), tuple(N), tuple(T), tuple(V), G, policy


def release_bytes(artifact):
    """Actual canonical payload bytes, including exact-number encoding costs."""
    _validate_release(artifact)
    return json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def export_release(region, partition, records, N, T, V, verifier=None, *, G=F(2)):
    """Project a completed finite provider certificate and aggregate a cohort.

    ``records`` may map cells to their records or be the selected records in
    partition order. Full finite certificates are required, even for releases
    failing an external reporting-capacity/sensitivity admission rule. Such
    diagnostic artifacts must be labeled by the caller and are not releases
    admitted under that rule. No outcome-dependent selection occurs here.

    The provider remains responsible for verifying the raw certification
    evidence. This export validates the completed-record contract; it does not
    reconstruct the cell envelope from the exported summaries.
    """
    from experiments import jmlr_verified_adaptive_v1_math as exact
    from experiments.jmlr_outcome_inference_v1_core import project_assignment

    if not region.get("valid"):
        raise ValueError("Cannot export from an invalid confidence region")
    if "alpha" in region and region["alpha"] != ALPHA_CAL:
        raise ValueError("Release requires the declared calibration allocation")
    partition = exact.validate_partition(partition, region["K"])
    vectors = []
    for name, values in (("N", N), ("T", T), ("V", V)):
        values = tuple(_integer(x, name) for x in values)
        if len(values) != region["K"]:
            raise ValueError("Provider counts must include every declared raw state")
        vectors.append(values)
    N, T, V = vectors
    if any(not v <= t <= n for n, t, v in zip(N, T, V)):
        raise ValueError("Provider counts require 0 <= V <= T <= N")
    G = _q(G, "G")
    if G < 1:
        raise ValueError("The raw sensitivity budget must be at least one")
    if isinstance(records, dict):
        selected = tuple(records[c] for c in partition)
    else:
        selected = tuple(records)
    if len(selected) != len(partition):
        raise ValueError("One complete cell record is required per selected group")
    U = []
    for cell, record in zip(partition, selected):
        if record.get("status") != "full" or record.get("coupled") is None:
            raise ValueError("Only completed finite cell certificates can be exported")
        if "cell" in record and tuple(record["cell"]) != cell:
            raise ValueError("Cell certificate belongs to a different selected group")
        u = _q(record["coupled"], "completed certificate")
        if u < 1:
            raise ValueError("Completed distortion bounds must be at least one")
        U.append(u)
    # The region is immutable scientific input throughout a provider block;
    # reuse its exact projection across profiles and independent cohort prefixes.
    projection_key = (partition, tuple(region['mass_lower']), tuple(region['mass_upper']),
                      tuple(region['propensity_lower']), tuple(region['propensity_upper']))
    cache = region.setdefault('_release_projection_cache', {})
    if projection_key not in cache:
        cache[projection_key] = project_assignment(region, partition)
    pbox, ebox, _ = cache[projection_key]
    policy = verifier.policy if verifier is not None else _policy(region.get("root_policy", asdict(RootPolicy())))
    artifact = {
        "version": VERSION,
        "raw_budget": str(G),
        "confidence": dict(CONFIDENCE),
        "root_policy": asdict(policy),
        "groups": [
            {"label": f"group_{j:04d}",
             "counts": {"size": sum(N[i] for i in cell),
                        "treated": sum(T[i] for i in cell),
                        "treated_successes": sum(V[i] for i in cell)},
             "mass_bounds": list(map(str, pbox[j])),
             "propensity_bounds": list(map(str, ebox[j])),
             "distortion_bound": str(U[j])}
            for j, cell in enumerate(partition)
        ],
    }
    _validate_release(artifact)
    return artifact


def analyze_release(artifact, verifier=None):
    """Infer using only anonymous aggregate counts and certified summaries.

    CP bounds allocate alpha_z over *all* released groups, including groups
    with zero treated observations. Numerical root fallback to [0,1] is
    retained and counted; it is distinct from operational release admission.
    No nominal conditional-on-admission coverage is claimed.
    """
    start = time.perf_counter()
    pbox, ebox, U, N, T, V, G, policy = _validate_release(artifact)
    if verifier is None:
        verifier = RootVerifier(policy)
    elif verifier.policy != policy:
        raise ValueError("Analyst verifier does not match the released root policy")
    hits_before, misses_before = verifier.cache_hits, verifier.cache_misses
    rb, roots = outcome_family(V, T, ALPHA_Z, verifier)
    certificate = max(U)
    methods = {
        "z_global": confidence_enclosure(pbox, ebox, rb, G * certificate),
        "z_cellwise": confidence_enclosure(pbox, ebox, rb, tuple(G * u for u in U)),
    }
    a, b = methods["z_global"]["interval"], methods["z_cellwise"]["interval"]
    if not a[0] <= b[0] <= b[1] <= a[1]:
        raise ArithmeticError("Cellwise refinement escaped the global enclosure")
    return {
        "group_count": len(U), "cohort_size": sum(N),
        "certificate": certificate, "transported_budget": G * certificate,
        "cell_budgets": tuple(G * u for u in U), "methods": methods,
        "z_outcome_bounds": rb, "z_outcome_roots": roots,
        "root_policy": asdict(policy),
        "root_fallbacks": sum(root["status"] == "fallback" for root in roots),
        "root_cache_hits": verifier.cache_hits - hits_before,
        "root_cache_misses": verifier.cache_misses - misses_before,
        # macOS reports bytes; Linux reports KiB. This is the process lifetime
        # high-water mark, so a shared batch retains prior analyses and cache.
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                          * (1 if sys.platform == "darwin" else 1024),
        "payload_bytes": len(release_bytes(artifact)),
        "compute_seconds": time.perf_counter() - start,
    }


def _jsonable(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _restrict_project_data(release_path):
    """Deny project data reads except the explicitly supplied release bundle.

    Installed before opening the bundle. Code imports remain available, while
    attempted access to any other noncode project file raises PermissionError.
    The caller also runs in a directory containing no other benchmark data.
    This is a data-flow demonstration, not a security isolation mechanism.
    """
    project = Path(__file__).resolve().parents[1]
    allowed_input = release_path.resolve()
    receipt = {"code_data_access_restricted": True,
               "allowed_data_input": str(allowed_input),
               "noncode_project_access_attempts": [],
               "data_input_open_count": 0}

    def audit(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        candidate = Path(args[0]).resolve()
        if candidate == allowed_input:
            receipt["data_input_open_count"] += 1
            return
        if candidate.is_relative_to(project) and candidate.suffix not in {".py", ".pyc", ".so"}:
            receipt["noncode_project_access_attempts"].append(str(candidate))
            raise PermissionError(f"Aggregate-only process attempted project data access: {candidate}")

    sys.addaudithook(audit)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze one aggregate-only release JSON file")
    parser.add_argument("release", type=Path)
    parser.add_argument("--batch", action="store_true", help="Input is an array of release artifacts")
    parser.add_argument("--restrict-project-data", action="store_true",
                        help="Deny other noncode project data inputs and return an access receipt")
    args = parser.parse_args(argv)
    receipt = _restrict_project_data(args.release) if args.restrict_project_data else None
    artifact = json.loads(args.release.read_text(encoding="utf-8"))
    if args.batch:
        if not isinstance(artifact, list) or not artifact:
            raise ValueError("Batch input must be a nonempty array of release artifacts")
        verifier = RootVerifier(_validate_release(artifact[0])[-1])
        answer = [analyze_release(item, verifier) for item in artifact]
    else:
        answer = analyze_release(artifact)
    if receipt is not None:
        for item in answer if args.batch else [answer]:
            item["data_access_receipt"] = dict(receipt)
    print(json.dumps(_jsonable(answer), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
