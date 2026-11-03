"""Fixed-tree pooled certification and nonselective diameter diagnostics.

The alternative is min(U_raw, V_pooled), a valid upper certificate on a
common raw-and-pooled confidence event. It is not the exact supremum over
the intersection region. No population law enters its construction.
"""
from __future__ import annotations

from fractions import Fraction as F
import math

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier, rational


def pooled_groups(tree_orders):
    """Distinct nonsingleton nodes of the predeclared fixed balanced trees."""
    orders = tuple(tuple(order) for order in tree_orders)
    if not orders:
        raise ValueError("At least one fixed tree is required")
    K = len(orders[0])
    if not K or any(sorted(order) != list(range(K)) for order in orders):
        raise ValueError("Every fixed tree order must permute the complete support")
    return tuple(sorted({cell for order in orders for cell in exact.build_tree(order) if len(cell) > 1}))


def build_regions(N, T, alpha, tree_orders, verifier=None, known_masses=None, *, pooled_cells=None):
    """Spend 4 alpha/5 on raw coordinates and alpha/5 on fixed pooled nodes.

    For a fixed node C, conditional on N_C, T_C is binomial with success
    q_C=P(A=1|S in C). This statement does not condition on the full vector
    of raw state counts. Overlapping-node dependence needs no independence
    assumption because error is allocated by a union bound.
    """
    alpha = rational(alpha)
    if not 0 < alpha < 1:
        raise ValueError("Confidence error must lie in (0,1)")
    N, T = tuple(N), tuple(T)
    tree_orders = tuple(tuple(order) for order in tree_orders)
    groups = pooled_groups(tree_orders)
    if len(N) != len(tree_orders[0]):
        raise ValueError("Fixed trees and count support differ")
    requested = groups if pooled_cells is None else tuple(sorted({exact._cell(cell, len(N)) for cell in pooled_cells}))
    if any(cell not in groups for cell in requested):
        raise ValueError("Requested pooled cells must belong to the full fixed-node collection")
    verifier = verifier or RootVerifier()
    alpha_raw, alpha_pooled = 4*alpha/5, alpha/5
    raw = exact.confidence_region(N, T, alpha_raw, known_masses=known_masses, verifier=verifier)
    gamma = alpha_pooled/len(groups) if groups else None
    intervals = {cell: verifier.interval(sum(T[i] for i in cell), sum(N[i] for i in cell), gamma)
                 for cell in requested}
    return {"raw_region": raw, "pooled_intervals": intervals, "pooled_groups": groups,
            "computed_pooled_groups": requested,
            "alpha": alpha, "alpha_raw": alpha_raw, "alpha_pooled": alpha_pooled,
            "gamma_pooled": gamma, "root_fallbacks": raw["root_fallbacks"]
            + sum(record["status"] == "fallback" for record in intervals.values())}


def direct_pooled_bound(cell, raw_region, pooled_interval):
    """Exact independent-endpoint odds bound for one fixed pooled node."""
    cell = exact._cell(cell, raw_region["K"])
    if len(cell) == 1:
        return {"status": "finite", "value": F(1), "reason": "singleton-identity"}
    if not raw_region["valid"]:
        return {"status": "unresolved", "value": None, "reason": "invalid-raw-region"}
    try:
        if isinstance(pooled_interval, dict):
            lo, hi = rational(pooled_interval["lower"]), rational(pooled_interval["upper"])
        else:
            lo, hi = map(rational, pooled_interval)
        if not 0 <= lo <= hi <= 1 or hi == 0 or lo == 1:
            raise ValueError("Pooled interval has no interior assignment law")
        lower, upper = raw_region["propensity_lower"], raw_region["propensity_upper"]
        if lo == 0 or hi == 1 or any(lower[i] == 0 or upper[i] == 1 for i in cell):
            return {"status": "infinite", "value": None,
                    "reason": "pooled-or-raw-propensity-boundary"}
        value = max(F(1), *(exact.odds(upper[i])/exact.odds(lo) for i in cell),
                    *(exact.odds(hi)/exact.odds(lower[i]) for i in cell))
        return {"status": "finite", "value": value, "reason": None}
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        return {"status": "unresolved", "value": None, "reason": str(error)}


def pooled_cell_bound(raw_record, cell, raw_region, pooled_intervals):
    """Complete the prescribed min(U_raw,V) only when its value is proved.

    A finite V alone is a valid upper certificate when raw optimization is
    unresolved, but is not a completed value of the prescribed minimum. Such
    records retain status=failure and save valid_upper_bound separately. A
    partial raw lower bound >= V proves the minimum equals V without finishing
    the raw optimization. Cells outside the fixed-node collection use U_raw.
    """
    cell = exact._cell(cell, raw_region["K"])
    if "cell" in raw_record and tuple(raw_record["cell"]) != cell:
        raise ValueError("The raw certificate belongs to a different cell")
    raw_status = raw_record.get("status")
    pooled = (direct_pooled_bound(cell, raw_region, pooled_intervals[cell])
              if cell in pooled_intervals else
              {"status": "unavailable", "value": None, "reason": "not-a-fixed-pooled-node"})
    answer = {"cell": cell, "coupled": None, "status": "failure", "completed": False,
              "exact_min_completed": False, "raw_status": raw_status,
              "pooled_status": pooled["status"], "pooled_bound": pooled["value"],
              "valid_upper_bound": pooled["value"], "rejection_bound": None,
              "reason": "raw-minimum-unresolved", "anchors": [],
              "certificate_semantics": "min_raw_supremum_and_pooled_bound_not_joint_supremum"}

    def complete(value, reason=None):
        return dict(answer, coupled=value, status="full", completed=True,
                    exact_min_completed=True, valid_upper_bound=value, reason=reason)

    if len(cell) == 1:
        return complete(F(1), "singleton-identity")
    if raw_status == "full" and raw_record.get("coupled") is not None:
        raw_value = rational(raw_record["coupled"])
        if raw_value < 1:
            raise ValueError("A completed raw bound must be at least one")
        if pooled["status"] == "unresolved":
            return dict(answer, valid_upper_bound=raw_value, reason="pooled-computation-unresolved")
        value = min(raw_value, pooled["value"]) if pooled["status"] == "finite" else raw_value
        return complete(value)
    if raw_status == "boundary":
        if pooled["status"] == "finite":
            return complete(pooled["value"], "raw-infinity-pooled-finite")
        if pooled["status"] in ("infinite", "unavailable"):
            return dict(answer, status="boundary", completed=True, exact_min_completed=True,
                        reason="mathematical-infinity")
        return dict(answer, reason="pooled-computation-unresolved")
    if raw_status == "rejected" and raw_record.get("rejection_bound") is not None:
        lower = rational(raw_record["rejection_bound"])
        if lower < 1:
            raise ValueError("A raw rejection lower bound must be at least one")
        if pooled["status"] == "finite" and lower >= pooled["value"]:
            return complete(pooled["value"], "partial-raw-lower-bound-proves-minimum")
        answer["rejection_bound"] = min(lower, pooled["value"]) if pooled["status"] == "finite" else lower
    return answer


def numerical_expansions(region):
    """Maximum saved verified expansion bounds, with root fallbacks explicit."""
    roots = region.get("roots", {})

    def family(name):
        records = roots.get(name) or ()
        values = [rational(record[key]) for record in records
                  for key in ("lower_expansion_bound", "upper_expansion_bound")]
        return max(values, default=F(0)), sum(r["status"] == "fallback" for r in records)

    ge, fe = family("propensity")
    gp, fp = family("mass")
    return {"g_e": ge, "g_p": gp, "propensity_root_fallbacks": fe,
            "mass_root_fallbacks": fp, "root_fallbacks": fe+fp,
            "saved_roots_available": bool(roots)}


def _truth(region, p, e):
    p, e = tuple(map(rational, p)), tuple(map(rational, e))
    if len(p) != region["K"] or len(e) != region["K"] or any(x <= 0 for x in p) or sum(p) != 1:
        raise ValueError("Truth masses must be positive and sum exactly to one on the declared support")
    if any(not 0 < x < 1 for x in e):
        raise ValueError("Truth propensities must be interior")
    return p, e


def realized_radii(region, p, e):
    """Privileged realized perturbation radii; never inputs to certification.

    Mass coordinates are projected through the bounded simplex before taking
    their maximal relative deviation. Exact odds ratios and relative errors
    are retained; logarithms are descriptive floating-point diagnostics.
    """
    p, e = _truth(region, p, e)
    if not region["valid"]:
        return {"status": "unresolved", "reason": "invalid-region", "epsilon_realized": None,
                "rho_realized": None, "b_realized": None, "truth_in_region": False}
    truth_in = all(a <= pi <= b and lo <= ei <= hi for a, b, lo, hi, pi, ei in
                   zip(region["mass_lower"], region["mass_upper"], region["propensity_lower"],
                       region["propensity_upper"], p, e))
    ratios, mass_bounds = [F(1)], []
    infinite = False
    for i, ei in enumerate(e):
        for endpoint in (region["propensity_lower"][i], region["propensity_upper"][i]):
            if endpoint in (0, 1):
                infinite = True
            else:
                ratio = exact.odds(endpoint)/exact.odds(ei)
                ratios.extend((ratio, 1/ratio))
        _, lo, hi, _ = exact._cell_mass(region, (i,))
        mass_bounds.append((F(lo, region["_D"]), F(hi, region["_D"])))
    rho = max(abs(endpoint/pi-1) for pi, bounds in zip(p, mass_bounds) for endpoint in bounds)
    R = None if infinite else max(ratios)
    epsilon = None if infinite else math.log(R.numerator)-math.log(R.denominator)
    b = None if infinite or rho >= 1 else 2*epsilon + math.log1p(float(rho))-math.log1p(-float(rho))
    return {"status": "finite" if b is not None else "infinite", "truth_in_region": truth_in,
            "epsilon_realized": epsilon, "logit_radius_ratio": R, "rho_realized": rho,
            "b_realized": b, "mass_projected_bounds": tuple(mass_bounds),
            "infinite_logit_radius": infinite, "rho_below_one": rho < 1}


def manuscript_bound(n, K, alpha, p_min, eta, known_mass=False, *, beta=F(1,20),
                     g_e=F(0), g_p=F(0), root_fallbacks=0, alpha_e_override=None):
    """The active manuscript's sufficient b_n with prespecified beta splits.

    All allocation constants remain exact; log/square-root evaluations here
    are diagnostics rather than additional verified selection predicates.
    """
    alpha, beta, p_min, eta, g_e, g_p = map(rational, (alpha, beta, p_min, eta, g_e, g_p))
    if (isinstance(n, bool) or not isinstance(n, int) or n <= 0 or isinstance(K, bool)
            or not isinstance(K, int) or K <= 0 or not 0 < alpha < 1 or not 0 < beta < 1
            or alpha+beta >= 1 or not 0 < p_min <= 1 or not 0 < eta <= F(1,2)
            or g_e < 0 or g_p < 0):
        raise ValueError("Invalid manuscript diameter inputs")
    alpha_e, alpha_p = (alpha, F(0)) if known_mass else (alpha/2, alpha/2)
    if alpha_e_override is not None:
        alpha_e = rational(alpha_e_override)
        if not 0 < alpha_e or alpha_e+alpha_p > alpha:
            raise ValueError("Propensity family allocation must respect the total error budget")
    beta_o = beta_e = beta/(2 if known_mass else 3)
    beta_p = F(0) if known_mass else beta/3
    Le, He = math.log(float(2*K/alpha_e)), math.log(float(2*K/beta_e))
    effective_n = float(n*p_min)
    re = (math.sqrt(Le)+math.sqrt(He))/math.sqrt(effective_n)+float(g_e)
    Lp = Hp = Ap = Bp = None
    rho = 0.
    if not known_mass:
        Lp, Hp = math.log(float(2*K/alpha_p)), math.log(float(2*K/beta_p))
        Ap = math.sqrt(2)*(math.sqrt(Lp)+math.sqrt(Hp))
        Bp = 2*math.sqrt(Lp*Hp)+2*Lp+2*Hp/3
        rho = Ap/math.sqrt(effective_n)+Bp/effective_n+float(g_p/p_min)
    occupancy_rhs = 8*math.log(float(K/beta_o))
    conditions = {"occupancy": effective_n >= occupancy_rhs, "propensity_radius": re < float(eta),
                  "relative_mass_radius": rho < 1, "no_root_fallback": root_fallbacks == 0}
    epsilon = re/((float(eta)-re)*(1-float(eta)+re)) if re < float(eta) else None
    b = (2*epsilon+math.log1p(rho)-math.log1p(-rho)
         if epsilon is not None and rho < 1 and all(conditions.values()) else None)
    return {"status": "finite" if b is not None else
            "invalid_theory_root_fallback" if root_fallbacks else "sufficient_conditions_fail",
            "b_n": b, "epsilon_n": epsilon, "rho_n": rho, "r_e": re,
            "conditions": conditions, "n_p_min": effective_n, "occupancy_rhs": occupancy_rhs,
            "p_min": p_min, "eta": eta, "alpha": alpha, "alpha_e": alpha_e, "alpha_p": alpha_p,
            "beta": beta, "beta_o": beta_o, "beta_e": beta_e, "beta_p": beta_p,
            "g_e": g_e, "g_p": g_p, "root_fallbacks": root_fallbacks,
            "L_e": Le, "H_e": He, "L_p": Lp, "H_p": Hp, "A_p": Ap, "B_p": Bp,
            "numerical_role": "descriptive_float_evaluation_of_manuscript_sufficient_bound"}


def manuscript_diagnostics(region, p, e, beta=F(1,20)):
    """Use the actual region allocation and truth-derived least mass/overlap."""
    p, e = _truth(region, p, e)
    expansions = numerical_expansions(region)
    result = manuscript_bound(int(region["n"]), region["K"], region["alpha"], min(p),
                              min(min(e), min(1-ei for ei in e)),
                              known_mass=region["known_masses"] is not None, beta=beta,
                              g_e=expansions["g_e"], g_p=expansions["g_p"],
                              root_fallbacks=expansions["root_fallbacks"],
                              alpha_e_override=(region["gamma"]*region["K"] if "gamma" in region else None))
    result["saved_roots_available"] = expansions["saved_roots_available"]
    if not expansions["saved_roots_available"]:
        result.update(status="missing_saved_root_evidence", b_n=None)
    return result


def truth_mass_slice(region, p):
    """Privileged feasible slice fixing p while retaining the same e intervals.

    When truth masses are outside the original mass box the fixed-mass region
    is supplied for diagnostics but is explicitly *not* a subset of that box.
    Comparisons relying on subset monotonicity are conditional on feasibility.
    """
    p = tuple(map(rational, p))
    if len(p) != region["K"] or any(x <= 0 for x in p) or sum(p) != 1:
        raise ValueError("Slice masses must be positive and sum exactly to one")
    feasible = bool(region["valid"] and all(a <= pi <= b for a, b, pi in
                    zip(region["mass_lower"], region["mass_upper"], p)))
    sliced = exact.region_from_bounds(p, p, region["propensity_lower"], region["propensity_upper"], p)
    for key in ("N", "T", "n", "alpha", "gamma", "root_policy", "root_fallbacks"):
        if key in region:
            sliced[key] = region[key]
    if "roots" in region:
        sliced["roots"] = {"propensity": region["roots"]["propensity"], "mass": None}
    return {"region": sliced, "truth_mass_feasible": feasible,
            "conditional_on_mass_feasibility": True,
            "propensity_intervals_unchanged": True, "original_alpha": region.get("alpha")}
