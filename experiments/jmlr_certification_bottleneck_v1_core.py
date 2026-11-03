"""Core calculations for the versioned finite-sample bottleneck study.

The module adds one raw-law confidence construction (conditional Sidak for
propensities, Bonferroni for unknown masses), certified positive-margin
population oracles, a bounded pilot-pooled screen, and lower-bound arithmetic.
It never writes source evidence or uses population truth for selection.
"""
from __future__ import annotations

from fractions import Fraction as Q
import math
from pathlib import Path
import time

from flint import arb, ctx

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier, rational
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import select_families
from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator
from experiments.jmlr_release_capacity_v1_search import candidate_cells
from experiments import jmlr_efficiency_gap_v1_historical as historical
from experiments import jmlr_efficiency_gap_v1_core as efficiency_core


PROJECT = Path(__file__).resolve().parents[1]


def sidak_individual_error(alpha_e, K, bits=80):
    """Largest dyadic eta with (1-eta)^K >= 1-alpha_e, exactly checked."""
    alpha_e = rational(alpha_e)
    if (isinstance(K, bool) or not isinstance(K, int) or K <= 0
            or isinstance(bits, bool) or not isinstance(bits, int) or bits < 16
            or not 0 < alpha_e < 1):
        raise ValueError("Invalid Sidak allocation inputs")
    denominator = 1 << bits
    target_num, target_den = (1-alpha_e).numerator, (1-alpha_e).denominator
    low, high = 0, denominator
    while low + 1 < high:
        middle = (low + high)//2
        safe = (denominator-middle)**K * target_den >= target_num * denominator**K
        if safe:
            low = middle
        else:
            high = middle
    eta = Q(low, denominator)
    product = (1-eta)**K
    if product < 1-alpha_e:
        raise ArithmeticError("Conservative Sidak product check failed")
    if high < denominator and Q(high, denominator) < 1:
        if (1-Q(high, denominator))**K >= 1-alpha_e:
            raise ArithmeticError("Returned Sidak dyadic is not maximal")
    return {"eta": eta, "bits": bits, "product_coverage": product,
            "nominal_coverage": 1-alpha_e, "coverage_slack": product-(1-alpha_e),
            "exact_product_verified": True}


def sidak_confidence_region(N, T, alpha=Q(1,20), known_masses=None,
                            verifier=None, bits=80):
    """Raw region with conditional-product propensity and valid mass coverage."""
    N, T = tuple(N), tuple(T)
    alpha = rational(alpha)
    if not N or len(N) != len(T) or not 0 < alpha < 1:
        raise ValueError("Invalid calibration counts/error")
    if any(isinstance(n, bool) or isinstance(t, bool) or not isinstance(n, int)
           or not isinstance(t, int) or n < 0 or t < 0 or t > n for n, t in zip(N,T)):
        raise ValueError("Require integer 0 <= T_i <= N_i")
    n, K = sum(N), len(N)
    if n <= 0:
        raise ValueError("Calibration total must be positive")
    if known_masses is not None:
        known_masses = tuple(rational(x) for x in known_masses)
        if len(known_masses) != K or any(x <= 0 for x in known_masses) or sum(known_masses) != 1:
            raise ValueError("Known masses must be positive and sum exactly to one")
    alpha_e, alpha_p = (alpha, Q(0)) if known_masses is not None else (alpha/2, alpha/2)
    allocation = sidak_individual_error(alpha_e, K, bits)
    eta_e = allocation["eta"]
    gamma_p = None if known_masses is not None else alpha_p/K
    verifier = verifier or RootVerifier()
    propensity = [verifier.interval(t, m, eta_e) for t, m in zip(T,N)]
    mass = None if known_masses is not None else [verifier.interval(m, n, gamma_p) for m in N]
    a = known_masses if mass is None else [record["lower"] for record in mass]
    b = known_masses if mass is None else [record["upper"] for record in mass]
    region = exact.region_from_bounds(a, b,
        [record["lower"] for record in propensity],
        [record["upper"] for record in propensity], known_masses)
    roots = propensity + (mass or [])
    region.update(N=N, T=T, n=n, alpha=alpha, alpha_e=alpha_e, alpha_p=alpha_p,
                  propensity_gamma=eta_e, mass_gamma=gamma_p,
                  sidak_allocation=allocation, roots={"propensity":propensity,"mass":mass},
                  root_policy=verifier.policy.__dict__,
                  root_fallbacks=sum(record["status"] == "fallback" for record in roots),
                  coverage_argument=("conditional product for propensity intervals; "
                                     "union bound between propensity and mass family events"))
    return region


def truth_in_region(region, law):
    return bool(region["valid"] and all(
        lo_p <= p <= hi_p and lo_e <= e <= hi_e
        for lo_p,hi_p,lo_e,hi_e,p,e in zip(
            region["mass_lower"],region["mass_upper"],region["propensity_lower"],
            region["propensity_upper"],law["p"],law["e"])))


def _arb_rational(value):
    value = rational(value)
    return arb(value.numerator)/value.denominator


def certified_exp_threshold(cap, delta, values, start_bits=160, max_bits=2560):
    """Classify exact rational values against c exp(-delta) with Arb."""
    cap, delta = rational(cap), rational(delta)
    values = tuple(rational(v) for v in values)
    if cap < 1 or delta < 0:
        raise ValueError("Invalid positive-margin threshold")
    old_prec = ctx.prec
    try:
        bits = start_bits
        while bits <= max_bits:
            ctx.prec = bits
            threshold = _arb_rational(cap) * (-_arb_rational(delta)).exp()
            one = arb(1)
            if threshold < one:
                return {"applicable":False,"reason":"threshold-below-one","precision_bits":bits,
                        "cap":cap,"delta":delta,"threshold_decimal":str(threshold)}
            classification = []
            unresolved = False
            for value in values:
                point = _arb_rational(value)
                if point <= threshold:
                    classification.append(True)
                elif point > threshold:
                    classification.append(False)
                else:
                    unresolved = True
                    break
            if not unresolved:
                feasible = [v for v,ok in zip(values,classification) if ok]
                infeasible = [v for v,ok in zip(values,classification) if not ok]
                if not feasible:
                    raise ArithmeticError("Identity must be feasible at every applicable threshold")
                lower = max(feasible)
                separator = lower if not infeasible else (lower+min(infeasible))/2
                if any((v <= separator) != ok for v,ok in zip(values,classification)):
                    raise ArithmeticError("Rational separator changed certified feasibility")
                return {"applicable":True,"cap":cap,"delta":delta,
                        "threshold_decimal":str(threshold),"precision_bits":bits,
                        "separator":separator,"classification":tuple(classification),
                        "closest_feasible":lower,
                        "closest_infeasible":None if not infeasible else min(infeasible),
                        "boundary_comparisons_certified":True}
            bits *= 2
        raise ArithmeticError("Could not separate a rational cell distortion from exp threshold")
    finally:
        ctx.prec = old_prec


def margin_oracles(source, original_rows, deltas, caps):
    """Retrospective J_delta^* rows in each realized candidate family."""
    saved = source["core"]
    order = tuple(saved["order"])
    trees = tuple(exact.build_tree(tuple(o)) for o in saved["tree_orders"])
    truth = {tuple(record["cell"]):record["value"] for record in saved["truth_cells"]}
    family_cells = tuple(tuple(c) for c in saved["family_cells"])
    values = tuple(truth[cell] for cell in family_cells)
    originals = {r["cap"]:r for r in original_rows if r["procedure"] == "original"}
    result = []
    for cap in caps:
        baseline_count = None
        for delta in deltas:
            certified = certified_exp_threshold(cap,delta,values)
            base = {**source["metadata"],"condition_id":Path(originals[cap]["source"]).parent.name,
                    "law_id":source["law_id"],"cap":cap,"delta":delta,
                    "selected_count":originals[cap]["selected_count"],
                    "selected_partition":originals[cap]["selected_partition"],
                    "selection_operational_cap":cap,"selection_changed":False,
                    "threshold_applicable":certified["applicable"],
                    "threshold_decimal":certified["threshold_decimal"],
                    "threshold_precision_bits":certified["precision_bits"]}
            if not certified["applicable"]:
                result.append(dict(base,oracle_count=None,signed_difference=None,
                                   positive_excess_count=None,positive_excess_over_K=None,
                                   selected_at_most_oracle=None,oracle_count_change_from_delta0=None,
                                   threshold_reason=certified["reason"]))
                continue
            value_map = {cell:truth[cell] for cell in family_cells}
            choice = select_families(order,trees,value_map,certified["separator"])["combined"]
            selected_truth = max(truth[cell] for cell in choice["partition"])
            if selected_truth > certified["separator"]:
                raise ArithmeticError("Margin oracle returned a noncertified partition")
            count = int(choice["cells"])
            if delta == 0:
                baseline_count = count
            if baseline_count is None:
                raise ArithmeticError("Delta grid must begin at zero")
            signed = int(base["selected_count"])-count
            result.append(dict(base,oracle_count=count,oracle_partition=choice["partition"],
                               oracle_true_distortion=selected_truth,
                               signed_difference=signed,positive_excess_count=max(signed,0),
                               positive_excess_over_K=Q(max(signed,0),base["K"]),
                               selected_at_most_oracle=base["selected_count"] <= count,
                               oracle_count_change_from_delta0=count-baseline_count,
                               boundary_comparisons_certified=True,
                               closest_feasible=certified["closest_feasible"],
                               closest_infeasible=certified["closest_infeasible"]))
    return result


def process_historical(path, baseline_path, protocol, verifier=None):
    started = time.perf_counter()
    source = historical._load(path)
    baseline = historical._load(baseline_path)
    saved = source["core"]
    law = historical._law(protocol,source["law_id"])
    N,T = tuple(saved["N"]),tuple(saved["T"])
    order = tuple(saved["order"]); trees = tuple(tuple(o) for o in saved["tree_orders"])
    cells = tuple(tuple(c) for c in saved["family_cells"])
    caps = tuple(protocol["caps"]); deltas = tuple(protocol["deltas"])
    oracle_rows = margin_oracles(source,baseline["rows"],deltas,caps)
    original_oracles = {cap:next(r for r in saved["oracles"] if r["family"]=="combined" and r["cap"]==cap)
                        for cap in caps}
    capacities = tuple(sorted({int(r["count"]) for r in original_oracles.values()}))
    baseline_partition = exact.canonical_partition(original_oracles[max(caps)]["partition"])
    alpha = next(m for m in saved["modes"] if m["mode"]=="unknown_mass")["region"]["alpha"]
    tick=time.perf_counter();region=sidak_confidence_region(N,T,alpha,verifier=verifier)
    confidence_seconds=time.perf_counter()-tick
    timings={}
    records,updates,pruning=historical._complete(region,cells,baseline_partition,threshold=max(caps),
        order=order,tree_orders=trees,capacities=capacities,timings=timings,
        certificate_time_key="sidak_certification_seconds")
    choices=historical.rank_capacity_search(order,trees,records,capacities)
    choices=historical._guard_ties(choices,pruning,records,order,trees)
    thresholds=historical._threshold_choices(order,trees,records,caps)
    evaluator=TruthEvaluator(law);covered=truth_in_region(region,law)
    rows=[]
    base_lookup={r["cap"]:r for r in baseline["rows"] if r["procedure"]=="original"}
    for cap in caps:
        selected=thresholds[cap];partition=exact.canonical_partition(selected["partition"])
        true=evaluator.partition(partition)["value"]
        if covered and selected["status"]=="finite" and true>selected["certificate"]:
            raise ArithmeticError("Sidak historical certificate failed on its raw event")
        B=int(original_oracles[cap]["count"]);choice=choices[B]
        rows.append({**source["metadata"],"condition_id":Path(path).parent.name,
            "law_id":law["id"],"procedure":"sidak","cap":cap,"alpha_total":alpha,
            "alpha_e":region["alpha_e"],"alpha_p":region["alpha_p"],
            "propensity_gamma":region["propensity_gamma"],"mass_gamma":region["mass_gamma"],
            "oracle_count":B,"selected_count":selected["cells"],
            "selected_partition":partition,"selected_certificate":selected["certificate"],
            "selection_status":selected["status"],"count_gap":selected["cells"]-B,
            "normalized_count_gap":Q(selected["cells"]-B,law["K"]),
            "selected_true_distortion":true,"unsafe_decision":bool(selected["certificate"]<=cap and true>cap),
            "region_covered":covered,"region_valid":region["valid"],
            "oracle_capacity_status":choice["status"],
            "oracle_capacity_certificate":choice["certificate"],
            "oracle_capacity_partition":choice["partition"],
            "oracle_capacity_groups":choice["cells"],
            "minimum_count_recovered":selected["cells"]==B,
            "original_selected_count":base_lookup[cap]["selected_count"],
            "selected_count_change_vs_original":selected["cells"]-base_lookup[cap]["selected_count"],
            "root_fallbacks":region["root_fallbacks"]})
    return {"kind":"historical","source":str(path),"baseline_source":str(baseline_path),
            "rows":rows,"margin_oracle_rows":oracle_rows,
            "allocation":{k:region[k] for k in ("alpha","alpha_e","alpha_p","propensity_gamma","mass_gamma","sidak_allocation")},
            "new_region":region,"new_cells":[historical._compact(r) for r in records.values()],
            "pruning":pruning,"timings":{"confidence_seconds":confidence_seconds,**timings,
                                           "compute_seconds":time.perf_counter()-started}}


def process_release(i,j,protocol,verifier=None):
    started=time.perf_counter();release_protocol=protocol["release_protocol"]
    path=Path(protocol["release_root"])/"panel"/"blocks"/f"{i}_{j:03d}.json.gz"
    source=historical._load(path);law=release_protocol["laws"][i]
    N,T=tuple(source["N"]),tuple(source["T"]);order=tuple(source["order"])
    trees=tuple(tuple(o) for o in release_protocol["tree_orders"][i])
    cells=tuple(tuple(r["cell"]) for r in source["cells"])
    capacities=tuple(release_protocol["capacity_grids"][i])
    alpha=source["region"]["alpha"]
    first=source["choices"][str(min(capacities))]["verified_capacity"]
    baseline=(exact.canonical_partition(first["partition"]) if first["partition"] is not None
              else (tuple(range(law["K"])),))
    tick=time.perf_counter();region=sidak_confidence_region(N,T,alpha,verifier=verifier)
    confidence_seconds=time.perf_counter()-tick;timings={}
    records,updates,pruning=historical._complete(region,cells,baseline,threshold=Q(3,2),
        order=order,tree_orders=trees,capacities=capacities,timings=timings,
        certificate_time_key="sidak_certification_seconds")
    choices=historical.rank_capacity_search(order,trees,records,capacities)
    choices=historical._guard_ties(choices,pruning,records,order,trees)
    evaluator=TruthEvaluator(law);covered=truth_in_region(region,law);rows=[]
    for B,choice in choices.items():
        true=None if choice["partition"] is None else evaluator.partition(choice["partition"])["value"]
        admitted=bool(choice["status"]=="finite" and choice["certificate"]<=Q(3,2))
        if covered and choice["status"]=="finite" and true>choice["certificate"]:
            raise ArithmeticError("Sidak release certificate failed on its raw event")
        rows.append({"law_index":i,"law_id":law["id"],"replication":j,"K":law["K"],
            "n":sum(N),"B":B,"cap":Q(3,2),"procedure":"sidak",
            "alpha_total":alpha,"alpha_e":region["alpha_e"],"alpha_p":region["alpha_p"],
            "propensity_gamma":region["propensity_gamma"],"mass_gamma":region["mass_gamma"],
            "status":choice["status"],"certificate":choice["certificate"],
            "selected_partition":choice["partition"],"selected_count":choice["cells"],
            "selected_true_distortion":true,"admitted":admitted,
            "unsafe_decision":bool(admitted and true>Q(3,2)),"region_covered":covered,
            "root_fallbacks":region["root_fallbacks"]})
    return {"kind":"release","source":str(path),"rows":rows,"new_region":region,
            "new_cells":[historical._compact(r) for r in records.values()],"pruning":pruning,
            "timings":{"confidence_seconds":confidence_seconds,**timings,
                       "compute_seconds":time.perf_counter()-started}}


def sidak_manuscript_bound(n,K,alpha,p_min,eta,known_mass=False,beta=Q(1,20),
                           g_e=Q(0),g_p=Q(0),root_fallbacks=0,bits=80):
    """Current analytical diameter substitution with Sidak's CP allocation."""
    alpha,beta,p_min,eta,g_e,g_p=map(rational,(alpha,beta,p_min,eta,g_e,g_p))
    alpha_e,alpha_p=(alpha,Q(0)) if known_mass else (alpha/2,alpha/2)
    allocation=sidak_individual_error(alpha_e,K,bits)
    beta_o=beta_e=beta/(2 if known_mass else 3);beta_p=Q(0) if known_mass else beta/3
    Le=math.log(2/float(allocation["eta"]));He=math.log(float(2*K/beta_e))
    effective=float(n*p_min);re=(math.sqrt(Le)+math.sqrt(He))/math.sqrt(effective)+float(g_e)
    Lp=Hp=Ap=Bp=None;rho=0.
    if not known_mass:
        Lp,Hp=math.log(float(2*K/alpha_p)),math.log(float(2*K/beta_p))
        Ap=math.sqrt(2)*(math.sqrt(Lp)+math.sqrt(Hp))
        Bp=2*math.sqrt(Lp*Hp)+2*Lp+2*Hp/3
        rho=Ap/math.sqrt(effective)+Bp/effective+float(g_p/p_min)
    conditions={"occupancy":effective>=8*math.log(float(K/beta_o)),
                "propensity_radius":re<float(eta),"relative_mass_radius":rho<1,
                "no_root_fallback":root_fallbacks==0}
    epsilon=re/((float(eta)-re)*(1-float(eta)+re)) if re<float(eta) else None
    b=(2*epsilon+math.log1p(rho)-math.log1p(-rho)
       if epsilon is not None and rho<1 and all(conditions.values()) else None)
    return {"status":"finite" if b is not None else "sufficient_conditions_fail",
            "b_n":b,"epsilon_n":epsilon,"rho_n":rho,"r_e":re,"conditions":conditions,
            "L_e":Le,"H_e":He,"L_p":Lp,"H_p":Hp,"A_p":Ap,"B_p":Bp,
            "alpha":alpha,"alpha_e":alpha_e,"alpha_p":alpha_p,
            "propensity_gamma":allocation["eta"],"sidak_product_coverage":allocation["product_coverage"],
            "beta":beta,"beta_o":beta_o,"beta_e":beta_e,"beta_p":beta_p,
            "p_min":p_min,"eta":eta,"g_e":g_e,"g_p":g_p,"root_fallbacks":root_fallbacks}


def process_recovery(replication,protocol,verifier=None):
    started=time.perf_counter();source_path=Path(protocol["efficiency_root"])/"panel"/"recovery"/f"{replication:03d}.json.gz"
    source=historical._load(source_path);policy=protocol["recovery_root_policy"]
    if verifier is None:
        from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy
        verifier=RootVerifier(RootPolicy(**policy))
    rows=[];cell=tuple(range(64));confidence_seconds=certification_seconds=0.
    for law_name,law in source["laws"].items():
        true=TruthEvaluator(law).cell(cell)["value"]
        for observation in source["count_paths"]["paths"][law_name]:
            N,T,n=tuple(observation["N"]),tuple(observation["T"]),observation["n"]
            for mass_mode in ("known_mass","unknown_mass"):
                masses=law["p"] if mass_mode=="known_mass" else None
                tick=time.perf_counter();region=sidak_confidence_region(N,T,Q(1,20),masses,verifier)
                confidence_seconds+=time.perf_counter()-tick
                tick=time.perf_counter();record=cell_certificate(region,cell)
                certification_seconds+=time.perf_counter()-tick
                status={"full":"finite","boundary":"infinite","failure":"unresolved"}.get(record["status"])
                if status is None: raise AssertionError("Unexpected recovery certificate status")
                certificate=record["coupled"] if status=="finite" else None
                covered=truth_in_region(region,law)
                if covered and certificate is not None and true>certificate:
                    raise ArithmeticError("Sidak recovery certificate failed on its raw event")
                certified=bool(status=="finite" and certificate<=Q(3,2))
                expansions=efficiency_core.numerical_expansions(region)
                bound=sidak_manuscript_bound(n,64,Q(1,20),min(law["p"]),
                    min(min(law["e"]),min(1-e for e in law["e"])),
                    known_mass=mass_mode=="known_mass",g_e=expansions["g_e"],g_p=expansions["g_p"],
                    root_fallbacks=region["root_fallbacks"])
                rows.append({"replication":replication,"law_id":law_name,"K":64,"n":n,
                    "mass_mode":mass_mode,"procedure":"sidak","status":status,
                    "certificate":certificate,"true_distortion":true,"certified":certified,
                    "unsafe_decision":bool(certified and true>Q(3,2)),"region_covered":covered,
                    "root_fallbacks":region["root_fallbacks"],"alpha_total":Q(1,20),
                    "alpha_e":region["alpha_e"],"alpha_p":region["alpha_p"],
                    "propensity_gamma":region["propensity_gamma"],"mass_gamma":region["mass_gamma"],
                    "analytical_status":bound["status"],"analytical_b_n":bound["b_n"],
                    "analytical_r_e":bound["r_e"],"analytical_rho_n":bound["rho_n"]})
    return {"kind":"recovery","source":str(source_path),"rows":rows,
            "timings":{"confidence_seconds":confidence_seconds,"certification_seconds":certification_seconds,
                       "compute_seconds":time.perf_counter()-started}}


def build_pilot_pooled_regions(N,T,alpha,cells,verifier=None):
    """Conditional pilot-family common event on an independent heldout sample."""
    alpha=rational(alpha);cells=tuple(sorted({tuple(c) for c in cells if len(c)>1}))
    verifier=verifier or RootVerifier();alpha_raw,alpha_pooled=4*alpha/5,alpha/5
    raw=exact.confidence_region(N,T,alpha_raw,verifier=verifier)
    gamma=alpha_pooled/len(cells)
    intervals={cell:verifier.interval(sum(T[i] for i in cell),sum(N[i] for i in cell),gamma)
               for cell in cells}
    return {"raw_region":raw,"pooled_intervals":intervals,"alpha":alpha,
            "alpha_raw":alpha_raw,"alpha_pooled":alpha_pooled,"gamma_pooled":gamma,
            "conditional_cell_count":len(cells)}


def process_pool_screen(path,protocol,verifier=None):
    started=time.perf_counter();source=historical._load(path);split=source["splits"][0]
    if split["fraction"] != "1/5" or source["metadata"]["replication"] != 10:
        raise ValueError("Pooled screen requires frozen 1/5 split at replication10")
    meta=source["metadata"];K=meta["K"]
    trees=tuple(tuple(o) for o in protocol["tree_orders"][str(K)])
    order=tuple(split["core"]["pilot_order"])
    cells=tuple(candidate_cells(order,trees));alpha=Q(1,20);cap=Q(3,2)
    heldout_N,heldout_T=tuple(split["heldout_N"]),tuple(split["heldout_T"])
    evidence=build_pilot_pooled_regions(heldout_N,heldout_T,alpha,cells,verifier)
    timings={}
    tick=time.perf_counter()
    raw={c:cell_certificate(evidence["raw_region"],c) for c in cells}
    timings["heldout_allocation_control_seconds"]=time.perf_counter()-tick
    tick=time.perf_counter()
    pooled={c:efficiency_core.pooled_cell_bound(raw[c],c,evidence["raw_region"],evidence["pooled_intervals"])
            for c in cells}
    timings["heldout_pooled_fusion_seconds"]=time.perf_counter()-tick
    control=historical._threshold_choices(order,trees,raw,(cap,))[cap]
    pooled_choice=historical._threshold_choices(order,trees,pooled,(cap,))[cap]
    full=next(r for r in source["reference"] if r["cap"]==cap)
    tick=time.perf_counter()
    heldout_full_region=exact.confidence_region(heldout_N,heldout_T,alpha,verifier=verifier)
    heldout_full_records={c:cell_certificate(heldout_full_region,c) for c in cells}
    heldout_full=historical._threshold_choices(order,trees,heldout_full_records,(cap,))[cap]
    timings["heldout_full_alpha_seconds"]=time.perf_counter()-tick
    tick=time.perf_counter()
    total_region=exact.confidence_region(tuple(source["N"]),tuple(source["T"]),alpha,verifier=verifier)
    total_records={c:cell_certificate(total_region,c) for c in cells}
    total_pilot=historical._threshold_choices(order,trees,total_records,(cap,))[cap]
    timings["full_data_pilot_family_seconds"]=time.perf_counter()-tick
    law=historical._law(protocol,source["metadata"]["id"])
    evaluator=TruthEvaluator(law)
    rows=[]
    for name,choice,true in (
        ("full_data_original",full,full["truth"]["exact_value"]),
        ("full_data_raw_pilot_family",total_pilot,None),
        ("heldout_raw_full_alpha",heldout_full,None),
        ("heldout_allocation_control",control,None),
        ("heldout_pilot_pooled",pooled_choice,None)):
        partition=choice.get("partition")
        if true is None:
            true=evaluator.partition(partition)["value"]
        certificate=choice.get("value") if name=="full_data_original" else choice.get("certificate")
        rows.append({**meta,"condition_id":meta["id"]+"_r"+str(meta["r"]),"procedure":name,
            "cap":cap,"selected_count":choice.get("cells",len(partition)),"selected_partition":partition,
            "selected_certificate":certificate,"selected_true_distortion":true,
            "selection_status":choice.get("status","finite"),
            "unsafe_decision":bool(certificate is not None and certificate<=cap and true>cap),
            "conditional_pooled_cells":evidence["conditional_cell_count"],
            "root_fallbacks":((evidence["raw_region"]["root_fallbacks"]+
                sum(r["status"]=="fallback" for r in evidence["pooled_intervals"].values()))
                if name in ("heldout_allocation_control","heldout_pilot_pooled") else
                heldout_full_region["root_fallbacks"] if name=="heldout_raw_full_alpha" else
                total_region["root_fallbacks"] if name=="full_data_raw_pilot_family" else 0)})
    return {"kind":"pool_screen","source":str(path),"rows":rows,
            "allocation":{k:evidence[k] for k in ("alpha","alpha_raw","alpha_pooled","gamma_pooled","conditional_cell_count")},
            "timings":{**timings,"compute_seconds":time.perf_counter()-started}}


def bernoulli_kl(x,y):
    x,y=map(float,(x,y))
    return x*math.log(x/y)+(1-x)*math.log((1-x)/(1-y))


def _arb_bernoulli_kl(x,y):
    x,y=map(rational,(x,y));ax,ay=_arb_rational(x),_arb_rational(y)
    return ax*(ax/ay).log()+(1-ax)*((1-ax)/(1-ay)).log()


def _certified_ceiling(builder,approximation,start_bits=160,max_bits=2560):
    """Prove an Arb expression lies in (m-1,m], hence ceil(expr)=m."""
    candidate=math.ceil(float(approximation));old_prec=ctx.prec
    try:
        bits=start_bits
        while bits<=max_bits:
            ctx.prec=bits;value=builder()
            if value > arb(candidate-1) and value <= arb(candidate):
                return {"ceiling":candidate,"unrounded":float(value),
                        "enclosure":str(value),"precision_bits":bits,
                        "adjacent_integer_inequalities_certified":True}
            bits*=2
        raise ArithmeticError("Could not certify lower-bound integer ceiling")
    finally:
        ctx.prec=old_prec


def one_cell_distortion(K,a,b):
    K=int(K);a,b=map(rational,(a,b));q=((K-1)*a+b)/K
    return max(Q(1),exact.odds(a)/exact.odds(q),exact.odds(q)/exact.odds(b))


def lower_bound_record(K,a,b,cap,alpha,gamma):
    K=int(K);a,b,cap,alpha,gamma=map(rational,(a,b,cap,alpha,gamma))
    if not (0<gamma<=Q(1,2) and alpha<1-gamma and one_cell_distortion(K,a,b)>cap):
        raise ValueError("Alternative or error targets do not satisfy the lower-bound assumptions")
    kappa=(b-a)**2/(a*(1-a))
    A_exact=(1-gamma-alpha)**2/(gamma*(1-gamma));A=float(A_exact)
    mixture=math.log1p(K*A)/math.log1p(float(kappa)/K)
    two=K*bernoulli_kl(1-gamma,alpha)/bernoulli_kl(a,b)
    mixture_cert=_certified_ceiling(
        lambda:(_arb_rational(1+K*A_exact).log()/
                _arb_rational(1+kappa/K).log()),mixture)
    two_cert=_certified_ceiling(
        lambda:K*_arb_bernoulli_kl(1-gamma,alpha)/_arb_bernoulli_kl(a,b),two)
    return {"K":K,"a":a,"b":b,"cap":cap,"alpha":alpha,"gamma":gamma,
            "distortion":one_cell_distortion(K,a,b),"kappa":kappa,
            "event_chi_square_constant":A,"event_chi_square_constant_exact":A_exact,
            "two_point_unrounded":two_cert["unrounded"],
            "two_point_ceiling":two_cert["ceiling"],"two_point_ceiling_certificate":two_cert,
            "mixture_unrounded":mixture_cert["unrounded"],
            "mixture_ceiling":mixture_cert["ceiling"],
            "mixture_ceiling_certificate":mixture_cert,
            "mixture_improvement_same_alternative":mixture-two}


def boundary_alternative(K,a,cap,overlap_floor,denominator=10**12):
    """Largest denominator-grid b<a with exact strict distortion above cap."""
    K=int(K);a,cap,overlap_floor=map(rational,(a,cap,overlap_floor))
    low=max(1,int(math.ceil(float(overlap_floor)*denominator)))
    high=int(math.floor(float(a)*denominator))-1
    while low<high:
        middle=(low+high+1)//2;b=Q(middle,denominator)
        if one_cell_distortion(K,a,b)>cap: low=middle
        else: high=middle-1
    b=Q(low,denominator)
    if b<overlap_floor or not one_cell_distortion(K,a,b)>cap:
        raise ArithmeticError("No admissible near-boundary alternative on requested grid")
    next_b=Q(low+1,denominator)
    return {"b":b,"denominator":denominator,"distortion":one_cell_distortion(K,a,b),
            "next_grid_b":next_b,"next_grid_distortion":one_cell_distortion(K,a,next_b),
            "strict_admissibility_exact":True,
            "grid_maximal":one_cell_distortion(K,a,next_b)<=cap}


def analytical_crossing(bound_fn,cap=Q(3,2),lower=1,upper=100_000_000):
    target=math.log(float(cap))
    def passed(n):
        record=bound_fn(n)
        return record["status"]=="finite" and record["b_n"]<target
    if not passed(upper): raise ValueError("Upper search limit does not satisfy sufficient criterion")
    while lower<upper:
        middle=(lower+upper)//2
        if passed(middle): upper=middle
        else: lower=middle+1
    return {"first_integer":lower,"at_threshold":bound_fn(lower),
            "previous_integer":bound_fn(lower-1),"strict_target_log":target}
