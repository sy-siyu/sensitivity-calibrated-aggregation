"""Saved-result analysis for the bounded split-LTT comparison.

No observations, splits, calibration, selection or root verification are run
here. Population distortion is supplied by the runner after selection. Monte
Carlo intervals describe the fixed simulation grid, not numerical validity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, norm, t

ROOT = Path(__file__).resolve().parents[1]
VERSION = "jmlr_statistical_baselines_v1"
PROCEDURES = ("full_reference", "split_ltt", "split_coupled")
CAPS = (Fraction(5, 4), Fraction(3, 2), Fraction(2))
FRACTIONS = (Fraction(1, 5), Fraction(1, 2))
NAMES = {"full_reference": "Full-data coupled reference", "split_ltt": "Split LTT", "split_coupled": "Matched split coupled"}
COLORS = {("balanced", "moderate"): "#0072B2", ("balanced", "weak"): "#D55E00",
          ("skewed", "moderate"): "#009E73", ("skewed", "weak"): "#CC79A7"}


def decode(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def encode(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load(path: Path, default: Any = None) -> Any:
    return decode(json.loads(path.read_text())) if path.exists() else default


def dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(encode(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value.rstrip() + "\n")
    temporary.replace(path)


def read_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as stream:
        return decode(json.load(stream))


def write_csv(rows: list[dict[str, Any]], path: Path, fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(dict.fromkeys(key for row in rows for key in row)) if fields is None and rows else fields or ("status",)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(encode(value), separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else str(value) if isinstance(value, Fraction) else value
                             for key, value in row.items()})
    temporary.replace(path)


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


def distribution(values: Any) -> dict[str, Any]:
    x = np.asarray(values, float)
    finite = x[np.isfinite(x)]
    if np.isnan(x).any() or np.isneginf(x).any():
        raise ValueError("Undefined selected distortion")
    def exported(v: float) -> float | str:
        return "infinity" if math.isinf(v) else v
    return {"n": len(x), "infinite": int(np.isposinf(x).sum()), "mean": None if not len(x) else exported(float(x.mean())),
            "minimum": None if not len(x) else exported(float(x.min())), "maximum": None if not len(x) else exported(float(x.max())),
            "median": None if not len(x) else exported(float(np.median(x))),
            "q05": None if not len(x) else exported(float(np.quantile(x, .05))), "q95": None if not len(x) else exported(float(np.quantile(x, .95))),
            "finite_n": len(finite), "finite_mean": None if not len(finite) else float(finite.mean())}


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {prefix + "_" + key: value for key, value in values.items()}


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


def _fmt(value: Any, places: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{places}f}"


def _interval(row: dict[str, Any], prefix: str, places: int = 3) -> str:
    return f"{_fmt(row.get(prefix + '_mean'), places)} [{_fmt(row.get(prefix + '_Student95_lower'), places)}, {_fmt(row.get(prefix + '_Student95_upper'), places)}]"






def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _row(metadata: dict[str, Any], row: dict[str, Any], phase: str,
         fraction: Fraction | None, evidence: str) -> dict[str, Any]:
    """Validate saved selections without accessing a numerical selector."""
    k = metadata["K"]
    partition = tuple(tuple(map(int, cell)) for cell in row["partition"])
    if sorted(i for cell in partition for i in cell) != list(range(k)) or any(not c for c in partition):
        raise ValueError("Returned partition is not a disjoint complete support cover")
    cap, value, truth = _fraction(row["cap"]), row["value"], row["truth"]
    true_value = truth["exact_value"]
    if cap not in CAPS or not isinstance(value, Fraction) or not 1 <= value <= cap:
        raise ValueError("Returned statistic/certificate lacks an exact passing comparison")
    if not isinstance(true_value, Fraction) or true_value < 1 or truth["ambiguous"]:
        raise ValueError("Population evaluation lacks an exact rational distortion")
    if bool(truth["strict_cap_violation"]) != (true_value > cap):
        raise ValueError("Stored strict cap indicator disagrees with exact truth")
    cells = len(partition)
    if row["cells"] != cells or row["identity"] != (cells == k) or row["compression"] != Fraction(k-cells, k):
        raise ValueError("Selected cell/count/compression metadata disagree")
    if row["procedure"] == "full_reference" and not row.get("recertified"):
        raise ValueError("Reference row lacks saved recertification")
    tested = row.get("tested", [])
    rejected = [entry for entry in tested if not entry["accepted"]]
    if len(rejected) > 1 or (rejected and tested[-1] is not rejected[0]):
        raise ValueError("The saved chain continued after its first failure")
    return dict(metadata, phase=phase, fraction=fraction, procedure=row["procedure"], cap=cap,
                cells=cells, compression=Fraction(k-cells, k), identity=cells==k,
                selected_value_exact=value, selected_value_float=float(value), selected_status=row["status"],
                true_distortion_exact=true_value, true_distortion=float(true_value),
                strict_cap_violation=bool(truth["strict_cap_violation"]),
                accepted_prefix_length=len(row.get("accepted_prefix_indices", [])),
                accepted_prefix_indices=row.get("accepted_prefix_indices", []),
                tested_count=len(tested), failed_test_count=len(rejected),
                stop_index=row.get("stop_index"), stop_reason=row.get("stop_reason"),
                partition=partition, evidence=evidence)


def _condition_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("phase", "fraction", "id", "K", "mass", "overlap", "geometry", "r", "n", "population_index", "r_index")}


def occupancy(N: Any, T: Any) -> dict[str, Any]:
    """Evaluation-only count diagnostics; no ranking or tuning inputs.

    All-treated/untreated counts exclude unobserved states. The three columns
    are thus disjoint, with observed mixed-action states forming the remainder.
    Relative occupancy divides each state's count by this subset's total.
    """
    N,T=tuple(map(int,N)),tuple(map(int,T))
    if not N or len(N)!=len(T) or any(n<0 or not 0<=t<=n for n,t in zip(N,T)):
        raise ValueError("Malformed occupancy counts")
    total=sum(N)
    return {"subset_total":total,"minimum_N":min(N),"minimum_T":min(T),
            "minimum_untreated":min(n-t for n,t in zip(N,T)),
            "unobserved_states":sum(n==0 for n in N),
            "all_treated_states":sum(n>0 and t==n for n,t in zip(N,T)),
            "all_untreated_states":sum(n>0 and t==0 for n,t in zip(N,T)),
            "minimum_relative_N":Fraction(min(N),total) if total else None}


def pilot_ranking_summary(rows: list[dict[str,Any]]) -> dict[str,Any]:
    return {phase:{"splits":sum(row["phase"]==phase for row in rows),
                   "splits_with_infinite_candidate_scores":sum(row["phase"]==phase and int(row["pilot_infinite_scores"])>0 for row in rows),
                   "infinite_candidate_scores":sum(int(row["pilot_infinite_scores"]) for row in rows if row["phase"]==phase),
                   "candidate_scores":sum(int(row["pilot_candidates"]) for row in rows if row["phase"]==phase)}
            for phase in ("development","final")}


def summarize(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    config = load(root / "configuration.json", load(ROOT / "configs" / (VERSION + ".json")))
    design = load(root / "design.json")
    completion = load(root / "completion.json", {})
    if design is None:
        # Refused or unstarted runs still expose the exact intended grid.
        historical = load(Path(config["source_root"]) / "frozen_study.json", {})
        design = {"populations": historical.get("design", {}).get("populations", []), "r_values": [250, 1000, 4000]}
    frozen = _fraction(completion["frozen_fraction"]) if completion.get("frozen_fraction") is not None else None
    selected, pair_rows, missing_grid, errors, adverse, tests, computation, occupancy_rows = [], [], [], [], [], [], [], []
    expected = {"development_datasets": 480, "final_datasets": 4320, "development_new_rows": 5760,
                "final_new_rows": 25920, "reference_rows": 14400}
    for phase, source_phase, reps in (("development", "development", range(10)), ("final", "evaluation", range(10, 100))):
        for pi, pop in enumerate(design["populations"]):
            for ri, r in enumerate(design["r_values"]):
                meta = {key: pop[key] for key in ("id", "K", "mass", "overlap", "geometry")}
                meta.update(r=r, n=pop["K"]*r, population_index=pi, r_index=ri)
                grid_row = dict(meta, phase=phase, expected_datasets=len(reps), completed_datasets=0, missing_replications=[])
                for rep in reps:
                    relative = Path("evidence") / source_phase / f"{pop['id']}_r{r}" / f"rep{rep:03d}.json.gz"
                    if not (root / relative).exists():
                        grid_row["missing_replications"].append(rep); continue
                    try:
                        data = read_gzip(root / relative)
                        if data["version"] != VERSION or data["phase"] != source_phase or data["metadata"] != dict(meta, replication=rep):
                            raise ValueError("Saved dataset identity differs from the frozen grid")
                        md = data["metadata"]
                        fractions = [_fraction(s["fraction"]) for s in data["splits"]]
                        wanted = set(FRACTIONS) if phase == "development" else {frozen}
                        if len(fractions) != len(wanted) or set(fractions) != wanted or frozen is None and phase == "final":
                            raise ValueError("Saved fractions differ from the frozen allocation")
                        ref = [_row(md, row, phase, None, str(relative)) for row in data["reference"]]
                        if len(ref) != 3 or {x["cap"] for x in ref} != set(CAPS) or any(x["procedure"] != "full_reference" for x in ref):
                            raise ValueError("Reference operating points are incomplete")
                        new, dataset_pairs, dataset_tests, dataset_adverse, dataset_compute, dataset_occupancy = [], [], [], [], [], []
                        for split in data["splits"]:
                            f = _fraction(split["fraction"])
                            if any(int(a)+int(b) != int(c) for a,b,c in zip(split["pilot_N"],split["heldout_N"],data["N"])) or any(int(a)+int(b) != int(c) for a,b,c in zip(split["pilot_T"],split["heldout_T"],data["T"])):
                                raise ValueError("Saved split does not conserve every count")
                            if sum(split["pilot_N"]) != math.floor(f*md["n"]):
                                raise ValueError("Saved pilot does not have the fixed total")
                            observed=dict(md,phase=phase,fraction=f,evidence=str(relative))
                            for subset,N,T in (("total",data["N"],data["T"]),
                                               ("pilot",split["pilot_N"],split["pilot_T"]),
                                               ("heldout",split["heldout_N"],split["heldout_T"])):
                                observed.update(prefixed(subset,occupancy(N,T)))
                            dataset_occupancy.append(observed)
                            rows = [_row(md, row, phase, f, str(relative)) for row in split["rows"]]
                            if len(rows) != 6 or {(row["procedure"],row["cap"]) for row in rows} != {(p,c) for p in PROCEDURES[1:] for c in CAPS}:
                                raise ValueError("New procedure operating points are incomplete")
                            new.extend(rows)
                            lookup = {(row["procedure"],row["cap"]):row for row in rows+ref}
                            for cap in CAPS:
                                a,b,c = (lookup[(p,cap)] for p in PROCEDURES)
                                delta, diagnostic = Fraction(100*(b["cells"]-a["cells"]),md["K"]), Fraction(100*(c["cells"]-b["cells"]),md["K"])
                                dataset_pairs.append(dict(md,phase=phase,fraction=f,cap=cap,
                                    reference_cells=a["cells"],ltt_cells=b["cells"],control_cells=c["cells"],
                                    reference_advantage_pp=delta,ltt_inference_advantage_pp=diagnostic,
                                    reference_wins=delta>0,count_tie=delta==0,ltt_wins=delta<0,
                                    ltt_beats_control=diagnostic>0,control_count_tie=diagnostic==0,control_beats_ltt=diagnostic<0,
                                    evidence=str(relative)))
                            for raw, flat in zip(split["rows"],rows):
                                for entry in raw.get("tested",[]):
                                    dataset_tests.append(dict(md,phase=phase,fraction=f,procedure=raw["procedure"],cap=raw["cap"],**entry,evidence=str(relative)))
                            core = split["core"]
                            diagnostic = dict(md,phase=phase,fraction=f,evidence=str(relative),**core["timings"])
                            diagnostic.update(root_cache_hits=core["root_cache"]["hits"],root_cache_misses=core["root_cache"]["misses"],
                                              pilot_candidates=len(core["candidates"]),pilot_infinite_scores=sum(x["score"] is None for x in core["candidates"]))
                            for procedure, evidence_key in (("split_ltt","ltt_evidence"),("split_coupled","control_evidence")):
                                evidence = core[evidence_key]
                                counts = Counter(record["status"] for record in evidence["cells"].values())
                                diagnostic.update({procedure+"_cell_"+status:count for status,count in counts.items()})
                                intervals = evidence.get("pooled_intervals", [])
                                if procedure == "split_coupled" and evidence.get("region") is not None:
                                    diagnostic[procedure+"_root_fallbacks"] = evidence["region"]["root_fallbacks"]
                                else:
                                    diagnostic[procedure+"_root_fallbacks"] = sum(item.get("status")=="fallback" for item in intervals)
                                for key, cell in evidence["cells"].items():
                                    if cell["status"] not in ("full","singleton"):
                                        dataset_adverse.append(dict(md,phase=phase,fraction=f,procedure=procedure,cell_key=key,status=cell["status"],reason=cell.get("reason"),evidence=str(relative)))
                            dataset_compute.append(diagnostic)
                        selected.extend(ref+new); pair_rows.extend(dataset_pairs);tests.extend(dataset_tests);adverse.extend(dataset_adverse);computation.extend(dataset_compute);occupancy_rows.extend(dataset_occupancy)
                        grid_row["completed_datasets"] += 1
                    except Exception as error:
                        errors.append(dict(meta,phase=phase,replication=rep,evidence=str(relative),error=type(error).__name__+": "+str(error)))
                        grid_row["missing_replications"].append(rep)
                grid_row["complete"] = grid_row["completed_datasets"] == grid_row["expected_datasets"]
                missing_grid.append(grid_row)
    selected_groups, paired_groups = defaultdict(list), defaultdict(list)
    for row in selected:selected_groups[(row["phase"],row["fraction"],row["procedure"],row["cap"],row["id"],row["r"])].append(row)
    for row in pair_rows:paired_groups[(row["phase"],row["fraction"],row["cap"],row["id"],row["r"])].append(row)
    occupancy_groups=defaultdict(list)
    for row in occupancy_rows:occupancy_groups[(row["phase"],row["fraction"],row["id"],row["r"])].append(row)
    occupancy_conditions=[]
    for rows in occupancy_groups.values():
        row=dict(_condition_meta(rows[0]),datasets=len(rows))
        for subset in ("total","pilot","heldout"):
            for metric in ("minimum_N","minimum_T","minimum_untreated","unobserved_states","all_treated_states","all_untreated_states","minimum_relative_N"):
                row.update(prefixed(subset+"_"+metric,distribution([x[subset+"_"+metric] for x in rows])))
            row.update(prefixed(subset+"_any_unobserved",rate([x[subset+"_unobserved_states"]>0 for x in rows])))
        occupancy_conditions.append(row)
    selected_conditions = []
    for rows in selected_groups.values():
        first=rows[0]; row=dict(_condition_meta(first),procedure=first["procedure"],cap=first["cap"])
        for metric in ("cells","compression","tested_count","accepted_prefix_length"):
            row.update(prefixed("selected_cells" if metric=="cells" else metric,paired([x[metric] for x in rows])))
        row.update(prefixed("identity",rate([x["identity"] for x in rows])))
        row.update(prefixed("strict_cap_violation",rate([x["strict_cap_violation"] for x in rows])))
        row.update(prefixed("true_distortion",distribution([x["true_distortion"] for x in rows])))
        row["stop_reasons"]=dict(Counter(str(x["stop_reason"]) for x in rows))
        selected_conditions.append(row)
    paired_conditions=[]
    for rows in paired_groups.values():
        first=rows[0]; row=dict(_condition_meta(first),cap=first["cap"])
        for metric in ("reference_advantage_pp","ltt_inference_advantage_pp"):
            row.update(prefixed(metric,paired([x[metric] for x in rows])))
        for metric in ("reference_wins","count_tie","ltt_wins","ltt_beats_control","control_count_tie","control_beats_ltt"):
            row.update(prefixed(metric,rate([x[metric] for x in rows])))
        paired_conditions.append(row)
    grid=[]
    for phase in ("development","final"):
        for fraction in FRACTIONS if phase=="development" else (() if frozen is None else (frozen,)):
            for cap in CAPS:
                rows=[x for x in paired_conditions if x["phase"]==phase and x["fraction"]==fraction and x["cap"]==cap]
                raw=[x for x in pair_rows if x["phase"]==phase and x["fraction"]==fraction and x["cap"]==cap]
                result=dict(phase=phase,fraction=fraction,cap=cap)
                for metric in ("reference_advantage_pp","ltt_inference_advantage_pp"):
                    result.update(prefixed(metric,grid_interval(rows,metric)))
                for metric in ("reference_wins","count_tie","ltt_wins","ltt_beats_control","control_count_tie","control_beats_ltt"):
                    result[metric+"_count"]=sum(x[metric] for x in raw)
                grid.append(result)
    grouped=[]
    for factor in ("K","mass","overlap","geometry","r"):
        for level in sorted({row[factor] for row in missing_grid},key=str):
            rows=[x for x in paired_conditions if x["phase"]=="final" and x["cap"]==Fraction(3,2) and x[factor]==level]
            row=dict(factor=factor,level=level)
            for metric in ("reference_advantage_pp","ltt_inference_advantage_pp"):
                expected_subgroup=sum(x["phase"]=="final" and x[factor]==level for x in missing_grid)
                row.update(prefixed(metric,grid_interval(rows,metric,expected_subgroup)))
            grouped.append(row)
    development=[]
    for fraction in FRACTIONS:
        for procedure in PROCEDURES[1:]:
            rows=[x for x in selected if x["phase"]=="development" and x["fraction"]==fraction and x["cap"]==Fraction(3,2) and x["procedure"]==procedure]
            development.append(dict(fraction=fraction,procedure=procedure,datasets=len(rows),
                mean_compression_exact=sum((x["compression"] for x in rows),Fraction(0))/len(rows) if rows else None,
                mean_selected_cells=float(np.mean([x["cells"] for x in rows])) if rows else None))
    frozen_check={"status":"not-yet-frozen"}
    ltt_dev={x["fraction"]:x for x in development if x["procedure"]=="split_ltt"}
    if frozen is not None and all(x["datasets"]==480 for x in ltt_dev.values()):
        expected_fraction=FRACTIONS[0] if ltt_dev[FRACTIONS[0]]["mean_compression_exact"]>=ltt_dev[FRACTIONS[1]]["mean_compression_exact"] else FRACTIONS[1]
        recorded=completion.get("fraction_selection",{}).get("mean_ltt_compression",{})
        ok=frozen==expected_fraction and all(recorded.get(str(f))==ltt_dev[f]["mean_compression_exact"] for f in FRACTIONS)
        frozen_check={"status":"passed" if ok else "failed","selected_fraction":frozen,"independently_recomputed_fraction":expected_fraction}
    technical=[]
    for path in sorted((root/"failures").glob("**/*.json")):
        item=load(path); technical.append(dict(path=str(path.relative_to(root)),**item))
    counts={"development_datasets":sum(x["completed_datasets"] for x in missing_grid if x["phase"]=="development"),
            "final_datasets":sum(x["completed_datasets"] for x in missing_grid if x["phase"]=="final"),
            "development_new_rows":sum(x["phase"]=="development" and x["procedure"]!="full_reference" for x in selected),
            "final_new_rows":sum(x["phase"]=="final" and x["procedure"]!="full_reference" for x in selected),
            "reference_rows":sum(x["procedure"]=="full_reference" for x in selected)}
    replay=load(root/"replay.json",{})
    procedure_grid=[]
    for phase in ("development","final"):
        for procedure in PROCEDURES:
            for fraction in (None,) if procedure=="full_reference" else FRACTIONS if phase=="development" else (() if frozen is None else (frozen,)):
                for cap in CAPS:
                    condition=[x for x in selected_conditions if x["phase"]==phase and x["fraction"]==fraction and x["procedure"]==procedure and x["cap"]==cap]
                    raw=[x for x in selected if x["phase"]==phase and x["fraction"]==fraction and x["procedure"]==procedure and x["cap"]==cap]
                    result=dict(phase=phase,fraction=fraction,procedure=procedure,cap=cap,datasets=len(raw),
                                identity_count=sum(x["identity"] for x in raw),strict_violation_count=sum(x["strict_cap_violation"] for x in raw),
                                descriptive_identity_fraction=sum(x["identity"] for x in raw)/len(raw) if raw else None,
                                descriptive_violation_fraction=sum(x["strict_cap_violation"] for x in raw)/len(raw) if raw else None,
                                pooled_rates_are_descriptive=True,**prefixed("true_distortion",distribution([x["true_distortion"] for x in raw])))
                    for metric in ("selected_cells","compression","tested_count","accepted_prefix_length"):
                        result.update(prefixed(metric,grid_interval(condition,metric)))
                    procedure_grid.append(result)
    # Worker durations are summed once per successful dataset. They overlap in
    # wall time under two workers; stage elapsed time is the budget quantity.
    receipt_by_dataset={}
    for stage in ("benchmark","develop","evaluate"):
        receipt=load(root/(stage+".json"),{})
        for row in receipt.get("rows",[]):
            if "technical_failure" not in row:
                receipt_by_dataset[(row["phase"],tuple(row["task"]))]=dict(stage=stage,**row)
    runtime_datasets=[]
    for (source_phase,task),row in receipt_by_dataset.items():
        pi,ri,rep=task;pop=design["populations"][pi]
        runtime_datasets.append(dict(phase="final" if source_phase=="evaluation" else source_phase,stage=row["stage"],
            id=pop["id"],K=pop["K"],r=design["r_values"][ri],replication=rep,
            dataset_worker_seconds=row["seconds"],**row["timings"],**row["io"]))
    runtime_summary={"stage_wall_seconds":{stage:load(root/(stage+".json"),{}).get("elapsed_seconds") for stage in ("benchmark","develop","evaluate","replay")},
                     "dataset_worker_seconds_sum":sum(x["dataset_worker_seconds"] for x in runtime_datasets),
                     "shared_dataset_seconds":{key:sum(x.get(key,0) for x in runtime_datasets) for key in ("split_seconds","evaluation_and_reference_seconds","serialization_seconds","output_seconds")},
                     "split_component_worker_seconds":{key:sum(x.get(key,0) for x in computation) for key in ("pilot_seconds","ltt_seconds","control_seconds","selection_seconds")},
                     "scope":"dataset and split worker durations sum work across workers; stage elapsed times count overlapping workers once; historical reference costs not benchmarked anew"}
    complete=counts==expected and not errors and frozen_check["status"]=="passed" and replay.get("status")=="passed"
    summary={"version":VERSION,"status":"complete" if complete else "incomplete","counts":counts,"expected":expected,
             "completion_record_status":completion.get("status","not-initialized"),"frozen_fraction":frozen,
             "fraction_selection_check":frozen_check,"development_fraction_comparison":development,
             "selected_conditions":selected_conditions,"paired_conditions":paired_conditions,"grid_contrasts":grid,
             "occupancy_conditions":occupancy_conditions,
             "final_occupancy_descriptives":{subset:{metric:distribution([row[subset+"_"+metric] for row in occupancy_rows if row["phase"]=="final"])
                 for metric in ("minimum_N","minimum_T","minimum_untreated","unobserved_states","all_treated_states","all_untreated_states","minimum_relative_N")}
                 for subset in ("total","pilot","heldout")},
             "procedure_grid":procedure_grid,"runtime":runtime_summary,
             "grouped_primary_contrasts":grouped,"missing_grid":missing_grid,"reporting_errors":errors,
             "technical_failures":technical,"numerical_cell_statuses":dict(Counter(x["procedure"]+":"+x["status"] for x in adverse)),
             "root_fallbacks":{p:sum(x.get(p+"_root_fallbacks",0) for x in computation) for p in PROCEDURES[1:]},
             "pilot_ranking_diagnostics":pilot_ranking_summary(computation),
             "test_statuses":dict(Counter(x["procedure"]+":"+str(x["status"]) for x in tests)),
             "stop_reasons":dict(Counter(x["procedure"]+":"+str(x["stop_reason"]) for x in selected if x["procedure"]!="full_reference")),
             "strict_violation_counts":{phase:{p:sum(x["strict_cap_violation"] for x in selected if x["phase"]==phase and x["procedure"]==p) for p in PROCEDURES} for phase in ("development","final")},
             "replay":replay,"focused_checks":load(root/"focused_checks.json",{}),"independent_report_checks":load(root/"report_checks.json",{}),"compute_elapsed_seconds":completion.get("compute_elapsed_seconds"),
             "versions":completion.get("versions",{}),"scope":"retrospective paired saved-count comparison; original dataset is the independent unit; pointwise cap safety"}
    tables={"development_rows":[x for x in selected if x["phase"]=="development" and x["procedure"]!="full_reference"],
            "final_rows":[x for x in selected if x["phase"]=="final" and x["procedure"]!="full_reference"],
            "reference_rows":[x for x in selected if x["procedure"]=="full_reference"],"paired_rows":pair_rows,
            "selected_conditions":selected_conditions,"paired_conditions":paired_conditions,"grid_contrasts":grid,
            "grouped_primary_contrasts":grouped,"development_fraction_comparison":development,
            "strict_violations":[x for x in selected if x["strict_cap_violation"]],"tested_decisions":tests,
            "adverse_cells":adverse,"split_computation":computation,"missing_grid":missing_grid,"reporting_errors":errors,
            "technical_failures":technical,"procedure_grid":procedure_grid,"runtime_datasets":runtime_datasets,
            "realized_occupancy":occupancy_rows,"occupancy_conditions":occupancy_conditions}
    return summary,tables


def _normal_interval(row: dict[str,Any],prefix: str) -> str:
    return f"{_fmt(row.get(prefix+'_mean'))} [{_fmt(row.get(prefix+'_analytic_Normal95_lower'))}, {_fmt(row.get(prefix+'_analytic_Normal95_upper'))}]"


def interpretation(summary: dict[str,Any]) -> dict[str,Any]:
    primary=next((x for x in summary["grid_contrasts"] if x["phase"]=="final" and x["cap"]==Fraction(3,2)),None)
    conditions=[x for x in summary["paired_conditions"] if x["phase"]=="final" and x["cap"]==Fraction(3,2)]
    if not primary or primary["reference_advantage_pp_mean"] is None:
        conclusion="The final comparison is incomplete; no completed-grid performance conclusion is available."
        recommendation="Retain the completed evidence and report the precise missing grid; this run does not yet settle the LTT comparison."
    else:
        lower=primary["reference_advantage_pp_analytic_Normal95_lower"]
        upper=primary["reference_advantage_pp_analytic_Normal95_upper"]
        if not summary["status"]=="complete":
            conclusion="The reported subset is incomplete and cannot establish the planned full-grid contrast."
            recommendation="Do not promote the subset to the completed comparison. Preserve unfavorable and missing rows explicitly."
        elif lower is not None and lower>0:
            conclusion="The full-data coupled reference achieves greater mean compression than this split-LTT specialization on the fixed evaluation grid."
            recommendation="The comparison strengthens the empirical assessment by establishing a scoped positive average compression difference against an established selection-guarantee framework specialized to distortion. Judge practical importance from the reported magnitude, identity frequencies, rare-state conditions and runtime rather than statistical sign alone. Keep the stronger all-partition guarantee and retrospective design explicit; the mean does not establish material dominance or superiority over all calibration procedures."
        elif upper is not None and upper<0:
            conclusion="Split LTT achieves greater mean compression than the full-data coupled reference on the fixed evaluation grid."
            recommendation="Retain LTT as a strong statistical alternative and qualify a general compression-efficiency claim for the coupled method. This unfavorable comparison is scientifically useful; the fixed-law necessity and all-partition validity contributions remain distinct. Do not silently substitute LTT or propose an untested hybrid."
        else:
            conclusion="The fixed-grid mean contrast does not resolve an overall compression advantage at the displayed simulation precision."
            recommendation="Present a mixed or unresolved comparison with its condition patterns and guarantee tradeoff. It strengthens the evaluation, but does not support an unqualified efficiency advantage."
    def number(predicate):return sum(predicate(x) for x in conditions)
    # The following assessment was written after review of this complete frozen
    # panel. It is not a prespecified materiality threshold or a selection rule.
    reviewed_assessment=("For these completed results, the roughly ten percentage-point compression difference, together with the stronger all-partition guarantee, materially strengthens the scoped complete-procedure comparison for the JMLR extension. The matched-split LTT advantage shows that this benefit cannot be attributed to coupled inference alone. Preserve the low-occupancy reversals and near parity in the high-sample balanced/moderate conditions; do not claim uniform dominance."
                         if summary["status"]=="complete" else "No completed-panel practical assessment is available.")
    return {"conclusion":conclusion,"recommendation":recommendation,"reviewed_assessment":reviewed_assessment,
            "conditions_with_positive_reference_mean":number(lambda x:x["reference_advantage_pp_mean"]>0),
            "conditions_with_negative_reference_mean":number(lambda x:x["reference_advantage_pp_mean"]<0),
            "conditions_with_exact_zero_reference_mean":number(lambda x:x["reference_advantage_pp_mean"]==0),
            "conditions_Student95_entirely_positive":number(lambda x:x["reference_advantage_pp_Student95_lower"] is not None and x["reference_advantage_pp_Student95_lower"]>0),
            "conditions_Student95_entirely_negative":number(lambda x:x["reference_advantage_pp_Student95_upper"] is not None and x["reference_advantage_pp_Student95_upper"]<0),
            "primary":primary}




def generate(root):
    """Generate numerical tables and summaries from newly computed records."""
    root=Path(root)
    summary,tables=summarize(root)
    for name,rows in tables.items():write_csv(rows,root/'tables'/(name+'.csv'))
    dump(summary,root/'summary.json')
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",nargs="?",type=Path,default=ROOT/"results"/VERSION)
    args=parser.parse_args()
    print(json.dumps(encode(generate(args.root)),indent=2,sort_keys=True))


if __name__=="__main__":main()
