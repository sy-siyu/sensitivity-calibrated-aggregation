"""Import-only adapter for the completed split-LTT comparison.

No calibration, candidate search, held-out testing or tuning is performed.
Original inference evidence is referenced in place. Population distortion is
an evaluation-only exact calculation under the pipeline's declared finite law.
"""
from __future__ import annotations

from fractions import Fraction as Q
import gzip
import json
import numbers
from pathlib import Path
from typing import Any, Callable

VERSION = "jmlr_verified_pipeline_v1"
SOURCE_VERSION = "jmlr_statistical_baselines_v1"
CAPS = (Q(5,4), Q(3,2), Q(2))
FROZEN_FRACTION = "1/5"


def decode(value: Any) -> Any:
    if isinstance(value,dict):
        if set(value)=={"numerator","denominator"}:return Q(int(value["numerator"]),int(value["denominator"]))
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(v) for v in value]
    return value


def read(path: str | Path) -> dict[str,Any]:
    path=Path(path)
    if path.suffix==".gz":
        with gzip.open(path,"rt") as stream:return decode(json.load(stream))
    return decode(json.loads(path.read_text()))


def same(left: Any,right: Any) -> bool:
    if isinstance(left,dict) and isinstance(right,dict):
        return set(left)==set(right) and all(same(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right


def partition(value: Any,K: int) -> tuple[tuple[int,...],...]:
    value=tuple(tuple(cell) for cell in value)
    if any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for cell in value for i in cell):
        raise ValueError("Partition state indices must be integers without coercion")
    result=tuple(sorted(tuple(sorted(int(i) for i in cell)) for cell in value))
    if any(not cell for cell in result) or sorted(i for cell in result for i in cell)!=list(range(K)):
        raise ValueError("Imported partition must cover the complete support exactly once")
    return result


def true_distortion(pi: Any,p: Any,e: Any) -> Q:
    p,e=tuple(p),tuple(e)
    if len(p)!=len(e) or any(not isinstance(v,Q) for v in p+e):
        raise TypeError("Evaluation requires declared rational law coordinates")
    if sum(p)!=1 or any(v<=0 for v in p) or any(not 0<v<1 for v in e):
        raise ValueError("Evaluation law is not a positive exact probability vector")
    result=Q(1)
    for cell in partition(pi,len(p)):
        if len(cell)==1:continue
        q=sum(p[i]*e[i] for i in cell)/sum(p[i] for i in cell)
        oq=q/(1-q)
        for i in cell:
            oi=e[i]/(1-e[i]);result=max(result,oi/oq,oq/oi)
    return result


def validate_source(root: str | Path) -> dict[str,Any]:
    """Check the original completed allocation once, before importing rows."""
    root=Path(root);completion=read(root/"completion.json");config=read(root/"configuration.json")
    if completion["version"]!=SOURCE_VERSION or completion["status"]!="complete":
        raise ValueError("The preserved split comparison is not complete")
    if completion["frozen_fraction"]!=FROZEN_FRACTION or completion["fraction_selection"]["evaluation_outputs_at_freeze"]!=0:
        raise ValueError("Import must preserve the pre-evaluation global fraction 1/5")
    expected={"development_datasets":480,"evaluation_datasets":4320,"development_new_rows":5760,
              "evaluation_new_rows":25920,"reference_rows":14400}
    if any(completion[k]!=v for k,v in expected.items()):raise ValueError("Historical comparison allocation differs")
    if Q(config["alpha"])!=Q(1,20) or tuple(map(Q,config["caps"]))!=CAPS:
        raise ValueError("Historical constants differ")
    if config["split_seed_recipe"]!=[20261012,901,"population_index","r_index","replication"]:
        raise ValueError("Historical split randomization differs")
    return {"status":"passed","source_root":str(root),"frozen_fraction":FROZEN_FRACTION,
            "allocation":expected,"fraction_selection":completion["fraction_selection"]}


def evidence_relative(metadata: dict[str,Any]) -> Path:
    phase="development" if int(metadata["replication"])<10 else "evaluation"
    return Path("evidence")/phase/f"{metadata['id']}_r{metadata['r']}"/f"rep{int(metadata['replication']):03d}.json.gz"


def _truth(row: dict[str,Any],p: Any,e: Any,cache: dict[Any,Q]) -> dict[str,Any]:
    pi=partition(row["partition"],len(p));cap=row["cap"]
    if pi not in cache:cache[pi]=true_distortion(pi,p,e)
    return {"exact_value":cache[pi],"strict_cap_violation":cache[pi]>cap,"ambiguous":False}


def _import_row(row: dict[str,Any],p: Any,e: Any,cache: dict[Any,Q]) -> dict[str,Any]:
    result=dict(row)
    result["partition"]=partition(row["partition"],len(p))
    result["old_truth"]=row["truth"]
    result["truth"]=_truth(result,p,e,cache)
    result["truth_value_changed"]=result["truth"]["exact_value"]!=row["truth"]["exact_value"]
    result["truth_predicate_changed"]=result["truth"]["strict_cap_violation"]!=row["truth"]["strict_cap_violation"]
    return result


def adapt_dataset(source_root: str | Path,metadata: dict[str,Any],p: Any,e: Any,
                  new_reference_rows: list[dict[str,Any]],*,law_id: str,
                  N: Any=None,T: Any=None,immutable_ids: Any=None) -> dict[str,Any]:
    """Link one historical baseline dataset to a new full-data selection.

    New references must provide exact ``cap``, ``partition``, ``cells`` and
    ``value`` fields. Their selection is performed by the launch owner before
    this adapter is called. All source splits, proposals and tests stay fixed.
    """
    relative=evidence_relative(metadata);source=read(Path(source_root)/relative)
    if not isinstance(law_id,str) or not law_id:raise ValueError("Declared pipeline law identifier is required")
    phase="development" if int(metadata["replication"])<10 else "evaluation"
    if source["version"]!=SOURCE_VERSION or source["phase"]!=phase or source["metadata"]!=metadata:
        raise ValueError("Historical baseline dataset identity differs")
    if N is not None and tuple(N)!=tuple(source["N"]):raise ValueError("Historical baseline state counts differ")
    if T is not None and tuple(T)!=tuple(source["T"]):raise ValueError("Historical baseline treated counts differ")
    if immutable_ids is not None and tuple(immutable_ids)!=tuple(source["immutable_ids"]):raise ValueError("Immutable state IDs differ")
    expected_fractions=("1/5","1/2") if phase=="development" else (FROZEN_FRACTION,)
    if tuple(s["fraction"] for s in source["splits"])!=expected_fractions:
        raise ValueError("Imported split allocation differs")
    cache={};splits=[]
    for index,split in enumerate(source["splits"]):
        if any(not isinstance(r["cap"],Q) for r in split["rows"]):raise ValueError("Imported caps must remain exact fractions")
        if len(split["rows"])!=6 or {(r["procedure"],r["cap"]) for r in split["rows"]}!={(m,c) for m in ("split_ltt","split_coupled") for c in CAPS}:
            raise ValueError("Historical split procedure/cap grid differs")
        splits.append({"fraction":split["fraction"],"source_split_index":index,
                       "rows":[_import_row(r,p,e,cache) for r in split["rows"]]})
    old={r["cap"]:r for r in source["reference"]}
    if len(old)!=3 or set(old)!=set(CAPS) or len(new_reference_rows)!=3 or {r["cap"] for r in new_reference_rows}!=set(CAPS):
        raise ValueError("Full reference operating points differ")
    references=[]
    for raw in new_reference_rows:
        row=dict(raw,procedure="full_reference")
        if "cells" not in row:row["cells"]=row["count"]
        if "value" not in row:row["value"]=row["certificate"]
        if isinstance(row["value"],dict):row["value"]=row["value"]["value"]
        pi=partition(row["partition"],len(p));cap=row["cap"]
        if row["cells"]!=len(pi) or not isinstance(row["value"],Q) or not 1<=row["value"]<=cap:
            raise ValueError("New full reference lacks an exact passing full certificate")
        row.update(partition=pi,identity=len(pi)==len(p),compression=Q(len(p)-len(pi),len(p)),
                   truth=_truth(row,p,e,cache),selection_provenance="new complete verified unknown-mass combined coupled search")
        previous=old[cap];old_pi=partition(previous["partition"],len(p))
        row.update(old_partition=old_pi,old_cells=previous["cells"],old_value=previous["value"],old_truth=previous["truth"],
                   old_partition_truth_under_new_law=_truth(previous,p,e,cache),
                   count_changed=row["cells"]!=previous["cells"],partition_changed=pi!=old_pi,
                   certificate_changed=row["value"]!=previous["value"],
                   truth_value_changed=row["truth"]["exact_value"]!=previous["truth"]["exact_value"],
                   truth_predicate_changed=row["truth"]["strict_cap_violation"]!=previous["truth"]["strict_cap_violation"])
        row["old_partition_truth_value_changed"]=row["old_partition_truth_under_new_law"]["exact_value"]!=previous["truth"]["exact_value"]
        row["old_partition_truth_predicate_changed"]=row["old_partition_truth_under_new_law"]["strict_cap_violation"]!=previous["truth"]["strict_cap_violation"]
        references.append(row)
    return {"version":VERSION,"source_version":SOURCE_VERSION,"metadata":metadata,"phase":phase,
            "source_root":str(Path(source_root)),"source_relative_path":str(relative),"law_id":law_id,
            "frozen_fraction":FROZEN_FRACTION,"splits":splits,"reference":references,
            "source_inference_recomputed":False,"tuning_repeated":False}


def iter_root_records(source: dict[str,Any]):
    """Every imported root record; the caller owns rigorous deduplication."""
    for split in source["splits"]:
        core=split["core"]
        yield from core["ltt_evidence"]["pooled_intervals"]
        region=core["control_evidence"]["region"]
        if region is not None:
            yield from region["roots"]["propensity"]
            yield from region["roots"]["mass"]


def replay_dataset(adapted: dict[str,Any],p: Any,e: Any,tree_orders: Any,*,
                   expected_metadata: dict[str,Any],N: Any,T: Any,immutable_ids: Any,
                   root_checker: Callable[[dict[str,Any]],Any],
                   new_reference_rows: list[dict[str,Any]],law_id: str,
                   expected_source_root: str | Path | None=None) -> dict[str,Any]:
    """Validate imported links and actual tests without repeating inference.

    ``root_checker`` must validate brackets with the rigorous backend and may
    cache by all record mathematical/policy inputs. Historical replay_split
    checks the split's integer/rational arithmetic and actual return path.
    """
    from experiments import jmlr_statistical_baselines_v1_core as core
    from experiments.jmlr_verified_pipeline_v1_replay import region_replay,cells_replay
    if adapted["version"]!=VERSION or adapted["frozen_fraction"]!=FROZEN_FRACTION:
        raise AssertionError("Adapter version/frozen fraction changed")
    if adapted["law_id"]!=law_id:raise AssertionError("Imported baseline exact law identifier changed")
    if expected_source_root is not None and Path(adapted["source_root"])!=Path(expected_source_root):
        raise AssertionError("Imported baseline source root changed")
    rebuilt=adapt_dataset(adapted["source_root"],expected_metadata,p,e,new_reference_rows,
                          law_id=law_id,N=N,T=T,immutable_ids=immutable_ids)
    if not same(adapted,rebuilt):raise AssertionError("Imported baseline rows or new reference linkage changed")
    source=read(Path(adapted["source_root"])/adapted["source_relative_path"])
    splits=core.nested_splits(N,T,expected_metadata["population_index"],expected_metadata["r_index"],
                             expected_metadata["replication"],immutable_ids=immutable_ids)
    selected=0;control_cells=0
    for split in source["splits"]:
        expected=splits[split["fraction"]]
        for key in ("pilot_N","pilot_T","heldout_N","heldout_T","pilot_size","heldout_size", "seed", "generator", "joint_order", "immutable_ids"):
            if not same(split[key],expected[key]):raise AssertionError("Imported actual split counts/metadata changed: "+key)
        core.replay_split(split["core"],split["pilot_N"],split["pilot_T"],split["heldout_N"],split["heldout_T"],
                          tree_orders,immutable_ids=immutable_ids,caps=CAPS)
        control=split["core"]["control_evidence"]
        if control["region"] is not None:
            region=region_replay(control["region"],{"K":len(p),"p":p,"e":e},root_checker)
            records=list(control["cells"].values())
            cells_replay(records,region,early_rejection=False)
            control_cells+=len(records)
        for raw,row in zip(split["core"]["selections"],split["rows"]):
            if any(not same(row.get(k),v) for k,v in raw.items()):raise AssertionError("Imported row differs from the actual split selector return")
            if row["fraction"]!=split["fraction"] or row["procedure"]!=("split_ltt" if raw["method"]=="ltt" else "split_coupled"):
                raise AssertionError("Imported method/procedure/fraction label changed")
            if not isinstance(row["value"],Q) or not 1<=row["value"]<=row["cap"]:
                raise AssertionError("Imported selection lacks exact passing arithmetic")
        selected+=len(split["rows"])
    roots=0
    for record in iter_root_records(source):
        answer=root_checker(record)
        if answer is False:raise AssertionError("Rigorous imported root replay failed")
        roots+=1
    return {"status":"passed","new_rows":selected,"reference_rows":3,"root_records":roots,
            "control_cells_rationally_replayed":control_cells,
            "source_inference_recomputed":False,"tuning_repeated":False}
