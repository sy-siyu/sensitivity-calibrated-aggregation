"""Matched saved-data split ablation; no observations or LTT tests are drawn.

One launch owner writes a new root. Completed dataset records are never replaced.
Source observations, proposal rankings, inference evidence and manuscripts are
read in place. Only C/D/E are computed; A/B/F selections are imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
import fcntl
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=PROJECT/'configs/jmlr_split_ablation_v1.json'
PROTOCOL=None


def now():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    value=Path(value)
    return value if value.is_absolute() else PROJECT/value


def read(filename):
    from experiments.jmlr_verified_pipeline_v1_baseline import read as load
    return load(filename)


def dump(filename,value):
    from experiments.jmlr_verified_adaptive_v1_math import encode
    filename=Path(filename);filename.parent.mkdir(parents=True,exist_ok=True)
    temporary=filename.with_name(filename.name+'.tmp')
    if filename.suffix=='.gz':
        with gzip.open(temporary,'wt') as stream:
            json.dump(encode(value),stream,separators=(',',':'),allow_nan=False)
    else:
        temporary.write_text(json.dumps(encode(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(filename)


@contextmanager
def ownership(root):
    with (root/'execution.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)






@lru_cache(maxsize=1)
def laws(filename):
    return {law['id']:law for law in read(filename)['large']}


def _compact(record):
    return {key:record.get(key) for key in ('cell','status','coupled','independent','rejection_bound',
                                           'reason','mixture_calls','closure_calls','anchors')}


def _metrics(engine):
    evidence=engine.evidence()
    return {**evidence['cache'],'backend_seconds':evidence['backend_seconds'],
            'queried_cells':len(evidence['cells'])}


def process_dataset(condition,replication,protocol):
    from experiments.jmlr_split_ablation_v1_inputs import load_dataset
    from experiments.jmlr_split_ablation_v1_core import (
        CellEngine,all_proposals,complete_pilot_search,stopping_diagnostic,validate_proposals)
    from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator
    from experiments.jmlr_verified_adaptive_v1_math import confidence_region,validate_partition
    from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier,RootPolicy
    started=time.perf_counter();timings={}
    name=condition['condition_id'];filename=f'rep{replication:03d}.json.gz'
    phase='development' if replication<10 else 'evaluation'
    split_path=Path(protocol['split_root'])/'evidence'/phase/name/filename
    full_path=Path(protocol['full_root'])/'evidence/large'/name/filename
    data=load_dataset(split_path,full_path,expected_policy=protocol['configuration']['root_policy'])
    assert data['metadata']['replication']==replication
    assert data['metadata']['id']==condition['id'] and data['metadata']['r']==condition['r']
    caps=tuple(map(F,protocol['configuration']['caps']))
    timings['load_inputs_seconds']=time.perf_counter()-started
    tick=time.perf_counter()
    region=data['heldout_region']
    constructed=region is None
    if constructed:
        region=confidence_region(data['heldout_N'],data['heldout_T'],F(1,20),
             verifier=RootVerifier(RootPolicy(**protocol['configuration']['root_policy'])))
    held=CellEngine(region,data['heldout_cells'])
    full=CellEngine(data['full_region'],data['full_cells'])
    validate_proposals(data['candidates'],data['pilot_order'],data['tree_orders'])
    timings['heldout_missing_confidence_seconds']=time.perf_counter()-tick
    tick=time.perf_counter();C=all_proposals(data['candidates'],held,caps)
    timings['C_seconds']=time.perf_counter()-tick
    timings['C_cell_backend_seconds']=held.evidence()['backend_seconds']
    tick=time.perf_counter();D=all_proposals(data['candidates'],full,caps)
    timings['D_seconds']=time.perf_counter()-tick
    timings['D_cell_backend_seconds']=full.evidence()['backend_seconds']
    tick=time.perf_counter();E=complete_pilot_search(data['pilot_order'],data['tree_orders'],full,caps)
    timings['E_seconds']=time.perf_counter()-tick
    timings['E_cell_backend_seconds']=full.evidence()['backend_seconds']-timings['D_cell_backend_seconds']
    # Imported decisions are authoritative; this checks their compatible reuse,
    # without rerunning LTT or reconstructing pilot rankings.
    by_index={r['candidate_index']:r for r in C['evaluations']}
    prefix_checks=0
    for cap,saved in data['B_by_cap'].items():
        for test in saved['tested']:
            decision=by_index[test['candidate_index']]['decisions'][cap]
            if bool(decision['accepted'])!=test['accepted']:
                raise AssertionError('C changed a preserved B prefix decision')
            if test['value'] is not None and decision['certificate']!=test['value']:
                raise AssertionError('C changed a preserved completed B certificate')
            prefix_checks+=1
    rows=[dict(row) for row in data['references']]
    for procedure,result in (('C',C),('D',D),('E',E)):
        for cap,choice in result['by_cap'].items():
            added=dict(choice,procedure=procedure,cap=cap,value_kind='coupled_certificate')
            if procedure in ('C','D'):
                index=choice['candidate_index']
                added.update(selected_pilot_score=data['candidates'][index]['score'],
                    selected_pilot_score_infinite=data['candidates'][index]['score'] is None,
                    infinite_pilot_proposals_passed=sum(e['pilot_score'] is None and e['decisions'][cap]['accepted']
                                                       for e in result['evaluations']))
            rows.append(added)
    assert len(rows)==18 and {(r['procedure'],r['cap']) for r in rows}=={(p,c) for p in 'ABCDEF' for c in caps}
    tick=time.perf_counter()
    law=laws(protocol['exact_law_path'])[data['metadata']['id']]
    evaluator=TruthEvaluator(law)
    for row in rows:
        pi=validate_partition(row['partition'],law['K'])
        assert row['cells']==len(pi) and isinstance(row['value'],F) and F(1)<=row['value']<=row['cap']
        truth=evaluator.partition(pi)['value']
        row.update(**data['metadata'],condition_id=name,partition=pi,
                   compression_pp=F(100*(law['K']-len(pi)),law['K']),
                   identity=len(pi)==law['K'],value_status='finite',true_distortion=truth,
                   strict_cap_violation=truth>row['cap'])
        row.setdefault('deterministic_tie_complete',True)
        row.setdefault('identity_fallback',False)
    timings['truth_evaluation_seconds']=time.perf_counter()-tick
    stopping=[];checks=0
    for cap in caps:
        method={r['procedure']:r for r in rows if r['cap']==cap}
        if method['C']['count_optimal'] and method['C']['cells']>method['B']['cells']:
            raise AssertionError('Completed C has more cells than B')
        if method['E']['count_optimal'] and method['E']['cells']>method['D']['cells']:
            raise AssertionError('Completed E has more cells than D')
        Q={p:r['compression_pp'] for p,r in method.items()}
        if Q['F']-Q['A']!=sum((Q[x]-Q[y] for x,y in [('B','A'),('C','B'),('D','C'),('E','D'),('F','E')]),F(0)):
            raise AssertionError('Paired sequence accounting identity failed')
        checks+=1
        diagnostic=stopping_diagnostic(data['B_by_cap'][cap],data['candidates'],C['evaluations'],cap)
        stop=diagnostic['saved_stop_index']
        failed=None if stop is None else data['candidates'][stop]
        stopping.append(dict(diagnostic,cap=cap,stop_index=diagnostic['saved_stop_index'],
             stop_reason=diagnostic['saved_stop_reason'],
             first_failed_partition=None if failed is None else failed['partition'],
             first_failed_cells=None if failed is None else failed['cells'],
             first_failed_pilot_score=None if failed is None else failed['score'],
             first_failed_pilot_score_infinite=None if failed is None else failed['score'] is None,
             later_proposal_passes=diagnostic['later_passing_count']>0,
             later_passing_fewer_cells=diagnostic['later_passing_coarser_count']>0,
             count_gain=diagnostic['count_improvement'],compression_gain_pp=diagnostic['compression_improvement_pp']))
    def new_evidence(engine,old):
        old_cells={tuple(r['cell']) for r in old}
        records=engine.evidence()
        if isinstance(records,dict):records=records.get('cells',records)
        if isinstance(records,dict):records=records.values()
        return [_compact(r) for r in records if tuple(r['cell']) not in old_cells]
    return {'version':protocol['version'],'metadata':data['metadata'],'condition_id':name,'phase':phase,
        'source_paths':{'split':str(split_path),'full':str(full_path)},
        'selection_information':'Saved counts, confidence regions, proposals and orders only; population law enters postselection truth evaluation',
        'rows':rows,'stopping':stopping,
        'proposal_status_counts':{p:dict(Counter(e['status'] for e in result['evaluations'])) for p,result in [('C',C),('D',D)]},
        'proposal_evaluations':{'C':C['evaluations'],'D':D['evaluations']},
        'pilot_infinite_proposals':sum(c['score'] is None for c in data['candidates']),
        'pilot_proposals':len(data['candidates']),'pilot_order':data['pilot_order'],
        'full_order':data['full_order'],'orders_equal':tuple(data['pilot_order'])==tuple(data['full_order']),
        'heldout_region_constructed':constructed,'root_fallbacks':{'heldout':region['root_fallbacks'],'full':data['full_region']['root_fallbacks']},
        'additional_cells':{'heldout':new_evidence(held,data['heldout_cells']),'full':new_evidence(full,data['full_cells'])},
        'cache':{'heldout':_metrics(held),'full':_metrics(full)},
        'timings':timings,'timing_scope':protocol['configuration']['execution']['timing_scope'],
        'checks':{'preserved_B_prefix_decisions':prefix_checks,'accounting_identities':checks,
                  'all_saved_proposals_in_pilot_family':True},
        'compute_seconds':time.perf_counter()-started}


def task_path(root,condition,replication,preflight=False):
    return Path(root)/('preflight' if preflight else 'panel')/condition['condition_id']/f'rep{replication:03d}.json.gz'


def worker(root,condition,replication,preflight=False):
    output=task_path(root,condition,replication,preflight)
    if output.exists():raise FileExistsError('Refusing to replace a completed dataset')
    start=time.perf_counter();record=process_dataset(condition,replication,PROTOCOL)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(development=preflight,peak_worker_rss_bytes=peak if sys.platform=='darwin' else peak*1024)
    tick=time.perf_counter();dump(output,record)
    return {'condition_id':condition['condition_id'],'replication':replication,'rows':len(record['rows']),
            'seconds':time.perf_counter()-start,'serialization_seconds':time.perf_counter()-tick,
            'bytes':output.stat().st_size,'peak_worker_rss_bytes':record['peak_worker_rss_bytes']}






