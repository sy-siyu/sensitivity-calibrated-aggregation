"""Resumable reporting-capacity experiment.

Run freeze, preflight, run and report from the repository root.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.set_int_max_str_digits(0)
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction as F
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time
import traceback
import fcntl
from contextlib import contextmanager

import numpy as np
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import exact_order
from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator
from experiments.jmlr_outcome_inference_v1_core import (
    outcome_profiles, outcome_family, confidence_enclosure, population_interval,
    project_assignment, intersect_enclosures, ALPHA_CAL, ALPHA_RAW, ALPHA_Z)
from experiments.jmlr_outcome_inference_v1_lp import build_region, solve_enclosure
from experiments.jmlr_outcome_inference_v1_sampling import counts
from experiments.jmlr_release_capacity_v1_search import candidate_cells, capacity_search, admissible
from experiments.jmlr_release_capacity_v1_interface import export_release, release_bytes
from experiments.jmlr_release_capacity_v1_sampling import build_benchmark, sample_block

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT/'configs/jmlr_release_capacity_v1.json'
ROOT = PROJECT/'results/jmlr_release_capacity_v1'
G, LAMBDA = F(2), F(3)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return exact.decode(json.load(stream))
    return exact.decode(json.loads(path.read_text()))


def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    if path.suffix == '.gz':
        with gzip.open(temporary, 'wt', compresslevel=3) as stream:
            json.dump(exact.encode(data), stream, separators=(',', ':'))
    else:
        temporary.write_text(json.dumps(exact.encode(data), indent=2)+'\n')
    os.replace(temporary, path)


@contextmanager
def ownership():
    with (ROOT/'.owner.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def grid(K):
    return tuple(sorted({(K+d-1)//d for d in (16,8,4,2,1)}))


def equal_mass_partition(N, order, B):
    """Closest whole-state cumulative mass, with deterministic early ties."""
    n = sum(N); K = len(order); cuts = [0]
    cumulative = [0]
    for i in order:
        cumulative.append(cumulative[-1]+N[i])
    for j in range(1, B):
        cut = min(range(cuts[-1]+1, K-(B-j)+1),
                  key=lambda x: (abs(B*cumulative[x]-j*n), x))
        cuts.append(cut)
    cuts.append(K)
    return exact.canonical_partition(tuple(order[a:b]) for a,b in zip(cuts,cuts[1:]))




def _summary_cell(record):
    return {k:v for k,v in record.items() if k!='anchors'} | {'anchors_evaluated':len(record.get('anchors',()))}


def select_provider(N,T,tree_orders,ids,capacities,verifier):
    tick=time.perf_counter()
    region=exact.confidence_region(N,T,ALPHA_CAL,verifier=verifier)
    construction=time.perf_counter()-tick
    ids=tuple(tuple(x) if isinstance(x,list) else x for x in ids)
    order=exact_order(N,T,ids)
    baselines={B:equal_mass_partition(N,order,B) for B in capacities}
    cells=candidate_cells(order,tree_orders); records={}
    tick=time.perf_counter()
    # Complete all baseline certificates, including above-budget ones.
    for c in sorted({c for p in baselines.values() for c in p}):
        records[c]=cell_certificate(region,c)
    smallest=baselines[min(capacities)]
    complete=all(records[c]['status']=='full' for c in smallest)
    incumbent=max(records[c]['coupled'] for c in smallest) if complete else None
    for c in cells:
        if c not in records:
            records[c]=cell_certificate(region,c,max_cap=incumbent)
    certification=time.perf_counter()-tick
    tick=time.perf_counter()
    search=capacity_search(order,tree_orders,records,capacities)
    selection=time.perf_counter()-tick
    choices={}
    for B in capacities:
        selected=search['by_capacity'][B]
        if selected['status']=='finite':
            assert max(records[c]['coupled'] for c in selected['partition'])==selected['certificate']
            assert all(records[c]['status']=='full' for c in selected['partition'])
        p=baselines[B]; rec=[records[c] for c in p]
        state=('unresolved' if any(r['status']=='failure' for r in rec) else
               'infinite' if any(r['status']=='boundary' for r in rec) else 'finite')
        baseline={'status':state,'completed':state!='unresolved','partition':p,
                  'cells':len(p),'capacity':B,'family':'contiguous',
                  'certificate':max(r['coupled'] for r in rec) if state=='finite' else None}
        if selected['status']=='finite' and state=='finite':
            assert selected['certificate']<=baseline['certificate']
        choices[B]={'verified_capacity':selected,'equal_calibration_mass':baseline}
    finite=[search['by_capacity'][B]['certificate'] for B in capacities if search['by_capacity'][B]['status']=='finite']
    assert finite==sorted(finite,reverse=True)
    assert search['by_capacity'][len(N)]['certificate']==1
    return {'region':region,'order':order,'records':records,'choices':choices,'search':search,
            'pruning_incumbent':incumbent,'timings':{'confidence_construction_seconds':construction,
             'cell_certification_seconds':certification,'partition_search_seconds':selection}}


def isolated_analyst(artifacts):
    """Fresh interpreter with only aggregate JSON as data, in an empty cwd."""
    with tempfile.TemporaryDirectory(prefix='jmlr_aggregate_analyst_') as folder:
        path=Path(folder)/'release_bundle.json'
        path.write_text(json.dumps(artifacts,separators=(',',':')))
        env=dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONDONTWRITEBYTECODE='1')
        tick=time.perf_counter()
        proc=subprocess.run([sys.executable,'-m','experiments.jmlr_release_capacity_v1_interface',
              str(path),'--batch','--restrict-project-data'],cwd=folder,env=env,
              capture_output=True,text=True,check=True,timeout=300)
        wall=time.perf_counter()-tick
        import re
        def rationals(value):
            if isinstance(value,str) and re.fullmatch(r'-?\d+(?:/\d+)?',value):
                return F(value)
            if isinstance(value,dict):
                return {k:rationals(v) for k,v in value.items()}
            if isinstance(value,list):
                return [rationals(v) for v in value]
            return value
        result=rationals(json.loads(proc.stdout))
        return result,wall


def width(interval):
    return float(interval[1]-interval[0])


def miss(interval,reference):
    return interval[0]>reference[0] or interval[1]<reference[1]


def process_block(i,j,preflight=False):
    started=time.perf_counter(); protocol=load(ROOT/'frozen_protocol.json')
    cfg=protocol['configuration'];law=protocol['laws'][i];K=law['K']
    profiles=protocol['profiles'][i];capacities=protocol['capacity_grids'][i]
    base=ROOT/('preflight' if preflight else 'panel')
    path=base/'blocks'/f'{i}_{j:03d}.json.gz'
    if path.exists():
        raise FileExistsError('Never overwrite a completed block '+str(path))
    tick=time.perf_counter()
    obs,streams=sample_block(law,profiles,i,j,preflight=preflight,
        record_states=protocol['reference_record_states'] if i==2 else None,
        master_seed=cfg['sampler']['study_master_seed'],
        development_seed=cfg['sampler']['development_master_seed'])
    N,T=counts(obs['cal_S'],obs['cal_A'],K=K)
    sample_seconds=time.perf_counter()-tick
    verifier=RootVerifier(RootPolicy(**cfg['root_policy']))
    provider=select_provider(N,T,protocol['tree_orders'][i],law['immutable_ids'],capacities,verifier)
    region=provider['region'];records=provider['records'];evaluator=TruthEvaluator(law)
    tick=time.perf_counter()
    truth_values={c:evaluator.cell(c)['value'] for c in records}
    oracle=capacity_search(provider['order'],protocol['tree_orders'][i],truth_values,capacities)
    oracle_seconds=time.perf_counter()-tick
    pbox=tuple(zip(region['mass_lower'],region['mass_upper']))
    ebox=tuple(zip(region['propensity_lower'],region['propensity_upper']))
    artifacts=[];requests=[];rows=[];outcomes=[];projection_cache={}
    projection_seconds=0.;raw_seconds=0.
    for pi,(profile,r) in enumerate(profiles.items()):
        raw_pop=population_interval(law['p'],law['e'],r,G)
        for ratio in cfg['outcome_m_per_K']:
            on,ot,ov=counts(obs['out_S'][:ratio*K],obs['out_A'][:ratio*K],obs['out_Y'][pi,:ratio*K],K=K)
            tick=time.perf_counter()
            raw_r,raw_roots=outcome_family(ov,ot,ALPHA_RAW,verifier)
            raw=confidence_enclosure(pbox,ebox,raw_r,G)
            raw_seconds+=time.perf_counter()-tick
            outcome_id=len(outcomes)
            outcomes.append({'profile':profile,'m_ratio':ratio,'N':on,'T':ot,'V':ov,
                             'raw_roots':raw_roots,'raw_confidence':raw,'raw_population_interval':raw_pop})
            for B in capacities:
                for method,choice in provider['choices'][B].items():
                    admitted=bool(region['valid'] and admissible(choice))
                    row={'law_id':law['id'],'law_index':i,'replication':j,'K':K,'profile':profile,
                         'm_ratio':ratio,'m':ratio*K,'B':B,'method':method,
                         'status':choice['status'],'admitted':admitted,'abstained':not admitted,
                         'groups':choice['cells'],'certificate':choice['certificate'],
                         'transported_budget':None if choice['certificate'] is None else G*choice['certificate'],
                         'oracle_distortion':oracle['by_capacity'][B]['certificate'],
                         'raw_population_width':width(raw_pop),'raw_stratified_width':width(raw['interval']),
                         'raw_stratified_miss':miss(raw['interval'],raw_pop),
                         'evaluation_fallback':choice['status']!='finite','outcome_id':outcome_id}
                    if choice['status']!='finite':
                        row.update(global_confidence_width=1.,cellwise_confidence_width=1.,
                                   admitted_and_global_miss=False,admitted_and_cellwise_miss=False,
                                   pooled_raw_width=None,pooled_raw_miss=None,
                                   aggregate_true_population_width=None,aggregate_certified_population_width=None,
                                   true_distortion=(evaluator.partition(choice['partition'])['value'] if choice['partition'] else None),
                                   payload_bytes=0,analyst_seconds=0.,operational_payload_bytes=0,
                                   uncomputed_diagnostic_reason=choice['status'])
                        rows.append(row);continue
                    p=choice['partition'];tick=time.perf_counter()
                    # Cache provider projections across outcomes for the same grouping.
                    if p not in projection_cache:
                        projection_cache[p]=project_assignment(region,p)
                    zp,ze,projection=projection_cache[p]
                    artifact=export_release(region,p,records,on,ot,ov,verifier)
                    projection_seconds+=time.perf_counter()-tick
                    tz=tuple(sum(ot[s] for s in c) for c in p)
                    vz=tuple(sum(ov[s] for s in c) for c in p)
                    zr,zroots=outcome_family(vz,tz,ALPHA_Z,verifier)
                    U=tuple(records[c]['coupled'] for c in p)
                    reference={'z_global':confidence_enclosure(zp,ze,zr,G*choice['certificate']),
                               'z_cellwise':confidence_enclosure(zp,ze,zr,tuple(G*u for u in U))}
                    tcells=[evaluator.cell(c) for c in p]
                    pp=tuple(c['mass'] for c in tcells);ep=tuple(c['propensity'] for c in tcells)
                    rz=tuple(sum(law['p'][s]*law['e'][s]*r[s] for s in c)/sum(law['p'][s]*law['e'][s] for s in c) for c in p)
                    distortion=max(c['value'] for c in tcells)
                    pop_true=population_interval(pp,ep,rz,G*distortion)
                    pop_cert=population_interval(pp,ep,rz,G*choice['certificate'])
                    row.update(true_distortion=distortion,
                         aggregate_true_population_width=width(pop_true),aggregate_certified_population_width=width(pop_cert),
                         zero_treated_groups=sum(t==0 for t in tz),artifact_index=len(artifacts),
                         payload_bytes=len(release_bytes(artifact)),
                         operational_payload_bytes=len(release_bytes(artifact)) if admitted else 0)
                    artifacts.append(artifact)
                    requests.append({'row_index':len(rows),'partition':p,'reference':reference,
                                     'zr':zr,'zroots':zroots,'population_true':pop_true,
                                     'population_certificate':pop_cert,'raw_population':raw_pop,
                                     'raw_interval':raw['interval'],'raw_r':raw_r})
                    rows.append(row)
    analyst_result,analyst_wall=isolated_analyst(artifacts)
    # Batch CLI supplies a receipt; all numerical results remain exact.
    analyses=analyst_result['analyses'] if isinstance(analyst_result,dict) else analyst_result
    receipt={k:v for k,v in analyst_result.items() if k!='analyses'} if isinstance(analyst_result,dict) else {}
    if not receipt and analyses:
        receipt=analyses[0].get('data_access_receipt',{})
    assert len(analyses)==len(requests)
    for analysis in analyses:
        access=analysis['data_access_receipt']
        assert access['code_data_access_restricted']
        assert access['noncode_project_access_attempts']==[]
        assert access['data_input_open_count']==1
    lp_cache={}; lp_seconds=0.; lp_records=[]
    for request,analysis in zip(requests,analyses):
        row=rows[request['row_index']]
        for name in ('z_global','z_cellwise'):
            assert tuple(analysis['methods'][name]['interval'])==request['reference'][name]['interval']
        assert analysis['z_outcome_bounds']==[list(x) for x in request['zr']] or tuple(map(tuple,analysis['z_outcome_bounds']))==request['zr']
        a,b=(analysis['methods'][name]['interval'] for name in ('z_global','z_cellwise'))
        key=(row['outcome_id'],request['partition'])
        if key not in lp_cache:
            tick=time.perf_counter()
            lp=solve_enclosure(build_region(pbox,ebox,request['raw_r'],request['partition'],request['zr'],G),
                              time_limit=cfg['execution']['LP_seconds_per_endpoint'])
            pool=intersect_enclosures(tuple(lp['interval']),tuple(request['raw_interval']),tuple(b))
            lp_seconds+=time.perf_counter()-tick
            lp_cache[key]=(lp,pool,len(lp_records));lp_records.append({'outcome_id':row['outcome_id'],'partition':request['partition'],'LP':lp,'pooled':pool})
        lp,pool,lp_id=lp_cache[key]
        row.update(global_confidence_width=width(a),cellwise_confidence_width=width(b),
                   pooled_raw_width=width(pool['interval']),global_miss=miss(a,request['raw_population']),
                   cellwise_miss=miss(b,request['raw_population']),pooled_raw_miss=miss(pool['interval'],request['raw_population']),
                   admitted_and_global_miss=bool(row['admitted'] and miss(a,request['raw_population'])),
                   admitted_and_cellwise_miss=bool(row['admitted'] and miss(b,request['raw_population'])),
                   analyst_seconds=analysis['compute_seconds'],analyst_root_fallbacks=analysis['root_fallbacks'],
                   analyst_peak_rss_bytes=analysis.get('peak_rss_bytes'),
                   analyst_cache_hits=analysis.get('root_cache_hits'),analyst_cache_misses=analysis.get('root_cache_misses'),
                   global_fallback=analysis['methods']['z_global']['fallback'],
                   cellwise_fallback=analysis['methods']['z_cellwise']['fallback'],
                   LP_status=lp['status'],LP_fallback=lp['fallback'],LP_id=lp_id,
                   empty_pooled_intersection=pool['empty_intersection'],aggregate_only_equal=True)
        event=all(l<=x<=u for (l,u),x in zip(analysis['z_outcome_bounds'],
            [sum(law['p'][s]*law['e'][s]*profiles[row['profile']][s] for s in c)/sum(law['p'][s]*law['e'][s] for s in c) for c in request['partition']]))
        reg_event=all(lo<=p<=hi and el<=e<=eu for lo,hi,el,eu,p,e in zip(region['mass_lower'],region['mass_upper'],region['propensity_lower'],region['propensity_upper'],law['p'],law['e']))
        if event and reg_event:
            assert not row['global_miss'] and not row['cellwise_miss']
    selected_cells={c for by_method in provider['choices'].values() for v in by_method.values()
                    if v['status']=='finite' for c in v['partition']}
    cells=[records[c] if c in selected_cells else _summary_cell(records[c]) for c in records]
    calibration_covered=all(lo<=p<=hi and el<=e<=eu for lo,hi,el,eu,p,e in zip(region['mass_lower'],region['mass_upper'],region['propensity_lower'],region['propensity_upper'],law['p'],law['e']))
    record={'version':cfg['version'],'law_index':i,'law_id':law['id'],'replication':j,
            'preflight':preflight,'streams':streams,'N':N,'T':T,'region':region,
            'order':provider['order'],'cells':cells,'choices':provider['choices'],
            'oracle':oracle,'pruning_incumbent':provider['pruning_incumbent'],
            'outcomes':outcomes,'rows':rows,'LP_records':lp_records,
            'analyst_results':analyses,'analyst_receipt':receipt,
            'diagnostics':{'calibration_covered':calibration_covered,'unobserved_states':sum(n==0 for n in N),
                          'cell_status_counts':{s:sum(r['status']==s for r in records.values()) for s in ('full','boundary','rejected','failure')},
                          'root_fallbacks':region['root_fallbacks'],
                          'root_cache_hits':verifier.cache_hits,'root_cache_misses':verifier.cache_misses},
            'timings':provider['timings']|{'sampling_seconds':sample_seconds,'oracle_seconds':oracle_seconds,
             'projection_export_seconds':projection_seconds,'raw_outcome_seconds':raw_seconds,
             'analyst_fresh_process_wall_seconds':analyst_wall,'analyst_inference_seconds':sum(a['compute_seconds'] for a in analyses),
             'pooled_raw_seconds':lp_seconds,'compute_seconds_before_serialization':time.perf_counter()-started},
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    tick=time.perf_counter()
    dump(base/'analyst_inputs'/f'{i}_{j:03d}.json.gz',artifacts)
    dump(path,record)
    serialization=time.perf_counter()-tick
    return {'law_index':i,'replication':j,'rows':len(rows),'admitted_rows':sum(r['admitted'] for r in rows),
            'seconds':time.perf_counter()-started,'serialization_seconds':serialization,
            'evidence_bytes':path.stat().st_size,'peak_rss_bytes':record['peak_rss_bytes'],
            'timings':record['timings'],'cell_status_counts':record['diagnostics']['cell_status_counts']}






