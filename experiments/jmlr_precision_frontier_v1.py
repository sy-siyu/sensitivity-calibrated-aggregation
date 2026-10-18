"""Resumable saved-block cap operating curves; this module draws no observations.

All historical data and implementations remain read-only. Selection receives
calibration-only arguments. Population laws are passed only after that call,
into the explicitly privileged diagnostic path. One launch owner writes blocks.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager
from datetime import datetime,timezone
from fractions import Fraction as F
import fcntl,gzip,json,multiprocessing as mp,numbers,platform,resource,shutil,sys,time,traceback
from pathlib import Path
sys.set_int_max_str_digits(0)
from experiments import jmlr_verified_adaptive_v1_math as exact

PROJECT=Path(__file__).resolve().parents[1]
CONFIG=PROJECT/'configs/jmlr_precision_frontier_v1.json'
PROTOCOL=None


def now():return datetime.now(timezone.utc).isoformat()
def resolved(value):
    p=Path(value);return p if p.is_absolute() else PROJECT/p

def load(filename):
    p=Path(filename)
    if p.suffix=='.gz':
        with gzip.open(p,'rt') as f:return exact.decode(json.load(f))
    return exact.decode(json.loads(p.read_text()))

def dump(filename,value):
    p=Path(filename);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
    if p.suffix=='.gz':
        with gzip.open(temp,'wt',compresslevel=3) as f:json.dump(exact.encode(value),f,separators=(',',':'),allow_nan=False)
    else:temp.write_text(json.dumps(exact.encode(value),indent=2,allow_nan=False)+'\n')
    temp.replace(p)

@contextmanager
def ownership(root):
    with (root/'.owner.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)




def integers(values,K,name):
    values=tuple(values)
    if len(values)!=K or any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or v<0 for v in values):
        raise ValueError(name+' must retain nonnegative integers on every declared state')
    return tuple(map(int,values))


def source_block(protocol,i,j,preflight=False):
    cfg=protocol['configuration'];source=Path(protocol['source_dir'])
    p=load(protocol['source_protocol']);law=p['laws'][i];K=law['K']
    path=source/('preflight' if preflight else 'panel')/'blocks'/f'{i}_{j:03d}.json.gz';b=load(path)
    if b['law_index']!=i or b['replication']!=j or b['law_id']!=law['id'] or bool(b['preflight'])!=preflight:
        raise ValueError('Source block identity/phase mismatch')
    N,T=integers(b['N'],K,'Calibration N'),integers(b['T'],K,'Calibration T')
    if sum(N)!=4000*K or any(t>n for n,t in zip(N,T)):raise ValueError('Saved calibration counts differ from design')
    if tuple(b['region']['N'])!=N or tuple(b['region']['T'])!=T:raise ValueError('Saved region count linkage differs')
    if b['region']['root_policy']!=cfg['root_policy'] or b['region']['alpha']!=F(1,40):raise ValueError('Source confidence policy differs')
    expected={(profile,ratio) for profile in p['profiles'][i] for ratio in (16,256)};found=set();shared={}
    for outcome in b['outcomes']:
        key=(outcome['profile'],outcome['m_ratio'])
        if key in found or key not in expected:raise ValueError('Source outcome grid differs')
        found.add(key);vectors=[integers(outcome[k],K,'Outcome '+k) for k in ('N','T','V')]
        n,t,v=vectors
        if sum(n)!=outcome['m_ratio']*K or any(not a<=bb<=c for a,bb,c in zip(v,t,n)):
            raise ValueError('Outcome cohort count constraints differ')
        if outcome['m_ratio'] in shared and shared[outcome['m_ratio']]!=(n,t):raise ValueError('Shared cohort differs between profiles')
        shared[outcome['m_ratio']]=(n,t)
    if found!=expected:raise ValueError('Source outcome conditions missing')
    return b,p,path


def process_block(protocol,i,j,preflight=False,integration_check=False):
    from experiments.jmlr_precision_frontier_v1_selection import select_thresholds,check_capacity_admission
    from experiments.jmlr_precision_frontier_v1_analysis import InferenceCache,operational_inference,five_stage_diagnostics
    from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier,RootPolicy
    from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator
    from experiments.jmlr_outcome_inference_v1_core import confidence_enclosure,outcome_family
    started=time.perf_counter();b,p,source=source_block(protocol,i,j,preflight);cfg=protocol['configuration']
    timings={'load_validate_seconds':time.perf_counter()-started};law_meta=protocol['laws'][i]
    # Deliberately pass no law probabilities, outcomes, target widths or capacity into selection.
    selected=select_thresholds(N=b['N'],T=b['T'],region=b['region'],order=b['order'],cells=b['cells'],
         tree_orders=p['tree_orders'][i],ids=p['laws'][i]['immutable_ids'],caps=tuple(map(F,cfg['caps'])),expected_root_policy=cfg['root_policy'])
    region=selected['region'];records=selected['records'];B=cfg['capacity_limits'][i]
    timings.update({'selection_'+k:v for k,v in selected['timings'].items()})
    # Population probabilities are used only after assignment-based choices exist.
    law=p['laws'][i];profiles=p['profiles'][i];truth=TruthEvaluator(law)
    cal_covered=bool(region['valid'] and all(lo<=pi<=hi and a<=ei<=z for lo,hi,a,z,pi,ei in
         zip(region['mass_lower'],region['mass_upper'],region['propensity_lower'],region['propensity_upper'],law['p'],law['e'])))
    selection_rows=[];by_cap={};diag_time=0.;analysis_time=0.
    for cap,choice in selected['by_cap'].items():
        complete=bool(choice['count_optimal'] and choice['deterministic_tie_complete'] and choice['computation_status']=='finite')
        pi=choice['partition'];U=choice['certificate'];d=truth.partition(pi)['value']
        admitted=bool(complete and len(pi)<=B and U<=cap)
        equivalence=None
        if cap==F(3,2):
            check=check_capacity_admission(choice,b['choices'][str(B)]['verified_capacity'],B)
            equivalence=True if check['checked'] and check['status']=='agrees' else None
        row={'law_index':i,'law_id':law['id'],'replication':j,'K':law['K'],'n':sum(b['N']),'cap':cap,'capacity_limit':B,
          'groups':len(pi),'partition':pi,'compression_pp':100*F(law['K']-len(pi),law['K']),
          'certificate':U,'true_distortion':d,'transported_budget':2*U,'budget_ceiling':2*cap,
          'budget_slack':2*(cap-U),'selection_complete':complete,'count_optimal':choice['count_optimal'],
          'deterministic_tie_complete':choice['deterministic_tie_complete'],'computation_status':choice['computation_status'],
          'identity':len(pi)==law['K'],'identity_fallback':choice['identity_fallback'],
          'admitted':admitted,'abstained':not admitted,'calibration_covered':cal_covered,
          'strict_distortion_cap_violation':d>cap,'source_admission_match':equivalence,
          'source_block':str(source),'selected_cell_bounds':tuple(records[c]['coupled'] for c in pi)}
        selection_rows.append(row);by_cap[cap]=row
    assert [x['groups'] for x in selection_rows if x['selection_complete']]==sorted([x['groups'] for x in selection_rows if x['selection_complete']],reverse=True)
    inference_cache=InferenceCache();verifier=RootVerifier(RootPolicy(**cfg['root_policy']))
    lp_saved={(a['outcome_id'],tuple(map(tuple,a['partition']))):a['LP'] for a in b['LP_records']}
    analysis_rows=[];new_inferences=[];inference_indices={};diagnostics_cache={};additional_costs=Counter();reuse_counts=Counter()
    for outcome_id,out in enumerate(b['outcomes']):
        for cap,choice in selected['by_cap'].items():
            part=choice['partition'];key=(outcome_id,part,tuple(records[c]['coupled'] for c in part))
            tick=time.perf_counter()
            inference=operational_inference(region,part,records,out['N'],out['T'],out['V'],verifier,
                  cache=inference_cache,raw_saved=out,raw_saved_policy=b['region']['root_policy'],
                  lp_saved=lp_saved.get((outcome_id,part)),lp_time_limit=cfg['execution']['LP_seconds_per_endpoint'])
            analysis_time+=time.perf_counter()-tick
            for name,value in inference['timings'].items():additional_costs[name]+=value
            reuse_counts.update({k:int(v) for k,v in inference['reuse'].items()})
            if key not in diagnostics_cache:
                tick=time.perf_counter();diagnostics_cache[key]=five_stage_diagnostics(law,profiles[out['profile']],region,part,records,inference)
                diag_time+=time.perf_counter()-tick
            diag=diagnostics_cache[key];base=by_cap[cap]
            row={k:v for k,v in base.items() if k not in ('partition','selected_cell_bounds','source_block')}
            row.update(profile=out['profile'],m_ratio=out['m_ratio'],m=out['m_ratio']*law['K'],outcome_id=outcome_id,
                 raw_population_exceeds_target=diag['global']['widths']['W0']>F(1,5),
                 raw_outcomes_covered=diag['events']['raw_outcomes_covered'],aggregate_outcomes_covered=diag['events']['aggregate_outcomes_covered'],
                 shared_coverage_event=diag['events']['shared_event'],projected_assignment_covered=diag['events']['projected_assignment_covered'],
                 LP_status=inference['LP']['status'],LP_fallback=inference['LP']['fallback'],
                 empty_pooled_intersection=inference['methods']['raw_pooled']['empty_intersection'],
                 raw_root_fallbacks=inference['raw_root_fallbacks'],aggregate_root_fallbacks=inference['aggregate_root_fallbacks'],
                 zero_treated_groups=sum(g['counts']['treated']==0 for g in inference['artifact']['groups']),
                 analyst_seconds=inference['timings']['analyst_seconds'],payload_bytes=inference['payload_bytes'],
                 operational_cache_hit=inference['reuse']['complete_cache_hit'],aggregate_only_equal=None)
            for method,result in inference['methods'].items():
                lo,hi=tuple(result['interval']);width=hi-lo
                numerical=bool(not result['fallback'] and (method!='raw_pooled' or not inference['LP']['fallback']))
                row.update({method+'_lower':lo,method+'_upper':hi,method+'_width':width,
                     method+'_miss':diag['entire_raw_interval_misses'][method],method+'_fallback':result['fallback'],
                     method+'_numerical_complete':numerical,method+'_target':width<=F(1,5),
                     method+'_admitted_target':bool(base['admitted'] and width<=F(1,5)) if method.startswith('z_') else None,
                     method+'_admitted_miss':bool(base['admitted'] and diag['entire_raw_interval_misses'][method]) if method.startswith('z_') else None})
            for prefix in ('global','cellwise'):
                stage=diag[prefix]
                for label,(lo,hi) in stage['intervals'].items():row.update({prefix+'_'+label+'_lower':lo,prefix+'_'+label+'_upper':hi,prefix+'_'+label+'_width':hi-lo})
                for name,val in stage['components'].items():row[prefix+'_'+('population_aggregation' if name=='population_aggregation_transport' else name)]=val
                row[prefix+'_total_excess']=stage['total_excess'];row[prefix+'_accounting_exact']=stage['accounting_exact']
                row[prefix+'_nesting']=all(stage['nesting'].values())
                for name,value in stage['nesting'].items():row[prefix+'_'+name]=value
            if integration_check and key not in inference_indices:
                # A bounded representative check of the integration formulas,
                # using exactly the analyst's r boxes and exported assignment boxes.
                a=inference['artifact'];pb=tuple(tuple(map(F,g['mass_bounds'])) for g in a['groups']);eb=tuple(tuple(map(F,g['propensity_bounds'])) for g in a['groups'])
                U=tuple(F(g['distortion_bound']) for g in a['groups']);rb=inference['z_outcome_bounds']
                reference=(confidence_enclosure(pb,eb,rb,2*max(U))['interval'],confidence_enclosure(pb,eb,rb,tuple(2*u for u in U))['interval'])
                assert reference==(inference['methods']['z_global']['interval'],inference['methods']['z_cellwise']['interval'])
                row['aggregate_only_equal']=True
            if key not in inference_indices:
                inference_indices[key]=len(new_inferences)
                # Raw roots/counts stay authoritative in the source block. Save
                # only changed aggregate/LP computation and a source reference.
                saved={'outcome_id':outcome_id,'partition':part,'cell_bounds':inference['cell_certificates'],
                       'artifact':inference['artifact'],'analyst':inference['analyst'],
                       'methods':inference['methods'],'diagnostics':diag,'reuse':inference['reuse'],
                       'timings':inference['timings']}
                if inference['reuse']['lp_saved_reused']:saved['LP_source_reference']={'block':str(source),'outcome_id':outcome_id,'partition':part}
                else:saved['LP']=inference['LP']
                new_inferences.append(saved)
            row['inference_id']=inference_indices[key];analysis_rows.append(row)
    expected=5*len(profiles)*2
    if len(selection_rows)!=5 or len(analysis_rows)!=expected:raise AssertionError('A committed condition is missing')
    timings.update(operational_seconds=analysis_time,diagnostics_seconds=diag_time,**{'incremental_'+k:v for k,v in additional_costs.items()})
    timings['worker_seconds_before_serialization']=time.perf_counter()-started
    return {'version':cfg['version'],'law_index':i,'law_id':law['id'],'replication':j,'preflight':preflight,
       'source_block':str(source),'source_protocol':protocol['source_protocol'],
       'selection_rows':selection_rows,'analysis_rows':analysis_rows,'selection_choices':selected['by_cap'],
       'additional_cell_records':selected['additional_cell_records'],'new_inferences':new_inferences,
       'calibration_diagnostics':{'calibration_covered':cal_covered,'source_root_fallbacks':b['region']['root_fallbacks'],'unobserved_states':sum(n==0 for n in b['N'])},
       'cache':{'selection':selected['cache'],'inference':inference_cache.statistics(),'reuse':dict(reuse_counts)},
       'timings':timings,'numerical_policy':cfg['root_policy'],
       'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)}










