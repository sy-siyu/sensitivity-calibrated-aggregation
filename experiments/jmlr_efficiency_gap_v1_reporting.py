"""Complete assignment-only efficiency-gap numerical tables.

Every committed condition and denominator is retained.  Historical threshold
selection, the separate oracle-count minimax diagnostic, capacity-constrained
release selection, and full-support recovery are distinct reporting objects.
The reporting code never changes certificates, selections, or observations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction as F
from functools import lru_cache
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

from experiments.jmlr_efficiency_gap_v1 import dump, load, now
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_release_capacity_v1_reporting import _statistics


PROCEDURES = ('original','allocation_control','pooled_alternative')
HIST_KEYS = ('condition_id','law_id','K','mass','overlap','geometry','r','n','cap','mass_mode','procedure')
RELEASE_KEYS = ('law_index','law_id','K','n','B','cap','mass_mode','procedure')
RECOVERY_KEYS = ('law_id','K','n','mass_mode','procedure')
KEYS = {'historical':HIST_KEYS,'release':RELEASE_KEYS,'recovery':RECOVERY_KEYS}
CONTINUOUS = (
    'count_gap','normalized_count_gap','normalized_compressible_gap','selected_count','compression',
    'oracle_count','threshold_selected_true_distortion','threshold_selected_certificate',
    'threshold_log_inflation','threshold_log_mass_inflation','threshold_log_propensity_inflation',
    'same_partition_true_mass_slice_certificate','true_mass_slice_count','mass_slice_count_gap',
    'd_star','oracle_margin_log','oracle_capacity_minimax_certificate','minimax_log_inflation',
    'certificate','true_distortion','certificate_inflation_log','certificate_minus_truth',
    'decomposition_log_total','decomposition_log_mass','decomposition_log_propensity','decomposition_log_selection',
    'realized_epsilon','realized_rho','realized_b','actual_over_realized_b',
    'analytical_b_n','analytical_epsilon_n','analytical_rho_n','analytical_r_e',
    'manuscript_reference_b_n','manuscript_reference_rho_n','manuscript_reference_r_e',
    'minimum_state_count','minimum_treated_or_control_count','expected_minimum_state_count',
    'population_min_mass','population_overlap','unobserved_states','states_without_treated',
    'states_without_control','empirical_minimum_propensity','g_e','g_p',
    'raw_root_fallbacks','pooled_root_fallbacks','root_fallbacks',
)
BINARY = (
    'count_recovered','positive_count_gap','negative_count_gap','certified',
    'admitted','abstained','unsafe_certification','threshold_cap_violation',
    'admitted_and_cap_violation','certificate_below_selected_truth',
    'calibration_failure','pooled_event_failure','procedure_event_failure',
    'used_certificate_event_failure','negative_certificate_inflation',
    'truth_mass_slice_nested','truth_mass_slice_not_nested','numerical_root_fallback',
    'analytical_bound_finite','realized_radius_finite','decomposition_identity_exact',
    'oracle_capacity_completed_and_at_most_cap',
)
METRICS = CONTINUOUS+BINARY
PAIR_METRICS = {
    'historical':('count_gap','normalized_count_gap','normalized_compressible_gap','selected_count','compression',
                  'threshold_selected_certificate','threshold_selected_true_distortion',
                  'threshold_log_inflation','threshold_log_mass_inflation','threshold_log_propensity_inflation',
                  'mass_slice_count_gap','count_recovered','threshold_cap_violation',
                  'certificate_below_selected_truth','calibration_failure','procedure_event_failure',
                  'oracle_capacity_minimax_certificate','minimax_log_inflation'),
    'release':('oracle_capacity_minimax_certificate','selected_count','compression','certificate_inflation_log',
               'minimax_log_inflation','decomposition_log_mass','decomposition_log_propensity',
               'decomposition_log_selection','admitted','admitted_and_cap_violation',
               'certificate_below_selected_truth','procedure_event_failure'),
    'recovery':('certificate','certificate_inflation_log','certified','unsafe_certification',
                'calibration_failure','used_certificate_event_failure','numerical_root_fallback'),
}


def _number(value):
    if value is None or not isinstance(value,(int,float,F)):
        return None
    number=float(value)
    return number if math.isfinite(number) else None


@lru_cache(maxsize=4096)
def _cp95(successes,n):
    """Outward verified conventional two-sided95% Clopper-Pearson endpoints."""
    if n<=0:
        return None,None,'not_computed'
    record=RootVerifier(RootPolicy(max_trials=max(n,1000))).interval(successes,n,F(1,20))
    return float(record['lower']),float(record['upper']),record['status']


def _stats(values,binary=False):
    values=list(values)
    answer=_statistics(values,probability=binary)
    finite=[_number(value) for value in values if _number(value) is not None]
    for name in ('minimum','q05','q25','median','q75','q95','maximum'):
        answer[name]=None
    if finite:
        quantiles=np.quantile(np.asarray(finite,dtype=float),[0,.05,.25,.5,.75,.95,1])
        answer.update(zip(('minimum','q05','q25','median','q75','q95','maximum'),map(float,quantiles)))
    answer.update(n_negative=sum(value<0 for value in finite),n_zero=sum(value==0 for value in finite),
                  n_positive=sum(value>0 for value in finite))
    if binary:
        if any(value not in (0.,1.) for value in finite):
            raise ValueError('Binomial interval requires binary block indicators')
        successes=sum(int(value) for value in finite)
        lo,hi,status=_cp95(successes,len(finite))
        answer.update(successes=successes,cp95_lower=lo,cp95_upper=hi,cp95_status=status)
    return answer


def _flatten_scalars(value,prefix=''):
    answer={}
    for key,item in value.items():
        name=prefix+str(key)
        if isinstance(item,dict):
            answer.update(_flatten_scalars(item,name+'_'))
        elif item is None or isinstance(item,(bool,str,int,float,F)):
            answer[name]=item
    return answer


def _log_ratio(numerator,denominator):
    ratio=F(numerator)/F(denominator)
    return math.log(ratio.numerator)-math.log(ratio.denominator)


def _enrich(row,block,kind,evidence_file,law=None):
    """Flatten reporting fields without reinterpreting historical selections."""
    answer=_flatten_scalars(row)
    answer.update(study_component=kind,evidence_file=evidence_file,
                  mass_mode=row.get('mass_mode','unknown_mass'))
    if kind=='historical':
        answer['condition_id']=Path(row['source']).parent.name
    raw_region=block.get('new_region')
    if kind=='recovery':
        law=block['laws'][row['law_id']]
        point=next(p for p in block['count_paths']['paths'][row['law_id']] if p['n']==row['n'])
        N,T=point['N'],point['T']
        answer.update(population_min_mass=min(law['p']),
                      population_overlap=min(min(e,1-e) for e in law['e']),
                      minimum_state_count=min(N),
                      minimum_treated_or_control_count=min(min(t,n-t) for n,t in zip(N,T)),
                      calibration_failure=not row['raw_region_covered'],
                      used_certificate_event_failure=(not row['certificate_event_covered']
                          if 'certificate_event_covered' in row else None),
                      root_fallbacks=row['root_fallbacks'],
                      optimization_status=row['status'],selected_count=1,
                      admitted=bool(row['certified']),abstained=not row['certified'],
                      certificate=row['certificate'],true_distortion=row['true_distortion'])
        answer['pooled_root_fallbacks']=int(bool(row.get('pooled_root_fallback'))
            and row['procedure']=='pooled_alternative')
        radius_prefix='radii_'
        answer.update(realized_epsilon=row.get(radius_prefix+'epsilon_realized'),
                      realized_rho=row.get(radius_prefix+'rho_realized'),
                      realized_b=row.get(radius_prefix+'b_realized'),
                      realized_status=row.get(radius_prefix+'status'))
        analytical={key[len('analytical_'):]:value for key,value in row.items() if key.startswith('analytical_')}
        answer['g_e']=row.get('analytical_g_e');answer['g_p']=row.get('analytical_g_p')
    else:
        N,T=raw_region['N'],raw_region['T']
        diagnostic_key='original' if row['procedure']=='original' else 'allocation_control'
        analytical=block['diagnostics'][diagnostic_key]
        raw_fallbacks=analytical['root_fallbacks']
        pooled_fallbacks=(sum(record['status']=='fallback' for record in
            block.get('pooled_interval_evidence',())) if row['procedure']=='pooled_alternative' else 0)
        answer.update(_flatten_scalars(analytical,'analytical_'))
        radii=row['realized_radii']
        answer.update(realized_epsilon=radii.get('epsilon_realized'),
                      realized_rho=radii.get('rho_realized'),realized_b=radii.get('b_realized'),
                      realized_status=radii['status'],
                      calibration_failure=not row['calibration_covered'],
                      pooled_event_failure=not row['pooled_event_covered'],
                      procedure_event_failure=not row['procedure_event_covered'],
                      raw_root_fallbacks=raw_fallbacks,pooled_root_fallbacks=pooled_fallbacks,
                      root_fallbacks=raw_fallbacks+pooled_fallbacks,g_e=analytical['g_e'],g_p=analytical['g_p'])
        answer['optimization_status']=row.get('selection_status') if kind=='historical' else row['oracle_capacity_status']
        answer['oracle_capacity_completed_and_at_most_cap']=bool(
            row['oracle_capacity_status']=='finite' and row['oracle_capacity_minimax_certificate']<=row['cap'])
        if kind=='historical':
            completed=row['selection_status']=='finite'
            answer.update(count_recovered=bool(completed and row['minimum_count_recovered']),
                          positive_count_gap=bool(completed and row['count_gap']>0),
                          negative_count_gap=bool(completed and row['count_gap']<0),
                          certificate_inflation_log=row['threshold_log_inflation'],
                          certificate=row['threshold_selected_certificate'],
                          true_distortion=row['threshold_selected_true_distortion'],
                          admitted=bool(completed and row['threshold_selected_certificate']<=row['cap']),
                          abstained=not(completed and row['threshold_selected_certificate']<=row['cap']))
            nested=row['truth_mass_slice_nested']
            answer['mass_slice_count_gap']=(row['selected_count']-row['true_mass_slice_count']
                if completed and row['true_mass_slice_selection_status']=='finite' else None)
            answer.update(truth_mass_slice_nested=nested,truth_mass_slice_not_nested=not nested)
            if row.get('threshold_decomposition_identity_exact'):
                if row['threshold_mass_inflation_ratio']*row['threshold_propensity_inflation_ratio']!=row['threshold_inflation_ratio']:
                    raise AssertionError('Actual threshold multiplicative decomposition does not match')
            if not completed:
                # An upper candidate remains reviewable in the dataset table,
                # but it is not a completed minimum-count optimum.
                for key in ('selected_count','count_gap','normalized_count_gap','normalized_compressible_gap',
                            'threshold_selected_certificate','threshold_selected_true_distortion',
                            'threshold_log_inflation','threshold_log_mass_inflation','threshold_log_propensity_inflation',
                            'certificate_inflation_log','certificate','true_distortion'):
                    answer['unresolved_candidate_'+key]=answer.get(key)
                    answer[key]=None
        else:
            answer.update(selected_count=row['oracle_capacity_groups'],
                          certificate=row['oracle_capacity_minimax_certificate'],
                          true_distortion=row.get('selected_true_distortion'),
                          abstained=not row['admitted'])
            if (law is not None and row['oracle_capacity_status'] in ('finite','infinite')
                    and row.get('oracle_capacity_partition') is not None):
                from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator
                selected_truth=TruthEvaluator(law).partition(row['oracle_capacity_partition'])['value']
                if answer['true_distortion'] is not None and answer['true_distortion']!=selected_truth:
                    raise AssertionError('Returned release partition population distortion changed')
                answer['true_distortion']=answer['selected_true_distortion']=selected_truth
            if answer['certificate'] is not None and answer['true_distortion'] is not None:
                answer['certificate_inflation_log']=_log_ratio(answer['certificate'],answer['true_distortion'])
            nested=row.get('truth_mass_feasible')
            answer.update(truth_mass_slice_nested=nested,
                          truth_mass_slice_not_nested=None if nested is None else not nested)
            for key,value in row.get('decomposition_logs',{}).items():
                answer['decomposition_log_'+key]=value
            if row.get('decomposition_identity_exact'):
                ratios=row['decomposition_ratios']
                if ratios['mass']*ratios['propensity']*ratios['selection']!=ratios['total']:
                    raise AssertionError('Capacity decomposition multiplicative identity does not match')
            answer['decomposition_identity_exact']=row.get('decomposition_identity_exact')
            answer['admitted_and_cap_violation']=bool(row['admitted'] and
                answer['true_distortion'] is not None and answer['true_distortion']>row['cap'])
        answer['certified']=bool(answer['admitted'])
    if kind in ('historical','release'):
        # Compression describes the returned partition, separately from
        # admission. An unresolved optimum supplies no completed estimate.
        count=answer.get('selected_count')
        answer['compression']=(1-F(count,answer['K']) if count is not None
            and answer['optimization_status'] in ('finite','infinite') else None)
    if answer.get('certificate_inflation_log') is None and answer.get('certificate') is not None:
        answer['certificate_inflation_log']=_log_ratio(answer['certificate'],answer['true_distortion'])
    actual=_number(answer.get('certificate_inflation_log'))
    realized=_number(answer.get('realized_b'))
    answer.update(expected_minimum_state_count=F(answer['n'])*answer['population_min_mass'],
                  unobserved_states=sum(n==0 for n in N),
                  states_without_treated=sum(t==0 for t in T),
                  states_without_control=sum(t==n for n,t in zip(N,T)),
                  empirical_minimum_propensity=min((F(t,n) for n,t in zip(N,T) if n),default=None),
                  numerical_root_fallback=bool(answer.get('root_fallbacks',0)),
                  analytical_bound_finite=analytical.get('status')=='finite',
                  realized_radius_finite=answer.get('realized_status')=='finite',
                  actual_over_realized_b=(actual/realized if actual is not None and realized is not None and realized>0 else None),
                  negative_certificate_inflation=(actual<0 if actual is not None else None),
                  certificate_below_selected_truth=(answer['certificate']<answer['true_distortion']
                      if answer.get('certificate') is not None and answer.get('true_distortion') is not None else None))
    return answer


def _groups(rows,keys):
    answer=defaultdict(list)
    for row in rows:
        answer[tuple(row[key] for key in keys)].append(row)
    return answer


def _condition_summary(rows,expected):
    rows=list(rows)
    if len({r['replication'] for r in rows})!=len(rows):
        raise AssertionError('A condition has duplicate independent blocks')
    result={'n_blocks':len(rows),'n_expected':expected,'n_missing_blocks':expected-len(rows)}
    result.update(n_negative_certificate_inflation=sum((_number(r.get('certificate_inflation_log')) or 0)<0 for r in rows),
                  n_negative_oracle_minimax_inflation=sum((_number(r.get('minimax_log_inflation')) or 0)<0 for r in rows))
    statuses=Counter(r['optimization_status'] for r in rows)
    for status in ('finite','infinite','unresolved'):
        result['n_'+status]=statuses[status]
    for field in ('optimization_status','selection_status','oracle_capacity_status','analytical_status',
                  'realized_status','decomposition_status','oracle_capacity_decomposition_status',
                  'true_mass_slice_selection_status','same_partition_true_mass_slice_status'):
        result[field+'_counts_json']=json.dumps(dict(sorted(Counter(r.get(field,'not_applicable') for r in rows).items())),sort_keys=True)
    for metric in METRICS:
        statistics_for_metric=_stats([row.get(metric) for row in rows],metric in BINARY)
        result.update({metric+'_'+key:value for key,value in statistics_for_metric.items()})
    nested=[r for r in rows if r.get('truth_mass_slice_nested') is True]
    nonnested=[r for r in rows if r.get('truth_mass_slice_nested') is False]
    result.update(n_nested_truth_mass_slice=len(nested),n_nonnested_truth_mass_slice=len(nonnested))
    for name,selected in (('nested',nested),('nonnested',nonnested)):
        for metric in ('mass_slice_count_gap','threshold_log_mass_inflation','threshold_log_propensity_inflation',
                       'decomposition_log_total','decomposition_log_mass','decomposition_log_propensity','decomposition_log_selection'):
            result.update({name+'_'+metric+'_'+key:value for key,value in _stats([r.get(metric) for r in selected]).items()})
    return result


def _paired(left,right,metric,expected):
    left,right=list(left),list(right)
    if len({r['replication'] for r in left})!=len(left) or len({r['replication'] for r in right})!=len(right):
        raise AssertionError('Paired comparison contains a duplicate block')
    left={r['replication']:r for r in left};right={r['replication']:r for r in right}
    common=sorted(set(left)&set(right));deltas=[];discordance=Counter()
    for rep in common:
        a,b=_number(left[rep].get(metric)),_number(right[rep].get(metric))
        if a is None or b is None:
            continue
        deltas.append(a-b)
        if metric in BINARY:
            discordance['both_true' if a and b else 'left_only' if a else 'right_only' if b else 'neither']+=1
    result={'metric':metric,'n_expected':expected,'n_matching_blocks':len(common),
            'n_unavailable_pairs':expected-len(deltas),'direction':'left minus right'}
    result.update({'difference_'+k:v for k,v in _stats(deltas).items()})
    if metric in BINARY:
        result.update({'discordant_'+key:discordance[key] for key in ('both_true','left_only','right_only','neither')})
    return result


def _equal_weight_overall(rows,condition_keys,expected_conditions=48,expected_reps=100):
    """Equal condition weights;100 rep-index averages preserve pairing."""
    grouped=_groups(rows,tuple(condition_keys)+('procedure',))
    conditions={key[:-1] for key in grouped}
    if len(conditions)!=expected_conditions:
        raise AssertionError('The primary equal-weight summary lacks a committed condition')
    output=[]
    for procedure in PROCEDURES:
        selected=[r for r in rows if r['procedure']==procedure]
        by_rep=defaultdict(list)
        for row in selected:
            by_rep[row['replication']].append(row)
        if len(by_rep)!=expected_reps or any(len(v)!=expected_conditions for v in by_rep.values()):
            raise AssertionError('Primary replicate clusters are incomplete')
        for metric in ('count_gap','normalized_count_gap','normalized_compressible_gap','selected_count','compression',
                       'oracle_count','threshold_log_inflation','threshold_log_mass_inflation',
                       'threshold_log_propensity_inflation','count_recovered','positive_count_gap',
                       'threshold_cap_violation','certificate_below_selected_truth','calibration_failure',
                       'procedure_event_failure','mass_slice_count_gap'):
            averages=[];unavailable=0
            for rep,values in sorted(by_rep.items()):
                numeric=[_number(r.get(metric)) for r in values]
                if any(value is None for value in numeric):
                    unavailable+=1
                    continue
                averages.append(statistics.fmean(numeric))
            result={'procedure':procedure,'metric':metric,'conditions_equal_weighted':expected_conditions,
                    'independent_replication_clusters':expected_reps,'unavailable_replication_clusters':unavailable,
                    'weighting':'equal weight for each of the original48conditions; mean within each replication index',
                    'uncertainty_scope':'mean-SE across independent replication-index averages; not a pooled binomial sample'}
            result.update(_stats(averages))
            output.append(result)
    return output


def _equal_weight_paired(rows,expected_conditions=48,expected_reps=100):
    """Paired contrasts of equally weighted condition averages.

    Binary condition averages are not Bernoulli observations. Their paired
    differences use mean-SE uncertainty, without a binomial formula.
    """
    groups=_groups(rows,('procedure','replication'))
    if len(groups)!=len(PROCEDURES)*expected_reps or any(
            len(values)!=expected_conditions for values in groups.values()):
        raise AssertionError('Equal-weight paired clusters are incomplete')
    output=[]
    for metric in PAIR_METRICS['historical']:
        averages={procedure:[] for procedure in PROCEDURES}
        for (procedure,rep),values in sorted(groups.items()):
            numeric=[_number(row.get(metric)) for row in values]
            value=statistics.fmean(numeric) if all(v is not None for v in numeric) else None
            averages[procedure].append({'replication':rep,'condition_average':value})
        for left,right in (('allocation_control','original'),('pooled_alternative','original'),
                           ('pooled_alternative','allocation_control')):
            result=_paired(averages[left],averages[right],'condition_average',expected_reps)
            result.update(metric=metric,left_procedure=left,right_procedure=right,
                          conditions_equal_weighted=expected_conditions,
                          uncertainty_scope='Paired differences of replication-index condition averages; mean-SE, not a binomial sample')
            output.append(result)
    return output


def _csv(path,rows):
    rows=list(rows);fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:('NA' if value is None else str(value) if isinstance(value,F) else
                                  json.dumps(value,sort_keys=True) if isinstance(value,(dict,list,tuple)) else value)
                             for key,value in row.items()})


def _receipts(root):
    path=root/'panel/receipts.jsonl'
    if not path.exists():
        return {}
    answer={}
    for line in path.read_text().splitlines():
        if line.strip():
            item=json.loads(line);answer[tuple(item['task'])]=item
    return answer


def _maximum_precision(value):
    maximum=0
    def visit(item):
        nonlocal maximum
        if isinstance(item,dict):
            if isinstance(item.get('precision'),int):
                maximum=max(maximum,item['precision'])
            for child in item.values():
                if isinstance(child,(dict,list,tuple)):
                    visit(child)
        elif isinstance(item,(list,tuple)):
            for child in item:
                visit(child)
    visit(value)
    return maximum


def _stage(block,kind,path,receipt):
    task=tuple(block['task'])
    raw=block.get('new_region')
    pooled=block.get('pooled_interval_evidence',[])
    if kind=='recovery':
        root_fallbacks=block['root_fallbacks']['row_occurrences']
        precision=max(r['maximum_precision_used'] for r in block['rows'])
    else:
        root_fallbacks=raw['root_fallbacks']+sum(r['status']=='fallback' for r in pooled)
        precision=_maximum_precision((raw,pooled))
    answer={'component':kind,'task':task,'evidence_file':str(path),
            'rows':len(block['rows']),'worker_seconds_before_serialization':block['worker_seconds_before_serialization'],
            'seconds_including_serialization':receipt.get('seconds'),
            'serialization_seconds':receipt.get('serialization_seconds'),
            'peak_rss_bytes':block['peak_rss_bytes'],'bytes':path.stat().st_size,
            'root_cache_entries':receipt.get('root_cache_entries'),
            'root_cache_hits_reported':block.get('root_cache',{}).get('hits'),
            'root_cache_misses_reported':block.get('root_cache',{}).get('misses'),
            'cache_counter_scope':'per-path deltas' if kind=='recovery' else 'cumulative worker counters; do not sum',
            'root_fallback_occurrences':root_fallbacks,'maximum_root_precision_used':precision,
            **{key:value for key,value in block.get('timings',{}).items()}}
    answer.update(_flatten_scalars(block.get('source_original_timings',{}),'source_original_'))
    answer['timing_interpretation']=block.get('timing_interpretation',
        'Shared computation with warm caches; source historical timings are not a matched cold-method benchmark')
    return answer


def _first_crossings(recovery_summaries):
    keys=('law_id','mass_mode','procedure')
    answer=[]
    for key,rows in sorted(_groups(recovery_summaries,keys).items()):
        rows=sorted(rows,key=lambda r:r['n'])
        point=next((r['n'] for r in rows if r['certified_mean'] is not None and r['certified_mean']>=.9),None)
        lower=next((r['n'] for r in rows if r['certified_cp95_lower'] is not None and r['certified_cp95_lower']>=.9),None)
        answer.append(dict(zip(keys,key),first_grid_point_rate_at_least_90pct=point,
                           first_grid_cp95_lower_at_least_90pct=lower,
                           grid_points=len(rows),largest_grid_n=max(r['n'] for r in rows),
                           scope='descriptive empirical grid crossing; pointwise intervals, not exact sample complexity'))
    return answer




def report(root,development=False):
    """Read the entire completed panel and write reproducible reporting outputs."""
    started=time.perf_counter();root=Path(root)
    if development:
        allowed=(Path(tempfile.gettempdir()).resolve(),Path('/private/tmp').resolve())
        if not any(root.resolve().is_relative_to(path) for path in allowed):
            raise ValueError('Development reporting requires a temporary fixture root')
    protocol=load(root/'frozen_protocol.json')
    config=protocol['configuration'];receipts=_receipts(root)
    expected={'historical':4800,'release':300,'recovery':config['recovery']['replications']}
    if development:
        expected={kind:len(list((root/'panel'/kind).glob('*.json.gz'))) for kind in expected}
        if not all(expected.values()):
            raise AssertionError('Development fixture must contain all three study components')
    all_rows={kind:[] for kind in expected};stages=[];stage_counts=Counter();status_counts=defaultdict(Counter)
    diagnostics=Counter();summary_rows={};paired=[];timing_summaries=[]
    cell_statuses=defaultdict(Counter)
    for kind,wanted in expected.items():
        paths=sorted((root/'panel'/kind).glob('*.json.gz'))
        if len(paths)!=wanted:
            raise AssertionError(f'Complete {kind} panel required: found {len(paths)} of {wanted}')
        seen=set()
        for path in paths:
            block=load(path);task=tuple(block['task'])
            if task in seen or task[0]!=kind or (block.get('development') and not development):
                raise AssertionError('Repeated, development, or mismatched committed task')
            seen.add(task)
            if kind=='historical':
                expected_grid={(cap,procedure) for cap in protocol['caps'] for procedure in PROCEDURES}
                actual={(r['cap'],r['procedure']) for r in block['rows']}
            elif kind=='release':
                i=block['law_index'];expected_grid={(B,p) for B in protocol['release_protocol']['capacity_grids'][i] for p in PROCEDURES}
                actual={(r['B'],r['procedure']) for r in block['rows']}
            else:
                expected_grid={(law,n,mode,p) for law in ('homogeneous','one_exceptional_state')
                               for n in config['recovery']['sample_sizes'] for mode in ('known_mass','unknown_mass') for p in PROCEDURES}
                actual={(r['law_id'],r['n'],r['mass_mode'],r['procedure']) for r in block['rows']}
            if actual!=expected_grid or len(block['rows'])!=len(expected_grid):
                raise AssertionError('Committed task omitted or repeated a condition')
            law=(protocol['release_protocol']['laws'][block['law_index']] if kind=='release' else None)
            enriched=[_enrich(row,block,kind,str(path.relative_to(root)),law) for row in block['rows']]
            comparison_keys={'historical':('cap',),'release':('B',),
                             'recovery':('law_id','n','mass_mode')}[kind]
            for values in _groups(enriched,comparison_keys).values():
                by_procedure={row['procedure']:row for row in values}
                controlled=by_procedure['allocation_control'];pooled=by_procedure['pooled_alternative']
                if kind=='historical' and all(row['optimization_status']=='finite' for row in (controlled,pooled)):
                    if pooled['selected_count']>controlled['selected_count']:
                        raise AssertionError('Pooled completed minimum count exceeds its allocation control')
                    diagnostics['structural_pooled_count_comparisons']+=1
                status_key='optimization_status' if kind=='recovery' else 'oracle_capacity_status'
                value_key='certificate' if kind=='recovery' else 'oracle_capacity_minimax_certificate'
                if controlled[status_key]=='finite' and pooled[status_key]=='infinite':
                    raise AssertionError('Pooled envelope became infinite with a finite control optimum')
                if all(row[status_key]=='finite' for row in (controlled,pooled)):
                    if pooled[value_key]>controlled[value_key]:
                        raise AssertionError('Pooled completed certificate exceeds its allocation control')
                    diagnostics['structural_pooled_certificate_comparisons']+=1
            if kind=='release':
                if any(row['selected_count'] is not None and row['selected_count']>row['B'] for row in enriched):
                    raise AssertionError('Returned release partition violates its reporting capacity')
                diagnostics['release_capacity_count_checks']+=len(enriched)
            all_rows[kind].extend(enriched)
            stages.append(_stage(block,kind,path,receipts.get(task,{})))
            for field in ('new_raw_cells','new_pooled_cells','original_completed_updates'):
                cell_statuses[kind+'_'+field].update(r['status'] for r in block.get(field,()))
            stage_counts[kind]+=1
            status_counts[kind].update(r['optimization_status'] for r in enriched)
            for row in enriched:
                for metric in BINARY:
                    if row.get(metric) is not None:
                        diagnostics[kind+'_'+metric+'_observations']+=1
                        diagnostics[kind+'_'+metric+'_true']+=bool(row[metric])
                if row.get('procedure_event_failure') is False and row.get('certificate_below_selected_truth'):
                    raise AssertionError('Used confidence event holds but a completed certificate misses selected truth')
                if row.get('used_certificate_event_failure') is False and row.get('certificate_below_selected_truth'):
                    raise AssertionError('Used recovery confidence event holds but the certificate misses truth')
                if kind=='release' and row.get('decomposition_identity_exact'):
                    diagnostics['release_exact_decomposition_rows']+=1
            del block
        if not development:
            intended=({('historical',k) for k in range(4800)} if kind=='historical' else
                      {('release',i,j) for i in range(3) for j in range(100)} if kind=='release' else
                      {('recovery',j) for j in range(config['recovery']['replications'])})
            if seen!=intended:
                raise AssertionError('Completed task identities differ from the frozen panel')
        keys=KEYS[kind];n_expected=(max(len(v) for v in _groups(all_rows[kind],keys).values())
            if development else config['recovery']['replications'] if kind=='recovery' else 100)
        groups=_groups(all_rows[kind],keys)
        summary_rows[kind]=[dict(zip(keys,key),**_condition_summary(values,n_expected))
                            for key,values in sorted(groups.items())]
        if any(row['n_blocks']!=n_expected for row in summary_rows[kind]):
            raise AssertionError('A condition lacks independent replicates')
        if not development and any({r['replication'] for r in values}!=set(range(n_expected)) for values in groups.values()):
            raise AssertionError('A condition has an unexpected replication index')
        base_keys=keys[:-1]
        for key,values in sorted(_groups(all_rows[kind],base_keys).items()):
            by_procedure={p:[r for r in values if r['procedure']==p] for p in PROCEDURES}
            for left,right in (('allocation_control','original'),('pooled_alternative','original'),('pooled_alternative','allocation_control')):
                for metric in PAIR_METRICS[kind]:
                    paired.append(dict(zip(base_keys,key),component=kind,left_procedure=left,right_procedure=right,
                                       **_paired(by_procedure[left],by_procedure[right],metric,n_expected)))
    counts={kind:len(rows) for kind,rows in all_rows.items()}
    if not development and counts!={'historical':43200,'release':4500,'recovery':43200}:
        raise AssertionError('Complete row counts differ from the frozen scientific panel')
    if not development and {kind:len(rows) for kind,rows in summary_rows.items()}!={'historical':432,'release':45,'recovery':144}:
        raise AssertionError('Complete condition grid differs from the frozen scientific panel')
    primary=[r for r in all_rows['historical'] if r['cap']==F(config['primary_cap'])]
    overall=_equal_weight_overall(primary,('condition_id',),
        len({r['condition_id'] for r in primary}) if development else 48,
        len({r['replication'] for r in primary}) if development else 100)
    overall_paired=_equal_weight_paired(primary,
        len({r['condition_id'] for r in primary}) if development else 48,
        len({r['replication'] for r in primary}) if development else 100)
    crossings=_first_crossings(summary_rows['recovery'])
    tables=root/'tables';tables.mkdir(exist_ok=True)
    for kind,rows in all_rows.items():
        rows.sort(key=lambda r:tuple(r[k] for k in KEYS[kind])+(r['replication'],))
        _csv(tables/f'{kind}_rows.csv',rows)
        _csv(tables/f'{kind}_condition_summary.csv',summary_rows[kind])
    _csv(tables/'paired_comparisons.csv',paired)
    _csv(tables/'historical_primary_equal_weight_summary.csv',overall)
    _csv(tables/'historical_primary_equal_weight_paired.csv',overall_paired)
    _csv(tables/'recovery_grid_crossings.csv',crossings)
    _csv(tables/'stage_times.csv',stages)
    distributions=[]
    for kind,summaries in summary_rows.items():
        keys=KEYS[kind]
        for condition in summaries:
            for metric in ('certificate_inflation_log','realized_b','analytical_b_n','manuscript_reference_b_n',
                           'oracle_margin_log','minimum_state_count','minimum_treated_or_control_count',
                           'expected_minimum_state_count','population_overlap','normalized_count_gap'):
                distributions.append({**{k:condition[k] for k in keys},'component':kind,'metric':metric,
                    **{key:condition.get(metric+'_'+key) for key in ('n','n_missing','n_nonfinite_or_nonnumeric','n_negative','n_zero','n_positive','minimum','q05','q25','median','q75','q95','maximum')},
                    'optimization_status_counts_json':condition['optimization_status_counts_json'],
                    'analytical_status_counts_json':condition['analytical_status_counts_json']})
    _csv(tables/'diagnostic_distributions.csv',distributions)
    slices=[]
    for kind in ('historical','release'):
        for condition in summary_rows[kind]:
            slices.append({**{k:condition[k] for k in KEYS[kind]},'component':kind,
                           **{k:v for k,v in condition.items() if k.startswith(('nested_','nonnested_','n_nested','n_nonnested'))},
                           'scope':'Nested-region attribution conditional on truth-mass feasibility; nonnested comparisons retained separately and descriptive'})
    _csv(tables/'truth_mass_slice_decomposition.csv',slices)
    for kind,values in _groups(stages,('component',)).items():
        stage_metrics=tuple(dict.fromkeys(('worker_seconds_before_serialization','seconds_including_serialization',
            'serialization_seconds','peak_rss_bytes','bytes','root_cache_entries','root_fallback_occurrences',
            'maximum_root_precision_used')+tuple(key for r in values for key in r if key.endswith('_seconds'))))
        for metric in stage_metrics:
            timing_summaries.append({'component':kind[0],'metric':metric,**_stats([r.get(metric) for r in values])})
    _csv(tables/'timing_summary.csv',timing_summaries)
    pass  # Public release generates numerical tables only.
    primary_lookup={(r['procedure'],r['metric']):r for r in overall}
    manifest=load(root/'manifest.json')
    result={'version':config['version'],'status':'development_report_complete' if development else 'report_complete',
            'development_fixture_only':bool(development),'generated_utc':now(),
            'independent_input_units':dict(stage_counts),'rows':counts,
            'conditions':{kind:len(rows) for kind,rows in summary_rows.items()},
            'status_counts':{kind:dict(value) for kind,value in status_counts.items()},
            'cell_record_status_counts':{key:dict(value) for key,value in cell_statuses.items()},
            'root_policy':config['root_policy'],'recovery_root_policy':config['recovery_root_policy'],
            'checks_and_counts':dict(diagnostics),
            'historical_primary_equal_weight_summary':overall,
            'historical_primary_equal_weight_paired':overall_paired,
            'historical_primary_original':{metric:primary_lookup['original',metric] for metric in (
                'count_gap','normalized_count_gap','normalized_compressible_gap','count_recovered',
                'threshold_log_inflation','threshold_cap_violation','certificate_below_selected_truth',
                'calibration_failure','mass_slice_count_gap')},
            'recovery_first_grid_crossings':crossings,
            'recovery_unsafe_alternative':[{k:r[k] for k in RECOVERY_KEYS+('unsafe_certification_successes',
                'unsafe_certification_n','unsafe_certification_mean','unsafe_certification_cp95_lower','unsafe_certification_cp95_upper')}
                for r in summary_rows['recovery'] if r['law_id']=='one_exceptional_state'],
            'timing_summary':timing_summaries,
            'maximum_provider_peak_rss_bytes':max(s['peak_rss_bytes'] for s in stages),
            'maximum_root_precision_observed':max(s['maximum_root_precision_used'] for s in stages),
            'maximum_root_cache_entries':max((s['root_cache_entries'] or 0 for s in stages)),
            'preflight':manifest.get('preflight'),'panel':manifest.get('panel'),
            'reference_sample_sizes':{'necessary_lower_reference':1900,'sufficient_diameter_reference':3000000,
                                      'direct_diameter_threshold':protocol['theoretical_reference'].get('direct_diameter_threshold')},
            'inference_scope':{
                'binary_condition_intervals':'Pointwise two-sided95% Clopper-Pearson with verified outward endpoints. Shared capacities/sizes/procedures are paired, not independent replications.',
                'continuous_and_paired':'Pointwise mean +/-1.96SE over independent block differences; all missing/infinite/unresolved denominators retained.',
                'overall':'Equal weight for each of the original48conditions at cap3/2, unknown mass. Uncertainty uses100 replication-index averages preserving cross-condition pairing; no K-dependent pooling.',
                'historical_actual_selection':'threshold_* fields concern the actual minimum-count threshold-selected partition. oracle_capacity_* and minimax_log_inflation concern a separate minimax optimization at the oracle count.',
                'unresolved_count_selection':'count_recovered requires a completed finite optimum. Unresolved candidates count as no established recovery and retain their separate status; their candidate counts/gaps remain in unresolved_candidate_* fields but are excluded from completed continuous count summaries.',
                'diameter':'Inflation is measured on the explicitly evaluated partition, not an all-partition maximum. Sufficient analytical b_n is a separate diagnostic with explicit applicability/root-fallback status.',
                'mass_slice':'Same propensity intervals and allocation, with true masses fixed. Attribution to a nested mass restriction is conditional on truth-mass feasibility. Nonnested slices and negative inflations are retained.',
                'recovery':'First90% grid crossing is descriptive and not exact sample complexity. Pointwise confidence is not simultaneous over nested sample sizes. Provider is not told homogeneous e.',
                'alternative':'Pooled alternative is min(Uraw,V) on its raw+pooled common event; it is not the exact supremum over the joint intersection. Control spends the same raw allocation as the alternative.',
                'timing':'Three-procedure computation shares regions, certificates and caches within a worker. Block costs are measured; procedure-specific speed differences cannot be inferred. Worker lifetime RSS/cache counters reflect prior tasks.',
                'scope':'Assignment-only reanalysis; no outcome experiments rerun, and no manuscript or historical output edits.',
            },'report_seconds':time.perf_counter()-started}
    dump(root/'summary.json',result)
    manifest['development_report_complete' if development else 'report_complete']=True
    manifest['report_generated_utc']=result['generated_utc']
    dump(root/'manifest.json',manifest)
    print(json.dumps({'status':result['status'],'rows':counts,'conditions':result['conditions'],
                      'report_seconds':result['report_seconds']}),flush=True)
    return result
