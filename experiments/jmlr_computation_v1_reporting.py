"""Complete computational scaling summaries from normalized saved cases.

Primary fresh-counts execution, capacity-minimax providers, dependent repeated
cohorts and separate preflight measurements have distinct denominators.  Stage
means support additive accounting; distributions retain censored elapsed work
and distinguish it from completed latency.  No benchmark is executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import gzip
import json
import math
from pathlib import Path
import statistics
import time

from experiments.jmlr_verified_adaptive_v1_math import decode, encode


SUPPORTS = (16,32,64,128,256)
CAPACITY_SUPPORTS = (64,128,256)
REPEAT_SUPPORTS = (64,256)
PREFLIGHT_SUPPORTS = (64,256)
STATUSES = ('completed','timeout','error','memory_limit')
CONDITION_KEYS = ('kind','law_index','law_id','K')
PROVIDER_STAGES = ('provider_validation_seconds','confidence_region_seconds',
    'family_seconds','cell_certification_seconds','search_seconds','instrumentation_seconds')
AGGREGATE_STAGES = (
    'input_load_validate_seconds','provider_validation_seconds',
    'confidence_region_seconds','family_seconds','cell_certification_seconds',
    'search_seconds','inference_input_validation_seconds',
    'projection_export_seconds','aggregate_validation_seconds',
    'aggregate_outcome_cp_seconds','aggregate_endpoint_optimization_seconds',
    'instrumentation_seconds','provider_accounting_remainder_seconds',
    'aggregate_unallocated_seconds')
RAW_STAGES = (
    'raw_outcome_cp_seconds','raw_endpoint_optimization_seconds',
    'lp_construction_seconds','lp_solver_matrices_seconds',
    'lp_solver_proposal_seconds','lp_certificate_construction_seconds',
    'lp_solver_orchestration_seconds','lp_exact_verification_seconds',
    'lp_intersection_seconds','raw_reference_unallocated_seconds')
INCLUSIVE_TIMINGS = (
    'aggregate_total_from_counts_seconds','aggregate_pipeline_seconds',
    'additional_raw_reference_seconds','inference_total_seconds',
    'provider_total_seconds','provider_total_from_counts_seconds',
    'raw_reference_additional_seconds','process_wall_seconds',
    'inprocess_work_wall_seconds','child_wall_until_serialization_seconds',
    'repeat_wall_seconds')
DIAGNOSTIC_TIMINGS = ('aggregate_accounting_remainder_seconds',
    'process_accounting_residual_seconds','process_wall_accounting_residual_seconds')
WORK_PREFIXES = ('counters_','root_work_','inference_work_',
                 'inference_raw_work_','inference_lp_work_','numerical_',
                 'work_','lp_work_')
STATUS_DEFINITIONS = (
    ('execution','completed','Process returned its measurement record; inspect optimization, root and LP diagnostics separately.','not_implied'),
    ('execution','timeout','Declared wall limit reached; elapsed work retained as censored, with stage/reason.','not_implied'),
    ('execution','memory_limit','Declared resident-memory limit reached; observed work/peak retained, not a completed latency draw.','not_implied'),
    ('execution','error','Execution error retained with reason; no inference of scientific infeasibility.','not_implied'),
    ('repeat','not_reached','Planned additional cohort was not measured because its primary prerequisite did not return.','not_implied'),
    ('confidence_root','verified','Numerical endpoint enclosure verified under configured grid, precision and resource policy.','confidence_enclosure'),
    ('confidence_root','fallback','Safe confidence-region enlargement, commonly [0,1], with reason; distinct from search failure or an informative endpoint.','valid_wider_confidence_enclosure'),
    ('cell','full','Completed finite cell envelope; coupled value may be used as an upper certificate.','completed_finite_upper_bound'),
    ('cell','rejected','Rigorous lower bound excludes the current cap/incumbent; partial anchors are not a completed upper envelope.','lower_bound_only'),
    ('cell','boundary','Completed mathematically infinite envelope due to a genuine boundary coordinate; distinct from numerical failure.','completed_infinite_envelope'),
    ('cell','failure','Cell computation unresolved; partial lower information does not prove infeasibility or a completed upper envelope.','unresolved'),
    ('search','finite','Finite candidate/value; count_optimal, completed and deterministic_tie_complete identify the actual proof scope.','inspect_proof_flags'),
    ('search','unresolved','Optimum or required deterministic tie remains unproved; a safe feasible candidate may still exist.','unresolved'),
    ('selection','ordinary_completed_identity','Identity selected by a completed optimization; distortion/certificate one, with no compression.','completed_identity'),
    ('selection','identity_fallback','Safe identity result retained separately; fallback alone does not establish the requested optimum.','feasible_fallback_not_optimality_proof'),
    ('LP','certified','Solver proposal passed exact bound-certificate verification; retained LP evidence supports the enclosure.','verified_LP_bound'),
    ('LP','solver_failure','Solver did not provide an accepted proposal; use the explicitly labeled certified fallback/intersection path.','no_verified_new_LP_bound'),
    ('LP','verification_failure','Numerical proposal failed exact verification; discard it and retain the labeled valid fallback.','no_verified_new_LP_bound'),
    ('intersection','empty_intersection','Conflicting enclosures trigger the labeled safe evaluation fallback; not a successful narrow interval.','safe_fallback'),
    ('counter','omitted_category','An absent category in a provided Counter mapping is zero; an absent whole diagnostic remains missing.','reporting_convention'),
)


def _number(value):
    if value is None or isinstance(value,bool) or not isinstance(value,(int,float,F)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    location = (len(values)-1)*probability
    lower = math.floor(location);upper = math.ceil(location)
    return values[lower]+(values[upper]-values[lower])*(location-lower)


def _statistics(values):
    values = list(values)
    finite = [v for value in values if (v:=_number(value)) is not None]
    result = {'n':len(finite),'n_records':len(values),
              'n_missing':sum(value is None for value in values),
              'n_nonfinite_or_nonnumeric':sum(value is not None and _number(value) is None for value in values),
              'mean':None,'sd':None,'minimum':None,'q25':None,'median':None,
              'q75':None,'iqr':None,'maximum':None,'sum':None}
    if finite:
        q25,q75 = _quantile(finite,.25),_quantile(finite,.75)
        result.update(mean=statistics.fmean(finite),
            sd=statistics.stdev(finite) if len(finite)>1 else None,
            minimum=min(finite),q25=q25,median=_quantile(finite,.5),
            q75=q75,iqr=q75-q25,maximum=max(finite),sum=math.fsum(finite))
    return result


def _flatten(value,prefix=''):
    result = {}
    for key,item in value.items():
        name = prefix+str(key)
        if key in ('cache_keys','selected_records') and isinstance(item,(list,tuple)):
            # Numerical inventories remain authoritative in saved case/evidence.
            # A derived CSV needs their count, not repeated very long fields.
            result[name+'_count'] = len(item)
        elif isinstance(item,dict):
            result.update(_flatten(item,name+'_'))
        elif item is None or isinstance(item,(bool,str,int,float,F)):
            result[name] = item
        else:
            result[name+'_json'] = json.dumps(encode(item),sort_keys=True)
    return result


def _group(rows,keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def _scope(name):
    if name in INCLUSIVE_TIMINGS:
        return 'inclusive_total'
    if name in DIAGNOSTIC_TIMINGS:
        return 'accounting_diagnostic'
    if name in AGGREGATE_STAGES:
        return 'aggregate_pipeline'
    if name in RAW_STAGES:
        return 'additional_raw_reference'
    if 'serialization' in name:
        return 'serialization'
    if 'overhead' in name or 'unallocated' in name:
        return 'unallocated_or_process_overhead'
    return 'other_exclusive_stage'


def _normalize(record):
    row = {name:record.get(name) for name in (
        'case_id','kind','law_index','law_id','K','replication','status',
        'stage_reached','reason','evidence_file','source_input','input_file',
        'input_path','evidence_path','input_bytes','evidence_bytes','record_bytes')}
    if row['status'] not in STATUSES:
        raise ValueError('Unrecognized process completion status')
    row['process_completed'] = row['status']=='completed'
    row['process_censored'] = row['status'] in ('timeout','memory_limit')
    row['has_repeat'] = isinstance(record.get('repeat'),dict) and bool(record['repeat'])
    for section in ('timings','memory','counters','selection','numerical','root_work','inference'):
        if isinstance(record.get(section),dict):
            row.update(_flatten(record[section],section+'_'))
    for name in ('process_wall_seconds','process_overhead_seconds',
                 'inprocess_work_wall_seconds','child_wall_until_serialization_seconds',
                 'process_wall_accounting_residual_seconds'):
        if name in record:
            row['timings_'+name] = record[name]
    row['memory_scope'] = ('whole fresh process lifetime, including raw reference and any repeated cohort; '
                           'not an aggregate-only peak and not stage-memory differences')
    row['input_boundary'] = 'saved finite-state sufficient counts; raw-record preprocessing not measured'
    timings = dict(record.get('timings') or {})
    # These remainders describe actual timing boundaries.  They do not turn
    # unmeasured future stages of a censored run into zero-time successes.
    provider_total = _number(timings.get('provider_total_from_counts_seconds'))
    provider_fields = ('input_load_validate_seconds',)+PROVIDER_STAGES
    if provider_total is not None and all(_number(timings.get(k)) is not None for k in provider_fields):
        timings['provider_accounting_remainder_seconds'] = provider_total-math.fsum(timings[k] for k in provider_fields)
    aggregate_total = _number(timings.get('aggregate_total_from_counts_seconds'))
    if aggregate_total is not None and all(_number(timings.get(k)) is not None for k in AGGREGATE_STAGES):
        timings['aggregate_accounting_remainder_seconds'] = aggregate_total-math.fsum(timings[k] for k in AGGREGATE_STAGES)
    main_exclusive = [v for k,v in (record.get('timings') or {}).items()
                      if k not in INCLUSIVE_TIMINGS+DIAGNOSTIC_TIMINGS
                      and 'serialization' not in k and _number(v) is not None]
    inprocess = _number(record.get('inprocess_work_wall_seconds'))
    repeated = record.get('repeat') or {}
    repeat_wall = _number(repeated.get('wall_seconds')) if isinstance(repeated,dict) else None
    if inprocess is not None:
        timings['inprocess_unallocated_seconds'] = inprocess-math.fsum(main_exclusive)-(repeat_wall or 0.)
        timings['repeated_work_seconds'] = repeat_wall or 0.
    child_wall = _number(record.get('child_wall_until_serialization_seconds'))
    serialization = [v for k,v in (record.get('timings') or {}).items()
                     if 'serialization' in k and _number(v) is not None]
    if child_wall is not None and inprocess is not None:
        timings['child_serialization_orchestration_seconds'] = child_wall-inprocess-math.fsum(serialization)
    if (_number(record.get('process_wall_seconds')) is not None and child_wall is not None
            and _number(record.get('process_overhead_seconds')) is not None):
        timings['process_accounting_residual_seconds'] = (record['process_wall_seconds']-
            child_wall-record['process_overhead_seconds'])
    row.update({'timings_'+k:v for k,v in timings.items()})
    if 'selection_identity' in row:
        row['selection_ordinary_completed_identity'] = bool(row['selection_identity']
            and row.get('selection_selection_complete') and not row.get('selection_identity_fallback'))
    for name,value in list(row.items()):
        if name.startswith(('numerical_','root_work_','inference_lp_work_')) and _number(value) is not None:
            if any(token in name for token in ('fallback','boundary','rejected','failure','unresolved','precision_escalated')):
                row['numerical_event_'+name] = value>0
    return row


def _repeats(record,planned=False):
    repeat = record.get('repeat')
    if (not isinstance(repeat,dict) or not repeat) and not planned:
        return None
    repeat = repeat if isinstance(repeat,dict) and repeat else {
        'status':'not_reached','reason':'primary execution did not produce its planned repeated-cohort measurement'}
    row = {name:record.get(name) for name in ('case_id','law_index','law_id','K','replication','evidence_file')}
    row.update(kind='repeat' if record['kind']=='primary' else record['kind']+'_repeat',
               parent_kind=record['kind'],parent_status=record['status'],
               status=repeat.get('status',record['status']),
               independent_calibration_dataset=False,
               reuse_scope='same completed assignment calibration and selected partition; fresh outcome cohort and outcome caches')
    for key,value in repeat.items():
        if isinstance(value,dict):
            row.update(_flatten(value,key+'_'))
        elif value is None or isinstance(value,(str,bool,int,float,F)):
            row[key] = value
        else:
            row[key+'_json'] = json.dumps(encode(value),sort_keys=True)
    if isinstance(repeat.get('totals'),dict):
        row.update(_flatten(repeat['totals'],'timings_'))
    if _number(repeat.get('wall_seconds')) is not None:
        row['timings_repeat_wall_seconds'] = repeat['wall_seconds']
    return row


def _validate(rows,repeats,primary_reps,secondary_reps,preflight_reps):
    identities = [(r['kind'],r['case_id']) for r in rows]
    if len(set(identities))!=len(identities):
        raise AssertionError('Duplicate measured process case')
    for kind,supports,reps in (('primary',SUPPORTS,primary_reps),
            ('capacity',CAPACITY_SUPPORTS,secondary_reps),('preflight',PREFLIGHT_SUPPORTS,preflight_reps)):
        expected = {(law,K,rep) for law in (0,1) for K in supports for rep in range(reps)}
        observed = [(r['law_index'],r['K'],r['replication']) for r in rows if r['kind']==kind]
        if len(observed)!=len(set(observed)) or set(observed)!=expected:
            raise AssertionError(f'Every committed {kind} case, including failed/censored cases, is required')
    if any(r['kind'] not in ('primary','capacity','preflight') for r in rows):
        raise AssertionError('Unexpected measurement kind')
    expected = {(law,K,rep) for law in (0,1) for K in REPEAT_SUPPORTS for rep in range(secondary_reps)}
    observed = [(r['law_index'],r['K'],r['replication']) for r in repeats if r['kind']=='repeat']
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise AssertionError('Every committed dependent repeated-use example is required')


def _status_rows(rows,keys=CONDITION_KEYS):
    result = []
    for key,condition in sorted(_group(rows,keys).items()):
        ids = dict(zip(keys,key));denominator = len(condition)
        statuses = Counter(r['status'] for r in condition)
        for status in STATUSES+('not_reached',):
            result.append(ids | {'diagnostic':'process_status','category':status,
                'n_cases':denominator,'n_recorded':denominator,'n_missing':0,
                'events':statuses[status],'fraction':statuses[status]/denominator,
                'denominator_scope':'measured process cases in this condition'})
        # Boolean/categorical completion diagnostics remain distinct from the
        # process exit status; a fast safe fallback is not a completed search.
        fields = sorted({field for r in condition for field,value in r.items()
            if field.startswith(('selection_','numerical_','inference_methods_','methods_',
                                 'root_work_','inference_lp_work_','reuse_')) and isinstance(value,(bool,str))
            and not field.endswith('_json')})
        for field in fields:
            available = [r[field] for r in condition if isinstance(r.get(field),(bool,str))]
            for category,count in sorted(Counter(str(v) for v in available).items()):
                result.append(ids | {'diagnostic':field,'category':category,
                    'n_cases':denominator,'n_recorded':len(available),'n_missing':denominator-len(available),
                    'events':count,'fraction':count/denominator,
                    'denominator_scope':'all measured cases; missing diagnostic counted explicitly'})
    return result


def _condition_summary(rows,keys=CONDITION_KEYS):
    summaries,stages,work = [],[],[]
    for key,condition in sorted(_group(rows,keys).items()):
        ids = dict(zip(keys,key));finished = [r for r in condition if r['status']=='completed']
        item = ids | {'n_cases':len(condition),'n_process_completed':len(finished),
            'n_timeout':sum(r['status']=='timeout' for r in condition),
            'n_error':sum(r['status']=='error' for r in condition),
            'n_memory_limit':sum(r['status']=='memory_limit' for r in condition),
            'n_not_reached':sum(r['status']=='not_reached' for r in condition),
            'n_carrying_repeat':sum(bool(r.get('has_repeat')) for r in condition),
            'stage_reached_counts_json':json.dumps(dict(Counter(str(r.get('stage_reached')) for r in condition)),sort_keys=True),
            'failure_reason_counts_json':json.dumps(dict(Counter(str(r.get('reason')) for r in condition if r['status']!='completed')),sort_keys=True),
            'time_scope':'fresh execution from finite-state sufficient counts; aggregate before raw references',
            'memory_scope':condition[0].get('memory_scope')}
        metrics = sorted({name for r in condition for name,value in r.items()
            if name.startswith(('timings_','memory_')+WORK_PREFIXES) and
            (value is None or _number(value) is not None)})
        for metric in metrics:
            stats = _statistics(r.get(metric) for r in condition)
            completed_stats = _statistics(r.get(metric) for r in finished)
            if metric.startswith(('timings_','memory_')):
                item.update({metric+'_'+name:value for name,value in stats.items()})
                item.update({'process_completed_'+metric+'_'+name:value for name,value in completed_stats.items()})
            if metric.startswith('timings_'):
                name = metric[len('timings_'):]
                stage = ids | {'stage':name,'scope':_scope(name),
                    'exclusive':name not in INCLUSIVE_TIMINGS+DIAGNOSTIC_TIMINGS,
                    'aggregate_accounting_member':name in AGGREGATE_STAGES+('aggregate_accounting_remainder_seconds',),
                    'process_accounting_member':name not in INCLUSIVE_TIMINGS+DIAGNOSTIC_TIMINGS+('provider_accounting_remainder_seconds',),
                    'stage_origin':('derived_remainder' if name in ('provider_accounting_remainder_seconds',
                        'inprocess_unallocated_seconds','child_serialization_orchestration_seconds')+DIAGNOSTIC_TIMINGS
                        else 'inclusive_measurement' if name in INCLUSIVE_TIMINGS else 'exclusive_measurement'),
                    'n_cases':len(condition),'n_process_completed':len(finished),
                    'elapsed_scope':'recorded elapsed values; censored cases are not completed-latency draws'}
                stage.update(stats)
                stage.update({'process_completed_'+name:value for name,value in completed_stats.items()})
                stages.append(stage)
            if metric.startswith(WORK_PREFIXES):
                work.append(ids | {'metric':metric,'n_cases':len(condition),**stats})
        summaries.append(item)
    return summaries,stages,work


def _accounting_rows(rows):
    result=[]
    for row in rows:
        entry={key:row[key] for key in ('case_id',)+CONDITION_KEYS+('replication','status')}
        aggregate=_number(row.get('timings_aggregate_total_from_counts_seconds'))
        fields=AGGREGATE_STAGES+('aggregate_accounting_remainder_seconds',)
        if aggregate is not None and all(_number(row.get('timings_'+k)) is not None for k in fields):
            total=math.fsum(row['timings_'+k] for k in fields)
            entry.update(aggregate_total_seconds=aggregate,aggregate_stage_sum_seconds=total,
                aggregate_residual_seconds=aggregate-total,
                aggregate_accounting_complete=abs(aggregate-total)<=1e-8*max(1,aggregate))
        else:
            entry.update(aggregate_total_seconds=aggregate,aggregate_stage_sum_seconds=None,
                aggregate_residual_seconds=None,aggregate_accounting_complete=None)
        wall=_number(row.get('timings_process_wall_seconds'))
        if wall is not None and _number(row.get('timings_child_wall_until_serialization_seconds')) is not None:
            values=[value for key,value in row.items() if key.startswith('timings_')
                and key[len('timings_'):] not in INCLUSIVE_TIMINGS+DIAGNOSTIC_TIMINGS+('provider_accounting_remainder_seconds',)
                and _number(value) is not None]
            total=math.fsum(values)
            entry.update(process_wall_seconds=wall,process_stage_sum_seconds=total,
                process_residual_seconds=wall-total,
                process_accounting_complete=abs(wall-total)<=1e-8*max(1,wall))
        else:
            entry.update(process_wall_seconds=wall,process_stage_sum_seconds=None,
                process_residual_seconds=None,process_accounting_complete=None)
        result.append(entry)
    return result


def _capacity_selections(records):
    result=[]
    for record in records:
        if record['kind']!='capacity':
            continue
        choices=(record.get('selection') or {}).get('by_capacity')
        if not isinstance(choices,dict):
            # A timed-out provider may not have returned its count tables.
            choices={str((record['K']+d-1)//d):None for d in (16,8,4,2,1)}
        for capacity,choice in choices.items():
            row={name:record.get(name) for name in ('case_id','kind','law_index','law_id','K',
                'replication','status','stage_reached','reason','evidence_file')}
            row['capacity']=int(capacity)
            row.update(_flatten(choice or {},'selection_'))
            row['selection_observed']=choice is not None
            result.append(row)
    return result


def _count_map_summary(records):
    """Counter omissions are zero only within a provided count mapping.

    This prevents reason/solver-category means being conditional on a positive
    count while preserving genuinely unavailable diagnostics of failed cases.
    """
    maps=[]
    for record in records:
        inference=record.get('inference') or {}
        sources=(('provider_root_fallback_reasons',(record.get('root_work') or {}).get('fallback_reason_counts')),
            ('aggregate_root_fallback_reasons',(inference.get('work') or {}).get('fallback_reason_counts')),
            ('raw_root_fallback_reasons',(inference.get('raw_work') or {}).get('fallback_reason_counts')),
            ('LP_solver_status', (inference.get('lp_work') or {}).get('solver_status')))
        for scope,value in sources:
            maps.append({name:record[name] for name in CONDITION_KEYS} | {'scope':scope,'mapping':value})
    result=[]
    for key,condition in sorted(_group(maps,CONDITION_KEYS+('scope',)).items()):
        observed=[{str(k):v for k,v in r['mapping'].items()} for r in condition if isinstance(r['mapping'],dict)]
        categories=sorted({str(category) for mapping in observed for category in mapping})
        for category in ['any']+categories:
            values=[(sum(mapping.values()) if all(_number(v) is not None for v in mapping.values()) else None)
                    if category=='any' else mapping.get(category,0) for mapping in observed]
            numeric=[value for value in values if _number(value) is not None]
            events=sum(value>0 for value in numeric)
            result.append(dict(zip(CONDITION_KEYS+('scope',),key)) | {'category':category,
                'n_cases':len(condition),'n_recorded_maps':len(observed),
                'n_missing_maps':len(condition)-len(observed),'n_numeric_counts':len(numeric),
                'total_occurrences':sum(numeric),'cases_with_occurrence':events,
                'fraction_all_cases':events/len(condition) if len(numeric)==len(condition) else None,
                'fraction_lower_bound':events/len(condition),
                'fraction_upper_bound':(events+len(condition)-len(numeric))/len(condition),
                'mean_count_recorded_maps':statistics.fmean(numeric) if numeric else None,
                'denominator_scope':'all process cases; missing maps explicit; absent keys of a provided Counter map mean zero'})
    return result


def _paired_operational(rows,repeats):
    """Descriptive matched costs; comparison objectives and units stay explicit."""
    primary={(r['law_index'],r['K'],r['replication']):r for r in rows if r['kind']=='primary'}
    pairs=[]
    for row in rows:
        if row['kind']!='capacity':
            continue
        reference=primary.get((row['law_index'],row['K'],row['replication']))
        if reference is None:
            continue
        a=_number(row.get('timings_provider_total_from_counts_seconds'))
        b=_number(reference.get('timings_provider_total_from_counts_seconds'))
        pairs.append({name:row[name] for name in ('law_index','law_id','K','replication','case_id')} | {
            'comparison':'capacity_minimax_provider_vs_fixed_cap_provider',
            'left_status':row['status'],'right_status':reference['status'],
            'left_seconds':a,'right_seconds':b,
            'difference_seconds':None if a is None or b is None else a-b,
            'ratio':None if a is None or not b else a/b,
            'scope':'matched counts; two different provider objectives; separate fresh processes; raw inference excluded'})
    for row in repeats:
        if row['kind']!='repeat':
            continue
        reference=primary.get((row['law_index'],row['K'],row['replication']))
        if reference is None:
            continue
        for metric,label in (('aggregate_pipeline_seconds','repeated_vs_initial_aggregate_inference'),
                             ('inference_total_seconds','repeated_vs_initial_all_inference')):
            a=_number(row.get('timings_'+metric));b=_number(reference.get('timings_'+metric))
            pairs.append({name:row[name] for name in ('law_index','law_id','K','replication','case_id')} | {
                'comparison':label,'left_status':row['status'],'right_status':reference['status'],
                'left_seconds':a,'right_seconds':b,
                'difference_seconds':None if a is None or b is None else a-b,
                'ratio':None if a is None or not b else a/b,
                'scope':'dependent additional cohort; assignment calibration/partition retained; fresh outcome caches'})
    summaries=[]
    for key,condition in sorted(_group(pairs,('comparison','law_index','law_id','K')).items()):
        item=dict(zip(('comparison','law_index','law_id','K'),key)) | {'n_matched_cases':len(condition)}
        for metric in ('left_seconds','right_seconds','difference_seconds','ratio'):
            item.update({metric+'_'+name:value for name,value in _statistics(r.get(metric) for r in condition).items()})
        summaries.append(item)
    return pairs,summaries


def _overviews(rows,repeats,condition_summaries,repeat_summaries,pair_summaries):
    """Small reading tables; full distributions and diagnostics remain separate."""
    condition_lookup={tuple(r[k] for k in CONDITION_KEYS):r for r in condition_summaries}
    outputs={'primary_overview':[],'capacity_overview':[],'repeat_overview':[]}
    for key,condition in sorted(_group(rows,CONDITION_KEYS).items(),key=lambda kv:(kv[0][0],kv[0][1],kv[0][3])):
        kind,law,law_id,K=key
        if kind not in ('primary','capacity'):
            continue
        summary=condition_lookup[key]
        def mean(name):
            return _statistics(r.get(name) for r in condition)['mean']
        item={'law_index':law,'law_id':law_id,'K':K,'n_cases':len(condition),
            'n_process_completed':summary['n_process_completed'],
            'n_selection_complete':sum(r.get('selection_selection_complete') is True for r in condition),
            'n_identity_fallback':sum(r.get('selection_identity_fallback') is True for r in condition),
            'n_carrying_repeat':sum(bool(r.get('has_repeat')) for r in condition)}
        total='aggregate_total_from_counts_seconds' if kind=='primary' else 'provider_total_from_counts_seconds'
        for stat in ('n','mean','median','q25','q75','iqr','maximum'):
            item['total_seconds_'+stat]=summary.get('timings_'+total+'_'+stat)
        for stage,label in (('confidence_region_seconds','confidence'),
                ('cell_certification_seconds','cell_certification'),('search_seconds','search')):
            value=summary.get('timings_'+stage+'_mean')
            item[label+'_seconds_mean']=value
            item[label+'_share_pct']=(100*value/item['total_seconds_mean']
                if value is not None and item['total_seconds_mean'] else None)
        stage_candidates={stage:summary.get('timings_'+stage+'_mean') for stage in AGGREGATE_STAGES}
        available={stage:value for stage,value in stage_candidates.items() if value is not None}
        item['dominant_stage']=max(available,key=available.get) if available else None
        for prefix,label in (('memory_peak_rss_bytes','whole_process_peak_rss_MiB'),
                            ('memory_aggregate_pipeline_peak_rss_bytes','before_raw_lifetime_peak_rss_MiB')):
            for stat in ('median','maximum'):
                value=summary.get(prefix+'_'+stat)
                item[label+'_'+stat]=None if value is None else value/2**20
        for metric,label in (('timings_process_wall_seconds','whole_process_wall_seconds'),
            ('timings_additional_raw_reference_seconds','additional_raw_reference_seconds'),
            ('timings_process_overhead_seconds','process_overhead_seconds')):
            for stat in ('mean','maximum'):
                item[label+'_'+stat]=summary.get(metric+'_'+stat)
        serial=[summary.get('timings_'+name+'_mean') for name in
                ('evidence_serialization_seconds','record_serialization_seconds')]
        item['serialization_seconds_mean']=sum(serial) if all(v is not None for v in serial) else None
        item.update(evidence_bytes_mean=mean('evidence_bytes'),
            evidence_bytes_sum=_statistics(r.get('evidence_bytes') for r in condition)['sum'],
            candidate_cells_mean=mean('counters_unique_candidate_cells'),
            anchors_evaluated_mean=mean('counters_anchors_evaluated'),
            mixture_calls_mean=mean('counters_mixture_calls_completed'))
        if kind=='primary':
            item.update(selected_groups_mean=mean('selection_groups'),
                selected_groups_minimum=_statistics(r.get('selection_groups') for r in condition)['minimum'],
                selected_groups_maximum=_statistics(r.get('selection_groups') for r in condition)['maximum'],
                certificate_mean=mean('selection_certificate'),
                aggregate_payload_bytes_mean=mean('inference_payload_bytes'))
        outputs[kind+'_overview'].append(item)
    for summary in sorted(repeat_summaries,key=lambda r:(r['kind'],r['law_index'],r['K'])):
        if summary['kind']!='repeat':
            continue
        condition=[r for r in repeats if r['kind']=='repeat' and r['law_index']==summary['law_index'] and r['K']==summary['K']]
        item={key:summary[key] for key in ('law_index','law_id','K','n_cases','n_process_completed','n_not_reached')}
        item['n_assignment_projection_reused']=sum(r.get('reuse_assignment_projection') is True for r in condition)
        item['n_outcome_root_cache_reused']=sum(r.get('reuse_outcome_root_cache') is True for r in condition)
        for metric,label in (('aggregate_pipeline_seconds','aggregate_seconds'),
            ('inference_total_seconds','all_inference_seconds'),('repeat_wall_seconds','repeat_wall_seconds')):
            for stat in ('n','mean','median','q25','q75','iqr','maximum'):
                item[label+'_'+stat]=summary.get('timings_'+metric+'_'+stat)
        paired=next((r for r in pair_summaries if r['comparison']=='repeated_vs_initial_aggregate_inference'
            and r['law_index']==summary['law_index'] and r['K']==summary['K']),{})
        item.update(initial_aggregate_seconds_mean=paired.get('right_seconds_mean'),
            repeat_minus_initial_aggregate_seconds_mean=paired.get('difference_seconds_mean'),
            repeat_to_initial_aggregate_ratio_median=paired.get('ratio_median'),
            repeat_to_initial_aggregate_ratio_maximum=paired.get('ratio_maximum'))
        outputs['repeat_overview'].append(item)
    return outputs


def summarize(records,primary_reps=20,secondary_reps=5,preflight_reps=1,require_complete=True):
    records = list(records)
    rows = [_normalize(record) for record in records]
    repeats = [row for record in records if (row:=_repeats(record,
        planned=record['kind']=='primary' and record['K'] in REPEAT_SUPPORTS
        and record['replication']<secondary_reps)) is not None]
    if require_complete:
        _validate(rows,repeats,primary_reps,secondary_reps,preflight_reps)
    rows.sort(key=lambda r:(r['kind'],r['law_index'],r['K'],r['replication']))
    condition,stages,counters = _condition_summary(rows)
    status = _status_rows(rows)
    repeat_condition,repeat_stages,repeat_counters = _condition_summary(repeats)
    repeat_status = _status_rows(repeats)
    # Whole-process wall work differs when an additional cohort is executed.
    # Retain these strata instead of portraying them as equal-work latencies.
    process_strata,_,_ = _condition_summary(rows,CONDITION_KEYS+('has_repeat',))
    capacity_selections=_capacity_selections(records)
    capacity_status=_status_rows(capacity_selections,CONDITION_KEYS+('capacity',))
    pairs,pair_summary=_paired_operational(rows,repeats)
    count_maps=_count_map_summary(records)
    accounting=_accounting_rows(rows)
    overviews=_overviews(rows,repeats,condition,repeat_condition,pair_summary)
    summary = {'status':'report_complete' if require_complete else 'fixture_report_complete',
        'primary_cases':sum(r['kind']=='primary' for r in rows),
        'capacity_cases':sum(r['kind']=='capacity' for r in rows),
        'preflight_cases':sum(r['kind']=='preflight' for r in rows),
        'repeated_cohorts':sum(r['kind']=='repeat' for r in repeats),
        'completed_repeated_cohorts':sum(r['kind']=='repeat' and r['status']=='completed' for r in repeats),
        'not_reached_repeated_cohorts':sum(r['kind']=='repeat' and r['status']=='not_reached' for r in repeats),
        'independent_calibration_primary_datasets':sum(r['kind']=='primary' for r in rows),
        'condition_count':len(condition),'repeat_condition_count':len(repeat_condition),
        'case_status_counts':{kind:dict(Counter(r['status'] for r in rows if r['kind']==kind))
                              for kind in ('primary','capacity','preflight')},
        'input_boundary':'saved finite-state calibration/outcome sufficient counts; input generation and raw preprocessing outside timers',
        'aggregate_boundary':'aggregate_total_from_counts_seconds excludes raw references, repeats, serialization and process startup',
        'process_boundary':'process_wall_seconds includes aggregate, additional raw references, any repeat, serialization, startup and shutdown',
        'memory_boundary':'fresh process lifetime high-water mark; includes additional raw work/repeat; no stage-memory subtraction',
        'repeat_interpretation':'dependent amortized costs with assignment law and selected partition retained; not additional independent calibrations',
        'censoring':'timeouts and memory limits retained with observed elapsed work; completed-latency summaries have separate denominators',
        'statistics':'descriptive case means and median/IQR/maximum; no heterogeneous pooled success rate',
        'accounting_checks':{field:dict(Counter(str(row.get(field)) for row in accounting))
            for field in ('aggregate_accounting_complete','process_accounting_complete')},
        'timing_accounting':{'aggregate':'sum aggregate_accounting_member stage means using the same observed cases',
            'process':'sum process_accounting_member stage means; inclusive totals excluded; derived inprocess remainder includes work not individually allocated',
            'negative_remainders':'retained as instrumentation diagnostics, never clipped to zero'},
        'tables':{'cases':'tables/cases.csv','condition_summary':'tables/condition_summary.csv',
            'stage_summary':'tables/stage_summary.csv','work_counters':'tables/work_counters_summary.csv',
            'status_denominators':'tables/status_denominators.csv','repeat_cases':'tables/repeat_cases.csv',
            'repeat_summary':'tables/repeat_summary.csv','repeat_stage_summary':'tables/repeat_stage_summary.csv',
            'repeat_status':'tables/repeat_status_denominators.csv','process_wall_strata':'tables/process_wall_strata.csv',
            'capacity_selections':'tables/capacity_selections.csv','capacity_status':'tables/capacity_status_denominators.csv',
            'fallback_reason_solver_status':'tables/fallback_reason_solver_status.csv',
            'timing_accounting':'tables/timing_accounting.csv',
            'status_definitions':'tables/numerical_status_definitions.csv',
            'primary_overview':'tables/primary_overview.csv',
            'capacity_overview':'tables/capacity_overview.csv',
            'repeat_overview':'tables/repeat_overview.csv',
            'paired_costs':'tables/paired_operational_costs.csv','paired_cost_summary':'tables/paired_operational_summary.csv'}}
    tables={'cases':rows,'condition_summary':condition,
        'stage_summary':stages,'work_counters_summary':counters,'status_denominators':status,
        'repeat_cases':repeats,'repeat_summary':repeat_condition,'repeat_stage_summary':repeat_stages,
        'repeat_work_counters_summary':repeat_counters,'repeat_status_denominators':repeat_status,
        'process_wall_strata':process_strata,'capacity_selections':capacity_selections,
        'capacity_status_denominators':capacity_status,'paired_operational_costs':pairs,
        'fallback_reason_solver_status':count_maps,
        'timing_accounting':accounting,
        'paired_operational_summary':pair_summary,
        'numerical_status_definitions':[dict(zip(('domain','status','definition','proof_scope'),entry))
                                        for entry in STATUS_DEFINITIONS]}
    tables.update(overviews)
    return {'summary':summary,'tables':tables}




def _csv(path,rows):
    rows = list(rows);columns = list(dict.fromkeys(k for r in rows for k in r))
    fractions = {k for r in rows for k,v in r.items() if isinstance(v,F)}
    columns += [key+'_float' for key in columns if key in fractions and key+'_float' not in columns]
    path = Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:
            row=dict(row)
            for key in fractions:
                row.setdefault(key+'_float',_number(row.get(key)))
            writer.writerow({k:('NA' if v is None else str(v) if isinstance(v,F)
                else json.dumps(encode(v),sort_keys=True) if isinstance(v,(dict,list,tuple)) else v)
                for k,v in row.items()})
    temporary.replace(path)


def _load(path):
    if Path(path).suffix=='.gz':
        with gzip.open(path,'rt') as stream:
            return decode(json.load(stream))
    return decode(json.loads(Path(path).read_text()))


def generate(root,primary_reps=20,secondary_reps=5,preflight_reps=1,figures=True,require_complete=True):
    root=Path(root);records=[];tick=time.perf_counter()
    for kind in ('primary','capacity','preflight'):
        for path in sorted((root/kind/'records').glob('*.json.gz')):
            record=_load(path)
            if record.get('kind',kind)!=kind:
                raise AssertionError('Case kind differs from its result directory')
            record['kind']=kind
            record['evidence_file']=str(path.relative_to(root))
            records.append(record)
    result=summarize(records,primary_reps,secondary_reps,preflight_reps,require_complete=require_complete)
    for name in ('manifest.json','execution.json','configuration.json'):
        path=root/name
        if path.exists():
            metadata=_load(path)
            if name=='manifest.json':
                result['summary']['environment']=metadata.get('environment')
            else:
                result['summary'][name.removesuffix('.json')]=metadata
    folder=root/'tables';folder.mkdir(exist_ok=True)
    for name,rows in result['tables'].items():
        _csv(folder/(name+'.csv'),rows)
    if figures:
        pass  # Public release generates numerical tables only.
    result['summary']['reporting_seconds']=time.perf_counter()-tick
    (root/'summary.json').write_text(json.dumps(encode(result['summary']),indent=2)+'\n')
    return result['summary']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'results/jmlr_computation_v1')
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    summary=generate(args.root,figures=not args.no_figures)
    print(json.dumps({k:summary[k] for k in ('status','primary_cases','capacity_cases','repeated_cohorts','preflight_cases')}))


if __name__=='__main__':
    main()
