"""Derived numerical tables for the frozen split-search ablation.

Only completed saved records are read.  This module never draws observations,
constructs confidence regions, tests proposals, or selects a partition.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction as F
import math
import os
from pathlib import Path

import numpy as np

from experiments import jmlr_statistical_baselines_v1_reporting as saved_report
from experiments.jmlr_verified_adaptive_v1_roots import rational
from experiments.jmlr_verified_adaptive_v1_math import validate_partition


VERSION = 'jmlr_split_ablation_v1'
PROCEDURES = tuple('ABCDEF')
CAPS = (F(5,4), F(3,2), F(2))
CONTRASTS = (('C','B'),('D','C'),('E','D'),('F','E'),
             ('C','A'),('F','C'),('F','A'),('B','A'))
META = ('id','K','mass','overlap','geometry','r','n','population_index','r_index','replication')
CONDITION_META = tuple(key for key in META if key != 'replication')
NAMES = {
    'A':'Split LTT; first failure', 'B':'Split coupled; first failure',
    'C':'Split coupled; all proposals', 'D':'Full coupled; pilot proposals',
    'E':'Full coupled; complete pilot-order family',
    'F':'Full coupled; complete full-data-order family',
}
CONTINUOUS = ('cells','compression_pp','true_distortion','value')
BINARY = ('identity','strict_cap_violation','count_optimal','identity_fallback',
          'deterministic_tie_complete','computation_complete','value_infinite','value_unresolved')
STOP_BINARY = ('later_proposal_passes','later_passing_fewer_cells','stopped')
STOP_CONTINUOUS = ('count_gain','compression_gain_pp')
PROPOSAL_STATUSES = ('finite','infinite','rigorous_above_cap','unresolved')


def _q(value):
    return rational(value)


def _condition_key(meta):
    return str(meta['id'])+'_r'+str(int(meta['r']))


def _numeric(value):
    if isinstance(value,bool):
        return int(value)
    try:
        result = float(value)
    except (TypeError,ValueError,OverflowError):
        return None
    return result if math.isfinite(result) else None


def _prefix(prefix, result):
    return {prefix+'_'+key:value for key,value in result.items()}


def _continuous(values):
    source = list(values)
    finite = [_numeric(value) for value in source]
    result = saved_report.paired([value for value in finite if value is not None])
    result.update(n_missing_or_nonfinite=sum(value is None for value in finite), n_total=len(source))
    return result


def _binary_moments(values):
    """Moments for the fixed-grid variance; per-condition binary CIs are CP."""
    result=saved_report.paired(list(map(int,values)))
    return {key:result[key] for key in ('n','mean','sample_sd')}


def _grid(rows, prefix, expected):
    result = saved_report.grid_interval(rows,prefix,expected_conditions=expected)
    # A partial panel does not redefine the prespecified equal-weight grid.
    if not result.get('all_conditions_present',False):
        result['available_condition_mean'] = result.get('mean')
        for key in ('mean','MCSE','analytic_Normal95_lower','analytic_Normal95_upper'):
            result[key] = None
        result['scope'] = 'prespecified fixed grid incomplete; no renormalized complete-grid estimate'
    return result


def _complete(row):
    if row.get('identity_fallback',False) or not row.get('deterministic_tie_complete',True):
        return False
    if 'computation_complete' in row:
        return bool(row['computation_complete'])
    status = str(row.get('computation_status','')).lower()
    return not row.get('identity_fallback',False) and not any(
        token in status for token in ('unresolved','incomplete','failure','error','fallback'))


def _normalize_row(row, metadata):
    result = dict(metadata)
    result.update(row.get('metadata',{}))
    result.update({key:value for key,value in row.items() if key != 'metadata'})
    for key in META:
        if key not in result:
            raise ValueError('Missing row metadata: '+key)
        if key in metadata and result[key] != metadata[key]:
            raise ValueError('Row and block metadata disagree: '+key)
    for key in ('K','r','n','population_index','r_index','replication','cells'):
        value = result[key]
        if isinstance(value,bool) or int(value) != value:
            raise ValueError('Expected integer metadata/count: '+key)
        result[key] = int(value)
    if result['procedure'] not in PROCEDURES:
        raise ValueError('Unknown procedure')
    result['cap'] = _q(result['cap'])
    if result['cap'] not in CAPS:
        raise ValueError('Uncommitted cap')
    result['partition'] = validate_partition(result['partition'],result['K'])
    if not 1 <= result['cells'] <= result['K'] or result['cells'] != len(result['partition']):
        raise ValueError('Returned count/partition disagree')
    expected = 100*F(result['K']-result['cells'],result['K'])
    result['compression_pp'] = _q(result['compression_pp'])
    if result['compression_pp'] != expected:
        raise ValueError('Compression does not match returned count')
    result['identity'] = bool(result['identity'])
    if result['identity'] != (result['cells']==result['K']):
        raise ValueError('Identity flag disagrees with partition')
    result['true_distortion'] = _q(result['true_distortion'])
    if bool(result['strict_cap_violation']) != (result['true_distortion']>result['cap']):
        raise ValueError('Strict cap violation is not the exact population predicate')
    if result.get('value') is not None:
        result['value'] = _q(result['value'])
    value_status = result.get('value_status')
    if value_status is None:
        if result['value'] is not None:
            value_status = 'finite'
        elif 'infinite' in str(result.get('computation_status','')).lower():
            value_status = 'infinite'
        else:
            value_status = 'unresolved'
    if value_status not in ('finite','infinite','unresolved'):
        raise ValueError('Unknown value completion status')
    if (value_status=='finite') != (result['value'] is not None):
        raise ValueError('Finite value status disagrees with exact value')
    result['value_status'] = value_status
    result['value_infinite'] = value_status=='infinite'
    result['value_unresolved'] = value_status=='unresolved'
    expected_kind = 'ltt_distortion_statistic' if result['procedure']=='A' else 'coupled_certificate'
    if result['value_kind'] != expected_kind:
        raise ValueError('LTT statistic and coupled certificate labels differ')
    result['identity_fallback'] = bool(result.get('identity_fallback',False))
    if result['identity_fallback'] and not result['identity']:
        raise ValueError('A safe identity fallback must return identity')
    result['count_optimal'] = bool(result.get('count_optimal',False))
    result['deterministic_tie_complete'] = bool(result.get('deterministic_tie_complete',False))
    result['computation_complete'] = _complete(result)
    result['condition_id'] = _condition_key(result)
    return result


def _condition_summary(rows, metadata, procedure, cap, expected_replications):
    result = {**metadata,'procedure':procedure,'procedure_name':NAMES[procedure],'cap':cap,
              'value_kind':'ltt_distortion_statistic' if procedure=='A' else 'coupled_certificate',
              'n_datasets':len(rows),'n_expected':expected_replications,
              'n_missing_datasets':expected_replications-len(rows)}
    for metric in CONTINUOUS:
        result.update(_prefix(metric,_continuous(row.get(metric) for row in rows)))
    for metric in BINARY:
        result.update(_prefix(metric,saved_report.rate([row[metric] for row in rows])))
        result.update(_prefix(metric+'_mean',_binary_moments([row[metric] for row in rows])))
    result['computation_status_counts'] = dict(Counter(row['computation_status'] for row in rows))
    result['value_status_counts'] = dict(Counter(row['value_status'] for row in rows))
    result['selection_scopes'] = sorted(set(str(row.get('selection_scope','')) for row in rows))
    return result


def _flatten_numeric(value, prefix=''):
    result = {}
    if isinstance(value,dict):
        for key,item in value.items():
            result.update(_flatten_numeric(item,prefix+'_'+str(key) if prefix else str(key)))
    elif isinstance(value,(int,float,F)) and not isinstance(value,bool) and _numeric(value) is not None:
        result[prefix] = value
    return result


def _proposal_counts_with_zeros(value, in_procedure=False):
    """Expand provided C/D Counters, preserving unavailable diagnostics.

    A saved Counter omits statuses observed zero times.  Only an actually
    supplied counter establishes those zeros; a missing method, cap or whole
    diagnostic remains missing. Cap decisions may nest caps above or below
    the C/D method key.
    """
    if not isinstance(value,dict):
        return value
    if in_procedure and all(not isinstance(item,dict) for item in value.values()):
        return {**dict.fromkeys(PROPOSAL_STATUSES,0),**value}
    return {key:_proposal_counts_with_zeros(item,in_procedure or key in ('C','D'))
            for key,item in value.items()}


def summarize(records, expected_conditions=48, expected_replications=90):
    """Summarize the fixed grid with within-condition paired MC uncertainty.

    Returns (summary, tables). An incomplete input remains explicitly
    incomplete; missing conditions are never silently reweighted away.
    """
    rows, stopping, cost, diagnostic_rows = [],[],[],[]
    # Keep only dataset keys. Provider evidence in a block can be much larger
    # than its 18 analysis rows and is not needed after those rows are read.
    blocks, conditions = set(),{}
    expected_reps = set(range(10,10+expected_replications))
    for record in records:
        metadata = dict(record['metadata'])
        key = (_condition_key(metadata),int(metadata['replication']))
        if key in blocks:
            raise ValueError('Duplicate dataset block: '+str(key))
        if key[1] not in expected_reps:
            raise ValueError('Development or uncommitted replication in final report')
        blocks.add(key)
        meta = {field:metadata[field] for field in CONDITION_META}
        meta['condition_id'] = key[0]
        if key[0] in conditions and conditions[key[0]] != meta:
            raise ValueError('Condition metadata changed between datasets')
        conditions[key[0]] = meta
        rows.extend(_normalize_row(row,metadata) for row in record['rows'])
        pilot=record.get('pilot_infinite_proposals')
        if pilot is not None and (isinstance(pilot,bool) or int(pilot)!=pilot or pilot<0):
            raise ValueError('Pilot infinite-score count must be nonnegative or unavailable')
        proposal_counts={family:_proposal_counts_with_zeros(record.get(family))
                         for family in ('proposal_status_counts','proposal_decision_status_counts')}
        diagnostic_rows.append({**metadata,'condition_id':key[0],
            'pilot_infinite_proposals':None if pilot is None else int(pilot),
            'pilot_has_infinite_score':None if pilot is None else bool(pilot>0),
            **proposal_counts})
        seen_caps = set()
        for item in record.get('stopping',[]):
            entry = {**metadata,**item,'condition_id':key[0],'cap':_q(item['cap'])}
            if entry['cap'] not in CAPS or entry['cap'] in seen_caps:
                raise ValueError('Duplicate/uncommitted stopping cap')
            seen_caps.add(entry['cap'])
            for field in ('later_proposal_passes','later_passing_fewer_cells'):
                entry[field] = bool(entry[field])
            if entry['later_passing_fewer_cells'] and not entry['later_proposal_passes']:
                raise ValueError('A later coarser pass requires a later pass')
            entry['stopped'] = entry.get('stop_index') is not None
            entry['count_gain'] = int(entry['count_gain'])
            entry['compression_gain_pp'] = _q(entry['compression_gain_pp'])
            stopping.append(entry)
        metrics = {}
        for family in ('timings','cache','cache_counts','cache_metrics','diagnostics','root_fallbacks',
                       'proposal_status_counts','proposal_decision_status_counts'):
            value=proposal_counts[family] if family in proposal_counts else record.get(family,{})
            metrics.update(_flatten_numeric(value,family))
        for metric,value in sorted(metrics.items()):
            cost.append({**metadata,'condition_id':key[0],'metric':metric,'value':value})

    indexed = {}
    by_method = defaultdict(list)
    for row in rows:
        key = (row['condition_id'],row['replication'],row['cap'],row['procedure'])
        if key in indexed:
            raise ValueError('Duplicate procedure/cap/dataset row')
        indexed[key] = row
        by_method[row['condition_id'],row['procedure'],row['cap']].append(row)
    accounting_checked = structural_CB = structural_ED = 0
    paired_rows = []
    paired_groups = defaultdict(list)
    for condition,replication in sorted(blocks):
        for cap in CAPS:
            selected = {p:indexed.get((condition,replication,cap,p)) for p in PROCEDURES}
            if not all(selected.values()):
                continue
            q = {p:row['compression_pp'] for p,row in selected.items()}
            if q['F']-q['A'] != sum((q[a]-q[b] for a,b in (('B','A'),('C','B'),('D','C'),('E','D'),('F','E'))),F()):
                raise ArithmeticError('Paired accounting identity failed')
            accounting_checked += 1
            for left,right,label in (('C','B','CB'),('E','D','ED')):
                if selected[left]['count_optimal'] and selected[left]['computation_complete'] and selected[right]['computation_complete']:
                    if selected[left]['cells']>selected[right]['cells']:
                        raise ArithmeticError('Structural count inequality failed: '+label)
                    if label=='CB':structural_CB += 1
                    else:structural_ED += 1
            entry = {**{field:selected['A'][field] for field in META},'condition_id':condition,'cap':cap,
                     'accounting_identity_exact':True}
            for left,right in CONTRASTS:
                label=left+'-'+right
                difference=q[left]-q[right]
                entry[label+'_pp']=difference
                paired_groups[condition,cap,label].append((replication,difference))
            paired_rows.append(entry)

    stop_index = {(x['condition_id'],x['replication'],x['cap']):x for x in stopping}
    if len(stop_index)!=len(stopping):raise ValueError('Duplicate stopping dataset record')
    for key,entry in stop_index.items():
        condition,replication,cap=key
        B=indexed.get((condition,replication,cap,'B'));C=indexed.get((condition,replication,cap,'C'))
        if B is not None and C is not None:
            if entry['count_gain']!=B['cells']-C['cells'] or entry['compression_gain_pp']!=C['compression_pp']-B['compression_pp']:
                raise ArithmeticError('Stopping improvement disagrees with B/C outputs')
            if C['count_optimal'] and C['computation_complete'] and B['computation_complete']:
                if entry['later_passing_fewer_cells'] != (entry['count_gain']>0):
                    raise ArithmeticError('Later passing improvement and complete all-proposal optimum disagree')

    method_conditions=[];paired_conditions=[];stopping_conditions=[]
    for condition,meta in sorted(conditions.items()):
        for cap in CAPS:
            for procedure in PROCEDURES:
                method_conditions.append(_condition_summary(by_method[condition,procedure,cap],meta,procedure,cap,expected_replications))
            for left,right in CONTRASTS:
                label=left+'-'+right
                values=[value for _,value in paired_groups[condition,cap,label]]
                item={**meta,'cap':cap,'contrast':label,'left':left,'right':right,
                      'direction':'left minus right; compression percentage points',
                      **_prefix('difference_pp',saved_report.paired(list(map(float,values)))),
                      'wins':sum(v>0 for v in values),'ties':sum(v==0 for v in values),'losses':sum(v<0 for v in values),
                      'n_expected':expected_replications,'n_missing_pairs':expected_replications-len(values)}
                paired_conditions.append(item)
            ss=[x for x in stopping if x['condition_id']==condition and x['cap']==cap]
            item={**meta,'cap':cap,'n_datasets':len(ss),'n_expected':expected_replications,
                  'n_missing_datasets':expected_replications-len(ss),
                  'stop_reason_counts':dict(Counter(str(x.get('stop_reason')) for x in ss))}
            for metric in STOP_BINARY:
                item.update(_prefix(metric,saved_report.rate([x[metric] for x in ss])))
                item.update(_prefix(metric+'_mean',_binary_moments([x[metric] for x in ss])))
            for metric in STOP_CONTINUOUS:item.update(_prefix(metric,_continuous(x[metric] for x in ss)))
            stopping_conditions.append(item)

    regimes = (250,1000,4000) if expected_conditions==48 else tuple(sorted({m['r'] for m in conditions.values()}))
    scopes=[('all',None,expected_conditions)]+[('r'+str(r),r,16 if expected_conditions==48 else sum(m['r']==r for m in conditions.values())) for r in regimes]
    method_grid=[];paired_grid=[];stopping_grid=[]
    for scope,r,H in scopes:
        for cap in CAPS:
            for procedure in PROCEDURES:
                cc=[x for x in method_conditions if x['cap']==cap and x['procedure']==procedure and (r is None or x['r']==r)]
                item={'scope':scope,'r':r,'cap':cap,'procedure':procedure,'procedure_name':NAMES[procedure],
                      'value_kind':'ltt_distortion_statistic' if procedure=='A' else 'coupled_certificate'}
                for metric in CONTINUOUS:item.update(_prefix(metric,_grid(cc,metric,H)))
                for metric in BINARY:
                    item.update(_prefix(metric,_grid(cc,metric+'_mean',H)))
                    item[metric+'_dataset_successes']=sum(x[metric+'_successes'] for x in cc)
                    item[metric+'_dataset_trials']=sum(x[metric+'_n'] for x in cc)
                method_grid.append(item)
            for left,right in CONTRASTS:
                label=left+'-'+right
                cc=[x for x in paired_conditions if x['cap']==cap and x['contrast']==label and (r is None or x['r']==r)]
                item={'scope':scope,'r':r,'cap':cap,'contrast':label,'left':left,'right':right,
                      **_prefix('difference_pp',_grid(cc,'difference_pp',H)),
                      'wins':sum(x['wins'] for x in cc),'ties':sum(x['ties'] for x in cc),'losses':sum(x['losses'] for x in cc),
                      'conditions_mean_positive':sum(x['difference_pp_mean'] is not None and x['difference_pp_mean']>0 for x in cc),
                      'conditions_mean_zero':sum(x['difference_pp_mean']==0 for x in cc),
                      'conditions_mean_negative':sum(x['difference_pp_mean'] is not None and x['difference_pp_mean']<0 for x in cc),
                      'count_scope':'descriptive dataset/condition counts; no pooled-binomial interval'}
                paired_grid.append(item)
            cc=[x for x in stopping_conditions if x['cap']==cap and (r is None or x['r']==r)]
            item={'scope':scope,'r':r,'cap':cap}
            for metric in STOP_BINARY:item.update(_prefix(metric,_grid(cc,metric+'_mean',H)))
            for metric in STOP_CONTINUOUS:item.update(_prefix(metric,_grid(cc,metric,H)))
            stopping_grid.append(item)

    cost_groups=defaultdict(list)
    for entry in cost:cost_groups[entry['condition_id'],entry['metric']].append(entry['value'])
    cost_conditions=[{**conditions[condition],'metric':metric,**_prefix('value',_continuous(values))}
                     for (condition,metric),values in sorted(cost_groups.items())]
    cost_grid=[]
    for scope,r,H in scopes:
        for metric in sorted({entry['metric'] for entry in cost}):
            cc=[x for x in cost_conditions if x['metric']==metric and (r is None or x['r']==r)]
            cost_grid.append({'scope':scope,'r':r,'metric':metric,**_prefix('value',_grid(cc,'value',H)),
                              'interpretation':'incremental/shared/cache-assisted execution; no cold end-to-end procedure speed comparison'})
    diagnostic_conditions=[]
    for condition,meta in sorted(conditions.items()):
        available=[x for x in diagnostic_rows if x['condition_id']==condition and x['pilot_has_infinite_score'] is not None]
        item={**meta,'n_pilot_score_diagnostic_available':len(available),
              'n_pilot_score_diagnostic_unavailable':sum(x['condition_id']==condition for x in diagnostic_rows)-len(available),
              **_prefix('pilot_has_infinite_score',saved_report.rate([x['pilot_has_infinite_score'] for x in available])),
              **_prefix('pilot_has_infinite_score_mean',_binary_moments([x['pilot_has_infinite_score'] for x in available])),
              **_prefix('pilot_infinite_proposals',_continuous(x['pilot_infinite_proposals'] for x in available))}
        diagnostic_conditions.append(item)
    diagnostic_grid=[]
    for scope,r,H in scopes:
        cc=[x for x in diagnostic_conditions if r is None or x['r']==r]
        diagnostic_grid.append({'scope':scope,'r':r,
            **_prefix('pilot_has_infinite_score',_grid(cc,'pilot_has_infinite_score_mean',H)),
            **_prefix('pilot_infinite_proposals',_grid(cc,'pilot_infinite_proposals',H))})
    expected_blocks=expected_conditions*expected_replications
    expected_rows=expected_blocks*len(PROCEDURES)*len(CAPS)
    missing_reps={condition:sorted(expected_reps-{rep for c,rep in blocks if c==condition}) for condition in conditions}
    missing_rows={condition+':rep'+str(rep):[p+':'+str(cap) for p in PROCEDURES for cap in CAPS
                  if (condition,rep,cap,p) not in indexed]
                  for condition,rep in sorted(blocks)}
    complete=(len(conditions)==expected_conditions and len(blocks)==expected_blocks and len(rows)==expected_rows
              and all(not values for values in missing_reps.values()) and all(not values for values in missing_rows.values())
              and len(paired_rows)==expected_blocks*len(CAPS) and len(stopping)==expected_blocks*len(CAPS))
    summary={'version':VERSION,'status':'complete' if complete else 'incomplete','complete':complete,
        'independent_datasets':len(blocks),'expected_datasets':expected_blocks,
        'conditions':len(conditions),'expected_conditions':expected_conditions,
        'replications_per_condition':expected_replications,'evaluation_replications':[10,9+expected_replications],
        'procedure_cap_rows':len(rows),'expected_procedure_cap_rows':expected_rows,
        'paired_dataset_cap_rows':len(paired_rows),'stopping_dataset_cap_rows':len(stopping),
        'all_returned_computations_complete':all(row['computation_complete'] for row in rows),
        'count_optimal_records':sum(row['count_optimal'] for row in rows),
        'identity_fallback_records':sum(row['identity_fallback'] for row in rows),
        'missing_replications_by_condition':missing_reps,
        'missing_procedure_cap_rows_by_dataset':{k:v for k,v in missing_rows.items() if v},
        'checks':{'accounting_identity_dataset_caps':accounting_checked,'structural_C_le_B_checks':structural_CB,
                  'structural_E_le_D_checks':structural_ED,'accounting_violations':0,'structural_violations':0},
        'computation_status_counts':{p:dict(Counter(x['computation_status'] for x in rows if x['procedure']==p)) for p in PROCEDURES},
        'value_status_counts':{p:dict(Counter(x['value_status'] for x in rows if x['procedure']==p)) for p in PROCEDURES},
        'methods_grid':method_grid,'paired_grid':paired_grid,'stopping_grid':stopping_grid,
        'diagnostic_grid':diagnostic_grid,
        'uncertainty':{'condition_continuous':'pointwise Student95 over independent datasets within each fixed condition',
                      'condition_binary':'pointwise two-sided Clopper-Pearson95 with actual independent dataset denominator',
                      'grid':'equal fixed-condition weights; variance H^-2 sum_h s_h^2/n_h; approximate normal95',
                      'pairing':'within-dataset differences; no replication-index clustering across conditions',
                      'scope':'Monte Carlo uncertainty for a retrospective matched fixed-grid ablation, distinct from each procedure confidence guarantee'},
        'timing_scope':'shared inputs, saved references, incremental calculations and cache reuse; no cold end-to-end speed ranking'}
    tables={'dataset_rows':rows,'method_conditions':method_conditions,'methods_grid':method_grid,
            'paired_dataset_rows':paired_rows,'paired_conditions':paired_conditions,'paired_grid':paired_grid,
            'stopping_rows':stopping,'stopping_conditions':stopping_conditions,'stopping_grid':stopping_grid,
            'cost_rows':cost,'cost_conditions':cost_conditions,'cost_grid':cost_grid,
            'diagnostic_rows':diagnostic_rows,'diagnostic_conditions':diagnostic_conditions,'diagnostic_grid':diagnostic_grid}
    return summary,tables




def generate(root, expected_conditions=48, expected_replications=90):
    """Regenerate only derived outputs from immutable saved block records."""
    root=Path(root)
    paths=sorted((root/'panel').rglob('rep*.json.gz'))
    summary,tables=summarize((saved_report.read_gzip(path) for path in paths),expected_conditions,expected_replications)
    for name,rows in tables.items():saved_report.write_csv(rows,root/'tables'/(name+'.csv'))
    summary['figures']=[]
    summary['saved_blocks_read']=len(paths)
    saved_report.dump(summary,root/'summary.json')
    return summary
