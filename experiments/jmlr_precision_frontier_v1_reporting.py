"""Derived evidence for the prespecified compression--distortion--width study.

Only saved block records are inputs.  Individual binary rates use two-sided
95% CP intervals; means and paired differences use block-level normal Monte
Carlo intervals.  No interval in this module asserts simultaneous coverage of
the outcome frontier or nominal coverage conditional on reporting admission.
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
import time

import numpy as np

from experiments.jmlr_release_capacity_v1_reporting import _statistics
from experiments.jmlr_verified_adaptive_v1_math import decode, encode


CAPS = (F(1), F(11,10), F(5,4), F(3,2), F(2))
WIDTH_TARGET = F(1,5)
METHODS = ('z_global', 'z_cellwise', 'raw_stratified', 'raw_pooled')
METHOD_LABELS = {
    'z_global':'Global aggregate', 'z_cellwise':'Cellwise aggregate',
    'raw_stratified':'Shared raw stratification',
    'raw_pooled':'Pooled raw (LP + intersection)'}
AGGREGATE_METHODS = ('z_global', 'z_cellwise')
COMPONENTS = ('population_aggregation', 'certificate_budget',
              'outcome_sampling', 'assignment_estimation')
COMPONENT_LABELS = ('Population aggregation + transport', 'Certificate budget',
                    'Outcome sampling', 'Assignment estimation')
SEL_KEYS = ('law_index','law_id','K','n','capacity_limit','cap')
ANALYSIS_KEYS = ('law_index','law_id','K','capacity_limit','profile','m_ratio','m','cap')
POP_KEYS = ('law_index','law_id','K','capacity_limit','profile','cap')
SELECTION_CONTINUOUS = (
    'groups','compression_pp','certificate','true_distortion',
    'transported_budget','budget_ceiling','budget_slack',
    'certificate_slack','distortion_certificate_gap')
SELECTION_BINARY = (
    'selection_complete','count_optimal','deterministic_tie_complete','identity',
    'identity_fallback','admitted','abstained','calibration_covered',
    'calibration_miss','strict_distortion_cap_violation','certificate_below_truth',
    'source_admission_match')
ANALYSIS_BINARY = (
    'admitted','abstained','selection_complete','calibration_covered',
    'calibration_miss','raw_outcomes_covered','aggregate_outcomes_covered',
    'raw_outcome_miss','aggregate_outcome_miss','shared_coverage_event',
    'global_nesting','cellwise_nesting','global_nesting_failure_on_shared_event',
    'cellwise_nesting_failure_on_shared_event','raw_population_exceeds_target',
    'LP_fallback','empty_pooled_intersection','aggregate_root_fallback',
    'raw_root_fallback','aggregate_only_equal',
    'cellwise_contained_in_global')
ANALYSIS_CONTINUOUS = (
    'groups','compression_pp','certificate','true_distortion','transported_budget',
    'budget_ceiling','aggregate_root_fallbacks','raw_root_fallbacks',
    'zero_treated_groups','analyst_seconds','payload_bytes')
METHOD_BINARY = ('miss','fallback','numerical_complete','target',
                 'admitted_target','admitted_and_miss')
METHOD_CONTINUOUS = ('width',)
DECOMPOSITION_METRICS = tuple(f'I{j}_width' for j in range(5)) + COMPONENTS + ('total_excess',)
POP_METRICS = tuple(f'{transport}_I{j}_width'
                    for transport in ('global','cellwise') for j in range(3))
QUANTILE_NAMES = ('minimum','q05','q25','median','q75','q95','maximum')
INFERENCE_STATISTICAL_COUNTERS = (
    'complete_hits','complete_misses','raw_hits','raw_misses',
    'aggregate_hits','aggregate_misses','lp_hits','lp_misses',
    'lp_solves','lp_saved_reuses')


def _number(value):
    if value is None or not isinstance(value,(bool,int,float,F)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _stats(values, binary=False):
    """Explicit availability counts; negative diagnostic differences retained."""
    values = list(values)
    answer = _statistics(values, probability=binary)
    finite = [_number(value) for value in values if _number(value) is not None]
    answer.update(n_records=len(values),
                  events=sum(int(v) for v in finite) if binary else None,
                  n_negative=sum(v<0 for v in finite),
                  n_zero=sum(v==0 for v in finite),
                  n_positive=sum(v>0 for v in finite))
    answer.update({name:None for name in QUANTILE_NAMES})
    if finite:
        answer.update(zip(QUANTILE_NAMES,map(float,np.quantile(
            finite,[0,.05,.25,.5,.75,.95,1]))))
    return answer


def _summarize_group(rows, continuous=(), binary=(), expected_blocks=100):
    rows = list(rows)
    replications = [row['replication'] for row in rows]
    if len(set(replications)) != len(replications):
        raise AssertionError('Repeated independent block in one condition')
    if len(rows)>expected_blocks:
        raise AssertionError('More independent blocks than declared')
    answer = {'n_blocks':len(rows),'n_expected_blocks':expected_blocks,
              'n_missing_blocks':expected_blocks-len(rows)}
    for name in continuous+binary:
        stats = _stats([row.get(name) for row in rows], name in binary)
        answer.update({f'{name}_{key}':value for key,value in stats.items()})
    return answer


def _paired(left, right, left_metric, right_metric=None, expected_blocks=100):
    """Left minus right, paired by independent block; never binomial CP."""
    right_metric = right_metric or left_metric
    left, right = list(left), list(right)
    if any(len({r['replication'] for r in rows}) != len(rows) for rows in (left,right)):
        raise AssertionError('Repeated independent block in paired contrast')
    left = {r['replication']:r for r in left}
    right = {r['replication']:r for r in right}
    common = sorted(left.keys() & right.keys())
    differences, admitted_differences = [], []
    for replication in common:
        a, b = _number(left[replication].get(left_metric)), _number(right[replication].get(right_metric))
        if a is None or b is None:
            continue
        # Preserve exact subtraction before display conversion when inputs are
        # rational.  In particular, identical rational differences have zero
        # Monte Carlo variance rather than cancellation noise.
        difference = left[replication][left_metric]-right[replication][right_metric]
        differences.append(difference)
        if left[replication].get('admitted') and right[replication].get('admitted'):
            admitted_differences.append(difference)
    answer = {'n_expected_blocks':expected_blocks,'n_matching_blocks':len(common),
              'n_unavailable_pairs':expected_blocks-len(differences),
              'left_metric':left_metric,'right_metric':right_metric,
              'direction':'left minus right',
              'uncertainty':'paired block mean difference; pointwise normal Monte Carlo interval',
              'both_admitted_scope':'descriptive; admission subsets can change with cap'}
    answer.update({f'difference_{key}':value for key,value in _stats(differences).items()})
    answer.update({f'both_admitted_difference_{key}':value
                   for key,value in _stats(admitted_differences).items()})
    return answer


def _group(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def _json_counts(values):
    return json.dumps(dict(sorted(Counter(str(value) for value in values).items())),sort_keys=True)


def _flatten(value, prefix=''):
    result = {}
    for key,item in value.items():
        name = prefix+str(key)
        if isinstance(item,dict):
            result.update(_flatten(item,name+'_'))
        elif item is None or isinstance(item,(str,bool,int,float,F)):
            result[name] = item
        elif isinstance(item,(list,tuple)):
            result[name+'_json'] = json.dumps(encode(item),sort_keys=True)
    return result


def _computation_cache(cache):
    """Expand omitted Counter zeros only inside a provided inference cache.

    Counter serialization omits never-incremented statistical counters.  Their
    absence is a measured zero, whereas an absent cache, unknown field, timing,
    or explicitly missing value remains unobserved.  Never mutate saved input.
    """
    if not isinstance(cache,dict) or not isinstance(cache.get('inference'),dict):
        return cache
    inference = dict.fromkeys(INFERENCE_STATISTICAL_COUNTERS,0)
    inference.update(cache['inference'])
    return dict(cache,inference=inference)


def _selection(row):
    row = dict(row)
    row['cap'] = F(row['cap'])
    if row['cap'] not in CAPS:
        raise AssertionError('Uncommitted distortion cap')
    if row.get('compression_pp') is None:
        row['compression_pp'] = F(100)*(1-F(row['groups'],row['K']))
    if row['compression_pp'] != F(100)*(1-F(row['groups'],row['K'])):
        raise AssertionError('Compression must be percentage points of states removed')
    for name, a, b in (
            ('budget_slack','budget_ceiling','transported_budget'),
            ('certificate_slack','cap','certificate'),
            ('distortion_certificate_gap','certificate','true_distortion')):
        row[name] = None if row.get(a) is None or row.get(b) is None else row[a]-row[b]
    row['calibration_miss'] = (None if row.get('calibration_covered') is None
                               else not row['calibration_covered'])
    row['certificate_below_truth'] = (None if row.get('certificate') is None
        or row.get('true_distortion') is None else row['certificate']<row['true_distortion'])
    row['computation_status'] = row.get('computation_status',row.get('selection_status',
        'finite' if row.get('selection_complete') else 'unresolved'))
    row['value_status'] = row.get('value_status',
        'finite' if _number(row.get('certificate')) is not None else 'unresolved')
    if row.get('admitted') is not None and row.get('abstained') is not None:
        if bool(row['admitted']) == bool(row['abstained']):
            raise AssertionError('Admission and abstention must be complements')
        expected = bool(row.get('selection_complete') and row['groups']<=row['capacity_limit'])
        if bool(row['admitted']) != expected:
            raise AssertionError('Admission requires completed selection within reporting capacity')
    if row.get('certificate') is not None:
        if row.get('transported_budget') != 2*row['certificate'] or row.get('budget_ceiling') != 2*row['cap']:
            raise AssertionError('Inference uses actual G*U, separately from ceiling G*c')
        if row.get('selection_complete') and row['certificate']>row['cap']:
            raise AssertionError('Completed selected certificate exceeds cap')
    return row


def _decomposition_check(row, transport):
    """Check exact saved endpoints/accounting without changing signed values."""
    widths = []
    for j in range(5):
        keys = [f'{transport}_I{j}_{name}' for name in ('lower','upper','width')]
        values = [row.get(k) for k in keys]
        if any(v is None for v in values):
            return None
        lower,upper,width = values
        if not 0<=lower<=upper<=1 or upper-lower!=width:
            raise AssertionError('Invalid saved diagnostic interval endpoints')
        widths.append(width)
    for j,name in enumerate(COMPONENTS):
        value = row.get(f'{transport}_{name}')
        if value is None:
            return None
        if value != widths[j+1]-widths[j]:
            raise AssertionError('Signed stage contribution differs from exact endpoint difference')
    contributions = [row[f'{transport}_{name}'] for name in COMPONENTS]
    if (sum(contributions)!=widths[4]-widths[0]
            or row.get(f'{transport}_total_excess')!=widths[4]-widths[0]):
        raise AssertionError('Five-stage accounting is not exact')
    method = 'z_global' if transport=='global' else 'z_cellwise'
    if any(row.get(f'{method}_{name}') != row.get(f'{transport}_I4_{name}')
           for name in ('lower','upper','width')):
        raise AssertionError('Final diagnostic interval differs from operational method interval')
    return True


def _analysis(row, selection):
    row = dict(row)
    row['cap'] = F(row['cap'])
    for name in SEL_KEYS+SELECTION_CONTINUOUS+SELECTION_BINARY+('computation_status','value_status'):
        if name in selection:
            if name in row and row[name]!=selection[name]:
                raise AssertionError(f'Analysis and assignment selection disagree on {name}')
            row.setdefault(name,selection[name])
    for name in ('raw','aggregate'):
        covered = row.get(f'{name}_outcomes_covered')
        row[f'{name}_outcome_miss'] = None if covered is None else not covered
    for transport in ('global','cellwise'):
        row[f'{transport}_accounting_exact'] = _decomposition_check(row,transport)
        nesting = row.get(f'{transport}_nesting')
        covered = row.get('shared_coverage_event')
        row[f'{transport}_nesting_failure_on_shared_event'] = (False if covered is False
            else None if covered is None or nesting is None else not nesting)
    for source in ('aggregate','raw'):
        fallbacks = row.get(f'{source}_root_fallbacks')
        row[f'{source}_root_fallback'] = None if fallbacks is None else bool(fallbacks)
    for method in METHODS:
        lower,upper,width = (row.get(f'{method}_{name}') for name in ('lower','upper','width'))
        if all(v is not None for v in (lower,upper,width)):
            if not 0<=lower<=upper<=1 or upper-lower!=width:
                raise AssertionError('Method width differs from saved exact endpoints')
        target = row.get(f'{method}_target')
        if width is not None and target is not None and bool(target)!=(width<=WIDTH_TARGET):
            raise AssertionError('Precision target must remain the fixed illustrative width 1/5')
        if method in AGGREGATE_METHODS:
            for suffix, source in (('admitted_target','target'),('admitted_and_miss','miss')):
                value = row.get(f'{method}_{source}')
                derived = False if not row.get('admitted') else None if value is None else bool(value)
                key = f'{method}_{suffix}'
                if key in row and row[key] is not None and row[key]!=derived:
                    raise AssertionError('Operational joint indicator has wrong admission denominator')
                row[key] = derived
        else:
            row[f'{method}_admitted_target'] = None
            row[f'{method}_admitted_and_miss'] = None
    lo,hi = row.get('z_global_lower'),row.get('z_global_upper')
    cl,ch = row.get('z_cellwise_lower'),row.get('z_cellwise_upper')
    row['cellwise_contained_in_global'] = (None if any(v is None for v in (lo,hi,cl,ch))
                                          else bool(lo<=cl<=ch<=hi))
    return row


def _deduplicate(rows, keys, fields):
    """Require repeated diagnostic values to agree before removing cohorts."""
    unique = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        entry = {name:row.get(name) for name in keys+tuple(fields)}
        if key in unique and unique[key]!=entry:
            raise AssertionError('Cohort-repeated population diagnostic changed')
        unique[key] = entry
    return list(unique.values())


def _selection_summaries(rows, expected_blocks):
    result = []
    for key,condition in sorted(_group(rows,SEL_KEYS).items()):
        summary = dict(zip(SEL_KEYS,key))
        summary.update(_summarize_group(condition,SELECTION_CONTINUOUS,SELECTION_BINARY,expected_blocks))
        for field in ('computation_status','value_status','selection_scope'):
            summary[field+'_counts_json'] = _json_counts(r.get(field,'not_recorded') for r in condition)
        completed = [r for r in condition if r.get('selection_complete')]
        summary['n_completed_selections'] = len(completed)
        for metric in ('groups','compression_pp','certificate','true_distortion'):
            summary.update({f'completed_{metric}_{name}':value for name,value in
                _stats([r.get(metric) for r in completed]).items()})
        result.append(summary)
    return result


def _analysis_summaries(rows, expected_blocks):
    analyses, methods, decompositions = [], [], []
    for key,condition in sorted(_group(rows,ANALYSIS_KEYS).items()):
        ids = dict(zip(ANALYSIS_KEYS,key))
        entry = ids | _summarize_group(condition,ANALYSIS_CONTINUOUS,ANALYSIS_BINARY,expected_blocks)
        for field in ('LP_status','computation_status','value_status'):
            entry[field+'_counts_json'] = _json_counts(r.get(field,'not_recorded') for r in condition)
        entry['n_admitted'] = sum(bool(r.get('admitted')) for r in condition)
        entry['n_abstained'] = sum(bool(r.get('abstained')) for r in condition)
        entry['global_accounting_exact_counts_json'] = _json_counts(r.get('global_accounting_exact') for r in condition)
        entry['cellwise_accounting_exact_counts_json'] = _json_counts(r.get('cellwise_accounting_exact') for r in condition)
        analyses.append(entry)
        for method in METHODS:
            converted = [dict(r,**{name:r.get(f'{method}_{name}')
                                   for name in METHOD_CONTINUOUS+METHOD_BINARY}) for r in condition]
            item = ids | {'method':method,'method_label':METHOD_LABELS[method]}
            item.update(_summarize_group(converted,METHOD_CONTINUOUS,METHOD_BINARY,expected_blocks))
            admitted = [r for r in converted if r.get('admitted')]
            item.update(n_admitted=len(admitted),n_abstained=len(condition)-len(admitted),
                        admission_scope=('reporting admission applies to aggregate release' if method in AGGREGATE_METHODS
                                         else 'raw reference has additional information; no release-admission rule'),
                        admitted_width_scope='descriptive conditional widths; no nominal conditional coverage',
                        width_scope='all blocks, including valid numerical fallback intervals',
                        width_target=WIDTH_TARGET)
            # Raw-reference conditional means are retained as paired descriptors
            # of the aggregate-admitted subset, not as raw admission decisions.
            item.update({f'admitted_width_{name}':value
                         for name,value in _stats([r.get('width') for r in admitted]).items()})
            item['n_numeric_complete'] = sum(r.get('numerical_complete') is True for r in converted)
            item['n_numeric_incomplete'] = sum(r.get('numerical_complete') is False for r in converted)
            item['n_numeric_completion_unrecorded'] = sum(r.get('numerical_complete') is None for r in converted)
            methods.append(item)
        for transport in ('global','cellwise'):
            converted = [dict(r,**{name:r.get(f'{transport}_{name}') for name in DECOMPOSITION_METRICS})
                         for r in condition]
            item = ids | {'transport':transport,'primary_presentation':transport=='global'}
            item.update(_summarize_group(converted,DECOMPOSITION_METRICS,(),expected_blocks))
            item['interpretation'] = ('Signed differences along the prespecified diagnostic sequence; '
                'population aggregation includes universal transport; I3 is privileged truth-assignment diagnostic')
            decompositions.append(item)
    return analyses,methods,decompositions


def _cap_contrasts(rows, keys, metrics, expected_blocks, kind):
    base_keys = tuple(k for k in keys if k!='cap')
    result = []
    for key,condition in sorted(_group(rows,base_keys).items()):
        by_cap = _group(condition,('cap',))
        pairs = [('consecutive_cap',CAPS[j],CAPS[j-1]) for j in range(1,len(CAPS))]
        pairs += [('cap_1_reference',cap,CAPS[0]) for cap in CAPS[1:]]
        for comparison,left_cap,right_cap in pairs:
            for metric in metrics:
                entry = dict(zip(base_keys,key)) | {'record_type':kind,'comparison':comparison,
                    'left_cap':left_cap,'right_cap':right_cap,'metric':metric}
                entry.update(_paired(by_cap.get((left_cap,),()),by_cap.get((right_cap,),()),
                                     metric,expected_blocks=expected_blocks))
                result.append(entry)
    return result


def _method_contrasts(rows, expected_blocks):
    result = []
    pairs = (('z_global','z_cellwise'),('z_global','raw_pooled'),
             ('z_global','raw_stratified'),('z_cellwise','raw_pooled'),
             ('z_cellwise','raw_stratified'),('raw_pooled','raw_stratified'))
    for key,condition in sorted(_group(rows,ANALYSIS_KEYS).items()):
        for left,right in pairs:
            for metric in ('width','target','miss'):
                entry = dict(zip(ANALYSIS_KEYS,key)) | {'left_method':left,'right_method':right,
                    'left_method_label':METHOD_LABELS[left],'right_method_label':METHOD_LABELS[right],
                    'metric':metric,'identity_inference_reference':right=='raw_stratified'}
                entry.update(_paired(condition,condition,f'{left}_{metric}',f'{right}_{metric}',expected_blocks))
                result.append(entry)
    return result


def _validate_design(records, selections, analyses, expected_blocks):
    if isinstance(expected_blocks,bool) or not isinstance(expected_blocks,int) or expected_blocks<1:
        raise ValueError('expected_blocks is the positive independent-block count per law')
    expected_profiles = {0:3,1:3,2:2}
    block_ids = [(r['law_index'],r['replication']) for r in records]
    if len(set(block_ids))!=len(block_ids):
        raise AssertionError('Duplicate saved block')
    if set(i for i,_ in block_ids)!={0,1,2}:
        raise AssertionError('All three frozen laws must be present')
    for law_index in range(3):
        observed = [rep for i,rep in block_ids if i==law_index]
        if set(observed)!=set(range(expected_blocks)):
            raise AssertionError('Every committed independent block is required')
        metadata = {(r['law_id'],r['K'],r['n'],r['capacity_limit'])
                    for r in selections if r['law_index']==law_index}
        if len(metadata)!=1:
            raise AssertionError('Frozen law metadata changed between blocks or caps')
        _,K,n,B = next(iter(metadata))
        if n!=4000*K or B!=(8,32,4)[law_index]:
            raise AssertionError('Calibration size or secondary reporting capacity changed')
        if any(r['m']!=r['m_ratio']*K for r in analyses if r['law_index']==law_index):
            raise AssertionError('Outcome-cohort size differs from the frozen ratio')
        profiles = {r['profile'] for r in analyses if r['law_index']==law_index}
        if len(profiles)!=expected_profiles[law_index]:
            raise AssertionError('All committed outcome profiles are required')
        for replication in range(expected_blocks):
            sr = [r for r in selections if r['law_index']==law_index and r['replication']==replication]
            ar = [r for r in analyses if r['law_index']==law_index and r['replication']==replication]
            if len(sr)!=len(CAPS) or {r['cap'] for r in sr}!=set(CAPS):
                raise AssertionError('Every block requires all five committed cap selections')
            wanted = {(profile,ratio,cap) for profile in profiles for ratio in (16,256) for cap in CAPS}
            observed = [(r['profile'],r['m_ratio'],r['cap']) for r in ar]
            if len(observed)!=len(set(observed)) or set(observed)!=wanted:
                raise AssertionError('Every block requires all committed profile/cohort/cap instances')
    if len(selections)!=15*expected_blocks or len(analyses)!=80*expected_blocks:
        raise AssertionError('Committed selection or analysis count mismatch')


def summarize(records, expected_blocks=100):
    """Return complete derived tables; reduced block counts support test fixtures.

    Even reduced fixtures retain the frozen laws/profile counts, both cohorts,
    and all five caps.  Missing scientific conditions never become a complete
    report.  The function neither reads population laws nor recomputes inference.
    """
    records = list(records)
    selections, analyses, computation = [], [], []
    for block in records:
        provenance = {'law_index':block['law_index'],'replication':block['replication'],
                      'evidence_file':block.get('evidence_file')}
        for row in block['selection_rows']+block['analysis_rows']:
            if any(name in row and row[name]!=provenance[name] for name in ('law_index','replication')):
                raise AssertionError('Saved row identifies a different independent block')
        selected = [_selection(dict(provenance,**row)) for row in block['selection_rows']]
        lookup = {r['cap']:r for r in selected}
        if len(lookup)!=len(selected):
            raise AssertionError('Repeated cap selection in a saved block')
        selections.extend(selected)
        for row in block['analysis_rows']:
            analyses.append(_analysis(dict(provenance,**row),lookup[F(row['cap'])]))
        cost = dict(provenance)
        for name in ('timings','cache','memory','root_statistics'):
            if isinstance(block.get(name),dict):
                value = _computation_cache(block[name]) if name=='cache' else block[name]
                cost.update(_flatten(value,name+'_'))
        for name in ('worker_seconds_before_serialization','peak_rss_bytes','peak_memory_bytes'):
            if name in block:
                cost[name] = block[name]
        computation.append(cost)
    _validate_design(records,selections,analyses,expected_blocks)
    selections.sort(key=lambda r:(r['law_index'],r['replication'],r['cap']))
    analyses.sort(key=lambda r:(r['law_index'],r['replication'],r['profile'],r['m_ratio'],r['cap']))
    # Assignment statistics originate in selection_rows, never repeated across
    # profiles/cohorts.  Population endpoints are repeated over cohorts only.
    population_fields = tuple(f'{transport}_I{j}_{name}' for transport in ('global','cellwise')
                              for j in range(3) for name in ('lower','upper','width'))
    population = _deduplicate(analyses,POP_KEYS+('replication',),
                             population_fields+('raw_population_exceeds_target',))
    population_summary = []
    for key,condition in sorted(_group(population,POP_KEYS).items()):
        item = dict(zip(POP_KEYS,key))
        item.update(_summarize_group(condition,POP_METRICS,('raw_population_exceeds_target',),expected_blocks))
        population_summary.append(item)
    assignment_summary = _selection_summaries(selections,expected_blocks)
    analysis_summary,method_summary,decomposition_summary = _analysis_summaries(analyses,expected_blocks)
    selection_pairs = _cap_contrasts(selections,SEL_KEYS,
        SELECTION_CONTINUOUS+('admitted','selection_complete'),expected_blocks,'selection')
    analysis_metrics = tuple(f'{method}_{metric}' for method in METHODS
                             for metric in ('width','target','miss','fallback','numerical_complete','admitted_target'))
    analysis_metrics += tuple(f'{transport}_{metric}' for transport in ('global','cellwise')
                              for metric in DECOMPOSITION_METRICS)
    analysis_pairs = _cap_contrasts(analyses,ANALYSIS_KEYS,analysis_metrics,expected_blocks,'analysis')
    identity_pairs = []
    for key,condition in sorted(_group(selections,SEL_KEYS).items()):
        for metric,reference in (('groups',condition[0]['K']),('compression_pp',0),
                                 ('certificate',F(1)),('true_distortion',F(1)),
                                 ('transported_budget',F(2))):
            right = [dict(r,**{metric:reference}) for r in condition]
            item = dict(zip(SEL_KEYS,key)) | {'comparison':'literal_identity_partition',
                'metric':metric,'reference_value':reference}
            item.update(_paired(condition,right,metric,expected_blocks=expected_blocks))
            identity_pairs.append(item)
    method_pairs = _method_contrasts(analyses,expected_blocks)
    computation_summary = []
    for key,condition in sorted(_group(computation,('law_index',)).items()):
        metrics = tuple(sorted({name for r in condition for name,value in r.items()
            if name not in ('law_index','replication','evidence_file') and
            (value is None or isinstance(value,(int,float,F))) and not isinstance(value,bool)}))
        item = {'law_index':key[0],
                'timing_scope':'additional cached execution; fixed batch order; no cold end-to-end speed comparison'}
        item.update(_summarize_group(condition,metrics,(),expected_blocks))
        for name in metrics:
            finite = [_number(r.get(name)) for r in condition]
            item[name+'_sum_available'] = sum(v for v in finite if v is not None)
        computation_summary.append(item)
    # Repeated raw references must be identical across caps, not merely equal
    # in their means.  Their outcome allocation is shared across procedures.
    raw_groups = _group(analyses,('law_index','replication','profile','m_ratio'))
    raw_invariant = all(len({tuple(r.get(f'raw_stratified_{name}') for name in ('lower','upper','width'))
                             for r in condition})==1 for condition in raw_groups.values())
    if not raw_invariant:
        raise AssertionError('Shared raw stratification changed across caps')
    monotonic = []
    for condition in _group(selections,('law_index','replication')).values():
        completed = sorted((r for r in condition if r.get('selection_complete')),key=lambda r:r['cap'])
        monotonic.append(all(a['groups']>=b['groups'] for a,b in zip(completed,completed[1:])))
    if not all(monotonic):
        raise AssertionError('Completed minimum group count increased with cap')
    summary = {
        'status':'report_complete','expected_independent_blocks_per_law':expected_blocks,
        'independent_blocks':len(records),'selection_rows':len(selections),
        'analysis_instances':len(analyses),'method_evaluations':len(METHODS)*len(analyses),
        'population_instances_after_cohort_deduplication':len(population),
        'assignment_conditions':len(assignment_summary),'population_conditions':len(population_summary),
        'analysis_conditions':len(analysis_summary),'method_conditions':len(method_summary),
        'decomposition_conditions':len(decomposition_summary),
        'caps':CAPS,'width_target':WIDTH_TARGET,'raw_sensitivity_budget':F(2),
        'methods':METHOD_LABELS,
        'uncertainty':{'continuous':'pointwise normal Monte Carlo intervals over independent blocks',
                       'binary':'two-sided 95% Clopper-Pearson intervals within each condition',
                       'paired':'normal Monte Carlo intervals for within-block differences, including binary differences',
                       'outcome_scope':'pointwise for prespecified cap/profile/cohort; no simultaneous frontier outcome claim',
                       'coverage_target':'containment of the entire raw identified population interval, not only the generating mean',
                       'admitted_widths':'descriptive only; no nominal conditional coverage claim'},
        'checks':{'shared_raw_invariant_across_caps':raw_invariant,
                  'completed_count_nonincreasing_across_caps':all(monotonic),
                  'global_accounting_counts':dict(Counter(str(r.get('global_accounting_exact')) for r in analyses)),
                  'cellwise_accounting_counts':dict(Counter(str(r.get('cellwise_accounting_exact')) for r in analyses)),
                  'cellwise_containment_counts':dict(Counter(str(r.get('cellwise_contained_in_global')) for r in analyses)),
                  'aggregate_interface_check_counts':dict(Counter(str(r.get('aggregate_only_equal')) for r in analyses)),
                  'identity_distortion_and_certificate_counts':dict(Counter(
                      str(r.get('true_distortion')==r.get('certificate')==1)
                      for r in selections if r.get('identity'))),
                  'identity_global_cellwise_shared_raw_interval_equality_counts':dict(Counter(
                      str(all(r.get(f'z_global_{name}')==r.get(f'z_cellwise_{name}')==
                          r.get(f'raw_stratified_{name}') for name in ('lower','upper','width')))
                      for r in analyses if r.get('identity'))),
                  'capacity_study_admission_match_counts':dict(Counter(
                      str(r.get('source_admission_match')) for r in selections if r['cap']==F(3,2)))},
        'selection_status_counts':dict(Counter(str(r.get('computation_status')) for r in selections)),
        'selection_value_status_counts':dict(Counter(str(r.get('value_status')) for r in selections)),
        'selection_complete_counts':dict(Counter(str(r.get('selection_complete')) for r in selections)),
        'analysis_LP_status_counts':dict(Counter(str(r.get('LP_status','not_recorded')) for r in analyses)),
        'assignment_diagnostic_counts':{name:dict(Counter(str(r.get(name)) for r in selections))
                                        for name in SELECTION_BINARY},
        'analysis_diagnostic_counts':{name:dict(Counter(str(r.get(name)) for r in analyses))
                                      for name in ANALYSIS_BINARY},
        'method_diagnostic_counts':{method:{name:dict(Counter(str(r.get(f'{method}_{name}'))
            for r in analyses)) for name in METHOD_BINARY} for method in METHODS},
        'diagnostic_count_scope':'counts across saved instances, not a pooled binomial rate; use condition tables for uncertainty',
        'timing_scope':'additional cache-dependent execution; no cold end-to-end performance comparison',
        'peak_rss_bytes_max':max((_number(r.get('peak_rss_bytes',r.get('peak_memory_bytes'))) or 0
                                  for r in computation),default=0) or None,
        'tables':{
            'selection_replications':'tables/selection_replications.csv',
            'analysis_replications':'tables/analysis_replications.csv',
            'assignment_summary':'tables/assignment_summary.csv',
            'population_summary':'tables/population_summary.csv',
            'method_width_target_summary':'tables/method_width_target_summary.csv',
            'decomposition_summary':'tables/decomposition_summary.csv',
            'coverage_numerical_summary':'tables/coverage_numerical_summary.csv',
            'cap_contrasts':'tables/paired_cap_contrasts.csv',
            'method_contrasts':'tables/paired_method_contrasts.csv',
            'identity_contrasts':'tables/paired_identity_selection_contrasts.csv',
            'computation':'tables/computation_blocks.csv',
            'computation_summary':'tables/computation_summary.csv'},
    }
    tables = {'selection_replications':selections,'analysis_replications':analyses,
              'assignment_summary':assignment_summary,'population_summary':population_summary,
              'method_width_target_summary':method_summary,'decomposition_summary':decomposition_summary,
              'coverage_numerical_summary':analysis_summary,
              'paired_cap_contrasts':selection_pairs+analysis_pairs,
              'paired_method_contrasts':method_pairs,'paired_identity_selection_contrasts':identity_pairs,
              'computation_blocks':computation,'computation_summary':computation_summary}
    return {'summary':summary,'tables':tables}


def _csv(path, rows):
    """Fractions keep exact strings; companion float columns are for plotting."""
    rows = list(rows)
    rational_keys = {key for row in rows for key,value in row.items() if isinstance(value,F)}
    columns = list(dict.fromkeys(key for row in rows for key in row))
    columns += [key+'_float' for key in columns if key in rational_keys and key+'_float' not in columns]
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    # Full exact endpoint tables can be large.  Stream rows rather than hold
    # an additional complete CSV string alongside all paired block records.
    with temporary.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            values = dict(row)
            for key in rational_keys:
                value = row.get(key)
                values.setdefault(key+'_float',_number(value))
            writer.writerow({key:('NA' if value is None else str(value) if isinstance(value,F)
                else json.dumps(encode(value),sort_keys=True) if isinstance(value,(dict,list,tuple))
                else value) for key,value in values.items()})
    temporary.replace(path)


def _load(path):
    if Path(path).suffix=='.gz':
        with gzip.open(path,'rt') as stream:
            return decode(json.load(stream))
    return decode(json.loads(Path(path).read_text()))






def generate(root, expected_blocks=100, figures=True):
    """Read the complete saved panel and write derived outputs only."""
    root = Path(root)
    paths = sorted((root/'panel'/'blocks').glob('*.json.gz'))
    if not paths:
        raise FileNotFoundError('No saved frontier panel blocks')
    tick = time.perf_counter()
    records = []
    for path in paths:
        record = _load(path)
        # New block evidence also contains LP witnesses and changed cell/root
        # records.  Reporting needs only flat rows and computation metadata;
        # retain at most one full block in memory while reading the panel.
        needed = ('law_index','law_id','replication','preflight','selection_rows',
                  'analysis_rows','timings','cache','peak_rss_bytes',
                  'peak_memory_bytes','worker_seconds_before_serialization',
                  'root_statistics','memory')
        compact = {name:record[name] for name in needed if name in record}
        compact['evidence_file'] = str(path.relative_to(root))
        records.append(compact)
        del record
    result = summarize(records,expected_blocks=expected_blocks)
    tables = root/'tables';tables.mkdir(exist_ok=True)
    for name,rows in result['tables'].items():
        _csv(tables/(name+'.csv'),rows)
    if figures:
        pass  # Public release generates numerical tables only.
    result['summary']['reporting_seconds'] = time.perf_counter()-tick
    (root/'summary.json').write_text(json.dumps(encode(result['summary']),indent=2)+'\n')
    return result['summary']


report = generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'results/jmlr_precision_frontier_v1')
    parser.add_argument('--expected-blocks',type=int,default=100)
    parser.add_argument('--no-figures',action='store_true')
    args = parser.parse_args()
    answer = generate(args.root,args.expected_blocks,not args.no_figures)
    print(json.dumps(encode({key:answer[key] for key in
        ('status','independent_blocks','selection_rows','analysis_instances','method_evaluations','reporting_seconds')})))


if __name__=='__main__':
    main()
