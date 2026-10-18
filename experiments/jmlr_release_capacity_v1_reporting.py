"""Tables and scientific figures for the complete reporting-capacity panel.

This module reads frozen evidence only.  All uncertainty calculations use
independent block replicates, with within-block differences for comparisons.
Admitted-only means are descriptive; admitted-and-miss probabilities retain
the unconditional block denominator.  Neither outcomes nor this reporting code
select reporting capacities or change admission decisions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import statistics
import time

from experiments.jmlr_release_capacity_v1 import dump, load, now
from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier


REPORTING_REVISION = '20261016_binary_rate_cp95'


METHODS = ('verified_capacity', 'equal_calibration_mass')
KEYS = ('law_index', 'law_id', 'K', 'profile', 'm_ratio', 'm', 'B', 'method')
CAL_KEYS = ('law_index', 'law_id', 'K', 'B', 'method')
PROBABILITIES = (
    'admitted', 'abstained', 'global_miss', 'cellwise_miss', 'raw_stratified_miss',
    'pooled_raw_miss', 'admitted_and_global_miss', 'admitted_and_cellwise_miss',
    'evaluation_fallback', 'global_fallback', 'cellwise_fallback', 'LP_fallback',
    'empty_pooled_intersection', 'aggregate_only_equal')
NUMERICS = (
    'groups', 'certificate', 'transported_budget', 'true_distortion', 'oracle_distortion',
    'raw_population_width', 'aggregate_true_population_width',
    'aggregate_certified_population_width', 'global_confidence_width',
    'cellwise_confidence_width', 'raw_stratified_width', 'pooled_raw_width',
    'payload_bytes', 'operational_payload_bytes', 'analyst_seconds',
    'zero_treated_groups', 'analyst_root_fallbacks', 'analyst_peak_rss_bytes',
    'analyst_cache_hits', 'analyst_cache_misses')
METRICS = NUMERICS + PROBABILITIES
COMPARISON_METRICS = (
    'admitted', 'groups', 'certificate', 'true_distortion', 'transported_budget',
    'aggregate_true_population_width', 'aggregate_certified_population_width',
    'global_confidence_width', 'cellwise_confidence_width', 'pooled_raw_width',
    'admitted_and_global_miss', 'admitted_and_cellwise_miss',
    'payload_bytes', 'operational_payload_bytes', 'analyst_seconds')


def _number(value):
    if value is None or not isinstance(value, (int, float, F)):
        return None
    answer = float(value)
    return answer if math.isfinite(answer) else None


@lru_cache(maxsize=None)
def _clopper_pearson95(events, trials):
    """Two-sided 95% CP interval, with verified outward numerical endpoints.

    This reports Monte Carlo uncertainty in an individual binary rate; it
    does not change the saved assignment or outcome confidence enclosures.
    General paired differences must never enter this single-binomial helper.
    """
    if (isinstance(events, bool) or isinstance(trials, bool)
            or not isinstance(events, int) or not isinstance(trials, int)
            or not 0 <= events <= trials):
        raise ValueError('Binomial rates require integer 0 <= events <= trials')
    if trials == 0:
        return None, None
    interval = RootVerifier().interval(events, trials, F(1,20))
    if interval['status'] == 'fallback':
        raise ArithmeticError('Reporting CP endpoints could not be verified')
    return float(interval['lower']), float(interval['upper'])


def _statistics(values, probability=False):
    """CP intervals for binary rates; normal mean intervals for other metrics.

    SD and SE remain descriptive Monte Carlo summaries for either type.
    Missing/nonfinite values remain counted rather than zero-imputed.
    """
    values = list(values)
    numeric = [_number(value) for value in values]
    finite = [value for value in numeric if value is not None]
    result = {'n': len(finite), 'n_missing': sum(value is None for value in values),
              'n_nonfinite_or_nonnumeric': sum(a is not None and b is None for a,b in zip(values,numeric)),
              'mean': None, 'sd': None, 'se': None, 'ci95_lower': None, 'ci95_upper': None,
              'ci95_method':('two_sided_95pct_clopper_pearson' if probability else
                             'normal_approximation_of_block_mean')}
    if not finite:
        return result
    if probability and any(value not in (0.,1.) for value in finite):
        raise ValueError('Individual binary rates require zero/one observations')
    mean = statistics.fmean(finite)
    result['mean'] = mean
    if len(finite) > 1:
        sd = statistics.stdev(finite)
        se = sd/math.sqrt(len(finite))
        lo, hi = mean-1.96*se, mean+1.96*se
        result.update(sd=sd, se=se, ci95_lower=lo, ci95_upper=hi)
    if probability:
        lo, hi = _clopper_pearson95(int(sum(finite)),len(finite))
        result.update(ci95_lower=lo, ci95_upper=hi)
    return result


def _summarize(rows, expected=100):
    """One condition, one observation per independent block."""
    rows = list(rows)
    replications = [row['replication'] for row in rows]
    if len(replications) != len(set(replications)):
        raise AssertionError('Repeated block in a condition summary')
    admitted = [row for row in rows if row['admitted']]
    finite = [row for row in rows if row['status'] == 'finite']
    statuses = Counter(row.get('status', 'missing') for row in rows)
    lp_statuses = Counter(row.get('LP_status', 'not_computed') for row in rows)
    answer = {'n':len(rows), 'n_expected':expected, 'n_missing_blocks':expected-len(rows),
              'n_admitted':len(admitted), 'n_abstained':sum(row['abstained'] for row in rows),
              'n_finite':statuses['finite'], 'n_infinite':statuses['infinite'],
              'n_unresolved':statuses['unresolved'],
              'status_counts_json':json.dumps(dict(sorted(statuses.items())),sort_keys=True),
              'LP_status_counts_json':json.dumps(dict(sorted(lp_statuses.items())),sort_keys=True),
              'admitted_precision_scope':'descriptive conditional means; no conditional nominal coverage'}
    for metric in METRICS:
        for prefix, selected in (('',rows), ('admitted_',admitted), ('finite_',finite)):
            stats = _statistics([row.get(metric) for row in selected], metric in PROBABILITIES)
            answer.update({f'{prefix}{metric}_{key}':value for key,value in stats.items()})
    for metric in ('certificate','transported_budget'):
        answer[f'{metric}_n_mathematically_infinite'] = statuses['infinite']
        answer[f'{metric}_n_unresolved'] = statuses['unresolved']
    # These are unconditional probabilities over all independent blocks. An
    # abstention contributes zero, even if no confidence interval was computed.
    for method in ('global','cellwise'):
        metric = f'admitted_and_{method}_miss'
        values = []
        for row in rows:
            if not row['admitted']:
                values.append(False)
            elif metric not in row:
                raise AssertionError('Admitted release lacks its coverage diagnostic')
            else:
                values.append(bool(row[metric]))
        answer.update({f'{metric}_{key}':value for key,value in _statistics(values,True).items()})
        answer[f'{metric}_failures'] = sum(values)
        n, failures = len(values), sum(values)
        answer[f'{metric}_supplementary_one_sided_exact95_upper_if_zero_failures'] = (
            1-math.exp(math.log(.05)/n) if n and failures==0 else None)
        answer[f'{metric}_supplementary_one_sided_exact95_lower_if_all_failures'] = (
            math.exp(math.log(.05)/n) if n and failures==n else None)
    return answer


def _paired_summary(left, right, left_metric, right_metric=None, expected=100):
    """Paired differences with an explicit count of unavailable block pairs."""
    right_metric = right_metric or left_metric
    left, right = list(left), list(right)
    if (len({row['replication'] for row in left})!=len(left)
            or len({row['replication'] for row in right})!=len(right)):
        raise AssertionError('Duplicate block in paired comparison')
    left = {row['replication']:row for row in left}
    right = {row['replication']:row for row in right}
    common = sorted(set(left)&set(right))
    differences, both_admitted = [], []
    for rep in common:
        a, b = _number(left[rep].get(left_metric)), _number(right[rep].get(right_metric))
        if a is None or b is None:
            continue
        differences.append(a-b)
        if left[rep]['admitted'] and right[rep]['admitted']:
            both_admitted.append(a-b)
    answer = {'n_expected':expected, 'n_matching_blocks':len(common),
              'n_unavailable_pairs':expected-len(differences),
              'left_metric':left_metric,'right_metric':right_metric,
              'direction':'left minus right',
              'inference_method':'paired_block_mean_difference_normal_approximation',
              'conditional_scope':'both-admitted comparison is descriptive'}
    answer.update({f'difference_{key}':value for key,value in _statistics(differences).items()})
    answer.update({f'both_admitted_difference_{key}':value
                   for key,value in _statistics(both_admitted).items()})
    return answer


def _csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with io.StringIO(newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:('NA' if value is None else str(value) if isinstance(value,F) else
                                  json.dumps(value,sort_keys=True) if isinstance(value,(list,tuple,dict)) else value)
                             for key,value in row.items()})
        payload = stream.getvalue().encode('utf-8')
    path = Path(path)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)


def _root_precision(root):
    values = []
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('precision','precision_bits','working_precision','arb_precision','max_precision_used') and isinstance(item,int):
                    values.append(item)
                elif isinstance(item,(dict,list,tuple)):
                    visit(item)
        elif isinstance(value,(list,tuple)):
            for item in value:
                visit(item)
    visit(root)
    return max(values) if values else None


def _rows_by_condition(rows, keys=KEYS):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups




def report(root, *, capacity_figure_only=False):
    """Validate and summarize the complete committed panel, never its blocks."""
    start = time.perf_counter(); root = Path(root)
    protocol = load(root/'frozen_protocol.json'); config = protocol['configuration']
    expected_reps = config['replications']
    paths = sorted((root/'panel/blocks').glob('*.json.gz'))
    expected_blocks = {(i,j) for i in range(len(protocol['laws'])) for j in range(expected_reps)}
    if len(paths) != len(expected_blocks):
        raise AssertionError(f'Complete panel required: found {len(paths)} of {len(expected_blocks)} blocks')
    rows, stages, cal_rows, seen_blocks = [], [], [], set()
    diagnostics = Counter(); lp_statuses = Counter(); cell_statuses = Counter()
    receipts = []; maximum_precision = None
    for path in paths:
        block = load(path); i,j = block['law_index'],block['replication']
        if (i,j) in seen_blocks or (i,j) not in expected_blocks or block['preflight']:
            raise AssertionError('Unexpected, repeated, or development block in committed panel')
        seen_blocks.add((i,j))
        expected = {(p,m,B,method) for p in protocol['profiles'][i]
                    for m in config['outcome_m_per_K'] for B in protocol['capacity_grids'][i]
                    for method in METHODS}
        actual = {(r['profile'],r['m_ratio'],r['B'],r['method']) for r in block['rows']}
        if len(block['rows']) != len(expected) or actual != expected:
            raise AssertionError('Block does not contain every committed condition exactly once')
        cal = {}
        for row in block['rows']:
            if (row['law_index'],row['replication']) != (i,j):
                raise AssertionError('Row block identity differs from enclosing evidence')
            if row['status'] not in ('finite','infinite','unresolved'):
                raise AssertionError('Unrecognized optimization status, refusing to hide it')
            if row['admitted'] and (row['status']!='finite' or row['groups']>row['B'] or 2*row['certificate']>3 or row['evaluation_fallback']):
                raise AssertionError('Saved admission violates the frozen operational constraint')
            key = (row['B'],row['method'])
            previous = cal.setdefault(key,row)
            for field in ('status','groups','certificate','true_distortion','admitted','oracle_distortion'):
                if previous.get(field) != row.get(field):
                    raise AssertionError('Assignment-based selection changed across outcome conditions')
            if row['status']=='finite':
                diagnostics['finite_rows'] += 1
                diagnostics['aggregate_only_equal_rows'] += row.get('aggregate_only_equal') is True
                if row.get('aggregate_only_equal') is not True:
                    raise AssertionError('Completed release lacks aggregate-only equality evidence')
            diagnostics['admitted_rows'] += row['admitted']
            diagnostics['evaluation_fallback_rows'] += row['evaluation_fallback']
            for field in ('global_miss','cellwise_miss','pooled_raw_miss','admitted_and_global_miss',
                          'admitted_and_cellwise_miss','global_fallback','cellwise_fallback','LP_fallback',
                          'empty_pooled_intersection'):
                diagnostics[f'{field}_rows'] += bool(row.get(field))
        for B in protocol['capacity_grids'][i]:
            a,b = (cal[B,method] for method in METHODS)
            if a['status']=='finite' and b['status']=='finite':
                diagnostics['structural_certificate_comparisons'] += 1
                if a['certificate']>b['certificate']:
                    raise AssertionError('Selector certificate exceeds contained baseline certificate')
            else:
                diagnostics['structural_certificate_comparisons_unavailable'] += 1
        if cal[protocol['laws'][i]['K'],'verified_capacity']['certificate'] != 1:
            raise AssertionError('Identity-capacity certificate differs from one')
        cal_rows.extend(cal.values()); rows.extend(block['rows'])
        analyses = block.get('analyst_results',[])
        per_receipts = [a.get('data_access_receipt',{}) for a in analyses]
        receipt = block.get('analyst_receipt') or (per_receipts[0] if per_receipts else {})
        restricted = bool(receipt.get('code_data_access_restricted'))
        forbidden = sum(len(r.get('noncode_project_access_attempts',[])) for r in per_receipts)
        all_copied = bool(per_receipts) and all(r.get('code_data_access_restricted')
                       and r.get('data_input_open_count')==1
                       and not r.get('noncode_project_access_attempts') for r in per_receipts)
        diagnostics['restricted_process_blocks'] += restricted
        diagnostics['all_per_analysis_receipts_blocks'] += all_copied
        diagnostics['forbidden_project_data_attempts'] += forbidden
        diagnostics['analyst_analyses'] += len(analyses)
        diagnostics['provider_root_fallbacks'] += block['diagnostics']['root_fallbacks']
        diagnostics['raw_outcome_root_fallbacks'] += sum(r['status']=='fallback' for o in block['outcomes'] for r in o['raw_roots'])
        diagnostics['analyst_root_fallbacks'] += sum(a['root_fallbacks'] for a in analyses)
        diagnostics['calibration_coverage_failed_blocks'] += not block['diagnostics']['calibration_covered']
        diagnostics['raw_stratified_miss_unique_outcomes'] += sum(
            o['raw_confidence']['interval'][0]>o['raw_population_interval'][0]
            or o['raw_confidence']['interval'][1]<o['raw_population_interval'][1] for o in block['outcomes'])
        diagnostics['unique_outcomes'] += len(block['outcomes'])
        diagnostics['LP_unique_solves'] += len(block['LP_records'])
        diagnostics['LP_unique_fallbacks'] += sum(l['LP']['fallback'] for l in block['LP_records'])
        lp_statuses.update(l['LP']['status'] for l in block['LP_records'])
        cell_statuses.update(block['diagnostics']['cell_status_counts'])
        precision = _root_precision({'region':block['region'],'outcomes':block['outcomes'],
                                     'analyst_results':analyses})
        if precision is not None:
            maximum_precision = max(maximum_precision or 0,precision)
        stage = {'law_index':i,'law_id':block['law_id'],'replication':j,
                 **block['timings'],'peak_rss_bytes':block['peak_rss_bytes'],
                 'analyst_peak_rss_bytes':max((a.get('peak_rss_bytes',0) for a in analyses),default=0),
                 'analyst_cache_hits':sum(a.get('root_cache_hits',0) for a in analyses),
                 'analyst_cache_misses':sum(a.get('root_cache_misses',0) for a in analyses),
                 'provider_cache_hits':block['diagnostics']['root_cache_hits'],
                 'provider_cache_misses':block['diagnostics']['root_cache_misses'],
                 'provider_root_fallbacks':block['diagnostics']['root_fallbacks'],
                 'analyst_root_fallbacks':sum(a['root_fallbacks'] for a in analyses),
                 'max_root_precision_used':precision,'evidence_bytes':path.stat().st_size,
                 'analyst_input_bytes':(root/'panel/analyst_inputs'/path.name).stat().st_size,
                 'restricted_data_receipt':restricted,'all_analysis_receipts_copied':all_copied}
        stages.append(stage)
        receipts.append({'law_index':i,'replication':j,'restricted':restricted,
                         'per_analysis_receipts':len(per_receipts),'all_copied':all_copied,
                         'forbidden_access_attempts':forbidden})
        if len(seen_blocks) % 50 == 0:
            print(json.dumps({'report_blocks_read':len(seen_blocks),'rows_read':len(rows)}),flush=True)
        del block
    if seen_blocks != expected_blocks or len(rows) != protocol['committed_condition_rows']:
        raise AssertionError('Final block/condition count differs from the frozen panel')
    rows.sort(key=lambda r:tuple(r[k] for k in KEYS)+ (r['replication'],))
    summaries = [dict(zip(KEYS,key),**_summarize(values,expected_reps))
                 for key,values in sorted(_rows_by_condition(rows).items())]
    calibration = [dict(zip(CAL_KEYS,key),**_summarize(values,expected_reps))
                   for key,values in sorted(_rows_by_condition(cal_rows,CAL_KEYS).items())]
    if len(summaries)!=160 or len(calibration)!=30 or any(r['n']!=expected_reps for r in summaries+calibration):
        raise AssertionError('Incomplete condition or calibration summary grid')
    paired = []
    condition_groups = _rows_by_condition(rows)
    without_method = KEYS[:-1]
    for key,values in sorted(_rows_by_condition(rows,without_method).items()):
        left = [r for r in values if r['method']==METHODS[0]]
        right = [r for r in values if r['method']==METHODS[1]]
        for metric in COMPARISON_METRICS:
            paired.append(dict(zip(without_method,key),comparison='selector_minus_equal_mass',
                               **_paired_summary(left,right,metric,expected=expected_reps)))
    for key,values in sorted(condition_groups.items()):
        for right_metric,label in [('cellwise_confidence_width','global_minus_cellwise'),
                                   ('pooled_raw_width','global_minus_pooled_raw'),
                                   ('raw_stratified_width','global_minus_raw_stratified')]:
            paired.append(dict(zip(KEYS,key),comparison=label,
                               **_paired_summary(values,values,'global_confidence_width',right_metric,expected_reps)))
    tables = root/'tables'; tables.mkdir(exist_ok=True)
    _csv(tables/'replications.csv',rows)
    _csv(tables/'condition_summary.csv',summaries)
    _csv(tables/'calibration_summary.csv',calibration)
    _csv(tables/'paired_comparisons.csv',paired)
    _csv(tables/'stage_times.csv',sorted(stages,key=lambda r:(r['law_index'],r['replication'])))
    pass  # Public release generates numerical tables only.
    manifest = load(root/'manifest.json')
    stage_totals = {key:sum(stage.get(key,0) or 0 for stage in stages)
                    for key in stages[0] if key.endswith('_seconds')}
    summary = {
        'version':config['version'],'generated_utc':now(),'status':'report_complete',
        'independent_blocks':len(seen_blocks),'replication_rows':len(rows),
        'conditions':len(summaries),'calibration_conditions':len(calibration),
        'paired_comparisons':len(paired),'replications_per_condition':expected_reps,
        'optimization_status_counts':dict(Counter(r['status'] for r in rows)),
        'LP_unique_status_counts':dict(lp_statuses),'provider_cell_status_counts':dict(cell_statuses),
        'checks_and_counts':dict(diagnostics),
        'all_aggregate_only_equal':diagnostics['aggregate_only_equal_rows']==diagnostics['finite_rows'],
        'all_restricted_process_receipts':diagnostics['restricted_process_blocks']==len(seen_blocks),
        'all_per_analysis_receipts_copied':diagnostics['all_per_analysis_receipts_blocks']==len(seen_blocks),
        'structural_certificate_violations':0,
        'provider_peak_rss_bytes':max(s['peak_rss_bytes'] for s in stages),
        'analyst_peak_rss_bytes':max(s['analyst_peak_rss_bytes'] for s in stages),
        'root_policy':config['root_policy'],'maximum_root_precision_observed':maximum_precision,
        'stage_worker_seconds':stage_totals,
        'preflight':manifest.get('preflight'), 'panel_wall_seconds':manifest.get('panel_wall_seconds'),
        'stage_time_interpretation':'Measured process times; analyst uses a shared cache in fixed profile, cohort-size, ascending-capacity, selector-then-baseline batch order. Timing comparisons depend on cache/order and are not method speed claims.',
        'payload_interpretation':'payload_bytes includes every completed finite diagnostic artifact; admitted_payload statistics describe operational releases. operational_payload_bytes is zero on abstention. Exact rational JSON encoding is included.',
        'uncertainty':'Individual binary-rate intervals use two-sided 95% Clopper-Pearson bounds with verified outward numerical endpoints. Other means and paired block differences retain mean +/-1.96 sample SD/sqrt(n); no single-binomial formula is applied to general paired differences. Supplementary ONE-SIDED exact 95% zero-failure upper bounds remain separate: 1-0.05**(1/n). These differ from the two-sided zero-event upper endpoint 1-0.025**(1/n). Admitted-only precision is descriptive; zero observed failures does not prove zero failure probability.',
        'reporting_revision':{'id':REPORTING_REVISION,'scope':'reporting_only',
            'frozen_configuration_preserved':True,'simulation_rerun':False,
            'paired_difference_inference':'unchanged paired-block normal approximation',
            'one_sided_exact_zero_failure_bounds':'preserved in separately named supplementary fields'},
        'coverage_scope':config['coverage_scope'],
        'admitted_precision_scope':'Admitted-only precision summaries are descriptive. Admitted-and-miss probabilities use all100 independent blocks per condition. No nominal conditional-on-admission coverage is claimed.',
        'status_interpretation':'finite means completed certificate; infinite is a mathematical envelope; unresolved is numerical computation, never proof of infeasibility. Evaluation fallback [0,1] is recorded separately and never counts as an admitted release.',
        'theory_interpretation':'Selector certificate <= baseline certificate follows structurally because the baseline belongs to the contiguous candidate family; it is checked with exact fractions, not an empirical superiority finding. Widths have no analogous theoretical ordering.',
        'report_seconds':time.perf_counter()-start,
    }
    if (root/'single_artifact_check.json').exists():
        audit = load(root/'single_artifact_check.json')
        summary['single_artifact_verification'] = {k:v for k,v in audit.items() if k!='records'}
    dump(root/'summary.json',summary)
    dump(root/'restricted_process_receipts.json',receipts)
    manifest['report_complete'] = True
    manifest['report_generated_utc'] = summary['generated_utc']
    manifest['report_rows'] = len(rows)
    dump(root/'manifest.json',manifest)
    print(json.dumps({'status':'report_complete','blocks':len(seen_blocks),'rows':len(rows),
                      'conditions':len(summaries),'report_seconds':summary['report_seconds']}),flush=True)
    return summary
