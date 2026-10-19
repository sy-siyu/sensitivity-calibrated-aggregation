"""Export the three current main-text tables from complete public reports.

No inference, population generation, manuscript parsing, or stored empirical
answers enter this module. Incomplete smoke reports are intentionally refused.
All paths are relative to the supplied reproduction workspace.
"""
from collections import Counter
import csv
from fractions import Fraction as F
import json
import math
from pathlib import Path

from experiments.jmlr_statistical_baselines_v1_reporting import grid_interval, paired
from experiments.jmlr_precision_frontier_v1_reporting import COMPONENTS, _decomposition_check
from experiments.jmlr_release_capacity_v1_reporting import _statistics


CAP = F(3, 2)
CONDITIONS = tuple((f'K{K}_{mass}_{overlap}_{geometry}', r)
                   for K in (16, 64) for mass in ('balanced', 'skewed')
                   for overlap in ('moderate', 'weak') for geometry in ('ordered', 'permuted')
                   for r in (250, 1000, 4000))
CONTRASTS = ('C-B', 'D-C', 'E-D', 'F-E', 'C-A', 'F-C', 'F-A', 'B-A')
PROFILES = ('aligned', 'alternating')


def _rows(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Required complete reporter output is missing: {path}')
    with path.open(newline='', encoding='utf-8') as stream:
        yield from csv.DictReader(stream)


def _q(value):
    try:
        return F(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f'Expected an available exact rational, received {value!r}') from error


def _int(value):
    result = _q(value)
    if result.denominator != 1:
        raise ValueError('Expected an integer')
    return result.numerator


def _bool(value):
    if value in (True, 'True', 'true', '1'): return True
    if value in (False, 'False', 'false', '0'): return False
    raise ValueError(f'Expected an available boolean, received {value!r}')


def _same(actual, expected, name):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11):
        raise ValueError(f'Reporter summary disagrees with dataset values: {name}')


def _case(row):
    condition = (row['id'], _int(row['r']))
    K, rep = _int(row['K']), _int(row['replication'])
    if K not in (16, 64) or not condition[0].startswith(f'K{K}_'):
        raise ValueError('State count and condition identifier disagree')
    return condition, rep, K


def _put(index, key, value):
    if key in index:
        raise ValueError(f'Duplicate reporter row: {key}')
    index[key] = value


def _complete(index, labels, reps, *, conditions=CONDITIONS):
    expected = {(condition, rep, label) for condition in conditions for rep in reps for label in labels}
    actual = set(index)
    if actual != expected:
        raise ValueError(f'Incomplete or incompatible paper panel: expected {len(expected)} unique rows, '
                         f'found {len(actual)}; missing {len(expected-actual)}, unexpected {len(actual-expected)}')


def _grid(groups):
    """Exact mean and the original equal fixed-condition MC variance."""
    if not groups or any(not values for values in groups):
        raise ValueError('Cannot summarize an empty condition')
    stats = [{'x_'+key: value for key, value in paired(list(map(float, values))).items()}
             for values in groups]
    result = grid_interval(stats, 'x', expected_conditions=len(groups))
    result['exact_mean'] = sum((sum(values, F())/len(values) for values in groups), F())/len(groups)
    _same(result['mean'], result['exact_mean'], 'equal-condition exact mean')
    return result


def _metric(index, label, name, conditions, reps):
    return _grid([[index[condition, rep, label][name] for rep in reps] for condition in conditions])


def _prefix(name, result):
    return {name+'_'+key: result[key] for key in
            ('mean', 'exact_mean', 'MCSE', 'analytic_Normal95_lower', 'analytic_Normal95_upper')}


def _assignment(rows, *, conditions=CONDITIONS, reps=range(100)):
    labels = tuple((family, method) for family in ('fixed', 'contiguous', 'combined')
                   for method in ('coupled', 'independent'))
    index = {}
    for row in rows:
        if row['mode'] != 'unknown_mass' or _q(row['cap']) != CAP: continue
        label = (row['family'], row['method'])
        if label not in labels: continue
        condition, rep, K = _case(row)
        count, oracle = _int(row['count']), _int(row['oracle_count'])
        if not 1 <= count <= K or not 1 <= oracle <= K:
            raise ValueError('Invalid selected/oracle count')
        compression, gap = F(100*(K-count), K), F(100*(count-oracle), K)
        if _q(row['compression'])*100 != compression or _int(row['oracle_gap']) != count-oracle:
            raise ValueError('Assignment compression or same-family oracle gap is inconsistent')
        certificate = _q(row['certificate'])
        if not 1 <= certificate <= CAP:
            raise ValueError('Selected assignment certificate is not completed within cap')
        _put(index, (condition, rep, label), {'count': count, 'oracle': oracle, 'K': K,
             'compression': compression, 'gap': gap, 'certificate': certificate})
    _complete(index, labels, reps, conditions=conditions)
    table = []
    for family, name in (('fixed', 'Fixed trees'), ('contiguous', 'Contiguous'), ('combined', 'Combined')):
        label = (family, 'coupled')
        for condition in conditions:
            for rep in reps:
                coupled, independent = index[condition, rep, label], index[condition, rep, (family, 'independent')]
                if coupled['oracle'] != independent['oracle']:
                    raise ValueError('Coupling comparison changes the population-oracle family')
                coupled['coupling'] = F(100*(independent['count']-coupled['count']), coupled['K'])
        recoveries = sum(index[c, r, label]['count'] == index[c, r, label]['oracle'] for c in conditions for r in reps)
        table.append({'family': name, 'confidence_region': 'Coupled CP',
            **_prefix('compression_percent', _metric(index, label, 'compression', conditions, reps)),
            **_prefix('coupling_gain_pp', _metric(index, label, 'coupling', conditions, reps)),
            **_prefix('oracle_gap_percent_K', _metric(index, label, 'gap', conditions, reps)),
            'recoveries': recoveries, 'datasets': len(conditions)*len(reps), 'conditions': len(conditions),
            'certificate_scope': 'completed selected certificate; global search-completion status is unavailable in this reporter table',
            'uncertainty': 'paired within-condition variation; equal fixed-condition normal MC interval'})
    return table, index


def _efficiency(rows, summaries, assignment, *, conditions=CONDITIONS, reps=range(100)):
    labels = ('original', 'allocation_control', 'pooled_alternative')
    index = {}
    for row in rows:
        if _q(row['cap']) != CAP or row.get('mass_mode', 'unknown_mass') != 'unknown_mass': continue
        label = row['procedure']
        if label not in labels: continue
        condition, rep, K = _case(row)
        count, oracle = _int(row['selected_count']), _int(row['oracle_count'])
        if not 1 <= count <= K or not 1 <= oracle <= K: raise ValueError('Invalid efficiency count')
        compression, gap = F(100*(K-count), K), F(100*(count-oracle), K)
        if _q(row['compression'])*100 != compression or _q(row['normalized_count_gap'])*100 != gap:
            raise ValueError('Efficiency compression or oracle gap is inconsistent')
        if _bool(row['count_recovered']) != (count == oracle):
            raise ValueError('Efficiency recovery indicator disagrees with actual threshold count')
        value = {'count': count, 'oracle': oracle, 'compression': compression, 'gap': gap,
                 'recovered': count == oracle, 'status': row['selection_status']}
        _put(index, (condition, rep, label), value)
        source = assignment[condition, rep, ('combined', 'coupled')]
        if label == 'original' and (count != source['count'] or oracle != source['oracle']
                                    or _q(row['selected_certificate']) != source['certificate']):
            raise ValueError('Efficiency original does not preserve the assignment SCA selection')
    _complete(index, labels, reps, conditions=conditions)
    reported = {}
    for row in summaries:
        if row['procedure'] in labels and row['metric'] in ('compression', 'normalized_count_gap', 'count_recovered'):
            _put(reported, (row['procedure'], row['metric']), row)
    output = []
    for label in labels:
        for metric, key in (('compression', 'compression'), ('normalized_count_gap', 'gap'), ('count_recovered', 'recovered')):
            saved = reported[label, metric]
            if _int(saved['conditions_equal_weighted']) != len(conditions) or _int(saved['n']) != len(reps):
                raise ValueError('Efficiency summary has an incomplete denominator')
            values = _metric(index, label, key, conditions, reps)
            factor = 1 if key == 'recovered' else 100
            _same(factor*_q(saved['mean']), values['exact_mean'], f'efficiency {label} {metric}')
        if label == 'original': continue
        entry = {'family': 'Combined', 'confidence_region':
                 'Allocation control' if label == 'allocation_control' else 'Fixed-group pooled',
                 'recoveries': sum(index[c, r, label]['recovered'] for c in conditions for r in reps),
                 'datasets': len(conditions)*len(reps), 'conditions': len(conditions),
                 'selection_unresolved': sum(index[c, r, label]['status'] != 'finite' for c in conditions for r in reps),
                 'uncertainty': 'retained efficiency reporter: normal MC interval over replication-index condition averages'}
        for metric, name, key in (('compression', 'compression_percent', 'compression'),
                                  ('normalized_count_gap', 'oracle_gap_percent_K', 'gap')):
            saved = reported[label, metric]
            entry.update({name+'_mean': 100*float(saved['mean']),
                          name+'_exact_mean': _metric(index, label, key, conditions, reps)['exact_mean'],
                          name+'_MCSE': 100*float(saved['se']),
                          name+'_analytic_Normal95_lower': 100*float(saved['ci95_lower']),
                          name+'_analytic_Normal95_upper': 100*float(saved['ci95_upper'])})
        output.append(entry)
    return output, {'original_assignment_matches': len(conditions)*len(reps),
                    'procedure_datasets': len(index), 'primary_cap': str(CAP)}


def _saved_grid(row, prefix, groups):
    result = _grid(groups)
    for suffix, expected in (('conditions', len(groups)), ('expected_conditions', len(groups)),
                             ('paired_datasets', sum(map(len, groups)))):
        if _int(row[prefix+'_'+suffix]) != expected:
            raise ValueError('Split grid summary has an incomplete denominator')
    if not _bool(row[prefix+'_all_conditions_present']):
        raise ValueError('Split grid summary declares missing conditions')
    for suffix in ('mean', 'MCSE', 'analytic_Normal95_lower', 'analytic_Normal95_upper'):
        _same(row[prefix+'_'+suffix], result[suffix], 'split '+prefix+' '+suffix)
        result[suffix] = float(row[prefix+'_'+suffix])
    return result


def _split(rows, method_grid, paired_rows, paired_grid, *, conditions=CONDITIONS, reps=range(10, 100)):
    index = {}
    for row in rows:
        if _q(row['cap']) != CAP: continue
        condition, rep, K = _case(row)
        cells = _int(row['cells'])
        if not 1 <= cells <= K: raise ValueError('Invalid split selected count')
        value = F(100*(K-cells), K)
        if _q(row['compression_pp']) != value: raise ValueError('Split compression disagrees with selected count')
        _put(index, (condition, rep, row['procedure']), {'compression': value, 'cells': cells,
             'optimal': _bool(row['count_optimal']), 'status': row['computation_status'],
             'complete': _bool(row['computation_complete']),
             'tie_complete': _bool(row['deterministic_tie_complete']),
             'identity': _bool(row['identity']), 'fallback': _bool(row['identity_fallback'])})
    _complete(index, tuple('ABCDEF'), reps, conditions=conditions)
    methods = {}
    for row in method_grid:
        if row['scope'] == 'all' and _q(row['cap']) == CAP: _put(methods, row['procedure'], row)
    if set(methods) != set('ABCDEF'): raise ValueError('Expected all six split grid procedures')
    method_table = []
    for label in 'ABCDEF':
        groups = [[index[c, r, label]['compression'] for r in reps] for c in conditions]
        result = _saved_grid(methods[label], 'compression_pp', groups)
        method_table.append({'procedure': label, 'procedure_name': methods[label]['procedure_name'],
            **_prefix('compression_percent', result), 'conditions': len(conditions),
            'datasets': len(conditions)*len(reps),
            'identity_count': sum(index[c, r, label]['identity'] for c in conditions for r in reps),
            'identity_fallback_count': sum(index[c, r, label]['fallback'] for c in conditions for r in reps),
            'selection_unresolved': sum(not index[c, r, label]['complete'] for c in conditions for r in reps),
            'count_not_proved_optimal': sum(not index[c, r, label]['optimal'] for c in conditions for r in reps),
            'deterministic_tie_incomplete': sum(not index[c, r, label]['tie_complete'] for c in conditions for r in reps)})
    pairs, structural = {}, Counter()
    for row in paired_rows:
        if _q(row['cap']) != CAP: continue
        condition, rep, _ = _case(row)
        values = {label: index[condition, rep, label]['compression'] for label in 'ABCDEF'}
        differences = {contrast: values[contrast[0]]-values[contrast[2]] for contrast in CONTRASTS}
        if any(_q(row[contrast+'_pp']) != value for contrast, value in differences.items()):
            raise ValueError('Split saved paired contrast differs from matched dataset outputs')
        if differences['F-A'] != sum((differences[x] for x in ('B-A', 'C-B', 'D-C', 'E-D', 'F-E')), F()):
            raise ValueError('Split signed accounting is not exact')
        if not _bool(row['accounting_identity_exact']): raise ValueError('Split saved exact-accounting flag is false')
        _put(pairs, (condition, rep, 'paired'), differences)
        for left, right in (('C', 'B'), ('E', 'D')):
            a, b = index[condition, rep, left], index[condition, rep, right]
            if a['optimal'] and a['complete'] and b['complete']:
                structural[left+'_le_'+right] += 1
                if a['cells'] > b['cells']: raise ValueError('Completed split structural inclusion failed')
    _complete(pairs, ('paired',), reps, conditions=conditions)
    grids = {}
    for row in paired_grid:
        if row['scope'] == 'all' and _q(row['cap']) == CAP: _put(grids, row['contrast'], row)
    if set(grids) != set(CONTRASTS): raise ValueError('Expected all eight split grid contrasts')
    contrasts = []
    for contrast in CONTRASTS:
        groups = [[pairs[c, r, 'paired'][contrast] for r in reps] for c in conditions]
        saved = grids[contrast]
        result = _saved_grid(saved, 'difference_pp', groups)
        values = [v for group in groups for v in group]
        counts = {'wins': sum(v > 0 for v in values), 'ties': sum(v == 0 for v in values),
                  'losses': sum(v < 0 for v in values)}
        condition_means = [sum(group, F())/len(group) for group in groups]
        counts.update(conditions_mean_positive=sum(v > 0 for v in condition_means),
                      conditions_mean_zero=sum(v == 0 for v in condition_means),
                      conditions_mean_negative=sum(v < 0 for v in condition_means))
        if any(_int(saved[name]) != count for name, count in counts.items()):
            raise ValueError('Split win/tie/loss counts disagree with dataset contrasts')
        entry = {'contrast': contrast, **_prefix('difference_pp', result), **counts,
                 'datasets': len(values), 'conditions': len(conditions)}
        for name in ('wins', 'ties', 'losses'):
            entry[name+'_fraction'] = float(F(counts[name], len(values)))
            entry[name+'_fraction_exact'] = F(counts[name], len(values))
        contrasts.append(entry)
    return method_table, contrasts, {'datasets': len(pairs), 'accounting_exact': len(pairs),
        'structural_checks': dict(structural), 'status_counts': dict(Counter(v['status'] for v in index.values())),
        'uncertainty': 'paired fixed-condition variance H^-2 sum_h s_h^2/n_h; descriptive fractions have no pooled-binomial interval'}


def _precision(rows, summaries, *, reps=range(100)):
    index, diagnostics = {}, {p: Counter() for p in PROFILES}
    for row in rows:
        if (_int(row['law_index']) != 0 or _q(row['cap']) != CAP or _int(row['m_ratio']) != 256
                or row['profile'] not in PROFILES): continue
        if row['law_id'] != 'K64_balanced_moderate_ordered' or _int(row['K']) != 64:
            raise ValueError('Precision table law differs from its declared balanced population')
        if (_q(row['transported_budget']) != 2*_q(row['certificate'])
                or _q(row['budget_ceiling']) != 2*CAP):
            raise ValueError('Precision table must retain actual certified budget, separately from the cap ceiling')
        profile, rep = row['profile'], _int(row['replication'])
        value = {key: _q(row[key]) for key in
                 [f'global_I{j}_{name}' for j in range(5) for name in ('lower', 'upper', 'width')]
                 +['global_'+name for name in COMPONENTS+('total_excess',)]
                 +['z_global_'+name for name in ('lower', 'upper', 'width')]}
        if _decomposition_check(value, 'global') is not True or not _bool(row['global_accounting_exact']):
            raise ValueError('Precision signed stage accounting is unavailable or inconsistent')
        nested = []
        for j in range(4):
            contained = (value[f'global_I{j+1}_lower'] <= value[f'global_I{j}_lower']
                         <= value[f'global_I{j}_upper'] <= value[f'global_I{j+1}_upper'])
            if contained != _bool(row[f'global_I{j}_in_I{j+1}']): raise ValueError('Precision saved nesting flag disagrees with endpoints')
            nested.append(contained)
        if all(nested) != _bool(row['global_nesting']): raise ValueError('Precision overall nesting flag disagrees with endpoints')
        shared = _bool(row['shared_coverage_event'])
        miss_on_shared = shared and not all(nested)
        if miss_on_shared != _bool(row['global_nesting_failure_on_shared_event']):
            raise ValueError('Precision shared-event nesting flag disagrees')
        d = diagnostics[profile]
        d.update(blocks=1, accounting_exact=1, nesting_failures=not all(nested),
                 shared_coverage_blocks=shared, nesting_failures_on_shared_event=miss_on_shared)
        for j, contained in enumerate(nested):
            d[f'I{j}_in_I{j+1}_blocks'] += contained
            d[f'I{j}_in_I{j+1}_failures'] += not contained
        for name in COMPONENTS: d[name+'_negative'] += value['global_'+name] < 0
        _put(index, (profile, rep, 'global'), value)
    _complete(index, ('global',), reps, conditions=PROFILES)
    saved = {}
    for row in summaries:
        if (_int(row['law_index']) == 0 and _q(row['cap']) == CAP and _int(row['m_ratio']) == 256
                and row['transport'] == 'global' and row['profile'] in PROFILES):
            _put(saved, row['profile'], row)
    if set(saved) != set(PROFILES): raise ValueError('Both precision profile summaries are required')
    stages, increments = [], []
    for j in range(5):
        entry = {'stage': 'I'+str(j)}
        for profile in PROFILES:
            source = saved[profile]
            if any(_int(source[key]) != len(reps) for key in ('n_blocks', 'n_expected_blocks', f'I{j}_width_n')):
                raise ValueError('Precision width summary has an incomplete denominator')
            if _int(source['n_missing_blocks']) != 0:
                raise ValueError('Precision summary declares missing blocks')
            values = [index[profile, r, 'global'][f'global_I{j}_width'] for r in reps]
            # Exact endpoints/accounting are checked per block. Rational sums
            # of unrelated endpoint denominators need not be compact payloads.
            stats = _statistics(values)
            for suffix in ('mean', 'ci95_lower', 'ci95_upper'):
                _same(source[f'I{j}_width_'+suffix], stats[suffix], f'precision {profile} I{j} {suffix}')
            entry.update({profile+'_mean': float(source[f'I{j}_width_mean']),
                          profile+'_ci95_lower': float(source[f'I{j}_width_ci95_lower']),
                          profile+'_ci95_upper': float(source[f'I{j}_width_ci95_upper']), profile+'_blocks': len(reps)})
        stages.append(entry)
    for name in COMPONENTS:
        entry = {'increment': name}
        for profile in PROFILES:
            values = [index[profile, r, 'global']['global_'+name] for r in reps]
            _same(saved[profile][name+'_mean'], _statistics(values)['mean'], f'precision {profile} {name}')
            entry.update({profile+'_mean': float(saved[profile][name+'_mean']),
                          profile+'_negative_blocks': sum(v < 0 for v in values)})
        increments.append(entry)
    return stages, increments, {p: dict(counts) for p, counts in diagnostics.items()}


def _csv(path, rows):
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def generate(workspace, output_dir=None):
    """Write main-table CSVs only after all required complete panels validate.

    Repeated methods/caps do not add datasets. Table3 includes all100 blocks
    in each profile, regardless of admission. Stage increments remain signed.
    """
    workspace = Path(workspace).resolve()
    paths = {
        'assignment': 'assignment/tables/assignment_rows.csv',
        'efficiency': 'efficiency/tables/historical_rows.csv',
        'efficiency_summary': 'efficiency/tables/historical_primary_equal_weight_summary.csv',
        'split': 'split/tables/dataset_rows.csv', 'split_methods': 'split/tables/methods_grid.csv',
        'split_pairs': 'split/tables/paired_dataset_rows.csv', 'split_contrasts': 'split/tables/paired_grid.csv',
        'precision': 'precision/tables/analysis_replications.csv',
        'precision_summary': 'precision/tables/decomposition_summary.csv'}
    read = lambda name: _rows(workspace/paths[name])
    first, assignment = _assignment(read('assignment'))
    extra, efficiency_checks = _efficiency(read('efficiency'), read('efficiency_summary'), assignment)
    methods, contrasts, split_checks = _split(read('split'), read('split_methods'),
                                             read('split_pairs'), read('split_contrasts'))
    stages, increments, precision_checks = _precision(read('precision'), read('precision_summary'))
    outputs = {'table1_assignment_search.csv': first+extra, 'table2_split_procedures.csv': methods,
               'table2_split_contrasts.csv': contrasts, 'table3_precision_stages.csv': stages,
               'table3_precision_increments.csv': increments}
    destination = workspace/'paper_tables' if output_dir is None else Path(output_dir).resolve()
    # Prevent an output argument from overwriting a reporter input.
    if any((destination/name).resolve() in {(workspace/p).resolve() for p in paths.values()} for name in outputs):
        raise ValueError('Paper-table output cannot overwrite a reporter input')
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items(): _csv(destination/name, rows)
    manifest = {'status': 'complete', 'cap': str(CAP), 'input_tables': paths,
        'outputs': {name: len(rows) for name, rows in outputs.items()},
        'assignment': {'independent_datasets': 4800, 'fixed_conditions': 48, 'replications_per_condition': 100,
                       'same_family_oracle': True, 'coupling_comparison': 'paired independent minus coupled count, normalized by K'},
        'efficiency_checks': efficiency_checks, 'split_checks': split_checks,
        'precision_checks': precision_checks,
        'precision_scope': {'law_index': 0, 'law_id': 'K64_balanced_moderate_ordered', 'profiles': list(PROFILES),
            'm_ratio': 256, 'method': 'z_global', 'blocks_per_profile': 100, 'admission_filter': False,
            'I3': 'privileged true assignment and operational outcome CP bounds at actual certified transported budget'},
        'scope': 'Numerical extraction of current main tables from complete public reporter outputs; no new inference or manuscript edits'}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return manifest
