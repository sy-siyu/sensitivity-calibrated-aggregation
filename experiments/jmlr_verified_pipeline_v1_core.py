"""Complete verified finite-family selections on the preserved assignment counts.

This module draws no samples and performs no file I/O. Approximate proposals
enter only the unchanged root verifier. Region endpoints, optimization, search
feasibility, selected truth and realized-family oracle predicates are rational.
"""
from __future__ import annotations

from fractions import Fraction as Q
import math
import numbers
import time

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator, validate_law
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate


ALPHA = Q(1, 20)
CAPS = (Q(5, 4), Q(3, 2), Q(2))
MODES = ('known_mass', 'unknown_mass')
METHODS = ('coupled', 'independent')
FAMILIES = ('fixed', 'contiguous', 'combined')


def counts(N, T):
    N, T = tuple(N), tuple(T)
    if not N or len(N) != len(T):
        raise ValueError('Counts need one common nonempty support')
    if any(isinstance(n, bool) or isinstance(t, bool)
           or not isinstance(n, numbers.Integral) or not isinstance(t, numbers.Integral)
           or n < 0 or not 0 <= t <= n for n, t in zip(N, T)):
        raise ValueError('Counts require integers 0 <= T_i <= N_i')
    N, T = tuple(map(int, N)), tuple(map(int, T))
    if sum(N) <= 0:
        raise ValueError('Counts require a positive total')
    return N, T


def exact_order(N, T, immutable_ids=None):
    N, T = counts(N, T)
    ids = tuple(range(len(N))) if immutable_ids is None else tuple(immutable_ids)
    if len(ids) != len(N) or len(set(ids)) != len(N):
        raise ValueError('Propensity order requires distinct immutable IDs')
    return tuple(sorted(range(len(N)), key=lambda i: (Q(2 * T[i] + 1, 2 * N[i] + 2), ids[i])))


def canonical_partitions(K=4):
    """Historical lexicographic restricted-growth order, constant first."""
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise ValueError('Known support size must be a positive integer')
    strings = [(0,)]
    for _ in range(1, K):
        strings = [prefix + (label,) for prefix in strings for label in range(max(prefix) + 2)]
    return tuple(tuple(tuple(i for i, label in enumerate(labels) if label == cell)
                       for cell in range(max(labels) + 1)) for labels in strings)


def _verifier(verifier, root_policy):
    if verifier is not None and root_policy is not None:
        raise ValueError('Supply a root verifier or a policy, not both')
    if verifier is not None:
        return verifier
    if isinstance(root_policy, dict):
        root_policy = exact.RootPolicy(**root_policy)
    return exact.RootVerifier(root_policy)


def _decision_values(records, method):
    # A partial rejection record is not an upper bound. Its independent value
    # is complete but above every cap by the certified rejection predicate.
    # Failed records never authorize a new selection.
    return {cell: (record[method] if record['status'] == 'full' else None)
            for cell, record in records.items()}


def partition_certificate(partition, values):
    values = [Q(1) if len(cell) == 1 else values[tuple(cell)] for cell in partition]
    return None if any(v is None for v in values) else max(values, default=Q(1))


def select_exhaustive(partitions, values, cap):
    cap = exact.rational(cap)
    eligible = [(index, partition_certificate(partition, values))
                for index, partition in enumerate(partitions)]
    eligible = [(index, value) for index, value in eligible if value is not None and value <= cap]
    if not eligible:
        raise ArithmeticError('Known-safe identity is unavailable')
    index, value = min(eligible, key=lambda r: (len(partitions[r[0]]), r[1], r[0]))
    return {'partition': exact.canonical_partition(partitions[index]), 'cells': len(partitions[index]),
            'certificate': value, 'index': index}


def select_families(order, trees, values, cap):
    fixed = exact.select_family(trees, values, cap)
    contiguous = exact.select_contiguous(order, values, cap)
    return {'fixed': fixed, 'contiguous': contiguous,
            'combined': exact.select_combined(fixed, contiguous)}


def _coverage(region, law):
    return bool(region['valid'] and
                all(region['mass_lower'][i] <= law['p'][i] <= region['mass_upper'][i]
                    and region['propensity_lower'][i] <= law['e'][i] <= region['propensity_upper'][i]
                    for i in range(law['K'])))


def _oracle_row(family, cap, choice, evaluator, scope='realized estimated-order family'):
    truth = evaluator.partition(choice['partition'])
    if truth['value'] != choice['certificate'] or truth['value'] > cap:
        raise ArithmeticError('Exact family oracle selected an infeasible population partition')
    row = {'family': family, 'cap': cap, 'partition': truth['partition'],
           'count': truth['count'], 'value': truth['value'], 'scope': scope,
           'min_cell_mass': truth['min_cell_mass'], 'max_cell_size': truth['max_cell_size']}
    if 'index' in choice:
        row['index'] = choice['index']
    if 'cuts' in choice:
        row['cuts'] = choice['cuts']
    if 'constituent' in choice:
        row['constituent'] = choice['constituent']
    return row


def _comparison(row, old, evaluator):
    """Optional linkable comparison with a historical source selection."""
    if old is None:
        return None
    old_partition = exact.validate_partition(old['partition'], evaluator.K)
    old_count = old.get('count', old.get('cells', len(old_partition)))
    if old_count != len(old_partition):
        raise ValueError('Historical selected count disagrees with its partition')
    old_value = old.get('certificate', old.get('value'))
    if isinstance(old_value, dict):
        old_value = old_value['value']
    if old_value is None:
        old_finite, old_binary, old_display = False, None, None
    elif isinstance(old_value, Q):
        old_finite, old_binary, old_display = True, old_value, float(old_value)
    else:
        display = float(old_value)
        if math.isnan(display):
            raise ValueError('Historical selected certificate is NaN')
        old_finite, old_display = math.isfinite(display), display if math.isfinite(display) else None
        old_binary = Q.from_float(display) if old_finite else None
    old_truth = evaluator.partition(old_partition)['value']
    new_value = row['certificate']
    return {
        'old_partition': old_partition, 'old_count': old_count,
        'old_certificate': old_binary, 'old_displayed_certificate': old_display,
        'old_certificate_infinite': not old_finite,
        'count_changed': row['count'] != old_count, 'partition_changed': row['partition'] != old_partition,
        'certificate_direction': ('new-smaller' if not old_finite or new_value < old_binary else
                                  'equal' if new_value == old_binary else 'new-larger'),
        'old_partition_new_law_truth': old_truth,
        'old_partition_new_law_cap_violation': old_truth > row['cap'],
        'source_identity': old.get('source_identity'),
        'old_truth_display': old.get('true_distortion', old.get('truth')),
        'old_oracle_count': old.get('oracle_count'),
    }


def _selected_label(mode, method, family, cap, choice, records, evaluator, oracle, old=None):
    partition = exact.validate_partition(choice['partition'], evaluator.K)
    if any(records[cell]['status'] != 'full' or records[cell][method] is None for cell in partition):
        raise ArithmeticError('Returned selection lacks a completed cell certificate')
    certificate = max(records[cell][method] for cell in partition)
    if certificate != choice['certificate'] or certificate > cap:
        raise ArithmeticError('Actual returned certificate differs or exceeds its exact cap')
    safe = exact.safe_accept(partition, records, cap, evaluator.K, method)
    if not safe['accepted'] or safe['output_partition'] != partition:
        raise ArithmeticError('Actual verified accept/return path disagrees with selector')
    truth = evaluator.partition(partition)
    row = {'mode': mode, 'method': method, 'family': family, 'cap': cap,
           'partition': partition, 'count': len(partition), 'certificate': certificate,
           'cells': len(partition), 'value': certificate,
           'true_distortion': truth['value'], 'covered': truth['value'] <= certificate,
           'cap_violation': truth['value'] > cap, 'identity': len(partition) == evaluator.K,
           'compression': Q(evaluator.K - len(partition), evaluator.K),
           'min_cell_mass': truth['min_cell_mass'], 'max_cell_size': truth['max_cell_size'],
           'oracle_count': oracle['count'], 'oracle_gap': len(partition) - oracle['count'],
           'status': 'verified-selection', 'accepted': True,
           'cuts': choice.get('cuts', ()), 'constituent': choice.get('constituent'),
           'partition_index': choice.get('index')}
    row['historical'] = _comparison(row, old, evaluator)
    return row


def _diagnostics(region, records, truth_values, partitions=None):
    status_counts = {name: sum(r['status'] == name for r in records.values())
                     for name in ('full', 'rejected', 'boundary', 'failure')}
    full_coverage = {}
    infinity = {}
    for method in METHODS:
        complete = [(cell, record[method]) for cell, record in records.items()
                    if record['status'] in ('full', 'boundary')]
        full_coverage[method] = all(value is None or truth_values[cell] <= value for cell, value in complete)
        infinity[method] = sum(value is None for cell, value in complete)
    result = {'root_fallbacks': region['root_fallbacks'], 'region_valid': region['valid'],
              'cell_status_counts': status_counts, 'full_cell_covered': full_coverage,
              'complete_cell_count': sum(status_counts[name] for name in ('full', 'boundary')),
              'partial_rejection_count': status_counts['rejected'],
              'technical_cell_failures': status_counts['failure'], 'complete_cell_infinities': infinity,
              'mixture_calls': sum(r.get('mixture_calls', 0) for r in records.values()),
              'closure_calls': sum(r.get('closure_calls', 0) for r in records.values())}
    if partitions is not None:
        result['all_partition_covered'] = {
            method: status_counts['failure'] == 0 and all(
                (value := partition_certificate(partition, {c: r[method] for c, r in records.items()})) is None
                or max(truth_values[c] for c in partition) <= value for partition in partitions)
            for method in METHODS}
    return result


def _old_lookup(old, mode, method, family, cap):
    if old is None:
        return None
    if isinstance(old, dict):
        return old.get((mode, method, family, cap), old.get((method, cap)) if family == 'exhaustive' else None)
    matching = [row for row in old if (row['mode'], row['method'], row['family'], row['cap']) ==
                (mode, method, family, cap)]
    if len(matching) > 1:
        raise ValueError('Duplicate historical selection key')
    return matching[0] if matching else None


def evaluate_large(N, T, law, tree_orders, immutable_ids=None, saved_order=None,
                   verifier=None, root_policy=None, old=None, evaluator=None):
    """Complete fixed, contiguous and combined search, both modes/methods/caps."""
    start = time.perf_counter()
    N, T = counts(N, T)
    validate_law(law)
    K = len(N)
    if law['K'] != K:
        raise ValueError('Dataset and exact population law have different supports')
    ids = tuple(law['immutable_ids'] if immutable_ids is None else immutable_ids)
    if ids != tuple(law['immutable_ids']):
        raise ValueError('Dataset immutable IDs differ from exact law coordinates')
    order = exact_order(N, T, ids)
    tree_orders = tuple(tuple(o) for o in tree_orders)
    if len(tree_orders) != 2 or any(sorted(o) != list(range(K)) for o in tree_orders):
        raise ValueError('Larger-support family requires its two frozen full trees')
    trees = tuple(exact.build_tree(o) for o in tree_orders)
    cells = tuple(sorted(set(exact.interval_cells(order)) | set().union(*(set(tree) for tree in trees))))
    if saved_order is not None and sorted(saved_order) != list(range(K)):
        raise ValueError('Saved historical order is not a support permutation')
    evaluator = TruthEvaluator(law) if evaluator is None else evaluator
    if evaluator.law != law:
        raise ValueError('Cached evaluator belongs to a different exact law')
    tick = time.perf_counter()
    truth_values = {cell: evaluator.cell(cell)['value'] for cell in cells}
    oracles = []
    oracle_lookup = {}
    for cap in CAPS:
        choices = select_families(order, trees, truth_values, cap)
        for family in FAMILIES:
            row = _oracle_row(family, cap, choices[family], evaluator)
            oracles.append(row)
            oracle_lookup[family, cap] = row
    true_order = tuple(sorted(range(K), key=lambda i: (law['e'][i], ids[i])))
    true_cells = exact.interval_cells(true_order)
    true_values = {cell: evaluator.cell(cell)['value'] for cell in true_cells}
    true_order_oracles = [_oracle_row('contiguous', cap, exact.select_contiguous(true_order, true_values, cap),
                                      evaluator, 'true-propensity-order contiguous reference') for cap in CAPS]
    truth_seconds = time.perf_counter() - tick
    verifier = _verifier(verifier, root_policy)
    begin_hits, begin_misses = verifier.cache_hits, verifier.cache_misses
    modes = []
    for mode in MODES:
        tick = time.perf_counter()
        region = exact.confidence_region(N, T, ALPHA,
                                         known_masses=law['p'] if mode == 'known_mass' else None,
                                         verifier=verifier)
        root_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        records = {cell: cell_certificate(region, cell, max_cap=max(CAPS)) for cell in cells}
        calibration_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        labels = []
        for method in METHODS:
            values = _decision_values(records, method)
            for cap in CAPS:
                choices = select_families(order, trees, values, cap)
                for family in FAMILIES:
                    labels.append(_selected_label(mode, method, family, cap, choices[family], records,
                                                  evaluator, oracle_lookup[family, cap],
                                                  _old_lookup(old, mode, method, family, cap)))
        diagnostics = _diagnostics(region, records, truth_values)
        diagnostics['raw_law_covered'] = _coverage(region, law)
        diagnostics['occupied_states'] = sum(n > 0 for n in N)
        modes.append({'mode': mode, 'region': region, 'cells': [records[cell] for cell in cells],
                      'labels': labels, 'diagnostics': diagnostics,
                      'timings': {'root_seconds': root_seconds, 'calibration_seconds': calibration_seconds,
                                  'selection_seconds': time.perf_counter() - tick}})
    all_truth_cells = tuple(sorted(set(cells) | set(true_cells)))
    return {
        'kind': 'larger-support-complete-family', 'law_id': law['id'], 'N': N, 'T': T,
        'immutable_ids': ids, 'order': order,
        'saved_order': None if saved_order is None else tuple(saved_order),
        'order_changed': saved_order is not None and order != tuple(saved_order),
        'tree_orders': tree_orders, 'family_cells': cells,
        'truth_cells': [evaluator.cell(cell) for cell in all_truth_cells],
        'oracles': oracles, 'true_order': true_order, 'true_order_oracles': true_order_oracles,
        'modes': modes,
        'root_cache': {'hits': verifier.cache_hits - begin_hits, 'misses': verifier.cache_misses - begin_misses},
        'timings': {'truth_and_oracle_seconds': truth_seconds, 'compute_seconds': time.perf_counter() - start},
    }


def evaluate_four(N, T, law, mode, verifier=None, root_policy=None, old=None, evaluator=None):
    """All 15 partitions, full certificates, and both comparator selectors."""
    start = time.perf_counter()
    N, T = counts(N, T)
    validate_law(law)
    if len(N) != 4 or law['K'] != 4 or mode not in MODES:
        raise ValueError('Four-state evaluation requires K=4 and a prescribed mass mode')
    evaluator = TruthEvaluator(law) if evaluator is None else evaluator
    if evaluator.law != law:
        raise ValueError('Cached evaluator belongs to a different exact law')
    partitions = canonical_partitions(4)
    cells = tuple(sorted(set(cell for partition in partitions for cell in partition)))
    truth_values = {cell: evaluator.cell(cell)['value'] for cell in cells}
    oracles = [_oracle_row('exhaustive', cap, select_exhaustive(partitions, truth_values, cap), evaluator,
                           'all 15 partitions of the known four-state support') for cap in CAPS]
    verifier = _verifier(verifier, root_policy)
    begin_hits, begin_misses = verifier.cache_hits, verifier.cache_misses
    tick = time.perf_counter()
    region = exact.confidence_region(N, T, ALPHA, known_masses=law['p'] if mode == 'known_mass' else None,
                                     verifier=verifier)
    root_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    records = {cell: cell_certificate(region, cell) for cell in cells}
    calibration_seconds = time.perf_counter() - tick
    labels, certificates = [], []
    tick = time.perf_counter()
    for method in METHODS:
        values = _decision_values(records, method)
        for index, partition in enumerate(partitions):
            value = partition_certificate(partition, values)
            statuses = tuple(sorted({records[cell]['status'] for cell in partition}))
            status = 'failure' if 'failure' in statuses else 'boundary' if 'boundary' in statuses else 'full'
            truth = max(truth_values[cell] for cell in partition)
            certificates.append({'method': method, 'index': index, 'partition': partition,
                                 'value': value, 'status': status,
                                 'true_distortion': truth,
                                 'covered': status != 'failure' and (value is None or truth <= value),
                                 'infinite': status == 'boundary',
                                 'merged_cell_sizes': tuple(len(cell) for cell in partition if len(cell) > 1)})
        for cap, oracle in zip(CAPS, oracles):
            choice = select_exhaustive(partitions, values, cap)
            labels.append(_selected_label(mode, method, 'exhaustive', cap, choice, records,
                                          evaluator, oracle, _old_lookup(old, mode, method, 'exhaustive', cap)))
    diagnostics = _diagnostics(region, records, truth_values, partitions)
    diagnostics.update(raw_law_covered=_coverage(region, law),
                       occupied_states=sum(n > 0 for n in N),
                       minimum_state_count=min(N),
                       minimum_propensity_count=min(min(t, n - t) for n, t in zip(N, T)),
                       partition_infinities={m: sum(r['method'] == m and r['infinite'] for r in certificates) for m in METHODS},
                       partition_failures={m: sum(r['method'] == m and r['status'] == 'failure' for r in certificates) for m in METHODS})
    return {
        'kind': 'four-state-exhaustive', 'law_id': law['id'], 'mode': mode,
        'N': N, 'T': T, 'partitions': partitions, 'region': region,
        'cells': [records[cell] for cell in cells], 'truth_cells': [evaluator.cell(cell) for cell in cells],
        'oracles': oracles, 'partition_certificates': certificates, 'labels': labels,
        'diagnostics': diagnostics,
        'root_cache': {'hits': verifier.cache_hits - begin_hits, 'misses': verifier.cache_misses - begin_misses},
        'timings': {'root_seconds': root_seconds, 'calibration_seconds': calibration_seconds,
                    'selection_seconds': time.perf_counter() - tick, 'compute_seconds': time.perf_counter() - start},
    }


def replay_core(result, law):
    """Rebuild mathematical decisions with saved verified roots and fresh exact optimization.

    The caller validates each root sign with the rigorous backend and verifies
    source identity. This function never reevaluates special functions. It
    reconstructs every cell optimization/rejection, complete family selector,
    oracle, selected truth and diagnostic. Runtime/cache/historical display
    fields are intentionally outside this mathematical comparison.
    """
    class SavedVerifier:
        def __init__(self, regions):
            self.cache_hits = self.cache_misses = 0
            self.intervals = {}
            policies = [region['root_policy'] for region in regions]
            if any(policy != policies[0] for policy in policies):
                raise AssertionError('Saved core uses incompatible numerical policies')
            self.policy = exact.RootPolicy(**policies[0])
            for region in regions:
                for interval in list(region['roots']['propensity']) + list(region['roots']['mass'] or []):
                    key = interval['x'], interval['m'], interval['gamma']
                    if key in self.intervals and self.intervals[key] != interval:
                        raise AssertionError('Conflicting saved evidence for one mathematical CP interval')
                    self.intervals[key] = interval

        def interval(self, x, m, gamma):
            key = int(x), int(m), exact.rational(gamma)
            if key not in self.intervals:
                raise AssertionError('Core replay lacks a required exact count/error root')
            self.cache_hits += 1
            return self.intervals[key]

    kind = result['kind']
    if kind == 'larger-support-complete-family':
        verifier = SavedVerifier([mode['region'] for mode in result['modes']])
        rebuilt = evaluate_large(result['N'], result['T'], law, result['tree_orders'],
                                 immutable_ids=result['immutable_ids'], saved_order=result['saved_order'],
                                 verifier=verifier)
        label_count = sum(len(mode['labels']) for mode in result['modes'])
        cells = sum(len(mode['cells']) for mode in result['modes'])
    elif kind == 'four-state-exhaustive':
        verifier = SavedVerifier([result['region']])
        rebuilt = evaluate_four(result['N'], result['T'], law, result['mode'], verifier=verifier)
        label_count, cells = len(result['labels']), len(result['cells'])
    else:
        raise ValueError('Unknown verified core result kind')

    def mathematical(value):
        if isinstance(value, dict):
            return {key: mathematical(item) for key, item in value.items()
                    if key not in ('timings', 'root_cache', 'historical') and not key.startswith('_')}
        if isinstance(value, (list, tuple)):
            return tuple(mathematical(item) for item in value)
        return value

    if mathematical(result) != mathematical(rebuilt):
        raise AssertionError('Reconstructed exact core decisions/population diagnostics differ')
    return {'status': 'passed', 'labels': label_count, 'cells': cells,
            'unique_saved_intervals': len(verifier.intervals)}
