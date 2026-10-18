"""Assignment-only reanalysis of preserved calibration and release blocks.

Historical threshold selections are retained as threshold selections; the new
minimax diagnostic is computed separately at the same-family oracle count.
Only new evidence is returned.  This module never modifies its source files,
uses outcomes, or draws a sample.
"""
from __future__ import annotations

from fractions import Fraction as Q
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import time
from types import MappingProxyType

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import select_families
from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator
from experiments.jmlr_release_capacity_v1_search import candidate_cells, cell_score


PROJECT = Path(__file__).resolve().parents[1]
CAPS = (Q(5, 4), Q(3, 2), Q(2))
# Both execution variants are checked against fully completed search. The
# bounded shared-load preflight retained optimistic refinement as the default;
# the optional batch variant is preserved to reproduce that development check.
BATCH_UPPER_RELEVANCE = False
TIMING_INTERPRETATION = (
    'Incremental process wall times on reused assignment inputs and source evidence; '
    'a worker may share its verified-root cache across records. Raw and pooled '
    'confidence construction is shared by allocation_control and pooled_alternative. '
    'Pooled fusion is the incremental alternative operation after control raw '
    'certification. These are not fair cold end-to-end method timings. Saved '
    'original timings retain their source execution scope.')


def _measure(timings, key, function, *args, **kwargs):
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings[key] = timings.get(key, 0.) + time.perf_counter()-started


def _timings():
    return {key: 0. for key in (
        'source_load_and_setup_seconds',
        'shared_raw_plus_pooled_confidence_construction_seconds',
        'allocation_control_raw_certification_seconds',
        'pooled_incremental_fusion_seconds',
        'pooled_incremental_refinement_certification_seconds',
        'original_missing_completion_seconds',
        'minimax_search_seconds', 'threshold_search_seconds',
        'mass_slice_diagnostics_seconds', 'other_analytical_diagnostics_seconds')}


def _load(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return exact.decode(json.load(stream))
    return exact.decode(json.loads(path.read_text()))


def _partition(partition):
    return exact.canonical_partition(partition)


def _restore(region):
    """Rebuild private integer caches from the saved authoritative endpoints."""
    rebuilt = exact.region_from_bounds(
        region['mass_lower'], region['mass_upper'],
        region['propensity_lower'], region['propensity_upper'],
        region.get('known_masses'))
    rebuilt.update({k: v for k, v in region.items() if not k.startswith('_')})
    return rebuilt


def _merge(left, right):
    # Cells are already canonical internally.  Avoid sorting every state's
    # membership repeatedly in the innermost minimax recurrence.
    return tuple(sorted(left + right))


@lru_cache(maxsize=4)
def _geometry(order, tree_orders):
    """Bounded process-local cache of immutable, score-free search geometry."""
    cells = candidate_cells(order, tree_orders)
    K = len(order)
    intervals = MappingProxyType({(i,j):tuple(sorted(order[i:j]))
                                 for j in range(1,K+1) for i in range(j)})
    predecessors = tuple(tuple(tuple(range(b-1,j)) for b in range(1,j+1))
                         for j in range(K+1))
    trees = tuple(MappingProxyType(exact.build_tree(leaves)) for leaves in tree_orders)
    roots = tuple(max(tree,key=len) for tree in trees)
    return cells,intervals,predecessors,trees,roots


def rank_capacity_search(order, tree_orders, values, capacities):
    """Exact bounded minimax DP with order-preserving integer score ranks.

    Rational scores are sorted exactly once; all subsequent comparisons are
    integer comparisons.  Local canonical ties agree with capacity_search.
    Lower and completed-upper tables keep unresolved computations distinct.
    Computation is restricted to counts at most max(capacities).
    """
    order = tuple(order)
    tree_orders = tuple(tuple(o) for o in tree_orders)
    cells,intervals,predecessors,trees,roots = _geometry(order,tree_orders)
    K = len(order)
    capacities = tuple(sorted(set(capacities)))
    if not capacities or any(isinstance(b, bool) or not isinstance(b, int) or not 1 <= b <= K for b in capacities):
        raise ValueError('Capacities must be integers in [1,K]')
    Bmax = max(capacities)
    scores = {c: cell_score(Q(1) if len(c) == 1 else values[c]) for c in cells}
    finite = sorted({Q(1)} | {s.lower for s in scores.values() if s.lower is not None}
                    | {s.upper for s in scores.values() if s.completed and s.upper is not None})
    ranks = {value: i + 1 for i, value in enumerate(finite)}
    infinity = len(finite) + 1
    back = {i + 1: value for i, value in enumerate(finite)}
    back[infinity] = None
    ranked = {c: (infinity if s.lower is None else ranks[s.lower],
                   (infinity if s.upper is None else ranks[s.upper]) if s.completed else None)
              for c, s in scores.items()}

    # Entries: lower score, lower partition, completed upper score/partition.
    def join_candidates(pairs):
        low, lp, high, hp = infinity + 1, None, infinity + 1, None
        for left, right in pairs:
            score = max(left[0], right[0])
            part = None
            if score <= low:
                part = _merge(left[1], right[1])
                if score < low or lp is None or part < lp:
                    low, lp = score, part
            if score == infinity:
                upper = infinity
                upart = part if part is not None else _merge(left[1], right[1])
            elif left[3] is not None and right[3] is not None:
                upper = max(left[2], right[2])
                if upper > high:
                    continue
                upart = _merge(left[3], right[3])
            else:
                continue
            if upper < high or hp is None or upper == high and upart < hp:
                high, hp = upper, upart
        if lp is None:
            raise ArithmeticError('A reachable recurrence lost its predecessor')
        return low, lp, high if hp is not None else None, hp

    def local(cell):
        lower, upper = ranked[cell]
        return lower, (cell,), upper, (cell,) if upper is not None else None

    H = [dict() for _ in range(K + 1)]
    H[0][0] = ranks[Q(1)], (), ranks[Q(1)], ()
    for j in range(1, K + 1):
        for b in range(1, min(j, Bmax) + 1):
            H[j][b] = join_candidates((H[i][b - 1], local(intervals[i, j]))
                                      for i in predecessors[j][b-1] if b - 1 in H[i])
    families = {'contiguous': H[K]}
    for index,(tree,root) in enumerate(zip(trees,roots)):
        def visit(cell):
            answer = {1: local(cell)}
            children = tree[cell]
            if children is not None:
                left, right = (visit(child) for child in children)
                for b in range(2, min(len(cell), Bmax) + 1):
                    answer[b] = join_candidates((left[lc], right[b-lc])
                                                for lc in sorted(left) if b-lc in right)
            return answer
        families['tree-' + str(index)] = visit(root)

    result = {}
    for B in capacities:
        options = [(entry, family) for family, table in families.items()
                   for count, entry in table.items() if count <= B]
        lower, lfamily = min(options, key=lambda pair: (pair[0][0], len(pair[0][1]), pair[0][1], pair[1]))
        uppers = [(entry, family) for entry, family in options if entry[3] is not None]
        if uppers:
            upper, ufamily = min(uppers, key=lambda pair: (pair[0][2], len(pair[0][3]), pair[0][3], pair[1]))
        else:
            upper, ufamily = None, None
        if lower[0] == infinity:
            status, part, family, cert = 'infinite', lower[1], lfamily, None
        elif upper is not None and upper[2] == lower[0]:
            status, part, family, cert = 'finite', upper[3], ufamily, back[upper[2]]
        else:
            status, part, family, cert = 'unresolved', None, None, None
        result[B] = {
            'capacity': B, 'status': status, 'completed': status != 'unresolved',
            'certificate': cert, 'partition': part, 'family': family,
            'cells': None if part is None else len(part), 'lower_bound': back[lower[0]],
            'lower_bound_infinite': lower[0] == infinity,
            'optimistic_partition': lower[1],
            'candidate_partition': None if upper is None else upper[3],
            'candidate_certificate': None if upper is None else back[upper[2]],
            'candidate_status': ('unavailable' if upper is None else
                                 'infinite' if upper[2] == infinity else 'finite'),
        }
    return result


def _compact(record):
    keys = ('cell', 'status', 'coupled', 'rejection_bound',
            'valid_upper_bound', 'exact_min_completed', 'reason', 'raw_bound', 'pooled_bound',
            'raw_status', 'pooled_status', 'winner', 'mixture_calls', 'closure_calls')
    return {k: record[k] for k in keys if k in record}


def _partition_record(partition, records):
    partition = _partition(partition)
    rec = [records[c] for c in partition]
    if any(r['status'] == 'boundary' for r in rec):
        return 'infinite', None
    if any(r['status'] != 'full' or r.get('coupled') is None for r in rec):
        return 'unresolved', None
    return 'finite', max(r['coupled'] for r in rec)


def _relevant_cells(order, tree_orders, records, B, certificate):
    """Cells occurring in a lower-bound-feasible at-most-B partition.

    Completing this set (or proving its cells strictly above certificate)
    preserves the canonical local recurrence ties as well as the optimum.
    """
    K = len(order)
    _,intervals,_,trees,roots = _geometry(tuple(order),tuple(tuple(o) for o in tree_orders))
    allowed = {c: cell_score(record).lower is not None
               and cell_score(record).lower <= certificate for c, record in records.items()}
    unreachable = K+1
    prefix, suffix = [unreachable]*(K+1), [unreachable]*(K+1)
    prefix[0], suffix[K] = 0, 0
    for j in range(1,K+1):
        prefix[j] = min((prefix[i]+1 for i in range(j) if allowed[intervals[i,j]]), default=unreachable)
    for i in range(K-1,-1,-1):
        suffix[i] = min((suffix[j]+1 for j in range(i+1,K+1) if allowed[intervals[i,j]]), default=unreachable)
    relevant = {cell for (i,j),cell in intervals.items()
                if allowed[cell] and prefix[i]+1+suffix[j] <= B}
    for tree,root in zip(trees,roots):
        minimum = {}
        def down(cell):
            children = tree[cell]
            split = unreachable if children is None else sum(down(child) for child in children)
            minimum[cell] = min(1 if allowed[cell] else unreachable, split)
            return minimum[cell]
        down(root)
        def up(cell, outside):
            if allowed[cell] and outside+1 <= B:
                relevant.add(cell)
            children = tree[cell]
            if children is not None:
                left,right = children
                up(left,outside+minimum[right]);up(right,outside+minimum[left])
        up(root,0)
    return relevant


def _lazy_complete(region, cells, baseline, saved, threshold, order, tree_orders, capacities,
                   certificate_fn=None, timings=None, certificate_time_key=None):
    """Refine lower/upper minimax tables, then resolve all relevant ties."""
    records = {} if saved is None else {tuple(c):dict(r) for c,r in saved.items()}
    updates, attempted_failures = {}, set()
    counters = {'initial_threshold_calls':0,'independent_seed_calls':0,
                'batch_upper_relevance_calls':0,'optimistic_completion_calls':0,
                'tie_completion_calls':0,'refinement_rounds':0}
    certificate_fn = cell_certificate if certificate_fn is None else certificate_fn

    def search(function,*args):
        return (function(*args) if timings is None else
                _measure(timings,'minimax_search_seconds',function,*args))

    def partial(record):
        return (record['status']=='rejected' or record.get('raw_status')=='rejected') and record.get('rejection_bound') is not None

    def evaluate(cell, cap, phase):
        if cell in attempted_failures:
            return False
        record = (certificate_fn(region,cell,max_cap=cap) if timings is None else
                  _measure(timings,certificate_time_key,certificate_fn,region,cell,max_cap=cap))
        records[cell] = record;updates[cell] = record;counters[phase] += 1
        if record['status'] == 'failure' and not partial(record):
            attempted_failures.add(cell)
        return record['status'] != 'failure' or partial(record)

    # These values suffice for every secondary threshold decision; old above-2
    # rejections remain lower bounds and need no repeated short certification.
    for cell in cells:
        old = records.get(cell)
        if old is not None and (old['status'] in ('full','boundary') or partial(old) and old['rejection_bound'] > threshold):
            continue
        evaluate(cell,threshold,'initial_threshold_calls')

    # The independent envelopes saved by the backend are completed upper
    # certificates. They propose feasible incumbents only, never the minimax
    # value of the prescribed coupled objective.
    proposals = {c: (r.get('coupled') if r['status']=='full' else
                     r.get('independent',r.get('valid_upper_bound')) if partial(r) else
                     r.get('valid_upper_bound'))
                 for c,r in records.items()}
    seeds = search(rank_capacity_search,order,tree_orders,proposals,capacities)
    for choice in seeds.values():
        if choice['status'] != 'finite':
            continue
        for cell in choice['partition']:
            if records[cell]['status'] not in ('full','boundary'):
                evaluate(cell,choice['certificate'],'independent_seed_calls')

    if BATCH_UPPER_RELEVANCE:
        # A completed feasible upper incumbent M guarantees the true optimum
        # is <=M. Every potentially optimal cell therefore belongs to a
        # lower-bound-feasible partition at that same threshold. Completing
        # this one batch preserves the entire optimum and canonical tie rule.
        choices = search(rank_capacity_search,order,tree_orders,records,capacities)
        requested = {}
        for B,choice in choices.items():
            if choice['status']=='unresolved' and choice['candidate_status']=='finite':
                M = choice['candidate_certificate']
                for cell in search(_relevant_cells,order,tree_orders,records,B,M):
                    if records[cell]['status'] not in ('full','boundary') and cell not in attempted_failures:
                        requested[cell] = max(requested.get(cell,M),M)
        for cell,cap in requested.items():
            evaluate(cell,cap,'batch_upper_relevance_calls')

    while True:
        choices = search(rank_capacity_search,order,tree_orders,records,capacities)
        pending = [choice for choice in choices.values() if choice['status']=='unresolved']
        if not pending:
            break
        requested = {}
        for choice in pending:
            cap = choice['candidate_certificate'] if choice['candidate_status']=='finite' else None
            for cell in choice['optimistic_partition']:
                if records[cell]['status'] not in ('full','boundary') and cell not in attempted_failures:
                    if cell not in requested or cap is None or requested[cell] is not None and cap>requested[cell]:
                        requested[cell] = cap
        if not requested:
            break
        counters['refinement_rounds'] += 1
        for cell,cap in requested.items():
            evaluate(cell,cap,'optimistic_completion_calls')

    # Objective equality alone does not identify every canonical local tie.
    # Any true optimum uses only lower-bound-feasible cells. Finish all such
    # unresolved cells, or prove their scores strictly above the relevant M.
    requested = {}
    for B,choice in choices.items():
        if choice['status'] != 'finite':
            continue
        M = choice['certificate']
        for cell in search(_relevant_cells,order,tree_orders,records,B,M):
            if records[cell]['status'] not in ('full','boundary') and cell not in attempted_failures:
                requested[cell] = max(requested.get(cell,M),M)
    for cell,cap in requested.items():
        evaluate(cell,cap,'tie_completion_calls')
    final = search(rank_capacity_search,order,tree_orders,records,capacities)
    unresolved_ties = set()
    unresolved_tie_capacities = {}
    for B,choice in final.items():
        if choice['status']=='finite':
            unknown = tuple(sorted(c for c in search(_relevant_cells,order,tree_orders,records,B,choice['certificate'])
                                   if records[c]['status'] not in ('full','boundary')))
            if unknown:
                unresolved_tie_capacities[B] = unknown
                unresolved_ties.update(unknown)
    state,incumbent = _partition_record(baseline,records)
    return records,updates,{'strategy':('threshold_then_lazy_minimax_and_relevant_ties'
                                       +('_batch_upper_relevance' if BATCH_UPPER_RELEVANCE else '')),
        'incumbent_status':state,'incumbent':incumbent,'pruning_cutoff':threshold,
        'finite_incumbents':{B:c['certificate'] for B,c in final.items() if c['status']=='finite'},
        'canonical_ties_completed':not unresolved_ties,'unresolved_tie_cells':tuple(sorted(unresolved_ties)),
        'unresolved_tie_capacities':unresolved_tie_capacities,
        **counters}


def _guard_ties(choices, pruning, records, order, tree_orders):
    """A proved optimum value does not complete an unresolved tie selection."""
    if pruning.get('canonical_ties_completed', True):
        return choices
    result = {}
    for B,choice in choices.items():
        unknown = (tuple(c for c in _relevant_cells(order,tree_orders,records,B,choice['certificate'])
                         if records[c]['status'] not in ('full','boundary'))
                   if choice['status']=='finite' else ())
        if unknown:
            choice = dict(choice, status='unresolved',completed=False,
                          objective_value_completed=True,objective_certificate=choice['certificate'],
                          reason='canonical-selection-tie-unresolved',tie_unresolved_cells=unknown,
                          candidate_partition=choice['partition'],candidate_certificate=choice['certificate'],
                          candidate_status='finite',certificate=None,partition=None,cells=None,family=None)
        result[B] = choice
    return result


def _complete(region, cells, baseline, saved=None, threshold=Q(2), *,
              order=None, tree_orders=None, capacities=None,timings=None,certificate_time_key=None):
    """Complete only cells that can change a threshold or minimax decision."""
    if capacities is not None:
        return _lazy_complete(region,cells,baseline,saved,threshold,order,tree_orders,capacities,
                               timings=timings,certificate_time_key=certificate_time_key)
    records = {} if saved is None else {tuple(c): dict(r) for c, r in saved.items()}
    updates = {}
    for c in baseline:
        if c not in records or records[c]['status'] not in ('full', 'boundary'):
            records[c] = cell_certificate(region, c)
            updates[c] = records[c]
    state, incumbent = _partition_record(baseline, records)
    cutoff = None if state != 'finite' else max(threshold, incumbent)
    for c in cells:
        old = records.get(c)
        if old is not None and old['status'] in ('full', 'boundary'):
            continue
        if (old is not None and old['status'] == 'rejected' and cutoff is not None
                and old.get('rejection_bound') is not None and old['rejection_bound'] > cutoff):
            continue
        records[c] = cell_certificate(region, c, max_cap=cutoff)
        updates[c] = records[c]
    return records, updates, {'incumbent_status': state, 'incumbent': incumbent, 'pruning_cutoff': cutoff}


def _threshold_choices(order, tree_orders, records, caps):
    trees = tuple(exact.build_tree(o) for o in tree_orders)
    upper = {c: r.get('coupled') if r['status'] == 'full' else None for c, r in records.items()}
    lower = {c: cell_score(r).lower for c, r in records.items()}
    result = {}
    for cap in caps:
        high = select_families(order, trees, upper, cap)['combined']
        low = select_families(order, trees, lower, cap)['combined']
        count_completed = high['cells'] == low['cells']
        completed = (count_completed and high['partition']==low['partition']
                     and high['certificate']==low['certificate'])
        result[cap] = dict(high, status='finite' if completed else 'unresolved',
                           completed=completed,count_completed=count_completed,
                           lower_count=low['cells'], upper_count=high['cells'], cap=cap)
    return result


def _log(value):
    # math.log(int) handles large integer mantissas without first overflowing
    # a float conversion of the rational itself.
    value = Q(value)
    if value <= 0:
        raise ValueError('Log diagnostic requires a positive exact ratio')
    return math.log(value.numerator) - math.log(value.denominator)


def _decomposition(choice, dstar, law, slice_region, mass_feasible, pooled_intervals=None):
    if choice['status'] != 'finite':
        return {'decomposition_status': choice['status'], 'truth_mass_feasible': mass_feasible}
    selected = TruthEvaluator(law).partition(choice['partition'])['value']
    rec = [cell_certificate(slice_region, c) for c in choice['partition']]
    if pooled_intervals is not None:
        from experiments import jmlr_efficiency_gap_v1_core as core
        rec = [core.pooled_cell_bound(record, c, slice_region, pooled_intervals)
               for record, c in zip(rec, choice['partition'])]
    if any(r['status'] != 'full' or r['coupled'] is None for r in rec):
        return {'decomposition_status': 'slice-unresolved-or-infinite',
                'truth_mass_feasible': mass_feasible, 'selected_true_distortion': selected}
    sliced = max(r['coupled'] for r in rec)
    U = choice['certificate']
    ratios = {'total': U/dstar, 'mass': U/sliced,
              'propensity': sliced/selected, 'selection': selected/dstar}
    if ratios['mass'] * ratios['propensity'] * ratios['selection'] != ratios['total']:
        raise ArithmeticError('Exact multiplicative decomposition failed')
    return {'decomposition_status': 'finite', 'truth_mass_feasible': mass_feasible,
            'selected_true_distortion': selected, 'true_mass_same_propensity_certificate': sliced,
            'decomposition_ratios': ratios,
            'decomposition_logs': {key: _log(value) for key, value in ratios.items()},
            'decomposition_identity_exact': True,
            'decomposition_attribution_scope': 'mass component is a nested-region comparison only when truth_mass_feasible; nonnegativity also requires applicable coverage'}


def _policy(protocol):
    configuration = protocol.get('configuration', {})
    policy = configuration.get('root_policy')
    return RootPolicy(**policy) if policy is not None else RootPolicy()


def _law(protocol, law_id):
    laws = protocol.get('historical_laws', protocol.get('laws'))
    if isinstance(laws, dict):
        return laws[law_id]
    return next(law for law in laws if law['id'] == law_id)


def _build_comparison(N, T, order, tree_orders, cells, baseline, alpha, verifier, timings=None, capacities=None):
    from experiments import jmlr_efficiency_gap_v1_core as core
    timings = {} if timings is None else timings
    evidence = _measure(timings, 'shared_raw_plus_pooled_confidence_construction_seconds',
                        core.build_regions, N, T, alpha, tree_orders, verifier=verifier)
    raw, updates, pruning = _complete(evidence['raw_region'], cells, baseline,
                                     order=order,tree_orders=tree_orders,capacities=capacities,
                                     timings=timings,certificate_time_key='allocation_control_raw_certification_seconds')
    tick = time.perf_counter()
    def fuse(record,cell):
        result = core.pooled_cell_bound(record,cell,evidence['raw_region'],evidence['pooled_intervals'])
        uppers = [v for v in (record.get('independent'),result.get('valid_upper_bound')) if v is not None]
        if uppers:
            result['independent'] = min(uppers)
        return result
    pooled = {c:fuse(raw[c],c) for c in cells}
    timings['pooled_incremental_fusion_seconds'] = timings.get('pooled_incremental_fusion_seconds', 0.) + time.perf_counter()-tick
    if capacities is not None:
        def pooled_certificate(region,cell,max_cap=None):
            old = raw[cell]
            if not (old['status'] in ('full','boundary') or old['status']=='rejected'
                    and max_cap is not None and old.get('rejection_bound') is not None
                    and old['rejection_bound'] > max_cap):
                raw[cell] = cell_certificate(region,cell,max_cap=max_cap)
            return fuse(raw[cell],cell)
        pooled,_,pooled_pruning = _lazy_complete(evidence['raw_region'],cells,baseline,pooled,Q(2),
                                                 order,tree_orders,capacities,pooled_certificate,
                                                 timings=timings,certificate_time_key='pooled_incremental_refinement_certification_seconds')
        pruning['pooled_incremental_refinement'] = pooled_pruning
    return evidence, raw, pooled, pruning


def _slice_selection(raw_region, law, cells, order, tree_orders, caps, pooled_intervals=None):
    """Privileged count diagnostic without changing propensity allocation."""
    from experiments import jmlr_efficiency_gap_v1_core as core
    sliced = core.truth_mass_slice(raw_region, law['p'])
    raw = {c: cell_certificate(sliced['region'], c, max_cap=max(caps)) for c in cells}
    records = raw if pooled_intervals is None else {
        c: core.pooled_cell_bound(raw[c], c, sliced['region'], pooled_intervals) for c in cells}
    return sliced, records, _threshold_choices(order, tree_orders, records, caps)


def _pooled_event(law, intervals):
    evaluator = TruthEvaluator(law)
    return all(record['lower'] <= evaluator.cell(c)['propensity'] <= record['upper']
               for c, record in intervals.items())


def _same_partition_slice(partition, sliced, pooled_intervals=None):
    records = {c: cell_certificate(sliced['region'], c) for c in partition}
    if pooled_intervals is not None:
        from experiments import jmlr_efficiency_gap_v1_core as core
        records = {c: core.pooled_cell_bound(records[c], c, sliced['region'], pooled_intervals)
                   for c in partition}
    return _partition_record(partition, records)


def process_historical(path, protocol, verifier=None):
    """Analyze one authoritative historical assignment record, with no writes."""
    from experiments import jmlr_efficiency_gap_v1_core as core
    started = time.perf_counter()
    timings = _timings()
    source = _load(path)
    saved = source['core']
    law = _law(protocol, source['law_id'])
    meta = dict(source['metadata'])
    N, T = tuple(saved['N']), tuple(saved['T'])
    order = tuple(saved['order'])
    trees = tuple(tuple(o) for o in saved['tree_orders'])
    cells = tuple(tuple(c) for c in saved['family_cells'])
    caps = tuple(Q(c) for c in protocol.get('caps', CAPS))
    original_mode = next(mode for mode in saved['modes'] if mode['mode'] == 'unknown_mass')
    region = _restore(original_mode['region'])
    truth = {tuple(c['cell']): c['value'] for c in saved['truth_cells']}
    oracles = {cap: next(row for row in saved['oracles'] if row['family'] == 'combined' and row['cap'] == cap)
               for cap in caps}
    capacity = {cap: int(oracles[cap]['count']) for cap in caps}
    Bs = tuple(sorted(set(capacity.values())))
    timings['source_load_and_setup_seconds'] = time.perf_counter()-started
    pop = _measure(timings, 'minimax_search_seconds', rank_capacity_search, order, trees, truth, Bs)
    baseline = _partition(oracles[max(caps)]['partition'])
    old_records = {tuple(r['cell']): r for r in original_mode['cells']}
    original, updates, pruning_original = _complete(region,cells,baseline,old_records,max(caps),
        order=order,tree_orders=trees,capacities=Bs,timings=timings,
        certificate_time_key='original_missing_completion_seconds')
    original_minimax = _measure(timings, 'minimax_search_seconds', rank_capacity_search, order, trees, original, Bs)
    original_minimax = _guard_ties(original_minimax,pruning_original,original,order,trees)
    alpha = region.get('alpha', Q(1, 20))
    verifier = RootVerifier(_policy(protocol)) if verifier is None else verifier
    evidence, control, pooled, pruning_control = _build_comparison(
        N, T, order, trees, cells, baseline, alpha, verifier, timings,Bs)
    procedures = {'original': (region, original, original_minimax),
                  'allocation_control': (evidence['raw_region'], control, _measure(timings, 'minimax_search_seconds', rank_capacity_search, order, trees, control, Bs)),
                  'pooled_alternative': (evidence['raw_region'], pooled, _measure(timings, 'minimax_search_seconds', rank_capacity_search, order, trees, pooled, Bs))}
    for name,pruning in (('allocation_control',pruning_control),
                         ('pooled_alternative',pruning_control.get('pooled_incremental_refinement',{}))):
        raw,records,choices=procedures[name]
        procedures[name]=(raw,records,_guard_ties(choices,pruning,records,order,trees))
    thresholds = {name: _measure(timings, 'threshold_search_seconds', _threshold_choices, order, trees, records, caps)
                  for name, (_, records, _) in procedures.items()}
    tick = time.perf_counter()
    slices = {'original': _slice_selection(region, law, cells, order, trees, caps),
              'allocation_control': _slice_selection(evidence['raw_region'], law, cells, order, trees, caps)}
    control_slice, control_slice_records, _ = slices['allocation_control']
    pooled_slice_records = {
        c: core.pooled_cell_bound(record, c, control_slice['region'], evidence['pooled_intervals'])
        for c, record in control_slice_records.items()}
    slices['pooled_alternative'] = (
        control_slice, pooled_slice_records, _threshold_choices(order, trees, pooled_slice_records, caps))
    timings['mass_slice_diagnostics_seconds'] += time.perf_counter()-tick
    rows = []
    tick = time.perf_counter()
    diagnostics = {'original': core.manuscript_diagnostics(region, law['p'], law['e']),
                   'allocation_control': core.manuscript_diagnostics(evidence['raw_region'], law['p'], law['e'])}
    pooled_covered = _pooled_event(law, evidence['pooled_intervals'])
    radii = {name: core.realized_radii(raw_region, law['p'], law['e'])
             for name, (raw_region, _, _) in procedures.items()}
    evaluator = TruthEvaluator(law)
    timings['other_analytical_diagnostics_seconds'] += time.perf_counter()-tick
    for name, (raw_region, records, minimax) in procedures.items():
        sliced, slice_records, slice_thresholds = slices[name]
        for cap in caps:
            B = capacity[cap]
            dstar = pop[B]['certificate']
            choice = minimax[B]
            threshold = thresholds[name][cap]
            source_selection = next(row for row in original_mode['labels']
                                    if row['method'] == 'coupled' and row['family'] == 'combined' and row['cap'] == cap)
            if name == 'original':
                if (threshold['cells'] != source_selection['count']
                        or threshold['certificate'] != source_selection['certificate']
                        or _partition(threshold['partition']) != _partition(source_selection['partition'])):
                    raise ArithmeticError('Original historical threshold selection changed')
                threshold = dict(source_selection, status='finite', completed=True,
                                 lower_count=source_selection['count'], upper_count=source_selection['count'])
            threshold_partition = _partition(threshold['partition'])
            threshold_D = _measure(timings, 'other_analytical_diagnostics_seconds', evaluator.partition, threshold_partition)['value']
            if name == 'original' and threshold_D != source_selection['true_distortion']:
                raise ArithmeticError('Historical selected population distortion changed')
            threshold_count = threshold.get('cells', threshold.get('count'))
            gap = threshold_count - B
            slice_state, slice_certificate = _measure(
                timings, 'mass_slice_diagnostics_seconds', _same_partition_slice, threshold_partition, sliced,
                evidence['pooled_intervals'] if name == 'pooled_alternative' else None)
            if (sliced['truth_mass_feasible'] and slice_state == 'finite'
                    and slice_certificate > threshold['certificate']):
                raise ArithmeticError('A feasible true-mass slice increased the same-partition certificate')
            procedure_covered = (radii[name]['truth_in_region']
                                 and (name != 'pooled_alternative' or pooled_covered))
            if procedure_covered and threshold_D > threshold['certificate']:
                raise ArithmeticError('A completed threshold certificate failed on its common event')
            row = {**meta, 'source': str(path), 'scope': 'historical_assignment_only',
                   'law_id': law['id'], 'procedure': name, 'alpha_total': alpha,
                   'alpha_raw': raw_region['alpha'], 'cap': cap, 'oracle_count': B,
                   'd_star': dstar, 'oracle_minimax_groups': pop[B]['cells'],
                   'oracle_margin_log': _log(cap/dstar), 'oracle_margin_ratio': cap/dstar,
                   'selected_count': threshold_count,
                   'selected_certificate': threshold['certificate'],
                   'selected_partition': threshold_partition,
                   'selection_status': threshold.get('status'),
                   'count_gap': gap, 'normalized_count_gap': Q(gap, law['K']),
                   'normalized_compressible_gap': Q(gap, law['K']-B) if law['K'] != B else None,
                   'threshold_selected_true_distortion': threshold_D,
                   'selected_true_distortion': threshold_D,
                   'threshold_selected_certificate': threshold['certificate'],
                   'threshold_log_inflation': _log(threshold['certificate']/threshold_D),
                   'threshold_inflation_ratio': threshold['certificate']/threshold_D,
                   'threshold_cap_violation': threshold_D > cap,
                   'minimum_state_count': min(N),
                   'minimum_treated_or_control_count': min(min(t,n-t) for n,t in zip(N,T)),
                   'population_min_mass': min(law['p']),
                   'population_overlap': min(min(e,1-e) for e in law['e']),
                   'calibration_covered': radii[name]['truth_in_region'],
                   'pooled_event_covered': pooled_covered,
                   'procedure_event_covered': procedure_covered,
                   'original_selected_count': source_selection['count'],
                   'historical_original_preserved': name == 'original',
                   'oracle_capacity_status': choice['status'],
                   'oracle_capacity_minimax_certificate': choice['certificate'],
                   'oracle_capacity_partition': choice['partition'],
                   'oracle_capacity_groups': choice['cells'],
                   'oracle_capacity_lower_bound': choice['lower_bound'],
                   'oracle_capacity_at_most_cap': (choice['certificate'] <= cap if choice['status'] == 'finite' else
                                                  False if choice['status'] == 'infinite' else None),
                   'minimum_count_recovered': threshold_count == B,
                   'true_mass_slice_count': slice_thresholds[cap]['cells'],
                   'true_mass_slice_selection_status': slice_thresholds[cap]['status'],
                   'original_true_mass_slice_count': slices['original'][2][cap]['cells'],
                   'same_partition_true_mass_slice_status': slice_state,
                   'same_partition_true_mass_slice_certificate': slice_certificate,
                   'truth_mass_slice_feasible': sliced['truth_mass_feasible'],
                   'truth_mass_slice_nested': sliced['truth_mass_feasible'],
                   'realized_radii': radii[name]}
            decomposition = _measure(timings, 'mass_slice_diagnostics_seconds', _decomposition,
                                      choice, dstar, law, sliced['region'], sliced['truth_mass_feasible'],
                                      evidence['pooled_intervals'] if name == 'pooled_alternative' else None)
            row.update({'oracle_capacity_'+key: value for key, value in decomposition.items()})
            if slice_state == 'finite':
                row.update(threshold_mass_inflation_ratio=threshold['certificate']/slice_certificate,
                           threshold_propensity_inflation_ratio=slice_certificate/threshold_D,
                           threshold_log_mass_inflation=_log(threshold['certificate']/slice_certificate),
                           threshold_log_propensity_inflation=_log(slice_certificate/threshold_D),
                           threshold_decomposition_identity_exact=True)
            if choice['status'] == 'finite':
                row['minimax_log_inflation'] = _log(choice['certificate']/dstar)
                row['inflation_within_margin'] = choice['certificate'] <= cap
            rows.append(row)
    return {'source': str(path), 'metadata': meta, 'law_id': law['id'], 'rows': rows,
            'diagnostics': diagnostics,
            'new_region': evidence['raw_region'],
            'pooled_interval_evidence': [dict(record, cell=c) for c, record in evidence['pooled_intervals'].items()],
            'allocation': {k: evidence[k] for k in ('alpha_raw', 'alpha_pooled', 'gamma_pooled')},
            'original_completed_updates': [_compact(r) for r in updates.values()],
            'true_mass_slice_evidence': {
                name: {'truth_mass_feasible': sliced['truth_mass_feasible'],
                       'propensity_intervals_unchanged': sliced['propensity_intervals_unchanged'],
                       'original_alpha': sliced['original_alpha'],
                       'cells': [_compact(record) for record in rec.values()],
                       'threshold_choices': choices}
                for name, (sliced, rec, choices) in slices.items()},
            'new_raw_cells': [_compact(r) for r in control.values()],
            'new_pooled_cells': [_compact(r) for r in pooled.values()],
            'pruning': {'original': pruning_original, 'new_raw': pruning_control},
            'root_cache': {'hits': verifier.cache_hits, 'misses': verifier.cache_misses},
            'timings': timings, 'timing_interpretation': TIMING_INTERPRETATION,
            'source_original_timings': {'unknown_mass_mode': original_mode.get('timings', {}),
                                        'core': saved.get('timings', {}),
                                        'record_compute_seconds': source.get('compute_seconds')},
            'compute_seconds': time.perf_counter()-started}


def process_release(i, j, protocol, verifier=None):
    """Use assignment evidence from one completed release block; ignore outcomes."""
    from experiments import jmlr_efficiency_gap_v1_core as core
    started = time.perf_counter()
    timings = _timings()
    release_protocol = protocol['release_protocol']
    root = Path(protocol.get('release_root', PROJECT/'results/jmlr_release_capacity_v1'))
    path = root/'panel'/'blocks'/f'{i}_{j:03d}.json.gz'
    source = _load(path)
    law = release_protocol['laws'][i]
    N, T = tuple(source['N']), tuple(source['T'])
    order = tuple(source['order'])
    trees = tuple(tuple(o) for o in release_protocol['tree_orders'][i])
    cells = tuple(tuple(r['cell']) for r in source['cells'])
    capacities = tuple(release_protocol['capacity_grids'][i])
    region = _restore(source['region'])
    original = {tuple(r['cell']): r for r in source['cells']}
    # The smallest-capacity selected partition is feasible at all grid entries.
    first = source['choices'][str(min(capacities))]['verified_capacity']
    # If the saved computation was unresolved, use the one-cell constituent
    # only to seek a numerical incumbent.  Identity would violate B<K and
    # would not justify excluding above-incumbent cells from that optimum.
    baseline = (_partition(first['partition']) if first['partition'] is not None
                else (tuple(range(law['K'])),))
    timings['source_load_and_setup_seconds'] = time.perf_counter()-started
    verifier = RootVerifier(_policy(protocol)) if verifier is None else verifier
    evidence, control, pooled, pruning = _build_comparison(
        N, T, order, trees, cells, baseline, region['alpha'], verifier, timings,capacities)
    alternatives = {
        'allocation_control': _measure(timings, 'minimax_search_seconds', rank_capacity_search, order, trees, control, capacities),
        'pooled_alternative': _measure(timings, 'minimax_search_seconds', rank_capacity_search, order, trees, pooled, capacities),
    }
    alternatives['allocation_control'] = _guard_ties(alternatives['allocation_control'],pruning,control,order,trees)
    alternatives['pooled_alternative'] = _guard_ties(alternatives['pooled_alternative'],
        pruning.get('pooled_incremental_refinement',{}),pooled,order,trees)
    # Reuse the completed original optimum rather than reinterpret its choice.
    original_choices = {B: source['choices'][str(B)]['verified_capacity'] for B in capacities}
    original_choices = {B: dict(row, partition=None if row['partition'] is None else _partition(row['partition']))
                        for B, row in original_choices.items()}
    alternatives = {'original': original_choices, **alternatives}
    oracle = {int(B): row for B, row in source['oracle']['by_capacity'].items()}
    rows = []
    tick = time.perf_counter()
    diagnostics = {'original': core.manuscript_diagnostics(region, law['p'], law['e']),
                   'allocation_control': core.manuscript_diagnostics(evidence['raw_region'], law['p'], law['e'])}
    pooled_covered = _pooled_event(law, evidence['pooled_intervals'])
    timings['other_analytical_diagnostics_seconds'] += time.perf_counter()-tick
    for name, choices in alternatives.items():
        raw = region if name == 'original' else evidence['raw_region']
        sliced = _measure(timings, 'mass_slice_diagnostics_seconds', core.truth_mass_slice, raw, law['p'])
        radii = _measure(timings, 'other_analytical_diagnostics_seconds', core.realized_radii, raw, law['p'], law['e'])
        for B, choice in choices.items():
            dstar = oracle[B]['certificate']
            row = {'scope': 'release_assignment_only', 'source': str(path), 'law_index': i,
                   'law_id': law['id'], 'replication': j, 'K': law['K'], 'n': sum(N),
                   'procedure': name, 'alpha_total': region['alpha'], 'alpha_raw': raw['alpha'],
                   'B': B, 'cap': Q(3, 2), 'd_star': dstar, 'oracle_count': oracle[B]['cells'],
                   'oracle_margin_log': _log(Q(3, 2)/dstar),
                   'oracle_margin_ratio': Q(3, 2)/dstar,
                   'oracle_capacity_status': choice['status'],
                   'oracle_capacity_minimax_certificate': choice['certificate'],
                   'oracle_capacity_partition': choice['partition'], 'oracle_capacity_groups': choice['cells'],
                   'admitted': choice['status'] == 'finite' and choice['certificate'] <= Q(3, 2),
                   'minimum_state_count': min(N),
                   'minimum_treated_or_control_count': min(min(t,n-t) for n,t in zip(N,T)),
                   'population_min_mass': min(law['p']),
                   'population_overlap': min(min(e,1-e) for e in law['e']),
                   'calibration_covered': radii['truth_in_region'],
                   'pooled_event_covered': pooled_covered,
                   'procedure_event_covered': radii['truth_in_region'] and (name != 'pooled_alternative' or pooled_covered),
                   'realized_radii': radii}
            row.update(_measure(timings, 'mass_slice_diagnostics_seconds', _decomposition,
                                choice, dstar, law, sliced['region'], sliced['truth_mass_feasible'],
                                evidence['pooled_intervals'] if name == 'pooled_alternative' else None))
            if choice['status'] == 'finite':
                row['minimax_log_inflation'] = _log(choice['certificate']/dstar)
                row['inflation_within_margin'] = choice['certificate'] <= Q(3, 2)
            rows.append(row)
    return {'source': str(path), 'law_id': law['id'], 'law_index': i, 'replication': j,
            'rows': rows, 'diagnostics': diagnostics, 'new_region': evidence['raw_region'],
            'pooled_interval_evidence': [dict(record, cell=c) for c, record in evidence['pooled_intervals'].items()],
            'allocation': {k: evidence[k] for k in ('alpha_raw', 'alpha_pooled', 'gamma_pooled')},
            'new_raw_cells': [_compact(r) for r in control.values()],
            'new_pooled_cells': [_compact(r) for r in pooled.values()], 'pruning': pruning,
            'root_cache': {'hits': verifier.cache_hits, 'misses': verifier.cache_misses},
            'timings': timings, 'timing_interpretation': TIMING_INTERPRETATION,
            'source_original_timings': {'assignment': {key: value for key, value in source.get('timings', {}).items()
                                                      if key in ('confidence_construction_seconds', 'cell_certification_seconds',
                                                                 'partition_search_seconds')},
                                        'scope': 'original release provider assignment stages, shared over original comparison groupings and capacities'},
            'compute_seconds': time.perf_counter()-started}
