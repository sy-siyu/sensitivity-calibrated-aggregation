"""Pilot-only proposals and verified split tests for the bounded LTT comparison.

The pilot score is deliberately untrusted binary floating-point arithmetic.
It chooses a fixed testing sequence only. Held-out acceptance uses the unchanged
verified CP backend and exact rational cap comparisons; no pilot value can
authorize acceptance. Infinity is exported as None, with an explicit status.
"""
from __future__ import annotations

from fractions import Fraction
import math
import numbers
import time

import numpy as np
import scipy
from scipy.special import betaincinv

from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier, rational
from experiments.jmlr_verified_adaptive_v1_math import (
    ONE, build_tree, canonical_partition, cell_certificate, confidence_region,
    interval_cells, odds, validate_partition,
)


ALPHA = Fraction(1, 20)
CAPS = (Fraction(5, 4), Fraction(3, 2), Fraction(2))


def _counts(N, T):
    N, T = tuple(N), tuple(T)
    if not N or len(N) != len(T):
        raise ValueError('Counts need a common nonempty support')
    for n, t in zip(N, T):
        if (isinstance(n, bool) or isinstance(t, bool)
                or not isinstance(n, numbers.Integral)
                or not isinstance(t, numbers.Integral)
                or n < 0 or not 0 <= t <= n):
            raise ValueError('Require integer counts 0 <= T_i <= N_i')
    return tuple(map(int, N)), tuple(map(int, T))


def cell_key(cell):
    """Compact immutable-state bitmask key, with no floating conversion."""
    return str(sum(1 << int(i) for i in cell))


def nested_splits(N, T, population_index, r_index, replication, immutable_ids=None):
    """Uniform fixed-size observation subsets, nested at 1/5 and 1/2.

    The sufficient joint counts are interleaved (state, action=0/1). The second
    hypergeometric draw is from the first draw's componentwise remainder.
    """
    N, T = _counts(N, T)
    ids = tuple(range(len(N))) if immutable_ids is None else tuple(immutable_ids)
    if len(ids) != len(N) or len(set(ids)) != len(N):
        raise ValueError('Require distinct immutable state identifiers')
    state_order = tuple(sorted(range(len(N)), key=lambda i: ids[i]))
    identifiers = (population_index, r_index, replication)
    if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) or i < 0
           for i in identifiers):
        raise ValueError('Split identifiers must be nonnegative integers')
    seed = [20261012, 901, *map(int, identifiers)]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    joint = np.array([v for i in state_order for v in (N[i] - T[i], T[i])], dtype=np.int64)
    n = sum(N)
    first_size, second_size = n // 5, n // 2
    first = rng.multivariate_hypergeometric(joint, first_size)
    extra = rng.multivariate_hypergeometric(joint - first, second_size - first_size)
    answer = {}
    for label, pilot in (('1/5', first), ('1/2', first + extra)):
        heldout = joint - pilot
        if np.any(pilot < 0) or np.any(heldout < 0) or not np.array_equal(pilot + heldout, joint):
            raise ArithmeticError('Hypergeometric split did not conserve joint counts')
        P, H = np.empty((len(N), 2), dtype=np.int64), np.empty((len(N), 2), dtype=np.int64)
        P[list(state_order)] = pilot.reshape(-1, 2)
        H[list(state_order)] = heldout.reshape(-1, 2)
        answer[label] = {
            'pilot_N': tuple(int(v) for v in P.sum(axis=1)),
            'pilot_T': tuple(int(v) for v in P[:, 1]),
            'heldout_N': tuple(int(v) for v in H.sum(axis=1)),
            'heldout_T': tuple(int(v) for v in H[:, 1]),
            'pilot_size': int(pilot.sum()), 'heldout_size': int(heldout.sum()),
            'fraction': Fraction(label), 'seed': seed,
            'generator': 'PCG64(SeedSequence)',
            'joint_order': 'immutable state, action 0 then action 1',
            'immutable_ids': ids,
        }
    return answer


def pilot_order(N, T, immutable_ids=None):
    N, T = _counts(N, T)
    ids = tuple(range(len(N))) if immutable_ids is None else tuple(immutable_ids)
    if len(ids) != len(N) or len(set(ids)) != len(N):
        raise ValueError('Require distinct immutable state identifiers')
    # Exact smoothed fractions avoid machine ties in the proposal ordering.
    return tuple(sorted(range(len(N)),
                        key=lambda i: (Fraction(2 * T[i] + 1, 2 * N[i] + 2), ids[i])))


def _score(value):
    """Internal pilot value; None is the exported infinity representation."""
    if value is None:
        return math.inf
    value = float(value)
    if math.isnan(value) or value < 1:
        raise ValueError('Pilot scores must be >=1 or infinity')
    return value


def _export_score(value):
    return float(value) if math.isfinite(value) else None


class PilotScorer:
    """Ordinary CP ranking values; never shared with held-out certification."""
    def __init__(self, N, T, alpha=ALPHA):
        self.N, self.T = _counts(N, T)
        self.alpha = rational(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError('Invalid pilot interval error')
        self.interval_cache = {}
        self.cell_cache = {}
        self.states = tuple(self.interval(t, n) for t, n in zip(self.T, self.N))

    def interval(self, x, m):
        key = (x, m)
        if key in self.interval_cache:
            return self.interval_cache[key]
        try:
            tail = float(self.alpha / 2)
            lower = 0.0 if x == 0 else float(betaincinv(x, m - x + 1, tail))
            upper = 1.0 if x == m else float(betaincinv(x + 1, m - x, 1 - tail))
            if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1):
                raise ArithmeticError('Invalid SciPy inverse-beta pilot interval')
            result = {'x': x, 'm': m, 'lower': lower, 'upper': upper,
                      'status': 'structural' if m == 0 else 'ordinary-cp', 'reason': None}
        except Exception as error:
            result = {'x': x, 'm': m, 'lower': 0.0, 'upper': 1.0,
                      'status': 'failure', 'reason': type(error).__name__ + ': ' + str(error)}
        self.interval_cache[key] = result
        return result

    def cell(self, cell):
        cell = tuple(sorted(cell))
        if cell in self.cell_cache:
            return self.cell_cache[cell]['value']
        if len(cell) == 1:
            value, status, reason = 1.0, 'singleton', None
        else:
            pooled = self.interval(sum(self.T[i] for i in cell), sum(self.N[i] for i in cell))
            involved = [self.states[i] for i in cell] + [pooled]
            if any(r['status'] == 'failure' for r in involved):
                value, status, reason = math.inf, 'failure', 'pilot-interval-failure'
            elif (pooled['lower'] <= 0 or pooled['upper'] >= 1
                  or any(self.states[i]['lower'] <= 0 or self.states[i]['upper'] >= 1 for i in cell)):
                value, status, reason = math.inf, 'boundary', 'pilot-boundary-infinity'
            else:
                try:
                    def o(v):
                        return v / (1 - v)
                    value = max(1.0, *(o(self.states[i]['upper']) / o(pooled['lower']) for i in cell),
                                *(o(pooled['upper']) / o(self.states[i]['lower']) for i in cell))
                    if not math.isfinite(value):
                        raise ArithmeticError('Nonfinite pilot odds')
                    status, reason = 'ordinary-score', None
                except Exception as error:
                    value, status = math.inf, 'failure'
                    reason = type(error).__name__ + ': ' + str(error)
        self.cell_cache[cell] = {'cell': cell, 'value': value, 'status': status, 'reason': reason}
        return value


def count_contiguous(order, cell_scores):
    """One minimax representative per count, earliest reachable predecessor.

    Reachability is not represented by a score: reachable infinity retains its
    valid cover and can never be confused with an impossible prefix.
    """
    order = tuple(order)
    K = len(order)
    if not K or len(set(order)) != K:
        raise ValueError('Invalid contiguous order')
    H = [[math.inf] * (K + 1) for _ in range(K + 1)]
    reach = [[False] * (K + 1) for _ in range(K + 1)]
    predecessor = [[None] * (K + 1) for _ in range(K + 1)]
    H[0][0], reach[0][0] = 1.0, True
    for j in range(1, K + 1):
        for m in range(1, j + 1):
            for i in range(m - 1, j):
                if not reach[i][m - 1]:
                    continue
                cell = tuple(sorted(order[i:j]))
                local = 1.0 if len(cell) == 1 else _score(cell_scores[cell])
                candidate = max(H[i][m - 1], local)
                if not reach[j][m] or candidate < H[j][m]:
                    H[j][m] = candidate
                    reach[j][m] = True
                    predecessor[j][m] = i
    answer = {}
    for m in range(1, K + 1):
        if not reach[K][m]:
            raise ArithmeticError('A contiguous count lost its reachable cover')
        cells, j, left = [], K, m
        while left:
            i = predecessor[j][left]
            if i is None:
                raise ArithmeticError('Missing reachable predecessor')
            cells.append(tuple(sorted(order[i:j])))
            j, left = i, left - 1
        if j != 0:
            raise ArithmeticError('Contiguous traceback did not reach the origin')
        answer[m] = {'cells': m, 'score': H[K][m], 'partition': canonical_partition(cells),
                     'family': 'contiguous'}
    return answer


def count_tree(tree, cell_scores):
    """Count-indexed tree convolution; score ties keep earliest left count."""
    if not tree:
        raise ValueError('Empty tree')
    root = max(tree, key=len)

    def visit(cell):
        children = tree[cell]
        value = 1.0 if len(cell) == 1 else _score(cell_scores[cell])
        answer = {1: {'cells': 1, 'score': value, 'partition': (tuple(sorted(cell)),)}}
        if children is None:
            return answer
        left, right = (visit(child) for child in children)
        for m in range(2, len(cell) + 1):
            best = None
            for ml in sorted(left):
                mr = m - ml
                if mr not in right:
                    continue
                score = max(left[ml]['score'], right[mr]['score'])
                if best is None or score < best['score']:
                    best = {'cells': m, 'score': score,
                            'partition': canonical_partition(left[ml]['partition'] + right[mr]['partition'])}
            if best is None:
                raise ArithmeticError('Tree count has no reachable pruning')
            answer[m] = best
        return answer

    return visit(root)


def candidates_from_scores(order, tree_orders, cell_scores):
    """Deterministic recurrence replay from saved pilot values alone."""
    order, tree_orders = tuple(order), tuple(tuple(o) for o in tree_orders)
    K = len(order)
    if len(tree_orders) != 2 or any(sorted(o) != list(range(K)) for o in tree_orders):
        raise ValueError('Require the two fixed complete tree leaf orders')
    representatives = [count_contiguous(order, cell_scores)]
    for index, leaf_order in enumerate(tree_orders):
        result = count_tree(build_tree(leaf_order), cell_scores)
        for choice in result.values():
            choice['family'] = 'tree-' + str(index)
        representatives.append(result)
    by_count = []
    for m in range(1, K + 1):
        choice = min((family[m] for family in representatives),
                     key=lambda r: (r['score'], r['partition'], r['family']))
        validate_partition(choice['partition'], K)
        by_count.append(dict(choice, score=_export_score(choice['score'])))
    sequence = sorted(by_count, key=lambda r: (_score(r['score']), -r['cells'], r['partition']))
    return [dict(record, candidate_index=i) for i, record in enumerate(sequence)]


def build_candidates(N, T, tree_orders, alpha=ALPHA, immutable_ids=None):
    """Only pilot counts and immutable geometry can enter this interface."""
    start = time.monotonic()
    N, T = _counts(N, T)
    order = pilot_order(N, T, immutable_ids)
    tree_orders = tuple(tuple(o) for o in tree_orders)
    if len(tree_orders) != 2 or any(sorted(o) != list(range(len(N))) for o in tree_orders):
        raise ValueError('Require the two fixed complete tree orders')
    scorer = PilotScorer(N, T, alpha)
    cells = set(interval_cells(order))
    for leaf_order in tree_orders:
        cells.update(build_tree(leaf_order))
    scores = {cell: scorer.cell(cell) for cell in sorted(cells)}
    candidates = candidates_from_scores(order, tree_orders, scores)
    evidence = {
        'alpha': rational(alpha),
        'method': 'ordinary SciPy inverse-beta CP; untrusted pilot ranking only',
        'scipy_version': scipy.__version__,
        'state_interval_count_keys': tuple((t, n) for t, n in zip(T, N)),
        'pooled_intervals': [scorer.interval_cache[key] for key in sorted(scorer.interval_cache)],
        'cell_scores': [dict(record, value=_export_score(record['value']))
                        for cell, record in sorted(scorer.cell_cache.items())],
        'contiguous_tie': 'earliest reachable predecessor',
        'tree_tie': 'earliest increasing feasible left-child count',
        'union_tie': 'pilot score, canonical returned representative, family label',
        'sequence_tie': 'pilot score, decreasing cell count, canonical partition',
    }
    return {'pilot_order': order, 'candidates': candidates, 'pilot_evidence': evidence,
            'pilot_seconds': time.monotonic() - start}


class HeldoutLTT:
    """Direct level-alpha distortion test; this value is not a joint bound."""
    def __init__(self, N, T, alpha=ALPHA, verifier=None):
        self.N, self.T = _counts(N, T)
        self.alpha = rational(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError('Invalid held-out interval error')
        self.verifier = RootVerifier() if verifier is None else verifier
        self.states = None
        self.pooled = {}
        self.cells = {}

    def _interval(self, x, m):
        key = (x, m)
        if key not in self.pooled:
            self.pooled[key] = self.verifier.interval(x, m, self.alpha)
        return self.pooled[key]

    def cell(self, cell):
        cell = tuple(sorted(cell))
        if cell in self.cells:
            return self.cells[cell]
        if len(cell) == 1:
            result = {'cell': cell, 'value': ONE, 'status': 'singleton', 'reason': None,
                      'pooled_count_key': None, 'plus': (), 'minus': ()}
            self.cells[cell] = result
            return result
        if self.states is None:
            self.states = tuple(self._interval(t, n) for t, n in zip(self.T, self.N))
        x, m = sum(self.T[i] for i in cell), sum(self.N[i] for i in cell)
        pooled = self._interval(x, m)
        involved = [self.states[i] for i in cell] + [pooled]
        result = {'cell': cell, 'value': None, 'status': 'failure', 'reason': None,
                  'pooled_count_key': (x, m), 'plus': (), 'minus': ()}
        if any(r['status'] == 'fallback' for r in involved):
            result.update(status='fallback', reason='verified-root-fallback')
        elif (pooled['lower'] <= 0 or pooled['upper'] >= 1
              or any(self.states[i]['lower'] <= 0 or self.states[i]['upper'] >= 1 for i in cell)):
            result.update(status='boundary', reason='genuine-boundary-infinity')
        else:
            try:
                plus = tuple((i, odds(self.states[i]['upper']) / odds(pooled['lower'])) for i in cell)
                minus = tuple((i, odds(pooled['upper']) / odds(self.states[i]['lower'])) for i in cell)
                result.update(value=max(ONE, *(v for _, v in plus), *(v for _, v in minus)),
                              status='full', plus=plus, minus=minus)
            except Exception as error:
                result.update(reason=type(error).__name__ + ': ' + str(error))
        self.cells[cell] = result
        return result

    def partition(self, partition):
        partition = validate_partition(partition, len(self.N))
        records = [self.cell(cell) for cell in partition]
        bad = next((r for r in records if r['value'] is None), None)
        return {
            'value': None if bad is not None else max((r['value'] for r in records), default=ONE),
            'status': 'full' if bad is None else bad['status'],
            'reason': None if bad is None else bad['reason'],
            'cell_keys': tuple(cell_key(c) for c in partition), 'rejection_bound': None,
        }

    def evidence(self):
        return {
            'alpha': self.alpha, 'gamma': self.alpha,
            'root_policy': dict(self.verifier.policy.__dict__),
            'state_interval_count_keys': None if self.states is None else tuple((t, n) for t, n in zip(self.T, self.N)),
            'pooled_intervals': [self.pooled[key] for key in sorted(self.pooled)],
            'cells': {cell_key(c): r for c, r in sorted(self.cells.items())},
        }


class HeldoutCoupled:
    """Unchanged alpha-level common-region envelope on the identical split."""
    def __init__(self, N, T, alpha=ALPHA, verifier=None):
        self.N, self.T = _counts(N, T)
        self.alpha = rational(alpha)
        self.verifier = RootVerifier() if verifier is None else verifier
        self.region = None
        self.cells = {}

    def cell(self, cell):
        cell = tuple(sorted(cell))
        if cell in self.cells:
            return self.cells[cell]
        if len(cell) == 1:
            result = {'cell': cell, 'coupled': ONE, 'independent': ONE, 'status': 'full',
                      'anchors': [], 'reason': None, 'rejection_bound': None,
                      'mixture_calls': 0, 'closure_calls': 0}
        else:
            if self.region is None:
                self.region = confidence_region(self.N, self.T, self.alpha, verifier=self.verifier)
            # Full envelopes for every queried cell keep the saved statistic
            # exact, independent of which cap happened to query it first.
            result = cell_certificate(self.region, cell)
        self.cells[cell] = result
        return result

    def partition(self, partition):
        partition = validate_partition(partition, len(self.N))
        records = [self.cell(cell) for cell in partition]
        bad = next((r for r in records if r['coupled'] is None), None)
        return {
            'value': None if bad is not None else max((r['coupled'] for r in records), default=ONE),
            'status': 'full' if bad is None else bad['status'],
            'reason': None if bad is None else bad['reason'],
            'cell_keys': tuple(cell_key(c) for c in partition), 'rejection_bound': None,
        }

    def evidence(self):
        return {'region': self.region, 'cells': {cell_key(c): r for c, r in sorted(self.cells.items())}}


def first_failure_select(candidates, cap, test_callback, K, method='test'):
    """Retain exactly a fixed sequence's accepted prefix, then minimize count.

    test_callback is not invoked for singleton identity or any proposal after
    the first failure. Callback errors are recorded failures, never acceptance.
    """
    cap = rational(cap)
    if cap < 1:
        raise ValueError('Distortion cap must be at least one')
    identity = tuple((i,) for i in range(K))
    best = {'partition': identity, 'cells': K, 'value': ONE}
    accepted, tested = [], []
    stop_index = stop_reason = None
    for index, candidate in enumerate(candidates):
        partition = validate_partition(candidate['partition'], K)
        if int(candidate['cells']) != len(partition):
            raise ValueError('Candidate count disagrees with partition')
        if len(partition) == K:
            accepted.append(index)
            continue
        try:
            result = dict(test_callback(partition))
            value = result.get('value')
            if value is not None:
                value = rational(value)
                if value < 1:
                    raise ValueError('A distortion statistic cannot be below one')
            passed = result.get('status') == 'full' and value is not None and value <= cap
        except Exception as error:
            result = {'value': None, 'status': 'failure',
                      'reason': type(error).__name__ + ': ' + str(error), 'rejection_bound': None}
            value, passed = None, False
        entry = {'candidate_index': index, 'value': value, 'status': result.get('status'),
                 'accepted': passed, 'reason': result.get('reason'),
                 'rejection_bound': result.get('rejection_bound'),
                 'cell_keys': result.get('cell_keys', ())}
        tested.append(entry)
        if not passed:
            stop_index = index
            stop_reason = result.get('reason') or ('statistic-above-cap' if value is not None else result.get('status', 'failure'))
            break
        accepted.append(index)
        if len(partition) < best['cells']:
            best = {'partition': partition, 'cells': len(partition), 'value': value}
    return dict(best, method=method, cap=cap,
                status='identity' if best['cells'] == K else 'certified-prefix',
                identity=best['cells'] == K,
                accepted_prefix_indices=tuple(accepted), stop_index=stop_index,
                stop_reason=stop_reason, tested=tested)


def evaluate_split(pilot_N, pilot_T, heldout_N, heldout_T, tree_orders,
                   caps=CAPS, alpha=ALPHA, root_policy=None, verifier=None,
                   immutable_ids=None):
    """One production split; all phases share cells across prespecified caps."""
    pilot_N, pilot_T = _counts(pilot_N, pilot_T)
    heldout_N, heldout_T = _counts(heldout_N, heldout_T)
    if len(pilot_N) != len(heldout_N) or sum(heldout_N) <= 0:
        raise ValueError('Pilot/held-out supports differ or held-out subset is empty')
    alpha = rational(alpha)
    if verifier is not None and root_policy is not None:
        raise ValueError('Supply verifier or root policy, not both')
    if isinstance(root_policy, dict):
        root_policy = RootPolicy(**root_policy)
    verifier = RootVerifier(root_policy) if verifier is None else verifier
    proposal = build_candidates(pilot_N, pilot_T, tree_orders, alpha, immutable_ids)
    ltt = HeldoutLTT(heldout_N, heldout_T, alpha, verifier)
    control = HeldoutCoupled(heldout_N, heldout_T, alpha, verifier)
    calibration = {'ltt': 0.0, 'split_coupled': 0.0}
    selection_seconds = 0.0
    selections = []
    for method, engine in (('ltt', ltt), ('split_coupled', control)):
        def timed_callback(partition):
            start = time.monotonic()
            try:
                return engine.partition(partition)
            finally:
                calibration[method] += time.monotonic() - start
        for cap in caps:
            start = time.monotonic()
            before = calibration[method]
            row = first_failure_select(proposal['candidates'], cap, timed_callback, len(pilot_N), method)
            elapsed = time.monotonic() - start
            selection_seconds += max(0.0, elapsed - (calibration[method] - before))
            selections.append(row)
    return {
        'pilot_order': proposal['pilot_order'], 'candidates': proposal['candidates'],
        'pilot_evidence': proposal['pilot_evidence'],
        'ltt_evidence': ltt.evidence(), 'control_evidence': control.evidence(),
        'selections': selections,
        'timings': {'pilot_seconds': proposal['pilot_seconds'],
                    'ltt_seconds': calibration['ltt'], 'control_seconds': calibration['split_coupled'],
                    'selection_seconds': selection_seconds},
        'root_cache': {'hits': verifier.cache_hits, 'misses': verifier.cache_misses},
    }


def _same(left, right):
    """List/tuple independent equality after JSON decoding, retaining numbers."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(x, y) for x, y in zip(left, right))
    return left == right


def replay_split(evidence, pilot_N, pilot_T, heldout_N, heldout_T, tree_orders,
                 immutable_ids=None, caps=CAPS):
    """Exact saved-decision replay; no special functions or mass optimization.

    Pilot inverse-beta values remain untrusted. Their saved arithmetic rebuilds
    the complete count recurrences and sequence. Verified root sign enclosures,
    count/error linkage, exact odds and the actual first-failure return are
    checked independently of any production root or optimizer invocation.
    """
    from experiments.jmlr_verified_adaptive_v1_replay_checks import _need, _root, validate_records
    from experiments.jmlr_verified_adaptive_v1_math import region_from_bounds

    pilot_N, pilot_T = _counts(pilot_N, pilot_T)
    N, T = _counts(heldout_N, heldout_T)
    K, n = len(N), sum(N)
    _need(len(pilot_N) == K and n > 0, 'Replay split support/total differs')
    order = pilot_order(pilot_N, pilot_T, immutable_ids)
    _need(tuple(evidence['pilot_order']) == order, 'Pilot order does not use pilot counts and immutable IDs')
    pilot = evidence['pilot_evidence']
    _need(pilot['alpha'] == ALPHA, 'Pilot error differs from frozen alpha')
    _need(tuple(tuple(k) for k in pilot['state_interval_count_keys']) == tuple(zip(pilot_T, pilot_N)),
          'Pilot state interval linkage differs')
    ordinary = {}
    for interval in pilot['pooled_intervals']:
        x, m = interval['x'], interval['m']
        _need(isinstance(x, int) and isinstance(m, int) and 0 <= x <= m, 'Invalid pilot interval counts')
        _need((x, m) not in ordinary, 'Duplicate pilot interval count cache')
        lower, upper = interval['lower'], interval['upper']
        _need(isinstance(lower, float) and isinstance(upper, float) and
              math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1,
              'Malformed ordinary pilot interval')
        if interval['status'] == 'failure':
            _need(lower == 0 and upper == 1 and bool(interval['reason']), 'Favorable pilot interval failure')
        else:
            _need(interval['status'] == ('structural' if m == 0 else 'ordinary-cp'),
                  'Pilot interval status differs from count case')
            _need((x != 0 or lower == 0) and (x != m or upper == 1), 'Pilot structural endpoint differs')
        ordinary[x, m] = interval
    states = tuple(ordinary[t, ni] for t, ni in zip(pilot_T, pilot_N))
    scores = {}
    for record in pilot['cell_scores']:
        cell = tuple(record['cell'])
        _need(cell and cell == tuple(sorted(set(cell))) and 0 <= cell[0] <= cell[-1] < K,
              'Malformed pilot cell')
        _need(cell not in scores, 'Duplicate pilot cell score')
        if len(cell) == 1:
            _need(record['status'] == 'singleton' and record['value'] == 1.0,
                  'Pilot singleton score differs from structural one')
        else:
            pooled = ordinary[sum(pilot_T[i] for i in cell), sum(pilot_N[i] for i in cell)]
            involved = [states[i] for i in cell] + [pooled]
            if record['status'] == 'failure':
                _need(record['value'] is None and bool(record['reason']), 'Favorable pilot score failure')
            elif (pooled['lower'] <= 0 or pooled['upper'] >= 1
                  or any(states[i]['lower'] <= 0 or states[i]['upper'] >= 1 for i in cell)):
                _need(record['status'] == 'boundary' and record['value'] is None,
                      'Pilot boundary score is finite')
            else:
                _need(all(r['status'] != 'failure' for r in involved), 'Pilot interval failure was ignored')
                def o(v):
                    return v / (1 - v)
                score = max(1.0, *(o(states[i]['upper']) / o(pooled['lower']) for i in cell),
                            *(o(pooled['upper']) / o(states[i]['lower']) for i in cell))
                _need(record['status'] == 'ordinary-score' and record['value'] == score,
                      'Pilot odds-ranking arithmetic differs')
        scores[cell] = record['value']
    expected_cells = set(interval_cells(order))
    for leaf_order in tree_orders:
        expected_cells.update(build_tree(leaf_order))
    _need(set(scores) == expected_cells, 'Pilot family cell-score coverage differs')
    candidates = candidates_from_scores(order, tree_orders, scores)
    _need(_same(evidence['candidates'], candidates), 'Count recurrence/candidate sequence replay differs')

    direct = evidence['ltt_evidence']
    _need(direct['alpha'] == direct['gamma'] == ALPHA, 'LTT did not use total interval error alpha')
    grid_bits = direct['root_policy']['grid_bits']
    roots = {}
    for interval in direct['pooled_intervals']:
        x, m = interval['x'], interval['m']
        _need((x, m) not in roots, 'Duplicate held-out LTT count cache')
        _root(interval, x, m, ALPHA, grid_bits)
        roots[x, m] = interval
    if direct['state_interval_count_keys'] is None:
        states = None
        _need(not roots, 'LTT saved roots without a queried nontrivial cell')
    else:
        _need(tuple(tuple(k) for k in direct['state_interval_count_keys']) == tuple(zip(T, N)),
              'Held-out LTT state count linkage differs')
        states = tuple(roots[t, ni] for t, ni in zip(T, N))
    ltt_cells = {}
    for key, record in direct['cells'].items():
        cell = tuple(record['cell'])
        _need(cell and cell == tuple(sorted(set(cell))) and 0 <= cell[0] <= cell[-1] < K,
              'Malformed held-out LTT cell')
        _need(key == cell_key(cell) and cell not in ltt_cells, 'LTT cell key/uniqueness differs')
        if len(cell) == 1:
            _need(record['status'] == 'singleton' and record['value'] == ONE and
                  record['pooled_count_key'] is None and not record['plus'] and not record['minus'],
                  'LTT singleton is not structural one')
        else:
            _need(states is not None, 'LTT merged cell lacks raw-state intervals')
            count_key = (sum(T[i] for i in cell), sum(N[i] for i in cell))
            _need(tuple(record['pooled_count_key']) == count_key, 'Pooled LTT counts do not match held-out cell')
            pooled = roots[count_key]
            involved = [states[i] for i in cell] + [pooled]
            if any(r['status'] == 'fallback' for r in involved):
                _need(record['value'] is None and record['status'] == 'fallback', 'LTT fallback authorized finite value')
            elif (pooled['lower'] <= 0 or pooled['upper'] >= 1
                  or any(states[i]['lower'] <= 0 or states[i]['upper'] >= 1 for i in cell)):
                _need(record['value'] is None and record['status'] == 'boundary', 'LTT boundary authorized finite value')
            elif record['status'] == 'failure':
                _need(record['value'] is None and bool(record['reason']), 'LTT arithmetic failure authorized finite value')
            else:
                plus = tuple((i, odds(states[i]['upper']) / odds(pooled['lower'])) for i in cell)
                minus = tuple((i, odds(pooled['upper']) / odds(states[i]['lower'])) for i in cell)
                value = max(ONE, *(v for _, v in plus), *(v for _, v in minus))
                _need(record['status'] == 'full' and record['value'] == value and
                      _same(record['plus'], plus) and _same(record['minus'], minus),
                      'Held-out LTT rational odds statistic differs')
        ltt_cells[cell] = record

    coupled = evidence['control_evidence']
    region = coupled['region']
    if region is None:
        _need(all(len(r['cell']) == 1 for r in coupled['cells'].values()),
              'Control merged cell lacks a calibrated common region')
        control_cells = {}
        for key, record in coupled['cells'].items():
            cell = tuple(record['cell'])
            _need(key == cell_key(cell) and record['coupled'] == record['independent'] == ONE and
                  record['status'] == 'full' and not record['anchors'], 'Malformed control singleton')
            control_cells[cell] = record
    else:
        _need(region['K'] == K and region['n'] == n and tuple(region['N']) == N and tuple(region['T']) == T,
              'Control confidence region count linkage differs')
        gamma = ALPHA / (2 * K)
        _need(region['alpha'] == ALPHA and region['gamma'] == gamma and
              region['known_masses'] is None and region['mode'] == 'unknown_mass',
              'Control alpha/mode allocation differs')
        prop, mass = region['roots']['propensity'], region['roots']['mass']
        _need(len(prop) == len(mass) == K, 'Control coordinate root coverage differs')
        grid_bits = region['root_policy']['grid_bits']
        for i in range(K):
            _root(prop[i], T[i], N[i], gamma, grid_bits)
            _root(mass[i], N[i], n, gamma, grid_bits)
            _need((prop[i]['lower'], prop[i]['upper']) ==
                  (region['propensity_lower'][i], region['propensity_upper'][i]),
                  'Control propensity endpoints differ from root returns')
            _need((mass[i]['lower'], mass[i]['upper']) ==
                  (region['mass_lower'][i], region['mass_upper'][i]),
                  'Control mass endpoints differ from root returns')
        restored = region_from_bounds(region['mass_lower'], region['mass_upper'],
                                      region['propensity_lower'], region['propensity_upper'])
        _need(restored['valid'] == region['valid'] and restored['diagnostics'] == region['diagnostics'],
              'Control exact region feasibility differs')
        _need(region['root_fallbacks'] == sum(r['status'] == 'fallback' for r in list(prop) + list(mass)),
              'Control root fallback count differs')
        for key, record in coupled['cells'].items():
            _need(key == cell_key(record['cell']), 'Control cell mask key differs')
        control_cells = validate_records(coupled['cells'].values(), region)

    def replay_partition(partition, method):
        records = ltt_cells if method == 'ltt' else control_cells
        field = 'value' if method == 'ltt' else 'coupled'
        selected = [records[tuple(cell)] for cell in partition]
        bad = next((r for r in selected if r[field] is None), None)
        return {'value': None if bad is not None else max((r[field] for r in selected), default=ONE),
                'status': 'full' if bad is None else bad['status'],
                'reason': None if bad is None else bad['reason'],
                'cell_keys': tuple(cell_key(c) for c in partition), 'rejection_bound': None}

    rows = {(row['method'], row['cap']): row for row in evidence['selections']}
    expected = {(method, rational(cap)) for method in ('ltt', 'split_coupled') for cap in caps}
    _need(len(rows) == len(evidence['selections']) and set(rows) == expected, 'Split selection grid differs')
    for method, cap in sorted(expected):
        choice = first_failure_select(candidates, cap,
                                      lambda partition: replay_partition(partition, method), K, method)
        _need(_same(rows[method, cap], choice), 'First-failure prefix/actual returned partition differs')
    return {'status': 'passed', 'selections': len(rows), 'candidates': len(candidates),
            'pilot_cells': len(scores), 'ltt_roots': len(roots),
            'ltt_cells': len(ltt_cells), 'control_cells': len(control_cells)}
