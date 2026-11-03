"""Exact minimax search under a reporting capacity.

This is the count-indexed counterpart of the existing verified threshold
search, using the recurrence structure in jmlr_statistical_baselines_v1_core.
No binary floating-point score conversion is used.  A bare ``None`` score
means a *completed mathematical infinity*.  Backend failure and partial
above-cap rejection records instead remain unresolved lower/upper intervals.

An unresolved cell has a certified lower bound (at least one) and no finite
upper bound.  Running both minimax recurrences can still prove an optimum
when every unresolved candidate is bounded below by a completed incumbent.
Thus partial rejection pruning is safe without treating a rejection as a
completed certificate.  A returned ``unresolved`` result is never a proof
that no admissible release exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from collections.abc import Mapping
import numbers

from experiments.jmlr_verified_adaptive_v1_math import (
    build_tree, canonical_partition, interval_cells, validate_partition,
)
from experiments.jmlr_verified_adaptive_v1_roots import rational

ONE = Fraction(1)


def _less(a, b):
    """Strict comparison of extended rationals, where None means +infinity."""
    return a is not None and (b is None or a < b)


def _maximum(a, b):
    return None if a is None or b is None else max(a, b)


def _finite(value):
    value = rational(value)
    if value < ONE:
        raise ValueError('Distortion bounds must be at least one')
    return value


@dataclass(frozen=True)
class _Score:
    lower: Fraction | None
    upper: Fraction | None
    completed: bool


def cell_score(value, method='coupled'):
    """Normalize exact scores while preserving completion semantics.

    Accepted forms are exact rational scalars; None for completed infinity;
    verified backend records; or explicit finite/infinite/unresolved records.
    A saved partial rejection supplies a lower bound only.  Its ``independent``
    field is deliberately not reused as the completed coupled certificate.
    """
    if not isinstance(value, Mapping):
        return _Score(None, None, True) if value is None else _Score(_finite(value), _finite(value), True)
    status = value.get('status')
    if status in ('boundary', 'infinite'):
        return _Score(None, None, True)
    if status in ('full', 'finite'):
        v = value.get(method, value.get('value', value.get('certificate')))
        if v is None:
            raise ValueError('A completed finite cell record needs an exact value')
        return _Score(_finite(v), _finite(v), True)
    if status not in ('failure', 'rejected', 'unresolved'):
        raise ValueError('Unknown cell certificate status: ' + str(status))
    lower = value.get('lower_bound', value.get('rejection_bound'))
    lower = ONE if lower is None else _finite(lower)
    return _Score(lower, None, False)


@dataclass(frozen=True)
class _Entry:
    lower: Fraction | None
    lower_partition: tuple
    lower_family: str
    upper: Fraction | None
    # None is unreachable/unavailable; an empty tuple is the reachable origin.
    upper_partition: tuple | None
    upper_family: str


def _entry(score, cell, family):
    partition = (cell,)
    return _Entry(score.lower, partition, family, score.upper,
                  partition if score.completed else None, family)


def _partition_key(partition, family):
    return len(partition), partition, family


def _best(entries):
    """Minimize lower and completed upper objectives separately."""
    entries = tuple(entries)
    if not entries:
        raise ValueError('No reachable candidates')
    lower = entries[0]
    upper = None
    for candidate in entries:
        if (_less(candidate.lower, lower.lower)
                or candidate.lower == lower.lower
                and _partition_key(candidate.lower_partition, candidate.lower_family)
                < _partition_key(lower.lower_partition, lower.lower_family)):
            lower = candidate
        if candidate.upper_partition is not None and (
                upper is None or _less(candidate.upper, upper.upper)
                or candidate.upper == upper.upper
                and _partition_key(candidate.upper_partition, candidate.upper_family)
                < _partition_key(upper.upper_partition, upper.upper_family)):
            upper = candidate
    return _Entry(lower.lower, lower.lower_partition, lower.lower_family,
                  None if upper is None else upper.upper,
                  None if upper is None else upper.upper_partition,
                  '' if upper is None else upper.upper_family)


def _combine(left, right, family):
    lower = _maximum(left.lower, right.lower)
    lower_partition = canonical_partition(left.lower_partition + right.lower_partition)
    if lower is None:
        # A known infinite cell makes this whole partition infinite even if
        # another cell's finite optimization did not complete.
        upper, upper_partition = None, lower_partition
    elif left.upper_partition is not None and right.upper_partition is not None:
        upper = _maximum(left.upper, right.upper)
        upper_partition = canonical_partition(left.upper_partition + right.upper_partition)
    else:
        upper, upper_partition = None, None
    return _Entry(lower, lower_partition, family, upper, upper_partition, family)


def _export(entry):
    completed = (entry.lower is None or entry.upper_partition is not None
                 and entry.upper == entry.lower)
    if entry.lower is None:
        partition, family = entry.lower_partition, entry.lower_family
        status, certificate = 'infinite', None
    elif completed:
        partition, family = entry.upper_partition, entry.upper_family
        status, certificate = 'finite', entry.upper
    else:
        partition, family = None, None
        status, certificate = 'unresolved', None
    return {
        'status': status, 'completed': completed, 'certificate': certificate,
        'partition': partition, 'cells': None if partition is None else len(partition),
        'family': family, 'lower_bound': entry.lower,
        'lower_bound_infinite': entry.lower is None,
        'candidate_partition': entry.upper_partition,
        'candidate_certificate': entry.upper,
        'candidate_status': ('unavailable' if entry.upper_partition is None else
                             'infinite' if entry.upper is None else 'finite'),
    }


def _order(order):
    order = tuple(order)
    if (not order or any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in order)
            or sorted(order) != list(range(len(order)))):
        raise ValueError('Order must include each declared support state exactly once')
    return tuple(map(int, order))


def _scores(cells, values, method):
    # Singleton distortion is structurally one, including unobserved states.
    return {cell: _Score(ONE, ONE, True) if len(cell) == 1 else cell_score(values[cell], method)
            for cell in cells}


def candidate_cells(order, tree_orders):
    order = _order(order)
    tree_orders = tuple(_order(o) for o in tree_orders)
    if len(tree_orders) != 2 or any(len(o) != len(order) for o in tree_orders):
        raise ValueError('Require the two fixed complete support tree orders')
    trees = tuple(build_tree(o) for o in tree_orders)
    return tuple(sorted(set(interval_cells(order)).union(*(set(tree) for tree in trees))))


def _contiguous(order, scores):
    K = len(order)
    H = [dict() for _ in range(K + 1)]
    H[0][0] = _Entry(ONE, (), 'contiguous', ONE, (), 'contiguous')
    for j in range(1, K + 1):
        for count in range(1, j + 1):
            candidates = []
            for i in range(count - 1, j):
                if count - 1 not in H[i]:
                    continue
                cell = tuple(sorted(order[i:j]))
                candidates.append(_combine(H[i][count - 1], _entry(scores[cell], cell, 'contiguous'),
                                           'contiguous'))
            if candidates:
                H[j][count] = _best(candidates)
    return H[K]


def count_contiguous(order, cell_scores, method='coupled'):
    """Exact minimax representative for each reachable *exact* cell count."""
    order = _order(order)
    scores = _scores(interval_cells(order), cell_scores, method)
    return {count: dict(_export(entry), count=count)
            for count, entry in _contiguous(order, scores).items()}


def _validate_tree(tree):
    if not tree:
        raise ValueError('Empty tree')
    root = max(tree, key=len)
    _order(root)
    visited = set()
    def visit(cell):
        if cell in visited or cell not in tree or tuple(sorted(cell)) != cell:
            raise ValueError('Malformed tree node')
        visited.add(cell)
        children = tree[cell]
        if children is None:
            if len(cell) != 1:
                raise ValueError('Every declared support state must be a tree leaf')
        else:
            if len(children) != 2 or set(children[0]) & set(children[1]) or tuple(sorted(children[0] + children[1])) != cell:
                raise ValueError('Children must disjointly partition the parent')
            for child in children:
                visit(child)
    visit(root)
    if visited != set(tree):
        raise ValueError('Tree has unreachable extra nodes')
    return root


def _tree(tree, scores, family):
    root = _validate_tree(tree)
    def visit(cell):
        answer = {1: _entry(scores[cell], cell, family)}
        children = tree[cell]
        if children is not None:
            left, right = (visit(child) for child in children)
            for count in range(2, len(cell) + 1):
                candidates = [_combine(left[lc], right[count - lc], family)
                              for lc in sorted(left) if count - lc in right]
                if candidates:
                    answer[count] = _best(candidates)
        return answer
    return visit(root)


def count_tree(tree, cell_scores, method='coupled', family='tree'):
    """Count-indexed minimax convolution over one constituent tree only."""
    _validate_tree(tree)
    scores = _scores(tree, cell_scores, method)
    return {count: dict(_export(entry), count=count)
            for count, entry in _tree(tree, scores, family).items()}


def capacity_search(order, tree_orders, cell_scores, capacities=None, method='coupled'):
    """Minimize the completed verified envelope over the union and |pi| <= B.

    A selected partition comes from exactly one constituent family.  Local
    recurrence ties compare canonical retained partitions; the final choice
    compares group count, canonical retained partition, and family label. The
    ``families`` tables also expose the exact-count optima for verification;
    these need not be monotone.  Only the at-most-capacity objective is nested.
    """
    order = _order(order)
    tree_orders = tuple(_order(o) for o in tree_orders)
    cells = candidate_cells(order, tree_orders)
    scores = _scores(cells, cell_scores, method)
    K = len(order)
    capacities = tuple(range(1, K + 1)) if capacities is None else tuple(capacities)
    if (len(set(capacities)) != len(capacities)
            or any(isinstance(b, bool) or not isinstance(b, numbers.Integral) or not 1 <= b <= K
                   for b in capacities)):
        raise ValueError('Capacities must be distinct integers in [1,K]')
    families = {'contiguous': _contiguous(order, scores)}
    for index, tree_order in enumerate(tree_orders):
        name = 'tree-' + str(index)
        families[name] = _tree(build_tree(tree_order), scores, name)
    by_count = {count: _best(table[count] for table in families.values() if count in table)
                for count in range(1, K + 1)}
    by_capacity = {int(b): dict(_export(_best(by_count[c] for c in range(1, int(b) + 1))), capacity=int(b))
                   for b in capacities}
    for choice in by_capacity.values():
        if choice['partition'] is not None:
            validate_partition(choice['partition'], K)
            if choice['cells'] > choice['capacity']:
                raise ArithmeticError('Capacity traceback violated the reporting limit')
    return {
        'by_capacity': by_capacity,
        'by_count': {c: dict(_export(e), count=c) for c, e in by_count.items()},
        'families': {f: {c: dict(_export(e), count=c) for c, e in table.items()}
                     for f, table in families.items()},
        'tie_rule': 'local canonical retained partitions; final group count, canonical retained partition, family label',
        'method': method,
    }


def admissible(choice, raw_budget=Fraction(2), aggregate_budget=Fraction(3)):
    """Admission uses a completed optimum and never an identity capacity fallback."""
    raw_budget, aggregate_budget = rational(raw_budget), rational(aggregate_budget)
    if raw_budget < 1 or aggregate_budget < 1:
        raise ValueError('Sensitivity budgets must be at least one')
    return bool(choice['status'] == 'finite' and choice['completed']
                and choice['cells'] <= choice['capacity']
                and raw_budget * choice['certificate'] <= aggregate_budget)
