"""One SCA procedure with two distinct operating objectives.

This module orchestrates the established verified routines. It does not
implement a second root solver, mixture optimizer, or search recurrence.
"""
from dataclasses import dataclass, field
from fractions import Fraction as Q
import numbers

import numpy as np
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier, rational
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import counts, exact_order, select_families
from experiments.jmlr_release_capacity_v1_search import candidate_cells, capacity_search
from experiments.jmlr_release_capacity_v1_interface import export_release, analyze_release
from experiments.jmlr_split_ablation_v1_core import _cell_lower, _relevant_unknown, _safe_selection


@dataclass
class SCAResult:
    """A certified selection and explicit optimization-completion information.

    A finite safe output need not establish an optimum if computation was
    unresolved. Mathematical infinity is distinct from unresolved computation.
    Provider state/memberships are not part of the exported analyst artifact.
    """
    objective: str
    partition: tuple | None
    certificate: Q | None
    status: str
    optimization_completed: bool
    metadata: dict
    _region: dict = field(repr=False)
    _records: dict = field(repr=False)

    @property
    def groups(self):
        return None if self.partition is None else len(self.partition)

    def admissible(self, *, raw_budget='2', aggregate_budget='3'):
        """Whether a completed result meets its capacity and sensitivity limits."""
        G, ceiling = rational(raw_budget), rational(aggregate_budget)
        if G < 1 or ceiling < 1:
            raise ValueError('Sensitivity budgets must be at least one')
        B = self.metadata.get('capacity', self._region['K'])
        return bool(self.optimization_completed and self.status == 'finite'
                    and self.partition is not None and len(self.partition) <= B
                    and self.certificate is not None and G*self.certificate <= ceiling)

    def release(self, N, T, successes, *, raw_budget='2'):
        """Export anonymous aggregate counts/bounds for an independent cohort.

        Coverage assumes independent calibration and outcome samples with the
        same population assignment law P(S, A): both state masses and treatment
        propensities. Observed counts need not match. Sharing only P(A | S) is
        insufficient: changed state masses can change aggregate propensities.

        The standard outcome interface requires calibration error1/40 and
        reserves1/80 each for raw/aggregate outcome families. Every selected
        group is retained, including groups with no treated observations.
        """
        if self.status != 'finite' or self.partition is None:
            raise ValueError('A completed finite selected certificate is required for export')
        return export_release(self._region, self.partition, self._records,
                              N, T, successes, G=rational(raw_budget))


class SCA:
    """Full-sample SCA using its principal Coupled Clopper–Pearson certificate.

    ``fit(cap=c)`` minimizes group count subject to completed certificate<=c.
    ``fit(capacity=B)`` minimizes the certificate over at mostB groups.
    These objectives may select different partitions. No population quantities
    or outcome observations enter either selection. The default numerical trial
    ceiling is2million; individual experiments retain their frozen policies.
    """
    def __init__(self, *, alpha='1/40', root_policy=None):
        self.alpha = rational(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError('alpha must lie strictly between zero and one')
        self.root_policy = RootPolicy(max_trials=2_000_000) if root_policy is None else root_policy
        if not isinstance(self.root_policy, RootPolicy):
            raise TypeError('root_policy must be a RootPolicy')

    def fit(self, N, T, *, cap=None, capacity=None, immutable_ids=None, tree_orders=None):
        """Use all supplied counts; default objective is minimum count at cap3/2.

        Both fixed trees are chosen before outcomes. Default trees are natural
        and a fixed permutation independent of assignments; a selected partition
        belongs to one complete constituent family, never a mixture of trees.
        """
        N, T = counts(N, T); K = len(N)
        if cap is not None and capacity is not None:
            raise ValueError('Specify a fixed cap OR a fixed capacity, not both')
        if capacity is None:
            cap = Q(3, 2) if cap is None else rational(cap)
            if cap < 1:
                raise ValueError('A distortion cap must be at least one')
        elif (isinstance(capacity, bool) or not isinstance(capacity, numbers.Integral)
              or not 1 <= capacity <= K):
            raise ValueError('capacity must be an integer between1 andK')
        ids = tuple(range(K)) if immutable_ids is None else tuple(immutable_ids)
        if len(ids) != K or len(set(ids)) != K:
            raise ValueError('Every declared state requires a distinct immutable ID')
        if tree_orders is None:
            tree_orders = (tuple(range(K)), tuple(map(int, np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([20261008, 101, K]))).permutation(K))))
        tree_orders = tuple(tuple(o) for o in tree_orders)
        verifier = RootVerifier(self.root_policy)
        region = exact.confidence_region(N, T, self.alpha, verifier=verifier)
        order = exact_order(N, T, ids)
        cells = candidate_cells(order, tree_orders)
        trees = tuple(exact.build_tree(o) for o in tree_orders)
        # A threshold rejection is not a completed upper bound. Capacity mode
        # computes the full table rather than recycling a threshold-pruned one.
        records = {c: cell_certificate(region, c, max_cap=cap) for c in cells}
        if capacity is None:
            upper = {c:r['coupled'] if r['status']=='full' else None for c,r in records.items()}
            lower = {c:_cell_lower(r) for c,r in records.items()}
            high = select_families(order, trees, upper, cap)['combined']
            low = select_families(order, trees, lower, cap)['combined']
            unknown = _relevant_unknown(order, trees, records, cap, high['cells'])
            chosen = _safe_selection(high, cap, K, 'full_sample_combined_family',
                count_optimal=high['cells']==low['cells'], tie_complete=not unknown,
                lower_count=low['cells'], upper_count=high['cells'],
                potentially_relevant_unresolved_cells=unknown)
            completed = bool(chosen['count_optimal'] and chosen['deterministic_tie_complete'])
            status = 'finite' if completed else 'unresolved'
            objective = 'minimum_groups_at_fixed_cap'
        else:
            chosen = dict(capacity_search(order, tree_orders, records, [capacity])['by_capacity'][int(capacity)])
            unknown = () if chosen['status'] != 'finite' else _relevant_unknown(
                order, trees, records, chosen['certificate'], int(capacity))
            completed = bool(chosen['completed'] and not unknown)
            status = chosen['status'] if completed else 'unresolved'
            chosen.update(deterministic_tie_complete=not unknown, count_optimal=completed)
            objective = 'minimum_certificate_at_fixed_capacity'
        metadata = dict(chosen, alpha=self.alpha, calibration_size=sum(N),
            full_assignment_sample=True, certificate_construction='Coupled Clopper–Pearson',
            family='two fixed-tree pruning families plus estimated-order contiguous partitions',
            order=order, tree_orders=tree_orders, root_policy=self.root_policy.__dict__,
            root_fallbacks=region['root_fallbacks'], root_cache_hits=verifier.cache_hits,
            cell_status_counts={s:sum(r['status']==s for r in records.values())
                                for s in ('full','rejected','boundary','failure')},
            secondary_ties='inherited local recurrence ties; no global secondary certificate optimum claimed')
        return SCAResult(objective, chosen['partition'], chosen['certificate'], status,
                         completed, metadata, region, records)
