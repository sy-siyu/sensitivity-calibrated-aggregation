"""Fresh, instrumented outcome inference from finite-state sufficient counts.

Aggregate work runs first using its own empty root cache and only the release
artifact. Raw-reference work then starts with a different empty cache. The
retained context is provider-side assignment information; it is never passed
to the aggregate analyst. Repeated use retains assignment projection but
recomputes outcome CP intervals and endpoints with fresh root caches.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F
import numbers
import resource
import sys
import time

from experiments import jmlr_outcome_inference_v1_lp as lp
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_outcome_inference_v1_core import (
    ALPHA_CAL, ALPHA_RAW, ALPHA_Z, confidence_enclosure, intersect_enclosures,
    outcome_family,
)
from experiments.jmlr_release_capacity_v1_interface import (
    _validate_release, export_release, release_bytes,
)
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier


VERSION = 'jmlr_computation_v1_inference'
G = F(2)
DEFAULT_POLICY = RootPolicy(max_trials=2_000_000)


def peak_rss_bytes():
    """Process-lifetime high-water mark; never a difference of stage peaks."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == 'darwin' else 1024)


class InstrumentedRootVerifier(RootVerifier):
    """Unchanged root arithmetic with request counters and exact-key cache."""
    def __init__(self, policy=None):
        super().__init__(DEFAULT_POLICY if policy is None else policy)
        self.interval_requests = 0
        self.interval_status_counts = Counter()

    def interval(self, x, m, gamma, proposal=None):
        self.interval_requests += 1
        answer = super().interval(x, m, gamma, proposal)
        self.interval_status_counts[answer['status']] += 1
        return answer


def root_work(verifier):
    """Count numerical work once per distinct computed lower/reflected root.

    The accompanying CP records contain the numerical proof evidence. Keys
    here connect that evidence to legitimate within-family cache reuse.
    """
    entries = tuple(verifier.cache.items())
    values = tuple(value for _, value in entries)
    precision = Counter(str(value.get('precision', 0)) for value in values)
    reasons = Counter(str(value.get('reason')) for value in values
                      if value.get('status') == 'fallback')
    return {
        'interval_requests': getattr(verifier, 'interval_requests', None),
        'interval_status_counts': dict(getattr(verifier, 'interval_status_counts', {})),
        'root_requests': verifier.cache_hits + verifier.cache_misses,
        'root_cache_hits': verifier.cache_hits,
        'root_cache_misses': verifier.cache_misses,
        'root_cache_entries': len(entries),
        'root_evaluations': sum(value.get('evaluations', 0) for value in values),
        'max_precision': max((value.get('precision', 0) for value in values), default=0),
        'precision_counts': dict(precision),
        'precision_escalated_roots': sum(value.get('precision', 0) > verifier.policy.initial_precision
                                        for value in values),
        'fallback_roots': sum(value.get('status') == 'fallback' for value in values),
        'fallback_reason_counts': dict(reasons),
        'cache_keys': tuple({'x': int(key[0]), 'm': int(key[1]), 'gamma': str(key[2]),
                             'policy': asdict(key[3]),
                             'proposal': None if key[4] is None else str(key[4])}
                            for key, _ in entries),
    }


def _counts(N, T, V, K):
    vectors = []
    for values in (N, T, V):
        values = tuple(values)
        if len(values) != K or any(isinstance(x, bool) or not isinstance(x, numbers.Integral)
                                   or x < 0 for x in values):
            raise ValueError('Cohort counts must include every declared state as nonnegative integers')
        vectors.append(tuple(map(int, values)))
    N, T, V = vectors
    if any(not v <= t <= n for n, t, v in zip(N, T, V)):
        raise ValueError('Cohort counts require 0 <= V <= T <= N')
    return N, T, V


def _policy(region, supplied):
    policy = DEFAULT_POLICY if supplied is None else supplied
    if not isinstance(policy, RootPolicy):
        raise ValueError('An immutable RootPolicy is required')
    if region.get('root_policy') != asdict(policy):
        raise ValueError('Calibration and outcome numerical policies must match')
    if region.get('alpha') != ALPHA_CAL or region.get('mode') != 'unknown_mass':
        raise ValueError('This study requires unknown-mass calibration at alpha=1/40')
    return policy


def aggregate_inference(artifact):
    """Aggregate-only path: artifact is the sole accepted data argument.

    Global and cellwise inference share the same CP family. Fresh root caches
    are intentional even when a later repeat happens to share count values.
    """
    start = time.perf_counter()
    pbox, ebox, U, N, T, V, budget, policy = _validate_release(artifact)
    if budget != G:
        raise ValueError('The computation study fixes G=2')
    validated = time.perf_counter()
    verifier = InstrumentedRootVerifier(policy)
    initial_entries = len(verifier.cache)
    tick = time.perf_counter()
    rbox, roots = outcome_family(V, T, ALPHA_Z, verifier)
    cp_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    methods = {
        'z_global': confidence_enclosure(pbox, ebox, rbox, G * max(U)),
        'z_cellwise': confidence_enclosure(pbox, ebox, rbox, tuple(G * u for u in U)),
    }
    endpoints_seconds = time.perf_counter() - tick
    outer, inner = (methods[key]['interval'] for key in ('z_global', 'z_cellwise'))
    contained = outer[0] <= inner[0] <= inner[1] <= outer[1]
    if not contained and not any(method['fallback'] for method in methods.values()):
        raise ArithmeticError('Completed cellwise interval escaped the global enclosure')
    work = root_work(verifier)
    return {
        'methods': methods, 'outcome_bounds': rbox, 'outcome_roots': roots,
        'certificate': max(U), 'global_budget': G * max(U),
        'cell_budgets': tuple(G * u for u in U), 'group_count': len(U),
        'cohort_size': sum(N), 'root_policy': asdict(policy),
        'root_fallbacks': sum(root['status'] == 'fallback' for root in roots),
        'cellwise_contained_in_global': contained,
        'cache_initial_entries': initial_entries, 'work': work,
        'stage_seconds': {
            'aggregate_validation_seconds': validated - start,
            'aggregate_outcome_cp_seconds': cp_seconds,
            'aggregate_endpoint_optimization_seconds': endpoints_seconds,
        },
        'compute_seconds': time.perf_counter() - start,
        'peak_rss_bytes': peak_rss_bytes(),
    }


@contextmanager
def _lp_instrumentation():
    """Scoped wrappers split existing LP work without changing its arithmetic.

    One active case per process is required. Solver proposal time excludes
    the nested rational certificate construction, avoiding double counting.
    """
    names = ('_matrix', '_proposal', 'certify_upper')
    originals = {name: getattr(lp, name) for name in names}
    timing = Counter()
    calls = Counter()

    def matrix(*args, **kwargs):
        start = time.perf_counter()
        try:
            return originals['_matrix'](*args, **kwargs)
        finally:
            timing['lp_solver_matrices_seconds'] += time.perf_counter() - start
            calls['matrix_calls'] += 1

    def certificate(*args, **kwargs):
        start = time.perf_counter()
        try:
            return originals['certify_upper'](*args, **kwargs)
        finally:
            timing['lp_certificate_construction_seconds'] += time.perf_counter() - start
            calls['certificate_construction_calls'] += 1

    def proposal(*args, **kwargs):
        before = timing['lp_certificate_construction_seconds']
        start = time.perf_counter()
        try:
            return originals['_proposal'](*args, **kwargs)
        finally:
            timing['lp_solver_proposal_seconds'] += (time.perf_counter() - start
                - (timing['lp_certificate_construction_seconds'] - before))
            calls['solver_proposal_calls'] += 1

    lp._matrix, lp._proposal, lp.certify_upper = matrix, proposal, certificate
    try:
        yield timing, calls
    finally:
        for name, value in originals.items():
            setattr(lp, name, value)


def _raw_references(region, partition, T, V, aggregate, policy, time_limit):
    stages = {}
    verifier = InstrumentedRootVerifier(policy)
    initial_entries = len(verifier.cache)
    tick = time.perf_counter()
    raw_box, roots = outcome_family(V, T, ALPHA_RAW, verifier)
    stages['raw_outcome_cp_seconds'] = time.perf_counter() - tick
    pbox = tuple(zip(region['mass_lower'], region['mass_upper']))
    ebox = tuple(zip(region['propensity_lower'], region['propensity_upper']))
    tick = time.perf_counter()
    raw = confidence_enclosure(pbox, ebox, raw_box, G)
    stages['raw_endpoint_optimization_seconds'] = time.perf_counter() - tick
    static_before = lp._static_rows.cache_info()._asdict()
    tick = time.perf_counter()
    lp_region = lp.build_region(pbox, ebox, raw_box, partition, aggregate['outcome_bounds'], G)
    stages['lp_construction_seconds'] = time.perf_counter() - tick
    with _lp_instrumentation() as (timing, calls):
        tick = time.perf_counter()
        evidence = lp.solve_enclosure(lp_region, time_limit=time_limit)
        total = time.perf_counter() - tick
    for key in ('lp_solver_matrices_seconds', 'lp_solver_proposal_seconds',
                'lp_certificate_construction_seconds'):
        stages[key] = timing[key]
    stages['lp_solver_orchestration_seconds'] = total - sum(timing.values())
    tick = time.perf_counter()
    try:
        verification = lp.verify_saved(evidence, lp_region)
    except (ValueError, ArithmeticError, KeyError, TypeError) as error:
        # Preserve proposed evidence and a conservative operational result.
        verification = {'status': 'verification_failure', 'reason': f'{type(error).__name__}: {error}'}
    stages['lp_exact_verification_seconds'] = time.perf_counter() - tick
    interval = (evidence['interval'] if verification['status'] == 'verified' else (F(0), F(1)))
    tick = time.perf_counter()
    pooled = intersect_enclosures(interval, raw['interval'], aggregate['methods']['z_cellwise']['interval'])
    pooled['lp_status'] = evidence['status']
    pooled['lp_verification_status'] = verification['status']
    pooled['lp_fallback'] = evidence['fallback'] or verification['status'] != 'verified'
    stages['lp_intersection_seconds'] = time.perf_counter() - tick
    return {
        'methods': {'raw_stratified': raw, 'raw_pooled': pooled},
        'outcome_bounds': raw_box, 'outcome_roots': roots, 'LP': evidence,
        'LP_verification': verification, 'stage_seconds': stages,
        'root_fallbacks': sum(root['status'] == 'fallback' for root in roots),
        'cache_initial_entries': initial_entries, 'work': root_work(verifier),
        'lp_work': {
            'nvars': lp_region.nvars, 'inequalities': len(lp_region.rows),
            'equalities': len(lp_region.equalities),
            'inequality_nonzeros': sum(map(len, lp_region.rows)),
            'equality_nonzeros': sum(map(len, lp_region.equalities)),
            'solver_status': {key: value['resolution'] for key, value in evidence['solver'].items()},
            'verification_status': verification['status'],
            'certificates_verified': sum(value is not None for value in evidence['certificates'].values())
                                     if verification['status'] == 'verified' else 0,
            'static_cache_before': static_before,
            'static_cache_after': lp._static_rows.cache_info()._asdict(), **dict(calls),
        },
    }


def infer_cohort(region, partition, records, N, T, V, *, policy=None,
                 lp_time_limit=10, retained=None):
    """Fresh aggregate pipeline, then additional raw-reference inference.

    The timer starts from already loaded sufficient counts and completed
    assignment selection. It excludes calibration/search and all population
    calculations. ``retained`` is exclusively an assignment-context reuse;
    outcome roots, endpoints, LP proposals and exact checks are recomputed.
    """
    start = time.perf_counter()
    stages = {}
    policy = _policy(region, policy)
    partition = exact.validate_partition(partition, region['K'])
    N, T, V = _counts(N, T, V, region['K'])
    stages['inference_input_validation_seconds'] = time.perf_counter() - start
    if retained is None:
        # A fresh primary case must not inherit the LP static-row cache from
        # another case. Normal reuse within the case/repeat remains allowed.
        lp._static_rows.cache_clear()
        clean_region = dict(region)
        clean_region.pop('_release_projection_cache', None)
        tick = time.perf_counter()
        artifact = export_release(clean_region, partition, records, N, T, V, G=G)
        stages['projection_export_seconds'] = time.perf_counter() - tick
        template = deepcopy(artifact)
        for group in template['groups']:
            group['counts'] = {'size': 0, 'treated': 0, 'treated_successes': 0}
        retained = {
            'version': VERSION, 'region': region, 'partition': partition,
            'records': records, 'policy': asdict(policy), 'release_template': template,
            'projection_evidence': clean_region['_release_projection_cache'],
        }
        projection_reused = False
    else:
        if retained.get('version') != VERSION or retained['region'] is not region or (
                retained['partition'] != partition or retained['policy'] != asdict(policy)):
            raise ValueError('Retained context must be the same completed assignment selection')
        tick = time.perf_counter()
        artifact = deepcopy(retained['release_template'])
        for group, cell in zip(artifact['groups'], partition):
            group['counts'] = {'size': sum(N[i] for i in cell),
                               'treated': sum(T[i] for i in cell),
                               'treated_successes': sum(V[i] for i in cell)}
        _validate_release(artifact)
        stages['projection_export_seconds'] = time.perf_counter() - tick
        projection_reused = True
    # No raw outcome routine has run or populated this aggregate root cache.
    aggregate_started = time.perf_counter() - start
    aggregate = aggregate_inference(artifact)
    aggregate_completed = time.perf_counter() - start
    stages.update(aggregate['stage_seconds'])
    stages['aggregate_unallocated_seconds'] = aggregate_completed - sum(stages.values())
    aggregate_peak = peak_rss_bytes()
    raw_started = time.perf_counter() - start
    raw = _raw_references(region, partition, T, V, aggregate, policy, lp_time_limit)
    stages.update(raw['stage_seconds'])
    raw_completed = time.perf_counter() - start
    stages['raw_reference_unallocated_seconds'] = (raw_completed - raw_started
                                                  - sum(raw['stage_seconds'].values()))
    methods = {**aggregate['methods'], **raw['methods']}
    payload_bytes = len(release_bytes(artifact))
    finish = time.perf_counter() - start
    stages['inference_unallocated_seconds'] = finish - sum(stages.values())
    return {
        'version': VERSION, 'artifact': artifact, 'methods': methods,
        'certificate': aggregate['certificate'], 'global_budget': aggregate['global_budget'],
        'cell_budgets': aggregate['cell_budgets'], 'payload_bytes': payload_bytes,
        'aggregate': aggregate, 'raw_reference': raw,
        'z_outcome_bounds': aggregate['outcome_bounds'], 'z_outcome_roots': aggregate['outcome_roots'],
        'raw_outcome_bounds': raw['outcome_bounds'], 'raw_outcome_roots': raw['outcome_roots'],
        'LP': raw['LP'], 'LP_verification': raw['LP_verification'],
        'stage_seconds': stages,
        'totals': {
            'aggregate_pipeline_seconds': aggregate_completed,
            'additional_raw_reference_seconds': raw_completed - raw_started,
            'inference_total_seconds': finish,
        },
        'chronology': {'aggregate_started_seconds': aggregate_started,
                       'aggregate_completed_seconds': aggregate_completed,
                       'raw_reference_started_seconds': raw_started,
                       'raw_reference_completed_seconds': raw_completed,
                       'aggregate_completed_before_raw_reference': aggregate_completed <= raw_started},
        'reuse': {'assignment_projection': projection_reused, 'outcome_root_cache': False,
                 'saved_outcome_roots': False, 'saved_LP': False},
        'cache_initial_entries': {'aggregate_outcomes': aggregate['cache_initial_entries'],
                                  'raw_outcomes': raw['cache_initial_entries']},
        'aggregate_pipeline_peak_rss_bytes': aggregate_peak,
        'peak_rss_bytes': peak_rss_bytes(), 'retained_assignment': retained,
    }


def infer_repeated_cohort(retained_assignment, N, T, V, *, lp_time_limit=10):
    """One additional cohort, retaining assignment evidence/projection only."""
    context = retained_assignment
    return infer_cohort(context['region'], context['partition'], context['records'], N, T, V,
                        policy=RootPolicy(**context['policy']), lp_time_limit=lp_time_limit,
                        retained=context)


def verify_saved_evidence(result):
    """Replay exact LP evidence without rerunning CP roots or an optimizer.

    This verifies the saved exact LP input region and bound expansions. The
    provider's assignment certificate and CP root evidence are separate.
    """
    evidence = result['LP']
    answer = lp.verify_saved(evidence)
    return {**answer, 'certificates_verified': sum(
        value is not None for value in evidence['certificates'].values())}
