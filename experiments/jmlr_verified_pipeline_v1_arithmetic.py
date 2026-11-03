"""Algebraically identical cell certificate with shared exact odds arithmetic.

The root backend is imported unchanged. The box/simplex allocation algorithm
is specialized to its scalar result: it retains the same breakpoints, sorted
allocation and exact witness checks, avoiding conversion of unused witness
coordinates to fractions. Fixed-mass mixtures use their exact linear identity.
Returned values, anchor order, early predicates and record fields retain the
historical verified interface, permitting exact record equality checks.
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm

from experiments import jmlr_verified_adaptive_v1_math as exact


def _endpoint_integers(region):
    key = (tuple(region['propensity_lower']), tuple(region['propensity_upper']))
    cached = region.get('_pipeline_endpoint_integers')
    if cached is None or cached[0] != key:
        lower, upper = key
        D = lcm(*(v.denominator for v in (*lower, *upper)))
        cached = (key, D,
                  tuple(v.numerator * (D // v.denominator) for v in lower),
                  tuple(v.numerator * (D // v.denominator) for v in upper))
        region['_pipeline_endpoint_integers'] = cached
    return cached[1:]


def mixture_value(region, cell, coefficients, maximize=False,
                  integer_coefficients=None, coefficient_denominator=None):
    """Same closed box/simplex extremum and checked witness, scalar result only.

    If p_i=P_i/D and v_i=V_i/Dc, the original witness objective check
    Σ p_i v_i / (S/D) = q is exactly Σ P_i V_i = sign*best_num.
    Thus checking this integer identity and the unchanged integer feasibility
    predicates replaces fraction conversion without weakening the check. A
    common coefficient denominator larger than the minimum merely scales all
    coefficient integers by one positive factor, preserving every ordering,
    breakpoint, ratio comparison and resulting rational extremum.
    """
    cell = exact._cell(cell, region['K'])
    if not region['valid']:
        raise ValueError('No admissible positive law')
    if integer_coefficients is None:
        values = tuple(exact.rational(v) for v in coefficients)
        if len(values) != region['K']:
            raise ValueError('Coefficient vector has wrong support')
        Dc = lcm(*(values[i].denominator for i in cell))
        integers = {i: values[i].numerator * (Dc // values[i].denominator) for i in cell}
    else:
        if (len(integer_coefficients) != region['K'] or not isinstance(coefficient_denominator, int)
                or coefficient_denominator <= 0):
            raise ValueError('Invalid exact integer coefficient representation')
        Dc, integers = coefficient_denominator, integer_coefficients
    if region['known_masses'] is not None:
        p = region['known_masses']
        S = sum(p[i] for i in cell)
        return {'value': sum(p[i] * Fraction(integers[i], Dc) for i in cell) / S,
                'status': 'exact'}
    a, b, D = region['_a'], region['_b'], region['_D']
    A, lo, hi, outside = exact._cell_mass(region, cell)
    if hi <= 0 or hi < lo:
        raise ValueError('No positive feasible cell mass')
    if lo == 0:
        value = (max if maximize else min)(integers[i] for i in cell)
        return {'value': Fraction(value, Dc), 'status': 'closure'}
    sign = -1 if maximize else 1
    ordered_coefficients = {i: integers[i] * sign for i in cell}
    order = sorted(cell, key=lambda i: (ordered_coefficients[i], i))
    numerator = sum(a[i] * ordered_coefficients[i] for i in cell)
    candidates = {lo, hi}
    step = A
    for i in order:
        step += b[i] - a[i]
        if lo <= step <= hi:
            candidates.add(step)
    S, pos, allocated = A, 0, 0
    best_num = best_S = None
    for target in sorted(candidates):
        extra = target - S
        while extra:
            while pos < len(order) and b[order[pos]] - a[order[pos]] == allocated:
                pos += 1
                allocated = 0
            if pos == len(order):
                raise ArithmeticError('Capacity allocation exhausted')
            i = order[pos]
            take = min(extra, b[i] - a[i] - allocated)
            numerator += take * ordered_coefficients[i]
            allocated += take
            extra -= take
            S += take
        if best_num is None or numerator * best_S < best_num * target:
            best_num, best_S = numerator, target
    p = list(a)
    remaining = best_S - A
    for i in order:
        take = min(remaining, b[i] - a[i])
        p[i] += take
        remaining -= take
    if remaining:
        raise ArithmeticError('Cell witness allocation failed')
    remaining = D - sum(p)
    for i in outside:
        take = min(remaining, b[i] - a[i])
        p[i] += take
        remaining -= take
    if (remaining or sum(p) != D or any(not a[i] <= p[i] <= b[i] for i in range(region['K']))
            or sum(p[i] for i in cell) != best_S):
        raise ArithmeticError('Exact mass witness failed feasibility')
    if sum(p[i] * integers[i] for i in cell) != sign * best_num:
        raise ArithmeticError('Witness objective mismatch')
    return {'value': Fraction(sign * best_num, Dc * best_S), 'status': 'exact'}


def cell_certificate(region, cell, max_cap=None):
    cell = exact._cell(cell, region['K'])
    if max_cap is not None:
        max_cap = exact.rational(max_cap)
    answer = {'cell': cell, 'coupled': None, 'independent': None, 'status': 'full',
              'anchors': [], 'rejection_bound': None, 'mixture_calls': 0,
              'closure_calls': 0, 'reason': None}
    if len(cell) == 1:
        return dict(answer, coupled=exact.ONE, independent=exact.ONE)
    if not region['valid']:
        return dict(answer, status='failure', reason='invalid-region')
    lower, upper = region['propensity_lower'], region['propensity_upper']
    if any(lower[i] <= 0 or upper[i] >= 1 for i in cell):
        return dict(answer, status='boundary', reason='genuine-boundary-infinity')

    known = region['known_masses']
    endpoint_denominator, lower_integers, upper_integers = _endpoint_integers(region)
    linear = None
    if known is not None:
        # For a fixed rational p, its singleton feasible mass witness is p.
        # With p=P/Dp and v=V/De, q_C=Σ(P_i V_i)/(De ΣP_i).
        # Changing only anchor i from L_i to U_i adds P_i(U_i-L_i)
        # to the integer numerator. The reciprocal anchor subtracts the same
        # term from the upper-vector numerator. These are exact identities;
        # no optimization or floating approximation is substituted here.
        key = (tuple(known), tuple(lower), tuple(upper))
        cached = region.get('_pipeline_known_linear')
        if cached is None or cached[0] != key:
            Dp = lcm(*(v.denominator for v in known))
            De = endpoint_denominator
            P = tuple(v.numerator * (Dp // v.denominator) for v in known)
            L, U = lower_integers, upper_integers
            if any(p <= 0 for p in P) or sum(P) != Dp:
                return dict(answer, status='failure', reason='invalid-known-mass-integer-witness')
            cached = (key, P, De,
                      tuple(p * v for p, v in zip(P, L)),
                      tuple(p * v for p, v in zip(P, U)),
                      tuple(p * (u - l) for p, u, l in zip(P, U, L)))
            region['_pipeline_known_linear'] = cached
        _, P, De, weighted_lower, weighted_upper, delta = cached
        S = sum(P[i] for i in cell)
        linear = (sum(weighted_lower[i] for i in cell),
                  sum(weighted_upper[i] for i in cell), De * S, delta)

    def optimize(coefficients, maximize, known_numerator=None, integer_coefficients=None):
        if linear is not None:
            value = Fraction(known_numerator, linear[2])
            answer['mixture_calls'] += 1
            return value
        result = mixture_value(region, cell, coefficients, maximize,
                               integer_coefficients=integer_coefficients,
                               coefficient_denominator=endpoint_denominator)
        answer['mixture_calls'] += 1
        answer['closure_calls'] += result['status'] == 'closure'
        return result['value']

    try:
        qlo = optimize(lower, False, None if linear is None else linear[0], lower_integers)
        qhi = optimize(upper, True, None if linear is None else linear[1], upper_integers)
        # Odds are increasing, so the extrema equal odds(min lower) and
        # odds(max upper) used by the unchanged routine. Compute each once.
        key = (tuple(lower), tuple(upper))
        cached = region.get('_pipeline_raw_odds')
        if cached is None or cached[0] != key:
            # Outside this queried cell, genuine boundary coordinates remain
            # None. They cannot be queried for finite merged-cell odds because
            # the boundary guard above returns first. The exact endpoint key
            # prevents stale reuse even if a caller replaces region vectors.
            cached = (key,
                      tuple(exact.odds(v) if 0 < v < 1 else None for v in lower),
                      tuple(exact.odds(v) if 0 < v < 1 else None for v in upper))
            region['_pipeline_raw_odds'] = cached
        lower_odds, upper_odds = cached[1], cached[2]
        minimum_odds = min(lower_odds[i] for i in cell)
        maximum_odds = max(upper_odds[i] for i in cell)
        independent = max(exact.ONE, maximum_odds / exact.odds(qlo),
                          exact.odds(qhi) / minimum_odds)
        answer.update(independent=independent,
                      independent_q_lower=qlo, independent_q_upper=qhi)
        coupled = exact.ONE
        order = sorted(cell,
                       key=lambda i: (max(upper_odds[i] / minimum_odds,
                                          maximum_odds / lower_odds[i]), i),
                       reverse=True)
        for i in order:
            integers = list(lower_integers) if linear is None else lower_integers
            if linear is None:
                integers[i] = upper_integers[i]
            qplus = optimize(lower, False, None if linear is None else linear[0] + linear[3][i], integers)
            integers = list(upper_integers) if linear is None else upper_integers
            if linear is None:
                integers[i] = lower_integers[i]
            qminus = optimize(upper, True, None if linear is None else linear[1] - linear[3][i], integers)
            plus = upper_odds[i] / exact.odds(qplus)
            minus = exact.odds(qminus) / lower_odds[i]
            coupled = max(coupled, plus, minus)
            answer['anchors'].append({'state': i, 'q_plus': qplus, 'q_minus': qminus,
                                      'plus': plus, 'minus': minus})
            if max_cap is not None and min(coupled, independent) > max_cap:
                return dict(answer, status='rejected',
                            rejection_bound=min(coupled, independent),
                            reason='proved-rejection-above-largest-cap')
        if coupled > independent:
            raise ArithmeticError('Exact coupled bound exceeds comparator')
        return dict(answer, coupled=coupled)
    except Exception as error:
        return dict(answer, coupled=None, independent=answer['independent'],
                    status='failure', reason=type(error).__name__ + ': ' + str(error))
