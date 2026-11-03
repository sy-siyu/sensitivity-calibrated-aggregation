"""Explicit exact nominal input laws for the preserved finite-support draws.

The first K-1 stored binary probabilities are retained and the last is the
residual. This declares NumPy's documented multinomial input-law convention;
it does not verify the sampler's floating-point internals or random generator.
Relabeling permutes an already declared whole law and never reapplies residuals.
"""
from __future__ import annotations

from fractions import Fraction as Q
from math import isfinite, lcm
import numbers

from experiments.jmlr_verified_adaptive_v1_math import canonical_partition, odds, validate_partition


CONVENTION = 'first K-1 stored binary masses; last mass exact residual; propensities stored binary rationals'


def _binary(values, name):
    values = tuple(values)
    result, hexadecimal = [], []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(name + ' requires stored real numeric scalars')
        f = float(value)
        if not isfinite(f):
            raise ValueError(name + ' contains a nonfinite stored float')
        result.append(Q.from_float(f))
        hexadecimal.append(f.hex())
    return tuple(result), tuple(hexadecimal)


def validate_law(law):
    K = law['K']
    p, e = tuple(law['p']), tuple(law['e'])
    if (isinstance(K, bool) or not isinstance(K, int) or K < 1
            or len(p) != K or len(e) != K
            or any(not isinstance(v, Q) for v in p + e)
            or any(v <= 0 for v in p) or sum(p) != 1
            or any(not 0 < v < 1 for v in e)):
        raise ValueError('Exact law requires positive rational unit masses and interior rational propensities')
    if law.get('convention') != CONVENTION or not law.get('id') or not law.get('source_identity'):
        raise ValueError('Exact law lacks its nominal convention/source identity')
    for key in ('mu0', 'mu1'):
        if key in law and (len(law[key]) != K or any(not isinstance(v, Q) or not 0 <= v <= 1 for v in law[key])):
            raise ValueError('Invalid exact binary outcome probability')
    return law


def nominal_law(population, source_identity, law_id=None):
    """Freeze the original generator-input coordinates exactly once."""
    if 'convention' in population or population.get('residual_applied_once'):
        raise ValueError('An already declared exact law must be relabeled as a whole, not readapted')
    original, p_hex = _binary(population['p'], 'Masses')
    e, e_hex = _binary(population['e'], 'Propensities')
    K = len(original)
    if not K or len(e) != K or any(v <= 0 for v in original):
        raise ValueError('Original stored law must have positive masses on one common support')
    residual = Q(1) - sum(original[:-1])
    if residual <= 0:
        raise ValueError('The nominal residual mass must be strictly positive')
    p = original[:-1] + (residual,)
    adjustments = tuple(a - b for a, b in zip(p, original))
    ids = tuple(population.get('immutable_ids', range(K)))
    if len(ids) != K or len(set(ids)) != K:
        raise ValueError('Law coordinates need distinct immutable IDs')
    law = {
        'id': str(population['id'] if law_id is None else law_id),
        'source_identity': str(source_identity), 'K': K, 'p': p, 'e': e,
        'immutable_ids': ids, 'convention': CONVENTION,
        'original_binary_p': original, 'original_binary_e': e,
        'original_float_hex': {'p': p_hex, 'e': e_hex},
        'original_mass_sum': sum(original), 'original_sum_discrepancy': sum(original) - 1,
        'mass_adjustments': adjustments, 'last_entry_adjustment': adjustments[-1],
        'generator_last_entry_id': ids[-1], 'residual_applied_once': True,
        'relabel_history': (),
        'qualification': 'Declared nominal iid input law; no verification of RNG or all sampler arithmetic',
    }
    for key in ('mu0', 'mu1'):
        if key in population:
            values, hexadecimal = _binary(population[key], key)
            law[key] = values
            law['original_float_hex'][key] = hexadecimal
    for key in ('mass', 'overlap', 'geometry', 'name'):
        if key in population:
            law[key] = population[key]
    validate_law(law)
    if sum(adjustments) != 1 - sum(original) or any(adjustments[:-1]):
        raise ArithmeticError('Nominal law altered a nonresidual mass')
    return law


def relabel_law(law, permutation, law_id=None):
    """new_index -> old_index; no second residual adjustment after relabeling."""
    validate_law(law)
    permutation = tuple(permutation)
    K = law['K']
    if (any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in permutation)
            or sorted(permutation) != list(range(K))):
        raise ValueError('Relabeling requires a complete state permutation')
    permutation = tuple(map(int, permutation))
    answer = dict(law)
    if law_id is not None:
        answer['id'] = str(law_id)
    for key in ('p', 'e', 'mu0', 'mu1', 'immutable_ids', 'original_binary_p',
                'original_binary_e', 'mass_adjustments'):
        if key in law:
            answer[key] = tuple(law[key][i] for i in permutation)
    answer['original_float_hex'] = {key: tuple(values[i] for i in permutation)
                                    for key, values in law['original_float_hex'].items()}
    answer['relabel_history'] = tuple(law['relabel_history']) + (permutation,)
    answer['last_entry_adjustment'] = law['last_entry_adjustment']
    validate_law(answer)
    if tuple(a - b for a, b in zip(answer['p'], answer['original_binary_p'])) != answer['mass_adjustments']:
        raise ArithmeticError('Relabeling changed the whole-law adjustment identity')
    return answer


class TruthEvaluator:
    """Exact population cell ratios using one common integer denominator.

    Only the extreme raw odds and the exact mass-weighted mixture are needed.
    This avoids repeated rational summation at every anchor and is algebraically
    identical to the maximum over all state/direction distortion ratios.
    """
    def __init__(self, law):
        validate_law(law)
        self.law, self.K = law, law['K']
        self.p, self.e = tuple(law['p']), tuple(law['e'])
        self.odds = tuple(odds(v) for v in self.e)
        Dp, De = lcm(*(v.denominator for v in self.p)), lcm(*(v.denominator for v in self.e))
        self.Dp, self.De = Dp, De
        self.pi = tuple(v.numerator * (Dp // v.denominator) for v in self.p)
        self.ei = tuple(v.numerator * (De // v.denominator) for v in self.e)
        self.pe = tuple(p * e for p, e in zip(self.pi, self.ei))
        self.cache = {}

    def cell(self, cell):
        cell = tuple(sorted(cell))
        if (not cell or len(set(cell)) != len(cell) or cell[0] < 0 or cell[-1] >= self.K
                or any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in cell)):
            raise ValueError('Invalid population cell')
        if cell not in self.cache:
            mass_integer = sum(self.pi[i] for i in cell)
            treated_integer = sum(self.pe[i] for i in cell)
            q = Q(treated_integer, self.De * mass_integer)
            oq = odds(q)
            value = Q(1) if len(cell) == 1 else max(Q(1), max(self.odds[i] for i in cell) / oq,
                                                  oq / min(self.odds[i] for i in cell))
            self.cache[cell] = {'cell': cell, 'value': value, 'mass': Q(mass_integer, self.Dp), 'propensity': q}
        return self.cache[cell]

    def partition(self, partition):
        partition = validate_partition(partition, self.K)
        cells = [self.cell(cell) for cell in partition]
        return {'partition': canonical_partition(partition), 'count': len(partition),
                'value': max(record['value'] for record in cells),
                'min_cell_mass': min(record['mass'] for record in cells),
                'max_cell_size': max(map(len, partition))}


def true_distortion(partition, law):
    return TruthEvaluator(law).partition(partition)['value']
