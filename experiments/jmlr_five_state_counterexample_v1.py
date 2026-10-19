"""Bounded exact search and verification of a small same-cap counterexample.

This optional diagnostic is separate from the paper experiment workflow.  It
uses exact rational arithmetic throughout, searches at most 20,000 fixed
candidates, and then verifies the discovered fixture by independently
enumerating all set partitions and all contiguous cuts of five states.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations
import json
import random


SEARCH_SEED = 20261020
SEARCH_LIMIT = 20_000
CAP = F(19, 10)
MASSES = tuple(F(value, 1110) for value in (13, 610, 21, 89, 377))
PROPENSITIES = tuple(F(value, 1000) for value in (545, 723, 764, 836, 874))


def set_partitions(items=(0, 1, 2, 3, 4)):
    """Enumerate each unrestricted set partition once by recursive insertion."""
    items = tuple(items)
    if not items:
        return ((),)
    first, rest = items[0], items[1:]
    result = []
    for previous in set_partitions(rest):
        result.append(((first,),) + previous)
        for index in range(len(previous)):
            result.append(previous[:index] + ((first,) + previous[index],) + previous[index + 1:])
    return tuple(result)


def contiguous_partitions(order=(0, 1, 2, 3, 4)):
    """Enumerate the independent cut-mask family in the supplied order."""
    order = tuple(order)
    answer = []
    for count in range(1, len(order) + 1):
        for cuts in combinations(range(1, len(order)), count - 1):
            bounds = (0,) + cuts + (len(order),)
            answer.append(tuple(order[left:right] for left, right in zip(bounds[:-1], bounds[1:])))
    return tuple(answer)


def cell_distortion(masses, propensities, cell):
    if len(cell) == 1:
        return F(1)
    mass = sum((masses[index] for index in cell), F())
    propensity = sum((masses[index] * propensities[index] for index in cell), F()) / mass
    aggregate_odds = propensity / (1 - propensity)
    return max(F(1), *(max(
        (propensities[index] / (1 - propensities[index])) / aggregate_odds,
        aggregate_odds / (propensities[index] / (1 - propensities[index])))
        for index in cell))


def partition_distortion(masses, propensities, partition):
    return max(cell_distortion(masses, propensities, cell) for cell in partition)


def _first_bounded_candidate():
    """Replay the finite discovery check; return the first exact separation."""
    generator = random.Random(SEARCH_SEED)
    all_partitions = set_partitions()
    unrestricted_two = tuple(partition for partition in all_partitions if len(partition) == 2)
    contiguous_two = tuple(partition for partition in contiguous_partitions() if len(partition) == 2)
    weight_grid = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)
    for trial in range(SEARCH_LIMIT):
        propensities = tuple(F(value, 1000) for value in sorted(generator.sample(range(20, 981), 5)))
        weights = tuple(generator.choice(weight_grid) for _ in range(5))
        masses = tuple(F(value, sum(weights)) for value in weights)
        unrestricted = min((partition_distortion(masses, propensities, partition), partition)
                           for partition in unrestricted_two)
        contiguous = min((partition_distortion(masses, propensities, partition), partition)
                         for partition in contiguous_two)
        one_cell = partition_distortion(masses, propensities, (tuple(range(5)),))
        if unrestricted[0] < min(contiguous[0], F(2)):
            cap = (unrestricted[0] + min(contiguous[0], F(2))) / 2
            if one_cell > cap:
                return trial, masses, propensities
    return None


def _states(partition):
    return [[index + 1 for index in cell] for cell in partition]


def verify():
    """Return a JSON-ready receipt after exhaustive exact verification."""
    discovered = _first_bounded_candidate()
    if discovered is None:
        raise ArithmeticError("bounded exact search no longer finds the retained fixture")
    trial, masses, propensities = discovered
    if masses != MASSES or propensities != PROPENSITIES or trial != 464:
        raise ArithmeticError("bounded exact search changed")

    unrestricted = set_partitions()
    contiguous = contiguous_partitions()
    if len(unrestricted) != 52 or len(set(unrestricted)) != 52:
        raise ArithmeticError("unrestricted Bell-number enumeration changed")
    if len(contiguous) != 16 or len(set(contiguous)) != 16:
        raise ArithmeticError("contiguous cut enumeration changed")

    unrestricted_count = min(len(partition) for partition in unrestricted
                             if partition_distortion(MASSES, PROPENSITIES, partition) <= CAP)
    contiguous_count = min(len(partition) for partition in contiguous
                           if partition_distortion(MASSES, PROPENSITIES, partition) <= CAP)
    unrestricted_best = min((partition_distortion(MASSES, PROPENSITIES, partition), partition)
                            for partition in unrestricted if len(partition) == 2)
    contiguous_best = min((partition_distortion(MASSES, PROPENSITIES, partition), partition)
                          for partition in contiguous if len(partition) == 2)
    contiguous_three = min((partition_distortion(MASSES, PROPENSITIES, partition), partition)
                           for partition in contiguous if len(partition) == 3)
    one_cell = partition_distortion(MASSES, PROPENSITIES, (tuple(range(5)),))

    expected = {
        "unrestricted_count": 2,
        "contiguous_count": 3,
        "unrestricted_best": F(25244179, 13307679),
        "contiguous_best": F(12892811, 6780186),
        "contiguous_three": F(435789, 341539),
        "one_cell": F(78993551, 26371351),
    }
    actual = {
        "unrestricted_count": unrestricted_count,
        "contiguous_count": contiguous_count,
        "unrestricted_best": unrestricted_best[0],
        "contiguous_best": contiguous_best[0],
        "contiguous_three": contiguous_three[0],
        "one_cell": one_cell,
    }
    if actual != expected:
        raise ArithmeticError("five-state exact verification changed")
    if not unrestricted_best[0] < CAP < contiguous_best[0] or not CAP <= 2:
        raise ArithmeticError("cap separation changed")

    return {
        "status": "verified",
        "search": {"seed": SEARCH_SEED, "candidate_limit": SEARCH_LIMIT,
                   "first_hit_zero_based": trial, "candidates_checked": trial + 1,
                   "arithmetic": "fractions.Fraction exact rationals"},
        "fixture": {"masses": [str(value) for value in MASSES],
                    "propensities": [str(value) for value in PROPENSITIES],
                    "cap": str(CAP)},
        "enumeration": {"unrestricted_partitions": len(unrestricted),
                        "contiguous_cut_partitions": len(contiguous),
                        "independent_generators": True},
        "optima": {"unrestricted_count": unrestricted_count,
                   "contiguous_count": contiguous_count,
                   "one_cell_distortion": str(one_cell),
                   "unrestricted_best_two_distortion": str(unrestricted_best[0]),
                   "unrestricted_best_two_partition": _states(unrestricted_best[1]),
                   "contiguous_best_two_distortion": str(contiguous_best[0]),
                   "contiguous_best_two_partition": _states(contiguous_best[1]),
                   "contiguous_best_three_distortion": str(contiguous_three[0]),
                   "contiguous_best_three_partition": _states(contiguous_three[1])},
        "comparison": "unrestricted_best_two < cap < contiguous_best_two; cap <= 2",
        "scope": "small deterministic exact fixture; not an empirical study or a universal gap bound",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
