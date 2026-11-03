"""Frozen finite-reference-population benchmark and paired study sampling.

The public IHDP file contributes covariates only.  Its 747 complete records are
the declared reference population, rather than a sample used to infer an
unknown population support.  The new assignment and binary potential-outcome
laws depend only on the declared state.  Hence their actual conditional odds
sensitivity is one and they belong to the prespecified class G=2 given S.

Empirical state masses are exact counts/747.  We sample a uniform reference
record with NumPy's bounded-integer sampler instead of rounding those masses
onto a dyadic grid.  Independent ideal random words/record draws give the stated
law; PCG64 supplies reproducible pseudorandom streams, not a proof of iid bits.
"""
from __future__ import annotations

from bisect import bisect_left
import csv
from fractions import Fraction as F
import hashlib
from pathlib import Path

import numpy as np

from experiments.jmlr_outcome_inference_v1_sampling import (
    COMPONENTS, bernoulli, categorical, counts, threshold)
from experiments.jmlr_verified_pipeline_v1_laws import CONVENTION, validate_law


STUDY_SEED = 2026101502
DEVELOPMENT_SEED = 202610150201
BENCHMARK_ID = 'IHDP747_bwquartile_momage_sex'
BENCHMARK_PROFILES = ('low_heterogeneity', 'high_heterogeneity')
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT / 'data/ihdp_npci_1.csv'
SOURCE_SHA256 = '2bdc8b94fdf2f669b1d3d49325fd9359d88418566545003867841312a1653411'
SOURCE_CSV = 'https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv'
SOURCE_LOADER = 'https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets.py'
SOURCE_VARIABLES = 'https://raw.githubusercontent.com/vdorie/npci/master/examples/ihdp_sim/data.R'


def exact_quantile(values, probability):
    """Linear sample quantile of exact decimal coordinates, without float ties."""
    values = tuple(sorted(map(F, values)))
    probability = F(probability)
    if not values or not 0 <= probability <= 1:
        raise ValueError('Quantiles require observations and probability in [0,1]')
    rank = probability * (len(values) - 1)
    lower = rank.numerator // rank.denominator
    weight = rank - lower
    if not weight:
        return values[lower]
    return (1 - weight) * values[lower] + weight * values[lower + 1]


def build_benchmark(path=None):
    """Declare the complete fixed benchmark before drawing any study cohorts.

    Returned ``record_states`` are a tuple suitable for frozen protocol files.
    ``law`` is compatible with the inherited TruthEvaluator.  The legacy
    ``convention`` field is retained solely for that validator; the authoritative
    ``exact_input_convention`` explicitly records counts/747 and no residual or
    dyadic mass conversion is applied.
    """
    path = DEFAULT_DATA if path is None else Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Reference-population file differs from the verified public IHDP CSV')
    rows = tuple(csv.reader(payload.decode('utf-8').splitlines()))
    if len(rows) != 747 or any(len(row) != 30 for row in rows):
        raise ValueError('The frozen IHDP population requires exactly 747 rows and 30 columns')
    # The official CEVAE loader uses x=data[:,5:].  Original column order in
    # vdorie/npci is bw,b.head,preterm,birth.o,nnhealth,momage,sex,... .
    birth_weight = tuple(F(row[5]) for row in rows)
    mother_age = tuple(F(row[10]) for row in rows)
    sex = tuple(F(row[11]) for row in rows)
    if any(value not in (F(0), F(1)) for value in sex):
        raise ValueError('Frozen infant-sex covariate must be binary')
    bw_cutpoints = tuple(exact_quantile(birth_weight, q) for q in (F(1,4), F(1,2), F(3,4)))
    age_cutpoint = exact_quantile(mother_age, F(1,2))
    # bisect_left puts equality into the lower bin: (previous,cutpoint].
    state_records = tuple((bisect_left(bw_cutpoints, bw), int(age > age_cutpoint), int(s))
                          for bw, age, s in zip(birth_weight, mother_age, sex))
    immutable_ids = tuple(sorted(set(state_records)))
    lookup = {state: i for i, state in enumerate(immutable_ids)}
    record_states = tuple(lookup[state] for state in state_records)
    state_counts = tuple(record_states.count(i) for i in range(len(immutable_ids)))
    masses = tuple(F(n, len(rows)) for n in state_counts)
    propensities = tuple(F(8 + 3*bw + 2*age + s, 32) for bw, age, s in immutable_ids)
    profiles = {
        'low_heterogeneity': tuple(F(12 + bw + age + s, 32) for bw, age, s in immutable_ids),
        'high_heterogeneity': tuple(F(1,8) if (bw + age + s) % 2 == 0 else F(7,8)
                                    for bw, age, s in immutable_ids),
    }
    targets = {name: sum((p*r for p, r in zip(masses, profile)), F())
               for name, profile in profiles.items()}
    construction = {
        'benchmark_id': BENCHMARK_ID,
        'reference_population': 'All 747 records in the verified public IHDP covariate file',
        'reference_population_size': len(rows),
        'source_sha256': digest,
        'source_bytes': len(payload),
        'sources': {'covariate_csv': SOURCE_CSV, 'covariate_loader': SOURCE_LOADER,
                    'variable_definitions': SOURCE_VARIABLES},
        'source_verification': {
            'public_csv_byte_identical': True,
            'verified_by': 'Study launch owner, before benchmark evaluation',
            'public_csv_bytes': 161908,
            'loader_definition': 'CEVAE datasets.py defines x = data[:, 5:]',
            'variable_order': 'vdorie/npci data.R: bw,b.head,preterm,birth.o,nnhealth,momage,sex,...',
        },
        'covariates': {
            'birth_weight': {'x_coordinate_one_based': 1, 'csv_column_one_based': 6,
                             'definition': 'Infant birth weight; standardized continuous covariate in this file'},
            'mother_age': {'x_coordinate_one_based': 6, 'csv_column_one_based': 11,
                           'definition': "Mother's age; standardized continuous covariate in this file"},
            'infant_sex': {'x_coordinate_one_based': 7, 'csv_column_one_based': 12,
                           'definition': 'Binary infant sex, preserving the source 0/1 coding'},
        },
        'state_definition': '(birth-weight quartile, mother-age above median, infant sex)',
        'selection_rationale': 'Small interpretable baseline covariates; chosen before assignment or outcome study draws',
        'quantile_rule': 'Exact linear quantiles at (N-1)*q of full-population decimal coordinates',
        'birth_weight_cutpoints': bw_cutpoints,
        'mother_age_cutpoint': age_cutpoint,
        'boundary_ties': {'birth_weight': tuple(sum(x == c for x in birth_weight) for c in bw_cutpoints),
                          'mother_age': sum(x == age_cutpoint for x in mother_age)},
        'tie_rule': 'Equality goes into the lower bin; mother-age bit is strictly above the median',
        'state_id_rule': 'Lexicographically sorted occupied state tuples, fixed before any draw',
        'immutable_ids': immutable_ids,
        'state_counts': state_counts,
        'support_rule': 'Positive occupied states in the complete reference population, including any absent from a cohort',
        'mass_rule': 'Exact state count / 747; no rounding or residual adaptation',
        'sampling_rule': 'Uniform reference-record index with replacement, then its immutable state',
        'assignment_law': 'e(S)=(8+3*birth_weight_bin+2*mother_age_bit+sex)/32',
        'potential_outcome_laws': {
            'low_heterogeneity': 'P(Y(1)=1|S)=(12+birth_weight_bin+mother_age_bit+sex)/32',
            'high_heterogeneity': 'P(Y(1)=1|S)=1/8 for even coordinate sum and 7/8 for odd sum',
            'control': 'P(Y(0)=1|S)=1/2',
            'conditional_independence': 'A, Y(1), and Y(0) independent conditional on S',
        },
        'sensitivity': {'conditioning_variable': 'The finite state S defined above',
                        'prespecified_G': F(2), 'actual_assignment_odds_sensitivity': F(1),
                        'interpretation': 'Known benchmark law belongs to the G=2 class conditional on S'},
        'target': 'E[Y(1)] in this declared 747-record benchmark population and generated outcome law',
        'population_targets': targets,
        'support_qualification': 'No claim that these are the support or masses of an unknown real-world population',
        'outcome_data_usage': 'Original treatments and outcomes in columns 1-5 are ignored',
    }
    law = {
        'id': BENCHMARK_ID, 'K': len(immutable_ids), 'p': masses, 'e': propensities,
        'immutable_ids': immutable_ids, 'convention': CONVENTION,
        'source_identity': f'IHDP747 reference covariates sha256:{digest}',
        'exact_input_convention': 'Direct exact empirical rational masses counts/747; dyadic generated propensities',
        'legacy_convention_qualification': 'Inherited validator compatibility marker only; no binary mass conversion is used',
        'residual_applied_once': False,
        'qualification': 'Known complete finite reference population; generated conditional assignment/outcome laws',
    }
    validate_law(law)
    return {'law': law, 'record_states': record_states, 'profiles': profiles,
            'construction': construction, 'targets': targets}


def stream_record(law_index, replication, preflight=False,
                  master_seed=STUDY_SEED, development_seed=DEVELOPMENT_SEED):
    """Six disjoint component keys, independent cohorts and development draws."""
    if any(isinstance(value, bool) or int(value) != value or value < 0
           for value in (law_index, replication, master_seed, development_seed)):
        raise ValueError('Stream coordinates must be nonnegative integers')
    master = int(development_seed if preflight else master_seed)
    return {name: [master, int(law_index), int(replication), component]
            for component, name in enumerate(COMPONENTS)}


def _reference_states(law, record_states):
    states = np.asarray(record_states)
    if (states.ndim != 1 or not len(states) or states.dtype.kind not in 'iu'
            or np.any(states < 0) or np.any(states >= law['K'])):
        raise ValueError('Reference-record states require valid immutable support indices')
    observed = np.bincount(states.astype(np.int64), minlength=law['K'])
    if tuple(F(int(n), len(states)) for n in observed) != tuple(law['p']):
        raise ValueError('Reference-record counts disagree with the declared exact masses')
    return states


def _bernoulli_by_state(words, probabilities, states):
    probabilities = tuple(map(F, probabilities))
    cuts = np.asarray([min(2**64-1, threshold(p)) for p in probabilities], dtype=np.uint64)
    answer = (words < cuts[states]).astype(np.uint8)
    certain = np.asarray([p == 1 for p in probabilities], dtype=bool)
    answer[certain[states]] = 1
    return answer


def sample_block(law, profiles, law_index, replication, preflight=False,
                 record_states=None, master_seed=STUDY_SEED, development_seed=DEVELOPMENT_SEED):
    """Independent calibration/outcome cohorts; shared paired profile uniforms.

    Outcome observations are stored at the maximum size 256K; smaller cohorts
    are its prefixes. Every capacity and comparison method reuses this block.
    Profile rows follow the mapping's declared insertion order.
    """
    validate_law(law)
    profiles = {name: tuple(map(F, values)) for name, values in profiles.items()}
    K = law['K']
    if not profiles or any(len(values) != K for values in profiles.values()):
        raise ValueError('Each prespecified outcome profile must cover the declared support')
    for values in profiles.values():
        for value in values:
            threshold(value)
    streams = stream_record(law_index, replication, preflight, master_seed, development_seed)

    def words(component, n):
        return np.random.PCG64(np.random.SeedSequence(streams[component])).random_raw(n)

    reference = None if record_states is None else _reference_states(law, record_states)

    def states(component, n):
        if reference is None:
            return categorical(words(component, n), law['p'])
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(streams[component])))
        indices = rng.integers(0, len(reference), size=n, dtype=np.int64)
        return np.asarray(reference[indices], dtype=np.uint8 if K <= 256 else np.uint32)

    ncal, mmax = 4000*K, 256*K
    calS = states('calibration_states', ncal)
    calA = _bernoulli_by_state(words('calibration_actions', ncal), law['e'], calS)
    S = states('outcome_states', mmax)
    A = _bernoulli_by_state(words('outcome_actions', mmax), law['e'], S)
    uy = words('treated_outcome_uniforms', mmax)
    u0 = words('control_outcome_uniforms', mmax)
    control = (u0 < np.uint64(2**63)).astype(np.uint8)
    Y = [np.where(A == 1, _bernoulli_by_state(uy, values, S), control).astype(np.uint8)
         for values in profiles.values()]
    return {'cal_S': calS, 'cal_A': calA, 'out_S': S, 'out_A': A,
            'out_Y': np.asarray(Y, dtype=np.uint8), 'treated_uniforms': uy,
            'control_uniforms': u0}, streams


def verify_observations(observations, law, profiles):
    """Check shapes, preserved support, and exact profile reconstruction."""
    K = law['K']
    if (len(observations['cal_S']) != 4000*K
            or len(observations['out_S']) != 256*K
            or observations['out_Y'].shape != (len(profiles), 256*K)):
        raise AssertionError('Stored cohort/profile sizes differ from the frozen design')
    counts(observations['cal_S'], observations['cal_A'], K=K)
    S, A = observations['out_S'], observations['out_A']
    for j, values in enumerate(profiles.values()):
        expected = np.where(A == 1, _bernoulli_by_state(observations['treated_uniforms'], values, S),
                            observations['control_uniforms'] < np.uint64(2**63)).astype(np.uint8)
        if not np.array_equal(expected, observations['out_Y'][j]):
            raise AssertionError('Stored outcomes differ from prespecified shared-uniform draws')
        counts(S, A, observations['out_Y'][j], K=K)
    return True
