"""Bounded public SCA checks using fresh counts and no research archives.

Synthetic rational tables below test objective/tie semantics. Numerical checks
use the unchanged public root policy, full supplied calibration counts, and
the established exact certification/search routines.
"""
from copy import deepcopy
from fractions import Fraction as F
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sca import SCA, analyze_release
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import exact_order, select_families
from experiments.jmlr_release_capacity_v1_search import candidate_cells, capacity_search
from experiments.jmlr_release_capacity_v1_interface import CONFIDENCE, release_bytes
from experiments.jmlr_outcome_inference_v1_core import (
    ALPHA_CAL, ALPHA_Z, confidence_enclosure, outcome_family, project_assignment,
)
from experiments.jmlr_statistical_baselines_v1_reporting import paired, rate


POLICY = RootPolicy(max_trials=2_000_000)
ROOT = Path(__file__).resolve().parents[1]


def constituents(order, tree_orders):
    """Independent enumeration; no selection recurrence is used here."""
    def tree_partitions(tree, cell):
        yield (cell,)
        children = tree[cell]
        if children is not None:
            left, right = children
            for a, b in itertools.product(tree_partitions(tree, left),
                                          tree_partitions(tree, right)):
                yield exact.canonical_partition(a + b)

    contiguous = []
    for bits in itertools.product((False, True), repeat=len(order)-1):
        cuts = (0,) + tuple(i+1 for i, cut in enumerate(bits) if cut) + (len(order),)
        contiguous.append(exact.canonical_partition(tuple(order[a:b])
                          for a, b in zip(cuts, cuts[1:])))
    families = {'contiguous': tuple(contiguous)}
    for i, tree_order in enumerate(tree_orders):
        tree = exact.build_tree(tree_order)
        families['tree-'+str(i)] = tuple(tree_partitions(tree, max(tree, key=len)))
    return families


def completed_value(partition, records):
    values = [records[c]['coupled'] if records[c]['status'] == 'full' else None
              for c in partition]
    return None if any(v is None for v in values) else max(values)


class PublicSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N = (4000,)*4
        cls.T = (1400, 1800, 2200, 2600)
        cls.ids = ('state-d', 'state-b', 'state-c', 'state-a')
        cls.tree_orders = ((0, 1, 2, 3), (2, 0, 3, 1))
        cls.order = exact_order(cls.N, cls.T, cls.ids)
        cls.trees = tuple(exact.build_tree(o) for o in cls.tree_orders)
        cls.region = exact.confidence_region(cls.N, cls.T, ALPHA_CAL,
                                             verifier=RootVerifier(POLICY))
        cls.cells = candidate_cells(cls.order, cls.tree_orders)
        cls.full_records = {c: cell_certificate(cls.region, c) for c in cls.cells}
        cls.families = constituents(cls.order, cls.tree_orders)

    def fit(self, **kwargs):
        return SCA(root_policy=POLICY).fit(self.N, self.T, immutable_ids=self.ids,
                                         tree_orders=self.tree_orders, **kwargs)

    def test_default_is_full_sample_coupled_combined_and_matches_direct_selection(self):
        # Use threshold-pruned records in the reference as the public API does.
        records = {c: cell_certificate(self.region, c, max_cap=F(3, 2)) for c in self.cells}
        values = {c: r['coupled'] if r['status'] == 'full' else None
                  for c, r in records.items()}
        expected = select_families(self.order, self.trees, values, F(3, 2))['combined']
        result = self.fit()
        self.assertEqual((result.partition, result.groups, result.certificate),
                         (expected['partition'], expected['cells'], expected['certificate']))
        self.assertEqual(result.objective, 'minimum_groups_at_fixed_cap')
        self.assertEqual(result.status, 'finite')
        self.assertTrue(result.optimization_completed)
        self.assertTrue(result.metadata['count_optimal'])
        self.assertTrue(result.metadata['deterministic_tie_complete'])
        self.assertEqual(result.metadata['alpha'], ALPHA_CAL)
        self.assertEqual(result.metadata['calibration_size'], sum(self.N))
        self.assertTrue(result.metadata['full_assignment_sample'])
        self.assertEqual(result.metadata['certificate_construction'], 'Coupled Clopper–Pearson')
        self.assertEqual(result.metadata['root_policy'], POLICY.__dict__)
        self.assertEqual(SCA().root_policy, POLICY)
        self.assertEqual(result.metadata['order'], self.order)
        self.assertEqual(result.metadata['tree_orders'], self.tree_orders)
        self.assertEqual(set(result._records), set(self.cells))
        self.assertTrue(any(result.partition in partitions for partitions in self.families.values()))
        feasible = [p for partitions in self.families.values() for p in partitions
                    if completed_value(p, self.full_records) is not None
                    and completed_value(p, self.full_records) <= F(3, 2)]
        self.assertEqual(result.groups, min(map(len, feasible)))
        self.assertTrue(result.admissible())
        self.assertFalse(result.admissible(aggregate_budget='1'))

    def test_every_capacity_matches_direct_and_exhaustive_family_values(self):
        expected = capacity_search(self.order, self.tree_orders, self.full_records)
        certificates = []
        for B in range(1, 5):
            with self.subTest(capacity=B):
                result = self.fit(capacity=B)
                choice = expected['by_capacity'][B]
                self.assertEqual((result.partition, result.groups, result.certificate),
                                 (choice['partition'], choice['cells'], choice['certificate']))
                self.assertEqual(result.objective, 'minimum_certificate_at_fixed_capacity')
                self.assertEqual(result.status, 'finite')
                self.assertTrue(result.optimization_completed)
                self.assertLessEqual(result.groups, B)
                self.assertEqual(result.metadata['capacity'], B)
                self.assertTrue(all(r['status'] != 'rejected' for r in result._records.values()))
                for family, partitions in self.families.items():
                    feasible = [completed_value(p, self.full_records) for p in partitions if len(p) <= B]
                    finite = [v for v in feasible if v is not None]
                    table = expected['families'][family]
                    actual = [table[k]['certificate'] for k in table if k <= B
                              and table[k]['status'] == 'finite']
                    self.assertEqual(min(actual), min(finite))
                all_values = [completed_value(p, self.full_records)
                              for ps in self.families.values() for p in ps if len(p) <= B]
                self.assertEqual(result.certificate, min(v for v in all_values if v is not None))
                certificates.append(result.certificate)
        self.assertTrue(all(a >= b for a, b in zip(certificates, certificates[1:])))
        self.assertEqual(certificates[-1], 1)
        self.assertEqual(self.fit(capacity=4).partition, ((0,), (1,), (2,), (3,)))

    def test_objectives_are_distinct_and_preserve_existing_local_ties(self):
        # A rational fixture makes the two objectives choose different outputs.
        # These are not empirical certification results.
        def certificate(region, cell, max_cap=None):
            value = (F(1) if len(cell) == 1 else
                     {(0, 1): F(6, 5), (1, 2): F(7, 5)}.get(cell, F(3)))
            return {'cell': cell, 'status': 'full', 'coupled': value}

        options = {'immutable_ids': tuple(range(4)),
                   'tree_orders': ((0, 2, 1, 3), (0, 2, 1, 3))}
        with patch('sca.api.cell_certificate', side_effect=certificate):
            threshold = SCA(root_policy=POLICY).fit(self.N, self.T, cap='3/2', **options)
            capacity = SCA(root_policy=POLICY).fit(self.N, self.T, capacity=3, **options)
        self.assertEqual(threshold.groups, 3)
        self.assertEqual(threshold.partition, ((0,), (1, 2), (3,)))
        self.assertEqual(threshold.certificate, F(7, 5))
        self.assertEqual(capacity.partition, ((0, 1), (2,), (3,)))
        self.assertEqual(capacity.certificate, F(6, 5))

    def test_k8_estimated_propensity_ties_use_immutable_ids_and_complete_support(self):
        N, T = (4000,)*8, (2000,)*8
        ids = (7, 1, 6, 2, 5, 3, 4, 0)
        trees = (tuple(range(8)), (7, 5, 3, 1, 6, 4, 2, 0))
        result = SCA(root_policy=POLICY).fit(N, T, immutable_ids=ids, tree_orders=trees)
        order = tuple(sorted(range(8), key=ids.__getitem__))
        self.assertEqual(result.metadata['order'], order)
        region = exact.confidence_region(N, T, ALPHA_CAL, verifier=RootVerifier(POLICY))
        records = {c: cell_certificate(region, c, max_cap=F(3, 2))
                   for c in candidate_cells(order, trees)}
        values = {c: r['coupled'] if r['status'] == 'full' else None for c, r in records.items()}
        direct = select_families(order, tuple(exact.build_tree(o) for o in trees), values, F(3, 2))['combined']
        self.assertEqual((result.partition, result.certificate), (direct['partition'], direct['certificate']))
        self.assertEqual(sorted(i for c in result.partition for i in c), list(range(8)))

    def test_mathematical_infinity_and_unresolved_computation_remain_distinct(self):
        def record(status):
            return lambda region, cell, max_cap=None: (
                {'cell': cell, 'status': 'full', 'coupled': F(1)} if len(cell) == 1 else
                {'cell': cell, 'status': status, 'coupled': None, 'lower_bound': F(1)})
        with patch('sca.api.cell_certificate', side_effect=record('boundary')):
            infinite = self.fit(capacity=1)
        self.assertEqual(infinite.status, 'infinite')
        self.assertTrue(infinite.optimization_completed)
        self.assertIsNone(infinite.certificate)
        self.assertFalse(infinite.admissible())
        with self.assertRaises(ValueError): infinite.release(self.N, self.T, (0,)*4)
        with patch('sca.api.cell_certificate', side_effect=record('failure')):
            unknown = self.fit(capacity=1)
            safe = self.fit()
        self.assertEqual(unknown.status, 'unresolved')
        self.assertFalse(unknown.optimization_completed)
        self.assertIsNone(unknown.partition)
        self.assertFalse(unknown.admissible())
        self.assertEqual(safe.status, 'unresolved')
        self.assertEqual(safe.certificate, 1)
        self.assertTrue(safe.metadata['identity_fallback'])
        self.assertFalse(safe.admissible())

    def test_assignment_only_contract_and_input_validation(self):
        self.assertEqual(tuple(inspect.signature(SCA.fit).parameters),
                         ('self', 'N', 'T', 'cap', 'capacity', 'immutable_ids', 'tree_orders'))
        with self.assertRaises(ValueError): self.fit(cap='3/2', capacity=2)
        for B in (0, 5, True, 1.5):
            with self.subTest(B=B), self.assertRaises(ValueError): self.fit(capacity=B)
        with self.assertRaises(ValueError): SCA().fit((1, 0), (2, 0))
        with self.assertRaises(ValueError): SCA().fit((True, 2), (0, 1))
        with self.assertRaises(ValueError): SCA().fit(self.N, self.T, immutable_ids=(0,)*4)
        with self.assertRaises(ValueError): SCA().fit(self.N, self.T, tree_orders=((0, 1, 2, 3),))
        with self.assertRaises(ValueError): SCA(alpha=0)
        with self.assertRaises(TypeError): SCA(root_policy=POLICY.__dict__)


class PublicAggregateAnalyst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal_N = (0, 4000, 4000, 8000)
        cls.cal_T = (0, 1600, 2400, 4000)
        cls.result = SCA(root_policy=POLICY).fit(cls.cal_N, cls.cal_T, cap=1)
        cls.N, cls.T, cls.V = (0, 10, 20, 30), (0, 0, 10, 15), (0, 0, 4, 8)
        cls.artifact = cls.result.release(cls.N, cls.T, cls.V)
        cls.analysis = analyze_release(cls.artifact)

    def test_declared_unobserved_state_is_in_identity_and_release(self):
        result = self.result
        self.assertEqual(result.partition, ((0,), (1,), (2,), (3,)))
        self.assertEqual(result.certificate, 1)
        self.assertTrue(result.optimization_completed)
        self.assertEqual(result._region['K'], 4)
        self.assertIn((0,), result._records)
        self.assertEqual(result._records[(0,)]['coupled'], 1)
        self.assertEqual(len(self.artifact['groups']), 4)
        self.assertEqual(self.artifact['groups'][0]['counts'],
                         {'size': 0, 'treated': 0, 'treated_successes': 0})
        self.assertEqual(self.artifact['groups'][1]['counts']['treated'], 0)
        self.assertEqual(self.analysis['group_count'], 4)
        self.assertEqual(self.analysis['cohort_size'], 60)

    def test_only_anonymous_aggregate_information_is_exported(self):
        self.assertEqual(set(self.artifact), {'version', 'groups', 'raw_budget', 'confidence', 'root_policy'})
        for j, group in enumerate(self.artifact['groups']):
            self.assertEqual(set(group), {'label', 'counts', 'mass_bounds',
                                         'propensity_bounds', 'distortion_bound'})
            self.assertEqual(group['label'], f'group_{j:04d}')
        self.assertEqual(self.artifact['confidence'], CONFIDENCE)
        self.assertEqual(self.analysis['payload_bytes'], len(release_bytes(self.artifact)))
        contaminated = deepcopy(self.artifact)
        contaminated['partition'] = self.result.partition
        with self.assertRaises(ValueError): analyze_release(contaminated)
        contaminated = deepcopy(self.artifact)
        contaminated['groups'][0]['raw_state_indices'] = [0]
        with self.assertRaises(ValueError): analyze_release(contaminated)

    def test_empty_treated_bounds_and_actual_certificate_match_provider_formulas(self):
        pbox, ebox, _ = project_assignment(self.result._region, self.result.partition)
        rb, roots = outcome_family(self.V, self.T, ALPHA_Z, RootVerifier(POLICY))
        self.assertEqual(rb[:2], ((F(0), F(1)),)*2)
        self.assertEqual([r['status'] for r in roots[:2]], ['structural', 'structural'])
        self.assertTrue(all(r['gamma'] == ALPHA_Z/4 for r in roots))
        self.assertEqual(self.analysis['root_fallbacks'], 0)
        self.assertEqual(self.analysis['certificate'], 1)
        self.assertEqual(self.analysis['transported_budget'], 2)
        self.assertEqual(self.analysis['cell_budgets'], (F(2),)*4)
        self.assertEqual(self.analysis['z_outcome_bounds'], rb)
        for method, budget in (('z_global', F(2)), ('z_cellwise', (F(2),)*4)):
            reference = confidence_enclosure(pbox, ebox, rb, budget)
            self.assertEqual(self.analysis['methods'][method], reference)
            self.assertFalse(reference['fallback'])
        self.assertEqual(self.analysis['methods']['z_global']['interval'],
                         self.analysis['methods']['z_cellwise']['interval'])

    def test_fresh_analyst_process_uses_release_as_its_only_data_input(self):
        with tempfile.TemporaryDirectory(prefix='sca_release_only_') as directory:
            path = Path(directory)/'release.json'
            path.write_text(json.dumps(self.artifact), encoding='utf-8')
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            completed = subprocess.run(
                [sys.executable, '-m', 'experiments.jmlr_release_capacity_v1_interface',
                 '--restrict-project-data', str(path)],
                cwd=directory, env=env, capture_output=True, text=True, check=True, timeout=30)
            fresh = json.loads(completed.stdout)
        self.assertEqual(fresh['certificate'], str(self.analysis['certificate']))
        self.assertEqual(fresh['transported_budget'], str(self.analysis['transported_budget']))
        for method in ('z_global', 'z_cellwise'):
            self.assertEqual(fresh['methods'][method]['interval'],
                             list(map(str, self.analysis['methods'][method]['interval'])))
        receipt = fresh['data_access_receipt']
        self.assertTrue(receipt['code_data_access_restricted'])
        self.assertEqual(receipt['noncode_project_access_attempts'], [])
        self.assertEqual(receipt['data_input_open_count'], 1)


class PublicMonteCarloSummaries(unittest.TestCase):
    def test_individual_binomial_boundaries_and_paired_differences_have_distinct_intervals(self):
        none, all_events = rate([False]*100), rate([True]*100)
        self.assertEqual((none['n'], none['successes'], none['rate'], none['CP95_lower']),
                         (100, 0, 0, 0))
        self.assertAlmostEqual(none['CP95_upper'], 1-math.pow(.025, 1/100), places=14)
        self.assertEqual((all_events['n'], all_events['successes'], all_events['rate'],
                          all_events['CP95_upper']), (100, 100, 1, 1))
        self.assertAlmostEqual(all_events['CP95_lower'], math.pow(.025, 1/100), places=14)
        differences = paired([-1, 1]*50)
        self.assertEqual(differences['n'], 100)
        self.assertEqual(differences['mean'], 0)
        self.assertGreater(differences['MCSE'], 0)
        self.assertLess(differences['Student95_lower'], 0)
        self.assertGreater(differences['Student95_upper'], 0)
        self.assertAlmostEqual(differences['Student95_lower'], -differences['Student95_upper'])
        self.assertNotIn('CP95_lower', differences)
        self.assertNotIn('Student95_lower', none)


if __name__ == '__main__':
    unittest.main()
