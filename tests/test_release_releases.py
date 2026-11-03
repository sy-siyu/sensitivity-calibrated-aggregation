"""Bounded fresh-generation and orchestration checks; no historical data input."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sca_reproduction import releases


def benchmark_fixture():
    law = releases.controlled_law(4, 0)
    law['id'] = 'public_covariate_fixture'
    profiles = {'low_heterogeneity': (F(1, 4),) * 4,
                'high_heterogeneity': (F(1, 8), F(7, 8), F(1, 8), F(7, 8))}
    return {'law': law, 'record_states': (0, 1, 2, 3), 'profiles': profiles,
            'construction': {'test_fixture_only': True},
            'targets': {key: sum(p * r for p, r in zip(law['p'], values))
                        for key, values in profiles.items()}}


class FreshStudies(unittest.TestCase):
    def test_generated_law_tree_and_integer_stream_conventions(self):
        from experiments.jmlr_computation_v1 import extended_law
        from experiments.jmlr_outcome_inference_v1_sampling import stream_record
        from experiments.jmlr_release_capacity_v1_sampling import stream_record as release_streams
        for K in (16, 64, 256):
            for index in (0, 1):
                law = releases.controlled_law(K, index)
                reference = extended_law(K, index)
                self.assertEqual((law['p'], law['e']), (reference['p'], reference['e']))
                self.assertEqual(sum(law['p']), 1)
            tree = releases.fixed_trees(K)
            self.assertEqual(tree[0], list(range(K)))
            self.assertEqual(sorted(tree[1]), list(range(K)))
        self.assertEqual(stream_record(2, 7)['outcome_actions'], [20261014, 2, 7, 3])
        self.assertEqual(release_streams(1, 7)['outcome_actions'], [2026101502, 1, 7, 3])
        self.assertNotEqual(release_streams(1, 7), release_streams(1, 7, True))

    def test_protocols_generated_before_evaluation_and_no_private_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('experiments.jmlr_release_capacity_v1_sampling.build_benchmark',
                       return_value=benchmark_fixture()) as build:
                protocol = releases.prepare_release(root / 'release', root / 'public.csv',
                                                     {'replications': 1, 'controlled_support': 4})
            build.assert_called_once_with(root / 'public.csv')
            self.assertTrue(protocol['frozen_before_any_evaluation'])
            self.assertEqual(protocol['committed_blocks'], 3)
            self.assertEqual(protocol['committed_condition_rows'], 96)
            self.assertFalse((root / 'release/panel').exists())
            self.assertNotIn('law_source', protocol['configuration'])
            self.assertNotIn('tree_source', protocol['configuration'])
            self.assertEqual(protocol['tree_orders'][2], releases.fixed_trees(4))
            with self.assertRaises(FileExistsError):
                releases.prepare_outcome(root / 'release')
            outcome = releases.prepare_outcome(root / 'outcome', {'replications': 1})
            self.assertEqual([x['K'] for x in outcome['laws']], [16, 64, 64])
            self.assertEqual(len(outcome['random_streams']), 3)
            precision = releases.prepare_precision(root / 'precision', root / 'release')
            self.assertEqual(precision['committed_selection_rows'], 15)
            self.assertEqual(precision['committed_analysis_instances'], 80)
            self.assertEqual(precision['committed_method_records'], 320)
            self.assertEqual(precision['configuration']['caps'], ['1', '11/10', '5/4', '3/2', '2'])

    def test_scientific_constants_not_silently_overridden(self):
        for config in ({'G': '3'}, {'root_policy': {'max_trials': 1}}, {'replications': True},
                       {'alpha_cal': '1/20'}, {'outcome_m_per_K': [16]}):
            with self.assertRaises(ValueError):
                releases.configuration('release', config)

    def test_fresh_k64_bridge_matches_direct_sampling(self):
        from experiments.jmlr_outcome_inference_v1_core import outcome_profiles
        from experiments.jmlr_outcome_inference_v1_sampling import counts
        from experiments.jmlr_release_capacity_v1_sampling import sample_block
        law = releases.controlled_law(64, 1)
        profiles = outcome_profiles(law)
        observations, streams = sample_block(law, profiles, 1, 0)
        cal, outcome, actual = releases.release_bridge_counts(1, 0)
        self.assertEqual(actual, streams)
        self.assertEqual((cal['N'], cal['T']), counts(observations['cal_S'], observations['cal_A'], K=64))
        aligned = list(profiles).index('aligned')
        self.assertEqual((outcome['N'], outcome['T'], outcome['V']), counts(
            observations['out_S'], observations['out_A'], observations['out_Y'][aligned], K=64))
        dev, _, development = releases.release_bridge_counts(1, 0, True)
        self.assertNotEqual(development, streams)
        self.assertNotEqual(dev, cal)

    def test_computation_inputs_need_no_release_directory_or_saved_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'computation'
            protocol = releases.prepare_computation(root, {
                'replications': 1, 'support_sizes': [16, 64],
                'preflight': {'support_sizes': [], 'replications_per_law': 1},
            })
            self.assertEqual(protocol['primary_cases'], 4)
            self.assertEqual(protocol['preflight_cases'], 0)
            self.assertEqual(releases.load(root / 'bridge_expected.json'), {})
            k64 = releases.load(root / 'inputs/primary/0_K64_r000.json.gz')
            cal, outcome, _ = releases.release_bridge_counts(0, 0)
            self.assertEqual((tuple(k64['N']), tuple(k64['T'])), (cal['N'], cal['T']))
            self.assertEqual({key: tuple(value) for key, value in k64['outcome'].items()}, outcome)
            self.assertNotIn('bridge_source', protocol['configuration'])
            self.assertEqual(protocol['configuration']['root_policy']['max_trials'], 2_000_000)

    def test_bounded_release_then_precision_use_new_evidence_and_resume(self):
        from experiments import jmlr_release_capacity_v1 as implementation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = implementation.ROOT, implementation.CONFIG
            with patch('experiments.jmlr_release_capacity_v1_sampling.build_benchmark', return_value=benchmark_fixture()):
                releases.prepare_release(root / 'release', root / 'public.csv',
                                         {'replications': 1, 'controlled_support': 4})
            answer = releases.run_release(root / 'release', blocks=[(0, 0)])
            self.assertEqual(answer['completed_blocks'], 1)
            self.assertEqual(answer['status'], 'panel_incomplete')
            self.assertEqual((implementation.ROOT, implementation.CONFIG), before)
            block = releases.load(root / 'release/panel/blocks/0_000.json.gz')
            self.assertEqual(sum(block['N']), 16000)
            self.assertEqual(len(block['outcomes']), 6)
            self.assertTrue(all(row.get('aggregate_only_equal', True) for row in block['rows']))
            again = releases.run_release(root / 'release', blocks=[(0, 0)])
            self.assertEqual(again['new_receipts'], [])
            releases.prepare_precision(root / 'precision', root / 'release', {'capacity_limits': [1, 1, 1]})
            result = releases.run_precision(root / 'precision', blocks=[(0, 0)])
            self.assertEqual(result['completed_blocks'], 1)
            precision = releases.load(root / 'precision/panel/blocks/0_000.json.gz')
            self.assertEqual(len(precision['selection_rows']), 5)
            self.assertEqual(len(precision['analysis_rows']), 30)
            self.assertTrue(all(row['count_optimal'] for row in precision['selection_rows']))

    def test_bounded_fresh_outcome_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'outcome'
            releases.prepare_outcome(root, {'replications': 1, 'outcome_supports': [4, 4, 4]})
            answer = releases.run_outcome(root, blocks=[(0, 0)])
            self.assertEqual(answer['completed_blocks'], 1)
            self.assertEqual(answer['status'], 'panel_incomplete')
            block = releases.load(root / 'blocks/0_000.json.gz')
            self.assertEqual(len(block['evaluations']), 12)
            replay = releases.load(root / 'replay_checks.json')
            self.assertEqual(replay['selection_replays'], 1)
            self.assertEqual(replay['outcome_replays'], 12)
            self.assertEqual(replay['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
