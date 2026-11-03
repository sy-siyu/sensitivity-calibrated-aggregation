"""Bounded fresh-input checks; no retained paper outputs are test fixtures."""
from copy import deepcopy
from fractions import Fraction as F
import math
from pathlib import Path
import tempfile
import unittest

from sca_reproduction import analyses
from sca_reproduction import assignment


class DesignAndScope(unittest.TestCase):
    def test_one_default_and_descriptive_ablation_labels(self):
        self.assertTrue(analyses.EFFICIENCY_LABELS['original'].startswith('SCA'))
        self.assertFalse(analyses.EFFICIENCY_LABELS['allocation_control'].startswith('SCA'))
        self.assertFalse(analyses.EFFICIENCY_LABELS['pooled_alternative'].startswith('SCA'))

    def test_default_grids_and_component_smoke_override(self):
        cfg=analyses._configuration('efficiency')
        self.assertEqual(cfg['recovery']['replications'],300)
        self.assertEqual(tuple(cfg['recovery']['sample_sizes']),analyses.RECOVERY_SIZES)
        self.assertEqual(cfg['root_policy']['max_trials'],1000000)
        self.assertEqual(cfg['recovery_root_policy']['max_trials'],4000000)
        small=analyses._configuration('efficiency',{'efficiency':{'recovery':{'replications':1,'sample_sizes':[128]}}})
        self.assertEqual(small['recovery']['sample_sizes'],[128])
        self.assertEqual(small['recovery']['master_seed'],cfg['recovery']['master_seed'])
        self.assertEqual(small['recovery']['development_seed'],cfg['recovery']['development_seed'])
        self.assertEqual(small['root_policy'],cfg['root_policy'])
        with self.assertRaises(ValueError):
            analyses._configuration('split',{'caps':['1','3/2','2']})
        with self.assertRaises(ValueError):
            analyses._limit([1],True)

    def test_theoretical_reference_is_recomputed_from_formula(self):
        from experiments.jmlr_efficiency_gap_v1_core import manuscript_bound
        reference=analyses.theoretical_reference();n=reference['direct_diameter_threshold']
        bound=manuscript_bound(n,64,F(1,20),F(1,64),F(1,4),beta=F(1,20))
        before=manuscript_bound(n-1,64,F(1,20),F(1,64),F(1,4),beta=F(1,20))
        self.assertLessEqual(bound['b_n'],math.log(1.5))
        self.assertGreater(before['b_n'],math.log(1.5))
        self.assertGreater(reference['necessary_lower_bound'],0)
        self.assertEqual(reference['sufficient_reference_n'],3000000)
        self.assertLessEqual(reference['sufficient_reference_bound']['b_n'],math.log(1.5))


class FreshDownstreamAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One declared K16 development dataset is enough to exercise the
        # current development and evaluation paths. No full panel.
        cls.design=assignment.build_design()
        cls.input=assignment.draw_case(0,0,0,cls.design)
        cls.full,cls.split=assignment.evaluate_case(cls.input,cls.design)

    def source_workspace(self,directory):
        workspace=Path(directory);root=workspace/'assignment'
        analyses._dump(root/'design.json',self.design)
        analyses._dump(root/'exact_laws.json',{'large':self.design['laws']})
        metadata=self.input['metadata'];condition=metadata['id']+'_r'+str(metadata['r'])
        full=root/'evidence/large'/condition/'rep000.json.gz'
        split=root/'splits/evidence/development'/condition/'rep000.json.gz'
        analyses._dump(full,self.full);analyses._dump(split,self.split)
        return workspace,full,split

    def test_efficiency_consumes_only_fresh_generated_source_and_resumes_prefix(self):
        with tempfile.TemporaryDirectory(prefix='sca-fresh-efficiency-') as directory:
            workspace,full,split=self.source_workspace(directory)
            original=(full.read_bytes(),split.read_bytes())
            result=analyses.run_efficiency(workspace,limit=1)
            self.assertEqual(result['new_records'],1)
            self.assertEqual(result['status'],'incomplete')
            paths=list((workspace/'efficiency/panel/historical').glob('*.json.gz'))
            record=analyses._load(paths[0])
            self.assertEqual(len(record['rows']),9)
            self.assertEqual({r['procedure'] for r in record['rows']},set(analyses.EFFICIENCY_LABELS))
            self.assertEqual(analyses.run_efficiency(workspace,limit=1)['new_records'],0)
            self.assertEqual((full.read_bytes(),split.read_bytes()),original)
            frozen=analyses._load(workspace/'efficiency/frozen_protocol.json')
            self.assertTrue(all(not Path(path).is_absolute() for path in frozen['historical_paths']))
            self.assertNotIn('JMLR_manuscript',str(frozen))
            partial=analyses.report(workspace,'efficiency')
            self.assertEqual(partial['status'],'incomplete')
            self.assertEqual(partial['rows'],9)
            with self.assertRaises(FileNotFoundError):
                analyses.run_efficiency(workspace)


    def test_current_evaluation_needs_no_old_fraction_choice_or_evidence(self):
        from unittest.mock import patch
        from experiments.jmlr_empirical_relevance_v1_core import evaluate_selected_ltt_with_ablation
        from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy
        source = assignment.draw_case(0, 0, 10, self.design)
        with patch.object(assignment.split_core, 'evaluate_split',
                          side_effect=AssertionError('Assignment must not evaluate the old split')):
            full, base = assignment.evaluate_case(source, self.design)
        self.assertEqual(base['splits'], [])
        self.assertEqual(full['baseline']['splits'], [])
        current = evaluate_selected_ltt_with_ablation(
            source['N'], source['T'], 0, 0, 10, source['immutable_ids'],
            self.design['tree_orders']['16'], RootPolicy().__dict__,
            full, self.design['laws'][0])
        self.assertEqual({row['procedure'] for row in current['rows']}, set('ABCDEF'))
        self.assertEqual(len(current['rows']), 18)

    def test_recovery_is_standalone_exact_count_generation_not_an_archive(self):
        with tempfile.TemporaryDirectory(prefix='sca-fresh-recovery-') as directory:
            workspace=Path(directory)
            config={'efficiency':{'recovery':{'replications':1,'sample_sizes':[128]}}}
            result=analyses.run_recovery(workspace,limit=1,config=config)
            self.assertEqual(result['new_records'],1)
            self.assertFalse((workspace/'assignment').exists())
            output=workspace/'efficiency/panel/recovery/000.json.gz'
            record=analyses._load(output);before=output.read_bytes()
            self.assertEqual(len(record['rows']),12)
            self.assertEqual({r['n'] for r in record['rows']},{128})
            self.assertEqual(analyses.run_recovery(workspace,limit=1)['new_records'],0)
            self.assertEqual(output.read_bytes(),before)
            self.assertEqual(analyses.report(workspace,'recovery')['status'],'incomplete')
            changed=deepcopy(config);changed['efficiency']['recovery']['sample_sizes']=[256]
            with self.assertRaises(ValueError):
                analyses.run_recovery(workspace,limit=1,config=changed)


if __name__=='__main__':
    unittest.main()
