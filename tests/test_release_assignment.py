"""Portable bounded checks: no historical archives or embedded result tables."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sca_reproduction import assignment
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy,RootVerifier


class AssignmentGenerators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.design=assignment.build_design()

    def test_original_grid_indices_and_fixed_geometry_streams(self):
        design=self.design
        self.assertEqual(len(design['populations']),16)
        self.assertEqual([(p['K'],p['mass'],p['overlap'],p['geometry']) for p in design['populations']],
            [(K,mass,overlap,geometry) for K in (16,64) for mass in ('balanced','skewed')
             for overlap in ('moderate','weak') for geometry in ('ordered','permuted')])
        for K in assignment.SUPPORTS:
            for component,observed in ((101,design['tree_orders'][str(K)][1]),(102,design['population_permutations'][str(K)])):
                expected=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261008,component,K]))).permutation(K)
                self.assertEqual(observed,expected.tolist())
            self.assertEqual(design['tree_orders'][str(K)][0],list(range(K)))

    def test_exact_nominal_law_applies_residual_once_after_raw_geometry(self):
        for population,law in zip(self.design['populations'],self.design['laws']):
            raw=tuple(F.from_float(p) for p in population['p'])
            self.assertEqual(law['p'][:-1],raw[:-1])
            self.assertEqual(law['p'][-1],1-sum(raw[:-1]))
            self.assertEqual(sum(law['p']),1)
            self.assertTrue(all(p>0 for p in law['p']))
            self.assertEqual(law['e'],tuple(F.from_float(e) for e in population['e']))
            self.assertEqual(law['immutable_ids'],tuple(range(law['K'])))
            self.assertTrue(law['residual_applied_once'])
            self.assertEqual(law['last_entry_adjustment'],law['p'][-1]-raw[-1])
            self.assertNotIn('/Users/',law['source_identity'])

    def test_original_multinomial_then_binomial_rng_sequence(self):
        for pi,ri,rep in ((0,0,0),(7,2,9),(8,2,10),(15,0,99)):
            population=self.design['populations'][pi]
            seed=[20261008,300,population['K'],population['mass_index'],population['overlap_index'],
                  population['geometry_index'],ri,rep]
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
            N=tuple(map(int,rng.multinomial(population['K']*assignment.R_VALUES[ri],population['p'])))
            T=tuple(map(int,rng.binomial(N,population['e'])))
            result=assignment.draw_case(pi,ri,rep,self.design)
            self.assertEqual((result['N'],result['T']),(N,T))
            self.assertEqual(result['assignment_seed'],seed)
            self.assertEqual(result['metadata']['population_index'],pi)
            self.assertEqual(result['phase'],'development' if rep<10 else 'evaluation')
            self.assertEqual(result,assignment.draw_case(pi,ri,rep,self.design))

    def test_original_nested_splits_conserve_counts_and_stream_identifiers(self):
        source=assignment.draw_case(15,0,10,self.design);N,T=source['N'],source['T'];splits=source['nested_splits']
        self.assertEqual(tuple(splits),('1/5','1/2'))
        for fraction,split in splits.items():
            self.assertEqual(split['seed'],[20261012,901,15,0,10])
            self.assertEqual(split['pilot_size'],sum(N)*F(fraction))
            self.assertEqual(tuple(a+b for a,b in zip(split['pilot_N'],split['heldout_N'])),N)
            self.assertEqual(tuple(a+b for a,b in zip(split['pilot_T'],split['heldout_T'])),T)
            self.assertEqual(len(split['pilot_N']),64)
            self.assertEqual(split['immutable_ids'],tuple(range(64)))
        self.assertTrue(all(a<=b for a,b in zip(splits['1/5']['pilot_N'],splits['1/2']['pilot_N'])))
        self.assertTrue(all(a<=b for a,b in zip(splits['1/5']['pilot_T'],splits['1/2']['pilot_T'])))

    def test_global_independent_development_precedes_evaluation(self):
        tasks=list(assignment.case_tasks())
        self.assertEqual(len(tasks),4800)
        self.assertTrue(all(rep<10 for _,_,rep in tasks[:480]))
        self.assertTrue(all(10<=rep<100 for _,_,rep in tasks[480:]))
        self.assertEqual(tasks[0],(0,0,0));self.assertEqual(tasks[480],(0,0,10))
        self.assertEqual(len(list(assignment.case_tasks('development'))),480)
        self.assertEqual(len(list(assignment.case_tasks('evaluation'))),4320)

    def test_invalid_identifiers_and_changed_design_are_rejected(self):
        for task in ((True,0,0),(0,3,0),(16,0,0),(0,0,100)):
            with self.subTest(task=task),self.assertRaises(ValueError):assignment.draw_case(*task,self.design)
        for changed in ({'pilot_fractions':['1/10','1/5','1/3']},{'root_policy':RootPolicy(max_trials=2_000_000).__dict__},
                        {'support_sizes':[16,32,64]},{'alpha':'1/40'}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):assignment._settings({'assignment':changed})


class BoundedFreshAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='sca_assignment_test_')
        cls.workspace=Path(cls.directory.name)
        cls.prepared=assignment.prepare(cls.workspace,smoke=True)
        cls.first=assignment.run(cls.workspace,limit=1)
        cls.root=cls.workspace/'assignment'
        cls.relative=Path('K16_balanced_moderate_ordered_r250/rep000.json.gz')
        cls.full_path=cls.root/'evidence/large'/cls.relative
        cls.split_path=cls.root/'splits/evidence/development'/cls.relative
        cls.full=assignment._read(cls.full_path);cls.split=assignment._read(cls.split_path)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_prepare_and_bounded_run_remain_explicitly_incomplete(self):
        self.assertEqual(self.prepared['prepared_cases'],1)
        self.assertFalse(self.prepared['complete']);self.assertFalse(self.first['complete'])
        self.assertEqual(self.first['completed_cases'],1)
        self.assertEqual(self.first['failures'],[])
        self.assertEqual(tuple(s['fraction'] for s in self.split['splits']),('1/5','1/2'))
        self.assertNotIn('frozen_fraction', self.full['baseline'])
        self.assertFalse((self.root/'fraction_selection.json').exists())
        self.assertFalse((self.root/'splits/evidence/evaluation').exists())
        self.assertTrue(self.full['baseline']['source_inference_recomputed'])
        self.assertFalse(self.full['historical_evidence_imported'])

    def test_fresh_full_core_has_complete_family_oracles_and_baselines(self):
        core=self.full['core'];K=core['N'].__len__()
        self.assertEqual(K,16);self.assertEqual(len(core['modes']),2)
        self.assertEqual(len(core['oracles']),9);self.assertEqual(len(core['true_order_oracles']),3)
        for mode in core['modes']:
            self.assertEqual(len(mode['labels']),18)
            records={tuple(r['cell']):r for r in mode['cells']}
            self.assertEqual(mode['region']['alpha'],F(1,20))
            for label in mode['labels']:
                partition=exact.validate_partition(label['partition'],K)
                self.assertEqual(label['count'],len(partition))
                self.assertEqual(label['certificate'],max(records[c][label['method']] for c in partition))
                self.assertLessEqual(label['certificate'],label['cap'])
                self.assertTrue(all(records[c]['status']=='full' for c in partition))
                self.assertIsNone(label['historical'])
        for split in self.split['splits']:
            self.assertEqual(len(split['core']['candidates']),K)
            self.assertEqual({r['cells'] for r in split['core']['candidates']},set(range(1,K+1)))
            self.assertEqual(len(split['rows']),6)
            self.assertEqual(split['core']['ltt_evidence']['alpha'],F(1,20))


    def test_generated_inputs_and_completed_pairs_resume_without_recomputation(self):
        before=(self.full_path.read_bytes(),self.split_path.read_bytes())
        with patch.object(assignment,'evaluate_case',side_effect=AssertionError('Completed output recomputed')):
            result=assignment.run(self.workspace,limit=1)
        self.assertEqual(result['completed_new'],0)
        self.assertEqual(before,(self.full_path.read_bytes(),self.split_path.read_bytes()))
        self.assertNotIn('fraction_selection', result)

    def test_generated_report_does_not_claim_complete_panel(self):
        result=assignment.report(self.workspace)
        self.assertFalse(result['complete']);self.assertEqual(result['independent_datasets'],1)
        self.assertEqual(result['condition_method_cap_rows'],36)
        for name in result['tables']:self.assertTrue((self.root/name).is_file())
        self.assertEqual(json.loads((self.root/'summary.json').read_text())['expected_datasets'],4800)

    def test_bad_input_metadata_or_root_allocation_rejected_before_work(self):
        design=assignment._read(self.root/'design.json');source=assignment._read(self.root/'inputs'/self.relative)
        changed=deepcopy(source);changed['metadata']['n']+=1
        with self.assertRaises(ValueError):assignment.evaluate_case(changed,design)
        with self.assertRaises(ValueError):assignment.evaluate_case(source,design,root_policy=RootPolicy(max_trials=2_000_000).__dict__)


if __name__=='__main__':unittest.main()
