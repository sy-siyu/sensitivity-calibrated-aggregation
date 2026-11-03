"""Small numerical fixtures for the main-table exporter; no saved results."""
from copy import deepcopy
import csv
from fractions import Fraction as F
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from sca_reproduction import paper_tables as paper


CONDITIONS = (('K16_balanced_moderate_ordered', 250), ('K64_skewed_weak_permuted', 4000))


def meta(condition, rep):
    law, r = condition
    return {'id': law, 'r': r, 'K': int(law.split('_')[0][1:]), 'replication': rep}




def continuous(values):
    values = list(map(float, values)); mean = statistics.fmean(values)
    se = statistics.stdev(values)/math.sqrt(len(values))
    return {'mean': mean, 'se': se, 'ci95_lower': mean-1.96*se, 'ci95_upper': mean+1.96*se}


def assignments():
    rows = []
    for condition in CONDITIONS:
        for rep in range(2):
            m = meta(condition, rep); K = m['K']
            for family, base in (('fixed', 3*K//4), ('contiguous', 5*K//8), ('combined', 9*K//16)):
                for method in ('coupled', 'independent'):
                    count = base+rep+(2 if method == 'independent' else 0)
                    oracle = (K//4 if K == 16 else K//2)+rep
                    rows.append({**m, 'mode': 'unknown_mass', 'method': method, 'family': family,
                        'cap': F(3, 2), 'count': count, 'oracle_count': oracle,
                        'compression': F(K-count, K), 'oracle_gap': count-oracle,
                        'certificate': F(6, 5) if method == 'coupled' else F(7, 5)})
    return rows


def efficiency(assignment_index):
    rows, summaries = [], []
    for condition in CONDITIONS:
        for rep in range(2):
            source = assignment_index[condition, rep, ('combined', 'coupled')]
            for procedure, extra in (('original', 0), ('allocation_control', 1), ('pooled_alternative', 2)):
                count, K, oracle = source['count']+extra, source['K'], source['oracle']
                rows.append({**meta(condition, rep), 'procedure': procedure, 'cap': F(3, 2),
                    'mass_mode': 'unknown_mass', 'selected_count': count, 'oracle_count': oracle,
                    'selected_certificate': source['certificate'], 'selection_status': 'finite',
                    'compression': F(K-count, K), 'normalized_count_gap': F(count-oracle, K),
                    'count_recovered': count == oracle})
    for procedure in ('original', 'allocation_control', 'pooled_alternative'):
        for metric in ('compression', 'normalized_count_gap', 'count_recovered'):
            clusters = [statistics.fmean(float(r[metric]) for r in rows
                        if r['procedure'] == procedure and r['replication'] == rep) for rep in range(2)]
            summaries.append({'procedure': procedure, 'metric': metric, 'conditions_equal_weighted': 2,
                              'n': 2, **continuous(clusters)})
    return rows, summaries




def relevance_tables():
    values = dict(A=10, B=8, C=9, D=12, E=23, F=22)
    common = dict(cap='3/2', selected_fraction='1/3', conditions=48, datasets=4320)
    def interval(prefix, value):
        return {prefix+'_mean': value, prefix+'_MCSE': .1,
                prefix+'_analytic_Normal95_lower': value-.196,
                prefix+'_analytic_Normal95_upper': value+.196}
    methods = [dict(common, procedure=name, **interval('compression_percent', value))
               for name,value in values.items()]
    contrasts=[]
    for name in paper.CONTRASTS:
        value=values[name[0]]-values[name[2]]
        contrasts.append(dict(common, contrast=name, **interval('difference_pp',value),
            wins=4320 if value>0 else 0, ties=4320 if value==0 else 0,
            losses=4320 if value<0 else 0))
    return methods,contrasts


def precision():
    rows, summaries = [], []
    for profile in paper.PROFILES:
        for rep in range(2):
            widths = [F(1, 10), F(1, 5), F(3, 10), F(1, 4) if rep == 0 else F(2, 5), F(1, 2)]
            if profile == 'alternating': widths = [v+F(1, 20) for v in widths]
            row = {'law_index': 0, 'law_id': 'K64_balanced_moderate_ordered', 'K': 64,
                'cap': F(3, 2), 'm_ratio': 256, 'profile': profile, 'replication': rep,
                'certificate': F(6, 5), 'transported_budget': F(12, 5), 'budget_ceiling': F(3),
                'global_accounting_exact': True, 'shared_coverage_event': rep != 0,
                'global_nesting_failure_on_shared_event': False, 'admitted': rep != 0}
            for j, width in enumerate(widths):
                row.update({f'global_I{j}_lower': F(), f'global_I{j}_upper': width,
                            f'global_I{j}_width': width})
            row.update({'z_global_lower': F(), 'z_global_upper': widths[-1], 'z_global_width': widths[-1],
                        'global_total_excess': widths[-1]-widths[0]})
            for j, component in enumerate(paper.COMPONENTS):
                row['global_'+component] = widths[j+1]-widths[j]
                row[f'global_I{j}_in_I{j+1}'] = widths[j] <= widths[j+1]
            row['global_nesting'] = all(widths[j] <= widths[j+1] for j in range(4))
            rows.append(row)
        summary = {'law_index': 0, 'cap': F(3, 2), 'm_ratio': 256, 'transport': 'global',
                   'profile': profile, 'n_blocks': 2, 'n_expected_blocks': 2, 'n_missing_blocks': 0}
        for j in range(5):
            stats = continuous([r[f'global_I{j}_width'] for r in rows if r['profile'] == profile])
            summary[f'I{j}_width_n'] = 2
            summary.update({f'I{j}_width_'+key: value for key, value in stats.items()})
        for component in paper.COMPONENTS:
            summary[component+'_mean'] = statistics.fmean(float(r['global_'+component]) for r in rows if r['profile'] == profile)
        summaries.append(summary)
    return rows, summaries


class PaperTableNumerics(unittest.TestCase):
    def test_assignment_normalization_and_paired_coupling(self):
        table, index = paper._assignment(assignments(), conditions=CONDITIONS, reps=range(2))
        self.assertEqual(len(table), 3)
        self.assertEqual(table[0]['coupling_gain_pp_exact_mean'], F(125, 16))
        self.assertEqual(table[0]['compression_percent_exact_mean'], F(1475, 64))
        self.assertEqual(table[0]['oracle_compression_percent_exact_mean'], F(3875, 64))
        pooled = F(100*(sum(v['K'] for (c,r,label),v in index.items()
                            if label==('fixed','coupled'))-
                        sum(v['oracle'] for (c,r,label),v in index.items()
                            if label==('fixed','coupled'))),
                   sum(v['K'] for (c,r,label),v in index.items()
                       if label==('fixed','coupled')))
        self.assertNotEqual(table[0]['oracle_compression_percent_exact_mean'],pooled)
        for row in table:
            self.assertEqual(row['datasets'], 4)
            self.assertEqual(row['conditions'], 2)
            self.assertEqual(row['recoveries'], 0)
            self.assertEqual(row['oracle_compression_percent_exact_mean'],F(3875,64))
        direct = [F(100*(v['count']-v['oracle']), v['K']) for (c, r, label), v in index.items()
                  if label == ('combined', 'coupled')]
        self.assertEqual(table[2]['oracle_gap_percent_K_exact_mean'], sum(direct)/4)

    def test_assignment_missing_duplicate_or_wrong_derived_quantity_refused(self):
        rows = assignments()
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError): paper._assignment(bad, conditions=CONDITIONS, reps=range(2))
        bad = deepcopy(rows); bad[0]['oracle_gap'] += 1
        with self.assertRaises(ValueError): paper._assignment(bad, conditions=CONDITIONS, reps=range(2))

    def test_efficiency_source_preservation_and_unfavorable_alternative(self):
        _, source = paper._assignment(assignments(), conditions=CONDITIONS, reps=range(2))
        rows, summaries = efficiency(source)
        table, checks = paper._efficiency(rows, summaries, source, conditions=CONDITIONS, reps=range(2))
        self.assertEqual(checks['original_assignment_matches'], 4)
        self.assertEqual(checks['combined_oracle_count_matches'], 12)
        self.assertTrue(all(row['oracle_compression_percent_exact_mean']==F(3875,64)
                            for row in table))
        self.assertLess(table[1]['compression_percent_mean'], table[0]['compression_percent_mean'])
        self.assertGreater(table[1]['oracle_gap_percent_K_mean'], table[0]['oracle_gap_percent_K_mean'])
        self.assertIn('oracle_gap_percent_K_exact_mean',table[0])
        bad = deepcopy(summaries); bad[0]['n'] = 1
        with self.assertRaises(ValueError): paper._efficiency(rows, bad, source, conditions=CONDITIONS, reps=range(2))
        bad = deepcopy(rows); bad[0]['selected_certificate'] = F(7, 5)
        with self.assertRaises(ValueError): paper._efficiency(bad, summaries, source, conditions=CONDITIONS, reps=range(2))
        bad = deepcopy(rows)
        changed = next(row for row in bad if row['procedure']=='allocation_control')
        changed['oracle_count'] += 1
        changed['normalized_count_gap'] = F(changed['selected_count']-changed['oracle_count'],changed['K'])
        changed['count_recovered'] = changed['selected_count']==changed['oracle_count']
        with self.assertRaisesRegex(ValueError,'population-oracle count'):
            paper._efficiency(bad, summaries, source, conditions=CONDITIONS, reps=range(2))



    def test_current_relevance_preserves_signed_contrasts(self):
        methods, contrasts, checks = paper._current_relevance(*relevance_tables())
        self.assertEqual(len(methods), 6)
        self.assertEqual(len(contrasts), 8)
        self.assertEqual(checks['selected_fraction'], '1/3')
        by_name = {row['contrast']: row for row in contrasts}
        self.assertEqual(by_name['F-E']['difference_pp_mean'], -1)
        self.assertEqual(by_name['F-E']['losses'], 4320)
        self.assertEqual(by_name['F-A']['difference_pp_mean'], 12)

    def test_old_fraction_and_incomplete_current_tables_are_rejected(self):
        methods, contrasts = relevance_tables()
        methods[0]['selected_fraction']='1/5'
        with self.assertRaises(ValueError): paper._current_relevance(methods, contrasts)
        methods, contrasts = relevance_tables()
        contrasts[0]['datasets']=4319
        with self.assertRaises(ValueError): paper._current_relevance(methods, contrasts)
        methods, contrasts = relevance_tables()
        contrasts[0]['wins']=0
        with self.assertRaises(ValueError): paper._current_relevance(methods, contrasts)
        methods, contrasts = relevance_tables()
        contrasts[0]['difference_pp_mean']+=1
        with self.assertRaises(ValueError): paper._current_relevance(methods, contrasts)

    def test_precision_keeps_negative_increment_and_all_blocks_without_admission_filter(self):
        rows, summaries = precision()
        stages, increments, checks = paper._precision(rows, summaries, reps=range(2))
        self.assertEqual(len(stages), 5); self.assertEqual(len(increments), 4)
        self.assertAlmostEqual(stages[3]['aligned_mean'], .325)
        self.assertEqual(stages[0]['aligned_blocks'], 2)
        self.assertEqual(checks['aligned']['accounting_exact'], 2)
        self.assertEqual(checks['aligned']['outcome_sampling_negative'], 1)
        self.assertEqual(checks['aligned']['nesting_failures'], 1)
        self.assertEqual(checks['aligned']['nesting_failures_on_shared_event'], 0)
        self.assertEqual(checks['aligned']['I2_in_I3_blocks'], 1)
        self.assertEqual(checks['aligned']['I2_in_I3_failures'], 1)
        self.assertAlmostEqual(sum(r['aligned_mean'] for r in increments),
                               stages[-1]['aligned_mean']-stages[0]['aligned_mean'])

    def test_precision_missing_blocks_bad_flags_and_cap_in_place_of_actual_budget_refused(self):
        rows, summaries = precision()
        with self.assertRaises(ValueError): paper._precision(rows[:-1], summaries, reps=range(2))
        bad = deepcopy(rows); bad[0]['global_I2_in_I3'] = True
        with self.assertRaises(ValueError): paper._precision(bad, summaries, reps=range(2))
        bad = deepcopy(rows); bad[0]['transported_budget'] = 3
        with self.assertRaises(ValueError): paper._precision(bad, summaries, reps=range(2))
        bad = deepcopy(summaries); bad[0]['I0_width_ci95_upper'] = 1
        with self.assertRaises(ValueError): paper._precision(rows, bad, reps=range(2))

    def test_public_generate_refuses_bounded_prefix_before_writing_tables(self):
        with tempfile.TemporaryDirectory(prefix='sca_paper_table_test_') as directory:
            root = Path(directory); path = root/'assignment/tables/assignment_rows.csv'
            path.parent.mkdir(parents=True)
            rows = assignments()
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Incomplete or incompatible paper panel'):
                paper.generate(root)
            self.assertFalse((root/'paper_tables').exists())


if __name__ == '__main__': unittest.main()
