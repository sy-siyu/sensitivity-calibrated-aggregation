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


def grid(groups, prefix):
    means = [statistics.fmean(map(float, values)) for values in groups]
    se = math.sqrt(sum(statistics.variance(map(float, values))/len(values)
                       for values in groups))/len(groups)
    mean = statistics.fmean(means)
    radius = statistics.NormalDist().inv_cdf(.975)*se
    return {prefix+'_conditions': len(groups), prefix+'_expected_conditions': len(groups),
            prefix+'_paired_datasets': sum(map(len, groups)), prefix+'_all_conditions_present': True,
            prefix+'_mean': mean, prefix+'_MCSE': se,
            prefix+'_analytic_Normal95_lower': mean-radius,
            prefix+'_analytic_Normal95_upper': mean+radius}


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
                    oracle = K//4
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


def splits():
    rows, methods, pairs, contrasts = [], [], [], []
    values = {}
    for condition in CONDITIONS:
        for rep in (10, 11):
            m = meta(condition, rep); K = m['K']
            counts = dict(A=K-2-(rep % 2), B=K-1, C=K-2, D=K-3, E=K-5, F=K-4)
            for procedure, cells in counts.items():
                compression = F(100*(K-cells), K)
                values[condition, rep, procedure] = compression
                rows.append({**m, 'procedure': procedure, 'cap': F(3, 2), 'cells': cells,
                    'compression_pp': compression, 'count_optimal': True,
                    'computation_status': 'imported_preserved_selection' if procedure in 'ABF' else 'finite',
                    'computation_complete': True, 'deterministic_tie_complete': True,
                    'identity': False, 'identity_fallback': False})
            pair = {**m, 'cap': F(3, 2), 'accounting_identity_exact': True}
            pair.update({contrast+'_pp': values[condition, rep, contrast[0]]-values[condition, rep, contrast[2]]
                         for contrast in paper.CONTRASTS})
            pairs.append(pair)
    for procedure in 'ABCDEF':
        groups = [[values[c, r, procedure] for r in (10, 11)] for c in CONDITIONS]
        methods.append({'scope': 'all', 'cap': F(3, 2), 'procedure': procedure,
                        'procedure_name': 'Fixture '+procedure, **grid(groups, 'compression_pp')})
    for contrast in paper.CONTRASTS:
        groups = [[values[c, r, contrast[0]]-values[c, r, contrast[2]] for r in (10, 11)] for c in CONDITIONS]
        all_values = [v for group in groups for v in group]
        means = [sum(group)/len(group) for group in groups]
        contrasts.append({'scope': 'all', 'cap': F(3, 2), 'contrast': contrast,
            **grid(groups, 'difference_pp'), 'wins': sum(v > 0 for v in all_values),
            'ties': sum(v == 0 for v in all_values), 'losses': sum(v < 0 for v in all_values),
            'conditions_mean_positive': sum(v > 0 for v in means),
            'conditions_mean_zero': sum(v == 0 for v in means),
            'conditions_mean_negative': sum(v < 0 for v in means)})
    return rows, methods, pairs, contrasts


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
        for row in table:
            self.assertEqual(row['datasets'], 4)
            self.assertEqual(row['conditions'], 2)
            self.assertEqual(row['recoveries'], 0)
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
        self.assertLess(table[1]['compression_percent_mean'], table[0]['compression_percent_mean'])
        self.assertGreater(table[1]['oracle_gap_percent_K_mean'], table[0]['oracle_gap_percent_K_mean'])
        bad = deepcopy(summaries); bad[0]['n'] = 1
        with self.assertRaises(ValueError): paper._efficiency(rows, bad, source, conditions=CONDITIONS, reps=range(2))
        bad = deepcopy(rows); bad[0]['selected_certificate'] = F(7, 5)
        with self.assertRaises(ValueError): paper._efficiency(bad, summaries, source, conditions=CONDITIONS, reps=range(2))

    def test_split_means_eight_contrasts_signed_accounting_and_fractions(self):
        fixtures = splits()
        methods, contrasts, checks = paper._split(*fixtures, conditions=CONDITIONS, reps=range(10, 12))
        self.assertEqual(len(methods), 6); self.assertEqual(len(contrasts), 8)
        self.assertEqual(checks['accounting_exact'], 4)
        self.assertEqual(checks['structural_checks'], {'C_le_B': 4, 'E_le_D': 4})
        self.assertTrue(all(row['selection_unresolved'] == 0 for row in methods))
        by_name = {r['contrast']: r for r in contrasts}
        self.assertEqual(by_name['F-E']['losses_fraction_exact'], 1)
        self.assertLess(by_name['F-E']['difference_pp_mean'], 0)
        self.assertAlmostEqual(by_name['F-A']['difference_pp_mean'],
            sum(by_name[c]['difference_pp_mean'] for c in ('B-A', 'C-B', 'D-C', 'E-D', 'F-E')))
        self.assertEqual(by_name['C-B']['wins'], 4)
        self.assertEqual(by_name['C-B']['datasets'], 4)

    def test_split_incomplete_grid_pair_and_wrong_win_counts_refused(self):
        fixtures = splits()
        bad = deepcopy(fixtures); bad[1][0]['compression_pp_paired_datasets'] = 3
        with self.assertRaises(ValueError): paper._split(*bad, conditions=CONDITIONS, reps=range(10, 12))
        bad = deepcopy(fixtures); bad[2].pop()
        with self.assertRaises(ValueError): paper._split(*bad, conditions=CONDITIONS, reps=range(10, 12))
        bad = deepcopy(fixtures); bad[3][0]['wins'] = 0
        with self.assertRaises(ValueError): paper._split(*bad, conditions=CONDITIONS, reps=range(10, 12))

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
