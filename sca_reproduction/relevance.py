"""Five-fraction LTT, propensity-logit clustering and RHC reproduction.

Uses assignment and release artifacts from the same workspace. The development
panel selects one-third for the A--F comparison on 4,320 held-out datasets.
The RHC illustration requires the checksum-verified public CSV.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from fractions import Fraction as F
import json
import math
from pathlib import Path
import statistics

from experiments.jmlr_empirical_relevance_v1_core import (
    LTT_CAPS,
    LTT_FRACTIONS,
    choose_ltt_fraction,
    evaluate_new_ltt_fractions,
    evaluate_selected_ltt_with_ablation,
    load_rhc,
    observed_split,
    release_clustering_block,
)
from . import assignment, releases


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_CAP = F(3, 2)
CONTRASTS = ('B-A', 'C-B', 'D-C', 'E-D', 'F-E', 'C-A', 'F-C', 'F-A')


def configuration():
    cfg = json.loads((PROJECT/'configs/jmlr_empirical_relevance_v1.json').read_text())
    if cfg['version'] != 'jmlr_empirical_relevance_v1':
        raise ValueError('empirical-relevance configuration version differs')
    if tuple(cfg['ltt_extension']['fractions']) != LTT_FRACTIONS:
        raise ValueError('five-fraction LTT grid differs')
    if (cfg['observed_case']['split']['calibration_size'],
            cfg['observed_case']['split']['outcome_size'],
            cfg['observed_case']['split']['count']) != (4014, 1721, 21):
        raise ValueError('RHC split protocol differs')
    if cfg['observed_case']['capacities'] != [2, 4, 8]:
        raise ValueError('RHC capacity grid differs')
    return cfg


def stratified_grid_summary(rows, value_key, condition_key='condition'):
    """Equal-weight fixed-condition mean and replication-cluster interval."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[condition_key]].append(float(row[value_key]))
    if not grouped:
        return {'conditions': 0, 'replications_per_condition': [], 'mean': math.nan,
                'se': math.nan, 'ci95_lower': math.nan, 'ci95_upper': math.nan}
    sizes = sorted(set(map(len, grouped.values())))
    if len(sizes) != 1:
        raise ValueError('fixed conditions have unequal replication counts')
    n = sizes[0]
    condition_keys = sorted(grouped)
    cluster_means = [statistics.fmean(grouped[key][index] for key in condition_keys)
                     for index in range(n)]
    mean = statistics.fmean(cluster_means)
    se = statistics.stdev(cluster_means)/math.sqrt(n) if n > 1 else 0.0
    return {'conditions': len(grouped), 'replications_per_condition': sizes,
            'mean': mean, 'se': se, 'ci95_lower': mean-1.96*se,
            'ci95_upper': mean+1.96*se}


def _condition(metadata):
    return metadata['id']+'_r'+str(metadata['r'])


def _relative(metadata):
    return assignment.relative_case(metadata)


def _bounded(paths, limit):
    paths = list(paths)
    if limit is None:
        return paths
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('limit must be a positive integer')
    return paths[:limit]


def _base_development_rows(full):
    answer = []
    for split in full['baseline']['splits']:
        for row in split['rows']:
            answer.append({'fraction': split['fraction'], 'procedure': row['procedure'],
                           'cap': row['cap'], 'cells': row['cells'],
                           'compression': row['compression'], 'status': row['status'],
                           'partition': row['partition'], 'source': 'assignment-base-fraction'})
    return answer


def _development_block(full, design, cfg):
    meta = full['metadata']; K = meta['K']
    new = evaluate_new_ltt_fractions(
        full['core']['N'], full['core']['T'], meta['population_index'], meta['r_index'],
        meta['replication'], tuple(range(K)), design['tree_orders'][str(K)], cfg['root_policy'])
    rows = _base_development_rows(full)
    for evaluation in new:
        for row in evaluation['rows']:
            rows.append({'fraction': evaluation['fraction'], 'procedure': row['procedure'],
                         'cap': row['cap'], 'cells': row['cells'],
                         'compression': row['compression'], 'status': row['status'],
                         'partition': row['partition'], 'source': cfg['version']})
    if {(row['fraction'], row['procedure'], row['cap']) for row in rows} != {
            (fraction, procedure, cap) for fraction in LTT_FRACTIONS
            for procedure in ('split_ltt', 'split_coupled') for cap in LTT_CAPS}:
        raise ArithmeticError('development fraction/procedure grid differs')
    return {'metadata': meta, 'condition': _condition(meta), 'rows': rows}


def _select(root, complete):
    blocks = [releases.load(path) for path in sorted((root/'ltt_development').glob('*/*.json.gz'))]
    totals = {fraction: F(0) for fraction in LTT_FRACTIONS}
    counts = {fraction: 0 for fraction in LTT_FRACTIONS}
    for block in blocks:
        for fraction in LTT_FRACTIONS:
            rows = [row for row in block['rows'] if row['fraction'] == fraction
                    and row['procedure'] == 'split_ltt' and row['cap'] == PRIMARY_CAP]
            if len(rows) != 1:
                raise ValueError('development LTT row differs')
            totals[fraction] += rows[0]['compression']; counts[fraction] += 1
    means = {fraction: totals[fraction]/counts[fraction]
             for fraction in LTT_FRACTIONS if counts[fraction]}
    selected = choose_ltt_fraction(means) if complete else '1/3'
    result = {'status': 'complete-development-selection' if complete else 'bounded-check-only',
              'selected_fraction': selected, 'mean_ltt_compression': means,
              'datasets_per_fraction': counts, 'development_data_used_only': True,
              'tie_rule': 'smallest fraction among exact maximizers',
              'partial_selection_claimed': False if not complete else None}
    if complete and selected != '1/3':
        raise ArithmeticError('complete public development panel did not reproduce selected fraction 1/3')
    releases.dump(root/'ltt_selection.json', result)
    return result


def run(workspace, rhc_data, *, limit=None):
    cfg = configuration(); workspace = Path(workspace).resolve(); root = workspace/'relevance'
    root.mkdir(parents=True, exist_ok=True); releases.dump(root/'configuration.json', cfg)
    design_path = workspace/'assignment/design.json'
    if not design_path.exists():
        raise FileNotFoundError('run assignment in this public workspace first')
    design = releases.load(design_path)
    full_paths = sorted((workspace/'assignment/evidence/large').glob('*/*.json.gz'))
    development = [path for path in full_paths if releases.load(path)['metadata']['replication'] < 10]
    evaluation = [path for path in full_paths if releases.load(path)['metadata']['replication'] >= 10]
    complete_upstream = len(development) == 480 and len(evaluation) == 4320
    if limit is None and not complete_upstream:
        raise FileNotFoundError('full relevance run requires all 4,800 fresh assignment artifacts')
    for path in _bounded(development, limit):
        full = releases.load(path); output = root/'ltt_development'/_relative(full['metadata'])
        if not output.exists():
            releases.dump(output, _development_block(full, design, cfg))
    selection = _select(root, complete_upstream and limit is None)
    if complete_upstream and limit is None:
        laws = {law['id']: law for law in design['laws']}
        for path in evaluation:
            full = releases.load(path); meta = full['metadata']; output = root/'ltt_evaluation'/_relative(meta)
            if output.exists():
                continue
            artifact = evaluate_selected_ltt_with_ablation(
                full['core']['N'], full['core']['T'], meta['population_index'], meta['r_index'],
                meta['replication'], tuple(range(meta['K'])), design['tree_orders'][str(meta['K'])],
                cfg['root_policy'], full, laws[full['law_id']], selected_fraction='1/3')
            releases.dump(output, {'metadata': meta, 'condition': _condition(meta), **artifact})

    release_protocol_path = workspace/'release/frozen_protocol.json'
    release_paths = sorted((workspace/'release/panel/blocks').glob('*.json.gz'))
    if limit is None and len(release_paths) != 300:
        raise FileNotFoundError('full relevance run requires 300 fresh release blocks')
    if release_paths and not release_protocol_path.exists():
        raise FileNotFoundError('release protocol is missing')
    protocol = releases.load(release_protocol_path) if release_protocol_path.exists() else None
    for path in _bounded(release_paths, limit):
        block = releases.load(path); output = root/'clustering'/path.name
        if not output.exists():
            releases.dump(output, release_clustering_block(block, protocol, cfg))

    rhc_path = Path(rhc_data).resolve()
    if not rhc_path.is_file():
        raise FileNotFoundError('verified RHC CSV is required; run prepare-data or pass --rhc-data')
    rhc = load_rhc(rhc_path, cfg['observed_case']['source_sha256'])
    observed_count = cfg['observed_case']['split']['count'] if limit is None else min(limit, 21)
    for split_index in range(observed_count):
        output = root/'observed'/f'split_{split_index:02d}.json.gz'
        if not output.exists():
            releases.dump(output, observed_split(rhc, split_index, cfg))

    counts = {'ltt_development': len(list((root/'ltt_development').glob('*/*.json.gz'))),
              'ltt_evaluation': len(list((root/'ltt_evaluation').glob('*/*.json.gz'))),
              'release_blocks': len(list((root/'clustering').glob('*.json.gz'))),
              'release_capacity_cases': sum(len(releases.load(path)['choices'])
                                            for path in (root/'clustering').glob('*.json.gz')),
              'rhc_splits': len(list((root/'observed').glob('*.json.gz')))}
    full = counts == {'ltt_development': 480, 'ltt_evaluation': 4320,
                      'release_blocks': 300, 'release_capacity_cases': 1500, 'rhc_splits': 21}
    manifest = {'version': cfg['version'], 'status': 'complete' if full else 'incomplete',
                'counts': counts, 'selected_fraction': selection['selected_fraction'],
                'independent_units': {'release_blocks': counts['release_blocks'],
                                      'capacity_cases_derived_from_blocks': counts['release_capacity_cases']},
                'rhc_dependence': 'one primary illustration plus dependent stability splits',
                'bounded_prefix': limit}
    releases.dump(root/'manifest.json', manifest)
    return manifest


def _write_csv(path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: str(value) if isinstance(value, F) else value
                             for key, value in row.items()})


def _ltt_tables(root, complete):
    development = [releases.load(path) for path in sorted((root/'ltt_development').glob('*/*.json.gz'))]
    dev_rows = []
    for block in development:
        for row in block['rows']:
            if row['procedure'] == 'split_ltt' and row['cap'] == PRIMARY_CAP:
                dev_rows.append({'condition': block['condition'],
                                 'replication': block['metadata']['replication'],
                                 'fraction': row['fraction'],
                                 'compression_pp': 100*float(row['compression'])})
    dev_summary = []
    for fraction in LTT_FRACTIONS:
        subset = [row for row in dev_rows if row['fraction'] == fraction]
        if subset:
            result = stratified_grid_summary(subset, 'compression_pp')
            dev_summary.append({'fraction': fraction, 'selected': fraction == '1/3', **result})

    evaluations = [releases.load(path) for path in sorted((root/'ltt_evaluation').glob('*/*.json.gz'))]
    values = defaultdict(list); contrasts = defaultdict(list); evaluation_exact = []
    for block in evaluations:
        rows = {row['procedure']: row for row in block['rows'] if row['cap'] == PRIMARY_CAP}
        if rows and set(rows) != set('ABCDEF'):
            raise ValueError('evaluation procedure grid differs')
        for method, row in rows.items():
            evaluation_exact.append({'condition': block['condition'],
                                     'replication': block['metadata']['replication'],
                                     'procedure': method,
                                     'compression_pp': row['compression_pp']})
            values[method].append({'condition': block['condition'],
                                   'replication': block['metadata']['replication'],
                                   'value': float(row['compression_pp'])})
        for contrast in CONTRASTS:
            left, right = contrast.split('-')
            if rows:
                contrasts[contrast].append({'condition': block['condition'],
                                            'replication': block['metadata']['replication'],
                                            'value': float(rows[left]['compression_pp']-
                                                           rows[right]['compression_pp'])})
    names = {'A': 'Split LTT baseline', 'B': 'Held-out first-failure ablation',
             'C': 'Held-out all-proposals ablation',
             'D': 'Full-sample proposal-list ablation',
             'E': 'Pilot-order family ablation', 'F': 'SCA (Coupled CP)'}
    methods_table = []
    for method, rows in sorted(values.items()):
        result = stratified_grid_summary(rows, 'value')
        methods_table.append({'procedure': method, 'procedure_name': names[method],
            'cap': str(PRIMARY_CAP), 'selected_fraction': '1/3',
            'compression_percent_mean': result['mean'], 'compression_percent_MCSE': result['se'],
            'compression_percent_analytic_Normal95_lower': result['ci95_lower'],
            'compression_percent_analytic_Normal95_upper': result['ci95_upper'],
            'conditions': result['conditions'], 'datasets': len(rows)})
    contrasts_table = []
    for name, rows in sorted(contrasts.items()):
        result = stratified_grid_summary(rows, 'value')
        contrasts_table.append({'contrast': name, 'difference_pp_mean': result['mean'],
            'cap': str(PRIMARY_CAP), 'selected_fraction': '1/3',
            'difference_pp_MCSE': result['se'],
            'difference_pp_analytic_Normal95_lower': result['ci95_lower'],
            'difference_pp_analytic_Normal95_upper': result['ci95_upper'],
            'conditions': result['conditions'], 'datasets': len(rows),
            'wins': sum(row['value'] > 0 for row in rows),
            'ties': sum(row['value'] == 0 for row in rows),
            'losses': sum(row['value'] < 0 for row in rows)})
    if complete and (len(evaluations) != 4320 or len(methods_table) != 6):
        raise ValueError('complete LTT report requires 4,320 coherent evaluation datasets')
    return dev_rows, evaluation_exact, dev_summary, methods_table, contrasts_table


def _observed_tables(root):
    blocks = [releases.load(path) for path in sorted((root/'observed').glob('*.json.gz'))]
    rows = []
    for block in blocks:
        for source in block['rows']:
            row = {key: value for key, value in source.items()
                   if not key.endswith(('_partition',)) and key != 'partition'}
            operational = bool(source['admitted'])
            row['operational_interval_released'] = operational
            if not operational:
                for key in tuple(row):
                    if key.endswith(('_lower', '_upper', '_width')):
                        row[key] = None
            rows.append(row)
    return rows


def report(workspace, *, require_complete=True):
    root = Path(workspace).resolve()/'relevance'; manifest = releases.load(root/'manifest.json')
    complete = manifest['status'] == 'complete'
    if require_complete and not complete:
        raise ValueError('complete relevance reporting requires the full generated panel')
    tables = root/'tables'; tables.mkdir(parents=True, exist_ok=True)
    dev_rows, evaluation_exact, dev_summary, methods, contrasts = _ltt_tables(root, complete)
    observed = _observed_tables(root)
    clustering = [releases.load(path) for path in sorted((root/'clustering').glob('*.json.gz'))]
    clustering_choices = [
        {'law_index': block['law_index'], 'replication': block['replication'],
         'capacity': int(capacity), 'groups': choice['cells'], 'status': choice['status'],
         'certificate': choice['certificate'],
         'admitted': bool(choice['status'] == 'finite' and choice['certificate'] <= PRIMARY_CAP)}
        for block in clustering for capacity, choice in block['choices'].items()]
    releases.dump(tables/'ltt_exact.json', {'development_rows': dev_rows,
                  'evaluation_rows': evaluation_exact,
                  'method_summary': methods, 'contrast_summary': contrasts})
    releases.dump(tables/'clustering_exact.json', clustering_choices)
    releases.dump(tables/'rhc_exact.json', observed)
    _write_csv(tables/'ltt_development.csv', dev_summary)
    _write_csv(tables/'ltt_methods.csv', methods)
    _write_csv(tables/'ltt_contrasts.csv', contrasts)
    _write_csv(tables/'clustering_capacity_cases.csv', clustering_choices)
    _write_csv(tables/'rhc_observed.csv', observed)
    primary = [row for row in observed if row['primary_split'] and row['capacity'] == 4]
    summary = {'version': manifest['version'], 'status': 'complete' if complete else 'incomplete',
               'selected_fraction': manifest['selected_fraction'], 'counts': manifest['counts'],
               'primary_rhc_capacity4': [{'method': row['method'], 'admitted': row['admitted'],
                                          'abstained': row['abstained'],
                                          'operational_interval_released': row['operational_interval_released']}
                                         for row in primary],
               'independent_release_blocks': manifest['counts']['release_blocks'],
               'derived_release_capacity_cases': manifest['counts']['release_capacity_cases'],
               'rhc_stability_splits_are_dependent': True,
               'full_precision': ['tables/ltt_exact.json', 'tables/clustering_exact.json',
                                  'tables/rhc_exact.json'],
               'rounded_displays': ['tables/ltt_development.csv', 'tables/ltt_methods.csv',
                                    'tables/ltt_contrasts.csv', 'tables/rhc_observed.csv']}
    releases.dump(root/'summary.json', summary)
    return summary
