"""Positive-margin oracle and sparse-mixture lower-bound reproduction."""
from __future__ import annotations

from collections import defaultdict
import csv
from fractions import Fraction as F
import json
import math
from pathlib import Path
import statistics

from experiments.jmlr_certification_bottleneck_v1_core import (
    analytical_crossing,
    boundary_alternative,
    lower_bound_record,
    margin_oracles,
)
from experiments.jmlr_efficiency_gap_v1_core import manuscript_bound
from . import releases


PROJECT = Path(__file__).resolve().parents[1]
CAPS = (F(5, 4), F(3, 2), F(2))
DELTAS = (F(0), F(1, 40), F(1, 20), F(1, 10), F(1, 5))


def configuration():
    cfg = json.loads((PROJECT/'configs/jmlr_certification_bottleneck_v1.json').read_text())
    if cfg['version'] != 'jmlr_certification_bottleneck_v1':
        raise ValueError('certification configuration version differs')
    if tuple(map(F, cfg['caps'])) != CAPS or tuple(map(F, cfg['oracle_log_margins'])) != DELTAS:
        raise ValueError('positive-margin grid differs from the configured study')
    if cfg['public_default_components'] != ['positive_margin_oracle', 'sparse_mixture_lower_bound']:
        raise ValueError('public certification defaults differ')
    return cfg


def cluster_summary(rows, metric, condition_count=48, replications=100):
    """Equal-condition summary clustered by shared replication index."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row['replication'])].append(float(row[metric]))
    if set(grouped) != set(range(replications)):
        raise ValueError('replication cluster grid is incomplete')
    if any(len(values) != condition_count for values in grouped.values()):
        raise ValueError('condition count differs within a replication cluster')
    values = [statistics.fmean(grouped[index]) for index in range(replications)]
    mean = statistics.fmean(values)
    se = statistics.stdev(values)/math.sqrt(replications) if replications > 1 else 0.0
    return {'n': replications, 'mean': mean, 'se': se,
            'lower': mean-1.96*se, 'upper': mean+1.96*se,
            'minimum': min(values), 'maximum': max(values)}


def _theory(cfg):
    lower = cfg['lower_bound']
    K, a, cap = int(lower['K']), F(lower['a']), F(lower['cap'])
    alpha, gamma = F(lower['alpha']), F(lower['gamma'])
    fixed = [lower_bound_record(K, a, F(b), cap, alpha, gamma)
             for b in lower['reference_b']]
    boundary = boundary_alternative(
        K, a, cap, F(lower['overlap_floor']), int(lower['near_boundary_denominator']))
    attained = lower_bound_record(K, a, boundary['b'], cap, alpha, gamma)
    sufficient = analytical_crossing(
        lambda n: manuscript_bound(n, K, alpha, F(1, K), F(1, 4), beta=F(1, 20)),
        cap=cap)
    if sufficient['first_integer'] != 2_852_384:
        raise ArithmeticError('Bonferroni sufficient crossing changed')
    return {
        'fixed_alternatives': fixed,
        'near_boundary_grid': boundary,
        'near_boundary_attained': attained,
        'sample_size_quantities': {
            'necessary_same_alternative': [
                {'b': row['b'], 'two_point_ceiling': row['two_point_ceiling'],
                 'sparse_mixture_ceiling': row['mixture_ceiling']}
                for row in fixed],
            'empirical_recovery_grid_first_complete_n': 256_000,
            'empirical_scope': 'first prespecified recovery-grid size with complete recovery; not an analytical necessity or sufficiency bound',
            'bonferroni_sufficient_first_integer': sufficient['first_integer'],
            'sufficient_scope': 'direct analytical CP-diameter criterion; not an executed sample size or optimal threshold',
        },
        'distinctions': {
            'experimental_alternative': 'positive-margin oracle counts over the 4,800 generated assignment datasets',
            'same_alternative_lower_bound': 'two-point and sparse-mixture calculations use the identical declared b within each row',
            'necessary_empirical_sufficient': 'lower-bound ceilings, the recovery-grid crossing, and the analytical sufficient integer are separate quantities',
        },
    }


def _original_rows(source, path):
    modes = [mode for mode in source['core']['modes'] if mode['mode'] == 'unknown_mass']
    if len(modes) != 1:
        raise ValueError('assignment artifact lacks one unknown-mass mode')
    labels = [row for row in modes[0]['labels']
              if row['method'] == 'coupled' and row['family'] == 'combined']
    if {row['cap'] for row in labels} != set(CAPS):
        raise ValueError('assignment artifact cap grid differs')
    return [{'procedure': 'original', 'cap': row['cap'],
             'selected_count': row['count'], 'selected_partition': row['partition'],
             'source': str(path)} for row in labels]


def run(workspace, *, limit=None):
    cfg = configuration()
    workspace, root = Path(workspace).resolve(), Path(workspace).resolve()/'certification'
    root.mkdir(parents=True, exist_ok=True)
    assignment = sorted((workspace/'assignment/evidence/large').glob('*/*.json.gz'))
    if limit is None and len(assignment) != cfg['historical_datasets']:
        raise FileNotFoundError('full certification reproduction requires 4,800 fresh assignment artifacts')
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('limit must be a positive integer')
        assignment = assignment[:limit]
    releases.dump(root/'configuration.json', cfg)
    releases.dump(root/'theory.json', _theory(cfg))
    completed = 0
    for path in assignment:
        source = releases.load(path)
        relative = path.relative_to(workspace/'assignment/evidence/large')
        output = root/'blocks'/relative
        if output.exists():
            completed += 1
            continue
        rows = margin_oracles(source, _original_rows(source, path), DELTAS, CAPS)
        releases.dump(output, {'version': cfg['version'], 'source': str(path.relative_to(workspace)),
                               'rows': rows, 'components': cfg['public_default_components']})
        completed += 1
    total = len(list((root/'blocks').glob('*/*.json.gz')))
    manifest = {'version': cfg['version'],
                'status': 'complete' if total == cfg['historical_datasets'] else 'incomplete',
                'assignment_datasets': total, 'expected_assignment_datasets': cfg['historical_datasets'],
                'new_or_checked': completed, 'bounded_prefix': limit,
                'default_components': cfg['public_default_components']}
    releases.dump(root/'manifest.json', manifest)
    return manifest


def _csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(value) if isinstance(value, F) else value
                             for key, value in row.items()})


def report(workspace, *, require_complete=True):
    root = Path(workspace).resolve()/'certification'
    cfg, theory = releases.load(root/'configuration.json'), releases.load(root/'theory.json')
    blocks = [releases.load(path) for path in sorted((root/'blocks').glob('*/*.json.gz'))]
    if require_complete and len(blocks) != cfg['historical_datasets']:
        raise ValueError('complete certification reporting requires all 4,800 datasets')
    rows = [row for block in blocks for row in block['rows']]
    tables = root/'tables'; tables.mkdir(parents=True, exist_ok=True)
    releases.dump(tables/'oracle_margin_exact.json', rows)
    releases.dump(tables/'lower_bound_exact.json', theory)
    rounded = []
    if rows:
        conditions = len({(row['id'], row['r']) for row in rows})
        replications = len({int(row['replication']) for row in rows})
        for cap in CAPS:
            for delta in DELTAS:
                subset = [row for row in rows if row['cap'] == cap and row['delta'] == delta
                          and row['oracle_count'] is not None]
                if len(subset) != conditions*replications:
                    if require_complete:
                        raise ValueError('positive-margin report grid is incomplete')
                    continue
                entry = {'cap': str(cap), 'delta': str(delta), 'datasets': len(subset),
                         'conditions': conditions, 'replication_clusters': replications}
                for metric in ('oracle_count', 'selected_count', 'positive_excess_over_K'):
                    value = cluster_summary(subset, metric, conditions, replications)
                    entry.update({metric+'_'+key: item for key, item in value.items()})
                rounded.append(entry)
    _csv(tables/'oracle_margin_summary.csv', rounded)
    sample = theory['sample_size_quantities']
    summary = {'version': cfg['version'],
               'status': 'complete' if len(blocks) == cfg['historical_datasets'] else 'incomplete',
               'datasets': len(blocks), 'positive_margin_rows': len(rows),
               'lower_bound_reference_ceiling': theory['fixed_alternatives'][0]['mixture_ceiling'],
               'empirical_recovery_grid_first_complete_n': sample['empirical_recovery_grid_first_complete_n'],
               'bonferroni_sufficient_first_integer': sample['bonferroni_sufficient_first_integer'],
               'full_precision': ['tables/oracle_margin_exact.json', 'tables/lower_bound_exact.json'],
               'rounded_display': 'tables/oracle_margin_summary.csv',
               'scope': theory['distinctions']}
    releases.dump(root/'summary.json', summary)
    return summary
