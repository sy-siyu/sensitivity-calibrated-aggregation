"""SCA release, outcome, precision and computation study orchestration.

Inputs are generated from the study configurations and the supplied public
IHDP covariate file. Results are written under the caller's output directory.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from experiments import jmlr_verified_adaptive_v1_math as exact


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = {
    'release': 'jmlr_release_capacity_v1',
    'outcome': 'jmlr_outcome_inference_v1',
    'precision': 'jmlr_precision_frontier_v1',
    'computation': 'jmlr_computation_v1',
}


def load(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return exact.decode(json.load(stream))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with (gzip.open(temporary, 'wt', compresslevel=3) if path.suffix == '.gz'
          else temporary.open('w')) as stream:
        json.dump(exact.encode(value), stream, allow_nan=False, separators=(',', ':'))
    temporary.replace(path)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def configuration(study, config=None):
    """Load a public numerical configuration, optionally with explicit overrides.

    Different support/replication counts are useful for smoke checks. Their
    generated protocols record the different design; they are never labeled
    the completed full paper panel. Scientific budgets and policies are not
    implicitly relaxed for smoke runs.
    """
    path = PROJECT / 'configs' / (CONFIG_NAMES[study] + '.json')
    base = json.loads(path.read_text())
    scientific = deepcopy(base)
    if config is not None:
        override = config if isinstance(config, dict) else json.loads(Path(config).read_text())
        base = _merge(base, override)
    for key in ('law_source', 'tree_source', 'bridge_source', 'bridge_expected_source'):
        base.pop(key, None)
    if isinstance(base.get('replications'), bool) or not isinstance(base.get('replications'), int) or base['replications'] < 1:
        raise ValueError('Replication count must be a positive integer')
    # The inherited kernels explicitly implement these constants. Do not let
    # an overridden metadata label silently disagree with the actual method.
    for key in ('G', 'Lambda_max', 'cap', 'calibration_n_per_K', 'outcome_m_per_K',
                'alpha_cal', 'alpha_raw_outcomes', 'alpha_aggregate_outcomes',
                'alpha_cell_outcomes', 'alpha_raw', 'alpha_z', 'mass_mode', 'root_policy'):
        if key in scientific and base[key] != scientific[key]:
            raise ValueError('The reproduction preserves its scientific/numerical setting: ' + key)
    if 'sampler' in scientific and base['sampler'] != scientific['sampler']:
        raise ValueError('The reproduction preserves the declared sampler streams')
    return base


def controlled_law(K, law_index):
    """Original exact stored-binary law convention, generated rather than read."""
    from experiments.jmlr_computation_v1 import extended_law
    if isinstance(K, bool) or not isinstance(K, int) or K < 2 or K & (K - 1):
        raise ValueError('Controlled support must be a power of two at least two')
    if law_index not in (0, 1):
        raise ValueError('Controlled law index must be zero or one')
    law = extended_law(K, law_index)
    law['source_identity'] = 'SCA controlled-law generator; exact stored-binary convention'
    return law


def fixed_trees(K, master_seed=20261008):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([master_seed, 101, K])))
    return [list(range(K)), rng.permutation(K).tolist()]


def _new_root(root, cfg):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg['output_dir'] = str(root)
    dump(root / 'configuration.json', cfg)
    return root


def _tasks(protocol, blocks=None, limit=None, preflight=False):
    count = 1 if preflight else protocol['configuration']['replications']
    allowed = [(i, j) for i in range(len(protocol['laws'])) for j in range(count)]
    if blocks is not None:
        chosen = [tuple(pair) for pair in blocks]
        if len(set(chosen)) != len(chosen) or any(pair not in allowed for pair in chosen):
            raise ValueError('Block coordinates must be distinct members of the frozen design')
    else:
        chosen = allowed
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('An explicit limit must be a positive number of blocks')
        chosen = chosen[:limit]
    return chosen, allowed


@contextmanager
def _release_paths(module, root):
    """Only the public checkout's process-local output pointers are changed."""
    previous = module.ROOT, module.CONFIG
    module.ROOT, module.CONFIG = root, root / 'configuration.json'
    try:
        yield
    finally:
        module.ROOT, module.CONFIG = previous


def _finish(root, stage, expected, receipts, failures, started):
    completed = sum((root / stage / 'blocks' / f'{i}_{j:03d}.json.gz').exists()
                    for i, j in expected)
    manifest = load(root / 'manifest.json')
    manifest[stage] = {'completed_blocks': completed, 'required_blocks': len(expected),
                       'wall_seconds': time.perf_counter() - started, 'failures': failures}
    manifest.update(status=stage + ('_complete' if completed == len(expected) and not failures else '_incomplete'),
                    completed_blocks=completed)
    dump(root / 'manifest.json', manifest)
    if failures:
        raise RuntimeError('SCA blocks failed; failure receipts are retained and completed blocks are preserved')
    return {'root': str(root), 'completed_blocks': completed, 'expected_blocks': len(expected),
            'new_receipts': receipts, 'status': manifest['status']}


def prepare_release(root, ihdp_csv, config=None):
    """Freeze laws/support/streams from generators and public covariates."""
    from experiments.jmlr_outcome_inference_v1_core import outcome_profiles
    from experiments.jmlr_release_capacity_v1 import grid
    from experiments.jmlr_release_capacity_v1_sampling import build_benchmark
    cfg = configuration('release', config)
    K = cfg.get('controlled_support', 64)
    laws = [controlled_law(K, i) for i in range(2)]
    benchmark = build_benchmark(Path(ihdp_csv))
    laws.append(benchmark['law'])
    cfg['controlled_laws'] = [law['id'] for law in laws[:2]]
    cfg['semi_synthetic_source'] = 'public IHDP CSV supplied explicitly to preparation'
    cfg['law_construction'] = 'fresh controlled generators and verified public finite IHDP reference'
    root = _new_root(root, cfg)
    profiles = [outcome_profiles(law) for law in laws[:2]] + [benchmark['profiles']]
    grids = [grid(law['K']) for law in laws]
    row_count = cfg['replications'] * sum(2 * len(g) * len(p) * len(cfg['outcome_m_per_K'])
                                         for g, p in zip(grids, profiles))
    protocol = {
        'public_study': 'SCA reporting capacity', 'frozen_utc': _now(), 'configuration': cfg,
        'laws': laws, 'profiles': profiles, 'tree_orders': [fixed_trees(law['K']) for law in laws],
        'reference_record_states': benchmark['record_states'],
        'semi_synthetic_construction': benchmark['construction'],
        'semi_synthetic_targets': benchmark['targets'], 'capacity_grids': grids,
        'committed_blocks': 3 * cfg['replications'], 'committed_condition_rows': row_count,
        'frozen_before_any_evaluation': True,
        'input_boundary': 'generators and verified public covariates; no saved study data',
    }
    dump(root / 'frozen_protocol.json', protocol)
    dump(root / 'manifest.json', {'status': 'frozen', 'public_study': 'SCA reporting capacity',
         'created_utc': _now(), 'independent_blocks_expected': protocol['committed_blocks'],
         'condition_rows_expected': row_count, 'workers': cfg['execution']['workers'],
         'numpy': np.__version__, 'generated_from_source': True})
    return protocol


def run_release(root, *, blocks=None, limit=None, preflight=False):
    from experiments import jmlr_release_capacity_v1 as study
    root = Path(root).resolve()
    protocol = load(root / 'frozen_protocol.json')
    requested, expected = _tasks(protocol, blocks, limit, preflight)
    stage = 'preflight' if preflight else 'panel'
    (root / stage / 'blocks').mkdir(parents=True, exist_ok=True)
    receipts, failures, started = [], [], time.perf_counter()
    with _release_paths(study, root), study.ownership():
        for i, j in requested:
            if (root / stage / 'blocks' / f'{i}_{j:03d}.json.gz').exists():
                continue
            try:
                receipt = study.process_block(i, j, preflight)
                receipts.append(receipt)
                dump(root / stage / 'receipts' / f'{i}_{j:03d}.json', receipt)
            except Exception:
                failure = {'law_index': i, 'replication': j, 'error': traceback.format_exc()}
                failures.append(failure)
                dump(root / stage / 'failures' / f'{i}_{j:03d}.json', failure)
        return _finish(root, stage, expected, receipts, failures, started)


def prepare_outcome(root, config=None):
    from experiments.jmlr_outcome_inference_v1_sampling import stream_record
    cfg = configuration('outcome', config)
    supports = cfg.get('outcome_supports', [16, 64, 64])
    if len(supports) != 3:
        raise ValueError('The outcome study declares three ordered law slots')
    laws = [controlled_law(K, int(i == 2)) for i, K in enumerate(supports)]
    cfg['laws'] = [law['id'] for law in laws]
    root = _new_root(root, cfg)
    protocol = {'public_study': 'SCA outcome inference', 'frozen_utc': _now(),
                'configuration': cfg, 'laws': laws,
                'tree_orders': {str(K): fixed_trees(K) for K in set(supports)},
                'random_streams': [{'law_index': i, 'law_id': law['id'], 'replication': j,
                                   'streams': stream_record(i, j)}
                                  for i, law in enumerate(laws) for j in range(cfg['replications'])],
                'protocol_frozen_before_new_outcome_comparisons': True,
                'input_boundary': 'fresh exact generators and preserved 20261014 sampler streams'}
    dump(root / 'frozen_protocol.json', protocol)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    dump(root / 'manifest.json', {'status': 'prepared', 'public_study': 'SCA outcome inference',
         'created_utc': _now(), 'compute_seconds': 0., 'stages': [],
         'configuration_sha256': digest(root / 'configuration.json'),
         'frozen_protocol_sha256': digest(root / 'frozen_protocol.json'),
         'calibration_datasets': 0, 'outcome_evaluations': 0, 'generated_from_source': True})
    dump(root / 'preservation.json', {})
    return protocol


def run_outcome(root, *, blocks=None, limit=None, preflight=False, replay=True):
    from experiments import jmlr_outcome_inference_v1 as study
    root = Path(root).resolve()
    protocol = load(root / 'frozen_protocol.json')
    requested, expected = _tasks(protocol, blocks, limit, preflight)
    base = root / 'preflight' if preflight else root
    base.mkdir(parents=True, exist_ok=True)
    receipts, failures, started = [], [], time.perf_counter()
    with study.ownership(root):
        for i, j in requested:
            if (base / 'blocks' / f'{i}_{j:03d}.json.gz').exists():
                continue
            try:
                receipt = study.process_block(base, i, j, protocol, preflight)
                receipt['evaluations'] = 3 * len(protocol['configuration']['outcome_m_per_K'])
                receipts.append(receipt)
                dump(base / 'receipts' / f'{i}_{j:03d}.json', receipt)
            except Exception:
                failure = {'law_index': i, 'replication': j, 'error': traceback.format_exc()}
                failures.append(failure)
                dump(base / 'failures' / f'{i}_{j:03d}.json', failure)
        complete = sum((base / 'blocks' / f'{i}_{j:03d}.json.gz').exists() for i, j in expected)
        manifest = load(root / 'manifest.json')
        if preflight:
            manifest['preflight'] = {'completed_blocks': complete, 'required_blocks': len(expected),
                                     'failures': failures}
        else:
            manifest.update(status='panel_complete' if complete == len(expected) and not failures else 'panel_incomplete',
                            calibration_datasets=complete,
                            outcome_evaluations=complete * 3 * len(protocol['configuration']['outcome_m_per_K']))
        manifest['compute_seconds'] += time.perf_counter() - started
        dump(root / 'manifest.json', manifest)
        all_receipts = [load(path) for path in sorted((base / 'receipts').glob('*.json'))]
        if not preflight:
            dump(root / 'panel_checks.json', {'status': 'passed' if complete == len(expected) and not failures else 'incomplete',
                 'blocks': all_receipts, 'calibration_datasets': complete,
                 'independent_outcome_blocks': complete, 'outcome_evaluations': manifest['outcome_evaluations']})
    if failures:
        raise RuntimeError('SCA outcome block failures are retained; resume only missing blocks')
    if replay:
        replay_outcome(root, blocks=requested, preflight=preflight)
    return {'root': str(root), 'completed_blocks': complete, 'expected_blocks': len(expected),
            'new_receipts': receipts, 'status': load(root / 'manifest.json')['status']}


def replay_outcome(root, *, blocks=None, limit=None, preflight=False):
    from experiments import jmlr_outcome_inference_v1 as study
    root = Path(root).resolve()
    protocol = load(root / 'frozen_protocol.json')
    requested, expected = _tasks(protocol, blocks, limit, preflight)
    if protocol['configuration']['outcome_m_per_K'] != [4, 16, 64, 256]:
        raise ValueError('Existing complete outcome replay requires its four prespecified prefixes')
    base = root / 'preflight' if preflight else root
    with study.ownership(root):
        for i, j in requested:
            path = base / 'verification' / f'{i}_{j:03d}.json'
            if not path.exists():
                dump(path, study.replay_block(base, i, j, protocol))
        receipts = [load(base / 'verification' / f'{i}_{j:03d}.json') for i, j in expected
                    if (base / 'verification' / f'{i}_{j:03d}.json').exists()]
        total = {'status': 'passed' if len(receipts) == len(expected) else 'incomplete', 'blocks': receipts,
                 **{key: sum(r[key] for r in receipts) for key in
                    ('selection_replays', 'outcome_replays', 'LP_bound_certificates')}}
        total['root_stats'] = {key: sum(r['root_stats'][key] for r in receipts)
                               for key in (receipts[0]['root_stats'] if receipts else ())}
        if not preflight:
            dump(root / 'replay_checks.json', total)
            manifest = load(root / 'manifest.json')
            if total['status'] == 'passed':
                manifest['status'] = 'replay_complete'
            dump(root / 'manifest.json', manifest)
        return total


def prepare_precision(root, release_root, config=None):
    cfg = configuration('precision', config)
    release_root = Path(release_root).resolve()
    source = load(release_root / 'frozen_protocol.json')
    sc = source['configuration']
    for key in ('calibration_n_per_K', 'outcome_m_per_K', 'G', 'alpha_cal',
                'alpha_raw_outcomes', 'alpha_aggregate_outcomes', 'mass_mode', 'root_policy'):
        if cfg[key] != sc[key]:
            raise ValueError('Fresh release and precision configurations differ: ' + key)
    cfg.update(replications=sc['replications'], independent_blocks=3 * sc['replications'],
               source_dir=str(release_root), source_protocol=str(release_root / 'frozen_protocol.json'),
               law_ids=[law['id'] for law in source['laws']])
    if list(map(F, cfg['caps'])) != [F(1), F(11, 10), F(5, 4), F(3, 2), F(2)]:
        raise ValueError('SCA precision study preserves all five prespecified caps')
    root = _new_root(root, cfg)
    analyses = sc['replications'] * sum(len(p) * 2 * 5 for p in source['profiles'])
    protocol = {'public_study': 'SCA precision operating curves', 'version': cfg['version'],
         'frozen_utc': _now(), 'configuration': cfg, 'source_dir': str(release_root),
         'source_protocol': str(release_root / 'frozen_protocol.json'),
         'laws': [{'law_index': i, 'id': law['id'], 'K': law['K'],
                   'profiles': list(source['profiles'][i]), 'capacity_limit': cfg['capacity_limits'][i]}
                  for i, law in enumerate(source['laws'])],
         'committed_selection_rows': 15 * sc['replications'], 'committed_analysis_instances': analyses,
         'committed_method_records': 4 * analyses, 'frozen_before_frontier_computation': True,
         'source_design_reuse': 'Freshly regenerated SCA release blocks only; same observations paired across caps'}
    dump(root / 'frozen_protocol.json', protocol)
    dump(root / 'manifest.json', {'status': 'frozen', 'created_utc': _now(),
         'public_study': 'SCA precision operating curves', 'new_observations': 0,
         'scope': {'independent_blocks': cfg['independent_blocks'],
                   'selection_rows': protocol['committed_selection_rows'], 'analysis_instances': analyses,
                   'method_records': 4 * analyses},
         'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                         'workers': cfg['execution']['workers'], 'native_threads': 1}})
    return protocol


def run_precision(root, *, blocks=None, limit=None, preflight=False):
    from experiments import jmlr_precision_frontier_v1 as study
    root = Path(root).resolve()
    protocol = load(root / 'frozen_protocol.json')
    requested, expected = _tasks(protocol, blocks, limit, preflight)
    stage = 'preflight' if preflight else 'panel'
    receipts, failures, started = [], [], time.perf_counter()
    with study.ownership(root):
        for i, j in requested:
            path = root / stage / 'blocks' / f'{i}_{j:03d}.json.gz'
            if path.exists():
                continue
            tick = time.perf_counter()
            try:
                record = study.process_block(protocol, i, j, preflight, integration_check=preflight)
                dump(path, record)
                receipt = {'law_index': i, 'replication': j,
                           'selection_rows': len(record['selection_rows']),
                           'analysis_instances': len(record['analysis_rows']),
                           'seconds': time.perf_counter() - tick, 'bytes': path.stat().st_size,
                           'timings': record['timings'], 'cache': record['cache']}
                receipts.append(receipt)
                dump(root / stage / 'receipts' / f'{i}_{j:03d}.json', receipt)
            except Exception:
                failure = {'law_index': i, 'replication': j, 'error': traceback.format_exc()}
                failures.append(failure)
                dump(root / stage / 'failures' / f'{i}_{j:03d}.json', failure)
        return _finish(root, stage, expected, receipts, failures, started)


def release_bridge_counts(law_index, replication, preflight=False):
    """Regenerate exactly the K64 release-study counts used by computation.

    No release certification, stored observations or frozen study files enter
    this bridge. Calibration/outcome draws retain the original release keys.
    """
    from experiments.jmlr_outcome_inference_v1_core import outcome_profiles
    from experiments.jmlr_outcome_inference_v1_sampling import counts
    from experiments.jmlr_release_capacity_v1_sampling import sample_block
    law = controlled_law(64, law_index)
    profiles = outcome_profiles(law)
    observations, streams = sample_block(law, profiles, law_index, replication, preflight=preflight)
    N, T = counts(observations['cal_S'], observations['cal_A'], K=64)
    index = list(profiles).index('aligned')
    on, ot, ov = counts(observations['out_S'], observations['out_A'], observations['out_Y'][index], K=64)
    return {'N': N, 'T': T}, {'N': on, 'T': ot, 'V': ov}, streams


def prepare_computation(root, config=None):
    """Generate the full computational count-input panel without saved studies."""
    from experiments import jmlr_computation_v1 as study
    cfg = configuration('computation', config)
    root = _new_root(root, cfg)
    laws, trees, inputs, streams = {}, {}, [], {}
    primary_reps = range(cfg['replications'])
    for K in cfg['support_sizes']:
        trees[str(K)] = fixed_trees(K, cfg['design_master_seed'])
        for i in range(2):
            law = controlled_law(K, i)
            laws[f'{i}_{K}'] = law
            phases = [('primary', primary_reps)]
            if K in cfg['preflight']['support_sizes']:
                phases.append(('preflight', range(cfg['preflight']['replications_per_law'])))
            for phase, reps in phases:
                for j in reps:
                    case = f'{i}_K{K}_r{j:03d}'
                    if K == 64:
                        cal, outcome, seeds = release_bridge_counts(i, j, phase == 'preflight')
                        seedinfo = {'bridge': 'fresh regeneration of preserved SCA release streams', 'source_streams': seeds}
                    else:
                        cal, cs = study.draw_counts(law, i, j, int(phase == 'preflight'), 0, cfg['sampling_master_seed'])
                        outcome, ys = study.draw_counts(law, i, j, int(phase == 'preflight'), 1, cfg['sampling_master_seed'])
                        seedinfo = {'calibration': cs, 'outcome': ys}
                    repeated = None
                    if phase == 'primary' and K in cfg['repeat_support_sizes'] and j < cfg['repeat_replications']:
                        repeated, rs = study.draw_counts(law, i, j, 0, 2, cfg['sampling_master_seed'])
                        seedinfo['repeat_outcome'] = rs
                    value = {'version': cfg['version'], 'case_id': case, 'law_id': law['id'],
                             'law_index': i, 'K': K, 'replication': j, 'phase': phase,
                             **cal, 'immutable_ids': law['immutable_ids'], 'tree_orders': trees[str(K)],
                             'outcome': outcome, 'repeat_outcome': repeated}
                    study.validate_input(value)
                    relative = Path('inputs') / phase / (case + '.json.gz')
                    dump(root / relative, value)
                    inputs.append(str(relative))
                    streams[phase + '/' + case] = seedinfo
    primary = 2 * len(cfg['support_sizes']) * cfg['replications']
    capacities = 2 * len(set(cfg['support_sizes']) & set(cfg['capacity_support_sizes'])) * min(cfg['replications'], cfg['capacity_replications'])
    repeats = 2 * len(set(cfg['support_sizes']) & set(cfg['repeat_support_sizes'])) * min(cfg['replications'], cfg['repeat_replications'])
    preflight = 2 * len(set(cfg['support_sizes']) & set(cfg['preflight']['support_sizes'])) * cfg['preflight']['replications_per_law']
    dump(root / 'population_laws.json', {'laws': laws, 'tree_orders': trees,
         'purpose': 'input generation and untimed truth diagnostics only'})
    dump(root / 'bridge_expected.json', {})
    dump(root / 'frozen_protocol.json', {'version': cfg['version'], 'public_study': 'SCA computational scaling',
         'frozen_utc': _now(), 'configuration': cfg, 'inputs': inputs, 'streams': streams,
         'primary_cases': primary, 'capacity_cases': capacities, 'repeated_cohorts': repeats,
         'preflight_cases': preflight, 'input_generation_outside_timing': True,
         'bridge_scope': 'K64 counts regenerated from original release streams; no saved selection reference is imported',
         'law_construction': 'original exact stored-binary controlled-law generator and final residual',
         'tree_construction': 'PCG64(SeedSequence([20261008,101,K])) fixed permutation'})
    dump(root / 'manifest.json', {'version': cfg['version'], 'status': 'inputs_frozen',
         'public_study': 'SCA computational scaling', 'created_utc': _now(),
         'environment': study.environment(), 'input_bytes': sum((root / p).stat().st_size for p in inputs),
         'completed_primary': 0, 'completed_capacity': 0, 'completed_repeats': 0,
         'generated_from_source': True})
    return load(root / 'frozen_protocol.json')
