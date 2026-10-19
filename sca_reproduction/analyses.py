"""Portable downstream assignment analyses using newly generated artifacts.

This adapter supplies paths and frozen design metadata to the research cores;
it does not replace their certificates, search, stopping rules or statistics.
The workspace contains generated inputs/results only.  It never requires an
archive of the paper's original observations, proposals or completed outputs.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
import gzip
import json
from pathlib import Path
import resource
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
POLICY = dict(grid_bits=48,initial_precision=128,max_precision=1024,
              max_evaluations=192,endpoint_seconds=5.,max_trials=1000000)
RECOVERY_SIZES = (2000,4000,8000,16000,32000,64000,128000,256000,
                  512000,1000000,2000000,3000000)
CAPS = (F(5,4),F(3,2),F(2))
EFFICIENCY_LABELS = {
    'original':'SCA (Coupled CP)',
    'allocation_control':'Allocation-control ablation',
    'pooled_alternative':'Certificate-construction ablation: pooled assignment bounds'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load(path):
    from experiments.jmlr_verified_adaptive_v1_math import decode
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        return decode(json.load(stream))


def _dump(path,value):
    from experiments.jmlr_verified_adaptive_v1_math import encode
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with (gzip.open(temporary,'wt',compresslevel=3) if path.suffix=='.gz' else temporary.open('w')) as stream:
        json.dump(encode(value),stream,allow_nan=False,separators=(',',':'))
    temporary.replace(path)


def _merge(base,updates):
    result=deepcopy(base)
    for key,value in updates.items():
        result[key]=_merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else deepcopy(value)
    return result


def _defaults(study):
    if study != 'efficiency':
        raise ValueError('Expected efficiency configuration')
    return _load(PROJECT/'configs/jmlr_efficiency_gap_v1.json')


def _configuration(study,config=None):
    if isinstance(config,(str,Path)):
        config=_load(config)
    config={} if config is None else deepcopy(config)
    if not isinstance(config,dict):
        raise TypeError('Configuration must be a dictionary or a JSON path')
    # The public CLI accepts either a component override or a combined smoke
    # configuration.  Other studies' settings must not leak into this one.
    if isinstance(config.get('studies'),dict):
        config=config['studies']
    if study in config and isinstance(config[study],dict):
        config=config[study]
    elif any(key in config for key in ('assignment','release','precision','computation','outcome','efficiency')):
        config={}
    result=_merge(_defaults(study),config)
    result.setdefault('execution',{})['workers']=1
    result.update(output_dir='efficiency',historical_source='assignment',release_source='release')
    if tuple(F(c) for c in result['caps'])!=CAPS:
        raise ValueError('These analyses preserve the three declared rational distortion caps')
    if result['root_policy']!=POLICY:
        raise ValueError('Retained analyses preserve the complete historical root policy')
    if study=='efficiency':
        if result['recovery_root_policy']!=dict(POLICY,max_trials=4000000):
            raise ValueError('Recovery preserves its declared four-million trial policy')
        if (result['recovery']['master_seed']!=2026101603
                or result['recovery']['development_seed']!=202610160301
                or F(result['recovery'].get('theoretical_eta','1/4'))!=F(1,4)):
            raise ValueError('Recovery preserves the declared streams and reference overlap floor')
    return result


@lru_cache(maxsize=1)
def theoretical_reference():
    """Recompute analytical references, never read a manuscript result JSON.

    The integer crossing uses the retained manuscript diagnostic function.
    Its logarithms are descriptive floating evaluations, not selection rules.
    The KL necessary bound uses80-digit Decimal arithmetic.  Three million
    is the prespecified sufficient-illustration design point, not a dataset.
    """
    from experiments.jmlr_efficiency_gap_v1_core import manuscript_bound
    import math
    def applicable(n):
        bound=manuscript_bound(n,64,F(1,20),F(1,64),F(1,4),beta=F(1,20))
        return bound['b_n'] is not None and bound['b_n']<=math.log(1.5)
    lower,upper=1,1
    while not applicable(upper):
        upper*=2
    while lower<upper:
        middle=(lower+upper)//2
        if applicable(middle):upper=middle
        else:lower=middle+1
    with localcontext() as context:
        context.prec=80
        def kl(p,q):
            return p*(p/q).ln()+(1-p)*((1-p)/(1-q)).ln()
        necessary=kl(Decimal(9)/10,Decimal(1)/20)/(Decimal(1)/64*kl(Decimal(1)/2,Decimal(4)/13))
        necessary_ceiling=int(necessary.to_integral_value(rounding=ROUND_CEILING))
    return {'direct_diameter_threshold':lower,'necessary_lower_bound':necessary_ceiling,
        'necessary_bound_decimal':str(necessary),'sufficient_reference_n':3000000,
        'sufficient_reference_bound':manuscript_bound(3000000,64,F(1,20),F(1,64),F(1,4),beta=F(1,20)),
        'scope':'Analytical reference calculations, distinct from empirical recovery-grid crossing or sample-complexity necessity of this selector'}


def _trees(assignment_root,K=64):
    filename=Path(assignment_root)/'design.json'
    if filename.exists():
        design=_load(filename)
        tree_map=design.get('tree_orders',design.get('design',{}).get('tree_orders',{}))
        if str(K) in tree_map:
            return tree_map[str(K)]
    # Fixed design geometry is generated before observations; recovery can
    # therefore run independently of assignment-data generation.
    import numpy as np
    generator=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261008,101,K])))
    return [list(range(K)),generator.permutation(K).tolist()]


def _relative(path,workspace):
    return str(Path(path).resolve().relative_to(Path(workspace).resolve()))


def _frozen(root,protocol):
    filename=root/'frozen_protocol.json'
    if filename.exists():
        old=_load(filename)
        if old['configuration']!=protocol['configuration']:
            raise ValueError('Existing workspace uses another frozen configuration; use a separate workspace')
        if old.get('recovery_tree_orders')!=protocol.get('recovery_tree_orders'):
            raise ValueError('Fixed recovery trees changed after the workspace was initialized')
        protocol['frozen_utc']=old['frozen_utc']
    root.mkdir(parents=True,exist_ok=True)
    _dump(filename,protocol);_dump(root/'configuration.json',protocol['configuration'])
    return protocol


def prepare_efficiency(workspace,config=None):
    """Bind newly generated assignment/release artifacts, without running them."""
    workspace=Path(workspace).resolve();root=workspace/'efficiency';assignment=workspace/'assignment'
    if config is None and (root/'configuration.json').exists():
        cfg=_load(root/'configuration.json')
    else:
        cfg=_configuration('efficiency',config)
    paths=sorted((assignment/'evidence/large').glob('*/*.json.gz'))
    law_path=assignment/'exact_laws.json'
    laws={law['id']:law for law in _load(law_path)['large']} if law_path.exists() else {}
    release_path=workspace/'release/frozen_protocol.json'
    release=_load(release_path) if release_path.exists() else None
    protocol={'configuration':cfg,'frozen_utc':_now(),'historical_laws':laws,
        'historical_paths':[_relative(path,workspace) for path in paths],
        'historical_root':'assignment','release_root':'release','release_protocol':release,
        'recovery_tree_orders':_trees(assignment),'recovery_root_policy':cfg['recovery_root_policy'],
        'caps':tuple(F(c) for c in cfg['caps']),
        'theoretical_reference':theoretical_reference(),
        'scope':{'historical_datasets':4800,'historical_comparison_rows':43200,
            'release_blocks':300,'release_capacity_rows':4500,
            'independent_recovery_paths':cfg['recovery']['replications'],
            'recovery_rows':12*len(cfg['recovery']['sample_sizes'])*cfg['recovery']['replications']},
        'provenance':'All upstream numerical evidence is generated in this public workspace, not loaded from a paper-results archive.',
        'public_default':'SCA: full-sample Coupled CP, complete combined family',
        'procedure_labels':EFFICIENCY_LABELS,
        'interpretation':'Allocation and certificate-construction variants are ablations; population laws enter diagnostics only.'}
    return _frozen(root,protocol)


def _runtime_efficiency(workspace,protocol):
    runtime=dict(protocol);workspace=Path(workspace).resolve()
    runtime['historical_root']=str(workspace/protocol['historical_root'])
    runtime['release_root']=str(workspace/protocol['release_root'])
    runtime['historical_paths']=[str(workspace/path) for path in protocol['historical_paths']]
    return runtime


def _limit(tasks,limit):
    if limit is None:
        return tasks
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<0:
        raise ValueError('limit must be a nonnegative prefix length')
    return tasks[:limit]


def _verifier(protocol,recovery=False):
    from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy,RootVerifier
    policy=protocol['recovery_root_policy'] if recovery else protocol['configuration']['root_policy']
    return RootVerifier(RootPolicy(**policy))


def _trim(verifier,config):
    execution=config.get('execution',{})
    if len(verifier.cache)>execution.get('root_cache_trim_threshold',30000):
        retained=list(verifier.cache.items())[-execution.get('root_cache_entries',20000):]
        verifier.cache.clear();verifier.cache.update(retained)


def _write_case(root,path,record,task,started):
    if path.exists():
        raise FileExistsError('Completed generated records are never overwritten')
    record=dict(record,task=task,development=False,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
        worker_seconds_before_serialization=time.perf_counter()-started)
    tick=time.perf_counter();_dump(path,record)
    receipt={'task':task,'rows':len(record.get('rows',())),
        'seconds':time.perf_counter()-started,'serialization_seconds':time.perf_counter()-tick,
        'bytes':path.stat().st_size,'peak_rss_bytes':record['peak_rss_bytes']}
    folder=root/'panel';folder.mkdir(exist_ok=True)
    with (folder/'receipts.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt)+'\n')
    return receipt


def _efficiency_counts(root):
    return {kind:len(list((Path(root)/'panel'/kind).glob('*.json.gz')))
            for kind in ('historical','release','recovery')}


def run_efficiency(workspace,limit=None,config=None):
    """Run historical + release assignment diagnostics; recovery is separate."""
    from experiments.jmlr_efficiency_gap_v1_historical import process_historical,process_release
    workspace=Path(workspace).resolve();root=workspace/'efficiency'
    frozen=prepare_efficiency(workspace,config);protocol=_runtime_efficiency(workspace,frozen)
    historical=[Path(path) for path in protocol['historical_paths']]
    release_paths=sorted((workspace/'release/panel/blocks').glob('*.json.gz'))
    if limit is None and (len(historical)!=4800 or len(release_paths)!=300):
        raise FileNotFoundError('Full efficiency analysis requires fresh run assignment (4800 records) and run release (300 blocks) first')
    tasks=[('historical',path) for path in historical]+[('release',path) for path in release_paths]
    verifier=_verifier(protocol);receipts=[]
    for kind,source in _limit(tasks,limit):
        name=(source.parent.name+'__'+source.name) if kind=='historical' else source.name
        output=root/'panel'/kind/name
        if output.exists():
            continue
        _trim(verifier,protocol['configuration']);started=time.perf_counter()
        if kind=='historical':
            record=process_historical(source,protocol,verifier=verifier)
            task=(kind,_relative(source,workspace))
        else:
            law_index,replication=map(int,source.name.removesuffix('.json.gz').split('_'))
            record=process_release(law_index,replication,protocol,verifier=verifier)
            task=(kind,law_index,replication)
        receipts.append(_write_case(root,output,record,task,started))
    result={'study':'efficiency','status':'incomplete' if limit is not None else 'assignment_components_complete',
        'new_records':len(receipts),'completed':_efficiency_counts(root),
        'remaining_component':'Run recovery before the full efficiency report',
        'scope':'SCA plus descriptive allocation/certificate-construction ablations; assignment-only'}
    _dump(root/'execution_summary.json',result)
    return result


def run_recovery(workspace,limit=None,config=None):
    """Generate independent exact nested count paths; no upstream data needed."""
    from experiments.jmlr_efficiency_gap_v1_recovery import process_recovery
    workspace=Path(workspace).resolve();root=workspace/'efficiency'
    protocol=prepare_efficiency(workspace,config);verifier=_verifier(protocol,recovery=True)
    wanted=_limit(list(range(protocol['configuration']['recovery']['replications'])),limit);written=0
    for replication in wanted:
        output=root/'panel/recovery'/f'{replication:03d}.json.gz'
        if output.exists():
            continue
        _trim(verifier,protocol['configuration']);started=time.perf_counter()
        record=process_recovery(replication,protocol,development=False,verifier=verifier)
        _write_case(root,output,record,('recovery',replication),started);written+=1
    result={'study':'recovery','status':'incomplete' if limit is not None else 'complete',
        'new_records':written,'completed_paths':_efficiency_counts(root)['recovery'],
        'expected_paths':protocol['configuration']['recovery']['replications'],
        'scope':'Prespecified full-support certificate diagnostic; not a complete partition-search study'}
    _dump(root/'recovery_execution_summary.json',result)
    return result






def run(workspace,study,limit=None,config=None):
    functions={'efficiency':run_efficiency,'recovery':run_recovery}
    if study not in functions:
        raise ValueError('Expected efficiency or recovery')
    return functions[study](workspace,limit=limit,config=config)


def _partial_report(root,study,paths):
    """Execution evidence for bounded smoke; no full-grid statistical claim."""
    from experiments.jmlr_efficiency_gap_v1_reporting import _csv
    from experiments.jmlr_verified_adaptive_v1_math import encode
    counts=Counter();rows=[]
    labels=EFFICIENCY_LABELS
    for path in paths:
        record=_load(path);kind=record.get('task',['assignment'])[0];counts[kind]+=1
        for row in record.get('rows',()):
            copied=dict(row,evidence_file=str(path.relative_to(root)))
            copied['procedure_label']=labels.get(row.get('procedure'),row.get('procedure'))
            copied={key:json.dumps(encode(value),sort_keys=True) if isinstance(value,(dict,list,tuple))
                    else value for key,value in copied.items()}
            rows.append(copied)
    folder=root/'tables';folder.mkdir(parents=True,exist_ok=True)
    _csv(folder/'partial_execution_rows.csv',rows)
    summary={'study':study,'status':'incomplete','processed_records':sum(counts.values()),
        'record_counts':dict(counts),'rows':len(rows),
        'scope':'Partial execution diagnostic only; not full-paper reproduction or a full-grid uncertainty summary'}
    _dump(root/('recovery_partial_summary.json' if study=='recovery' else 'partial_summary.json'),summary)
    return summary


def _recovery_report(root):
    """Standalone full recovery statistics, reusing the original paired/CP code."""
    from experiments import jmlr_efficiency_gap_v1_reporting as saved
    root=Path(root);protocol=_load(root/'frozen_protocol.json');paths=sorted((root/'panel/recovery').glob('*.json.gz'))
    rows=[];seen=set()
    expected={(law,n,mode,procedure) for law in ('homogeneous','one_exceptional_state')
        for n in RECOVERY_SIZES for mode in ('known_mass','unknown_mass') for procedure in EFFICIENCY_LABELS}
    for path in paths:
        block=_load(path)
        replication=int(path.name.removesuffix('.json.gz'))
        actual={(row['law_id'],row['n'],row['mass_mode'],row['procedure']) for row in block['rows']}
        if replication in seen or not 0<=replication<300 or actual!=expected or len(block['rows'])!=len(expected):
            raise AssertionError('Complete unique recovery paths and all prespecified conditions are required')
        if block.get('development') or any(row['replication']!=replication for row in block['rows']):
            raise AssertionError('Development or mismatched path cannot enter the recovery panel')
        seen.add(replication)
        rows.extend(saved._enrich(row,block,'recovery',str(path.relative_to(root))) for row in block['rows'])
    if seen!=set(range(300)):
        raise AssertionError('All300 independent recovery paths are required')
    summaries=[]
    for key,condition in sorted(saved._groups(rows,saved.RECOVERY_KEYS).items()):
        summaries.append(dict(zip(saved.RECOVERY_KEYS,key))|saved._condition_summary(condition,300))
    paired=[]
    for key,condition in sorted(saved._groups(rows,('law_id','n','mass_mode')).items()):
        methods=saved._groups(condition,('procedure',))
        for left,right in (('allocation_control','original'),('pooled_alternative','allocation_control')):
            for metric in saved.PAIR_METRICS['recovery']:
                paired.append(dict(zip(('law_id','n','mass_mode'),key))|
                    {'left_procedure':left,'right_procedure':right}|
                    saved._paired(methods[(left,)],methods[(right,)],metric,300))
    destination=root/'recovery_report';(destination/'tables').mkdir(parents=True,exist_ok=True)
    for name,data in (('replications',rows),('condition_summary',summaries),
                      ('paired_comparisons',paired),('first_grid_crossings',saved._first_crossings(summaries))):
        saved._csv(destination/'tables'/(name+'.csv'),data)
    summary={'study':'recovery','status':'complete','independent_paths':300,'condition_count':len(summaries),
        'rows':len(rows),'procedure_labels':EFFICIENCY_LABELS,'theoretical_reference':protocol['theoretical_reference'],
        'uncertainty':'Pointwise two-sided95% CP for individual binary rates; mean +/-1.96SE for continuous metrics and paired independent-path differences',
        'scope':'Full-support certificate diagnostic, with full fixed pooled-collection error allocation; no complete partition search or exact sample-complexity claim'}
    _dump(destination/'summary.json',summary)
    return summary


def report(workspace,study):
    """Dispatch complete original statistics; bounded prefixes stay explicit."""
    workspace=Path(workspace).resolve()
    if study not in ('efficiency','recovery'):
        raise ValueError('Expected efficiency or recovery')
    root=workspace/'efficiency';counts=_efficiency_counts(root)
    if study=='efficiency' and counts=={'historical':4800,'release':300,'recovery':300}:
        from experiments.jmlr_efficiency_gap_v1_reporting import report as full_report
        prepare_efficiency(workspace)
        return full_report(root)
    if study=='recovery' and counts['recovery']==300:
        protocol=_load(root/'frozen_protocol.json')
        if tuple(protocol['configuration']['recovery']['sample_sizes'])==RECOVERY_SIZES:
            return _recovery_report(root)
    paths=sorted((root/'panel/recovery').glob('*.json.gz')) if study=='recovery' else sorted((root/'panel').glob('*/*.json.gz'))
    return _partial_report(root,study,paths)
