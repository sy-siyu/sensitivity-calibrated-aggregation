"""Generators and numerical prerequisites for the assignment studies.

SCA is the full-sample Coupled CP combined-family procedure. Independent CP,
known masses, restricted families and split procedures are comparison controls
or ablations. LTT is the statistical baseline. Compatibility fields preserve
the numerical modules' data contracts.

The generator uses the original NumPy multinomial/binomial calls and PCG64
streams. Exact nominal laws retain the first K-1 stored binary masses and set
the final mass to the exact residual. This declares the input-law convention;
it does not certify the PRNG or all floating-point sampler internals.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from fractions import Fraction as F
import gzip
import json
import math
import numbers
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.special import expit

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy
from experiments.jmlr_verified_pipeline_v1_laws import nominal_law,TruthEvaluator
from experiments.jmlr_verified_pipeline_v1_core import evaluate_large
from experiments import jmlr_statistical_baselines_v1_core as split_core


VERSION='sca_assignment_reproduction_v1'
MASTER_SEED=20261008
SUPPORTS=(16,64)
R_VALUES=(250,1000,4000)
CAPS=(F(5,4),F(3,2),F(2))
PILOT_FRACTIONS=('1/5','1/2')
PAPER_FRACTION='1/5'
ALPHA=F(1,20)


def _read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:return exact.decode(json.load(stream))
    return exact.decode(json.loads(path.read_text()))


def _write(path,value,*,immutable=False):
    path=Path(path);payload=json.dumps(exact.encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if immutable and path.exists():
        if exact.encode(_read(path))!=exact.encode(value):
            raise ValueError('Existing generated input/configuration differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    if path.suffix=='.gz':
        with temporary.open('wb') as stream:
            with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,compresslevel=5) as compressed:compressed.write(payload)
    else:temporary.write_bytes(payload+b'\n')
    temporary.replace(path)


def _settings(config=None):
    config={} if config is None else config
    config=config.get('assignment',config)
    policy=dict(config.get('root_policy',RootPolicy().__dict__))
    if policy!=RootPolicy().__dict__:
        raise ValueError('Assignment reproduction retains the historical complete root policy')
    frozen={'master_seed':MASTER_SEED,'support_sizes':list(SUPPORTS),
        'mass_profiles':['balanced','skewed'],'overlaps':['moderate','weak'],
        'geometries':['ordered','permuted'],'observations_per_state':list(R_VALUES),
        'replications':100,'alpha':'1/20','caps':['5/4','3/2','2'],
        'pilot_fractions':list(PILOT_FRACTIONS),'frozen_fraction':PAPER_FRACTION,
        'development_replications':[0,10],'evaluation_replications':[10,100],
        'root_policy':policy}
    for key,value in frozen.items():
        if key in config and config[key]!=value:
            raise ValueError('Retained assignment design differs: '+key)
    return frozen


def _integer(value,name,maximum):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or not 0<=value<maximum:
        raise ValueError(name+' is outside the retained grid')
    return int(value)


def build_design():
    """Recreate the 16 original generator-input vectors and two fixed trees."""
    populations=[];trees={};permutations={};laws=[]
    for K in SUPPORTS:
        tree=np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER_SEED,101,K]))).permutation(K)
        permutation=np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER_SEED,102,K]))).permutation(K)
        trees[str(K)]=[list(range(K)),tree.tolist()];permutations[str(K)]=permutation.tolist()
        z=2*np.arange(K)/(K-1)-1
        for mi,mass in enumerate(('balanced','skewed')):
            p0=np.ones(K) if mass=='balanced' else (np.arange(K)+1.)**(-1.5)
            p0=p0/p0.sum()
            for oi,overlap in enumerate(('moderate','weak')):
                e0=expit(math.log(3 if overlap=='moderate' else 19)*z)
                for gi,geometry in enumerate(('ordered','permuted')):
                    indices=np.arange(K) if geometry=='ordered' else permutation
                    p,e=p0[indices].copy(),e0[indices].copy()
                    population={'id':f'K{K}_{mass}_{overlap}_{geometry}','K':K,'mass':mass,
                        'overlap':overlap,'geometry':geometry,'mass_index':mi,'overlap_index':oi,
                        'geometry_index':gi,'p':p.tolist(),'e':e.tolist(),'effective_mass_sum':float(p.sum())}
                    populations.append(population)
                    laws.append(nominal_law(population,'generator:sca_reproduction.assignment.build_design;seed=20261008'))
    return {'version':VERSION,'populations':populations,'tree_orders':trees,
        'population_permutations':permutations,'r_values':list(R_VALUES),'laws':laws,
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
        'sampling_scope':'Original NumPy multinomial then binomial from one PCG64 stream; declared exact nominal input laws, not a PRNG/sampler proof'}


def case_tasks(phase='all'):
    if phase not in ('all','development','evaluation'):
        raise ValueError('Unknown assignment phase')
    phases=('development','evaluation') if phase=='all' else (phase,)
    for current in phases:
        for pi in range(16):
            for ri in range(3):
                for replication in range(10) if current=='development' else range(10,100):
                    yield pi,ri,replication


def draw_case(population_index,r_index,replication,design=None):
    """Generate sufficient counts and the original nested fixed-size splits."""
    pi=_integer(population_index,'Population index',16)
    ri=_integer(r_index,'Sample-size index',3);rep=_integer(replication,'Replication',100)
    design=build_design() if design is None else design;pop=design['populations'][pi];K=pop['K']
    seed=[MASTER_SEED,300,K,pop['mass_index'],pop['overlap_index'],pop['geometry_index'],ri,rep]
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    N=tuple(map(int,rng.multinomial(K*R_VALUES[ri],pop['p'])))
    T=tuple(map(int,rng.binomial(N,pop['e'])))
    ids=tuple(range(K))
    metadata={key:pop[key] for key in ('id','K','mass','overlap','geometry')}
    metadata.update(r=R_VALUES[ri],n=K*R_VALUES[ri],population_index=pi,r_index=ri,replication=rep)
    return {'version':VERSION,'phase':'development' if rep<10 else 'evaluation','metadata':metadata,
        'N':N,'T':T,'immutable_ids':ids,'assignment_seed':seed,'generator':'PCG64(SeedSequence)',
        'nested_splits':split_core.nested_splits(N,T,pi,ri,rep,immutable_ids=ids)}


def relative_case(metadata):
    return Path(metadata['id']+'_r'+str(metadata['r']))/('rep%03d.json.gz'%metadata['replication'])


def _truth_rows(rows,law):
    evaluator=TruthEvaluator(law);answer=[]
    for source in rows:
        row=dict(source);partition=exact.validate_partition(row['partition'],law['K'])
        value=evaluator.partition(partition)['value']
        row.update(partition=partition,cells=len(partition),identity=len(partition)==law['K'],
            compression=F(law['K']-len(partition),law['K']),
            truth={'exact_value':value,'strict_cap_violation':value>row['cap'],'ambiguous':False})
        answer.append(row)
    return answer


def evaluate_case(input_record,design,*,root_policy=None,fraction_choice_established=False):
    """Recompute full SCA, controls/oracles and split LTT/held-out controls.

    The return is (full_artifact, split_artifact). The legacy-format wrappers
    feed the retained efficiency and split adapters, but all numerical evidence
    is generated by this call. No source archive is read. Pilot scores remain
    untrusted floats and infinite scores remain present as explicit None.
    """
    metadata=dict(input_record['metadata']);pi=metadata['population_index'];K=metadata['K']
    N,T=tuple(input_record['N']),tuple(input_record['T']);ids=tuple(input_record['immutable_ids'])
    if (len(N)!=K or len(T)!=K or sum(N)!=metadata['n'] or metadata['r']*K!=sum(N)
            or input_record['phase']!=('development' if metadata['replication']<10 else 'evaluation')):
        raise ValueError('Input counts, phase or metadata differ')
    policy=RootPolicy().__dict__ if root_policy is None else dict(root_policy)
    if policy!=RootPolicy().__dict__:raise ValueError('Retained assignment root policy differs')
    law=design['laws'][pi];trees=design['tree_orders'][str(K)]
    if law['id']!=metadata['id'] or tuple(law['immutable_ids'])!=ids:
        raise ValueError('Input law/state identifiers differ')
    started=time.perf_counter()
    full_core=evaluate_large(N,T,law,trees,immutable_ids=ids,root_policy=policy)
    reference=[dict(row,procedure='full_reference') for mode in full_core['modes'] if mode['mode']=='unknown_mass'
               for row in mode['labels'] if row['method']=='coupled' and row['family']=='combined']
    phase=input_record['phase'];fractions=PILOT_FRACTIONS if phase=='development' else (PAPER_FRACTION,)
    splits=[]
    for fraction in fractions:
        split=dict(input_record['nested_splits'][fraction])
        core=split_core.evaluate_split(split['pilot_N'],split['pilot_T'],split['heldout_N'],split['heldout_T'],
            trees,caps=CAPS,alpha=ALPHA,root_policy=policy,immutable_ids=ids)
        rows=_truth_rows([dict(row,procedure='split_ltt' if row['method']=='ltt' else 'split_coupled',fraction=fraction)
                          for row in core['selections']],law)
        splits.append(dict(split,fraction=fraction,core=core,rows=rows))
    split_artifact={'version':'jmlr_statistical_baselines_v1','reproduction_version':VERSION,
        'generation':'generated observations, split/proposal/certificate calculations',
        'phase':phase,'metadata':metadata,'N':N,'T':T,'immutable_ids':ids,
        'reference':_truth_rows(reference,law),'splits':splits}
    # These aliases reproduce the old adapter schema. During development the
    # named paper fraction is a compatibility parameter, not a new tuning result.
    baseline={'version':'sca_fresh_assignment_baseline_linkage_v1','phase':phase,'metadata':metadata,
        'frozen_fraction':PAPER_FRACTION,'fraction_choice_established':bool(fraction_choice_established),
        'fraction_scope':'Recomputed complete independent development choice' if fraction_choice_established else
                         'Declared paper fraction for development/bounded integration; no completed tuning claim',
        'splits':[{'fraction':s['fraction'],'source_split_index':i,'rows':s['rows']} for i,s in enumerate(splits)],
        'reference':reference,'source_inference_recomputed':True,'tuning_repeated':bool(fraction_choice_established)}
    full_artifact={'version':'jmlr_verified_pipeline_v1','reproduction_version':VERSION,'kind':'large',
        'metadata':metadata,'law_id':law['id'],'source':'fresh_generator','core':full_core,'baseline':baseline,
        'compute_seconds':time.perf_counter()-started,'historical_evidence_imported':False}
    return full_artifact,split_artifact


def _root(workspace):return Path(workspace)/'assignment'


def prepare(workspace,config=None,smoke=False,limit=None):
    """Generate count inputs only; a bounded prefix never changes the design."""
    settings=_settings(config);root=_root(workspace);design=build_design()
    _write(root/'configuration.json',settings,immutable=True)
    _write(root/'design.json',design,immutable=True)
    _write(root/'exact_laws.json',{'version':VERSION,'large':design['laws']},immutable=True)
    bound=1 if smoke and limit is None else limit
    if bound is not None and (isinstance(bound,bool) or not isinstance(bound,int) or bound<1):
        raise ValueError('A bounded prefix limit must be a positive integer')
    generated=0
    for index,task in enumerate(case_tasks()):
        if bound is not None and index>=bound:break
        record=draw_case(*task,design);path=root/'inputs'/relative_case(record['metadata'])
        _write(path,record,immutable=True);generated+=1
    count=len(list((root/'inputs').glob('*/rep*.json.gz')))
    summary={'version':VERSION,'prepared_cases':count,'expected_cases':4800,'complete':count==4800,
             'bounded_prefix':bound,'generated_or_checked_this_call':generated,
             'input_generation_scope':'Original finite-count RNG and nested splits; no numerical confidence or selection work'}
    _write(root/'inputs_manifest.json',summary);return summary


def freeze_fraction(workspace):
    """Recompute the original global development rule before evaluation."""
    root=_root(workspace);design=_read(root/'design.json')
    totals={fraction:F() for fraction in PILOT_FRACTIONS};counts={fraction:0 for fraction in PILOT_FRACTIONS}
    for pi,ri,rep in case_tasks('development'):
        pop=design['populations'][pi]
        metadata={'id':pop['id'],'r':R_VALUES[ri],'replication':rep}
        path=root/'splits/evidence/development'/relative_case(metadata)
        if not path.exists():return {'complete':False,'selected_fraction':None,'datasets_per_fraction':counts}
        source=_read(path);K=source['metadata']['K']
        for split in source['splits']:
            fraction=split['fraction']
            row=[r for r in split['rows'] if r['procedure']=='split_ltt' and r['cap']==F(3,2)]
            if len(row)!=1:raise ValueError('Development LTT label grid differs')
            totals[fraction]+=F(K-row[0]['cells'],K);counts[fraction]+=1
    if any(counts[fraction]!=480 for fraction in PILOT_FRACTIONS):raise ValueError('Incomplete development design')
    means={fraction:totals[fraction]/480 for fraction in PILOT_FRACTIONS}
    chosen='1/5' if means['1/5']>=means['1/2'] else '1/2'
    result={'complete':True,'selected_fraction':chosen,'paper_fraction':PAPER_FRACTION,
        'matches_paper_fraction':chosen==PAPER_FRACTION,'mean_ltt_compression':means,'datasets_per_fraction':counts,
        'criterion':'Equal fixed-condition weight at cap3/2; exact ties choose1/5; development0..9 only',
        'evaluation_data_used':False}
    _write(root/'fraction_selection.json',result,immutable=True)
    return result


def run(workspace,config=None,limit=None):
    """Recompute the retained grid or an explicitly incomplete bounded prefix."""
    settings=_settings(config);prepare(workspace,config,limit=limit)
    root=_root(workspace);design=_read(root/'design.json');completed_new=0;failed=[];seen=0;frozen=None
    for index,task in enumerate(case_tasks()):
        if limit is not None and index>=limit:break
        pi,ri,rep=task;pop=design['populations'][pi]
        relative=relative_case({'id':pop['id'],'r':R_VALUES[ri],'replication':rep})
        full_path=root/'evidence/large'/relative;phase='development' if rep<10 else 'evaluation'
        split_path=root/'splits/evidence'/phase/relative
        if phase=='evaluation' and frozen is None:
            frozen=freeze_fraction(workspace)
            if not frozen['complete'] or not frozen['matches_paper_fraction']:
                raise RuntimeError('Complete development did not reproduce the paper fraction; inspect fraction_selection.json and the numerical environment')
        if full_path.exists() and split_path.exists():seen+=1;continue
        if full_path.exists() or split_path.exists():
            raise RuntimeError('Incomplete generated artifact pair; preserve/repair this workspace before resuming: '+str(relative))
        try:
            inputs=_read(root/'inputs'/relative)
            full,split=evaluate_case(inputs,design,root_policy=settings['root_policy'],fraction_choice_established=phase=='evaluation')
            _write(split_path,split);_write(full_path,full);completed_new+=1;seen+=1
        except Exception as error:
            failure={'task':task,'relative_case':str(relative),'error':type(error).__name__+': '+str(error)}
            failed.append(failure);_write(root/'failures'/relative,failure)
    if frozen is None:frozen=freeze_fraction(workspace)
    available=len(list((root/'evidence/large').glob('*/rep*.json.gz')))
    summary={'version':VERSION,'status':'complete' if available==4800 and not failed else 'incomplete',
        'complete':available==4800 and not failed,'completed_cases':available,'expected_cases':4800,
        'completed_new':completed_new,'checked_prefix_cases':seen,'bounded_prefix_limit':limit,
        'fraction_selection':frozen,'failures':failed,
        'numerical_scope':'Verified calculations on generated counts'}
    _write(root/'run_manifest.json',summary);return summary


def report(workspace,config=None):
    """Create tables from newly generated outputs, without embedded results."""
    _settings(config)
    from experiments.jmlr_statistical_baselines_v1_reporting import paired,rate
    root=_root(workspace);paths=sorted((root/'evidence/large').glob('*/rep*.json.gz'))
    tables=root/'tables';tables.mkdir(parents=True,exist_ok=True);groups=defaultdict(list)
    fields=('id','K','mass','overlap','geometry','r','n','population_index','r_index','replication',
            'mode','method','family','cap','count','certificate','true_distortion','cap_violation',
            'compression','oracle_count','oracle_gap','identity','raw_law_covered')
    with (tables/'assignment_rows.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for path in paths:
            record=_read(path);meta=record['metadata']
            for mode in record['core']['modes']:
                for label in mode['labels']:
                    row={**meta,**label,'raw_law_covered':mode['diagnostics']['raw_law_covered']}
                    writer.writerow({key:row[key] for key in fields})
                    groups[(meta['id'],meta['r'],label['mode'],label['method'],label['family'],label['cap'])].append(row)
    conditions=[]
    for (law,r,mode,method,family,cap),rows in sorted(groups.items()):
        result={'law_id':law,'r':r,'mode':mode,'method':method,'family':family,'cap':cap,'datasets':len(rows)}
        for metric in ('count','compression','oracle_gap'):
            result.update({metric+'_'+key:value for key,value in paired([float(row[metric]) for row in rows]).items()})
        for metric in ('cap_violation','raw_law_covered','identity'):
            result.update({metric+'_'+key:value for key,value in rate([row[metric] for row in rows]).items()})
        conditions.append(result)
    if conditions:
        with (tables/'assignment_conditions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(conditions[0]));writer.writeheader();writer.writerows(conditions)
    summary={'version':VERSION,'complete':len(paths)==4800,'independent_datasets':len(paths),
        'expected_datasets':4800,'condition_method_cap_rows':len(conditions),
        'principal_procedure':'SCA: unknown masses, full-sample Coupled CP, complete combined family',
        'uncertainty':'Pointwise within-condition Student95 means and two-sided CP95 binary rates; repeated labels do not add datasets',
        'tables':['tables/assignment_rows.csv','tables/assignment_conditions.csv']}
    _write(root/'summary.json',summary);return summary
