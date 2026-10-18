"""Nested sufficient-count recovery paths for the efficiency-gap study.

The benchmark laws are exact: p_i=1/64, homogeneous e_i=1/2, and a second
law changes only e_0 to4/13. Non-dyadic4/13 assignments use unbiased bounded
integers in{0,...,12}, with success iff the integer is below4. Millions of
individual records are never materialized. State-count and half-action
increments are shared across laws; independent path replicates have disjoint
streams. All procedures and mass modes use each path's same sufficient counts.
"""
from __future__ import annotations

from fractions import Fraction as F
import math
import resource
import sys
import time

import numpy as np

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_laws import CONVENTION, TruthEvaluator


RECOVERY_K = 64
RECOVERY_SIZES = (2000,4000,8000,16000,32000,64000,128000,256000,512000,
                  1000000,2000000,3000000)
RECOVERY_LAWS = ('homogeneous','one_exceptional_state')
PROCEDURES = ('original','allocation_control','pooled_alternative')
MASS_MODES = ('known_mass','unknown_mass')
RECOVERY_MASTER_SEED = 2026101603
RECOVERY_DEVELOPMENT_SEED = 202610160301
STREAM_COMPONENTS = ('state_count_increments','half_action_increments',
                     'exceptional_rational_action_increments')


def recovery_laws():
    """Known benchmark laws, with no inherited binary residual adaptation."""
    answer = {}
    for name in RECOVERY_LAWS:
        e = (F(1,2),)*RECOVERY_K
        if name=='one_exceptional_state':
            e = (F(4,13),)+e[1:]
        answer[name] = {
            'id':name,'K':RECOVERY_K,'p':(F(1,RECOVERY_K),)*RECOVERY_K,'e':e,
            'immutable_ids':tuple(range(RECOVERY_K)),
            'source_identity':'Prespecified exact rational recovery benchmark',
            'convention':CONVENTION,
            'exact_input_convention':'Direct exact rational p_i=1/64; e_i=1/2 except declared e_0=4/13',
            'legacy_convention_qualification':'Validator compatibility marker only; no residual adaptation',
        }
    return answer


def recovery_streams(replication,development=False,master_seed=RECOVERY_MASTER_SEED,
                     development_seed=RECOVERY_DEVELOPMENT_SEED):
    if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<0
           for x in (replication,master_seed,development_seed)):
        raise ValueError('Recovery stream coordinates must be nonnegative integers')
    if master_seed==development_seed:
        raise ValueError('Development and evaluation streams must be disjoint')
    seed = int(development_seed if development else master_seed)
    return {name:[seed,int(replication),component] for component,name in enumerate(STREAM_COMPONENTS)}


def nested_count_paths(replication,sample_sizes=RECOVERY_SIZES,
                       master_seed=RECOVERY_MASTER_SEED,
                       development_seed=RECOVERY_DEVELOPMENT_SEED,development=False):
    """Draw independent increments and return both laws' nested count paths.

    Uniform multinomial state probabilities and binomial1/2 inputs are exact
    binary values. Only the exceptional state's occupancy needs non-dyadic
    Bernoulli draws; its integer sampler uses rejection internally rather than
    an approximate floating threshold. NumPy/PCG64 is a reproducible numerical
    implementation, not a formal verification of its random-generator internals.
    """
    sizes = tuple(sample_sizes)
    if (not sizes or any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<=0 for n in sizes)
            or tuple(sorted(set(sizes)))!=sizes):
        raise ValueError('Recovery sample sizes must be distinct increasing positive integers')
    streams = recovery_streams(replication,development,master_seed,development_seed)
    rng = {name:np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
           for name,key in streams.items()}
    N = np.zeros(RECOVERY_K,dtype=np.int64)
    homogeneous = np.zeros(RECOVERY_K,dtype=np.int64)
    alternative = np.zeros(RECOVERY_K,dtype=np.int64)
    paths = {name:[] for name in RECOVERY_LAWS}
    increments = [];previous=0
    probabilities = np.full(RECOVERY_K,1/RECOVERY_K,dtype=np.float64)
    for n in sizes:
        increment = int(n)-previous
        ni = rng['state_count_increments'].multinomial(increment,probabilities)
        ti = rng['half_action_increments'].binomial(ni,.5)
        exceptional = rng['exceptional_rational_action_increments'].integers(
            0,13,size=int(ni[0]),dtype=np.uint8)
        alt0 = int(np.count_nonzero(exceptional<4))
        N += ni; homogeneous += ti; alternative += ti
        alternative[0] += alt0-int(ti[0])
        for name,T in zip(RECOVERY_LAWS,(homogeneous,alternative)):
            paths[name].append({'n':int(n),'N':tuple(map(int,N)),'T':tuple(map(int,T))})
        increments.append({'n':int(n),'increment':increment,'N':tuple(map(int,ni)),
                           'T_homogeneous':tuple(map(int,ti)),'alternative_state0_treated':alt0})
        previous=int(n)
    return {'replication':int(replication),'development':bool(development),
            'streams':streams,'sample_sizes':tuple(map(int,sizes)),
            'paths':{name:tuple(values) for name,values in paths.items()},
            'increments':tuple(increments),
            'sampling':{
                'states':'Uniform64-state multinomial sufficient-count increments, exact binary1/64 input',
                'homogeneous_actions':'Binomial(N_i_increment,1/2) for every state',
                'exceptional_actions':'Uniform integer in0,...,12, success below4; only state0 occupancy drawn',
                'coupling':'Both laws share N and all half-action increments; alternative replaces only state0 treated increments',
                'nestedness':'Independent increments accumulated at the prespecified size grid',
                'qualification':'Declared exact nominal law and rational integer sampling; no formal PRNG iid or NumPy implementation theorem',
            }}


def sample_recovery_path(replication,protocol,development=False):
    config = protocol.get('configuration',protocol)['recovery']
    return nested_count_paths(replication,config['sample_sizes'],config['master_seed'],
                              config['development_seed'],development)


def _scalars(value,prefix=''):
    answer = {}
    if not isinstance(value,dict):
        return answer
    for key,item in value.items():
        name = f'{prefix}{key}'
        if isinstance(item,dict):
            answer.update(_scalars(item,name+'_'))
        elif item is None or isinstance(item,(str,bool,int,float,F)):
            answer[name] = item
    return answer


def _root_summary(regions,pooled_root):
    roots = [pooled_root]
    for region in regions:
        roots.extend(region['roots']['propensity'])
        roots.extend(region['roots']['mass'] or ())
    precision = 0
    def visit(value):
        nonlocal precision
        if isinstance(value,dict):
            if isinstance(value.get('precision'),int):
                precision=max(precision,value['precision'])
            for item in value.values():
                if isinstance(item,(dict,list,tuple)):
                    visit(item)
        elif isinstance(value,(list,tuple)):
            for item in value:
                visit(item)
    visit(roots)
    return {'root_intervals':len(roots),'root_fallbacks':sum(r['status']=='fallback' for r in roots),
            'maximum_precision_used':precision}


def process_recovery(replication,protocol,development=False,verifier=None):
    """Evaluate the full-support cell only; return a replayable144-row record.

    The pooled procedure keeps its1/100 total pooled error allocated across
    the entire fixed collection of nonsingleton tree nodes, but constructs only
    the full-support interval used by this recovery diagnostic. It never gives
    the full pooled error budget to that one cell. No partition search occurs.
    """
    from experiments.jmlr_efficiency_gap_v1_core import (
        pooled_groups, pooled_cell_bound, realized_radii, manuscript_diagnostics,
        manuscript_bound, build_regions)
    started=time.perf_counter()
    config=protocol['configuration'];recovery=config['recovery']
    if development and replication!=0:
        raise ValueError('The prespecified recovery development process uses replication0 only')
    if not development and not 0<=replication<recovery['replications']:
        raise ValueError('Recovery replication is outside the committed panel')
    policy_data=protocol.get('recovery_root_policy',config['root_policy'])
    policy=RootPolicy(**policy_data)
    if policy.max_trials<max(recovery['sample_sizes']):
        raise ValueError('Root policy trial budget does not cover the complete recovery size grid')
    if verifier is None:
        verifier=RootVerifier(policy)
    elif verifier.policy!=policy:
        raise ValueError('Shared recovery verifier does not match the frozen root policy')
    cache_before=(verifier.cache_hits,verifier.cache_misses,len(verifier.cache))
    tick=time.perf_counter();draws=sample_recovery_path(replication,protocol,development)
    sampling_seconds=time.perf_counter()-tick
    laws=recovery_laws();cell=tuple(range(RECOVERY_K))
    trees=protocol['recovery_tree_orders']
    groups=pooled_groups(trees)
    if cell not in groups:
        raise AssertionError('Full-support diagnostic cell is absent from the fixed pooled collection')
    alpha=F(1,20);alpha_raw=F(1,25);alpha_pooled=F(1,100)
    gamma_pooled=alpha_pooled/len(groups)
    reference_eta=F(recovery.get('theoretical_eta','1/4'))
    rows=[];evidence=[];confidence_seconds=0.;certification_seconds=0.;diagnostic_seconds=0.
    for law_name,law in laws.items():
        if not 0<reference_eta<=min(min(law['e']),min(1-e for e in law['e'])):
            raise ValueError('Prespecified manuscript reference overlap is invalid for the recovery law')
        true=TruthEvaluator(law).cell(cell)['value']
        for observation in draws['paths'][law_name]:
            N,T,n=observation['N'],observation['T'],observation['n']
            for mode in MASS_MODES:
                masses=law['p'] if mode=='known_mass' else None
                tick=time.perf_counter()
                original=exact.confidence_region(N,T,alpha,known_masses=masses,verifier=verifier)
                constructed=build_regions(N,T,alpha,trees,verifier=verifier,
                                          known_masses=masses,pooled_cells=(cell,))
                controlled=constructed['raw_region']
                pooled_root=constructed['pooled_intervals'][cell]
                if (constructed['gamma_pooled']!=gamma_pooled or constructed['alpha_raw']!=alpha_raw
                        or tuple(constructed['computed_pooled_groups'])!=(cell,)):
                    raise AssertionError('Recovery pooled allocation or requested-cell computation changed')
                confidence_seconds+=time.perf_counter()-tick
                tick=time.perf_counter()
                original_record=cell_certificate(original,cell)
                raw_record=cell_certificate(controlled,cell)
                pooled_record=pooled_cell_bound(raw_record,cell,controlled,{cell:pooled_root})
                certification_seconds+=time.perf_counter()-tick
                records={'original':original_record,'allocation_control':raw_record,
                         'pooled_alternative':pooled_record}
                regions={'original':original,'allocation_control':controlled,'pooled_alternative':controlled}
                diagnostics={};reference_diagnostics={};radii={}
                tick=time.perf_counter()
                for name,region in (('original',original),('allocation_control',controlled)):
                    diagnostics[name]=manuscript_diagnostics(region,law['p'],law['e'],beta=F(1,20))
                    reference_diagnostics[name]=manuscript_bound(n,RECOVERY_K,region['alpha'],
                        min(law['p']),reference_eta,known_mass=mode=='known_mass',beta=F(1,20),
                        g_e=diagnostics[name]['g_e'],g_p=diagnostics[name]['g_p'],
                        root_fallbacks=region['root_fallbacks'])
                    radii[name]=realized_radii(region,law['p'],law['e'])
                diagnostic_seconds+=time.perf_counter()-tick
                roots=_root_summary((original,controlled),pooled_root)
                evidence_id=len(evidence)
                evidence.append({'law_id':law_name,'n':n,'mass_mode':mode,
                                 'N':N,'T':T,'original_region':original,'allocation_region':controlled,
                                 'pooled_root':pooled_root,'certificates':records,
                                 'radii':radii,'analytical_diagnostics':diagnostics,
                                 'manuscript_reference_diagnostics':reference_diagnostics,
                                 'root_summary':roots})
                for procedure in PROCEDURES:
                    record=records[procedure];region=regions[procedure]
                    status={'full':'finite','boundary':'infinite','failure':'unresolved'}.get(record['status'])
                    if status is None:
                        raise AssertionError('Full-support evaluation returned an unexpected partial certificate')
                    certificate=record['coupled'] if status=='finite' else None
                    if status=='finite' and certificate is None:
                        raise AssertionError('Completed finite result lacks a certificate')
                    certified=bool(status=='finite' and region['valid'] and certificate<=F(3,2))
                    covered=bool(region['valid'] and all(a<=p<=b and l<=e<=u for a,b,l,u,p,e in zip(
                        region['mass_lower'],region['mass_upper'],region['propensity_lower'],region['propensity_upper'],law['p'],law['e'])))
                    q_true=sum((p*e for p,e in zip(law['p'],law['e'])),F(0))
                    pooled_covered=pooled_root['lower']<=q_true<=pooled_root['upper']
                    inflation_ratio=None if certificate is None else certificate/true
                    inflation=(None if inflation_ratio is None else
                        math.log(inflation_ratio.numerator)-math.log(inflation_ratio.denominator))
                    diagnostic_key='original' if procedure=='original' else 'allocation_control'
                    row={'replication':int(replication),'development':bool(development),
                         'law_id':law_name,'K':RECOVERY_K,'n':n,'mass_mode':mode,'procedure':procedure,
                         'status':status,'certificate':certificate,'true_distortion':true,
                         'certificate_reason':record.get('reason'),
                         'raw_optimization_status':record.get('raw_status',record['status']),
                         'pooled_bound_status':record.get('pooled_status'),
                         'certificate_semantics':record.get('certificate_semantics','exact_raw_region_supremum'),
                         'certificate_minus_truth':None if certificate is None else certificate-true,
                         'certificate_inflation_log':inflation,
                         'certificate_inflation_status':status,
                         'raw_certificate':record.get('raw_coupled',record.get('raw_bound',raw_record['coupled'] if procedure=='pooled_alternative' else certificate)),
                         'pooled_bound':record.get('pooled_bound',record.get('V')),
                         'valid_upper_bound':record.get('valid_upper_bound',certificate),
                         'threshold':F(3,2),'certified':certified,
                         'unsafe_certification':bool(certified and true>F(3,2)),
                         'raw_region_covered':covered,'region_valid':region['valid'],
                         'full_support_pooled_interval_covered':pooled_covered,
                         'certificate_event_covered':bool(covered and (pooled_covered if procedure=='pooled_alternative' else True)),
                         'coverage_scope':'raw event plus used full-support pooled interval; other unused pooled intervals not computed',
                         'raw_root_fallbacks':region['root_fallbacks'],
                         'pooled_root_fallback':pooled_root['status']=='fallback',
                         'root_fallbacks':region['root_fallbacks']+(pooled_root['status']=='fallback' if procedure=='pooled_alternative' else 0),
                         'maximum_precision_used':roots['maximum_precision_used'],
                         'raw_alpha':alpha if procedure=='original' else alpha_raw,
                         'pooled_alpha':alpha_pooled if procedure=='pooled_alternative' else F(0),
                         'pooled_group_count':len(groups),'pooled_gamma':gamma_pooled,
                         'evidence_id':evidence_id,
                         **_scalars(radii[diagnostic_key],'radii_'),
                         **_scalars(diagnostics[diagnostic_key],'analytical_'),
                         **_scalars(reference_diagnostics[diagnostic_key],'manuscript_reference_')}
                    rows.append(row)
    expected=len(laws)*len(recovery['sample_sizes'])*len(MASS_MODES)*len(PROCEDURES)
    if len(rows)!=expected:
        raise AssertionError('Recovery path omitted a committed condition')
    return {'version':'jmlr_efficiency_gap_v1','replication':int(replication),
            'development':bool(development),'count_paths':draws,'laws':laws,
            'procedures':PROCEDURES,'mass_modes':MASS_MODES,
            'pooled_groups':groups,'alpha':alpha,'alpha_raw':alpha_raw,'alpha_pooled':alpha_pooled,
            'gamma_pooled':gamma_pooled,'root_policy':policy_data,
            'manuscript_reference_eta':reference_eta,
            'rows':rows,'evidence':evidence,
            'root_cache':{'hits':verifier.cache_hits-cache_before[0],
                          'misses':verifier.cache_misses-cache_before[1],
                          'size_start':cache_before[2],'size_end':len(verifier.cache),
                          'reuse_scope':'Caller may reuse a policy-matched worker cache across paths; counters are per-path deltas'},
            'root_fallbacks':{'region_interval_occurrences':sum(e['root_summary']['root_fallbacks'] for e in evidence),
                              'row_occurrences':sum(r['root_fallbacks'] for r in rows),
                              'rows_with_fallback':sum(r['root_fallbacks']>0 for r in rows)},
            'timings':{'sampling_seconds':sampling_seconds,
                       'confidence_construction_seconds':confidence_seconds,
                       'cell_certification_seconds':certification_seconds,
                       'analytical_diagnostics_seconds':diagnostic_seconds,
                       'compute_seconds':time.perf_counter()-started},
            'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
            'interpretation':'Full-support recovery diagnostic only; independent paths with nested sample sizes, shared methods/modes/laws. Pointwise confidence allocation at each size, without a simultaneous-over-path claim.'}
