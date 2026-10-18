"""Saved-count operational inference and separate privileged width diagnostics.

The operational function accepts assignments, selected cells, and outcome
counts only. Its aggregate analyst receives the existing allowlisted release.
Population laws enter exclusively in five_stage_diagnostics after inference.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from fractions import Fraction as F
import numbers
import time

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy,RootVerifier,rational
from experiments.jmlr_outcome_inference_v1_core import (
    ALPHA_CAL,ALPHA_RAW,ALPHA_Z,confidence_enclosure,outcome_family,
    population_interval,intersect_enclosures,
)
from experiments.jmlr_release_capacity_v1_interface import export_release,analyze_release,release_bytes
from experiments.jmlr_outcome_inference_v1_lp import build_region,solve_enclosure,verify_saved

G=F(2)


class InferenceCache:
    """Caller-owned, block-local cache; keys include complete scientific inputs."""
    def __init__(self):
        self.raw={};self.aggregate={};self.lp={};self.complete={};self.counts=Counter()

    def statistics(self):
        return dict(self.counts,raw_entries=len(self.raw),aggregate_entries=len(self.aggregate),
                    lp_entries=len(self.lp),complete_entries=len(self.complete))


def _counts(N,T,V,K):
    values=[]
    for vector in (N,T,V):
        vector=tuple(vector)
        if len(vector)!=K or any(isinstance(x,bool) or not isinstance(x,numbers.Integral) or x<0 for x in vector):
            raise ValueError('Outcome vectors require nonnegative integer counts on the complete support')
        values.append(tuple(map(int,vector)))
    N,T,V=values
    if any(not v<=t<=n for n,t,v in zip(N,T,V)):
        raise ValueError('Outcome counts require 0 <= V <= T <= N')
    return N,T,V


def _selected(region,partition,records):
    partition=exact.validate_partition(partition,region['K'])
    if isinstance(records,Mapping):
        selected=tuple(records[cell] for cell in partition)
    else:
        selected=tuple(records)
    if len(selected)!=len(partition):
        raise ValueError('One completed certificate is required for every selected cell')
    U=[]
    for cell,record in zip(partition,selected):
        if record.get('status')!='full' or record.get('coupled') is None:
            raise ValueError('A partial rejection/infinite/unresolved cell is not a completed finite certificate')
        if 'cell' in record and tuple(record['cell'])!=cell:
            raise ValueError('Selected cell and certificate membership differ')
        value=rational(record['coupled'])
        if value<1 or len(cell)==1 and value!=1:
            raise ValueError('Selected distortion bounds must be >=1, with singleton exactly one')
        U.append(value)
    return partition,selected,tuple(U)


def _context(region):
    if (region.get('known_masses') is not None or region.get('mode')!='unknown_mass'
            or rational(region['alpha'])!=ALPHA_CAL or not region['valid']):
        raise ValueError('Inference requires the valid frozen unknown-mass calibration region at alpha=1/40')
    policy=RootPolicy(**region['root_policy'])
    pbox=tuple((rational(a),rational(b)) for a,b in zip(region['mass_lower'],region['mass_upper']))
    ebox=tuple((rational(a),rational(b)) for a,b in zip(region['propensity_lower'],region['propensity_upper']))
    key=(region['K'],tuple(region['N']),tuple(region['T']),ALPHA_CAL,
         tuple(sorted(asdict(policy).items())),pbox,ebox)
    return key,pbox,ebox,policy


def _saved_raw(saved,T,V,K,policy,saved_policy):
    metadata=saved_policy if saved_policy is not None else saved.get('root_policy')
    if isinstance(metadata,RootPolicy):
        metadata=asdict(metadata)
    if metadata is None or RootPolicy(**metadata)!=policy:
        raise ValueError('Saved raw outcome reuse requires the matching declared numerical root policy')
    roots=tuple(saved.get('raw_roots',saved.get('raw_outcome_roots',())))
    if len(roots)!=K:
        raise ValueError('Saved raw outcome roots do not include the complete support')
    bounds=[]
    for root,v,t in zip(roots,V,T):
        if (isinstance(root['x'],bool) or isinstance(root['m'],bool)
                or not isinstance(root['x'],numbers.Integral) or not isinstance(root['m'],numbers.Integral)
                or root['x']!=v or root['m']!=t or rational(root['gamma'])!=ALPHA_RAW/K
                or root['status'] not in ('verified','structural','fallback')):
            raise ValueError('Saved raw outcome roots differ in counts, allocation, or completion contract')
        lo,hi=rational(root['lower']),rational(root['upper'])
        if not 0<=lo<=hi<=1 or root['status']=='fallback' and (lo,hi)!=(F(0),F(1)):
            raise ValueError('Malformed saved raw outcome bounds')
        bounds.append((lo,hi))
    if 'raw_outcome_bounds' in saved and tuple(map(tuple,saved['raw_outcome_bounds']))!=tuple(bounds):
        raise ValueError('Saved raw outcome bound aliases disagree')
    return tuple(bounds),roots


def _seed_raw_brackets(verifier,roots):
    """Reuse compatible verified root evidence, including conservative fallback.

    RootVerifier caches the lower root and the upper endpoint's reflected
    lower root. Seeding these exact inputs makes identity aggregate CP use
    precisely the shared raw intervals, even if a historical resource fallback
    would otherwise be recomputed successfully in this process.
    """
    for record in roots:
        x,m,gamma=int(record['x']),int(record['m']),rational(record['gamma'])
        if x:
            verifier.cache[x,m,gamma,verifier.policy,None]=record['lower_bracket']
        if x<m:
            verifier.cache[m-x,m,gamma,verifier.policy,None]=record['upper_bracket']['reflected_bracket']


def operational_inference(region,partition,cell_records,N,T,V,verifier=None,*,cache=None,
                          raw_saved=None,raw_saved_policy=None,lp_saved=None,lp_time_limit=10):
    """Four operational procedures, with actual selected budgets and no laws.

    raw_saved may be the historical outcome record containing raw_roots and
    raw_confidence. Its policy is supplied explicitly by the source region.
    lp_saved may be an existing LP dictionary or its {'LP': dictionary} wrapper;
    exact input rows and both rational certificates are verified before reuse.
    A cache hit returns zero incremental stage cost, not the original cost.
    """
    started=time.perf_counter();cache=cache if cache is not None else InferenceCache()
    context,pbox,ebox,policy=_context(region)
    verifier=verifier or RootVerifier(policy)
    if verifier.policy!=policy:
        raise ValueError('Outcome verifier differs from the frozen calibration root policy')
    partition,selected,U=_selected(region,partition,cell_records)
    N,T,V=_counts(N,T,V,region['K'])
    key=(context,partition,U,N,T,V,ALPHA_RAW,ALPHA_Z,G,lp_time_limit)
    zero={'raw_seconds':0.,'projection_export_seconds':0.,'analyst_seconds':0.,'lp_seconds':0.}
    if key in cache.complete:
        cache.counts['complete_hits']+=1
        return dict(cache.complete[key],timings=dict(zero,total_seconds=time.perf_counter()-started),
                    reuse={'complete_cache_hit':True,'raw_cache_hit':True,'aggregate_cache_hit':True,
                           'lp_cache_hit':True,'raw_saved_reused':False,'lp_saved_reused':False,'lp_computed':False})
    cache.counts['complete_misses']+=1
    timings=dict(zero);reuse={'complete_cache_hit':False,'raw_saved_reused':False,'lp_saved_reused':False,'lp_computed':False}
    raw_key=(context,N,T,V,ALPHA_RAW,G)
    reuse['raw_cache_hit']=raw_key in cache.raw
    if raw_key not in cache.raw:
        tick=time.perf_counter()
        if raw_saved is None:
            rb,raw_roots=outcome_family(V,T,ALPHA_RAW,verifier)
        else:
            rb,raw_roots=_saved_raw(raw_saved,T,V,region['K'],policy,raw_saved_policy)
            reuse['raw_saved_reused']=True
        raw=confidence_enclosure(pbox,ebox,rb,G)
        if raw_saved is not None and 'raw_confidence' in raw_saved and tuple(raw_saved['raw_confidence']['interval'])!=raw['interval']:
            raise ValueError('Saved raw confidence interval differs from the same-region formula')
        cache.raw[raw_key]=(rb,raw_roots,raw)
        timings['raw_seconds']=time.perf_counter()-tick;cache.counts['raw_misses']+=1
    else:
        cache.counts['raw_hits']+=1
    rb,raw_roots,raw=cache.raw[raw_key]
    _seed_raw_brackets(verifier,raw_roots)
    tick=time.perf_counter()
    artifact=export_release(region,partition,selected,N,T,V,verifier,G=G)
    encoded=release_bytes(artifact)
    timings['projection_export_seconds']=time.perf_counter()-tick
    aggregate_key=(partition,encoded,ALPHA_CAL,ALPHA_Z,G,tuple(sorted(asdict(policy).items())))
    reuse['aggregate_cache_hit']=aggregate_key in cache.aggregate
    if aggregate_key not in cache.aggregate:
        tick=time.perf_counter()
        # This analyst invocation has exactly one data argument: the release.
        analyst=analyze_release(artifact,verifier)
        cache.aggregate[aggregate_key]=analyst
        timings['analyst_seconds']=time.perf_counter()-tick;cache.counts['aggregate_misses']+=1
    else:
        analyst=cache.aggregate[aggregate_key];cache.counts['aggregate_hits']+=1
    zr=analyst['z_outcome_bounds']
    methods={'raw_stratified':raw,**analyst['methods']}
    if len(partition)==region['K'] and not raw['interval']==methods['z_global']['interval']==methods['z_cellwise']['interval']:
        raise ArithmeticError('Identity aggregate and shared raw enclosures differ under the same allocation')
    tick=time.perf_counter()
    lp_region=build_region(pbox,ebox,rb,partition,zr,G)
    lp_key=(context,rb,partition,zr,G,ALPHA_RAW,ALPHA_Z,lp_time_limit)
    reuse['lp_cache_hit']=lp_key in cache.lp
    if lp_key not in cache.lp:
        if lp_saved is not None:
            lp=lp_saved['LP'] if 'LP' in lp_saved else lp_saved
            verify_saved(lp,region=lp_region)
            reuse['lp_saved_reused']=True;cache.counts['lp_saved_reuses']+=1
        else:
            lp=solve_enclosure(lp_region,time_limit=lp_time_limit)
            reuse['lp_computed']=True;cache.counts['lp_solves']+=1
        cache.lp[lp_key]=lp;cache.counts['lp_misses']+=1
    else:
        lp=cache.lp[lp_key];cache.counts['lp_hits']+=1
    methods['raw_pooled']=intersect_enclosures(tuple(map(rational,lp['interval'])),raw['interval'],methods['z_cellwise']['interval'])
    methods['raw_pooled']['lp_status']=lp['status']
    timings['lp_seconds']=time.perf_counter()-tick
    z_pbox=tuple(tuple(map(F,group['mass_bounds'])) for group in artifact['groups'])
    z_ebox=tuple(tuple(map(F,group['propensity_bounds'])) for group in artifact['groups'])
    result={'partition':partition,'cell_certificates':U,'certificate':max(U),
            'global_budget':G*max(U),'cell_budgets':tuple(G*u for u in U),'methods':methods,
            'artifact':artifact,'analyst':analyst,'projection':{'mass_bounds':z_pbox,'propensity_bounds':z_ebox},
            'raw_outcome_bounds':rb,'raw_outcome_roots':raw_roots,
            'z_outcome_bounds':zr,'z_outcome_roots':analyst['z_outcome_roots'],
            'LP':lp,'root_policy':asdict(policy),'payload_bytes':len(encoded),
            'raw_root_fallbacks':sum(r['status']=='fallback' for r in raw_roots),
            'aggregate_root_fallbacks':analyst['root_fallbacks'],
            'timings':dict(timings,total_seconds=time.perf_counter()-started),'reuse':reuse}
    cache.complete[key]=result
    return result


def _contains(outer,inner):
    return outer[0]<=inner[0]<=inner[1]<=outer[1]


def five_stage_diagnostics(law,r,region,partition,cell_records,inference):
    """Privileged exact five-stage intervals and signed width differences.

    No event miss is filtered, and no component is truncated at zero. Nesting
    is checked only under the stated relevant event for each adjacent edge.
    """
    partition,_,U=_selected(region,partition,cell_records)
    if partition!=inference['partition'] or U!=inference['cell_certificates']:
        raise ValueError('Diagnostic partition/certificates differ from operational inference')
    K=region['K'];p,e,r=(tuple(map(rational,x)) for x in (law['p'],law['e'],r))
    if (len(p)!=K or len(e)!=K or len(r)!=K or sum(p)!=1 or any(x<=0 for x in p)
            or any(not 0<x<1 for x in e) or any(not 0<=x<=1 for x in r)):
        raise ValueError('Diagnostic population must be a positive exact finite law with valid binary outcomes')
    pp=[];ep=[];rp=[];D=[]
    for cell in partition:
        mass=sum(p[i] for i in cell);treated=sum(p[i]*e[i] for i in cell)
        q=treated/mass;mean=sum(p[i]*e[i]*r[i] for i in cell)/treated
        oq=exact.odds(q)
        distortion=max(F(1),*(exact.odds(e[i])/oq for i in cell),*(oq/exact.odds(e[i]) for i in cell))
        pp.append(mass);ep.append(q);rp.append(mean);D.append(distortion)
    pp,ep,rp,D=map(tuple,(pp,ep,rp,D))
    projection=inference['projection'];zr=inference['z_outcome_bounds'];rb=inference['raw_outcome_bounds']
    calibration=bool(region['valid'] and all(a<=pi<=b and lo<=ei<=hi for a,b,lo,hi,pi,ei in
        zip(region['mass_lower'],region['mass_upper'],region['propensity_lower'],region['propensity_upper'],p,e)))
    aggregate_outcomes=all(lo<=ri<=hi for (lo,hi),ri in zip(zr,rp))
    raw_outcomes=all(lo<=ri<=hi for (lo,hi),ri in zip(rb,r))
    projected_assignment=all(a<=pi<=b and lo<=ei<=hi for (a,b),(lo,hi),pi,ei in
                             zip(projection['mass_bounds'],projection['propensity_bounds'],pp,ep))
    cell_certificates_cover=all(d<=u for d,u in zip(D,U))
    global_certificate_covers=max(D)<=max(U)
    if calibration and (not cell_certificates_cover or not projected_assignment):
        raise ArithmeticError('Calibration-event certificate/projection implication failed')
    I0=population_interval(p,e,r,G)
    output={}
    names=('population_aggregation_transport','certificate_budget','outcome_sampling','assignment_estimation')
    for name,true_budget,budget,certificate_event,method in (
        ('global',G*max(D),G*max(U),global_certificate_covers,'z_global'),
        ('cellwise',tuple(G*d for d in D),tuple(G*u for u in U),cell_certificates_cover,'z_cellwise')):
        I1=population_interval(pp,ep,rp,true_budget)
        I2=population_interval(pp,ep,rp,budget)
        privileged=confidence_enclosure(tuple((x,x) for x in pp),tuple((x,x) for x in ep),zr,budget)
        I3=privileged['interval'];I4=inference['methods'][method]['interval']
        stages=(I0,I1,I2,I3,I4);widths=tuple(hi-lo for lo,hi in stages)
        differences=tuple(b-a for a,b in zip(widths,widths[1:]))
        if sum(differences,F(0))!=widths[-1]-widths[0]:
            raise ArithmeticError('Signed five-stage width accounting failed')
        nesting=tuple(_contains(b,a) for a,b in zip(stages,stages[1:]))
        applicable=(True,certificate_event,aggregate_outcomes,projected_assignment)
        if any(event and not nested for event,nested in zip(applicable,nesting)):
            raise ArithmeticError('Five-stage interval nesting failed on its relevant event')
        output[name]={'intervals':dict(zip(('I0','I1','I2','I3','I4'),stages)),
                      'widths':dict(zip(('W0','W1','W2','W3','W4'),widths)),
                      'components':dict(zip(names,differences)),'total_excess':widths[-1]-widths[0],
                      'accounting_exact':True,'nesting':dict(zip(('I0_in_I1','I1_in_I2','I2_in_I3','I3_in_I4'),nesting)),
                      'nesting_event_applicable':dict(zip(('I0_in_I1','I1_in_I2','I2_in_I3','I3_in_I4'),applicable)),
                      'true_budgets':true_budget,'certified_budgets':budget,'I3_enclosure':privileged}
    failures={name:not _contains(result['interval'],I0) for name,result in inference['methods'].items()}
    if calibration and raw_outcomes and failures['raw_stratified']:
        raise ArithmeticError('Raw stratification missed its population interval on the raw joint event')
    if calibration and aggregate_outcomes and (failures['z_global'] or failures['z_cellwise']):
        raise ArithmeticError('Transported aggregate enclosure missed the raw interval on its joint event')
    if calibration and raw_outcomes and aggregate_outcomes and failures['raw_pooled']:
        raise ArithmeticError('Pooled raw enclosure missed the raw interval on the shared joint event')
    return {'global':output['global'],'cellwise':output['cellwise'],
            'population':{'p_C':pp,'e_C':ep,'r_C':rp,'cell_distortions':D,'distortion':max(D),'raw_interval':I0},
            'events':{'calibration_covered':calibration,'raw_outcomes_covered':raw_outcomes,
                      'aggregate_outcomes_covered':aggregate_outcomes,'projected_assignment_covered':projected_assignment,
                      'cell_certificates_cover':cell_certificates_cover,'global_certificate_covers':global_certificate_covers,
                      'shared_event':calibration and raw_outcomes and aggregate_outcomes},
            'entire_raw_interval_misses':failures,
            'scope':'Pointwise outcome analysis; common calibration event across caps; signed sequence diagnostics, no conditional-on-admission claim'}
