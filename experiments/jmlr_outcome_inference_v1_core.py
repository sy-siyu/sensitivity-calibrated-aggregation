"""Outcome-blind calibration and finite-sample binary outcome enclosures.

This new wrapper passes every confidence allocation explicitly. Historical
modules/constants are imported unchanged; no privileged law enters selection
or confidence construction. Population diagnostics are a separate function.
"""
from __future__ import annotations

from fractions import Fraction as F
import time

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier, RootPolicy
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import exact_order, select_families
from experiments.jmlr_verified_pipeline_v1_laws import TruthEvaluator

G = F(2)
CAP = F(3, 2)
ALPHA_CAL = F(1, 40)
ALPHA_RAW = ALPHA_Z = F(1, 80)
ALPHA_STANDALONE_ASSIGNMENT = ALPHA_STANDALONE_OUTCOME = F(1, 40)
METHODS = ('raw_stratified', 'z_global', 'z_cellwise', 'raw_pooled',
           'raw_standalone', 'z_untransported')
SHARED_METHODS = METHODS[:4]
PROFILES = ('constant', 'aligned', 'alternating')


def pair(interval):
    return tuple(F(x) for x in interval)


def binary_coeff(e, r, budget, upper=False):
    e, r, budget = F(e), F(r), F(budget)
    if not 0 <= e <= 1 or not 0 <= r <= 1 or budget < 1:
        raise ValueError('Binary coefficients require probabilities and budget >= 1')
    h = min(budget*r, 1-(1-r)/budget) if upper else max(r/budget, 1-budget*(1-r))
    return e*r+(1-e)*h


def population_interval(p, e, r, budgets):
    if isinstance(budgets, (int, F)):
        budgets = (F(budgets),)*len(p)
    if not len(p) == len(e) == len(r) == len(budgets) or sum(p) != 1:
        raise ValueError('Population vectors must align and masses sum exactly to one')
    return (sum((F(w)*binary_coeff(a,b,g) for w,a,b,g in zip(p,e,r,budgets)), F()),
            sum((F(w)*binary_coeff(a,b,g,True) for w,a,b,g in zip(p,e,r,budgets)), F()))


def mass_linear_extremum(bounds, coefficients, maximize=False):
    """Exact linear objective over a bounded simplex, with a feasible witness."""
    bounds = tuple(map(pair, bounds)); coefficients = tuple(map(F, coefficients))
    if len(bounds) != len(coefficients) or any(not 0 <= a <= b <= 1 for a,b in bounds):
        raise ValueError('Malformed bounded mass simplex')
    p = [a for a,b in bounds]; remaining = 1-sum(p)
    if remaining < 0 or sum(b for a,b in bounds) < 1:
        raise ValueError('Empty bounded mass simplex')
    order = sorted(range(len(p)), key=lambda i: ((-1 if maximize else 1)*coefficients[i], i))
    for i in order:
        take = min(remaining, bounds[i][1]-p[i]); p[i] += take; remaining -= take
    if remaining or sum(p) != 1 or any(not a <= x <= b for x,(a,b) in zip(p,bounds)):
        raise ArithmeticError('Mass witness is infeasible')
    value = sum((x*c for x,c in zip(p,coefficients)), F())
    return {'value':value, 'witness':tuple(p), 'coefficients':coefficients,
            'maximize':bool(maximize), 'status':'exact'}


def confidence_enclosure(pbox, ebox, rbox, budgets):
    if isinstance(budgets,(int,F)):
        budgets=(F(budgets),)*len(pbox)
    try:
        if not len(pbox)==len(ebox)==len(rbox)==len(budgets) or not pbox:
            raise ValueError('Enclosure inputs require one common nonempty support')
        if any(not 0<=a<=b<=1 for bounds in (ebox,rbox) for a,b in bounds):
            raise ValueError('Malformed probability interval')
        lower = tuple(binary_coeff(e[0],r[0],g) for e,r,g in zip(ebox,rbox,budgets))
        upper = tuple(binary_coeff(e[0],r[1],g,True) for e,r,g in zip(ebox,rbox,budgets))
        lo = mass_linear_extremum(pbox,lower)
        hi = mass_linear_extremum(pbox,upper,True)
        if not 0 <= lo['value'] <= hi['value'] <= 1:
            raise ArithmeticError('Malformed outcome enclosure')
        return {'interval':(lo['value'],hi['value']), 'lower_optimization':lo,
                'upper_optimization':hi, 'budgets':tuple(budgets),
                'status':'verified', 'fallback':False}
    except (ValueError,ArithmeticError) as error:
        return {'interval':(F(0),F(1)), 'status':'fallback', 'fallback':True,
                'reason':str(error), 'budgets':tuple(budgets)}


def choose_calibration(N,T,tree_orders,immutable_ids,verifier=None):
    """Selection uses only calibration assignments and fixed design inputs."""
    start=time.perf_counter(); verifier=verifier or RootVerifier()
    region=exact.confidence_region(N,T,ALPHA_CAL,verifier=verifier)
    order=exact_order(N,T,immutable_ids)
    trees=tuple(exact.build_tree(o) for o in tree_orders)
    cells=tuple(sorted(set(exact.interval_cells(order))|set().union(*(set(t) for t in trees))))
    records={c:cell_certificate(region,c,max_cap=CAP) for c in cells}
    values={c:(r['coupled'] if r['status']=='full' else None) for c,r in records.items()}
    choices=select_families(order,trees,values,CAP)
    choice=choices['combined']; partition=exact.validate_partition(choice['partition'],len(N))
    if any(records[c]['status']!='full' or records[c]['coupled'] is None
           or records[c]['coupled']>CAP for c in partition):
        raise ArithmeticError('Selection lacks complete within-cap certificates')
    selected=tuple(records[c] for c in partition)
    certificate=max(r['coupled'] for r in selected)
    if certificate!=choice['certificate']:
        raise ArithmeticError('Partition certificate mismatch')
    return {'N':tuple(map(int,N)), 'T':tuple(map(int,T)), 'region':region,
            'order':order, 'tree_orders':tuple(tuple(o) for o in tree_orders),
            'immutable_ids':tuple(immutable_ids), 'family_cells':cells,
            'cell_records':tuple(records[c] for c in cells), 'choices':choices,
            'partition':partition, 'selected_cells':selected, 'certificate':certificate,
            'count':len(partition), 'cap':CAP, 'alpha':ALPHA_CAL,
            'compute_seconds':time.perf_counter()-start}


def project_assignment(region,partition):
    """Coordinate projection of the raw simplex and exact cell mixtures."""
    projection=[]; pbox=[]; ebox=[]
    for cell in partition:
        _,lo,hi,_=exact._cell_mass(region,cell)
        mass=(F(lo,region['_D']),F(hi,region['_D']))
        l=exact.mixture_extremum(region,cell,region['propensity_lower'])
        u=exact.mixture_extremum(region,cell,region['propensity_upper'],True)
        pbox.append(mass);ebox.append((l['value'],u['value']))
        projection.append({'cell':cell,'mass':mass,'propensity':ebox[-1],
                           'lower_mixture':l,'upper_mixture':u})
    return tuple(pbox),tuple(ebox),tuple(projection)


def outcome_family(successes,treated,alpha,verifier):
    if len(successes)!=len(treated) or not successes:
        raise ValueError('Outcome count vectors must align')
    gamma=F(alpha)/len(successes)
    roots=tuple(verifier.interval(int(x),int(m),gamma) for x,m in zip(successes,treated))
    return tuple((r['lower'],r['upper']) for r in roots),roots


def intersect_enclosures(*intervals):
    lower=max(x[0] for x in intervals);upper=min(x[1] for x in intervals)
    if lower>upper:
        return {'interval':(F(0),F(1)), 'empty_intersection':True,
                'attempted_intersection':(lower,upper),'fallback':True,
                'status':'empty-intersection-fallback'}
    return {'interval':(lower,upper),'attempted_intersection':(lower,upper),
            'empty_intersection':False,'fallback':False,'status':'shared-event-intersection'}


def infer_outcome(calibration,N,T,V,verifier=None,lp_saved=None):
    """Six procedures; this signature deliberately has no population inputs."""
    from experiments.jmlr_outcome_inference_v1_lp import build_region,solve_enclosure,verify_saved
    start=time.perf_counter();verifier=verifier or RootVerifier()
    region=calibration['region'];partition=calibration['partition'];K=region['K']
    N,T,V=tuple(map(int,N)),tuple(map(int,T)),tuple(map(int,V))
    if len(N)!=K or len(T)!=K or len(V)!=K or any(not 0<=v<=t<=n for n,t,v in zip(N,T,V)):
        raise ValueError('Outcome counts require 0 <= V <= T <= N')
    pbox=tuple(zip(region['mass_lower'],region['mass_upper']))
    ebox=tuple(zip(region['propensity_lower'],region['propensity_upper']))
    raw_r,raw_roots=outcome_family(V,T,ALPHA_RAW,verifier)
    Tz=tuple(sum(T[i] for i in c) for c in partition)
    Vz=tuple(sum(V[i] for i in c) for c in partition)
    zr,z_roots=outcome_family(Vz,Tz,ALPHA_Z,verifier)
    tick=time.perf_counter()
    z_p,z_e,projection=project_assignment(region,partition)
    U=tuple(r['coupled'] for r in calibration['selected_cells'])
    methods={'raw_stratified':confidence_enclosure(pbox,ebox,raw_r,G),
             'z_global':confidence_enclosure(z_p,z_e,zr,G*calibration['certificate']),
             'z_cellwise':confidence_enclosure(z_p,z_e,zr,tuple(G*u for u in U)),
             'z_untransported':confidence_enclosure(z_p,z_e,zr,G)}
    a,b=methods['z_global']['interval'],methods['z_cellwise']['interval']
    if not a[0]<=b[0]<=b[1]<=a[1]:
        raise ArithmeticError('Cellwise refinement escaped global enclosure')
    if len(partition)==K and not methods['raw_stratified']['interval']==a==b:
        raise ArithmeticError('Identity enclosures differ under shared allocation')
    direct_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    lp_region=build_region(pbox,ebox,raw_r,partition,zr,G)
    if lp_saved is None:
        lp=solve_enclosure(lp_region)
    else:
        verify_saved(lp_saved,region=lp_region);lp=lp_saved
    methods['raw_pooled']=intersect_enclosures(pair(lp['interval']),methods['raw_stratified']['interval'],b)
    methods['raw_pooled']['lp_status']=lp['status']
    lp_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    sn=tuple(a+b for a,b in zip(calibration['N'],N))
    st=tuple(a+b for a,b in zip(calibration['T'],T))
    standalone=exact.confidence_region(sn,st,ALPHA_STANDALONE_ASSIGNMENT,verifier=verifier)
    sr,sroots=outcome_family(V,T,ALPHA_STANDALONE_OUTCOME,verifier)
    methods['raw_standalone']=confidence_enclosure(tuple(zip(standalone['mass_lower'],standalone['mass_upper'])),
        tuple(zip(standalone['propensity_lower'],standalone['propensity_upper'])),sr,G)
    # Plug-ins are descriptive point estimates, never honest confidence intervals.
    phat=tuple(F(n,sum(calibration['N'])) for n in calibration['N'])
    ehat=tuple(F(t,n) if n else F(1,2) for t,n in zip(calibration['T'],calibration['N']))
    rhat=tuple(F(v,t) if t else F(1,2) for v,t in zip(V,T))
    pzhat=tuple(sum(phat[i] for i in c) for c in partition)
    ezhat=tuple(sum(phat[i]*ehat[i] for i in c)/p if p else F(1,2) for c,p in zip(partition,pzhat))
    rzhat=tuple(F(v,t) if t else F(1,2) for v,t in zip(Vz,Tz))
    plugins={'raw_stratified':population_interval(phat,ehat,rhat,G),
             'z_global':population_interval(pzhat,ezhat,rzhat,G*calibration['certificate']),
             'z_cellwise':population_interval(pzhat,ezhat,rzhat,tuple(G*u for u in U)),
             'z_untransported':population_interval(pzhat,ezhat,rzhat,G)}
    plugins['raw_pooled']=plugins['raw_stratified']
    plugins['raw_standalone']=population_interval(tuple(F(n,sum(sn)) for n in sn),
        tuple(F(t,n) if n else F(1,2) for t,n in zip(st,sn)),rhat,G)
    return {'N':N,'T':T,'V':V,'Tz':Tz,'Vz':Vz,'raw_outcome_roots':raw_roots,
            'z_outcome_roots':z_roots,'raw_outcome_bounds':raw_r,'z_outcome_bounds':zr,
            'projection':projection,'methods':methods,'LP':lp,
            'standalone_region':standalone,'standalone_outcome_roots':sroots,
            'standalone_outcome_bounds':sr,'plugins':plugins,
            'timings':{'direct_enclosure_seconds':direct_seconds,'lp_seconds':lp_seconds,
                       'standalone_and_plugin_seconds':time.perf_counter()-tick,
                       'compute_seconds':time.perf_counter()-start}}


def outcome_profiles(law):
    order=tuple(sorted(range(law['K']),key=lambda i:(law['e'][i],law['immutable_ids'][i])))
    values={name:[F(1,2)]*law['K'] for name in PROFILES}
    for rank,i in enumerate(order):
        values['aligned'][i]=F(1,8) if rank<law['K']//2 else F(7,8)
        values['alternating'][i]=F(1,8) if rank%2==0 else F(7,8)
    return {k:tuple(v) for k,v in values.items()}


def population_references(law,r,calibration):
    """Privileged laws appear only here, after selection and inference."""
    p,e=law['p'],law['e'];partition=calibration['partition'];evaluator=TruthEvaluator(law)
    cells=tuple(evaluator.cell(c) for c in partition)
    pz=tuple(c['mass'] for c in cells);ez=tuple(c['propensity'] for c in cells)
    rz=tuple(sum(p[i]*e[i]*r[i] for i in c)/sum(p[i]*e[i] for i in c) for c in partition)
    d=tuple(c['value'] for c in cells);U=tuple(c['coupled'] for c in calibration['selected_cells'])
    return {'raw':population_interval(p,e,r,G),'pz':pz,'ez':ez,'rz':rz,
            'true_cell_distortions':d,'true_distortion':max(d),
            'z_global_true':population_interval(pz,ez,rz,G*max(d)),
            'z_cellwise_true':population_interval(pz,ez,rz,tuple(G*x for x in d)),
            'z_global_certificate':population_interval(pz,ez,rz,G*calibration['certificate']),
            'z_cellwise_certificate':population_interval(pz,ez,rz,tuple(G*x for x in U)),
            'z_untransported':population_interval(pz,ez,rz,G)}


def diagnose(law,r,calibration,inference):
    pop=population_references(law,r,calibration);raw=pop['raw'];reg=calibration['region']
    calibration_covered=bool(reg['valid'] and all(a<=p<=b and l<=e<=u for a,b,l,u,p,e in
        zip(reg['mass_lower'],reg['mass_upper'],reg['propensity_lower'],reg['propensity_upper'],law['p'],law['e'])))
    raw_covered=all(a<=x<=b for (a,b),x in zip(inference['raw_outcome_bounds'],r))
    z_covered=all(a<=x<=b for (a,b),x in zip(inference['z_outcome_bounds'],pop['rz']))
    methods={}
    for name in METHODS:
        lo,hi=pair(inference['methods'][name]['interval'])
        methods[name]={'interval':(lo,hi),'width':hi-lo,'contains_raw_set':lo<=raw[0] and hi>=raw[1],
                       'lower_miss':lo>raw[0],'upper_miss':hi<raw[1],
                       'vacuous':lo==0 and hi==1,'fallback':inference['methods'][name]['fallback']}
    shared_event=calibration_covered and raw_covered and z_covered
    joint_containment=all(methods[name]['contains_raw_set'] for name in SHARED_METHODS)
    if shared_event and not joint_containment:
        raise ArithmeticError('A shared-event coverage implication failed')
    sr=inference['standalone_region']
    standalone_assignment_covered=bool(sr['valid'] and all(a<=p<=b and l<=e<=u for a,b,l,u,p,e in
        zip(sr['mass_lower'],sr['mass_upper'],sr['propensity_lower'],sr['propensity_upper'],law['p'],law['e'])))
    standalone_outcomes_covered=all(a<=x<=b for (a,b),x in zip(inference['standalone_outcome_bounds'],r))
    standalone_event=standalone_assignment_covered and standalone_outcomes_covered
    if standalone_event and not methods['raw_standalone']['contains_raw_set']:
        raise ArithmeticError('A standalone coverage implication failed')
    decompositions={}
    width=lambda x:x[1]-x[0]
    for name in ['z_global','z_cellwise']:
        loss=width(pop[name+'_true'])-width(raw)
        certificate=width(pop[name+'_certificate'])-width(pop[name+'_true'])
        statistical=methods[name]['width']-width(pop[name+'_certificate'])
        total=methods[name]['width']-width(raw)
        if loss+certificate+statistical!=total:
            raise ArithmeticError('Signed decomposition does not sum exactly')
        decompositions[name]={'population_loss':loss,'certificate_extra':certificate,
                              'statistical_extra':statistical,'total_excess':total}
    return {'population':pop,'methods':methods,'decompositions':decompositions,
            'calibration_covered':calibration_covered,'raw_outcomes_covered':raw_covered,
            'z_outcomes_covered':z_covered,'shared_event':shared_event,
            'joint_raw_set_containment':joint_containment,
            'standalone_assignment_covered':standalone_assignment_covered,
            'standalone_outcomes_covered':standalone_outcomes_covered,
            'standalone_event':standalone_event,
            'zero_raw_treated':sum(t==0 for t in inference['T']),
            'zero_cell_treated':sum(t==0 for t in inference['Tz'])}


def clean_mathematical(record):
    if isinstance(record,dict):
        return {k:clean_mathematical(v) for k,v in record.items()
                if k not in ('timings','compute_seconds','elapsed_seconds') and not k.startswith('_')}
    if isinstance(record,(list,tuple)):
        return tuple(clean_mathematical(v) for v in record)
    return record
