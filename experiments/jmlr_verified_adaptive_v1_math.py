"""Exact finite-support envelopes over a verified enclosing CP region.

All authoritative inputs/outputs and decisions are rational. Binary floats are
used only outside this module for historical displays and root proposals.
"""
from __future__ import annotations
from fractions import Fraction
from math import lcm
import numbers
from experiments.jmlr_verified_adaptive_v1_roots import (
    rational, fraction_json, fraction_from_json, RootPolicy, RootVerifier,
    cp_interval, tail_difference, ball_bounds)

ONE=Fraction(1);ZERO=Fraction(0)


def outward_decimal(value,places=12):
    if value is None:return 'infinity'
    value=rational(value);scale=10**places
    integer=-((-value.numerator*scale)//value.denominator)
    sign='-' if integer<0 else '';integer=abs(integer)
    return f'{sign}{integer//scale}.{integer%scale:0{places}d}' if places else str((-1 if sign else 1)*integer)


def certificate_json(value,status='full'):
    return {'infinite':value is None,'status':status,'value':None if value is None else fraction_json(value),
            'upper_decimal':outward_decimal(value)}


def encode(value):
    if isinstance(value,Fraction):return fraction_json(value)
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items() if not str(k).startswith('_')}
    if isinstance(value,(tuple,list)):return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value,dict):
        if set(value)=={'numerator','denominator'}:return fraction_from_json(value)
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(v) for v in value]
    return value


def _cell(cell,K):
    cell=tuple(cell)
    if not cell or any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for i in cell):
        raise ValueError('Cell must contain integer state indices')
    cell=tuple(sorted(int(i) for i in cell))
    if len(set(cell))!=len(cell) or cell[0]<0 or cell[-1]>=K:raise ValueError('Invalid cell indices')
    return cell


def canonical_partition(partition):
    return tuple(sorted(tuple(sorted(int(i) for i in cell)) for cell in partition))


def validate_partition(partition,K):
    cells=tuple(_cell(c,K) for c in partition)
    if sorted(i for c in cells for i in c)!=list(range(K)):
        raise ValueError('Partition must contain every known state exactly once')
    return canonical_partition(cells)


def region_from_bounds(a,b,l,u,known_masses=None):
    a,b,l,u=[tuple(rational(v) for v in x) for x in (a,b,l,u)]
    K=len(a)
    if not K or any(len(x)!=K for x in (b,l,u)):raise ValueError('Bounds require one common known support')
    if known_masses is not None:
        masses=tuple(rational(x) for x in known_masses)
        if len(masses)!=K or any(p<=0 for p in masses) or sum(masses)!=1:
            raise ValueError('Known masses must be positive rationals summing exactly to one; no normalization')
        if any(not a[i]<=masses[i]<=b[i] for i in range(K)):raise ValueError('Known masses outside declared bounds')
        a=b=masses
    else:masses=None
    diagnostics=[]
    if any(not 0<=a[i]<=b[i]<=1 or not 0<=l[i]<=u[i]<=1 for i in range(K)):
        diagnostics.append('malformed-box')
    if any(bi<=0 for bi in b) or sum(a)>1 or sum(b)<1 or sum(a)==1 and any(ai==0 for ai in a):
        diagnostics.append('no-positive-mass-law')
    if any(ui<=0 or li>=1 for li,ui in zip(l,u)):
        diagnostics.append('no-interior-propensity-law')
    D=lcm(*(v.denominator for v in (*a,*b)))
    ai=tuple(v.numerator*(D//v.denominator) for v in a)
    bi=tuple(v.numerator*(D//v.denominator) for v in b)
    return {'K':K,'mass_lower':a,'mass_upper':b,'propensity_lower':l,'propensity_upper':u,
            'known_masses':masses,'mode':'known_mass' if masses is not None else 'unknown_mass',
            'valid':not diagnostics,'diagnostics':diagnostics,
            '_D':D,'_a':ai,'_b':bi,'_sum_a':sum(ai),'_sum_b':sum(bi),'_cell_cache':{}}


def confidence_region(N,T,alpha=Fraction(1,20),known_masses=None,verifier=None):
    N,T=tuple(N),tuple(T);alpha=rational(alpha)
    if not N or len(N)!=len(T) or not 0<alpha<1:raise ValueError('Invalid calibration counts/error')
    for n,t in zip(N,T):
        if isinstance(n,bool) or isinstance(t,bool) or not isinstance(n,numbers.Integral) or not isinstance(t,numbers.Integral) or n<0 or t<0 or t>n:
            raise ValueError('Require integer 0 <= T_i <= N_i')
    N,T=tuple(map(int,N)),tuple(map(int,T));n=sum(N);K=len(N)
    if n<=0:raise ValueError('Calibration total must be positive')
    if known_masses is not None:
        known_masses=tuple(rational(x) for x in known_masses)
        if len(known_masses)!=K or any(x<=0 for x in known_masses) or sum(known_masses)!=1:
            raise ValueError('Known masses must sum exactly to one and be positive')
    verifier=RootVerifier() if verifier is None else verifier
    gamma=alpha/(K if known_masses is not None else 2*K)
    propensity=[verifier.interval(t,m,gamma) for t,m in zip(T,N)]
    mass=None if known_masses is not None else [verifier.interval(m,n,gamma) for m in N]
    a=known_masses if mass is None else [d['lower'] for d in mass]
    b=known_masses if mass is None else [d['upper'] for d in mass]
    region=region_from_bounds(a,b,[d['lower'] for d in propensity],[d['upper'] for d in propensity],known_masses)
    region.update(N=N,T=T,n=n,alpha=alpha,gamma=gamma,roots={'propensity':propensity,'mass':mass},
                  root_policy=verifier.policy.__dict__,root_fallbacks=sum(d['status']=='fallback' for d in propensity+(mass or [])))
    return region


def _cell_mass(region,cell):
    cache=region['_cell_cache']
    if cell not in cache:
        a,b,D=region['_a'],region['_b'],region['_D'];A=sum(a[i] for i in cell);B=sum(b[i] for i in cell)
        lower=max(A,D-(region['_sum_b']-B),0);upper=min(B,D-(region['_sum_a']-A),D)
        outside=tuple(i for i in range(region['K']) if i not in cell)
        cache[cell]=(A,lower,upper,outside)
    return cache[cell]


def mixture_extremum(region,cell,coefficients,maximize=False):
    """Exact sorted allocation, all feasible capacity breakpoints, checked witness.

    The witness may have zero coordinates: it belongs to the closed polytope;
    positive laws approach it by mixing with a positive feasible vector.
    """
    K=region['K'];cell=_cell(cell,K)
    if not region['valid']:raise ValueError('No admissible positive law')
    v=tuple(rational(x) for x in coefficients)
    if len(v)!=K:raise ValueError('Coefficient vector has wrong support')
    if region['known_masses'] is not None:
        p=region['known_masses'];S=sum(p[i] for i in cell)
        return {'value':sum(p[i]*v[i] for i in cell)/S,'witness':p,'status':'exact',
                'cell_mass':S,'breakpoints':1}
    a,b,D=region['_a'],region['_b'],region['_D'];A,lo,hi,outside=_cell_mass(region,cell)
    if hi<=0 or hi<lo:raise ValueError('No positive feasible cell mass')
    if lo==0:
        return {'value':(max if maximize else min)(v[i] for i in cell),'witness':None,
                'status':'closure','cell_mass':ZERO,'breakpoints':0}
    Dc=lcm(*(v[i].denominator for i in cell))
    coefficients_int={i:v[i].numerator*(Dc//v[i].denominator)*(-1 if maximize else 1) for i in cell}
    order=sorted(cell,key=lambda i:(coefficients_int[i],i))
    base=sum(a[i]*coefficients_int[i] for i in cell)
    candidates={lo,hi};step=A
    for i in order:
        step+=b[i]-a[i]
        if lo<=step<=hi:candidates.add(step)
    candidates=sorted(candidates)
    # Increasing candidate S permits one allocation sweep.
    numerator=base;S=A;pos=0;allocated=0;best_num=None;best_S=None
    for target in candidates:
        extra=target-S
        while extra:
            while pos<len(order) and b[order[pos]]-a[order[pos]]==allocated:
                pos+=1;allocated=0
            if pos==len(order):raise ArithmeticError('Capacity allocation exhausted')
            i=order[pos];take=min(extra,b[i]-a[i]-allocated)
            numerator+=take*coefficients_int[i];allocated+=take;extra-=take;S+=take
        if best_num is None or numerator*best_S<best_num*target:
            best_num,best_S=numerator,target
    p=list(a);remaining=best_S-A
    for i in order:
        take=min(remaining,b[i]-a[i]);p[i]+=take;remaining-=take
    if remaining:raise ArithmeticError('Cell witness allocation failed')
    remaining=D-sum(p)
    for i in outside:
        take=min(remaining,b[i]-a[i]);p[i]+=take;remaining-=take
    if remaining or sum(p)!=D or any(not a[i]<=p[i]<=b[i] for i in range(K)) or sum(p[i] for i in cell)!=best_S:
        raise ArithmeticError('Exact mass witness failed feasibility')
    q=Fraction(best_num, Dc*best_S)*(-1 if maximize else 1)
    if sum(p[i]*v[i] for i in cell)/best_S!=q:raise ArithmeticError('Witness objective mismatch')
    return {'value':q,'witness':tuple(Fraction(pi,D) for pi in p),'status':'exact',
            'cell_mass':Fraction(best_S,D),'breakpoints':len(candidates)}


def odds(x):
    x=rational(x)
    if not 0<x<1:raise ValueError('Odds require interior rational argument')
    return x/(1-x)


def cell_certificate(region,cell,max_cap=None):
    cell=_cell(cell,region['K'])
    if max_cap is not None:max_cap=rational(max_cap)
    answer={'cell':cell,'coupled':None,'independent':None,'status':'full','anchors':[],
            'rejection_bound':None,'mixture_calls':0,'closure_calls':0,'reason':None}
    if len(cell)==1:return dict(answer,coupled=ONE,independent=ONE)
    if not region['valid']:return dict(answer,status='failure',reason='invalid-region')
    lower,upper=region['propensity_lower'],region['propensity_upper']
    if any(lower[i]<=0 or upper[i]>=1 for i in cell):
        return dict(answer,status='boundary',reason='genuine-boundary-infinity')
    def optimize(v,maximize):
        result=mixture_extremum(region,cell,v,maximize)
        answer['mixture_calls']+=1;answer['closure_calls']+=result['status']=='closure'
        return result['value']
    try:
        qlo,qhi=optimize(lower,False),optimize(upper,True)
        independent=max(ONE,odds(max(upper[i] for i in cell))/odds(qlo),odds(qhi)/odds(min(lower[i] for i in cell)))
        answer.update(independent=independent,independent_q_lower=qlo,independent_q_upper=qhi)
        coupled=ONE
        # Extreme propensity anchors first are a computation choice only.
        order=sorted(cell,key=lambda i:(max(odds(upper[i])/odds(min(lower[j] for j in cell)),
                                           odds(max(upper[j] for j in cell))/odds(lower[i])),i),reverse=True)
        for i in order:
            v=list(lower);v[i]=upper[i];qplus=optimize(v,False)
            v=list(upper);v[i]=lower[i];qminus=optimize(v,True)
            plus,minus=odds(upper[i])/odds(qplus),odds(qminus)/odds(lower[i])
            coupled=max(coupled,plus,minus)
            answer['anchors'].append({'state':i,'q_plus':qplus,'q_minus':qminus,'plus':plus,'minus':minus})
            if max_cap is not None and min(coupled,independent)>max_cap:
                return dict(answer,status='rejected',rejection_bound=min(coupled,independent),
                            reason='proved-rejection-above-largest-cap')
        if coupled>independent:raise ArithmeticError('Exact coupled bound exceeds comparator')
        return dict(answer,coupled=coupled)
    except Exception as error:
        return dict(answer,coupled=None,independent=answer['independent'],status='failure',reason=type(error).__name__+': '+str(error))


def interval_cells(order):
    order=tuple(order)
    if len(set(order))!=len(order):raise ValueError('Repeated state in order')
    return tuple(sorted(tuple(sorted(order[i:j])) for j in range(1,len(order)+1) for i in range(j)))


def build_tree(order):
    order=tuple(order);tree={}
    def visit(leaves):
        cell=tuple(sorted(leaves))
        if len(leaves)==1:tree[cell]=None
        else:
            middle=len(leaves)//2;left,right=visit(leaves[:middle]),visit(leaves[middle:]);tree[cell]=(left,right)
        return cell
    if not order or len(set(order))!=len(order):raise ValueError('Invalid tree leaf order')
    visit(order);return tree


def leq(value,cap):
    if value is None:return False
    value=rational(value)
    if value<1:raise ValueError('Certificate values must be at least one')
    return value<=cap


def _choice(partition,values,**extra):
    partition=canonical_partition(partition)
    vals=[ONE if len(c)==1 else values[c] for c in partition]
    certificate=None if any(v is None for v in vals) else max(vals,default=ONE)
    return dict(partition=partition,cells=len(partition),certificate=certificate,**extra)


def select_contiguous(order,values,cap):
    order=tuple(order);cap=rational(cap);K=len(order)
    if cap<1 or len(set(order))!=K:raise ValueError('Invalid order or cap')
    F=[0]+[K+1]*K;predecessor=[-1]*(K+1)
    for j in range(1,K+1):
        for i in range(j):
            cell=tuple(sorted(order[i:j]));value=ONE if j-i==1 else values[cell]
            if leq(value,cap) and F[i]+1<F[j]:F[j],predecessor[j]=F[i]+1,i
    partition=[];cuts=[];j=K
    while j:
        i=predecessor[j]
        if i<0:raise ArithmeticError('Identity unavailable')
        partition.append(tuple(sorted(order[i:j])))
        if i:cuts.append(i)
        j=i
    return _choice(partition,values,cuts=tuple(sorted(cuts)),predecessors=tuple(predecessor))


def _choice_key(choice):
    return choice['cells'],(choice['certificate'] is None,choice['certificate'] or ZERO),choice['partition']


def select_family(trees,values,cap):
    cap=rational(cap);choices=[]
    for tree in trees:
        root=max(tree,key=len)
        def visit(cell):
            value=ONE if len(cell)==1 else values[cell]
            candidates=[]
            if leq(value,cap):candidates.append(_choice((cell,),values))
            children=tree[cell]
            if children is not None:
                left,right=map(visit,children)
                candidates.append(_choice(left['partition']+right['partition'],values))
            if not candidates:raise ArithmeticError('Identity unavailable')
            return min(candidates,key=_choice_key)
        choices.append(visit(root))
    return min(choices,key=_choice_key)


def select_combined(fixed_choice,contig_choice):
    choice=min((fixed_choice,contig_choice),key=_choice_key)
    return dict(choice,constituent='fixed' if choice is fixed_choice else 'contiguous')


def safe_accept(partition,certificates,cap,K,method='coupled'):
    """The actual safe return path: proposed choice or explicitly labeled identity."""
    partition=validate_partition(partition,K);cap=rational(cap)
    if cap<1:raise ValueError('Identity requires cap >= 1')
    values=[];failures=[]
    for cell in partition:
        record=certificates.get(cell,{})
        if len(cell)==1:value=ONE
        elif record.get('status') not in ('full','boundary','failure'):value=None
        else:value=record.get(method)
        values.append(value)
        if not leq(value,cap):failures.append(cell)
    accepted=not failures;identity=tuple((i,) for i in range(K))
    proposed_certificate=None if any(x is None for x in values) else max(values,default=ONE)
    return {'proposed_partition':partition,'proposed_certificate':proposed_certificate,'accepted':accepted,
            'output_partition':partition if accepted else identity,
            'output_certificate':proposed_certificate if accepted else ONE,
            'identity_fallback':not accepted,'status':'accepted' if accepted else 'rejected-to-identity',
            'failed_cells':tuple(failures)}
