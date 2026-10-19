"""Fresh-process rigorous sign replay and exact saved-decision reconstruction.

Evaluates every distinct nonstructural bracket with Arb and repeats exact cell
optimization/selection from the saved observations without sampling.
"""
from __future__ import annotations
from fractions import Fraction as Q
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments import jmlr_verified_adaptive_v1_replay_checks as linkage
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, tail_difference, ball_bounds
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate


def same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


class RigorousRootReplay:
    """Deduplicate proved root inputs, endpoints, and immutable root policy."""
    def __init__(self,policy=None):
        self.policy=RootPolicy() if policy is None else policy
        self.seen=set();self.records=0;self.sign_evaluations=0;self.cache_hits=0

    def __call__(self,record):
        x,m,gamma=record['x'],record['m'],record['gamma']
        linkage._root(record,x,m,gamma,self.policy.grid_bits)
        self.records+=1
        for successes,bracket in ((x,record['lower_bracket']),
                                  (m-x,record['upper_bracket']['reflected_bracket'])):
            key=(successes,m,gamma,bracket['low'],bracket['high'],bracket['status'],self.policy)
            if key in self.seen:
                self.cache_hits+=1;continue
            if bracket['status']=='verified':
                for endpoint,direction in ((bracket['low'],-1),(bracket['high'],1)):
                    if endpoint in (0,1):
                        difference=endpoint-gamma/2
                        if not (difference<0 if direction<0 else difference>0):
                            raise AssertionError('Structural incomplete-beta sign differs')
                        continue
                    proved=False;precision=self.policy.initial_precision
                    while precision<=self.policy.max_precision:
                        ball=tail_difference(endpoint,successes,m-successes+1,gamma/2,precision)
                        self.sign_evaluations+=1
                        if ball.is_finite():
                            lo,hi=ball_bounds(ball)
                            if hi<0 if direction<0 else lo>0:
                                proved=True;break
                        precision*=2
                    if not proved:raise AssertionError('Fresh Arb evaluation cannot prove saved bracket sign')
            elif bracket['status'] not in ('structural','fallback'):
                raise AssertionError('Unsupported saved root status')
            self.seen.add(key)
        return True

    def receipt(self):
        return {'root_records':self.records,'distinct_brackets':len(self.seen),
                'fresh_arb_sign_evaluations':self.sign_evaluations,'deduplicated_brackets':self.cache_hits}


def region_replay(region,law,root_checker):
    K=law['K'];N=tuple(region['N']);T=tuple(region['T']);n=sum(N)
    known=region['mode']=='known_mass';gamma=Q(1,20)/(K if known else 2*K)
    if region['alpha']!=Q(1,20) or region['gamma']!=gamma or region['n']!=n:
        raise AssertionError('Region constants/count total changed')
    if known and not same(region['known_masses'],law['p']):
        raise AssertionError('Known mass region differs from the declared nominal law')
    if not known and region['known_masses'] is not None:raise AssertionError('Unknown mode uses population masses')
    roots=region['roots'];prop=roots['propensity'];mass=roots['mass']
    if len(prop)!=K or (known and mass is not None) or (not known and len(mass)!=K):
        raise AssertionError('Missing coordinate root evidence')
    for i,record in enumerate(prop):
        linkage._root(record,T[i],N[i],gamma,region['root_policy']['grid_bits']);root_checker(record)
    if not known:
        for i,record in enumerate(mass):
            linkage._root(record,N[i],n,gamma,region['root_policy']['grid_bits']);root_checker(record)
    a=tuple(law['p']) if known else tuple(r['lower'] for r in mass)
    b=a if known else tuple(r['upper'] for r in mass)
    l=tuple(r['lower'] for r in prop);u=tuple(r['upper'] for r in prop)
    rebuilt=exact.region_from_bounds(a,b,l,u,tuple(law['p']) if known else None)
    for key in ('K','mass_lower','mass_upper','propensity_lower','propensity_upper','known_masses','mode','valid','diagnostics'):
        if not same(region[key],rebuilt[key]):raise AssertionError('Saved region reconstruction differs: '+key)
    if region['root_fallbacks']!=sum(r['status']=='fallback' for r in prop+(mass or [])):
        raise AssertionError('Root fallback diagnostic changed')
    return rebuilt


def cells_replay(records,region,*,early_rejection):
    """Recompute every saved optimizer value, including partial rejection anchors."""
    linkage.validate_records(records,region)
    region=exact.region_from_bounds(region['mass_lower'],region['mass_upper'],
                                   region['propensity_lower'],region['propensity_upper'],region['known_masses'])
    for record in records:
        rebuilt=cell_certificate(region,tuple(record['cell']),max_cap=Q(2) if early_rejection else None)
        if not same(record,rebuilt):raise AssertionError('Exact cell calculation changed: '+str(record['cell']))
    return True
