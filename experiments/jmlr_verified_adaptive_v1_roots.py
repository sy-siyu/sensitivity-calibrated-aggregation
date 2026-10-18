"""Verified binomial root brackets using python-flint 0.8.0 / Arb.

SciPy supplies an untrusted starting point only. Every accepted sign follows
from an Arb enclosure of I_z(a,b)-t with exact rational t. Decisions never use
midpoints, decimal strings, additive tolerances, or the proposal's accuracy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from fractions import Fraction
from functools import lru_cache
import math
import numbers
import time
from flint import arb, ctx
from scipy.special import betaincinv


def rational(value):
    if isinstance(value, Fraction): return value
    if isinstance(value, bool): raise ValueError('Boolean is not a rational input')
    if isinstance(value, int): return Fraction(value)
    if isinstance(value, str): return Fraction(value)
    # Accept integer scalar objects without accepting binary floats.
    if type(value).__module__.startswith('numpy') and type(value).__name__.startswith(('int','uint')):
        return Fraction(int(value))
    raise ValueError('Use explicit integers, Fraction, or rational strings; floats are not exact input constants')


def fraction_json(value):
    value=rational(value)
    return {'numerator':str(value.numerator),'denominator':str(value.denominator)}


def fraction_from_json(value):
    return Fraction(int(value['numerator']),int(value['denominator']))


def _arb_rational(value):
    value=rational(value)
    return arb(value.numerator)/arb(value.denominator)


def _dyadic_from_arb(value):
    mantissa,exponent=value.man_exp(); m,e=int(mantissa),int(exponent)
    return Fraction(m*(1<<e)) if e>=0 else Fraction(m,1<<(-e))


def ball_bounds(value):
    if not value.is_finite(): raise ArithmeticError('Nonfinite Arb enclosure')
    return (_dyadic_from_arb(value.lower()),_dyadic_from_arb(value.upper()))


def tail_difference(z,a,b,t,precision=128):
    z,t=rational(z),rational(t)
    if not 0<=z<=1 or not isinstance(a,int) or not isinstance(b,int) or a<=0 or b<=0:
        raise ValueError('Invalid incomplete-beta arguments')
    with ctx.workprec(int(precision)):
        return _arb_rational(z).beta_lower(a,b,regularized=True)-_arb_rational(t)


@dataclass(frozen=True)
class RootPolicy:
    grid_bits: int=48
    initial_precision: int=128
    max_precision: int=1024
    max_evaluations: int=192
    endpoint_seconds: float=5.0
    max_trials: int=1_000_000
    def __post_init__(self):
        if not 4<=self.grid_bits<=256 or self.initial_precision<32 or self.max_precision<self.initial_precision or self.max_evaluations<0 or self.endpoint_seconds<0 or self.max_trials<0:
            raise ValueError('Invalid finite root policy')


def _counts(x,m):
    if isinstance(x,(bool,)) or isinstance(m,(bool,)) or not isinstance(x,numbers.Integral) or not isinstance(m,numbers.Integral):
        raise ValueError('Counts must be integers')
    x,m=int(x),int(m)
    if m<0 or not 0<=x<=m: raise ValueError('Counts require 0 <= x <= m')
    return x,m


def _root_bracket(x,m,gamma,policy,proposal=None):
    """Bracket lower CP root I_z(x,m-x+1)=gamma/2; x>0."""
    start=time.monotonic(); D=1<<policy.grid_bits; t=gamma/2
    evaluations=0; max_precision_used=0; evidence={}
    def fail(reason):
        return {'low':Fraction(0),'high':Fraction(1),'width':Fraction(1),
                'status':'fallback','reason':reason,'evaluations':evaluations,
                'precision':max_precision_used,'elapsed_seconds':time.monotonic()-start,
                'sign_low':-1,'sign_high':1,'evidence':{},'expansion_bound':Fraction(1)}
    if m>policy.max_trials: return fail('trial-budget')
    if policy.max_evaluations==0 or policy.endpoint_seconds==0: return fail('endpoint-resource-budget')
    def sign(g):
        nonlocal evaluations,max_precision_used
        if g==0: return -1
        if g==D: return 1
        if g in evidence: return evidence[g]['sign']
        precision=policy.initial_precision
        while precision<=policy.max_precision:
            if evaluations>=policy.max_evaluations or time.monotonic()-start>policy.endpoint_seconds:
                raise TimeoutError('endpoint-resource-budget')
            diff=tail_difference(Fraction(g,D),x,m-x+1,t,precision)
            evaluations+=1;max_precision_used=max(max_precision_used,precision)
            if diff.is_finite():
                lo,hi=ball_bounds(diff)
                if hi<0 or lo>0:
                    s=-1 if hi<0 else 1
                    evidence[g]={'sign':s,'difference_lower':lo,'difference_upper':hi,'precision':precision}
                    return s
            precision*=2
        return 0 # No sign proof; equality may lie here. Never accept this point.
    try:
        if proposal is None:
            proposal=float(betaincinv(x,m-x+1,float(t)))
        if isinstance(proposal,Fraction): p=proposal
        elif isinstance(proposal,(int,float)) and math.isfinite(proposal): p=Fraction(proposal)
        else: p=Fraction(1,2)
        g=min(D,max(0,(p.numerator*D)//p.denominator))
        # Widen until both strict directions are proved. Arbitrarily wrong or
        # exactly-root proposals are safe; unknown signs are never accepted.
        radius=1
        while True:
            low=max(0,g-radius);high=min(D,g+radius+1)
            sl,sh=sign(low),sign(high)
            if sl==-1 and sh==1: break
            if radius>=D: raise ArithmeticError('Could not prove structural root bracket')
            radius=min(D,radius*2)
        # Refine a wide bracket to at most two grid spacings. If midpoint sign
        # remains undecidable, prove its neighboring grid points instead.
        while high-low>2:
            middle=(low+high)//2;sm=sign(middle)
            if sm<0: low=middle
            elif sm>0: high=middle
            else:
                left,right=max(low,middle-1),min(high,middle+1)
                if sign(left)==-1 and sign(right)==1: low,high=left,right
                else: raise ArithmeticError('Unresolved root sign at maximum precision')
        L,H=Fraction(low,D),Fraction(high,D)
        def endpoint_evidence(g,s):
            if g in (0,D):
                value=-t if g==0 else 1-t
                return {'sign':s,'difference_lower':value,'difference_upper':value,'precision':0,'structural':True}
            return evidence[g]
        return {'low':L,'high':H,'width':H-L,'status':'verified','reason':None,
                'sign_low':-1,'sign_high':1,'precision':max_precision_used,
                'evaluations':evaluations,'elapsed_seconds':time.monotonic()-start,
                'evidence':{'low':endpoint_evidence(low,-1),'high':endpoint_evidence(high,1)},
                'expansion_bound':H-L}
    except Exception as error:
        return fail(type(error).__name__+': '+str(error))


def _structural(value):
    value=Fraction(value)
    return {'low':value,'high':value,'width':Fraction(0),'status':'structural',
            'reason':None,'sign_low':0,'sign_high':0,'precision':0,'evaluations':0,
            'elapsed_seconds':0.,'evidence':{},'expansion_bound':Fraction(0)}


class RootVerifier:
    """Process-local exact-key cache. Numerical policy is immutable per instance."""
    def __init__(self,policy=None):
        self.policy=RootPolicy() if policy is None else policy
        self.cache={};self.cache_hits=0;self.cache_misses=0
    def root(self,x,m,gamma,proposal=None):
        key=(x,m,gamma,self.policy,proposal)
        if key in self.cache:
            self.cache_hits+=1;return self.cache[key]
        self.cache_misses+=1
        answer=_root_bracket(x,m,gamma,self.policy,proposal)
        self.cache[key]=answer
        return answer
    def interval(self,x,m,gamma,proposal=None):
        x,m=_counts(x,m);gamma=rational(gamma)
        if not 0<gamma<1: raise ValueError('Individual error must be in (0,1)')
        lower=_structural(0) if x==0 else self.root(x,m,gamma,proposal)
        reflected=_structural(0) if x==m else self.root(m-x,m,gamma,proposal)
        upper=dict(reflected,low=1-reflected['high'],high=1-reflected['low'],reflected=True,
                   sign_low=reflected['sign_high'],sign_high=reflected['sign_low'],
                   evidence={'low':reflected['evidence'].get('high',{}),'high':reflected['evidence'].get('low',{})},
                   reflected_bracket=reflected)
        failed=lower['status']=='fallback' or reflected['status']=='fallback'
        status='fallback' if failed else ('structural' if m==0 else 'verified')
        L,U=(Fraction(0),Fraction(1)) if failed else (lower['low'],upper['high'])
        return {'x':x,'m':m,'gamma':gamma,'lower':L,'upper':U,'status':status,
                'lower_bracket':lower,'upper_bracket':upper,
                'lower_expansion_bound':Fraction(1) if failed else lower['width'],
                'upper_expansion_bound':Fraction(1) if failed else reflected['width']}


def cp_interval(x,m,gamma,policy=None,proposal=None):
    return RootVerifier(policy).interval(x,m,gamma,proposal)
