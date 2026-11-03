"""Integer-threshold sampling of the frozen dyadic nominal laws.

An ideal independent uniform 64-bit word gives the exact stated probability;
PCG64 is a reproducible pseudorandom implementation, not a proved iid source.
"""
from fractions import Fraction as F
import numpy as np
from experiments.jmlr_outcome_inference_v1_core import outcome_profiles,PROFILES

STUDY_SEED=20261014
PREFLIGHT_SEED=2026101401
COMPONENTS=('calibration_states','calibration_actions','outcome_states',
            'outcome_actions','treated_outcome_uniforms','control_outcome_uniforms')


def threshold(probability):
    p=F(probability)
    if not 0<=p<=1 or p.denominator&(p.denominator-1) or p.denominator>2**64:
        raise ValueError('Sampler needs a dyadic probability representable on 64 bits')
    return p.numerator*((2**64)//p.denominator)


def categorical(words,probabilities):
    probabilities=tuple(map(F,probabilities))
    if any(p<=0 for p in probabilities) or sum(probabilities)!=1:
        raise ValueError('Categorical law must be positive and exactly normalized')
    cumulative=[];total=F()
    for p in probabilities[:-1]:
        total+=p;cumulative.append(threshold(total))
    thresholds=np.asarray(cumulative,dtype=np.uint64)
    return np.searchsorted(thresholds,np.asarray(words,dtype=np.uint64),side='right').astype(np.uint8)


def bernoulli(words,probabilities):
    words=np.asarray(words,dtype=np.uint64)
    probabilities=tuple(map(F,probabilities))
    if words.shape!=(len(probabilities),):
        raise ValueError('One Bernoulli threshold per observation is required')
    # Exact handling of probability one avoids casting 2**64 into uint64.
    cuts=np.asarray([min(2**64-1,threshold(p)) for p in probabilities],dtype=np.uint64)
    result=(words<cuts).astype(np.uint8)
    for i,p in enumerate(probabilities):
        if p==1:result[i]=1
    return result


def counts(S,A,Y=None,K=None):
    if K is None:raise ValueError('The known population support size is required')
    S,A=np.asarray(S),np.asarray(A)
    if S.shape!=A.shape or np.any(S<0) or np.any(S>=K) or np.any((A!=0)&(A!=1)):
        raise ValueError('Malformed observed assignments')
    N=np.bincount(S,minlength=K);T=np.bincount(S[A==1],minlength=K)
    if Y is None:return tuple(map(int,N)),tuple(map(int,T))
    Y=np.asarray(Y)
    if Y.shape!=S.shape or np.any((Y!=0)&(Y!=1)):raise ValueError('Malformed binary labels')
    V=np.bincount(S[(A==1)&(Y==1)],minlength=K)
    return tuple(map(int,N)),tuple(map(int,T)),tuple(map(int,V))


def stream_record(law_index,replication,preflight=False):
    master=PREFLIGHT_SEED if preflight else STUDY_SEED
    return {name:[master,int(law_index),int(replication),component]
            for component,name in enumerate(COMPONENTS)}


def sample_block(law,law_index,replication,preflight=False):
    streams=stream_record(law_index,replication,preflight)
    def words(component,n):
        return np.random.PCG64(np.random.SeedSequence(streams[component])).random_raw(n)
    K=law['K'];ncal=4000*K;mmax=256*K
    calS=categorical(words('calibration_states',ncal),law['p'])
    # Vectorize thresholds without converting any probability to a float.
    e_thresholds=np.asarray([threshold(x) for x in law['e']],dtype=np.uint64)
    calA=(words('calibration_actions',ncal)<e_thresholds[calS]).astype(np.uint8)
    S=categorical(words('outcome_states',mmax),law['p'])
    A=(words('outcome_actions',mmax)<e_thresholds[S]).astype(np.uint8)
    uy=words('treated_outcome_uniforms',mmax);u0=words('control_outcome_uniforms',mmax)
    profiles=outcome_profiles(law)
    Y=[]
    for name in PROFILES:
        cuts=np.asarray([threshold(x) for x in profiles[name]],dtype=np.uint64)
        Y.append(np.where(A==1,uy<cuts[S],u0<np.uint64(2**63)).astype(np.uint8))
    return {'cal_S':calS,'cal_A':calA,'out_S':S,'out_A':A,'out_Y':np.asarray(Y,dtype=np.uint8),
            'treated_uniforms':uy,'control_uniforms':u0},streams


def verify_observations(observations,law):
    K=law['K'];profiles=outcome_profiles(law)
    assert len(observations['cal_S'])==4000*K and len(observations['out_S'])==256*K
    S,A=observations['out_S'],observations['out_A']
    assert observations['out_Y'].shape==(3,256*K)
    for j,name in enumerate(PROFILES):
        cuts=np.asarray([threshold(x) for x in profiles[name]],dtype=np.uint64)
        expected=np.where(A==1,observations['treated_uniforms']<cuts[S],
                          observations['control_uniforms']<np.uint64(2**63)).astype(np.uint8)
        assert np.array_equal(observations['out_Y'][j],expected)
    counts(observations['cal_S'],observations['cal_A'],K=K)
    counts(S,A,observations['out_Y'][0],K=K)
    return True
