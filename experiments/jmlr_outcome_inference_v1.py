"""Launch owner for the fixed finite-sample outcome study and evidence replay.

Run from project root with .venv/bin/python -m experiments.jmlr_outcome_inference_v1
followed by freeze, preflight, run, replay, or report. No historical entry point
is launched or patched. New evidence is exclusively under this study's root.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import sys
sys.set_int_max_str_digits(0)
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed,TimeoutError
from contextlib import contextmanager
from datetime import datetime,timezone
from fractions import Fraction as F
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import time
import traceback
import numpy as np

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy,RootVerifier
from experiments.jmlr_verified_pipeline_v1_replay import RigorousRootReplay
from experiments.jmlr_outcome_inference_v1_core import (
    choose_calibration,infer_outcome,outcome_profiles,diagnose,PROFILES,clean_mathematical)
from experiments.jmlr_outcome_inference_v1_sampling import (
    sample_block,stream_record,counts,threshold,verify_observations)

PROJECT=Path(__file__).resolve().parents[1]
CONFIG=PROJECT/'configs/jmlr_outcome_inference_v1.json'
ROOT=PROJECT/'results/jmlr_outcome_inference_v1'


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return exact.decode(json.loads(Path(path).read_text()))


def dump(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(exact.encode(value),indent=2)+'\n');os.replace(temporary,path)


def read_block(path):
    with gzip.open(path,'rt') as stream:return exact.decode(json.load(stream))


def write_block(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with gzip.open(temporary,'wt',compresslevel=6) as stream:
        json.dump(exact.encode(value),stream,separators=(',',':'))
    os.replace(temporary,path)


@contextmanager
def ownership(root):
    with (root/'.owner.lock').open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A study owner is already active; no duplicate launch')
        handle.seek(0);handle.truncate();handle.write(str(os.getpid()));handle.flush()
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)








def block_path(root,i,j):return root/'blocks'/f'{i}_{j:03d}.json.gz'


def process_block(root,i,j,protocol,preflight=False):
    start=time.perf_counter();law=protocol['laws'][i];K=law['K'];root=Path(root)
    (root/'blocks').mkdir(exist_ok=True,parents=True);(root/'observations').mkdir(exist_ok=True,parents=True)
    path=block_path(root,i,j);obs_path=root/'observations'/f'{i}_{j:03d}.npz'
    if path.exists() or obs_path.exists():raise FileExistsError('Refusing to overwrite sampled block '+str(path))
    tick=time.perf_counter();obs,streams=sample_block(law,i,j,preflight);verify_observations(obs,law)
    N,T=counts(obs['cal_S'],obs['cal_A'],K=K)
    sampling_seconds=time.perf_counter()-tick
    verifier=RootVerifier(RootPolicy(**protocol['configuration']['root_policy']))
    tick=time.perf_counter()
    cal=choose_calibration(N,T,protocol['tree_orders'][str(K)],law['immutable_ids'],verifier)
    calibration_seconds=time.perf_counter()-tick
    profiles=outcome_profiles(law);evaluations=[];tick=time.perf_counter()
    for pi,profile in enumerate(PROFILES):
        for ratio in protocol['configuration']['outcome_m_per_K']:
            m=ratio*K
            on,ot,ov=counts(obs['out_S'][:m],obs['out_A'][:m],obs['out_Y'][pi,:m],K=K)
            inference=infer_outcome(cal,on,ot,ov,verifier)
            diagnostics=diagnose(law,profiles[profile],cal,inference)
            evaluations.append({'profile':profile,'m_ratio':ratio,'m':m,
                                'inference':inference,'diagnostics':diagnostics})
    outcome_seconds=time.perf_counter()-tick
    record={'version':'jmlr_outcome_inference_v1','law_id':law['id'],'law_index':i,'K':K,
            'replication':j,'preflight':preflight,'streams':streams,
            'observations':str(obs_path.relative_to(root)),
            'calibration':cal,'evaluations':evaluations,
            'root_cache':{'hits':verifier.cache_hits,'misses':verifier.cache_misses},
            'timings':{'sampling_seconds':sampling_seconds,'calibration_seconds':calibration_seconds,
                       'outcome_seconds':outcome_seconds}}
    tick=time.perf_counter()
    temporary=obs_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**obs)
    os.replace(temporary,obs_path)
    write_block(path,record)
    serialization_seconds=time.perf_counter()-tick
    return {'law_index':i,'replication':j,'path':str(path),'evaluations':12,
            'sampling_seconds':sampling_seconds,'calibration_seconds':calibration_seconds,
            'outcome_seconds':outcome_seconds,'serialization_seconds':serialization_seconds,
            'compute_seconds':time.perf_counter()-start,'bytes':path.stat().st_size+obs_path.stat().st_size,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def collect_roots(value,result=None):
    result={} if result is None else result
    if isinstance(value,dict):
        if {'x','m','gamma','lower_bracket','upper_bracket','lower','upper'}<=set(value):
            key=value['x'],value['m'],value['gamma']
            if key in result and clean_mathematical(result[key])!=clean_mathematical(value):
                raise AssertionError('Conflicting evidence for one count/error interval')
            result[key]=value
        else:
            for v in value.values():collect_roots(v,result)
    elif isinstance(value,(list,tuple)):
        for v in value:collect_roots(v,result)
    return result


class SavedVerifier:
    def __init__(self,intervals,policy):
        self.intervals=intervals;self.policy=RootPolicy(**policy);self.cache_hits=self.cache_misses=0
    def interval(self,x,m,gamma):
        key=int(x),int(m),F(gamma)
        if key not in self.intervals:raise AssertionError('Missing saved root evidence '+str(key))
        self.cache_hits+=1;return self.intervals[key]


def replay_block(root,i,j,protocol):
    start=time.perf_counter();root=Path(root);record=read_block(block_path(root,i,j))
    law=protocol['laws'][i];K=law['K']
    if (record['law_id'],record['law_index'],record['K'],record['replication'])!=(law['id'],i,K,j):
        raise AssertionError('Block identity changed')
    if record['streams']!=stream_record(i,j,record['preflight']):raise AssertionError('Random-stream identity changed')
    with np.load(root/record['observations'],allow_pickle=False) as z:obs={k:z[k] for k in z.files}
    verify_observations(obs,law)
    N,T=counts(obs['cal_S'],obs['cal_A'],K=K)
    if tuple(record['calibration']['N'])!=N or tuple(record['calibration']['T'])!=T:
        raise AssertionError('Selection counts differ from saved observations')
    roots=collect_roots(record);checker=RigorousRootReplay(RootPolicy(**protocol['configuration']['root_policy']))
    for r in roots.values():checker(r)
    verifier=SavedVerifier(roots,protocol['configuration']['root_policy'])
    cal=choose_calibration(N,T,protocol['tree_orders'][str(K)],law['immutable_ids'],verifier)
    if clean_mathematical(cal)!=clean_mathematical(record['calibration']):
        raise AssertionError('Complete family selection replay differs')
    profiles=outcome_profiles(law);seen=set();lp_certificates=0
    for ev in record['evaluations']:
        profile,ratio=ev['profile'],ev['m_ratio'];m=ratio*K
        if (profile,ratio) in seen or profile not in PROFILES or ratio not in protocol['configuration']['outcome_m_per_K'] or ev['m']!=m:
            raise AssertionError('Repeated or changed outcome operating point')
        seen.add((profile,ratio));pi=PROFILES.index(profile)
        on,ot,ov=counts(obs['out_S'][:m],obs['out_A'][:m],obs['out_Y'][pi,:m],K=K)
        inference=infer_outcome(cal,on,ot,ov,verifier,lp_saved=ev['inference']['LP'])
        if clean_mathematical(inference)!=clean_mathematical(ev['inference']):
            raise AssertionError('Outcome enclosure or exact LP evidence replay differs')
        diagnostics=diagnose(law,profiles[profile],cal,inference)
        if clean_mathematical(diagnostics)!=clean_mathematical(ev['diagnostics']):
            raise AssertionError('Population/coverage/decomposition replay differs')
        lp_certificates+=2*(not inference['LP']['fallback'])
    if len(seen)!=12:raise AssertionError('Missing profile/prefix evaluation')
    return {'law_index':i,'replication':j,'status':'passed','selection_replays':1,'outcome_replays':12,
            'LP_bound_certificates':lp_certificates,'root_stats':checker.receipt(),
            'compute_seconds':time.perf_counter()-start}


def benchmark_block(root,i,j,protocol):
    start=time.perf_counter();result=process_block(root,i,j,protocol,True)
    replay=replay_block(root,i,j,protocol)
    return dict(result,replay_seconds=replay['compute_seconds'],root_stats=replay['root_stats'],
                total_path_seconds=time.perf_counter()-start)


def parallel(tasks,worker,workers,remaining,progress):
    context=mp.get_context('spawn');executor=ProcessPoolExecutor(max_workers=workers,mp_context=context)
    futures={executor.submit(worker,*args):args for args in tasks};results=[]
    try:
        for future in as_completed(futures,timeout=remaining):
            answer=future.result();results.append(answer);progress(answer,len(results),len(tasks))
    except BaseException:
        for f in futures:f.cancel()
        for process in executor._processes.values():process.terminate()
        executor.shutdown(wait=True,cancel_futures=True);raise
    else:executor.shutdown(wait=True)
    return results








