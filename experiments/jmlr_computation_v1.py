"""Fresh sequential computation study; all paths in the portable CLI are relative.

Generate compact count inputs once, preflight, freeze resources, then run. Only
this entry point launches cases. Historical evidence is never a timed input.
"""
from __future__ import annotations
import os
THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for _key in THREAD_KEYS: os.environ[_key]='1'
import argparse,atexit
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction as F
import fcntl,gzip,json,math,numbers,platform,re,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.set_int_max_str_digits(0)
PROJECT=Path(__file__).resolve().parents[1]
CONFIG=PROJECT/'configs/jmlr_computation_v1.json'

def now():return datetime.now(timezone.utc).isoformat()
def load(path):
    from experiments.jmlr_verified_adaptive_v1_math import decode
    p=Path(path)
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as stream:return decode(json.load(stream))
def dump(path,value):
    from experiments.jmlr_verified_adaptive_v1_math import encode
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
    with (gzip.open(temp,'wt',compresslevel=3) if p.suffix=='.gz' else temp.open('w')) as stream:
        json.dump(encode(value),stream,separators=(',',':'),allow_nan=False)
    temp.replace(p)
def root_for(cfg):
    p=Path(cfg['output_dir']);return p if p.is_absolute() else PROJECT/p
@contextmanager
def owner(root):
    with (root/'.owner.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)
def rss_bytes():
    v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v if sys.platform=='darwin' else 1024*v)
def environment():
    import numpy,scipy,flint
    def sysctl(key):
        try:return subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip()
        except (OSError,subprocess.CalledProcessError):return None
    return dict(python=sys.version,platform=platform.platform(),processor=sysctl('machdep.cpu.brand_string') or platform.processor(),
       physical_memory_bytes=sysctl('hw.memsize'),logical_cpus=os.cpu_count(),numpy=numpy.__version__,scipy=scipy.__version__,
       python_flint=flint.__version__,FLINT=flint.__FLINT_VERSION__,
       native_threads={key:os.environ[key] for key in THREAD_KEYS})

def extended_law(K,law_index):
    """The original stored-binary generator formula, then one exact residual."""
    import numpy as np
    from scipy.special import expit
    from experiments.jmlr_verified_pipeline_v1_laws import nominal_law
    mass,overlap=('balanced','moderate') if law_index==0 else ('skewed','weak')
    z=2*np.arange(K)/(K-1)-1
    p=np.ones(K) if mass=='balanced' else (np.arange(K)+1.)**(-1.5)
    p=p/p.sum();e=expit(math.log(3 if overlap=='moderate' else 19)*z)
    return nominal_law(dict(id=f'K{K}_{mass}_{overlap}_ordered',K=K,p=p.tolist(),e=e.tolist(),
          mass=mass,overlap=overlap,geometry='ordered',immutable_ids=list(range(K))),
          'jmlr_computation_v1: original controlled-law formula, design seed 20261008')

def cut128(p):
    p=F(p)
    if not 0<=p<=1 or p.denominator&(p.denominator-1) or p.denominator>2**128:
        raise ValueError('Exact dyadic probability does not fit the declared 128-bit sampler')
    v=p.numerator*((1<<128)//p.denominator)
    return v>>64,v&((1<<64)-1)
def draw_counts(law,law_index,replication,phase,cohort,master=2026101701):
    import numpy as np
    K=law['K'];n=(4000 if cohort==0 else 256)*K
    components={}
    def words(component):
        seed=[master,int(phase),K,int(law_index),int(replication),int(cohort),int(component)]
        components[str(component)]=seed
        w=np.random.PCG64(np.random.SeedSequence(seed)).random_raw(2*n)
        return w[:n],w[n:]
    high,low=words(0);cuts=[];total=F()
    for p in law['p'][:-1]:total+=p;cuts.append(cut128(total))
    hs=np.asarray([h for h,l in cuts],dtype=np.uint64)
    if len(set(map(int,hs)))!=len(hs):raise ValueError('Sampler requires distinct high-word cumulative cuts')
    S=np.searchsorted(hs,high,side='right')
    for j,(h,l) in enumerate(cuts):
        if l:S[(high==h)&(low<l)]=j
    def binary(probabilities,component):
        h,l=words(component);cs=[cut128(p) for p in probabilities]
        ch=np.asarray([min(a,2**64-1) for a,b in cs],dtype=np.uint64)
        cl=np.asarray([b for a,b in cs],dtype=np.uint64)
        result=(h<ch[S])|((h==ch[S])&(l<cl[S]))
        for j,p in enumerate(probabilities):
            if p==1:result[S==j]=True
        return result
    A=binary(law['e'],1)
    N=tuple(map(int,np.bincount(S,minlength=K)));T=tuple(map(int,np.bincount(S[A],minlength=K)))
    if cohort==0:return dict(N=N,T=T),components
    from experiments.jmlr_outcome_inference_v1_core import outcome_profiles
    Y=binary(outcome_profiles(law)['aligned'],2)
    V=tuple(map(int,np.bincount(S[A&Y],minlength=K)))
    return dict(N=N,T=T,V=V),components

def validate_input(data):
    expected={'version','case_id','law_id','law_index','K','replication','phase','N','T','immutable_ids','tree_orders','outcome','repeat_outcome'}
    if set(data)!=expected:raise ValueError('Operational input allowlist excludes population laws and saved evidence')
    K=data['K']
    for key in ('N','T','immutable_ids'):
        v=data[key]
        if len(v)!=K or any(isinstance(x,bool) or not isinstance(x,numbers.Integral) or x<0 for x in v):raise ValueError('Invalid '+key)
    if len(set(data['immutable_ids']))!=K or sum(data['N'])!=4000*K or any(t>n for n,t in zip(data['N'],data['T'])):raise ValueError('Invalid complete calibration support')
    if len(data['tree_orders'])!=2 or any(sorted(t)!=list(range(K)) for t in data['tree_orders']):raise ValueError('Invalid two-tree family')
    for out in (data['outcome'],data['repeat_outcome']):
        if out is None:continue
        if set(out)!={'N','T','V'} or any(len(out[k])!=K for k in out):raise ValueError('Invalid cohort support')
        if any(isinstance(x,bool) or not isinstance(x,numbers.Integral) for v in out.values() for x in v):raise ValueError('Noninteger cohort')
        if sum(out['N'])!=256*K or any(not 0<=v<=t<=n for n,t,v in zip(out['N'],out['T'],out['V'])):raise ValueError('Invalid cohort counts')
    return data


def public_inference(result):
    return {key:value for key,value in result.items() if key!='retained_assignment'}

def selection_summary(context,kind):
    if kind=='capacity':
        choices=context['by_capacity']
        return dict(by_capacity={str(k):v for k,v in choices.items()},
                    selection_complete=all(v.get('completed',v.get('status')=='finite') for v in choices.values()))
    c=dict(context['choice']);part=context['partition']
    return c|dict(groups=len(part),partition=part,identity=len(part)==context['region']['K'],
                  transported_budget=2*c.get('certificate',F(1)),
                  selection_complete=bool(c.get('count_optimal',False) and c.get('deterministic_tie_complete',False)))

def run_case(input_path,output_path,kind,cfg,root):
    # Imports are included in fresh-process wall time and visible process overhead,
    # not silently charged to a later procedure's numerical stage.
    from experiments.jmlr_computation_v1_provider import provider
    from experiments.jmlr_computation_v1_inference import infer_cohort,infer_repeated_cohort
    from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy
    started=time.perf_counter();progress_path=output_path.with_suffix('.progress.json')
    progress_overhead=0.;last_write=0.;stage='input_loading';data={}
    execution=load(root/'execution.json') if (root/'execution.json').exists() else cfg['preflight']
    memory_limit=execution['resident_limit_bytes']
    def progress(event):
        nonlocal progress_overhead,last_write,stage
        t=time.perf_counter();stage=event.get('stage',stage)
        if event.get('event')!='cells_completed' or t-last_write>5:
            dump(progress_path,dict(stage_reached=stage,elapsed_seconds=t-started,peak_rss_bytes=rss_bytes(),event=event));last_write=t
        if rss_bytes()>memory_limit:raise MemoryError('declared resident-memory limit exceeded')
        progress_overhead+=time.perf_counter()-t
    try:
        tick=time.perf_counter();data=validate_input(load(input_path));load_seconds=time.perf_counter()-tick
        policy=RootPolicy(**cfg['root_policy'])
        context=provider(data['N'],data['T'],data['immutable_ids'],data['tree_orders'],
             mode='capacity' if kind=='capacity' else 'threshold',policy=policy,progress_callback=progress)
        timings={'input_load_validate_seconds':load_seconds}
        measurements=context['measurements']
        timings.update(measurements['stage_seconds'])
        timings['provider_unallocated_seconds']=measurements['unallocated_seconds']
        chosen=selection_summary(context,kind);selection_done=time.perf_counter()
        evidence=dict(version=cfg['version'],case_id=data['case_id'],kind=kind,
            region={k:v for k,v in context['region'].items() if not k.startswith('_')},
            order=context['order'],selection=chosen,
            selected_records=[context['records'][tuple(c)] for c in context['partition']] if kind!='capacity' else
                 [context['records'][c] for c in sorted({tuple(c) for v in context['by_capacity'].values() if v.get('partition') for c in v['partition']})])
        result=None;repeated=None
        if kind!='capacity':
            progress(dict(stage='projection_export_and_inference',event='start'))
            out=data['outcome']
            result=infer_cohort(context['region'],context['partition'],context['records'],out['N'],out['T'],out['V'],
                  policy=policy,lp_time_limit=cfg['lp_endpoint_seconds'])
            timings.update(result['stage_seconds'])
            timings['aggregate_total_from_counts_seconds']=(selection_done-started)+result['totals']['aggregate_pipeline_seconds']
            timings['provider_total_from_counts_seconds']=selection_done-started
            timings['raw_reference_additional_seconds']=result['totals']['additional_raw_reference_seconds']
            timings.update(result['totals'])
            evidence['inference']=public_inference(result)
            if data['repeat_outcome'] is not None and kind=='primary':
                progress(dict(stage='repeated_cohort',event='start'));out=data['repeat_outcome'];tick=time.perf_counter()
                again=infer_repeated_cohort(result['retained_assignment'],out['N'],out['T'],out['V'],lp_time_limit=cfg['lp_endpoint_seconds'])
                repeated=dict(status='completed',timings=again['stage_seconds'],totals=again['totals'],wall_seconds=time.perf_counter()-tick,
                              reuse=again['reuse'],work=again['aggregate']['work'],methods=again['methods'],payload_bytes=again['payload_bytes'],
                              LP_verification=again['LP_verification'],root_fallbacks=again['aggregate']['root_fallbacks'])
                evidence['repeat_inference']=public_inference(again)
        else:timings['provider_total_from_counts_seconds']=selection_done-started
        stages_done=time.perf_counter()
        record={k:data[k] for k in ('case_id','law_id','law_index','K','replication')}
        record.update(kind=kind,status='completed',stage_reached='complete',reason=None,timings=timings,
            memory=dict(peak_rss_bytes=rss_bytes(),aggregate_pipeline_peak_rss_bytes=result['aggregate_pipeline_peak_rss_bytes'] if result else None,
                        unit='bytes; Darwin ru_maxrss is already bytes, Linux KiB multiplied by1024; lifetime peaks, never differences'),
            selection=chosen,counters=measurements['work'],root_work=measurements.get('root_work'),repeat=repeated,
            numerical=dict(provider=measurements.get('root_work'),cell_statuses=measurements['work'].get('cell_status_counts'),
                aggregate_root_fallbacks=result['aggregate']['root_fallbacks'] if result else None,
                raw_root_fallbacks=result['raw_reference'].get('root_fallbacks') if result else None,
                LP_verification=result['LP_verification'] if result else None,
                LP_status=result['LP'].get('status') if result else None,
                chronology=result['chronology'] if result else None,
                cache_initial_entries=result['cache_initial_entries'] if result else None),
            inference=dict(methods=result['methods'],work=result['aggregate']['work'],raw_work=result['raw_reference']['work'],
                           lp_work=result['raw_reference']['lp_work'],payload_bytes=result['payload_bytes'],reuse=result['reuse']) if result else None,
            inprocess_work_wall_seconds=stages_done-started,progress_instrumentation_seconds=progress_overhead,
            input_path=str(input_path.relative_to(root)),environment_threads={k:os.environ[k] for k in THREAD_KEYS})
        evidence_path=root/kind/'evidence'/(data['case_id']+'.json.gz')
        tick=time.perf_counter();dump(evidence_path,evidence);timings['evidence_serialization_seconds']=time.perf_counter()-tick
        record['evidence_bytes']=evidence_path.stat().st_size;record['evidence_path']=str(evidence_path.relative_to(root))
        # This record-write estimate is measured separately; the final parent
        # receipt write is in unallocated process overhead.
        tick=time.perf_counter();dump(output_path,record);timings['record_serialization_seconds']=time.perf_counter()-tick
        record['child_wall_until_serialization_seconds']=time.perf_counter()-started
        dump(output_path,record)
    except Exception as error:
        record={k:data.get(k) for k in ('case_id','law_id','law_index','K','replication')}
        record.update(kind=kind,status='memory_limit' if isinstance(error,MemoryError) else 'error',stage_reached=stage,
          reason=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),timings={},memory={'peak_rss_bytes':rss_bytes()},
          inprocess_work_wall_seconds=time.perf_counter()-started,repeat=None)
        dump(output_path,record);raise

def launch_case(root,cfg,input_path,kind,limit_seconds):
    data=load(input_path);output=root/kind/'records'/(data['case_id']+'.json.gz')
    if output.exists():return load(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    log=root/kind/'logs'/(data['case_id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();status=None
    command=[sys.executable,'-m','experiments.jmlr_computation_v1','case','--root',str(root),'--input',str(input_path),'--output',str(output),'--kind',kind]
    env=os.environ.copy();env.update({k:'1' for k in THREAD_KEYS});env['PYTHONPATH']=str(PROJECT)
    with log.open('w') as handle:
        child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:child.wait(timeout=limit_seconds)
        except subprocess.TimeoutExpired:
            status='timeout';os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    wall=time.perf_counter()-started
    if output.exists():record=load(output)
    else:
        prog=output.with_suffix('.progress.json');partial=load(prog) if prog.exists() else {}
        record={k:data[k] for k in ('case_id','law_id','law_index','K','replication')}
        record.update(kind=kind,status=status or 'error',stage_reached=partial.get('stage_reached','startup'),
          reason='per-case wall-clock resource limit' if status else 'process exited without completed record',timings={},
          memory={'peak_rss_bytes':partial.get('peak_rss_bytes')},partial_progress=partial,repeat=None)
    for key in ('case_id','law_id','law_index','K','replication'):
        if record.get(key) is None:record[key]=data[key]
    record['process_wall_seconds']=wall;record['exit_code']=child.returncode;record['process_id']=child.pid
    match=re.search(r'JMLR_PROCESS_PEAK_RSS_BYTES=(\d+)',log.read_text())
    if match:
        record['memory']['peak_rss_bytes']=int(match.group(1))
        record['memory']['peak_source']='child atexit getrusage(RUSAGE_SELF) lifetime high-water mark, including serialization; normalized bytes'
    # Whole-process overhead excludes all actual main/repeat work, leaving
    # imports/startup/shutdown and parent-visible unallocated time explicit.
    accounted=record.get('child_wall_until_serialization_seconds',record.get('inprocess_work_wall_seconds',0))
    record['process_overhead_seconds']=max(0.,wall-accounted)
    record['process_wall_accounting_residual_seconds']=wall-accounted-record['process_overhead_seconds']
    dump(output,record)
    print(json.dumps(dict(kind=kind,case_id=data['case_id'],status=record['status'],wall_seconds=wall,stage=record['stage_reached'])),flush=True)
    return record

def run_panel(root,cfg,preflight=False,limit=None):
    if preflight:
        inputs=sorted((root/'inputs/preflight').glob('*.json.gz'),key=lambda p:(load(p)['K'],p.name));case_limit=cfg['preflight']['case_limit_seconds']
        for path in inputs[:limit]:launch_case(root,cfg,path,'preflight',case_limit)
        return
    if not (root/'execution.json').exists():raise RuntimeError('Run separate preflight and freeze resource limits first')
    execution=load(root/'execution.json');paths=sorted((root/'inputs/primary').glob('*.json.gz'),key=lambda p:(load(p)['K'],p.name))
    jobs=[(p,'primary') for p in paths]
    jobs += [(p,'capacity') for p in paths if load(p)['K'] in cfg['capacity_support_sizes'] and load(p)['replication']<5]
    completed=0
    for path,kind in jobs:
        output=root/kind/'records'/(load(path)['case_id']+'.json.gz')
        if output.exists():continue
        if limit is not None and completed>=limit:break
        launch_case(root,cfg,path,kind,execution['capacity_case_limit_seconds' if kind=='capacity' else 'primary_case_limit_seconds']);completed+=1
        manifest=load(root/'manifest.json')
        for k in ('primary','capacity'):
            rr=[load(p) for p in (root/k/'records').glob('*.json.gz')]
            manifest['completed_'+k]=len(rr);manifest['successful_'+k]=sum(r['status']=='completed' for r in rr)
            if k=='primary':manifest['completed_repeats']=sum(r.get('repeat') is not None for r in rr)
        finished=manifest.get('completed_primary')==200 and manifest.get('completed_capacity')==30
        manifest.update(status='computed' if finished else 'running',last_update_utc=now());dump(root/'manifest.json',manifest)

def freeze_resources(root,cfg):
    if (root/'execution.json').exists():raise FileExistsError('Resource configuration already frozen')
    rows=[load(p) for p in (root/'preflight/records').glob('*.json.gz')]
    if len(rows)!=4:raise RuntimeError('All four committed preflight records are required')
    largest=max(r['process_wall_seconds'] for r in rows)
    # Capacity all-count DP can be substantially dearer than threshold DP;
    # keep a generous independent bound rather than infer it from fast fallback.
    primary_limit=max(600,math.ceil(4*largest));capacity_limit=max(1800,math.ceil(8*largest))
    execution=dict(frozen_utc=now(),active_timed_cases=1,native_threads=1,
       primary_case_limit_seconds=primary_limit,capacity_case_limit_seconds=capacity_limit,
       resident_limit_bytes=8589934592,preflight_max_wall_seconds=largest,
       rationale='primary max(600s,4x largest preflight); capacity max(1800s,8x largest preflight); 8GiB lifetime resident limit checked at provider checkpoints',
       forecast_primary_seconds=20*sum(r['process_wall_seconds'] for r in rows)*1.5,
       forecast_caveat='four development cases cannot identify tail latency; capacity all-count work forecast is uncertain',
       implementation='original verified arithmetic, new instrumentation only')
    dump(root/'execution.json',execution);print(json.dumps(execution),flush=True)

def verify_evidence(path):
    from experiments.jmlr_computation_v1_inference import verify_saved_evidence
    e=load(path);result=verify_saved_evidence(e['inference'])
    from experiments.jmlr_verified_pipeline_v1_replay import RigorousRootReplay
    from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy
    replay=RigorousRootReplay(RootPolicy(**e['region']['root_policy']))
    r=e['region'];K=r['K'];gamma=F(1,40)/(2*K)
    if r['alpha']!=F(1,40) or r['gamma']!=gamma:raise AssertionError('Saved allocation differs')
    for i,record in enumerate(r['roots']['propensity']):
        if (record['x'],record['m'],record['gamma'])!=(r['T'][i],r['N'][i],gamma):raise AssertionError('Saved propensity counts differ')
        if (r['propensity_lower'][i],r['propensity_upper'][i])!=(record['lower'],record['upper']):raise AssertionError('Saved propensity bounds differ')
        replay(record)
    for i,record in enumerate(r['roots']['mass']):
        if (record['x'],record['m'],record['gamma'])!=(r['N'][i],sum(r['N']),gamma):raise AssertionError('Saved mass counts differ')
        if (r['mass_lower'][i],r['mass_upper'][i])!=(record['lower'],record['upper']):raise AssertionError('Saved mass bounds differ')
        replay(record)
    for field in ('z_outcome_roots','raw_outcome_roots'):
        for record in e['inference'][field]:replay(record)
    from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
    from experiments.jmlr_verified_adaptive_v1_math import region_from_bounds
    r=e['region'];region=region_from_bounds(r['mass_lower'],r['mass_upper'],r['propensity_lower'],r['propensity_upper'])
    if 'K' not in region:raise AssertionError('Malformed saved region')
    count=0
    for record in e['selected_records']:
        recalculated=cell_certificate(region,record['cell'])
        if recalculated['status']!='full' or recalculated['coupled']!=record['coupled']:raise AssertionError('Saved cell certificate mismatch')
        count+=1
    result['selected_cells_verified']=count;result['roots']=replay.receipt()
    result['scope']='fresh Arb sign replay, selected-cell recomputation and exact LP certificate checks; not a replay of search optimality or raw-data sampling'
    from experiments.jmlr_verified_adaptive_v1_math import encode
    print(json.dumps(encode(result)),flush=True);return result


def main():
    parser=argparse.ArgumentParser(description='Internal isolated SCA computational case worker')
    commands=parser.add_subparsers(dest='command',required=True)
    case=commands.add_parser('case')
    for name in ('root','input','output'):case.add_argument('--'+name,type=Path,required=True)
    case.add_argument('--kind',choices=('preflight','primary','capacity'),required=True)
    args=parser.parse_args()
    cfg=load(args.root/'configuration.json')
    run_case(args.input,args.output,args.kind,cfg,args.root)

if __name__=='__main__':main()
