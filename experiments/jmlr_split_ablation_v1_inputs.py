"""Read-only adapter for preserved split-LTT and verified full-data evidence.

No observations, proposals, roots, or certificates are generated here. Returned
regions and candidate dictionaries retain their saved contents. Population laws
and population-truth evaluation deliberately do not enter this interface.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction as F
import gzip
import json
import math
import numbers
from pathlib import Path

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, rational
from experiments.jmlr_verified_pipeline_v1_core import exact_order

ALPHA = F(1,20)
CAPS = (F(5,4), F(3,2), F(2))


def _read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path,'rt') as stream:
            return exact.decode(json.load(stream))
    return exact.decode(json.loads(path.read_text()))


def _same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(_same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    return a==b


def _integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value<minimum:
        raise ValueError(name+' must be an integer at least '+str(minimum))
    return int(value)


def _counts(N,T,name):
    N,T = tuple(N),tuple(T)
    if not N or len(N)!=len(T):
        raise ValueError(name+' counts must have a common nonempty support')
    N = tuple(_integer(v,name+' N') for v in N)
    T = tuple(_integer(v,name+' T') for v in T)
    if any(t>n for n,t in zip(N,T)):
        raise ValueError(name+' counts require 0 <= T <= N')
    return N,T


def _policy(value):
    if isinstance(value,RootPolicy):
        return asdict(value)
    if not isinstance(value,dict) or set(value)!=set(asdict(RootPolicy())):
        raise ValueError('Numerical policy fields differ from the established root policy')
    policy = RootPolicy(**value)
    return asdict(policy)


def _partition(value,K):
    return exact.validate_partition(value,K)


def _root(record,x,m,gamma):
    if _integer(record['x'],'root successes')!=x or _integer(record['m'],'root trials')!=m:
        raise ValueError('Saved root counts do not match their declared sample')
    if rational(record['gamma'])!=gamma:
        raise ValueError('Saved root allocation differs from the region allocation')
    lo,hi = rational(record['lower']),rational(record['upper'])
    if not 0<=lo<=hi<=1 or record['status'] not in ('verified','structural','fallback'):
        raise ValueError('Malformed saved verified root interval')
    if record['status']=='fallback' and (lo,hi)!=(F(0),F(1)):
        raise ValueError('A saved root fallback must retain [0,1]')
    return lo,hi


def _region(region,N,T,policy,name):
    """Validate linkage, preserving the returned endpoints without rerooting."""
    K,n = len(N),sum(N)
    if (region['K']!=K or region.get('known_masses') is not None
            or region['mode']!='unknown_mass' or rational(region['alpha'])!=ALPHA
            or rational(region['gamma'])!=ALPHA/(2*K)
            or _integer(region['n'],name+' total',1)!=n):
        raise ValueError(name+' support, allocation, or unknown-mass mode differs')
    rN,rT = _counts(region['N'],region['T'],name+' region')
    if (rN,rT)!=(N,T) or _policy(region['root_policy'])!=policy:
        raise ValueError(name+' saved region sample/policy differs')
    roots = region['roots']
    if len(roots['propensity'])!=K or roots['mass'] is None or len(roots['mass'])!=K:
        raise ValueError(name+' must retain both raw root families on the full support')
    vectors = [region[k] for k in ('mass_lower','mass_upper','propensity_lower','propensity_upper')]
    if any(len(v)!=K for v in vectors):
        raise ValueError(name+' endpoint vector support differs')
    gamma = ALPHA/(2*K)
    for i in range(K):
        pl,pu = _root(roots['mass'][i],N[i],n,gamma)
        el,eu = _root(roots['propensity'][i],T[i],N[i],gamma)
        if (pl,pu,el,eu)!=tuple(rational(v[i]) for v in vectors):
            raise ValueError(name+' endpoints do not match their saved root records')
    fallback_count = sum(r['status']=='fallback' for r in roots['propensity']+roots['mass'])
    if region.get('root_fallbacks')!=fallback_count:
        raise ValueError(name+' root fallback count differs')


def _row(value,K):
    cap = rational(value['cap'])
    cells = _integer(value.get('cells',value.get('count')),'selected cell count',1)
    partition = _partition(value['partition'],K)
    bound = rational(value.get('value',value.get('certificate')))
    if cap not in CAPS or cells!=len(partition) or not 1<=bound<=cap:
        raise ValueError('Imported selection lacks a passing exact statistic/certificate')
    if 'count' in value and value['count']!=cells:
        raise ValueError('Selected count aliases differ')
    if 'certificate' in value and value['certificate']!=bound:
        raise ValueError('Selected certificate aliases differ')
    if bool(value['identity'])!=(cells==K):
        raise ValueError('Imported identity flag differs from the selected partition')
    return cap,cells,partition,bound


def _compare_selection(a,b,K):
    if _row(a,K)!=_row(b,K):
        raise ValueError('Preserved count, partition, or exact selected value changed')
    for field in ('status','identity','stop_index','stop_reason','accepted_prefix_indices','tested'):
        if field in a and field in b and not _same(a[field],b[field]):
            raise ValueError('Preserved selector return/stop trace changed: '+field)


def _normalized(row,procedure,K):
    cap,cells,partition,value = _row(row,K)
    stop = (None if procedure=='F' else {key:row[key] for key in
            ('stop_index','stop_reason','accepted_prefix_indices','tested')})
    return {'procedure':procedure,'cap':cap,'cells':cells,'partition':partition,'value':value,
            'value_kind':'ltt_distortion_statistic' if procedure=='A' else 'coupled_certificate',
            'identity':cells==K,'computation_status':'imported_preserved_selection',
            'count_optimal':True,
            'selection_scope':'full_order_combined_family' if procedure=='F' else 'saved_accepted_prefix',
            'source_status':row['status'],'source_stop':stop}


def _check_prefix(selection,candidates,K):
    """Replay only saved exact test returns, without inference or new tests."""
    from experiments.jmlr_statistical_baselines_v1_core import first_failure_select
    pending = iter(selection['tested'])
    consumed = 0

    def saved_test(partition):
        nonlocal consumed
        entry = next(pending)
        index = _integer(entry['candidate_index'],'tested proposal index')
        if index>=K or _partition(candidates[index]['partition'],K)!=partition:
            raise ValueError('Saved test is not linked to the proposal being checked')
        consumed += 1
        return entry

    replayed = first_failure_select(candidates,selection['cap'],saved_test,K,selection['method'])
    _compare_selection(selection,replayed,K)
    if consumed!=len(selection['tested']):
        raise ValueError('Saved first-failure trace contains unqueried tests')


def _one(rows,predicate,name):
    found = [r for r in rows if predicate(r)]
    if len(found)!=1:
        raise ValueError('Expected exactly one '+name)
    return found[0]


def load_dataset(split_path,full_path,expected_policy=None):
    """Return one matched saved dataset, with nine normalized A/B/F rows.

    Both source phases are accepted for disjoint development checks. A final
    panel caller must restrict the returned phase to evaluation (reps10–99).
    Every pilot proposal is retained, including untrusted infinite scores.
    """
    split_source,full = _read(split_path),_read(full_path)
    if (split_source['version']!='jmlr_statistical_baselines_v1'
            or full['version']!='jmlr_verified_pipeline_v1'):
        raise ValueError('Unexpected authoritative source versions')
    metadata = split_source['metadata']
    if not _same(metadata,full['metadata']):
        raise ValueError('Split and full-data dataset metadata differ')
    rep = _integer(metadata['replication'],'replication')
    _integer(metadata['population_index'],'population index')
    _integer(metadata['r_index'],'sample-size index')
    if not isinstance(metadata['id'],str) or not metadata['id']:
        raise ValueError('Dataset identifier must be a nonempty string')
    if rep>=100:
        raise ValueError('Replication is outside the frozen development/evaluation panel')
    phase = 'development' if rep<10 else 'evaluation'
    if split_source['phase']!=phase:
        raise ValueError('Saved split phase differs from its replication range')
    core = full['core']
    N,T = _counts(split_source['N'],split_source['T'],'full sample')
    if _counts(core['N'],core['T'],'pipeline sample')!=(N,T):
        raise ValueError('Authoritative full sample counts differ')
    K = len(N)
    if (_integer(metadata['K'],'K',1)!=K or _integer(metadata['n'],'n',1)!=sum(N)
            or _integer(metadata['r'],'n/K',1)*K!=sum(N)):
        raise ValueError('Metadata sample size/support do not match counts')
    ids = tuple(split_source['immutable_ids'])
    if len(ids)!=K or len(set(ids))!=K or tuple(core['immutable_ids'])!=ids:
        raise ValueError('Immutable state identifiers differ')
    split = _one(split_source['splits'],lambda r:r['fraction']=='1/5','saved frozen-fraction split')
    if tuple(split['immutable_ids'])!=ids:
        raise ValueError('Split immutable state identifiers differ')
    pilot_N,pilot_T = _counts(split['pilot_N'],split['pilot_T'],'pilot')
    heldout_N,heldout_T = _counts(split['heldout_N'],split['heldout_T'],'held-out')
    if len(pilot_N)!=K or len(heldout_N)!=K:
        raise ValueError('Split counts must retain the complete support')
    if (tuple(a+b for a,b in zip(pilot_N,heldout_N))!=N
            or tuple(a+b for a,b in zip(pilot_T,heldout_T))!=T):
        raise ValueError('Pilot and held-out counts do not conserve the full counts')
    if (sum(pilot_N)!=sum(N)//5 or sum(pilot_N)*5!=sum(N)
            or _integer(split['pilot_size'],'pilot size')!=sum(pilot_N)
            or _integer(split['heldout_size'],'held-out size',1)!=sum(heldout_N)):
        raise ValueError('The saved split does not preserve the frozen one-fifth allocation')
    seed = [20261012,901,metadata['population_index'],metadata['r_index'],rep]
    if (split['seed']!=seed or split['generator']!='PCG64(SeedSequence)'
            or split['joint_order']!='immutable state, action 0 then action 1'):
        raise ValueError('Saved split randomization provenance differs')
    proposal = split['core']
    pilot_order = tuple(proposal['pilot_order'])
    full_order = tuple(core['order'])
    if pilot_order!=exact_order(pilot_N,pilot_T,ids) or full_order!=exact_order(N,T,ids):
        raise ValueError('Saved pilot/full order does not match exact smoothed count ordering')
    tree_orders = tuple(tuple(order) for order in core['tree_orders'])
    if len(tree_orders)!=2 or any(sorted(order)!=list(range(K)) for order in tree_orders):
        raise ValueError('The two complete fixed-tree orders differ from the established family')
    candidates = proposal['candidates']
    if len(candidates)!=K or {_integer(r['cells'],'candidate count',1) for r in candidates}!=set(range(1,K+1)):
        raise ValueError('The complete saved pilot list must include every count exactly once')
    tree_nodes = [set(exact.build_tree(order)) for order in tree_orders]
    contiguous = set(exact.interval_cells(pilot_order))
    for i,candidate in enumerate(candidates):
        if _integer(candidate['candidate_index'],'candidate index')!=i:
            raise ValueError('Saved proposal indices/order differ')
        partition = _partition(candidate['partition'],K)
        if len(partition)!=candidate['cells']:
            raise ValueError('Proposal partition/count differ')
        family = candidate['family']
        nodes = contiguous if family=='contiguous' else tree_nodes[int(family[-1])] if family in ('tree-0','tree-1') else None
        if nodes is None or any(c not in nodes for c in partition):
            raise ValueError('Saved proposal is outside its declared constituent family')
        score = candidate['score']
        if score is not None and (isinstance(score,bool) or not isinstance(score,(int,float))
                                  or not math.isfinite(score) or score<1):
            raise ValueError('Pilot score must be an untrusted finite ranking value or saved infinity')
    full_mode = _one(core['modes'],lambda r:r['mode']=='unknown_mass','unknown-mass full-data region')
    full_region = full_mode['region']
    policy = _policy(full_region['root_policy']) if expected_policy is None else _policy(expected_policy)
    _region(full_region,N,T,policy,'Full-data')
    heldout_region = proposal['control_evidence']['region']
    if heldout_region is not None:
        _region(heldout_region,heldout_N,heldout_T,policy,'Held-out')
    elif any(len(record['cell'])>1 for record in proposal['control_evidence']['cells'].values()):
        raise ValueError('A saved merged coupled cell cannot lack its held-out region')
    if (_policy(proposal['ltt_evidence']['root_policy'])!=policy
            or rational(proposal['ltt_evidence']['alpha'])!=ALPHA
            or rational(proposal['ltt_evidence']['gamma'])!=ALPHA):
        raise ValueError('Preserved LTT confidence allocation/policy differs')
    baseline = full['baseline']
    if (baseline['phase']!=phase or baseline['frozen_fraction']!='1/5'
            or not _same(baseline['metadata'],metadata)):
        raise ValueError('Pipeline baseline linkage differs from this saved split')
    adapted_split = _one(baseline['splits'],lambda r:r['fraction']=='1/5','pipeline frozen-fraction split')
    references,B_by_cap = [],{}
    for procedure,method,label in (('A','ltt','split_ltt'),('B','split_coupled','split_coupled')):
        for cap in CAPS:
            row = _one(split['rows'],lambda r:r['procedure']==label and rational(r['cap'])==cap,'split selection')
            selection = _one(proposal['selections'],lambda r:r['method']==method and rational(r['cap'])==cap,'saved selector return')
            imported = _one(adapted_split['rows'],lambda r:r['procedure']==label and rational(r['cap'])==cap,'pipeline imported selection')
            _compare_selection(row,selection,K);_compare_selection(row,imported,K)
            _check_prefix(selection,candidates,K)
            stop = selection['stop_index']
            if stop is not None:
                stop = _integer(stop,'first-failure index')
                if not 0<=stop<K or not selection['tested'] or selection['tested'][-1]['candidate_index']!=stop or selection['tested'][-1]['accepted']:
                    raise ValueError('Saved first-failure index is not linked to its final failed test')
            references.append(_normalized(row,procedure,K))
            if procedure=='B':
                B_by_cap[cap] = selection
    for cap in CAPS:
        row = _one(full_mode['labels'],lambda r:r['method']=='coupled' and r['family']=='combined' and rational(r['cap'])==cap,'full-data combined coupled selection')
        imported = _one(baseline['reference'],lambda r:rational(r['cap'])==cap,'pipeline full-data reference')
        _compare_selection(row,imported,K)
        references.append(_normalized(row,'F',K))
    return {'metadata':metadata,'phase':phase,'N':N,'T':T,'immutable_ids':ids,
            'pilot_N':pilot_N,'pilot_T':pilot_T,'heldout_N':heldout_N,'heldout_T':heldout_T,
            'pilot_order':pilot_order,'candidates':candidates,'tree_orders':tree_orders,
            'full_order':full_order,'heldout_region':heldout_region,'full_region':full_region,
            'heldout_cells':tuple(proposal['control_evidence']['cells'].values()),
            'full_cells':tuple(full_mode['cells']),'references':references,'B_by_cap':B_by_cap,
            'source_paths':{'split':str(Path(split_path).resolve()),'full':str(Path(full_path).resolve())}}
