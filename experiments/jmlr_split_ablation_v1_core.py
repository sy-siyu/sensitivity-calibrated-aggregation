"""Exact assignment-only C/D/E ablations on preserved confidence regions.

Pilot scores determine a saved proposal sequence only. This module never
consults a population law, changes the split, or redefines LTT/first-failure
selection. A per-region CellEngine reuses completed saved coupled certificates
and uses the preserved optimized arithmetic for missing cells. Partial above-2
rejections remain rejection proofs, never completed coupled upper bounds.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction as F
import math
import numbers
import time
from types import MappingProxyType

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, rational
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import counts, select_families


CAPS=(F(5,4),F(3,2),F(2))
MAX_CAP=F(2)


def _integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value<minimum:
        raise ValueError(name+' must be an integer without coercion')
    return int(value)


def _caps(caps):
    result=tuple(rational(c) for c in caps)
    if not result or len(set(result))!=len(result) or any(not 1<=c<=MAX_CAP for c in result):
        raise ValueError('Caps must be distinct exact rationals in [1,2]')
    return result


def _partition(partition,K):
    # exact.validate_partition validates raw index types before canonicalizing.
    return exact.validate_partition(partition,K)


def _geometry(order,tree_orders):
    order=tuple(order);K=len(order)
    if not K or any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for i in order) or sorted(order)!=list(range(K)):
        raise ValueError('Pilot order must permute every immutable state index')
    order=tuple(map(int,order))
    orders=tuple(tuple(o) for o in tree_orders)
    if len(orders)!=2 or any(len(o)!=K or any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for i in o)
                              or sorted(o)!=list(range(K)) for o in orders):
        raise ValueError('Exactly two fixed complete tree orders are required')
    orders=tuple(tuple(map(int,o)) for o in orders)
    return order,orders,tuple(exact.build_tree(o) for o in orders)


def family_membership(partition,order,tree_orders):
    """Return constituents containing the *whole* partition, without mixing."""
    order,_,trees=_geometry(order,tree_orders);partition=_partition(partition,len(order))
    positions={state:i for i,state in enumerate(order)}
    families=[]
    if all(max(positions[i] for i in c)-min(positions[i] for i in c)+1==len(c) for c in partition):
        families.append('contiguous')
    selected=set(partition)
    for index,tree in enumerate(trees):
        def visit(cell):
            if cell in selected:
                return True
            children=tree[cell]
            return children is not None and all(visit(child) for child in children)
        if visit(max(tree,key=len)):
            families.append('tree-'+str(index))
    return tuple(families)


def _candidates(candidates,K):
    result=[]
    for index,candidate in enumerate(candidates):
        if not isinstance(candidate,Mapping):
            raise ValueError('Saved proposals must be mappings')
        partition=_partition(candidate['partition'],K)
        count=_integer(candidate['cells'],'Proposal cell count',1)
        if count!=len(partition):
            raise ValueError('Saved proposal count disagrees with its partition')
        if 'candidate_index' in candidate and _integer(candidate['candidate_index'],'Candidate index')!=index:
            raise ValueError('Saved proposal indices or ordering changed')
        result.append(dict(candidate,partition=partition,cells=count,candidate_index=index))
    if len(result)!=K or sorted(c['cells'] for c in result)!=list(range(1,K+1)):
        raise ValueError('The complete saved proposal list must contain one representative per cell count')
    return tuple(result)


def validate_proposals(candidates,order,tree_orders):
    """Validate every saved proposal, including infinite pilot-score entries."""
    order,orders,_=_geometry(order,tree_orders)
    checked=[]
    for candidate in _candidates(candidates,len(order)):
        families=family_membership(candidate['partition'],order,orders)
        if not families:
            raise ValueError('A saved proposal mixes constituent families or belongs to none')
        declared=candidate.get('family')
        if declared is not None and declared not in families:
            raise ValueError('Saved proposal family label disagrees with its whole partition')
        checked.append({'candidate_index':candidate['candidate_index'],'cells':candidate['cells'],
                        'families':families,'declared_family':declared})
    return tuple(checked)


def _record(cell,record):
    record=dict(record)
    if 'cell' in record and tuple(record['cell'])!=cell:
        raise ValueError('Saved certificate belongs to another cell')
    status=record.get('status')
    if status not in ('full','boundary','rejected','failure'):
        raise ValueError('Unrecognized saved coupled-cell status')
    for key in ('coupled','independent','rejection_bound','lower_bound'):
        if record.get(key) is not None:
            record[key]=rational(record[key])
            if record[key]<1:
                raise ValueError('Distortion values and rigorous lower bounds must be at least one')
    if status=='full' and record.get('coupled') is None:
        raise ValueError('A completed finite cell requires its exact coupled certificate')
    if status!='full' and record.get('coupled') is not None:
        raise ValueError('Partial, infinite, or unresolved cells cannot carry a completed coupled value')
    if status=='rejected' and record.get('rejection_bound') is None:
        raise ValueError('A saved rejection requires a rigorous lower bound')
    if len(cell)==1 and (status!='full' or record.get('coupled')!=1):
        raise ValueError('Singleton certificates must equal one')
    return dict(record,cell=cell)


def _cell_lower(record):
    if record['status']=='boundary':
        return None
    if record['status']=='full':
        return record['coupled']
    return record.get('rejection_bound',record.get('lower_bound')) or F(1)


class CellEngine:
    """One immutable public confidence context, with its own cell cache.

    Records supplied by the input adapter must be associated with this saved
    region; their proof evidence is not independently reconstructed here.
    ``records_context`` optionally enforces an explicit context-key match.
    No cache is shared between engines, samples, or confidence allocations.
    """
    def __init__(self,region,records=(),*,records_context=None):
        if not isinstance(region,Mapping):
            raise ValueError('A saved verified confidence region is required')
        N,T=counts(region['N'],region['T']);K=len(N)
        if _integer(region['K'],'Support size',1)!=K or region.get('known_masses') is not None or region.get('mode')!='unknown_mass':
            raise ValueError('This ablation uses the saved unknown-mass region on the complete support')
        if _integer(region['n'],'Calibration total',1)!=sum(N):
            raise ValueError('Saved region total does not match its integer counts')
        alpha,gamma=rational(region['alpha']),rational(region['gamma'])
        if not 0<alpha<1 or gamma!=alpha/(2*K):
            raise ValueError('Unknown-mass confidence allocation does not match alpha/(2K)')
        policy_inputs=dict(region['root_policy'])
        if set(policy_inputs)!=set(RootPolicy().__dict__):
            raise ValueError('Saved region must declare the complete numerical root policy')
        for name,value in policy_inputs.items():
            if name=='endpoint_seconds':
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError('Root endpoint time budget must be a finite numeric value')
            else:
                _integer(value,'Root policy '+name)
        policy=RootPolicy(**policy_inputs)
        rebuilt=exact.region_from_bounds(region['mass_lower'],region['mass_upper'],
                                         region['propensity_lower'],region['propensity_upper'])
        if bool(region.get('valid'))!=rebuilt['valid']:
            raise ValueError('Saved confidence-region validity flag disagrees with its exact endpoints')
        rebuilt.update(N=N,T=T,n=sum(N),alpha=alpha,gamma=gamma,root_policy=dict(policy.__dict__))
        self._region=rebuilt;self.K=K;self.N=N;self.T=T;self.alpha=alpha;self.gamma=gamma;self.policy=policy
        self.context_key=(K,N,T,alpha,gamma,tuple(sorted(policy.__dict__.items())),
                          tuple(rebuilt['mass_lower']),tuple(rebuilt['mass_upper']),
                          tuple(rebuilt['propensity_lower']),tuple(rebuilt['propensity_upper']))
        if records_context is not None and records_context!=self.context_key:
            raise ValueError('Saved-cell cache context differs from the confidence region')
        self._saved={};self._cells={}
        values=records.values() if isinstance(records,Mapping) else records
        for source in values:
            cell=exact._cell(source['cell'],K)
            if cell in self._saved:
                raise ValueError('Duplicate saved cell certificate')
            normalized=_record(cell,source)
            if not rebuilt['valid'] and len(cell)>1 and normalized['status']!='failure':
                raise ValueError('An invalid raw region cannot provide a completed envelope or rigorous rejection')
            self._saved[cell]=normalized
        self._counters=Counter(saved_cells=len(self._saved),cache_hits=0,cache_misses=0,
            backend_calls=0,saved_records_reused=0,structural_singletons=0,
            saved_partial_or_failure_recomputed=0)
        self._backend_seconds=0.
        self.source_root_fallbacks=_integer(region.get('root_fallbacks',0),'Saved root fallback count')

    @property
    def region(self):
        return MappingProxyType(self._region)

    @property
    def cells(self):
        return MappingProxyType(self._cells)

    def cell(self,cell):
        cell=exact._cell(cell,self.K)
        if cell in self._cells:
            self._counters['cache_hits']+=1
            return self._cells[cell]
        self._counters['cache_misses']+=1
        old=self._saved.get(cell)
        if len(cell)==1:
            result={'cell':cell,'status':'full','coupled':F(1),'independent':F(1),
                    'rejection_bound':None,'anchors':[],'reason':'singleton-identity'}
            self._counters['structural_singletons']+=1
        elif old is not None and (old['status'] in ('full','boundary') or
                                  old['status']=='rejected' and old['rejection_bound']>MAX_CAP):
            result=old;self._counters['saved_records_reused']+=1
        else:
            tick=time.perf_counter()
            result=cell_certificate(self._region,cell,max_cap=MAX_CAP)
            self._backend_seconds+=time.perf_counter()-tick;self._counters['backend_calls']+=1
            if old is not None:
                self._counters['saved_partial_or_failure_recomputed']+=1
        result=_record(cell,result)
        self._cells[cell]=result
        return result

    def partition(self,partition,cap=None):
        partition=_partition(partition,self.K)
        records=tuple(self.cell(cell) for cell in partition)
        boundary=any(r['status']=='boundary' for r in records)
        lowers=tuple(_cell_lower(r) for r in records)
        lower=None if boundary else max(lowers,default=F(1))
        complete=all(r['status']=='full' for r in records)
        certificate=max((r['coupled'] for r in records),default=F(1)) if complete else None
        status=('infinite' if boundary else 'finite' if complete else
                'rigorous_above_cap' if lower>MAX_CAP else 'unresolved')
        result={'partition':partition,'cells':len(partition),'status':status,
                'certificate':certificate,'value':certificate,'lower_bound':lower,
                'completed_envelope':bool(complete or boundary),
                'cell_status_counts':dict(Counter(r['status'] for r in records)),
                'reason':next((r.get('reason') for r in records if r['status']!='full'),None)}
        if cap is not None:
            cap=_caps((cap,))[0]
            result.update(_decision(result,cap))
        return result

    def evidence(self):
        return {'context_key':self.context_key,'N':self.N,'T':self.T,'alpha':self.alpha,
                'gamma':self.gamma,'root_policy':dict(self.policy.__dict__),
                'cells':tuple(self._cells[c] for c in sorted(self._cells)),
                'cache':dict(self._counters),'backend_seconds':self._backend_seconds,
                'source_root_fallbacks':self.source_root_fallbacks,
                'cell_status_counts':dict(Counter(r['status'] for r in self._cells.values())),
                'scope':'Per-engine context-bound cell cache; saved evidence reused, missing cells computed with max_cap2'}


def _decision(evaluation,cap):
    if evaluation['status']=='infinite':
        status,accepted='infinite',False
    elif evaluation['certificate'] is not None:
        accepted=evaluation['certificate']<=cap
        status='finite' if accepted else 'rigorous_above_cap'
    elif evaluation['lower_bound'] is not None and evaluation['lower_bound']>cap:
        status,accepted='rigorous_above_cap',False
    else:
        status,accepted='unresolved',False
    return {'cap':cap,'status':status,'accepted':accepted,
            'certificate':evaluation['certificate'],'lower_bound':evaluation['lower_bound'],
            'completed_envelope':evaluation['completed_envelope']}


def _safe_selection(choice,cap,K,scope,*,count_optimal,tie_complete,blocking_indices=(),**extra):
    certificate=choice['certificate']
    if certificate is None or certificate>cap or len(choice['partition'])!=choice['cells']:
        raise ArithmeticError('Returned selection lacks a completed passing certificate')
    complete=bool(count_optimal and tie_complete)
    identity=choice['cells']==K
    return {**choice,'cap':cap,'value':certificate,'value_status':'finite',
            'status':'finite' if complete else 'unresolved',
            'computation_status':'finite' if complete else 'unresolved',
            'count_optimal':bool(count_optimal),'deterministic_tie_complete':bool(tie_complete),
            'identity':identity,'identity_fallback':bool(identity and not complete),
            'safe_fallback':not complete,'selection_scope':scope,
            'blocking_indices':tuple(blocking_indices),
            'secondary_objective_scope':'saved proposal order' if scope=='all_saved_pilot_proposals'
                else 'preserved local dynamic-program ties; no global secondary-certificate optimum claimed',
            **extra}


def all_proposals(candidates,engine,caps=CAPS):
    """Evaluate the full saved list, retaining every pilot-score infinity."""
    caps=_caps(caps);candidates=_candidates(candidates,engine.K);evaluations=[]
    for candidate in candidates:
        evaluation=engine.partition(candidate['partition'])
        evaluations.append(dict(evaluation,candidate_index=candidate['candidate_index'],
                                pilot_score=candidate.get('score'),pilot_family=candidate.get('family'),
                                decisions={cap:_decision(evaluation,cap) for cap in caps}))
    by_cap={}
    for cap in caps:
        passing=[row for row in evaluations if row['decisions'][cap]['accepted']]
        if not passing:
            raise ArithmeticError('Complete proposal list lost its structural identity')
        best=min(passing,key=lambda row:(row['cells'],row['candidate_index']))
        unknown=[row for row in evaluations if row['decisions'][cap]['status']=='unresolved']
        lower_count=[row['candidate_index'] for row in unknown if row['cells']<best['cells']]
        earlier_tie=[row['candidate_index'] for row in unknown if row['cells']==best['cells']
                     and row['candidate_index']<best['candidate_index']]
        choice={key:best[key] for key in ('partition','cells','certificate','candidate_index')}
        by_cap[cap]=_safe_selection(choice,cap,engine.K,'all_saved_pilot_proposals',
            count_optimal=not lower_count,tie_complete=not earlier_tie,
            blocking_indices=tuple(lower_count+earlier_tie),
            unresolved_proposal_indices=tuple(row['candidate_index'] for row in unknown),
            proposal_status_counts=dict(Counter(row['decisions'][cap]['status'] for row in evaluations)))
    return {'evaluations':tuple(evaluations),'by_cap':by_cap,'candidate_count':len(candidates),
            'cache':engine.evidence()['cache']}


def _relevant_unknown(order,trees,records,cap,B):
    """Unknown cells in a lower-feasible at-most-B member of one constituent."""
    K=len(order);unreachable=K+1
    allowed={c:_cell_lower(r) is not None and _cell_lower(r)<=cap for c,r in records.items()}
    unknown={c for c,r in records.items() if r['status'] not in ('full','boundary')}
    intervals={(i,j):tuple(sorted(order[i:j])) for j in range(1,K+1) for i in range(j)}
    prefix,suffix=[unreachable]*(K+1),[unreachable]*(K+1);prefix[0]=suffix[K]=0
    for j in range(1,K+1):
        prefix[j]=min((prefix[i]+1 for i in range(j) if allowed[intervals[i,j]]),default=unreachable)
    for i in range(K-1,-1,-1):
        suffix[i]=min((suffix[j]+1 for j in range(i+1,K+1) if allowed[intervals[i,j]]),default=unreachable)
    relevant={c for (i,j),c in intervals.items() if c in unknown and allowed[c] and prefix[i]+1+suffix[j]<=B}
    for tree in trees:
        minimum={}
        def down(cell):
            children=tree[cell]
            split=unreachable if children is None else sum(down(child) for child in children)
            minimum[cell]=min(1 if allowed[cell] else unreachable,split)
            return minimum[cell]
        root=max(tree,key=len);down(root)
        def up(cell,outside):
            if cell in unknown and allowed[cell] and outside+1<=B:
                relevant.add(cell)
            children=tree[cell]
            if children is not None:
                left,right=children;up(left,outside+minimum[right]);up(right,outside+minimum[left])
        up(root,0)
    return tuple(sorted(relevant))


def complete_pilot_search(order,tree_orders,engine,caps=CAPS):
    """Existing combined threshold search, with the complete saved pilot order."""
    order,_,trees=_geometry(order,tree_orders);caps=_caps(caps)
    if len(order)!=engine.K:
        raise ValueError('Pilot geometry and confidence support differ')
    cells=tuple(sorted(set(exact.interval_cells(order)).union(*(set(tree) for tree in trees))))
    records={cell:engine.cell(cell) for cell in cells}
    upper={c:r['coupled'] if r['status']=='full' else None for c,r in records.items()}
    lower={c:_cell_lower(r) for c,r in records.items()}
    by_cap={}
    for cap in caps:
        high=select_families(order,trees,upper,cap)['combined']
        low=select_families(order,trees,lower,cap)['combined']
        candidate={'partition':high['partition'],'cells':high['cells'],'certificate':high['certificate'],
                   'constituent':high['constituent']}
        unresolved=_relevant_unknown(order,trees,records,cap,high['cells'])
        count_complete=high['cells']==low['cells']
        tie_complete=not unresolved
        result=_safe_selection(candidate,cap,engine.K,'complete_pilot_order_combined_family',
            count_optimal=count_complete,tie_complete=tie_complete,
            lower_count=low['cells'],upper_count=high['cells'],
            potentially_relevant_unresolved_cells=unresolved)
        # Retain all deterministic auxiliary fields from the actual high search.
        result['search_choice']=high
        if not family_membership(result['partition'],order,tree_orders):
            raise ArithmeticError('Combined search returned a mixed-family partition')
        by_cap[cap]=result
    return {'by_cap':by_cap,'family_cells':cells,
            'cell_status_counts':dict(Counter(r['status'] for r in records.values())),
            'cache':engine.evidence()['cache']}


def _by_cap(mapping,cap):
    if cap in mapping:
        return mapping[cap]
    if str(cap) in mapping:
        return mapping[str(cap)]
    raise ValueError('Evaluation lacks the requested prespecified cap')


def stopping_diagnostic(B_row,candidates,evaluations,cap):
    """Describe continuing *after* B's saved first failure, without changing B."""
    cap=_caps((cap,))[0]
    K=sum(len(c) for c in B_row['partition']);candidates=_candidates(candidates,K)
    Bcount=_integer(B_row['cells'],'Saved B selected count',1)
    if Bcount!=len(_partition(B_row['partition'],K)):
        raise ValueError('Saved B count differs from its selected partition')
    if 'cap' in B_row and rational(B_row['cap'])!=cap:
        raise ValueError('Saved B row has a different cap')
    root=evaluations if isinstance(evaluations,Mapping) else None
    rows=tuple(evaluations['evaluations'] if root is not None else evaluations)
    if len(rows)!=len(candidates) or any(row['candidate_index']!=i or row['partition']!=candidates[i]['partition'] for i,row in enumerate(rows)):
        raise ValueError('All-proposal evaluations do not match the saved sequence')
    stop=B_row.get('stop_index')
    if stop is not None:
        stop=_integer(stop,'Saved first-failure index')
        if stop>=len(candidates):
            raise ValueError('Saved first failure is outside the proposal sequence')
    later=tuple(range(stop+1,len(rows))) if stop is not None else ()
    passing=tuple(i for i in later if _by_cap(rows[i]['decisions'],cap)['accepted'])
    coarser=tuple(i for i in passing if rows[i]['cells']<Bcount)
    unresolved=tuple(i for i in later if _by_cap(rows[i]['decisions'],cap)['status']=='unresolved')
    unresolved_coarser=tuple(i for i in unresolved if rows[i]['cells']<Bcount)
    best_later=min((rows[i]['cells'] for i in passing),default=None)
    improvement=max(0,Bcount-best_later) if best_later is not None else 0
    failed_now_passes=(bool(_by_cap(rows[stop]['decisions'],cap)['accepted']) if stop is not None else False)
    return {'saved_stop_index':stop,'saved_stop_reason':B_row.get('stop_reason'),
            'stop_index':stop,'stop_reason':B_row.get('stop_reason'),
            'B_selected_count':Bcount,'saved_failed_proposal_now_passing':failed_now_passes,
            'later_passing':bool(passing),'later_passing_count':len(passing),
            'later_passing_indices':passing,'later_passing_coarser':bool(coarser),
            'later_passing_coarser_count':len(coarser),'later_passing_coarser_indices':coarser,
            'best_later_count':best_later,'count_improvement':improvement,
            'compression_improvement_pp':F(100*improvement,K),
            'later_unresolved_indices':unresolved,'later_unresolved_coarser_indices':unresolved_coarser,
            'later_search_complete':not unresolved,
            'all_proposals_count_optimal':(_by_cap(root['by_cap'],cap)['count_optimal'] if root is not None else None)}
