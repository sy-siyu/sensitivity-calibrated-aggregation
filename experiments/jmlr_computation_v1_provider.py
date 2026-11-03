"""Fresh verified assignment providers with bounded, explicit instrumentation.

The input boundary is finite-state calibration counts and immutable geometry.
No saved numerical evidence, outcomes, population law or precision target is
accepted. Threshold and capacity modes build separate confidence/cell tables.
Projection/export is owned by the subsequent inference stage, not performed
here. Historical numerical and search modules are imported without edits.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction as F
import math
import numbers
import time

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, rational
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate
from experiments.jmlr_verified_pipeline_v1_core import counts,exact_order,select_families
from experiments.jmlr_release_capacity_v1_search import candidate_cells,capacity_search
from experiments.jmlr_split_ablation_v1_core import _cell_lower,_relevant_unknown,_safe_selection


ALPHA=F(1,40)
CAP=F(3,2)
DEFAULT_POLICY=RootPolicy(max_trials=2_000_000)
CELL_STATUSES=('full','rejected','boundary','failure')


def capacity_grid(K):
    return tuple(sorted({(K+d-1)//d for d in (16,8,4,2,1)}))


def _id(value):
    return tuple(_id(x) for x in value) if isinstance(value,(tuple,list)) else value


def _policy(value):
    value=DEFAULT_POLICY if value is None else value
    if isinstance(value,RootPolicy):
        value=value.__dict__
    if not isinstance(value,Mapping) or set(value)!=set(DEFAULT_POLICY.__dict__):
        raise ValueError('A complete numerical root policy is required')
    for name,number in value.items():
        if name=='endpoint_seconds':
            if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number):
                raise ValueError('Endpoint time limit must be finite')
        elif isinstance(number,bool) or not isinstance(number,numbers.Integral):
            raise ValueError('Root precision and resource policy fields must be integers')
    return RootPolicy(**dict(value))


def _progress(callback,stage,event='start',**details):
    if callback is not None:
        callback({'stage':stage,'event':event,**details})


def _equal_mass_partition(N,order,B):
    """The release provider's unchanged whole-state incumbent construction."""
    n=sum(N);K=len(order);cuts=[0];cumulative=[0]
    for i in order:
        cumulative.append(cumulative[-1]+N[i])
    for j in range(1,B):
        cuts.append(min(range(cuts[-1]+1,K-(B-j)+1),
                        key=lambda x:(abs(B*cumulative[x]-j*n),x)))
    cuts.append(K)
    return exact.canonical_partition(tuple(order[a:b]) for a,b in zip(cuts,cuts[1:]))


def _search_work(K,trees,mode,capacities,capacity_relevance_scans=0):
    """Exact loop-work equivalents, not total arithmetic/bit-operation counts."""
    intervals=K*(K+1)//2
    nodes=sum(len(tree) for tree in trees)
    if mode=='threshold':
        return {'kind':'exact implemented-loop equivalents',
            'threshold_family_passes':2,
            'contiguous_transition_checks':2*intervals,
            'tree_node_visits':2*nodes,
            'relevance_prefix_suffix_checks':2*intervals,
            'relevance_tree_node_visits':2*nodes}
    pair_transitions=0;tree_checks=0
    for tree in trees:
        for cell,children in tree.items():
            if children is not None:
                left,right=children
                pair_transitions+=len(left)*len(right)
                tree_checks+=(len(cell)-1)*len(left)
    return {'kind':'exact implemented-loop equivalents',
        'contiguous_inner_loop_checks':K*(K+1)*(K+2)//6,
        'contiguous_combine_transitions':K+(K-1)*K*(K+1)//6,
        'tree_convolution_membership_checks':tree_checks,
        'tree_combine_transitions':pair_transitions,
        'at_most_count_candidates':sum(capacities),
        'exact_count_states_contiguous':intervals+1,
        'exact_count_states_tree':sum(sum(len(c) for c in tree) for tree in trees),
        'capacity_tie_relevance_scans':capacity_relevance_scans}


def _record_work(region,records):
    statuses=Counter({status:0 for status in CELL_STATUSES})
    sizes=Counter();anchors=mixtures=closure=0;max_num=max_den=0
    reasons=Counter()
    def bits(value):
        nonlocal max_num,max_den
        if isinstance(value,F):
            max_num=max(max_num,value.numerator.bit_length())
            max_den=max(max_den,value.denominator.bit_length())
    for name in ('mass_lower','mass_upper','propensity_lower','propensity_upper'):
        for value in region[name]:bits(value)
    for cell,record in records.items():
        status=record.get('status')
        if status not in CELL_STATUSES:
            raise ValueError('Unrecognized fresh certificate status')
        statuses[status]+=1;sizes[len(cell)]+=1
        aa=record.get('anchors',())
        anchors+=len(aa);mixtures+=record.get('mixture_calls',0);closure+=record.get('closure_calls',0)
        if record.get('reason'):reasons[str(record['reason'])]+=1
        for name in ('coupled','independent','rejection_bound','independent_q_lower','independent_q_upper'):
            bits(record.get(name))
        for anchor in aa:
            for value in anchor.values():bits(value)
    cache=region.get('_cell_cache',{})
    return {'cell_status_counts':dict(statuses),'cell_size_histogram':dict(sorted(sizes.items())),
        'unique_candidate_cells':len(records),'candidate_memberships':sum(map(len,records)),
        'candidate_size_squares':sum(len(c)**2 for c in records),'largest_candidate_cell':max(map(len,records)),
        'anchors_evaluated':anchors,'mixture_calls_completed':mixtures,'closure_mixture_calls':closure,
        'full_finite_cells':statuses['full'],'rigorous_rejected_cells':statuses['rejected'],
        'mathematical_infinite_cells':statuses['boundary'],'unresolved_cells':statuses['failure'],
        'cell_reason_counts':dict(reasons),'cell_cache_entries':len(cache),
        'cell_cache_outside_memberships':sum(len(value[3]) for value in cache.values()),
        'retained_anchor_records':anchors,'retained_fraction_numerator_bits_max':max_num,
        'retained_fraction_denominator_bits_max':max_den,
        'mixture_counter_scope':'Completed mixture calls reported by the unchanged backend; an interrupted/raising attempt need not be included',
        'fraction_counter_scope':'Retained endpoints/certificate/anchor fractions only; not maximum intermediate integer operands'}


def _threshold(order,trees,records):
    upper={c:r['coupled'] if r['status']=='full' else None for c,r in records.items()}
    lower={c:_cell_lower(r) for c,r in records.items()}
    high=select_families(order,trees,upper,CAP)['combined']
    low=select_families(order,trees,lower,CAP)['combined']
    unresolved=_relevant_unknown(order,trees,records,CAP,high['cells'])
    choice={key:high[key] for key in ('partition','cells','certificate','constituent')}
    return _safe_selection(choice,CAP,len(order),'complete_calibration_order_combined_family',
        count_optimal=high['cells']==low['cells'],tie_complete=not unresolved,
        lower_count=low['cells'],upper_count=high['cells'],
        potentially_relevant_unresolved_cells=unresolved,search_choice=high)


def _capacity_choices(order,trees,records,search):
    """Keep objective completion separate from conservative tie completion."""
    output={}
    for B,source in search['by_capacity'].items():
        choice=dict(source);objective_completed=bool(source['completed'])
        unresolved=()
        if source['status']=='finite':
            unresolved=_relevant_unknown(order,trees,records,source['certificate'],B)
        tie_complete=objective_completed and not unresolved
        complete=objective_completed and tie_complete
        partition=choice['partition']
        selected=() if partition is None else tuple(records[cell] for cell in partition)
        if source['status']=='finite' and any(r['status']!='full' for r in selected):
            raise ArithmeticError('Finite capacity result lacks completed selected cells')
        choice.update(objective_completed=objective_completed,deterministic_tie_complete=tie_complete,
            optimization_completed=complete,completed=complete,
            computation_status=source['status'] if complete else 'unresolved',
            count_optimal=complete,
            value_status='finite' if source['status']=='finite' else source['status'],
            selected_records=selected,potentially_relevant_unresolved_cells=unresolved,
            identity=bool(partition is not None and len(partition)==len(order)),
            identity_fallback=False,safe_fallback=False)
        output[B]=choice
    return output


def provider(N,T,ids,tree_orders,mode='threshold',policy=None,progress_callback=None):
    """Build a fresh threshold or capacity provider from sufficient counts.

    Each call starts with empty root, arithmetic-cell and projection caches.
    Normal reuse inside the call is counted. Capacity mode constructs its own
    full baseline certificates and finite incumbent; it never receives a
    threshold-pruned table. The returned verifier is assignment-only: callers
    must use a distinct fresh analyst verifier for aggregate timing.
    """
    started=time.perf_counter();times={}
    _progress(progress_callback,'validation')
    tick=time.perf_counter()
    from experiments.jmlr_computation_v1_inference import InstrumentedRootVerifier,root_work
    N,T=counts(N,T);K=len(N);policy=_policy(policy)
    if mode not in ('threshold','capacity'):
        raise ValueError('Provider mode must be threshold or capacity')
    ids=tuple(_id(i) for i in ids)
    try:
        if len(ids)!=K or len(set(ids))!=K:
            raise ValueError('Immutable IDs must be distinct on the whole support')
        sorted(ids)
    except TypeError as error:
        raise ValueError('Immutable IDs must be distinct comparable values') from error
    tree_orders=tuple(tuple(order) for order in tree_orders)
    if len(tree_orders)!=2 or any(len(o)!=K or any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for i in o)
                                 or sorted(o)!=list(range(K)) for o in tree_orders):
        raise ValueError('Exactly two complete fixed tree orders are required')
    tree_orders=tuple(tuple(map(int,o)) for o in tree_orders)
    verifier=InstrumentedRootVerifier(policy)
    if verifier.cache:
        raise ArithmeticError('Fresh provider root cache is not empty')
    times['provider_validation_seconds']=time.perf_counter()-tick
    _progress(progress_callback,'validation','complete')

    _progress(progress_callback,'confidence_region')
    tick=time.perf_counter();region=exact.confidence_region(N,T,ALPHA,verifier=verifier)
    times['confidence_region_seconds']=time.perf_counter()-tick
    initial_cell_cache_entries=len(region['_cell_cache'])
    if initial_cell_cache_entries or region.get('_release_projection_cache'):
        raise ArithmeticError('Fresh provider arithmetic/projection cache is not empty')
    _progress(progress_callback,'confidence_region','complete',root_fallbacks=region['root_fallbacks'])

    _progress(progress_callback,'family')
    tick=time.perf_counter();order=exact_order(N,T,ids);trees=tuple(exact.build_tree(o) for o in tree_orders)
    family_cells=candidate_cells(order,tree_orders)
    capacities=capacity_grid(K) if mode=='capacity' else ()
    baselines={B:_equal_mass_partition(N,order,B) for B in capacities}
    times['family_seconds']=time.perf_counter()-tick
    _progress(progress_callback,'family','complete',candidate_cells=len(family_cells))

    _progress(progress_callback,'cell_certification')
    tick=time.perf_counter();records={};baseline_calls=0;incumbent=None
    if mode=='capacity':
        for cell in sorted({cell for pi in baselines.values() for cell in pi}):
            records[cell]=cell_certificate(region,cell);baseline_calls+=1
        smallest=baselines[min(capacities)]
        if all(records[cell]['status']=='full' for cell in smallest):
            incumbent=max(records[cell]['coupled'] for cell in smallest)
    step=max(1,len(family_cells)//100)
    for index,cell in enumerate(family_cells):
        if cell not in records:
            records[cell]=cell_certificate(region,cell,max_cap=CAP if mode=='threshold' else incumbent)
        if (index+1)%step==0 or index+1==len(family_cells):
            _progress(progress_callback,'cell_certification','progress',cells_completed=index+1,total_cells=len(family_cells))
    times['cell_certification_seconds']=time.perf_counter()-tick
    _progress(progress_callback,'cell_certification','complete')

    _progress(progress_callback,'search')
    tick=time.perf_counter();choice=None;by_capacity={};search=None
    if mode=='threshold':
        choice=_threshold(order,trees,records)
        choice['optimization_completed']=bool(choice['count_optimal'] and choice['deterministic_tie_complete'])
        choice['completed']=choice['optimization_completed']
    else:
        search=capacity_search(order,tree_orders,records,capacities)
        by_capacity=_capacity_choices(order,trees,records,search)
        finite=[by_capacity[B]['certificate'] for B in capacities if by_capacity[B]['objective_completed'] and by_capacity[B]['value_status']=='finite']
        if finite!=sorted(finite,reverse=True):
            raise ArithmeticError('Completed at-most-capacity objectives increase with capacity')
        if by_capacity[K]['certificate']!=1:
            raise ArithmeticError('Capacity identity certificate must equal one')
    times['search_seconds']=time.perf_counter()-tick
    _progress(progress_callback,'search','complete')

    tick=time.perf_counter();work=_record_work(region,records)
    work.update(baseline_cell_certification_calls=baseline_calls,cell_certification_calls=len(records),
        initial_root_cache_entries=0,initial_cell_cache_entries=initial_cell_cache_entries,
        initial_projection_cache_entries=0,
        search_work=_search_work(K,trees,mode,capacities,
            sum(choice['status']=='finite' for choice in by_capacity.values())),
        confidence_root_fallbacks=region['root_fallbacks'],
        source_saved_evidence_reused=False,capacity_incumbent=incumbent)
    roots=root_work(verifier)
    times['instrumentation_seconds']=time.perf_counter()-tick
    total=time.perf_counter()-started
    measurements={'stage_seconds':times,'provider_total_seconds':total,
        'unallocated_seconds':total-sum(times.values()),'work':work,'root_work':roots,
        'projection_export_scope':'Not performed by provider; measured fresh by infer_cohort',
        'timing_scope':'Fresh execution from finite calibration counts; normal within-call caches only'}
    partition=None if choice is None else choice['partition']
    selected=() if partition is None else tuple(records[cell] for cell in partition)
    _progress(progress_callback,'complete','complete',provider_seconds=total)
    return {'mode':mode,'K':K,'N':N,'T':T,'ids':ids,'tree_orders':tree_orders,'trees':trees,
        'order':order,'region':region,'verifier':verifier,'records':records,'family_cells':family_cells,
        'choice':choice,'partition':partition,'selected_records':selected,'by_capacity':by_capacity,
        'capacities':capacities,'baselines':baselines,'pruning_incumbent':incumbent,'search':search,
        'measurements':measurements,'policy':dict(policy.__dict__)}
