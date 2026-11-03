"""Assignment-only minimum-count selections on saved release-study regions.

This module accepts no population law, outcome cohort or precision target and
performs no file I/O. Saved endpoints are reused, while private exact-arithmetic
caches are rebuilt. Threshold feasibility and deterministic local/family ties
are those of the established verified pipeline. In particular this is not
certificate minimization at a prescribed capacity.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction as F
import numbers
import time

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, rational
from experiments.jmlr_verified_pipeline_v1_core import counts, exact_order
from experiments.jmlr_split_ablation_v1_core import (
    CellEngine, complete_pilot_search, family_membership,
)


CAPS=(F(1),F(11,10),F(5,4),F(3,2),F(2))
ALPHA=F(1,40)
N_PER_K=4000


def _integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value<minimum:
        raise ValueError(name+' must be an integer at least '+str(minimum))
    return int(value)


def _immutable_id(value):
    return tuple(_immutable_id(x) for x in value) if isinstance(value,(list,tuple)) else value


def _caps(caps):
    caps=tuple(rational(cap) for cap in caps)
    if len(caps)!=len(CAPS) or set(caps)!=set(CAPS):
        raise ValueError('This study requires the complete frozen exact cap grid')
    return CAPS


def _saved_records(cells):
    """Copy record containers without modifying the historical evidence."""
    if isinstance(cells,Mapping):
        result=[]
        for cell,record in cells.items():
            if not isinstance(record,Mapping):
                raise ValueError('Saved cell records must be mappings')
            cell=tuple(cell)
            if 'cell' in record and tuple(record['cell'])!=cell:
                raise ValueError('Cell-record mapping key differs from its saved cell')
            result.append(dict(record,cell=cell))
        return tuple(result)
    return tuple(dict(record) for record in cells)


def select_thresholds(*,N,T,region,order,cells,tree_orders,ids,caps=CAPS,
                      expected_root_policy=None,records_context=None):
    """Return minimum-count certified selections using calibration data alone.

    Every selected cell has a completed finite coupled bound. A safe finite
    candidate can still have unresolved count optimality or deterministic ties;
    those statuses are retained separately. Saved partial rejections with lower
    bound <=2 are recomputed by CellEngine, whereas lower bounds >2 suffice to
    exclude a cell from every committed threshold without completing its upper
    envelope. Mathematical infinity is represented only by a boundary record.

    ``cells`` must be saved evidence associated with ``region``. The optional
    ``records_context`` checks an explicit context key; this routine does not
    independently reconstruct the provenance of historical cell certificates.
    """
    started=time.perf_counter()
    N,T=counts(N,T);K=len(N);caps=_caps(caps)
    if sum(N)!=N_PER_K*K:
        raise ValueError('Saved calibration size must equal 4000 times support size')
    if not isinstance(region,Mapping):
        raise ValueError('A saved calibration region is required')
    source_N,source_T=counts(region['N'],region['T'])
    if (source_N,source_T)!=(N,T):
        raise ValueError('Calibration counts differ from the saved region')
    if rational(region['alpha'])!=ALPHA:
        raise ValueError('Saved calibration allocation must equal 1/40')
    try:
        ids=tuple(_immutable_id(i) for i in ids)
        derived_order=exact_order(N,T,ids)
    except (TypeError,ValueError) as error:
        raise ValueError('Distinct comparable immutable IDs are required on the full support') from error
    order=tuple(order)
    if any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for i in order) or order!=derived_order:
        raise ValueError('Saved order differs from calibration-estimated propensity/immutable-ID order')
    order=tuple(map(int,order))
    tree_orders=tuple(tuple(o) for o in tree_orders)
    # This validates both complete trees before any cell certification occurs.
    family_membership(tuple((i,) for i in range(K)),order,tree_orders)
    saved=_saved_records(cells)
    engine=CellEngine(region,saved,records_context=records_context)
    expected=RootPolicy() if expected_root_policy is None else expected_root_policy
    if isinstance(expected,Mapping):
        if set(expected)!=set(RootPolicy().__dict__):
            raise ValueError('Expected root policy must contain every policy field')
        expected=RootPolicy(**dict(expected))
    if not isinstance(expected,RootPolicy) or engine.policy!=expected:
        raise ValueError('Saved numerical root policy differs from the frozen study policy')
    setup_seconds=time.perf_counter()-started

    family_cells=tuple(sorted(set(exact.interval_cells(order)).union(
        *(set(exact.build_tree(o)) for o in tree_orders))))
    tick=time.perf_counter()
    records={cell:engine.cell(cell) for cell in family_cells}
    certification_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    search=complete_pilot_search(order,tree_orders,engine,caps)
    search_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    by_cap={}
    for cap,source in search['by_cap'].items():
        choice=dict(source)
        selected=tuple(records[cell] for cell in choice['partition'])
        if any(record['status']!='full' or record.get('coupled') is None for record in selected):
            raise ArithmeticError('Selected partition lacks completed finite cell bounds')
        cell_U=tuple(record['coupled'] for record in selected)
        U=max(cell_U,default=F(1))
        if U!=choice['certificate'] or not F(1)<=U<=cap:
            raise ArithmeticError('Selected certificate is not its actual completed cell maximum')
        completed=bool(choice['count_optimal'] and choice['deterministic_tie_complete'])
        if choice['identity'] and U!=1:
            raise ArithmeticError('Structural identity certificate must equal one')
        choice.update(K=K,U=U,cell_U=cell_U,
            cell_bounds={cell:record['coupled'] for cell,record in zip(choice['partition'],selected)},
            selected_cells=selected,completed=completed,optimization_completed=completed,
            region_valid=bool(engine.region['valid']),
            selection_scope='complete_calibration_order_combined_family')
        by_cap[cap]=choice
    completed_counts=[by_cap[cap]['cells'] for cap in caps if by_cap[cap]['completed']]
    if completed_counts!=sorted(completed_counts,reverse=True):
        raise ArithmeticError('Completed minimum-count selections increase with the cap')
    validation_seconds=time.perf_counter()-tick

    saved_by_cell={tuple(record['cell']):record for record in saved}
    additional=[]
    for cell,record in records.items():
        old=saved_by_cell.get(cell)
        reused=(len(cell)==1 and old is not None or old is not None and
                (old['status'] in ('full','boundary') or old['status']=='rejected'
                 and rational(old['rejection_bound'])>CAPS[-1]))
        if not reused:
            additional.append(record)
    evidence=engine.evidence()
    # A mutable rebuilt copy supports export_release's projection cache, while
    # none of the source region's private caches or extra metadata are reused.
    rebuilt=dict(engine.region)
    rebuilt['root_fallbacks']=engine.source_root_fallbacks
    return {'K':K,'N':N,'T':T,'ids':ids,'order':order,'tree_orders':tree_orders,'caps':caps,
        'region':rebuilt,'records':records,'by_cap':by_cap,'family_cells':family_cells,
        'additional_cell_records':tuple(additional),'context_key':engine.context_key,
        'cell_status_counts':dict(Counter(record['status'] for record in records.values())),
        'cache':evidence['cache'],'backend_seconds':evidence['backend_seconds'],
        'source_root_fallbacks':engine.source_root_fallbacks,
        'timings':{'region_rebuild_validation_seconds':setup_seconds,
            'cell_certification_seconds':certification_seconds,'partition_search_seconds':search_seconds,
            'selection_validation_seconds':validation_seconds,'total_seconds':time.perf_counter()-started},
        'timing_scope':'Additional cached computation on saved calibration endpoints and compatible cell evidence; no root reconstruction or cold end-to-end benchmark',
        'checks':{'completed_count_monotone':True,'full_support_preserved':True,
                  'outcome_or_population_inputs_accepted':False}}


def check_capacity_admission(choice,saved_choice,B,*,cap=F(3,2)):
    """Check the threshold/capacity existence equivalence when both complete."""
    B=_integer(B,'Reporting capacity',1);cap=rational(cap)
    if cap!=F(3,2) or rational(choice['cap'])!=cap:
        raise ValueError('The preserved release-study comparison is at cap 3/2')
    if B>_integer(choice['K'],'Support size',1):
        raise ValueError('Reporting capacity exceeds complete support')
    saved_choice=saved_choice.get('verified_capacity',saved_choice)
    if 'capacity' in saved_choice and _integer(saved_choice['capacity'],'Saved capacity',1)!=B:
        raise ValueError('Saved capacity differs from the comparison capacity')
    status=saved_choice.get('status')
    source_completed=bool(saved_choice.get('completed',False))
    threshold_completed=bool(choice.get('completed',False))
    if not (threshold_completed and source_completed and status in ('finite','infinite')):
        return {'checked':False,'status':'not_checked_unresolved','cap':cap,'capacity':B,
                'threshold_admitted':None,'capacity_admitted':None}
    if not choice.get('region_valid',True):
        return {'checked':False,'status':'not_checked_invalid_region','cap':cap,'capacity':B,
                'threshold_admitted':None,'capacity_admitted':None}
    threshold_admitted=choice['cells']<=B
    capacity_admitted=(status=='finite' and rational(saved_choice['certificate'])<=cap
                       and _integer(saved_choice['cells'],'Saved selected count',1)<=B)
    if threshold_admitted!=capacity_admitted:
        raise ArithmeticError('Completed threshold/capacity admission decisions disagree')
    return {'checked':True,'status':'agrees','cap':cap,'capacity':B,
            'threshold_admitted':threshold_admitted,'capacity_admitted':capacity_admitted}
