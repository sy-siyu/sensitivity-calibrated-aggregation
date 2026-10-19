"""The five main paper layouts, drawn only from public reporting tables.

Install the optional plotting dependencies before calling :func:`plot`.  No
simulation, confidence-bound calculation, or uncertainty calculation occurs in
this module.  Full-grid checks are deliberately stricter than an ordinary CSV
plot: a bounded smoke run is not a paper reproduction.  Synthetic layout tests
may opt into ``allow_partial=True``; those figures carry a visible watermark.
"""
from __future__ import annotations

import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import tempfile


STUDIES = ('efficiency', 'split', 'release', 'precision', 'computation')
BASENAMES = dict(zip(STUDIES, ('statistical_efficiency', 'split_search_contrasts',
    'release_capacity', 'precision_operating_curves', 'computational_cost')))
CAPS = tuple(map(Fraction, ('1', '11/10', '5/4', '3/2', '2')))
RECOVERY_SIZES = (2000, 4000, 8000, 16000, 32000, 64000, 128000,
                  256000, 512000, 1000000, 2000000, 3000000)
PROCEDURES = ('original', 'allocation_control', 'pooled_alternative')
CONTRASTS = (
    ('C-B', 'Remove stopping'),
    ('D-C', 'Full sample,\nfixed proposals'),
    ('E-D', 'Complete pilot-order\nsearch'),
    ('F-E', 'Full-data order'),
    ('C-A', 'Held-out, all proposals\nminus LTT'),
    ('F-C', 'SCA minus held-out,\nall proposals'),
    ('F-A', 'SCA minus LTT'),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    _require(rows, f'No reporting rows in {path}')
    return rows


def _number(row, key):
    try:
        value = float(Fraction(row[key]))
    except (KeyError, ValueError, ZeroDivisionError, TypeError, OverflowError) as error:
        raise ValueError(f'Missing or nonfinite plotting value {key}: {row.get(key)!r}') from error
    _require(math.isfinite(value), f'Nonfinite plotting value {key}')
    return value


def _integer(row, key):
    value = _number(row, key)
    _require(value.is_integer(), f'Expected integer {key}')
    return int(value)


def _cap(row):
    try:
        return Fraction(row['cap'])
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as error:
        raise ValueError('Missing exact distortion cap') from error


def _interval(row, metric, lower='_ci95_lower', upper='_ci95_upper', mean='_mean'):
    result = tuple(_number(row, metric+suffix) for suffix in (mean, lower, upper))
    y, lo, hi = result
    _require(lo <= y+1e-12 and y <= hi+1e-12 and lo <= hi,
             f'Inconsistent saved interval for {metric}')
    return result


def _denominator(row, key, expected, partial):
    value = _integer(row, key)
    _require(value > 0 and (partial or value == expected),
             f'{key} must be {expected}, found {value}; incomplete inputs require allow_partial=True')


def _unique(rows, key, expected, label):
    result = {}
    for row in rows:
        item = key(row)
        _require(item not in result, f'Duplicate {label}: {item}')
        result[item] = row
    _require(set(result) == set(expected), f'Missing or unexpected {label} points')
    return result


def _summary(root, partial, checks=()):
    path = root/'summary.json'
    with path.open() as stream:
        summary = json.load(stream)
    if not partial:
        _require(summary.get('status') in ('complete', 'report_complete'),
                 f'{root.name} report is incomplete; run its full public report first')
        _require(not summary.get('development_fixture_only', False),
                 'Development reporting cannot supply a complete paper figure')
        for key, expected in checks:
            _require(summary.get(key) == expected,
                     f'{root.name} summary requires {key}={expected!r}')
    return summary


def _load_efficiency(root, partial):
    _summary(root, partial, (
        ('independent_input_units', {'historical':4800, 'release':300, 'recovery':300}),
        ('conditions', {'historical':432, 'release':45, 'recovery':144}),
    ))
    historical = [r for r in _rows(root/'tables/historical_rows.csv')
                  if r.get('procedure') == 'original' and _cap(r) == Fraction(3,2)]
    _require(historical, 'No SCA cap-3/2 historical oracle-count points')
    identities = set()
    by_condition = {}
    for row in historical:
        identity = (row['condition_id'], _integer(row, 'replication'))
        _require(identity not in identities, 'Repeated historical calibration block')
        identities.add(identity)
        by_condition.setdefault(identity[0], set()).add(identity[1])
        _require(row.get('mass_mode') == 'unknown_mass', 'Expected unknown-mass historical points')
        _require(row.get('oracle_capacity_status') == 'finite',
                 'A nonfinite/unresolved oracle-count point cannot be silently removed')
        _number(row, 'oracle_margin_log'); _number(row, 'minimax_log_inflation')
    if not partial:
        _require(len(historical) == 4800 and len(by_condition) == 48 and
                 all(reps == set(range(100)) for reps in by_condition.values()),
                 'Efficiency figure requires all48 conditions and100 calibration blocks each')
    recovery = _rows(root/'tables/recovery_condition_summary.csv')
    expected = {(law, mode, procedure, n) for law in ('homogeneous', 'one_exceptional_state')
        for mode in ('known_mass', 'unknown_mass') for procedure in PROCEDURES for n in RECOVERY_SIZES}
    indexed = _unique(recovery, lambda r:(r['law_id'], r['mass_mode'], r['procedure'],
        _integer(r, 'n')), expected, 'recovery condition')
    for row in indexed.values():
        _denominator(row, 'n_blocks', 300, partial)
        _denominator(row, 'certified_n', 300, partial)
        _interval(row, 'certified', '_cp95_lower', '_cp95_upper')
    return historical, indexed


def _load_split(root, partial):
    _summary(root, partial, (('conditions', 48), ('replications_per_condition', 90)))
    rows = [r for r in _rows(root/'tables/paired_grid.csv')
            if r.get('scope') == 'all' and _cap(r) == Fraction(3,2)
            and r.get('contrast') in dict(CONTRASTS)]
    indexed = _unique(rows, lambda r:r['contrast'], dict(CONTRASTS), 'split contrast')
    for row in indexed.values():
        for key, expected in (('difference_pp_conditions',48),
                              ('difference_pp_expected_conditions',48),
                              ('difference_pp_paired_datasets',4320)):
            _denominator(row, key, expected, partial)
        if not partial:
            _require(row.get('difference_pp_all_conditions_present') == 'True',
                     'Split contrast lacks one or more fixed conditions')
        _interval(row, 'difference_pp', '_analytic_Normal95_lower', '_analytic_Normal95_upper')
    return indexed


def _load_release(root, partial):
    _summary(root, partial, (('calibration_conditions',30), ('replications_per_condition',100)))
    rows = _rows(root/'tables/calibration_summary.csv')
    expected = {(law, method, B) for law, K in enumerate((64,64,16))
        for method in ('verified_capacity','equal_calibration_mass')
        for B in (K//16,K//8,K//4,K//2,K)}
    indexed = _unique(rows, lambda r:(_integer(r,'law_index'),r['method'],_integer(r,'B')),
                      expected, 'release-capacity condition')
    for (law, method, B), row in indexed.items():
        _require(_integer(row,'K') == (64,64,16)[law], 'Release support changed')
        _denominator(row, 'n', 100, partial)
        for metric in ('certificate','true_distortion','oracle_distortion'):
            _denominator(row, metric+'_n', 100, partial)
            _interval(row, metric)
        _denominator(row, 'admitted_n', 100, partial)
        _require(0 <= _number(row,'admitted_mean') <= 1, 'Invalid admission rate')
    return indexed


def _load_precision(root, partial):
    _summary(root, partial, (('expected_independent_blocks_per_law',100),
        ('independent_blocks',300), ('assignment_conditions',15), ('method_conditions',320)))
    assignment = _unique(_rows(root/'tables/assignment_summary.csv'),
        lambda r:(_integer(r,'law_index'),_cap(r)),
        {(law,cap) for law in range(3) for cap in CAPS}, 'precision assignment condition')
    methods = _rows(root/'tables/method_width_target_summary.csv')
    expected = {(law,profile,m,method,cap) for law, profiles in
        ((0,('constant','aligned','alternating')),(1,('constant','aligned','alternating')),
         (2,('low_heterogeneity','high_heterogeneity')))
        for profile in profiles for m in (16,256)
        for method in ('z_global','z_cellwise','raw_stratified','raw_pooled') for cap in CAPS}
    indexed = _unique(methods, lambda r:(_integer(r,'law_index'),r['profile'],
        _integer(r,'m_ratio'),r['method'],_cap(r)), expected, 'precision method condition')
    for row in assignment.values():
        _denominator(row,'n_blocks',100,partial)
        _denominator(row,'compression_pp_n',100,partial)
        _interval(row,'compression_pp')
    for row in indexed.values():
        _denominator(row,'n_blocks',100,partial)
        _denominator(row,'width_n',100,partial)
        _interval(row,'width')
    return assignment, indexed


def _load_computation(root, partial):
    _summary(root, partial, (('primary_cases',200), ('capacity_cases',30),
        ('independent_calibration_primary_datasets',200)))
    answer = []
    for table, supports, count in (('primary_overview',(16,32,64,128,256),20),
                                   ('capacity_overview',(64,128,256),5)):
        indexed = _unique(_rows(root/'tables'/(table+'.csv')),
            lambda r:(_integer(r,'law_index'),_integer(r,'K')),
            {(law,K) for law in range(2) for K in supports}, table)
        for row in indexed.values():
            for key in ('n_cases','n_process_completed','n_selection_complete','total_seconds_n'):
                _denominator(row,key,count,partial)
            median, lo, hi = _interval(row,'total_seconds','_q25','_q75','_median')
            _require(lo > 0, 'Log computation plot requires positive measured time')
        answer.append(indexed)
    return answer


def _coordinates(rows, *, panel, xkey, metric, lower='_ci95_lower',
                 upper='_ci95_upper', mean='_mean', **labels):
    receipt = []
    for row in rows:
        y, lo, hi = _interval(row,metric,lower,upper,mean)
        receipt.append(dict(panel=panel, **labels, x=row[xkey], mean=y, lower=lo, upper=hi,
            source_x=xkey, source_mean=metric+mean, source_lower=metric+lower,
            source_upper=metric+upper))
    return receipt


def _errors(ax, rows, xkey, metric, *, lower='_ci95_lower', upper='_ci95_upper',
            mean='_mean', **style):
    x = [_number(row,xkey) for row in rows]
    points = [_interval(row,metric,lower,upper,mean) for row in rows]
    y, lo, hi = (list(values) for values in zip(*points))
    return ax.errorbar(x,y,yerr=[[max(0,a-b) for a,b in zip(y,lo)],
        [max(0,a-b) for a,b in zip(hi,y)]], **style)


def _draw_efficiency(plt, data):
    from matplotlib.ticker import ScalarFormatter, FixedLocator, NullFormatter
    historical, recovery = data
    fig, axes = plt.subplots(1,2,figsize=(6,3.55))
    fig.subplots_adjust(left=.10,right=.98,bottom=.28,top=.87,wspace=.47)
    x = [_number(r,'oracle_margin_log') for r in historical]
    y = [_number(r,'minimax_log_inflation') for r in historical]
    axes[0].scatter(x,y,s=5,alpha=.22,color='#1976b5',linewidths=0)
    axes[0].plot([0,max(x)],[0,max(x)],':',color='black',label='Equality')
    axes[0].set_yscale('symlog',linthresh=.1,linscale=1)
    axes[0].set_ylim(-.01,max(max(y)*1.15,.1))
    axes[0].yaxis.set_major_locator(FixedLocator([0,.1,1,10]))
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xticks([0,.05,.1,.15,.2])
    axes[0].set(xlabel='Best population log margin',ylabel='Oracle-count log inflation',
                title='SCA oracle-count barrier')
    axes[0].legend(loc='lower right',frameon=False,handlelength=1.3)
    receipt = [dict(panel='oracle_count_barrier', condition_id=r['condition_id'],
        replication=r['replication'],x=r['oracle_margin_log'],mean=r['minimax_log_inflation'],
        source_x='oracle_margin_log',source_mean='minimax_log_inflation') for r in historical]
    for procedure,label,color,style,marker in (
        ('original','SCA (Coupled CP)','#1976b5','-','o'),
        ('allocation_control','Allocation control','#cc650c','--','s'),
        ('pooled_alternative','Fixed-group pooled','#24883e','-.','^')):
        rows = [recovery['homogeneous','unknown_mass',procedure,n] for n in RECOVERY_SIZES]
        vals = [_interval(r,'certified','_cp95_lower','_cp95_upper') for r in rows]
        axes[1].plot(RECOVERY_SIZES,[v[0] for v in vals],linestyle=style,marker=marker,
            markersize=3,linewidth=1.4,color=color,label=label)
        axes[1].fill_between(RECOVERY_SIZES,[v[1] for v in vals],[v[2] for v in vals],
                             color=color,alpha=.13)
        receipt += _coordinates(rows,panel='unknown_mass_recovery',xkey='n',metric='certified',
            lower='_cp95_lower',upper='_cp95_upper',procedure=procedure)
    axes[1].axhline(.9,color='#555555',linestyle=':',linewidth=1)
    axes[1].set_xscale('log')
    axes[1].set_xticks([2000,16000,128000,1000000],labels=['2k','16k','128k','1m'])
    axes[1].set(ylim=(-.02,1.035),yticks=[0,.5,1],xlabel='Assignment observations',
                ylabel='One-cell recovery fraction',title='Unknown-mass recovery')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,
               frameon=False,handlelength=1.3,columnspacing=.9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15)
    return fig, receipt


def _draw_split(plt, indexed):
    fig, ax = plt.subplots(figsize=(6,3.8))
    fig.subplots_adjust(left=.365,right=.675,bottom=.14,top=.95)
    receipt = []
    for y,(contrast,label) in enumerate(CONTRASTS):
        row = indexed[contrast]
        m,lo,hi = _interval(row,'difference_pp','_analytic_Normal95_lower','_analytic_Normal95_upper')
        color = '#1976b5' if y < 4 else '#333333'
        ax.errorbar(m,y,xerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt='o',color=color,
                    markersize=3.5,capsize=2,linewidth=1.3)
        # Keep the leading decimal point when dropping the leading zero.
        def number(value):
            return f'{value:.3f}'.replace('-0.','-.').replace('0.','.',1) if abs(value)<1 else f'{value:.3f}'
        ax.text(1.06,y,f'{number(m)} [{number(lo)}, {number(hi)}]',
            transform=ax.get_yaxis_transform(),va='center',clip_on=False)
        receipt.append(dict(panel='compression_contrasts',contrast=contrast,cap=row['cap'],
            x=m,y=y,mean=m,lower=lo,upper=hi,source_x='difference_pp_mean',
            source_mean='difference_pp_mean',source_lower='difference_pp_analytic_Normal95_lower',
            source_upper='difference_pp_analytic_Normal95_upper'))
    ax.set_yticks(range(len(CONTRASTS)),[label for contrast,label in CONTRASTS])
    ax.set(ylim=(len(CONTRASTS)-.5,-.5),xlim=(-6,16),xticks=[-5,0,5,10,15],
           xlabel='Compression difference (pp)')
    ax.axvline(0,color='#555555',linewidth=.8)
    ax.axhline(3.5,color='#aaaaaa',linewidth=.6)
    ax.grid(axis='x',alpha=.16)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=8)
    return fig, receipt


def _draw_release(plt, indexed):
    from matplotlib.ticker import ScalarFormatter, FixedLocator, NullFormatter
    fig, axes = plt.subplots(3,1,figsize=(6,7))
    fig.subplots_adjust(left=.15,right=.97,bottom=.16,top=.90,hspace=1.05)
    receipt = []
    for law,(ax,title) in enumerate(zip(axes,('Balanced controlled law','Skewed controlled law',
                                               'IHDP reference population'))):
        ax.set_title(title,pad=38)
        for method,color,short,label in (
            ('verified_capacity','#1976b5','SCA','SCA'),
            ('equal_calibration_mass','#cc650c','B','Equal-mass baseline')):
            rows = [r for (i,method_key,B),r in indexed.items() if i==law and method_key==method]
            rows.sort(key=lambda r:_integer(r,'B'))
            x = [_integer(r,'B') for r in rows]
            for metric,style,marker,alpha in (('certificate','-','o',1),
                                              ('true_distortion','--',None,.7)):
                _errors(ax,rows,'B',metric,color=color,linestyle=style,marker=marker,
                    markersize=4,capsize=2,linewidth=1.4,alpha=alpha,
                    label=label+(': certificate' if metric=='certificate' else ': true distortion'))
                receipt += _coordinates(rows,panel='release_capacity',xkey='B',metric=metric,
                                         law_index=law,method=method)
            for row in rows:
                admitted = round(_number(row,'admitted_mean')*_integer(row,'n'))
                ax.annotate(f"{short} {admitted}/{_integer(row,'n')}",
                    xy=(_integer(row,'B'),1),xycoords=('data','axes fraction'),
                    xytext=(-6 if _integer(row,'B')==x[-1] else 0,
                            14 if short=='SCA' else 3),textcoords='offset points',
                    color=color,ha='center',va='bottom',annotation_clip=False)
            if method == 'verified_capacity':
                ax.plot(x,[_number(r,'oracle_distortion_mean') for r in rows],':',color='black',
                    marker='x',markersize=4,label='Same-family population oracle')
                receipt += _coordinates(rows,panel='release_capacity',xkey='B',metric='oracle_distortion',
                                         law_index=law,method='population_oracle')
        ax.axhline(1.5,color='#b72121',linestyle='-.',linewidth=1.2,
                   label='Admissible certificate: 1.5')
        ax.set_xscale('log',base=2); ax.set_xticks(x)
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator(([1,1.2,1.5,1.8],[1,10,100],[1,1.5,2,3])[law]))
        ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel('Distortion / certificate')
        if law == 2:
            ax.set_xlabel('Maximum reporting groups B')
        ax.grid(alpha=.16); ax.spines[['top','right']].set_visible(False)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,
               bbox_to_anchor=(.5,.005),columnspacing=.8,handlelength=2)
    return fig, receipt


def _draw_precision(plt, data):
    assignment, methods = data
    fig, axes = plt.subplots(2,2,figsize=(6,5.1))
    fig.subplots_adjust(left=.10,right=.985,top=.89,bottom=.20,hspace=.49,wspace=.30)
    receipt = []
    ax = axes[0,0]
    for law,label,color,marker in ((0,'Balanced','#2478b5','o'),
        (1,'Weak overlap','#d27622','s'),(2,'IHDP reference','#37864b','^')):
        rows = [assignment[law,cap] for cap in CAPS]
        _errors(ax,rows,'cap','compression_pp',color=color,marker=marker,
                markersize=3,capsize=1.5,linewidth=1.1,label=label)
        receipt += _coordinates(rows,panel='compression',xkey='cap',metric='compression_pp',law_index=law)
    ax.set_title('(a) SCA compression',loc='left')
    ax.set(ylabel='Compression (%)',ylim=(-4,102),yticks=[0,25,50,75,100])
    handles,labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,1.005),ncol=3,
               frameon=False,columnspacing=1.4,handlelength=1.5)
    for ax,law,profile,title in ((axes[0,1],0,'aligned','(b) Balanced / aligned'),
        (axes[1,0],1,'aligned','(c) Weak overlap / aligned'),
        (axes[1,1],2,'high_heterogeneity','(d) IHDP / high heterogeneity')):
        for method,m,color,marker,style,label in (
            ('z_global',16,'#2478b5','o','-','SCA + global inference, $m/K=16$'),
            ('z_global',256,'#c04a36','^','-','SCA + global inference, $m/K=256$'),
            ('raw_stratified',256,'#555555',None,'--','Shared raw, $m/K=256$'),
            ('raw_pooled',256,'#8d5da6','s',':','Pooled raw, $m/K=256$')):
            rows = [methods[law,profile,m,method,cap] for cap in CAPS]
            _errors(ax,rows,'cap','width',color=color,marker=marker,linestyle=style,
                markersize=3,capsize=1.5,linewidth=1.1,label=label)
            receipt += _coordinates(rows,panel='confidence_width',xkey='cap',metric='width',
                law_index=law,profile=profile,m_ratio=m,method=method)
        ax.axhline(.2,color='#999999',linewidth=.8,linestyle=(0,(1,2)))
        ax.set_title(title,loc='left')
        ax.set(ylabel='Confidence width',ylim=(.15,.93),yticks=[.2,.4,.6,.8])
    for ax in axes.flat:
        ax.set(xlabel='Distortion cap $c$',xlim=(.97,2.03))
        ax.set_xticks([float(c) for c in CAPS],labels=['1','1.1','1.25','1.5','2'])
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.14)
    handles,labels = axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.51,.015),ncol=2,
               frameon=False,columnspacing=1.2,handlelength=2)
    return fig, receipt


def _draw_computation(plt, data):
    fig, axes = plt.subplots(1,2,figsize=(6,2.85))
    fig.subplots_adjust(left=.10,right=.985,bottom=.19,top=.77,wspace=.29)
    receipt = []
    for ax,indexed,title,panel in zip(axes,data,
        ('(a) Fixed-cap SCA +\naggregate outcome inference',
         '(b) Capacity-constrained SCA,\nall five reporting limits'),('primary','capacity')):
        for law,color,marker,style,label in ((0,'#2478b5','o','-','Balanced / moderate overlap'),
            (1,'#d27622','s','--','Skewed / weak overlap')):
            rows = sorted((r for (i,K),r in indexed.items() if i==law),key=lambda r:_integer(r,'K'))
            _errors(ax,rows,'K','total_seconds',lower='_q25',upper='_q75',mean='_median',
                color=color,marker=marker,linestyle=style,markersize=3.5,capsize=2,linewidth=1.2,label=label)
            receipt += _coordinates(rows,panel=panel,xkey='K',metric='total_seconds',
                lower='_q25',upper='_q75',mean='_median',law_index=law)
        ax.set_xscale('log',base=2)
        ks = sorted({_integer(r,'K') for r in indexed.values()})
        ax.set_xticks(ks,labels=[str(k) for k in ks]); ax.set_yscale('log')
        ax.set(xlabel='Declared support $K$',ylabel='Seconds')
        ax.set_title(title,loc='left',pad=7)
        ax.spines[['top','right']].set_visible(False); ax.grid(which='major',alpha=.17)
    axes[0].set_yticks([.3,1,3,10,30],labels=['0.3','1','3','10','30']); axes[0].set_ylim(.3,75)
    axes[1].set_yticks([3,10,30,100,300,1000],labels=['3','10','30','100','300','1000'])
    axes[1].set_ylim(2,1500)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,1.015),ncol=2,
               frameon=False,columnspacing=1.1,handlelength=1.8)
    return fig, receipt


_LOADERS = dict(zip(STUDIES, (_load_efficiency,_load_split,_load_release,
                              _load_precision,_load_computation)))
_DRAWERS = dict(zip(STUDIES, (_draw_efficiency,_draw_split,_draw_release,
                              _draw_precision,_draw_computation)))


def plot(study, workspace, output_dir=None, *, allow_partial=False):
    """Draw current main-paper figures from completed public reporter outputs.

    ``workspace`` is the common public reproduction workspace, containing the
    study subdirectories (not a historical research directory).  All selected
    inputs are checked before any figure is written.  ``allow_partial`` is an
    explicit layout-fixture option: expected plotting grids still must exist,
    but reduced replication denominators are accepted and visibly labeled.
    The returned manifest and per-figure coordinates identify every plotted
    statistic.  Figures use saved intervals verbatim, with no new uncertainty
    estimates and no selection based on the plotted values.
    """
    _require(study in (*STUDIES,'all'), f'Unknown paper figure study {study!r}')
    workspace = Path(workspace).resolve()
    studies = STUDIES if study == 'all' else (study,)
    loaded = {name:_LOADERS[name](workspace/name,allow_partial) for name in studies}
    try:
        # Respect an explicit caller setting.  Otherwise keep font/config caches
        # in the operating system's temporary area, including sandboxed runs.
        os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'sca-matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise RuntimeError('Paper figures require the optional plotting dependencies') from error
    destination = Path(output_dir).resolve() if output_dir is not None else workspace/'paper_figures'
    destination.mkdir(parents=True,exist_ok=True)
    result = {'study':study,'partial_fixture':bool(allow_partial),'figures':{},
        'scope':'Current main-paper layouts from public reporting tables; saved uncertainty unchanged'}
    style = {'font.size':9.3,'axes.labelsize':9.3,'axes.titlesize':10,'legend.fontsize':9.3,
        'xtick.labelsize':9.3,'ytick.labelsize':9.3,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        for name in studies:
            fig = None
            try:
                with plt.rc_context({'axes.titlesize':9.3 if name=='efficiency' else 10}):
                    fig, receipt = _DRAWERS[name](plt,loaded[name])
                if allow_partial:
                    fig.text(.5,.5,'PARTIAL LAYOUT FIXTURE',ha='center',va='center',
                        rotation=27,fontsize=22,color='#a00000',alpha=.22,zorder=1000)
                basename = BASENAMES[name]
                files = {}
                for suffix in ('pdf','png'):
                    path = destination/(basename+'.'+suffix)
                    fig.savefig(path,dpi=200)
                    files[suffix] = str(path)
                coordinates = destination/(basename+'_coordinates.csv')
                fields = list(dict.fromkeys(key for row in receipt for key in row))
                with coordinates.open('w',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=fields)
                    writer.writeheader(); writer.writerows(receipt)
                result['figures'][name] = dict(files,coordinates=str(coordinates),points=len(receipt))
            finally:
                if fig is not None:
                    plt.close(fig)
    (destination/('plot_manifest_'+study+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result
