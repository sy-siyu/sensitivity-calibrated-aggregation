"""Synthetic reporting tables only; no simulations or retained paper outputs."""
import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest

from sca_reproduction import plots

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'sca-test-matplotlib'))


def write_rows(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def metric(name, mean, n, lo=None, hi=None):
    return {name+'_mean':mean,name+'_n':n,
            name+'_ci95_lower':mean-.01 if lo is None else lo,
            name+'_ci95_upper':mean+.02 if hi is None else hi}


def fixture(workspace, full=False):
    """All layout points, with invented values and explicit fixture scope."""
    count = 100 if full else 2
    summaries = {
        'efficiency':{'independent_input_units':{'historical':4800,'release':300,'recovery':300},
                      'conditions':{'historical':432,'release':45,'recovery':144}},
        'split':{'conditions':48,'replications_per_condition':90},
        'release':{'calibration_conditions':30,'replications_per_condition':100},
        'precision':{'expected_independent_blocks_per_law':100,'independent_blocks':300,
                     'assignment_conditions':15,'method_conditions':320},
        'computation':{'primary_cases':200,'capacity_cases':30,
                       'independent_calibration_primary_datasets':200},
    }
    for name, data in summaries.items():
        root = workspace/name; root.mkdir(parents=True)
        data['status'] = 'report_complete' if full else 'incomplete'
        (root/'summary.json').write_text(json.dumps(data))
    rows = []
    for condition in range(48 if full else 1):
        for rep in range(count):
            rows.append(dict(condition_id=f'synthetic_{condition}',replication=rep,
                procedure='original',cap='3/2',mass_mode='unknown_mass',oracle_capacity_status='finite',
                oracle_margin_log=.05+condition/1000,minimax_log_inflation=.4+rep/1000,
                threshold_log_inflation=999))
    write_rows(workspace/'efficiency/tables/historical_rows.csv',rows)
    rows = []
    for law in ('homogeneous','one_exceptional_state'):
        for mode in ('known_mass','unknown_mass'):
            for procedure in plots.PROCEDURES:
                for n in plots.RECOVERY_SIZES:
                    rows.append(dict(law_id=law,mass_mode=mode,procedure=procedure,n=n,
                        n_blocks=300 if full else count,certified_n=300 if full else count,
                        certified_mean=.4,certified_cp95_lower=.17,certified_cp95_upper=.68))
    write_rows(workspace/'efficiency/tables/recovery_condition_summary.csv',rows)
    rows = []
    for contrast,label in plots.CONTRASTS:
        rows.append(dict(scope='all',cap='3/2',contrast=contrast,
            difference_pp_conditions=48 if full else 1,
            difference_pp_expected_conditions=48 if full else 1,
            difference_pp_paired_datasets=4320 if full else count,
            difference_pp_all_conditions_present=full,difference_pp_mean=3.1,
            difference_pp_analytic_Normal95_lower=-1.234,
            difference_pp_analytic_Normal95_upper=8.4))
    write_rows(workspace/'split/tables/paired_grid.csv',rows)
    rows = []
    for law,K in enumerate((64,64,16)):
        for method in ('verified_capacity','equal_calibration_mass'):
            for B in (K//16,K//8,K//4,K//2,K):
                rows.append(dict(law_index=law,K=K,method=method,B=B,n=count,
                    **metric('certificate',1.3,count),**metric('true_distortion',1.15,count),
                    **metric('oracle_distortion',1.1,count),**metric('admitted',.5,count)))
    write_rows(workspace/'release/tables/calibration_summary.csv',rows)
    rows = [dict(law_index=law,cap=str(cap),n_blocks=count,
                 **metric('compression_pp',50,count)) for law in range(3) for cap in plots.CAPS]
    write_rows(workspace/'precision/tables/assignment_summary.csv',rows)
    rows = []
    for law, profiles in ((0,('constant','aligned','alternating')),
        (1,('constant','aligned','alternating')),(2,('low_heterogeneity','high_heterogeneity'))):
        for profile in profiles:
            for ratio in (16,256):
                for method in ('z_global','z_cellwise','raw_stratified','raw_pooled'):
                    for cap in plots.CAPS:
                        rows.append(dict(law_index=law,profile=profile,m_ratio=ratio,method=method,
                            cap=str(cap),n_blocks=count,**metric('width',.4,count),
                            admitted_width_mean=.01))
    write_rows(workspace/'precision/tables/method_width_target_summary.csv',rows)
    for table,supports,n in (('primary_overview',(16,32,64,128,256),20),
                             ('capacity_overview',(64,128,256),5)):
        rows = [dict(law_index=law,K=K,n_cases=n if full else count,
            n_process_completed=n if full else count,n_selection_complete=n if full else count,
            total_seconds_n=n if full else count,total_seconds_median=K/8,
            total_seconds_q25=K/10,total_seconds_q75=K/6,total_seconds_mean=999)
            for law in range(2) for K in supports]
        write_rows(workspace/'computation/tables'/(table+'.csv'),rows)


class PaperInputChecks(unittest.TestCase):
    def test_full_synthetic_grid_validates_and_values_keep_their_scope(self):
        with tempfile.TemporaryDirectory(prefix='sca-paper-fixture-') as directory:
            workspace = Path(directory); fixture(workspace,full=True)
            data = {name:loader(workspace/name,False) for name,loader in plots._LOADERS.items()}
            self.assertEqual(len(data['efficiency'][0]),4800)
            self.assertEqual(float(data['efficiency'][0][0]['minimax_log_inflation']),.4)
            self.assertEqual(len(data['release']),30)
            self.assertEqual(len(data['precision'][1]),320)
            self.assertEqual(len(data['computation'][0]),10)

    def test_partial_report_refused_by_default_before_output(self):
        with tempfile.TemporaryDirectory(prefix='sca-paper-fixture-') as directory:
            workspace = Path(directory); fixture(workspace)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                plots.plot('all',workspace)
            self.assertFalse((workspace/'paper_figures').exists())

    def test_denominator_duplicate_and_unresolved_points_are_not_dropped(self):
        with tempfile.TemporaryDirectory(prefix='sca-paper-fixture-') as directory:
            workspace = Path(directory); fixture(workspace,full=True)
            path = workspace/'precision/tables/method_width_target_summary.csv'
            rows = plots._rows(path); rows[0]['width_n'] = 99; write_rows(path,rows)
            with self.assertRaisesRegex(ValueError,'width_n must be 100'):
                plots._load_precision(workspace/'precision',False)
            path = workspace/'release/tables/calibration_summary.csv'
            rows = plots._rows(path); write_rows(path,rows+[rows[0]])
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                plots._load_release(workspace/'release',False)
            path = workspace/'efficiency/tables/historical_rows.csv'
            rows = plots._rows(path); rows[0]['oracle_capacity_status'] = 'unresolved'; write_rows(path,rows)
            with self.assertRaisesRegex(ValueError,'cannot be silently removed'):
                plots._load_efficiency(workspace/'efficiency',False)

    def test_missing_and_inconsistent_saved_intervals_fail(self):
        with self.assertRaisesRegex(ValueError,'Inconsistent'):
            plots._interval({'x_mean':'1','x_ci95_lower':'1.1','x_ci95_upper':'2'},'x')
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            plots._number({'x':'NA'},'x')
        with self.assertRaisesRegex(ValueError,'Unknown paper'):
            plots.plot('generic','/unused')

    def test_runtime_module_has_no_private_source_or_solver_dependency(self):
        source = inspect.getsource(plots)
        for forbidden in ('JMLR_manuscript','Desktop/CPRL','results/jmlr_',
                          'manuscript_bound','RootVerifier','cell_certificate'):
            self.assertNotIn(forbidden,source)


@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Optional plotting dependency unavailable')
class PaperRendering(unittest.TestCase):
    def test_five_partial_layouts_and_receipts_use_saved_values(self):
        with tempfile.TemporaryDirectory(prefix='sca-paper-fixture-') as directory:
            workspace = Path(directory); fixture(workspace)
            before = {p:p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
            result = plots.plot('all',workspace,allow_partial=True)
            self.assertTrue(result['partial_fixture'])
            self.assertEqual(set(result['figures']),set(plots.STUDIES))
            for study,n in (('efficiency',38),('split',7),('release',75),('precision',75),('computation',16)):
                record = result['figures'][study]
                self.assertEqual(record['points'],n)
                self.assertTrue(Path(record['pdf']).read_bytes().startswith(b'%PDF'))
                self.assertTrue(Path(record['png']).read_bytes().startswith(b'\x89PNG'))
            split = plots._rows(Path(result['figures']['split']['coordinates']))
            self.assertEqual(float(split[0]['mean']),3.1)
            self.assertEqual(float(split[0]['lower']),-1.234)
            self.assertEqual(float(split[0]['upper']),8.4)
            recovery = [r for r in plots._rows(Path(result['figures']['efficiency']['coordinates']))
                        if r['panel']=='unknown_mass_recovery']
            self.assertEqual(len(recovery),36)
            self.assertTrue(all(float(r['lower'])==.17 and float(r['upper'])==.68
                                for r in recovery))
            precision = plots._rows(Path(result['figures']['precision']['coordinates']))
            self.assertTrue(all(float(r['mean'])==.4 for r in precision if r['panel']=='confidence_width'))
            computation = plots._rows(Path(result['figures']['computation']['coordinates']))
            self.assertEqual(float(computation[0]['mean']),2)
            self.assertTrue(all(r['source_mean']=='total_seconds_median' for r in computation))
            self.assertTrue(all(path.read_bytes()==content for path,content in before.items()))
            from matplotlib import pyplot as plt
            self.assertFalse(plt.get_fignums())

    def test_release_admission_annotations_fit_export_canvas(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        with tempfile.TemporaryDirectory(prefix='sca-paper-fixture-') as directory:
            workspace = Path(directory); fixture(workspace,full=True)
            data = plots._load_release(workspace/'release',False)
            for row in data.values():
                row['admitted_mean'] = '1'
            with plt.rc_context({'font.size':9.3,'axes.labelsize':9.3,
                                 'axes.titlesize':10,'legend.fontsize':9.3,
                                 'xtick.labelsize':9.3,'ytick.labelsize':9.3}):
                fig,receipt = plots._draw_release(plt,data)
                try:
                    # The PNG export uses200dpi; test text extents with that
                    # renderer so clipping cannot hide behind a lower DPI.
                    fig.set_dpi(200); fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    annotations = [text for ax in fig.axes for text in ax.texts]
                    self.assertEqual(len(annotations),30)
                    for text in annotations:
                        bounds = text.get_window_extent(renderer)
                        self.assertGreaterEqual(bounds.x0,fig.bbox.x0)
                        self.assertLessEqual(bounds.x1,fig.bbox.x1)
                        self.assertGreaterEqual(bounds.y0,fig.bbox.y0)
                        self.assertLessEqual(bounds.y1,fig.bbox.y1)
                    rightmost = [text for ax in fig.axes for text in ax.texts
                                 if text.xy[0]==max(t.xy[0] for t in ax.texts)]
                    self.assertEqual(len(rightmost),6)
                    self.assertTrue(all(text.get_ha()=='right' for text in rightmost))
                    self.assertTrue(all('100/100' in text.get_text() for text in rightmost))
                    fig.savefig(workspace/'release_annotations.png',dpi=200)
                finally:
                    plt.close(fig)

    def test_partial_watermark_is_explicit_on_each_figure(self):
        from unittest.mock import patch
        from matplotlib.figure import Figure
        original = Figure.savefig; watermarks = []
        def save(fig,*args,**kwargs):
            watermarks.append(any(t.get_text()=='PARTIAL LAYOUT FIXTURE' for t in fig.texts))
            return original(fig,*args,**kwargs)
        with tempfile.TemporaryDirectory(prefix='sca-paper-fixture-') as directory:
            workspace = Path(directory); fixture(workspace)
            with patch.object(Figure,'savefig',save):
                plots.plot('split',workspace,output_dir=workspace/'custom',allow_partial=True)
        self.assertEqual(watermarks,[True,True])


if __name__ == '__main__':
    unittest.main()
