"""SCA examples, source-first experiments, tables and optional paper plots."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import runpy

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

from . import assignment, certification, relevance, releases
from . import analyses

PROJECT = Path(__file__).resolve().parents[1]
CURRENT_STUDIES = ('assignment','release','certification','relevance','recovery',
                   'efficiency','precision','outcome','computation')
STUDIES = CURRENT_STUDIES


def _configuration(path):
    config = json.loads(Path(path).read_text())
    if config.get('profile') not in ('smoke','full-paper'):
        raise ValueError('Choose the separate paper or smoke configuration')
    return config


def _override(config, study):
    return config.get('studies',{}).get(study,{})


def _guard_workspace(workspace,config,*,report_only=False):
    """Never reinterpret a smoke workspace as a completed paper design."""
    workspace=Path(workspace).resolve()
    path=workspace/'reproduction_configuration.json'
    requested={'public_configuration':config,
               'component_configurations':{p.name:json.loads(p.read_text())
                    for p in sorted((PROJECT/'configs').glob('jmlr_*.json'))}}
    if path.exists():
        if json.loads(path.read_text())!=requested:
            raise ValueError('Workspace configuration differs; use a separate workspace')
    else:
        if report_only or any((workspace/name).exists() for name in STUDIES):
            raise ValueError('No matching CLI configuration receipt; use a new workspace')
        workspace.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(requested,indent=2)+'\n')


def _run(study, workspace, config, data, limit=None, rhc_data=None):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    root = workspace/study
    if study == 'assignment':
        result=assignment.run(workspace,limit=limit)
        if result['failures']:raise RuntimeError('Assignment failures retained; inspect run_manifest.json')
        return result
    if study in ('efficiency','recovery'):
        return analyses.run(workspace,study,limit=limit,config=config)
    if study == 'certification':
        return certification.run(workspace,limit=limit)
    if study == 'relevance':
        return relevance.run(workspace,rhc_data or 'data/rhc.csv',limit=limit)
    if study == 'release':
        if not (root/'frozen_protocol.json').exists():
            if not Path(data).is_file():
                raise FileNotFoundError('Run prepare-data first, then provide --data PATH')
            releases.prepare_release(root,data,_override(config,study))
        return releases.run_release(root,limit=limit)
    if study == 'outcome':
        if not (root/'frozen_protocol.json').exists():
            releases.prepare_outcome(root,_override(config,study))
        return releases.run_outcome(root,limit=limit)
    if study == 'precision':
        if not (workspace/'release/frozen_protocol.json').exists():
            raise FileNotFoundError('Run release in the same workspace first')
        if not (root/'frozen_protocol.json').exists():
            releases.prepare_precision(root,workspace/'release',_override(config,study))
        return releases.run_precision(root,limit=limit)
    if study == 'computation':
        from experiments import jmlr_computation_v1 as computation
        if not (root/'frozen_protocol.json').exists():
            releases.prepare_computation(root,_override(config,study))
        cfg = releases.load(root/'configuration.json')
        with computation.owner(root):
            if config['profile'] == 'smoke':
                # Explicit small-support execution checks do not freeze a
                # substitute four-case resource policy for the paper panel.
                inputs=sorted((root/'inputs/primary').glob('*.json.gz'))
                for path in inputs[:limit]:
                    for kind in ('primary','capacity'):
                        record=computation.launch_case(root,cfg,path,kind,600)
                        if record['status']!='completed':
                            raise RuntimeError('Computation smoke failed; its original status record is retained')
                return {'status':'smoke_only','root':str(root)}
            if not (root/'execution.json').exists():
                computation.run_panel(root,cfg,preflight=True)
                computation.freeze_resources(root,cfg)
            computation.run_panel(root,cfg,limit=limit)
        return releases.load(root/'manifest.json')
    raise ValueError('Unknown study')


def _partial_report(root, study):
    """Descriptive rows/statuses only; no incomplete-grid inferential summary."""
    if not (root/'frozen_protocol.json').exists():
        raise FileNotFoundError('Prepare/run the study before reporting it')
    folders = ['panel/blocks'] if study in ('release','precision') else ['blocks']
    paths = [p for folder in folders for p in sorted((root/folder).glob('*.json.gz'))]
    rows = []
    for path in paths:
        block = releases.load(path)
        for key in ('rows','selection_rows','analysis_rows','evaluations'):
            for row in block.get(key,[]):
                rows.append(dict(record_type=key,evidence_file=str(path.relative_to(root)),**row))
    tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for row in rows for k in row})
    with (tables/'partial_rows.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(releases.exact.encode(v),sort_keys=True)
                             if isinstance(v,(dict,list,tuple)) else str(v)
                             for k,v in row.items()})
    summary={'study':study,'status':'incomplete','blocks':len(paths),'rows':len(rows),
             'scope':'Bounded execution check only; no full-panel scientific conclusions or uncertainty intervals'}
    releases.dump(root/'partial_summary.json',summary)
    return summary


def _report(study,workspace,config):
    workspace=Path(workspace).resolve();root=workspace/study
    if study=='assignment':return assignment.report(workspace)
    if study in ('efficiency','recovery'):return analyses.report(workspace,study)
    if study=='certification':return certification.report(
        workspace,require_complete=config['profile']=='full-paper')
    if study=='relevance':return relevance.report(
        workspace,require_complete=config['profile']=='full-paper')
    if study=='computation':
        from experiments.jmlr_computation_v1_reporting import generate
        return generate(root,figures=False,require_complete=config['profile']=='full-paper')
    if config['profile']=='smoke':return _partial_report(root,study)
    if study=='release':
        from experiments.jmlr_release_capacity_v1_reporting import report
        return report(root)
    if study=='precision':
        from experiments.jmlr_precision_frontier_v1_reporting import generate
        return generate(root,figures=False)
    if study=='outcome':
        from experiments.jmlr_outcome_inference_v1_reporting import generate_reports
        return generate_reports(root)
    raise ValueError('Unknown study')


def smoke(workspace,config,data,rhc_data='data/rhc.csv'):
    if config['profile']!='smoke':raise ValueError('Smoke requires configs/smoke.json')
    _guard_workspace(workspace,config)
    results={}
    for study in CURRENT_STUDIES:
        # All dependent release blocks are small but include the complete
        # IHDP finite support. Original assignment K16 is checked as a prefix.
        limit=1 if study in ('assignment','certification','relevance','recovery','computation') else (2 if study=='efficiency' else None)
        results[study]=_run(study,workspace,config,data,limit,rhc_data)
        results[study+'_tables']=_report(study,workspace,config)
    receipt={'status':'smoke_complete','scientific_scope':'Execution checks only; full paper studies were not run',
             'studies':results,'plotting':'not run; optional paper plots require complete reported panels'}
    releases.dump(Path(workspace)/'smoke_receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('example',help='Run the principal full-sample SCA example')
    commands.add_parser('example-bounded',help='Run bounded-outcome export and reconstruction')
    commands.add_parser('example-drift',help='Run mass-drift export and admission checks')
    data=commands.add_parser('prepare-data',help='Download/check public IHDP and RHC inputs (untracked)')
    data.add_argument('--directory',default='data')
    data.add_argument('--dataset',choices=('all','ihdp','rhc'),default='all')
    plots=commands.add_parser('plot',help='Render current empirical figures or the retained computation diagnostic')
    plots.add_argument('study',choices=('efficiency','split','release','precision','computation','all'))
    plots.add_argument('--workspace',default='outputs/paper')
    plots.add_argument('--output-dir',default=None)
    tables=commands.add_parser('paper-tables',help='Assemble the current main-paper tables from complete reported outputs')
    tables.add_argument('--workspace',default='outputs/paper')
    tables.add_argument('--output-dir',default=None)
    for name in ('run','report','smoke'):
        command=commands.add_parser(name)
        if name!='smoke':command.add_argument('study',choices=(*STUDIES,'all'))
        command.add_argument('--workspace',default='outputs/smoke' if name=='smoke' else 'outputs/paper')
        command.add_argument('--config',default='configs/smoke.json' if name=='smoke' else 'configs/paper.json')
        command.add_argument('--data',default='data/ihdp_npci_1.csv')
        command.add_argument('--rhc-data',default='data/rhc.csv')
        if name=='run':command.add_argument('--limit',type=int,help='Explicit incomplete prefix; never a full reproduction')
    args=parser.parse_args()
    if args.command=='example':
        runpy.run_path(str(PROJECT/'examples/synthetic.py'),run_name='__main__');return
    if args.command=='example-bounded':
        runpy.run_path(str(PROJECT/'examples/bounded_outcomes.py'),run_name='__main__');return
    if args.command=='example-drift':
        runpy.run_path(str(PROJECT/'examples/mass_drift.py'),run_name='__main__');return
    if args.command=='prepare-data':
        from .data import prepare_data
        result=prepare_data(args.directory,args.dataset)
    elif args.command=='plot':
        from .plots import plot
        result=plot(args.study,args.workspace,output_dir=args.output_dir)
    elif args.command=='paper-tables':
        from .paper_tables import generate
        result=generate(args.workspace,output_dir=args.output_dir)
    else:
        config=_configuration(args.config)
        if args.command=='smoke':result=smoke(args.workspace,config,args.data,args.rhc_data)
        else:
            _guard_workspace(args.workspace,config,report_only=args.command=='report')
            result={}
            for study in (CURRENT_STUDIES if args.study=='all' else (args.study,)):
                if args.command=='run':result[study]=_run(
                    study,args.workspace,config,args.data,args.limit,args.rhc_data)
                else:result[study]=_report(study,args.workspace,config)
    print(json.dumps(releases.exact.encode(result),indent=2,default=str))


if __name__=='__main__':main()
