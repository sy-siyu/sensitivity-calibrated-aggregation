import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sca_reproduction import __main__ as cli


class PublicOrchestration(unittest.TestCase):
    def test_frozen_workspace_rejects_profile_or_template_change(self):
        smoke=cli._configuration(cli.PROJECT/'configs/smoke.json')
        paper=cli._configuration(cli.PROJECT/'configs/paper.json')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'work'
            cli._guard_workspace(root,smoke)
            cli._guard_workspace(root,smoke)
            with self.assertRaises(ValueError):cli._guard_workspace(root,paper)
            altered=dict(smoke,description='different committed invocation')
            with self.assertRaises(ValueError):cli._guard_workspace(root,altered)

    def test_report_does_not_create_unprepared_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'release'
            with self.assertRaises(FileNotFoundError):cli._partial_report(root,'release')
            self.assertFalse(root.exists())

    def test_computation_smoke_propagates_worker_failure(self):
        from experiments import jmlr_computation_v1 as computation
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'computation'
            (root/'inputs/primary').mkdir(parents=True)
            (root/'inputs/primary/example.json.gz').touch()
            (root/'frozen_protocol.json').write_text('{}')
            (root/'configuration.json').write_text('{}')
            with patch.object(computation,'launch_case',return_value={'status':'timeout'}):
                with self.assertRaisesRegex(RuntimeError,'smoke failed'):
                    cli._run('computation',directory,{'profile':'smoke'},'unused',1)


if __name__=='__main__':unittest.main()
