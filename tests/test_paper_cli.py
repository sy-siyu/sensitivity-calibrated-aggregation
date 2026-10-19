"""Dispatch checks only: plotting/table commands must not launch experiments."""
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from sca_reproduction import __main__ as cli


class PaperCommands(unittest.TestCase):
    def test_plot_reads_reported_workspace_without_experiment_dispatch(self):
        with patch.object(sys,'argv',['sca-reproduce','plot','release','--workspace','outputs/example']), \
             patch('sca_reproduction.plots.plot',return_value={'status':'complete'}) as plot, \
             patch.object(cli,'_run',side_effect=AssertionError('Must not run experiments')), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        plot.assert_called_once_with('release','outputs/example',output_dir=None)

    def test_paper_tables_reads_reported_workspace_without_experiment_dispatch(self):
        with patch.object(sys,'argv',['sca-reproduce','paper-tables','--workspace','outputs/example']), \
             patch('sca_reproduction.paper_tables.generate',return_value={'status':'complete'}) as tables, \
             patch.object(cli,'_run',side_effect=AssertionError('Must not run experiments')), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        tables.assert_called_once_with('outputs/example',output_dir=None)


if __name__=='__main__':unittest.main()
