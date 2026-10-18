"""Sensitivity-calibrated aggregation (SCA).

The public procedure always uses the full supplied assignment sample, Coupled
Clopper–Pearson certification, and the complete combined search family.
Baselines and ablations live in ``sca_reproduction`` rather than this API.
"""
from .api import SCA, SCAResult, analyze_release

__all__ = ['SCA', 'SCAResult', 'analyze_release']
