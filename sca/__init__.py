"""Sensitivity-calibrated aggregation (SCA).

The public procedure always uses the full supplied assignment sample, Coupled
Clopper–Pearson certification, and the complete combined search family.
Baselines and ablations live in ``sca_reproduction`` rather than this API.
"""
from experiments.jmlr_inference_extensions_v1 import (
    ConfidenceAllocation,
    confidence_allocation,
)
from .api import SCA, SCAResult, analyze_outcome_release, analyze_release

__all__ = ['SCA', 'SCAResult', 'ConfidenceAllocation', 'confidence_allocation',
           'analyze_release', 'analyze_outcome_release']
