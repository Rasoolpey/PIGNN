"""
Comparison utilities package for validating solver results against DIgSILENT PowerFactory outputs.

This package provides:
- DIgSILENT CSV/Excel export parsers
- Error metric calculation (voltage, flow, losses)
- Batch comparison utilities for contingency analysis
- Report generation for validation studies
"""

from .digsilent_parser import DIgSILENTParser
from .error_metrics import ErrorCalculator, ComparisonResults
from .batch_comparator import BatchComparator
from .report_generator import ComparisonReportGenerator

__all__ = [
    'DIgSILENTParser',
    'ErrorCalculator',
    'ComparisonResults', 
    'BatchComparator',
    'ComparisonReportGenerator'
]