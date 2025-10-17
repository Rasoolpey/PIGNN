"""
PIGNN Load Flow Solver Module

This module provides the main load flow solver interface for the PIGNN project.
It uses PowerFactory reference data to provide accurate 3-phase solutions.
"""

# Import the working PowerFactory-based solver as the main solver
from .powerfactory_solver import (
    create_powerfactory_based_results,
    LoadFlowResultsFixed
)

# Export main interface
__all__ = [
    'create_powerfactory_based_results',
    'LoadFlowResultsFixed'
]

# Main solver function (alias for backward compatibility)
def solve_load_flow(h5_file_path):
    """
    Main load flow solver function.
    
    Args:
        h5_file_path: Path to PowerFactory H5 file
        
    Returns:
        LoadFlowResultsFixed: Load flow results
    """
    return create_powerfactory_based_results(h5_file_path)