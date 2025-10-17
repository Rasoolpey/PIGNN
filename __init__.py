"""
Three-Phase Power Grid Graph Package

A modular, physics-informed graph structure for representing unbalanced 
three-phase power systems with inter-phase coupling.

Key Features:
- Three separate phase graphs with coupling matrices
- Physics-based coupling calculations (Carson's equations)
- Component-specific models (generators, loads, lines, transformers)
- H5 data integration from PowerFactory simulations
- Comprehensive validation framework
- 3D visualization with coupling representation

Example usage:
    from power_grid_graph.data.h5_loader import H5DataLoader
    from power_grid_graph.data.graph_builder import GraphBuilder
    from power_grid_graph.visualization.graph_plotter import ThreePhaseGraphPlotter
    
    # Load data and build graph
    loader = H5DataLoader("scenario_0.h5")
    data = loader.load_all_data()
    
    builder = GraphBuilder(base_mva=100.0)
    graph = builder.build_from_h5_data(data)
    
    # Visualize
    plotter = ThreePhaseGraphPlotter(graph)
    fig, ax = plotter.plot_3d_coupled_graph()
"""

__version__ = "1.0.0"
__author__ = "Power Systems Research"

# Core components
from .core.graph_base import PowerGridGraph, PhaseType, Node, Edge, CouplingMatrix
from .core.node_types import Generator, Load, Bus
from .core.edge_types import TransmissionLine, Transformer

# Data loading and building
from .data.h5_loader import H5DataLoader
from .data.graph_builder import GraphBuilder

# Physics modules
from .physics.coupling_models import CouplingCalculator
from .physics.impedance_matrix import AdmittanceMatrixBuilder
from .physics.load_flow_solver import ThreePhaseLoadFlowSolver, LoadFlowResults

# Utilities
from .utils.validators import GraphValidator

# Visualization
from .visualization.graph_plotter import ThreePhaseGraphPlotter

__all__ = [
    # Core
    'PowerGridGraph', 'PhaseType', 'Node', 'Edge', 'CouplingMatrix',
    'Generator', 'Load', 'Bus',
    'TransmissionLine', 'Transformer',
    
    # Data
    'H5DataLoader', 'GraphBuilder',
    
    # Physics
    'CouplingCalculator', 'AdmittanceMatrixBuilder',
    
    # Utils
    'GraphValidator',
    
    # Visualization
    'ThreePhaseGraphPlotter'
]