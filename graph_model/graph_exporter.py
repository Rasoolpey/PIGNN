"""
Graph to H5 Exporter

This module bridges the existing PowerGrid graph representation
to the H5 file format, extracting and organizing data from the
graph model for storage.

Author: PIGNN Project
Date: 2025-10-19
Version: 2.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_base import PowerGridGraph, PhaseType
from core.node_types import Generator, Load, Bus
from core.edge_types import TransmissionLine, Transformer
from graph_model.h5_writer import (
    PowerGridH5Writer,
    create_default_generator_parameters,
    create_default_exciter_parameters,
    create_default_governor_parameters
)

logger = logging.getLogger(__name__)


class GraphToH5Exporter:
    """
    Export PowerGridGraph graph model to H5 format.
    
    This class extracts data from the graph representation and
    organizes it according to the H5 format specification.
    """
    
    def __init__(self, graph: PowerGridGraph):
        """
        Initialize exporter with a PowerGridGraph graph.
        
        Args:
            graph: PowerGridGraph graph instance
        """
        self.graph = graph
        self.phase_names = ['phase_a', 'phase_b', 'phase_c']
    
    def export(self, 
               filepath: str,
               include_dynamics: bool = False,
               use_default_dynamics: bool = True):
        """
        Export graph to H5 file.
        
        Args:
            filepath: Output H5 file path
            include_dynamics: Whether to include dynamic RMS parameters
            use_default_dynamics: Use default parameters if actual data not available
        """
        logger.info(f"Exporting graph to {filepath}...")
        
        with PowerGridH5Writer(filepath, mode='w') as writer:
            # 1. Metadata
            self._write_metadata(writer)
            
            # 2. Topology
            self._write_topology(writer)
            
            # 3. Per-phase data
            self._write_all_phases(writer)
            
            # 4. Coupling matrices (if available)
            self._write_coupling(writer)
            
            # 5. Steady-state results
            self._write_steady_state(writer)
            
            # 6. Dynamic models (optional)
            if include_dynamics:
                self._write_dynamics(writer, use_defaults=use_default_dynamics)
            
            # 7. Scenarios (if multiple scenarios exist)
            self._write_scenarios(writer)
        
        logger.info(f"✓ Export complete: {filepath}")
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def _write_metadata(self, writer: PowerGridH5Writer):
        """Extract and write metadata from graph."""
        # Get system info from graph properties
        properties = getattr(self.graph, 'properties', {})
        
        grid_name = properties.get('name', 'PowerGridGraph')
        base_mva = getattr(self.graph, 'base_mva', 100.0)
        base_freq = getattr(self.graph, 'frequency_hz', 60.0)
        
        # Count unique buses (nodes are organized as node_id -> {phase -> Node})
        num_buses = len(self.graph.nodes)
        
        writer.write_metadata(
            grid_name=grid_name,
            base_mva=base_mva,
            base_frequency_hz=base_freq,
            num_buses=num_buses,
            num_phases=3,
            description=properties.get('description', '')
        )
    
    def _write_topology(self, writer: PowerGridH5Writer):
        """Extract and write topology."""
        # PowerGridGraph has edges organized as edge_id -> {phase -> Edge}
        # We'll use this structure to build the topology
        
        n_edges = len(self.graph.edges)
        from_bus = np.zeros(n_edges, dtype=np.int64)
        to_bus = np.zeros(n_edges, dtype=np.int64)
        edge_type = np.zeros(n_edges, dtype=np.int64)
        edge_names = []
        
        # Create node index mapping (node_id -> index)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(self.graph.nodes.keys())}
        
        # Extract topology from phase A (topology is same for all phases)
        from core.graph_base import PhaseType
        phase_a = PhaseType.PHASE_A
        
        for i, (edge_id, phase_edges) in enumerate(self.graph.edges.items()):
            edge_a = phase_edges[phase_a]  # Get phase A edge
            
            from_bus[i] = node_to_idx[edge_a.from_node_id]
            to_bus[i] = node_to_idx[edge_a.to_node_id]
            
            # Determine edge type based on edge_type attribute
            if edge_a.edge_type == 'line':
                edge_type[i] = 0
            elif edge_a.edge_type == 'transformer':
                edge_type[i] = 1
            else:
                edge_type[i] = 2  # other/switch
            
            edge_names.append(edge_id.encode())
        
        edge_list = {
            'from_bus': from_bus,
            'to_bus': to_bus,
            'edge_type': edge_type,
            'edge_names': edge_names
        }
        
        writer.write_topology(edge_list=edge_list)
    
    def _write_all_phases(self, writer: PowerGridH5Writer):
        """Extract and write data for all three phases."""
        from core.graph_base import PhaseType
        phases = [(PhaseType.PHASE_A, 'phase_a'), 
                  (PhaseType.PHASE_B, 'phase_b'),
                  (PhaseType.PHASE_C, 'phase_c')]
        
        for phase_enum, phase_name in phases:
            node_data, edge_data = self._extract_phase_data(phase_enum)
            writer.write_phase_data(phase_name, node_data, edge_data)
    
    def _extract_phase_data(self, phase: PhaseType) -> Tuple[Dict, Dict]:
        """
        Extract node and edge data for a single phase from PowerGridGraph.
        
        Args:
            phase: PhaseType enum (PHASE_A, PHASE_B, or PHASE_C)
            
        Returns:
            Tuple of (node_data_dict, edge_data_dict)
        """
        n_nodes = len(self.graph.nodes)
        n_edges = len(self.graph.edges)
        
        # Initialize node data arrays
        node_data = {
            'bus_ids': np.arange(n_nodes, dtype=np.int64),
            'bus_names': np.array([node_id.encode() for node_id in self.graph.nodes.keys()], dtype='S50'),
            'bus_types': np.zeros(n_nodes, dtype=np.int64),
            'base_voltages_kV': np.full(n_nodes, 138.0, dtype=np.float64),
            'voltages_pu': np.ones(n_nodes, dtype=np.float64),
            'angles_deg': np.zeros(n_nodes, dtype=np.float64),
            'P_injection_MW': np.zeros(n_nodes, dtype=np.float64),
            'Q_injection_MVAR': np.zeros(n_nodes, dtype=np.float64),
            'P_generation_MW': np.zeros(n_nodes, dtype=np.float64),
            'Q_generation_MVAR': np.zeros(n_nodes, dtype=np.float64),
            'P_load_MW': np.zeros(n_nodes, dtype=np.float64),
            'Q_load_MVAR': np.zeros(n_nodes, dtype=np.float64),
            'shunt_G_pu': np.zeros(n_nodes, dtype=np.float64),
            'shunt_B_pu': np.zeros(n_nodes, dtype=np.float64),
        }
        
        # Fill node data from graph structure
        for i, (node_id, phase_nodes) in enumerate(self.graph.nodes.items()):
            node = phase_nodes[phase]  # Get the specific phase node
            
            # Extract data from node properties
            if hasattr(node, 'properties'):
                props = node.properties
                node_data['bus_types'][i] = props.get('bus_type', 0)
                node_data['base_voltages_kV'][i] = props.get('base_voltage_kV', 138.0)
                node_data['voltages_pu'][i] = props.get('voltage_pu', 1.0)
                node_data['angles_deg'][i] = props.get('angle_deg', 0.0)
                node_data['P_injection_MW'][i] = props.get('P_injection_MW', 0.0)
                node_data['Q_injection_MVAR'][i] = props.get('Q_injection_MVAR', 0.0)
                node_data['P_generation_MW'][i] = props.get('P_generation_MW', 0.0)
                node_data['Q_generation_MVAR'][i] = props.get('Q_generation_MVAR', 0.0)
                node_data['P_load_MW'][i] = props.get('P_load_MW', 0.0)
                node_data['Q_load_MVAR'][i] = props.get('Q_load_MVAR', 0.0)
        
        # Initialize edge data arrays
        node_to_idx = {node_id: idx for idx, node_id in enumerate(self.graph.nodes.keys())}
        
        edge_data = {
            'from_bus': np.zeros(n_edges, dtype=np.int64),
            'to_bus': np.zeros(n_edges, dtype=np.int64),
            'element_id': np.array([edge_id.encode() for edge_id in self.graph.edges.keys()], dtype='S50'),
            'element_type': np.zeros(n_edges, dtype=np.int64),
            'R_pu': np.zeros(n_edges, dtype=np.float64),
            'X_pu': np.zeros(n_edges, dtype=np.float64),
            'B_shunt_pu': np.zeros(n_edges, dtype=np.float64),
            'rating_MVA': np.zeros(n_edges, dtype=np.float64),
            'length_km': np.zeros(n_edges, dtype=np.float64),
            'in_service': np.ones(n_edges, dtype=bool),
        }
        
        # Fill edge data from graph structure  
        for i, (edge_id, phase_edges) in enumerate(self.graph.edges.items()):
            edge = phase_edges[phase]  # Get the specific phase edge
            
            edge_data['from_bus'][i] = node_to_idx[edge.from_node_id]
            edge_data['to_bus'][i] = node_to_idx[edge.to_node_id]
            
            # Extract edge properties
            if hasattr(edge, 'properties'):
                props = edge.properties
                if edge.edge_type == 'line':
                    edge_data['element_type'][i] = 0
                elif edge.edge_type == 'transformer':
                    edge_data['element_type'][i] = 1
                else:
                    edge_data['element_type'][i] = 2
                
                edge_data['R_pu'][i] = props.get('R_pu', 0.0)
                edge_data['X_pu'][i] = props.get('X_pu', 0.0)
                edge_data['B_shunt_pu'][i] = props.get('B_pu', 0.0)
                edge_data['rating_MVA'][i] = props.get('rating_MVA', 0.0)
                edge_data['length_km'][i] = props.get('length_km', 0.0)
        
        return node_data, edge_data
    
    def _write_coupling(self, writer: PowerGridH5Writer):
        """
        Write three-phase coupling matrices if available.
        
        Note: This requires coupling data to be stored in the graph model.
        For now, we'll write placeholder or skip if not available.
        """
        # Check if graph has coupling data
        if not hasattr(self.graph, 'coupling_data'):
            logger.info("No coupling data available - skipping")
            return
        
        coupling_data = self.graph.coupling_data
        
        # Lines
        if 'lines' in coupling_data:
            line_data = coupling_data['lines']
            writer.write_line_coupling(
                line_names=line_data.get('names', []),
                Z_matrix_abc=line_data.get('Z_matrix_abc'),
                Y_shunt_abc=line_data.get('Y_shunt_abc'),
                Z0_ohm=line_data.get('Z0_ohm'),
                Z1_ohm=line_data.get('Z1_ohm'),
                Z2_ohm=line_data.get('Z2_ohm')
            )
        
        # Transformers
        if 'transformers' in coupling_data:
            trafo_data = coupling_data['transformers']
            writer.write_transformer_coupling(
                transformer_names=trafo_data.get('names', []),
                winding_config=trafo_data.get('winding_config', []),
                vector_group=trafo_data.get('vector_group'),
                phase_shift_deg=trafo_data.get('phase_shift_deg'),
                Z_matrix_primary=trafo_data.get('Z_matrix_primary'),
                Z_matrix_secondary=trafo_data.get('Z_matrix_secondary'),
                connection_matrix=trafo_data.get('connection_matrix')
            )
    
    def _write_steady_state(self, writer: PowerGridH5Writer):
        """Write steady-state power flow results."""
        # Check if power flow has been solved
        if not hasattr(self.graph, 'power_flow_results'):
            logger.warning("No power flow results available")
            return
        
        results = self.graph.power_flow_results
        
        writer.write_power_flow_results(
            converged=results.get('converged', False),
            iterations=results.get('iterations', 0),
            max_mismatch=results.get('max_mismatch', 0.0),
            total_generation_MW=results.get('total_generation_MW', 0.0),
            total_load_MW=results.get('total_load_MW', 0.0),
            total_losses_MW=results.get('total_losses_MW', 0.0),
            max_voltage_pu=results.get('max_voltage_pu', 1.0),
            min_voltage_pu=results.get('min_voltage_pu', 1.0)
        )
        
        # Write Y-matrices if available
        if hasattr(self.graph, 'Y_matrix'):
            writer.write_admittance_matrix(
                Y_single_phase=self.graph.Y_matrix.get('single_phase'),
                Y_three_phase=self.graph.Y_matrix.get('three_phase')
            )
    
    def _write_dynamics(self, writer: PowerGridH5Writer, use_defaults: bool = True):
        """
        Write dynamic RMS parameters.
        
        Args:
            use_defaults: If True, use default parameters when actual data not available
        """
        logger.info("Writing dynamic parameters...")
        
        # Get all generators from phase A (same generators on all phases)
        graph_a = self.graph.get_phase_graph('a')
        generators = [node_obj for _, node_attr in graph_a.nodes(data=True)
                     if (node_obj := node_attr.get('node_object')) and isinstance(node_obj, Generator)]
        
        if len(generators) == 0:
            logger.warning("No generators found - skipping dynamics")
            return
        
        n_gen = len(generators)
        
        # Extract generator info
        gen_names = [gen.name for gen in generators]
        gen_buses = [getattr(gen, 'bus_name', f'bus_{i}') for i, gen in enumerate(generators)]
        gen_phases = ['abc' for _ in generators]  # Assume three-phase
        
        # Check if generators have dynamic parameters
        has_dynamics = any(hasattr(gen, 'H_s') for gen in generators)
        
        if not has_dynamics and not use_defaults:
            logger.warning("No dynamic parameters in generators and use_defaults=False")
            return
        
        # Build parameters dictionary
        if has_dynamics:
            # Extract from generators
            gen_params = self._extract_generator_parameters(generators)
            model_types = [getattr(gen, 'model_type', 'GENCLS') for gen in generators]
        else:
            # Use defaults
            logger.info("Using default dynamic parameters")
            gen_params = create_default_generator_parameters(n_gen)
            model_types = ['GENCLS'] * n_gen
            
            # Add ratings from generators
            gen_params['S_rated_MVA'] = np.array([getattr(gen, 'S_rated_MVA', 100.0) 
                                                  for gen in generators])
            gen_params['V_rated_kV'] = np.array([getattr(gen, 'V_rated_kV', 138.0) 
                                                 for gen in generators])
        
        writer.write_generator_dynamics(
            names=gen_names,
            buses=gen_buses,
            phases=gen_phases,
            model_type=model_types,
            parameters=gen_params
        )
        
        # Exciters (use defaults for now)
        if use_defaults:
            exc_params = create_default_exciter_parameters(n_gen, 'SEXS')
            writer.write_exciter_models(
                names=[f'{name}_EXC' for name in gen_names],
                generator_names=gen_names,
                model_type=['SEXS'] * n_gen,
                parameters=exc_params
            )
        
        # Governors (use defaults for now)
        if use_defaults:
            gov_params = create_default_governor_parameters(n_gen, 'TGOV1')
            writer.write_governor_models(
                names=[f'{name}_GOV' for name in gen_names],
                generator_names=gen_names,
                model_type=['TGOV1'] * n_gen,
                parameters=gov_params
            )
        
        logger.info(f"✓ Dynamic parameters written for {n_gen} generators")
    
    def _extract_generator_parameters(self, generators: List[Generator]) -> Dict[str, np.ndarray]:
        """Extract dynamic parameters from generator objects."""
        n_gen = len(generators)
        
        params = {}
        
        # List of all possible parameters
        param_names = [
            'H_s', 'D_pu',
            'xd_pu', 'xq_pu', 'xd_prime_pu', 'xq_prime_pu',
            'xd_double_prime_pu', 'xq_double_prime_pu', 'xl_pu', 'ra_pu',
            'Td0_prime_s', 'Tq0_prime_s', 'Td0_double_prime_s', 'Tq0_double_prime_s',
            'S10', 'S12',
            'S_rated_MVA', 'V_rated_kV'
        ]
        
        # Extract each parameter
        for param in param_names:
            values = np.array([getattr(gen, param, np.nan) for gen in generators])
            
            # Check if any valid values exist
            if not np.all(np.isnan(values)):
                params[param] = values
        
        return params
    
    def _write_scenarios(self, writer: PowerGridH5Writer):
        """Write multiple scenarios if they exist."""
        if not hasattr(self.graph, 'scenarios') or len(self.graph.scenarios) == 0:
            return
        
        logger.info(f"Writing {len(self.graph.scenarios)} scenarios...")
        
        for scenario_id, scenario_data in self.graph.scenarios.items():
            writer.write_scenario(
                scenario_id=scenario_id,
                description=scenario_data.get('description', ''),
                voltages_pu=scenario_data['voltages_pu'],
                angles_deg=scenario_data['angles_deg'],
                P_injections_MW=scenario_data['P_injections_MW'],
                Q_injections_MVAR=scenario_data['Q_injections_MVAR'],
                contingency_description=scenario_data.get('contingency_description', ''),
                power_flow_converged=scenario_data.get('converged', True)
            )


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def export_graph_to_h5(graph: PowerGridGraph, 
                       filepath: str,
                       include_dynamics: bool = False,
                       use_default_dynamics: bool = True):
    """
    Convenience function to export a PowerGridGraph to H5.
    
    Args:
        graph: PowerGridGraph instance
        filepath: Output file path
        include_dynamics: Include RMS dynamic parameters
        use_default_dynamics: Use defaults if actual data not available
    """
    exporter = GraphToH5Exporter(graph)
    exporter.export(filepath, include_dynamics, use_default_dynamics)
    logger.info(f"Graph exported to {filepath}")
