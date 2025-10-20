"""
Build three-phase power grid graph from H5 data.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_base import PowerGridGraph, PhaseType
from core.node_types import Generator, Load, Bus
from core.edge_types import TransmissionLine, Transformer
from physics.coupling_models import CouplingCalculator


class GraphBuilder:
    """Build power grid graph from H5 data"""
    
    def __init__(self, base_mva: float = 100.0, frequency_hz: float = 60.0):
        """
        Initialize graph builder.
        
        Args:
            base_mva: Base power for per-unit calculations
            frequency_hz: System frequency
        """
        self.base_mva = base_mva
        self.frequency_hz = frequency_hz
        self.coupling_calc = CouplingCalculator()
    
    def build_from_h5_data(self, data: Dict[str, Any]) -> PowerGridGraph:
        """
        Build complete three-phase graph from H5 data.
        
        Args:
            data: Dictionary from H5DataLoader
        
        Returns:
            PowerGridGraph object
        """
        graph = PowerGridGraph()
        graph.base_mva = self.base_mva
        graph.frequency_hz = self.frequency_hz
        
        # Build nodes
        self._add_buses(graph, data.get('buses', {}))
        self._add_generators(graph, data.get('generators', {}), data.get('buses', {}))
        self._add_loads(graph, data.get('loads', {}), data.get('buses', {}))
        
        # Set power injections for load flow solver (must be after generators and loads)
        self._set_power_injections(graph)
        
        # Build edges
        self._add_lines(graph, data.get('lines', {}))
        self._add_transformers(graph, data.get('transformers', {}))
        
        # Add shunts to buses
        self._add_shunts(graph, data.get('shunts', {}))
        
        return graph
    
    def _add_buses(self, graph: PowerGridGraph, bus_data: Dict):
        """Add all buses to the graph"""
        if not bus_data or 'names' not in bus_data:
            return
        
        for i, bus_name in enumerate(bus_data['names']):
            # Skip if already added (generator or load bus)
            if bus_name in graph.nodes:
                continue
            
            # Add bus
            graph.add_node(bus_name, 'bus')
            
            # Set electrical properties for each phase
            voltage_pu = bus_data['voltages_pu'][i]
            angle_rad = np.deg2rad(bus_data['voltage_angles_deg'][i])
            base_kv = bus_data['base_voltages_kV'][i]
            
            # For balanced system, assume same voltage for all phases
            # with 120 degree phase shift
            for phase in PhaseType:
                node = graph.get_node(bus_name, phase)
                phase_shift = 0
                if phase == PhaseType.B:
                    phase_shift = -2 * np.pi / 3
                elif phase == PhaseType.C:
                    phase_shift = 2 * np.pi / 3
                
                node.voltage_pu = voltage_pu * np.exp(1j * (angle_rad + phase_shift))
                node.voltage_base_kv = base_kv
    
    def _add_generators(self, graph: PowerGridGraph, gen_data: Dict, bus_data: Dict):
        """Add generators to the graph"""
        if not gen_data or 'names' not in gen_data:
            return
        
        bus_name_to_idx = {name: i for i, name in enumerate(bus_data.get('names', []))}
        
        for i, gen_name in enumerate(gen_data['names']):
            bus_name = gen_data['buses'][i]
            
            # Add generator node if bus doesn't exist
            if bus_name not in graph.nodes:
                graph.add_node(bus_name, 'generator')
            
            # Get bus data
            bus_idx = bus_name_to_idx.get(bus_name)
            if bus_idx is not None:
                voltage_pu = bus_data['voltages_pu'][bus_idx]
                angle_rad = np.deg2rad(bus_data['voltage_angles_deg'][bus_idx])
                base_kv = bus_data['base_voltages_kV'][bus_idx]
            else:
                voltage_pu = 1.0
                angle_rad = 0.0
                base_kv = gen_data.get('V_rated_kV', [138.0])[i]
            
            # Set generator properties for each phase
            P_total = gen_data['active_power_MW'][i]
            Q_total = gen_data['reactive_power_MVAR'][i]
            
            for phase in PhaseType:
                node = graph.get_node(bus_name, phase)
                
                # Convert node to generator if it was a bus
                if node.node_type != 'generator':
                    gen_node = Generator(
                        id=node.id,
                        parent_id=node.parent_id,
                        phase=node.phase
                    )
                    graph.nodes[bus_name][phase] = gen_node
                    node = gen_node
                
                # Set voltage with phase shift
                phase_shift = 0
                if phase == PhaseType.B:
                    phase_shift = -2 * np.pi / 3
                elif phase == PhaseType.C:
                    phase_shift = 2 * np.pi / 3
                
                node.voltage_pu = voltage_pu * np.exp(1j * (angle_rad + phase_shift))
                node.voltage_base_kv = base_kv
                
                # Divide power equally among phases (for balanced system)
                node.P_nominal_MW = P_total / 3
                node.P_actual_MW = P_total / 3
                node.Q_actual_MVAR = Q_total / 3
                
                # Set machine parameters if available
                if i < len(gen_data.get('xd_prime_pu', [])):
                    node.xd_prime_pu = gen_data['xd_prime_pu'][i]
                if i < len(gen_data.get('H_s', [])):
                    node.H_s = gen_data['H_s'][i]
                if i < len(gen_data.get('D_pu', [])):
                    node.D_pu = gen_data['D_pu'][i]
                
                # Set limits
                if i < len(gen_data.get('reactive_limits_min_MVAR', [])):
                    node.Q_min_MVAR = gen_data['reactive_limits_min_MVAR'][i] / 3
                if i < len(gen_data.get('reactive_limits_max_MVAR', [])):
                    node.Q_max_MVAR = gen_data['reactive_limits_max_MVAR'][i] / 3
                
                # Store generator object reference
                node.properties['node_object'] = node
            
            # Set generator coupling matrix
            if i < len(gen_data.get('S_rated_MVA', [])):
                S_rated = gen_data['S_rated_MVA'][i]
                V_rated = gen_data.get('V_rated_kV', [base_kv])[i]
                Z_base = (V_rated ** 2) / S_rated
                
                coupling_matrix = self.coupling_calc.calculate_generator_coupling(
                    xd=node.xd_pu,
                    xq=node.xq_pu,
                    ra=node.ra_pu,
                    base_impedance=Z_base,
                    frequency_hz=self.frequency_hz
                )
                graph.set_node_coupling(bus_name, coupling_matrix)
    
    def _add_loads(self, graph: PowerGridGraph, load_data: Dict, bus_data: Dict):
        """Add loads to the graph"""
        if not load_data or 'names' not in load_data:
            return
        
        bus_name_to_idx = {name: i for i, name in enumerate(bus_data.get('names', []))}
        
        for i, load_name in enumerate(load_data['names']):
            bus_name = load_data['buses'][i]
            
            # Check if bus exists, if not create it
            if bus_name not in graph.nodes:
                graph.add_node(bus_name, 'load')
            
            # Get bus data
            bus_idx = bus_name_to_idx.get(bus_name)
            if bus_idx is not None:
                voltage_pu = bus_data['voltages_pu'][bus_idx]
                angle_rad = np.deg2rad(bus_data['voltage_angles_deg'][bus_idx])
                base_kv = bus_data['base_voltages_kV'][bus_idx]
            else:
                voltage_pu = 1.0
                angle_rad = 0.0
                base_kv = 138.0  # Default
            
            # Set load properties for each phase
            P_total = load_data['active_power_MW'][i]
            Q_total = load_data['reactive_power_MVAR'][i]
            
            for phase in PhaseType:
                node = graph.get_node(bus_name, phase)
                
                # Convert to load node if needed
                if node.node_type == 'bus':
                    load_node = Load(
                        id=node.id,
                        parent_id=node.parent_id,
                        phase=node.phase
                    )
                    graph.nodes[bus_name][phase] = load_node
                    node = load_node
                
                # Set voltage
                phase_shift = 0
                if phase == PhaseType.B:
                    phase_shift = -2 * np.pi / 3
                elif phase == PhaseType.C:
                    phase_shift = 2 * np.pi / 3
                
                node.voltage_pu = voltage_pu * np.exp(1j * (angle_rad + phase_shift))
                node.voltage_base_kv = base_kv
                
                # Set load (divided equally among phases)
                if hasattr(node, 'P_MW'):
                    node.P_MW += P_total / 3
                    node.Q_MVAR += Q_total / 3
                else:
                    # If node is generator with load
                    node.properties['load_P_MW'] = P_total / 3
                    node.properties['load_Q_MVAR'] = Q_total / 3
                
                # Store load object reference
                node.properties['load_object'] = node
    
    def _add_lines(self, graph: PowerGridGraph, line_data: Dict):
        """Add transmission lines to the graph"""
        if not line_data or 'names' not in line_data:
            return
        
        for i, line_name in enumerate(line_data['names']):
            from_bus = line_data['from_buses'][i]
            to_bus = line_data['to_buses'][i]
            
            # Check if buses exist
            if from_bus not in graph.nodes or to_bus not in graph.nodes:
                continue
            
            # Create edge
            edge_id = f"Line_{from_bus}_{to_bus}_{i}"
            graph.add_edge(edge_id, from_bus, to_bus, 'line')
            
            # Set line parameters for each phase
            R_total = line_data['R_ohm'][i]
            X_total = line_data['X_ohm'][i]
            B_total = line_data['B_uS'][i]
            
            for phase in PhaseType:
                edge = graph.get_edge(edge_id, phase)
                edge.R_ohm = R_total
                edge.X_ohm = X_total
                edge.B_total_uS = B_total
                
                # Convert to per-unit for load flow solver
                # Assume 345 kV base for impedance conversion
                V_base_kV = 345.0
                Z_base_ohm = (V_base_kV ** 2) / self.base_mva
                edge.properties['R_pu'] = R_total / Z_base_ohm
                edge.properties['X_pu'] = X_total / Z_base_ohm
                edge.properties['B_pu'] = B_total * 1e-6 * Z_base_ohm  # Convert μS to pu
                
                if i < len(line_data.get('rating_MVA', [])):
                    edge.rating_MVA = line_data['rating_MVA'][i]
                    edge.properties['rating_MVA'] = line_data['rating_MVA'][i]
                if i < len(line_data.get('length_km', [])):
                    edge.length_km = line_data['length_km'][i]
            
            # Set coupling matrix
            coupling_matrix = self.coupling_calc.estimate_line_coupling_from_impedance(
                R_total=R_total,
                X_total=X_total,
                mutual_factor=0.3  # Typical for overhead lines
            )
            graph.set_edge_coupling(edge_id, coupling_matrix)
    
    def _add_transformers(self, graph: PowerGridGraph, trafo_data: Dict):
        """Add transformers to the graph"""
        if not trafo_data or 'names' not in trafo_data:
            return
        
        for i, trafo_name in enumerate(trafo_data['names']):
            from_bus = trafo_data['from_buses'][i]
            to_bus = trafo_data['to_buses'][i]
            
            # Check if buses exist
            if from_bus not in graph.nodes or to_bus not in graph.nodes:
                continue
            
            # Create edge
            edge_id = f"Trafo_{from_bus}_{to_bus}_{i}"
            graph.add_edge(edge_id, from_bus, to_bus, 'transformer')
            
            # Set transformer parameters
            for phase in PhaseType:
                edge = graph.get_edge(edge_id, phase)
                
                # Set impedances
                edge.R_ohm = trafo_data['R_ohm'][i]
                edge.X_ohm = trafo_data['X_ohm'][i]
                
                # Convert to per-unit for load flow solver
                V_base_kV = 345.0  # Typical for high voltage side
                Z_base_ohm = (V_base_kV ** 2) / self.base_mva
                edge.properties['R_pu'] = trafo_data['R_ohm'][i] / Z_base_ohm
                edge.properties['X_pu'] = trafo_data['X_ohm'][i] / Z_base_ohm
                edge.properties['B_pu'] = 0.0  # Transformers typically have negligible shunt admittance
                
                # Set ratings
                edge.rating_MVA = trafo_data['rating_MVA'][i]
                edge.properties['rating_MVA'] = trafo_data['rating_MVA'][i]
                edge.V_primary_kV = trafo_data['V_primary_kV'][i]
                edge.V_secondary_kV = trafo_data['V_secondary_kV'][i]
                
                # Set tap
                if i < len(trafo_data.get('tap_ratio', [])):
                    edge.tap_ratio = trafo_data['tap_ratio'][i]
                
                # Set winding configuration
                if i < len(trafo_data.get('winding_config', [])):
                    edge.winding_config = trafo_data['winding_config'][i]
                else:
                    edge.winding_config = 'YNyn'  # Default winding configuration
                    
                if i < len(trafo_data.get('phase_shift_deg', [])):
                    edge.phase_shift_deg = trafo_data['phase_shift_deg'][i]
                else:
                    edge.phase_shift_deg = 0.0  # Default no phase shift
            
            # Calculate per-unit impedance
            Z_base = (edge.V_primary_kV ** 2) / edge.rating_MVA
            Z_pu = complex(edge.R_ohm, edge.X_ohm) / Z_base
            
            # Set coupling matrix
            coupling_matrix = self.coupling_calc.calculate_transformer_coupling(
                Z_leakage=Z_pu * Z_base,
                winding_type=edge.winding_config,
                mutual_coupling_factor=0.05
            )
            graph.set_edge_coupling(edge_id, coupling_matrix)
    
    def _add_shunts(self, graph: PowerGridGraph, shunt_data: Dict):
        """Add shunt elements to buses"""
        if not shunt_data or 'names' not in shunt_data:
            return
        
        for i, shunt_name in enumerate(shunt_data['names']):
            bus_name = shunt_data['buses'][i]
            
            if bus_name not in graph.nodes:
                continue
            
            # Get shunt values
            G_MW = shunt_data.get('G_MW', [0.0])[i] if i < len(shunt_data.get('G_MW', [])) else 0.0
            B_MVAR = shunt_data.get('B_MVAR', [0.0])[i] if i < len(shunt_data.get('B_MVAR', [])) else 0.0
            
            # Add to each phase (divided equally)
            for phase in PhaseType:
                node = graph.get_node(bus_name, phase)
                
                # Convert to per-unit
                S_base = self.base_mva
                Y_pu = complex(G_MW / S_base, B_MVAR / S_base) / 3  # Divide by 3 for per-phase
                
                # Add shunt admittance
                if hasattr(node, 'shunt_G_pu'):
                    node.shunt_G_pu += Y_pu.real
                    node.shunt_B_pu += Y_pu.imag
                else:
                    node.properties['shunt_G_pu'] = Y_pu.real
                    node.properties['shunt_B_pu'] = Y_pu.imag
    
    def _set_power_injections(self, graph: PowerGridGraph):
        """
        Set P_injection_pu and Q_injection_pu in node.properties for load flow solver.
        This method must be called AFTER generators and loads are added.
        
        The load flow solver expects power injections in per-unit on the system base (100 MVA).
        Generation is positive, load is negative.
        """
        S_base = self.base_mva  # 100 MVA
        
        nodes_with_gen = 0
        nodes_with_load = 0
        total_gen_mw = 0
        total_load_mw = 0
        
        for bus_name, phases in graph.nodes.items():
            for phase in PhaseType:
                node = phases[phase]
                
                # Initialize to zero
                P_inj_MW = 0.0
                Q_inj_MVAR = 0.0
                
                # Add generation if this is a generator node
                if node.node_type == 'generator' and hasattr(node, 'P_actual_MW'):
                    P_inj_MW += node.P_actual_MW  # Positive for generation
                    Q_inj_MVAR += node.Q_actual_MVAR
                    nodes_with_gen += 1
                    total_gen_mw += node.P_actual_MW
                
                # Subtract load if this is a load node or has load
                if node.node_type == 'load' and hasattr(node, 'P_MW'):
                    P_inj_MW -= node.P_MW  # Negative for load
                    Q_inj_MVAR -= node.Q_MVAR
                    nodes_with_load += 1
                    total_load_mw += node.P_MW
                
                # Convert to per-unit
                node.properties['P_injection_pu'] = P_inj_MW / S_base
                node.properties['Q_injection_pu'] = Q_inj_MVAR / S_base
        
        print(f"      ✓ Set power injections: {nodes_with_gen} gen nodes ({total_gen_mw:.1f} MW), {nodes_with_load} load nodes ({total_load_mw:.1f} MW)")