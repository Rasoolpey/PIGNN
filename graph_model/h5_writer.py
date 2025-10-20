"""
H5 Writer for Power Grid Graph Data

This module implements the H5 file format specification for storing
three-phase power grid graphs with steady-state and dynamic parameters.

Author: PIGNN Project
Date: 2025-10-19
Version: 2.0
"""

import h5py
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from scipy.sparse import csr_matrix, issparse
import logging

logger = logging.getLogger(__name__)


class PowerGridH5Writer:
    """
    Write power grid data to H5 file following the PIGNN format specification.
    
    This class handles the complete storage of:
    - Network topology
    - Per-phase data (a, b, c)
    - Three-phase coupling matrices
    - Steady-state power flow results
    - Dynamic RMS simulation parameters (optional)
    - Multiple scenarios (optional)
    """
    
    def __init__(self, filepath: str, mode: str = 'w'):
        """
        Initialize H5 writer.
        
        Args:
            filepath: Path to H5 file
            mode: File mode ('w' for write, 'a' for append)
        """
        self.filepath = filepath
        self.mode = mode
        self.file = None
        
    @classmethod
    def default_filename(cls) -> str:
        """Return a recommended default filename for graph model exports."""
        return 'graph_model/Graph_model.h5'
        
    def __enter__(self):
        """Context manager entry"""
        self.file = h5py.File(self.filepath, self.mode)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.file:
            self.file.close()
    
    # ========================================================================
    # METADATA
    # ========================================================================
    
    def write_metadata(self, 
                       grid_name: str,
                       base_mva: float,
                       base_frequency_hz: float,
                       num_buses: int,
                       num_phases: int = 3,
                       description: str = "",
                       num_lines: Optional[int] = None,
                       num_transformers: Optional[int] = None,
                       num_generators: Optional[int] = None,
                       num_loads: Optional[int] = None):
        """
        Write file metadata.
        
        Args:
            grid_name: System name (e.g., "IEEE39_three_phase")
            base_mva: System base power (MVA)
            base_frequency_hz: Nominal frequency (Hz)
            num_buses: Number of buses
            num_phases: Number of phases (default 3)
            description: Optional description
        """
        logger.info(f"Writing metadata for {grid_name}...")
        
        metadata_group = self.file.create_group('metadata')
        
        metadata_group.attrs['version'] = '2.0'
        metadata_group.attrs['creation_date'] = datetime.now().isoformat()
        metadata_group.attrs['grid_name'] = grid_name
        metadata_group.attrs['base_mva'] = base_mva
        metadata_group.attrs['base_frequency_hz'] = base_frequency_hz
        metadata_group.attrs['num_buses'] = num_buses
        metadata_group.attrs['num_phases'] = num_phases
        if description:
            metadata_group.attrs['description'] = description
        # Optional system counts for quick inspection
        if num_lines is not None:
            metadata_group.attrs['num_lines'] = num_lines
        if num_transformers is not None:
            metadata_group.attrs['num_transformers'] = num_transformers
        if num_generators is not None:
            metadata_group.attrs['num_generators'] = num_generators
        if num_loads is not None:
            metadata_group.attrs['num_loads'] = num_loads
        
        logger.info("✓ Metadata written")
    
    # ========================================================================
    # TOPOLOGY
    # ========================================================================
    
    def write_topology(self,
                       adjacency_matrix: Optional[Union[np.ndarray, csr_matrix]] = None,
                       edge_list: Optional[Dict[str, np.ndarray]] = None):
        """
        Write graph topology.
        
        Args:
            adjacency_matrix: Sparse or dense adjacency matrix
            edge_list: Dict with 'from_bus', 'to_bus', 'edge_type'
        """
        logger.info("Writing topology...")
        
        topology_group = self.file.create_group('topology')
        
        # Write adjacency matrix if provided
        if adjacency_matrix is not None:
            adj_group = topology_group.create_group('adjacency_matrix')
            
            if issparse(adjacency_matrix):
                # Convert to CSR if not already
                if not isinstance(adjacency_matrix, csr_matrix):
                    adjacency_matrix = csr_matrix(adjacency_matrix)
                
                adj_group.create_dataset('data', data=adjacency_matrix.data)
                adj_group.create_dataset('indices', data=adjacency_matrix.indices)
                adj_group.create_dataset('indptr', data=adjacency_matrix.indptr)
                adj_group.create_dataset('shape', data=adjacency_matrix.shape)
            else:
                # Dense matrix - convert to sparse
                sparse_adj = csr_matrix(adjacency_matrix)
                adj_group.create_dataset('data', data=sparse_adj.data)
                adj_group.create_dataset('indices', data=sparse_adj.indices)
                adj_group.create_dataset('indptr', data=sparse_adj.indptr)
                adj_group.create_dataset('shape', data=adjacency_matrix.shape)
        
        # Write edge list if provided
        if edge_list is not None:
            edge_group = topology_group.create_group('edge_list')
            edge_group.create_dataset('from_bus', data=edge_list['from_bus'])
            edge_group.create_dataset('to_bus', data=edge_list['to_bus'])
            edge_group.create_dataset('edge_type', data=edge_list['edge_type'])
        
        logger.info("✓ Topology written")
    
    # ========================================================================
    # PER-PHASE DATA
    # ========================================================================
    
    def write_phase_data(self,
                         phase_name: str,
                         node_data: Dict[str, np.ndarray],
                         edge_data: Dict[str, np.ndarray]):
        """
        Write data for a single phase (a, b, or c).
        
        Args:
            phase_name: 'phase_a', 'phase_b', or 'phase_c'
            node_data: Dictionary with node datasets
            edge_data: Dictionary with edge datasets
        """
        logger.info(f"Writing {phase_name} data...")
        
        phases_group = self.file.require_group('phases')
        phase_group = phases_group.create_group(phase_name)
        
        # Write node data
        nodes_group = phase_group.create_group('nodes')
        for key, value in node_data.items():
            if value is not None:
                self._write_dataset(nodes_group, key, value)
        
        # Write edge data
        edges_group = phase_group.create_group('edges')
        for key, value in edge_data.items():
            if value is not None:
                self._write_dataset(edges_group, key, value)
        
        logger.info(f"✓ {phase_name} data written")
    
    def write_all_phases(self,
                         phase_a_nodes: Dict[str, np.ndarray],
                         phase_a_edges: Dict[str, np.ndarray],
                         phase_b_nodes: Dict[str, np.ndarray],
                         phase_b_edges: Dict[str, np.ndarray],
                         phase_c_nodes: Dict[str, np.ndarray],
                         phase_c_edges: Dict[str, np.ndarray]):
        """
        Write data for all three phases at once.
        
        Args:
            phase_a_nodes: Phase A node data
            phase_a_edges: Phase A edge data
            phase_b_nodes: Phase B node data
            phase_b_edges: Phase B edge data
            phase_c_nodes: Phase C node data
            phase_c_edges: Phase C edge data
        """
        self.write_phase_data('phase_a', phase_a_nodes, phase_a_edges)
        self.write_phase_data('phase_b', phase_b_nodes, phase_b_edges)
        self.write_phase_data('phase_c', phase_c_nodes, phase_c_edges)
    
    # ========================================================================
    # COUPLING MATRICES
    # ========================================================================
    
    def write_line_coupling(self,
                            line_names: List[str],
                            Z_matrix_abc: Optional[np.ndarray] = None,
                            Y_shunt_abc: Optional[np.ndarray] = None,
                            Z0_ohm: Optional[np.ndarray] = None,
                            Z1_ohm: Optional[np.ndarray] = None,
                            Z2_ohm: Optional[np.ndarray] = None):
        """
        Write three-phase coupling matrices for transmission lines.
        
        Args:
            line_names: List of line identifiers
            Z_matrix_abc: Full 3x3 impedance matrices (N_lines, 3, 3) [Ohm]
            Y_shunt_abc: Shunt admittance matrices (N_lines, 3, 3) [S]
            Z0_ohm: Zero-sequence impedances (alternative to Z_matrix_abc)
            Z1_ohm: Positive-sequence impedances
            Z2_ohm: Negative-sequence impedances
        """
        logger.info("Writing line coupling matrices...")
        
        coupling_group = self.file.require_group('coupling')
        line_coupling_group = coupling_group.create_group('line_coupling')
        
        # Write line names
        self._write_dataset(line_coupling_group, 'line_names', 
                           np.array(line_names, dtype='S'))
        
        # Write 3x3 matrices if provided
        if Z_matrix_abc is not None:
            line_coupling_group.create_dataset('Z_matrix_abc', data=Z_matrix_abc,
                                              dtype=np.complex128)
        
        if Y_shunt_abc is not None:
            line_coupling_group.create_dataset('Y_shunt_abc', data=Y_shunt_abc,
                                              dtype=np.complex128)
        
        # Write sequence impedances if provided (alternative representation)
        if Z0_ohm is not None or Z1_ohm is not None or Z2_ohm is not None:
            seq_group = line_coupling_group.create_group('sequence_impedances')
            if Z0_ohm is not None:
                seq_group.create_dataset('Z0_ohm', data=Z0_ohm, dtype=np.complex128)
            if Z1_ohm is not None:
                seq_group.create_dataset('Z1_ohm', data=Z1_ohm, dtype=np.complex128)
            if Z2_ohm is not None:
                seq_group.create_dataset('Z2_ohm', data=Z2_ohm, dtype=np.complex128)
        
        logger.info("✓ Line coupling matrices written")
    
    def write_transformer_coupling(self,
                                   transformer_names: List[str],
                                   winding_config: List[str],
                                   vector_group: Optional[List[str]] = None,
                                   phase_shift_deg: Optional[np.ndarray] = None,
                                   Z_matrix_primary: Optional[np.ndarray] = None,
                                   Z_matrix_secondary: Optional[np.ndarray] = None,
                                   connection_matrix: Optional[np.ndarray] = None):
        """
        Write three-phase coupling matrices for transformers.
        
        Args:
            transformer_names: List of transformer identifiers
            winding_config: Winding configurations (e.g., 'YNyn0', 'Dyn11')
            vector_group: IEC vector group notation
            phase_shift_deg: Phase shifts in degrees
            Z_matrix_primary: Primary side 3x3 impedance matrices
            Z_matrix_secondary: Secondary side 3x3 impedance matrices
            connection_matrix: Connection/coupling 3x3 matrices
        """
        logger.info("Writing transformer coupling matrices...")
        
        coupling_group = self.file.require_group('coupling')
        trafo_coupling_group = coupling_group.create_group('transformer_coupling')
        
        # Write transformer names
        self._write_dataset(trafo_coupling_group, 'transformer_names',
                           np.array(transformer_names, dtype='S'))
        
        # Write winding configurations
        self._write_dataset(trafo_coupling_group, 'winding_config',
                           np.array(winding_config, dtype='S'))
        
        if vector_group is not None:
            self._write_dataset(trafo_coupling_group, 'vector_group',
                               np.array(vector_group, dtype='S'))
        
        if phase_shift_deg is not None:
            trafo_coupling_group.create_dataset('phase_shift_deg', data=phase_shift_deg)
        
        # Write 3x3 impedance matrices
        if Z_matrix_primary is not None:
            trafo_coupling_group.create_dataset('Z_matrix_primary', data=Z_matrix_primary,
                                               dtype=np.complex128)
        
        if Z_matrix_secondary is not None:
            trafo_coupling_group.create_dataset('Z_matrix_secondary', data=Z_matrix_secondary,
                                               dtype=np.complex128)
        
        if connection_matrix is not None:
            trafo_coupling_group.create_dataset('connection_matrix', data=connection_matrix,
                                               dtype=np.complex128)
        
        logger.info("✓ Transformer coupling matrices written")
    
    # ========================================================================
    # STEADY-STATE RESULTS
    # ========================================================================
    
    def write_power_flow_results(self,
                                  converged: bool,
                                  iterations: int,
                                  max_mismatch: float,
                                  total_generation_MW: float,
                                  total_load_MW: float,
                                  total_losses_MW: float,
                                  max_voltage_pu: float,
                                  min_voltage_pu: float):
        """
        Write power flow solution results.
        
        Args:
            converged: Convergence status
            iterations: Number of iterations
            max_mismatch: Maximum power mismatch (pu)
            total_generation_MW: Total generation
            total_load_MW: Total load
            total_losses_MW: Total losses
            max_voltage_pu: Maximum bus voltage
            min_voltage_pu: Minimum bus voltage
        """
        logger.info("Writing power flow results...")
        
        ss_group = self.file.require_group('steady_state')
        pf_group = ss_group.create_group('power_flow_results')
        
        pf_group.attrs['converged'] = converged
        pf_group.attrs['iterations'] = iterations
        pf_group.attrs['max_mismatch'] = max_mismatch
        pf_group.attrs['total_generation_MW'] = total_generation_MW
        pf_group.attrs['total_load_MW'] = total_load_MW
        pf_group.attrs['total_losses_MW'] = total_losses_MW
        pf_group.attrs['max_voltage_pu'] = max_voltage_pu
        pf_group.attrs['min_voltage_pu'] = min_voltage_pu
        pf_group.attrs['timestamp'] = datetime.now().isoformat()
        
        logger.info("✓ Power flow results written")
    
    def write_admittance_matrix(self,
                                Y_single_phase: Optional[csr_matrix] = None,
                                Y_three_phase: Optional[csr_matrix] = None):
        """
        Write admittance matrices.
        
        Args:
            Y_single_phase: Per-phase Y-matrix (N×N) sparse
            Y_three_phase: Full 3-phase Y-matrix (3N×3N) sparse
        """
        logger.info("Writing admittance matrices...")
        
        ss_group = self.file.require_group('steady_state')
        y_group = ss_group.create_group('admittance_matrix')
        
        if Y_single_phase is not None:
            y1_group = y_group.create_group('Y_single_phase')
            self._write_sparse_complex_matrix(y1_group, Y_single_phase)
        
        if Y_three_phase is not None:
            y3_group = y_group.create_group('Y_three_phase')
            self._write_sparse_complex_matrix(y3_group, Y_three_phase)
        
        logger.info("✓ Admittance matrices written")
    
    # ========================================================================
    # DYNAMIC MODELS (RMS Parameters)
    # ========================================================================
    
    def write_generator_dynamics(self,
                                  names: List[str],
                                  buses: List[str],
                                  phases: List[str],
                                  model_type: List[str],
                                  parameters: Dict[str, np.ndarray]):
        """
        Write generator dynamic parameters for RMS simulation.
        
        Args:
            names: Generator names
            buses: Connected bus names
            phases: Connected phases ('a', 'b', 'c', or 'abc')
            model_type: Model types ('GENCLS', 'GENROU', etc.)
            parameters: Dictionary of parameter arrays:
                - H_s, D_pu (mechanical)
                - xd_pu, xq_pu, xd_prime_pu, etc. (electrical)
                - Td0_prime_s, Tq0_prime_s, etc. (time constants)
                - S10, S12 (saturation)
                - S_rated_MVA, V_rated_kV (ratings)
        """
        logger.info("Writing generator dynamic parameters...")
        
        dyn_group = self.file.require_group('dynamic_models')
        gen_group = dyn_group.create_group('generators')
        
        # Write identifiers
        self._write_dataset(gen_group, 'names', np.array(names, dtype='S'))
        self._write_dataset(gen_group, 'buses', np.array(buses, dtype='S'))
        self._write_dataset(gen_group, 'phases', np.array(phases, dtype='S'))
        self._write_dataset(gen_group, 'model_type', np.array(model_type, dtype='S'))
        
        # Write all parameters
        for param_name, param_values in parameters.items():
            if param_values is not None:
                gen_group.create_dataset(param_name, data=param_values)
        
        logger.info("✓ Generator dynamics written")
    
    def write_exciter_models(self,
                              names: List[str],
                              generator_names: List[str],
                              model_type: List[str],
                              parameters: Dict[str, np.ndarray]):
        """
        Write excitation system (AVR) parameters.
        
        Args:
            names: Exciter names
            generator_names: Associated generator names
            model_type: Exciter model types ('SEXS', 'IEEEAC1A', etc.)
            parameters: Dictionary of parameter arrays:
                - Ka, Ta_s, Efd_max, Efd_min (generic)
                - Model-specific parameters
        """
        logger.info("Writing exciter models...")
        
        dyn_group = self.file.require_group('dynamic_models')
        exc_group = dyn_group.create_group('exciters')
        
        self._write_dataset(exc_group, 'names', np.array(names, dtype='S'))
        self._write_dataset(exc_group, 'generator_names', np.array(generator_names, dtype='S'))
        self._write_dataset(exc_group, 'model_type', np.array(model_type, dtype='S'))
        
        for param_name, param_values in parameters.items():
            if param_values is not None:
                exc_group.create_dataset(param_name, data=param_values)
        
        logger.info("✓ Exciter models written")
    
    def write_governor_models(self,
                               names: List[str],
                               generator_names: List[str],
                               model_type: List[str],
                               parameters: Dict[str, np.ndarray]):
        """
        Write governor/turbine parameters.
        
        Args:
            names: Governor names
            generator_names: Associated generator names
            model_type: Governor model types ('TGOV1', 'HYGOV', etc.)
            parameters: Dictionary of parameter arrays:
                - R_pu, Tg_s, Tt_s, Dt_pu, Pmax_pu, Pmin_pu
        """
        logger.info("Writing governor models...")
        
        dyn_group = self.file.require_group('dynamic_models')
        gov_group = dyn_group.create_group('governors')
        
        self._write_dataset(gov_group, 'names', np.array(names, dtype='S'))
        self._write_dataset(gov_group, 'generator_names', np.array(generator_names, dtype='S'))
        self._write_dataset(gov_group, 'model_type', np.array(model_type, dtype='S'))
        
        for param_name, param_values in parameters.items():
            if param_values is not None:
                gov_group.create_dataset(param_name, data=param_values)
        
        logger.info("✓ Governor models written")
    
    def write_dynamic_loads(self,
                            names: List[str],
                            buses: List[str],
                            model_type: List[str],
                            parameters: Dict[str, np.ndarray]):
        """
        Write dynamic load models (optional).
        
        Args:
            names: Load names
            buses: Connected bus names
            model_type: Load model types ('ZIP', 'exponential', etc.)
            parameters: Model-specific parameters
        """
        logger.info("Writing dynamic load models...")
        
        dyn_group = self.file.require_group('dynamic_models')
        load_group = dyn_group.create_group('dynamic_loads')
        
        self._write_dataset(load_group, 'names', np.array(names, dtype='S'))
        self._write_dataset(load_group, 'buses', np.array(buses, dtype='S'))
        self._write_dataset(load_group, 'model_type', np.array(model_type, dtype='S'))
        
        for param_name, param_values in parameters.items():
            if param_values is not None:
                load_group.create_dataset(param_name, data=param_values)
        
        logger.info("✓ Dynamic load models written")
    
    # ========================================================================
    # INITIAL CONDITIONS
    # ========================================================================
    
    def write_initial_conditions(self,
                                  rotor_angles_rad: np.ndarray,
                                  rotor_speeds_pu: Optional[np.ndarray] = None,
                                  field_voltages_pu: Optional[np.ndarray] = None,
                                  mechanical_power_pu: Optional[np.ndarray] = None):
        """
        Write initial dynamic states for RMS simulation.
        
        Args:
            rotor_angles_rad: Initial rotor angles (rad)
            rotor_speeds_pu: Initial rotor speeds (pu), default 1.0
            field_voltages_pu: Initial field voltages (pu)
            mechanical_power_pu: Initial mechanical power (pu)
        """
        logger.info("Writing initial conditions...")
        
        ic_group = self.file.create_group('initial_conditions')
        gen_states_group = ic_group.create_group('generator_states')
        
        gen_states_group.create_dataset('rotor_angles_rad', data=rotor_angles_rad)
        
        if rotor_speeds_pu is None:
            rotor_speeds_pu = np.ones_like(rotor_angles_rad)
        gen_states_group.create_dataset('rotor_speeds_pu', data=rotor_speeds_pu)
        
        if field_voltages_pu is not None:
            gen_states_group.create_dataset('field_voltages_pu', data=field_voltages_pu)
        
        if mechanical_power_pu is not None:
            gen_states_group.create_dataset('mechanical_power_pu', data=mechanical_power_pu)
        
        logger.info("✓ Initial conditions written")
    
    # ========================================================================
    # SCENARIOS
    # ========================================================================
    
    def write_scenario(self,
                       scenario_id: str,
                       description: str,
                       voltages_pu: np.ndarray,
                       angles_deg: np.ndarray,
                       P_injections_MW: np.ndarray,
                       Q_injections_MVAR: np.ndarray,
                       contingency_description: str = "",
                       power_flow_converged: bool = True):
        """
        Write a single operating scenario.
        
        Args:
            scenario_id: Unique scenario identifier
            description: Scenario description
            voltages_pu: Voltage magnitudes (3*N_buses)
            angles_deg: Voltage angles (3*N_buses)
            P_injections_MW: Active power injections (3*N_buses)
            Q_injections_MVAR: Reactive power injections (3*N_buses)
            contingency_description: What changed in this scenario
            power_flow_converged: Convergence status
        """
        logger.info(f"Writing scenario {scenario_id}...")
        
        scenarios_group = self.file.require_group('scenarios')
        
        # Update registry if exists, create if not
        if 'scenario_registry' not in scenarios_group:
            reg_group = scenarios_group.create_group('scenario_registry')
            reg_group.create_dataset('scenario_ids', data=np.array([scenario_id], dtype='S'),
                                    maxshape=(None,), chunks=True)
            reg_group.create_dataset('descriptions', data=np.array([description], dtype='S'),
                                    maxshape=(None,), chunks=True)
            reg_group.create_dataset('timestamps', 
                                    data=np.array([datetime.now().isoformat()], dtype='S'),
                                    maxshape=(None,), chunks=True)
        else:
            # Append to existing registry
            reg_group = scenarios_group['scenario_registry']
            for ds_name, new_value in [('scenario_ids', scenario_id),
                                       ('descriptions', description),
                                       ('timestamps', datetime.now().isoformat())]:
                ds = reg_group[ds_name]
                old_len = ds.shape[0]
                ds.resize((old_len + 1,))
                ds[old_len] = new_value
        
        # Write scenario data
        scenario_group = scenarios_group.create_group(f'scenario_{scenario_id}')
        scenario_group.create_dataset('voltages_pu', data=voltages_pu)
        scenario_group.create_dataset('angles_deg', data=angles_deg)
        scenario_group.create_dataset('P_injections_MW', data=P_injections_MW)
        scenario_group.create_dataset('Q_injections_MVAR', data=Q_injections_MVAR)
        scenario_group.attrs['contingency_description'] = contingency_description
        scenario_group.attrs['power_flow_converged'] = power_flow_converged
        
        logger.info(f"✓ Scenario {scenario_id} written")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _write_dataset(self, group: h5py.Group, name: str, data: np.ndarray):
        """Helper to write a dataset with proper handling of string arrays."""
        if data.dtype.kind in ['U', 'S', 'O']:  # String types
            # Ensure bytes with proper dtype
            if data.dtype.kind == 'U':
                # Convert unicode to bytes
                data = np.array([s.encode('utf-8') if isinstance(s, str) else s 
                                for s in data.flat]).reshape(data.shape)
            # Let h5py infer the string dtype from the data
            group.create_dataset(name, data=data)
        else:
            group.create_dataset(name, data=data)
    
    def _write_sparse_complex_matrix(self, group: h5py.Group, matrix: csr_matrix):
        """Write a sparse complex matrix in CSR format."""
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        group.create_dataset('data_real', data=matrix.data.real)
        group.create_dataset('data_imag', data=matrix.data.imag)
        group.create_dataset('indices', data=matrix.indices)
        group.create_dataset('indptr', data=matrix.indptr)
        group.create_dataset('shape', data=matrix.shape)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_default_generator_parameters(n_generators: int) -> Dict[str, np.ndarray]:
    """
    Create default generator dynamic parameters for initial testing.
    
    These are typical/default values. MUST be replaced with actual
    PowerFactory data for realistic simulations.
    
    Args:
        n_generators: Number of generators
        
    Returns:
        Dictionary of parameter arrays with defaults
    """
    return {
        # Mechanical
        'H_s': np.full(n_generators, 5.0),
        'D_pu': np.full(n_generators, 2.0),
        
        # Electrical - Reactances
        'xd_pu': np.full(n_generators, 1.8),
        'xq_pu': np.full(n_generators, 1.7),
        'xd_prime_pu': np.full(n_generators, 0.3),
        'xq_prime_pu': np.full(n_generators, 0.55),
        'xd_double_prime_pu': np.full(n_generators, 0.25),
        'xq_double_prime_pu': np.full(n_generators, 0.25),
        'xl_pu': np.full(n_generators, 0.15),
        'ra_pu': np.full(n_generators, 0.003),
        
        # Time Constants
        'Td0_prime_s': np.full(n_generators, 8.0),
        'Tq0_prime_s': np.full(n_generators, 0.4),
        'Td0_double_prime_s': np.full(n_generators, 0.03),
        'Tq0_double_prime_s': np.full(n_generators, 0.05),
        
        # Saturation
        'S10': np.full(n_generators, 0.0),
        'S12': np.full(n_generators, 0.0),
    }


def create_default_exciter_parameters(n_generators: int, model_type: str = 'SEXS') -> Dict[str, np.ndarray]:
    """
    Create default exciter parameters (SEXS - Simple Exciter).
    
    Args:
        n_generators: Number of generators
        model_type: Exciter model type (default 'SEXS')
        
    Returns:
        Dictionary of exciter parameters
    """
    if model_type == 'SEXS':
        return {
            'Ka': np.full(n_generators, 200.0),
            'Ta_s': np.full(n_generators, 0.05),
            'Tb_s': np.full(n_generators, 10.0),
            'Tc_s': np.full(n_generators, 1.0),
            'Efd_max': np.full(n_generators, 5.0),
            'Efd_min': np.full(n_generators, -5.0),
        }
    else:
        # Placeholder for other models
        return {}


def create_default_governor_parameters(n_generators: int, model_type: str = 'TGOV1') -> Dict[str, np.ndarray]:
    """
    Create default governor parameters (TGOV1 - Simple Governor).
    
    Args:
        n_generators: Number of generators
        model_type: Governor model type (default 'TGOV1')
        
    Returns:
        Dictionary of governor parameters
    """
    if model_type == 'TGOV1':
        return {
            'R_pu': np.full(n_generators, 0.05),
            'Dt_pu': np.full(n_generators, 0.0),
            'Tg_s': np.full(n_generators, 0.2),
            'Tt_s': np.full(n_generators, 0.5),
            'Pmax_pu': np.full(n_generators, 1.0),
            'Pmin_pu': np.full(n_generators, 0.0),
        }
    else:
        return {}
