"""
H5 data loader for IEEE39 bus system data.
Loads data from PowerFactory simulation results.
"""

import h5py
import numpy as np
from typing import Dict, List, Optional, Any


class H5DataLoader:
    """Load power system data from H5 files"""
    
    def __init__(self, h5_path: str):
        """
        Initialize loader with H5 file path.
        
        Args:
            h5_path: Path to H5 file
        """
        self.h5_path = h5_path
        self._file = None
        self._data_cache = {}
    
    def __enter__(self):
        """Context manager entry"""
        self._file = h5py.File(self.h5_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._file:
            self._file.close()
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all data from H5 file.
        Automatically detects format (old detailed_system_data or new phases/phase_a)
        """
        with h5py.File(self.h5_path, 'r') as f:
            # Detect which format this H5 file uses
            if 'phases' in f and 'phase_a' in f['phases']:
                # New Graph_model.h5 format
                return self._load_graph_model_format(f)
            else:
                # Old detailed_system_data format
                return self._load_legacy_format(f)
    
    def _load_graph_model_format(self, f: h5py.File) -> Dict[str, Any]:
        """Load data from new Graph_model.h5 format (phases/phase_a/...)"""
        phase_a_nodes = f['phases/phase_a/nodes']
        phase_a_edges = f['phases/phase_a/edges']
        topology = f['topology/edge_list']
        
        # Load bus data
        bus_names = [n.decode() if isinstance(n, bytes) else str(n) for n in phase_a_nodes['bus_names'][:]]
        num_buses = len(bus_names)
        
        buses = {
            'names': bus_names,
            'voltages_pu': phase_a_nodes['voltages_pu'][:],
            'voltage_angles_deg': phase_a_nodes['angles_deg'][:],
            'base_voltages_kV': phase_a_nodes['base_voltages_kV'][:],
            'active_injection_MW': phase_a_nodes['P_injection_MW'][:],
            'reactive_injection_MVAR': phase_a_nodes['Q_injection_MVAR'][:],
            'in_service': np.ones(num_buses, dtype=bool)  # Assume all in service
        }
        
        # Load generator data from dynamic_models
        generators = {}
        if 'dynamic_models' in f and 'generators' in f['dynamic_models']:
            gen_group = f['dynamic_models/generators']
            gen_names = [n.decode() if isinstance(n, bytes) else str(n) for n in gen_group['names'][:]]
            gen_buses = [b.decode() if isinstance(b, bytes) else str(b) for b in gen_group['buses'][:]]
            num_gens = len(gen_names)
            
            # Get generation from phase_a nodes
            gen_P_MW = np.zeros(num_gens)
            gen_Q_MVAR = np.zeros(num_gens)
            for i, bus_name in enumerate(gen_buses):
                if bus_name in bus_names:
                    bus_idx = bus_names.index(bus_name)
                    gen_P_MW[i] = phase_a_nodes['P_generation_MW'][bus_idx] * 3.0  # Convert back to total (3-phase)
                    gen_Q_MVAR[i] = phase_a_nodes['Q_generation_MVAR'][bus_idx] * 3.0
            
            generators = {
                'names': gen_names,
                'buses': gen_buses,
                'active_power_MW': gen_P_MW,
                'reactive_power_MVAR': gen_Q_MVAR,
                'voltage_setpoint_pu': np.ones(num_gens),  # Default
                'S_rated_MVA': gen_group['Sn_MVA'][:],
                'V_rated_kV': gen_group['Un_kV'][:],
                'xd_prime_pu': gen_group['xd_prime_pu'][:],
                'H_s': gen_group['H_s'][:],
                'D_pu': gen_group['D_pu'][:],
                'reactive_limits_min_MVAR': gen_group['Q_min_MVAR'][:],
                'reactive_limits_max_MVAR': gen_group['Q_max_MVAR'][:]
            }
        
        # Load load data from phase_a nodes
        loads = {}
        load_names = []
        load_buses = []
        load_P_MW = []
        load_Q_MVAR = []
        
        for i, bus_name in enumerate(bus_names):
            P_load = phase_a_nodes['P_load_MW'][i] * 3.0  # Convert to total (3-phase)
            Q_load = phase_a_nodes['Q_load_MVAR'][i] * 3.0
            if P_load > 0 or Q_load > 0:
                load_names.append(f"Load_{bus_name}")
                load_buses.append(bus_name)
                load_P_MW.append(P_load)
                load_Q_MVAR.append(Q_load)
        
        if load_names:
            loads = {
                'names': load_names,
                'buses': load_buses,
                'active_power_MW': np.array(load_P_MW),
                'reactive_power_MVAR': np.array(load_Q_MVAR),
                'power_factor': np.array([0.9] * len(load_names)),
                'load_type': ['constant_power'] * len(load_names)
            }
        
        # Load line and transformer data
        lines, transformers = self._split_edges_from_graph_model(f, bus_names)
        
        return {
            'scenario_info': {
                'scenario_id': 0,
                'contingency_type': 'normal',
                'description': 'Graph_model.h5 data',
                'timestamp': '2025-10-20T00:00:00'
            },
            'buses': buses,
            'generators': generators,
            'loads': loads,
            'lines': lines,
            'transformers': transformers,
            'shunts': {},
            'y_matrix': None
        }
    
    def _split_edges_from_graph_model(self, f: h5py.File, bus_names: List[str]) -> tuple:
        """Split edges into lines and transformers from Graph_model.h5"""
        edges = f['phases/phase_a/edges']
        topology = f['topology/edge_list']
        
        num_edges = len(edges['from_bus'])
        
        lines_data = {
            'names': [],
            'from_buses': [],
            'to_buses': [],
            'R_ohm': [],
            'X_ohm': [],
            'B_uS': [],
            'rating_MVA': [],
            'length_km': []
        }
        
        transformers_data = {
            'names': [],
            'from_buses': [],
            'to_buses': [],
            'R_ohm': [],
            'X_ohm': [],
            'rating_MVA': [],
            'V_primary_kV': [],
            'V_secondary_kV': []
        }
        
        for i in range(num_edges):
            from_bus_idx = int(edges['from_bus'][i])
            to_bus_idx = int(edges['to_bus'][i])
            edge_type = int(topology['edge_type'][i]) if 'edge_type' in topology else 0
            element_id = edges['element_id'][i].decode() if isinstance(edges['element_id'][i], bytes) else str(edges['element_id'][i])
            
            from_bus_name = bus_names[from_bus_idx] if from_bus_idx < len(bus_names) else f"Bus_{from_bus_idx}"
            to_bus_name = bus_names[to_bus_idx] if to_bus_idx < len(bus_names) else f"Bus_{to_bus_idx}"
            
            # Determine if line or transformer based on element_id or edge_type
            # edge_type: 0=line, 1=transformer (from EdgeType enum)
            is_transformer = edge_type == 1 or 'Trafo' in element_id or 'T_' in element_id
            
            if is_transformer:
                transformers_data['names'].append(element_id)
                transformers_data['from_buses'].append(from_bus_name)
                transformers_data['to_buses'].append(to_bus_name)
                transformers_data['R_ohm'].append(float(edges['R_pu'][i]))  # Note: R_pu in file
                transformers_data['X_ohm'].append(float(edges['X_pu'][i]))  # Note: X_pu in file
                transformers_data['rating_MVA'].append(float(edges['rating_MVA'][i]))
                # Get voltage levels from bus data
                from_bus_idx_safe = from_bus_idx if from_bus_idx < len(f['phases/phase_a/nodes/base_voltages_kV']) else 0
                to_bus_idx_safe = to_bus_idx if to_bus_idx < len(f['phases/phase_a/nodes/base_voltages_kV']) else 0
                transformers_data['V_primary_kV'].append(float(f['phases/phase_a/nodes/base_voltages_kV'][from_bus_idx_safe]))
                transformers_data['V_secondary_kV'].append(float(f['phases/phase_a/nodes/base_voltages_kV'][to_bus_idx_safe]))
            else:
                lines_data['names'].append(element_id)
                lines_data['from_buses'].append(from_bus_name)
                lines_data['to_buses'].append(to_bus_name)
                lines_data['R_ohm'].append(float(edges['R_pu'][i]))  # Note: R_pu in file
                lines_data['X_ohm'].append(float(edges['X_pu'][i]))  # Note: X_pu in file
                lines_data['B_uS'].append(float(edges['B_shunt_pu'][i]) * 1e6)  # Convert pu to uS
                lines_data['rating_MVA'].append(float(edges['rating_MVA'][i]))
                lines_data['length_km'].append(float(edges['length_km'][i]))
        
        # Convert lists to arrays
        lines = {k: np.array(v) if isinstance(v, list) and k not in ['names', 'from_buses', 'to_buses'] else v 
                 for k, v in lines_data.items()}
        transformers = {k: np.array(v) if isinstance(v, list) and k not in ['names', 'from_buses', 'to_buses'] else v 
                       for k, v in transformers_data.items()}
        
        return lines, transformers
    
    def _load_legacy_format(self, f: h5py.File) -> Dict[str, Any]:
        """Load data from old detailed_system_data format"""
        data = {
            'scenario_info': self._load_scenario_info(f),
            'buses': self._load_bus_data(f),
            'generators': self._load_generator_data(f),
            'loads': self._load_load_data(f),
            'lines': self._load_line_data(f),
            'transformers': self._load_transformer_data(f),
            'shunts': self._load_shunt_data(f),
            'y_matrix': self._load_y_matrix(f) if 'y_matrix' in f else None
        }
        return data
    
    def _load_scenario_info(self, f: h5py.File) -> Dict:
        """Load scenario metadata"""
        if 'scenario_metadata' not in f:
            return {
                'scenario_id': 0,
                'contingency_type': 'normal',
                'description': 'Unknown scenario',
                'timestamp': '2025-01-01T00:00:00'
            }
        
        meta = f['scenario_metadata']
        try:
            return {
                'scenario_id': meta.get('scenario_id', [0])[()],
                'contingency_type': meta.get('contingency_type', [b'normal'])[()].decode() if meta.get('contingency_type') is not None else 'normal',
                'description': meta.get('description', [b'Unknown scenario'])[()].decode() if meta.get('description') is not None else 'Unknown scenario',
                'timestamp': meta.get('execution_timestamp', [b'2025-01-01T00:00:00'])[()].decode() if meta.get('execution_timestamp') is not None else '2025-01-01T00:00:00'
            }
        except Exception:
            return {
                'scenario_id': 0,
                'contingency_type': 'normal',
                'description': 'Unknown scenario',
                'timestamp': '2025-01-01T00:00:00'
            }
    
    def _load_bus_data(self, f: h5py.File) -> Dict:
        """Load bus data"""
        if 'detailed_system_data/buses' not in f:
            return {}
        
        buses = f['detailed_system_data/buses']
        num_buses = len(buses['names'])
        
        return {
            'names': [n.decode() for n in buses['names'][:]],
            'voltages_pu': buses.get('voltages_pu', np.ones(num_buses))[:],
            'voltage_angles_deg': buses.get('voltage_angles_deg', np.zeros(num_buses))[:],
            'base_voltages_kV': buses.get('base_voltages_kV', np.full(num_buses, 138.0))[:],
            'active_injection_MW': buses.get('active_injection_MW', np.zeros(num_buses))[:],
            'reactive_injection_MVAR': buses.get('reactive_injection_MVAR', np.zeros(num_buses))[:],
            'in_service': buses.get('in_service', np.ones(num_buses, dtype=bool))[:]
        }
    
    def _load_generator_data(self, f: h5py.File) -> Dict:
        """Load generator data"""
        if 'detailed_system_data/generators' not in f:
            return {}
        
        gens = f['detailed_system_data/generators']
        num_gens = len(gens['names'])
        
        return {
            'names': [n.decode() for n in gens['names'][:]],
            'buses': [b.decode() for b in gens['buses'][:]],
            'active_power_MW': gens.get('active_power_MW', np.zeros(num_gens))[:],
            'reactive_power_MVAR': gens.get('reactive_power_MVAR', np.zeros(num_gens))[:],
            'voltage_setpoint_pu': gens.get('voltage_setpoint_pu', np.ones(num_gens))[:],
            'S_rated_MVA': gens.get('S_rated_MVA', np.full(num_gens, 100.0))[:],
            'V_rated_kV': gens.get('V_rated_kV', np.full(num_gens, 138.0))[:],
            'xd_prime_pu': gens.get('xd_prime_pu', np.full(num_gens, 0.25))[:],
            'H_s': gens.get('H_s', np.full(num_gens, 5.0))[:],
            'D_pu': gens.get('D_pu', np.full(num_gens, 2.0))[:],
            'reactive_limits_min_MVAR': gens.get('reactive_limits_min_MVAR', np.full(num_gens, -100.0))[:],
            'reactive_limits_max_MVAR': gens.get('reactive_limits_max_MVAR', np.full(num_gens, 100.0))[:]
        }
    
    def _load_load_data(self, f: h5py.File) -> Dict:
        """Load load data"""
        if 'detailed_system_data/loads' not in f:
            return {}
        
        loads = f['detailed_system_data/loads']
        num_loads = len(loads['names'])
        
        return {
            'names': [n.decode() for n in loads['names'][:]],
            'buses': [b.decode() for b in loads['buses'][:]],
            'active_power_MW': loads.get('active_power_MW', np.zeros(num_loads))[:],
            'reactive_power_MVAR': loads.get('reactive_power_MVAR', np.zeros(num_loads))[:],
            'power_factor': loads.get('power_factor', np.full(num_loads, 0.9))[:],
            'load_type': [t.decode() if isinstance(t, bytes) else 'constant_power' 
                         for t in loads.get('load_type', [b'constant_power'] * num_loads)]
        }
    
    def _load_line_data(self, f: h5py.File) -> Dict:
        """Load transmission line data"""
        if 'detailed_system_data/lines' not in f:
            return {}
        
        lines = f['detailed_system_data/lines']
        num_lines = len(lines['names'])
        
        return {
            'names': [n.decode() for n in lines['names'][:]],
            'from_buses': [b.decode() for b in lines['from_buses'][:]],
            'to_buses': [b.decode() for b in lines['to_buses'][:]],
            'R_ohm': lines.get('R_ohm', np.full(num_lines, 0.01))[:],
            'X_ohm': lines.get('X_ohm', np.full(num_lines, 0.1))[:],
            'B_uS': lines.get('B_uS', np.zeros(num_lines))[:],
            'rating_MVA': lines.get('rating_MVA', np.full(num_lines, 100.0))[:],
            'length_km': lines.get('length_km', np.full(num_lines, 10.0))[:],
            'in_service': lines.get('in_service', np.ones(num_lines, dtype=bool))[:]
        }
    
    def _load_transformer_data(self, f: h5py.File) -> Dict:
        """Load transformer data"""
        if 'detailed_system_data/transformers' not in f:
            return {}
        
        trafos = f['detailed_system_data/transformers']
        num_trafos = len(trafos['names'])
        
        return {
            'names': [n.decode() for n in trafos['names'][:]],
            'from_buses': [b.decode() for b in trafos['from_buses'][:]],
            'to_buses': [b.decode() for b in trafos['to_buses'][:]],
            'R_ohm': trafos['R_ohm'][:],
            'X_ohm': trafos['X_ohm'][:],
            'rating_MVA': trafos.get('rating_MVA', np.full(num_trafos, 100.0))[:],
            'V_primary_kV': trafos.get('V_primary_kV', np.full(num_trafos, 138.0))[:],
            'V_secondary_kV': trafos.get('V_secondary_kV', np.full(num_trafos, 69.0))[:],
            'tap_ratio': trafos.get('tap_ratio', np.ones(num_trafos))[:],
            'winding_config': [c.decode() if isinstance(c, bytes) else 'YY' 
                              for c in trafos.get('winding_config', [b'YY'] * num_trafos)],
            'phase_shift_deg': trafos.get('phase_shift_deg', np.zeros(num_trafos))[:]
        }
    
    def _load_shunt_data(self, f: h5py.File) -> Dict:
        """Load shunt data"""
        if 'detailed_system_data/shunts' not in f:
            return {}
        
        shunts = f['detailed_system_data/shunts']
        return {
            'names': [n.decode() for n in shunts['names'][:]],
            'buses': [b.decode() for b in shunts['buses'][:]],
            'G_MW': shunts.get('conductance', np.array([]))[:],
            'B_MVAR': shunts.get('capacitive', np.array([]))[:]
        }
    
    def _load_y_matrix(self, f: h5py.File) -> Optional[Dict]:
        """Load Y-matrix if available"""
        if 'y_matrix' not in f:
            return None
        
        y_data = f['y_matrix']
        
        # Check if sparse or dense
        if 'data' in y_data and 'indices' in y_data and 'indptr' in y_data:
            # Sparse matrix
            return {
                'format': 'sparse',
                'data': y_data['data'][:],
                'indices': y_data['indices'][:],
                'indptr': y_data['indptr'][:],
                'shape': (y_data['matrix_size'][()], y_data['matrix_size'][()]),
                'properties': self._load_dict_attributes(y_data['matrix_properties'])
            }
        elif 'Y_real' in y_data and 'Y_imag' in y_data:
            # Dense matrix
            return {
                'format': 'dense',
                'Y_real': y_data['Y_real'][:],
                'Y_imag': y_data['Y_imag'][:],
                'properties': self._load_dict_attributes(y_data['matrix_properties'])
            }
        
        return None
    
    def _load_dict_attributes(self, group: h5py.Group) -> Dict:
        """Load attributes from H5 group into dict"""
        result = {}
        for key in group.keys():
            try:
                result[key] = group[key][()]
            except:
                result[key] = None
        return result