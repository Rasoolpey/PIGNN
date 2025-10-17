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
        """Load all data from H5 file"""
        with h5py.File(self.h5_path, 'r') as f:
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