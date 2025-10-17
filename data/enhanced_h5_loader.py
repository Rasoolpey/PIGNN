"""
Enhanced H5 data loader with proper power injection data extraction.
This fixes the missing power injection issue identified in our diagnosis.
"""

import h5py
import numpy as np
from typing import Dict, List, Optional, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EnhancedH5DataLoader:
    """Enhanced H5 data loader that extracts complete power system data"""
    
    def __init__(self, h5_path: str):
        """
        Initialize enhanced loader with H5 file path.
        
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
    
    def load_complete_system_data(self) -> Dict[str, Any]:
        """Load complete system data with proper power injections"""
        with h5py.File(self.h5_path, 'r') as f:
            print("Enhanced H5 Data Loading...")
            
            # Load basic system data
            data = {
                'scenario_info': self._load_scenario_info(f),
                'buses': self._load_enhanced_bus_data(f),
                'generators': self._load_enhanced_generator_data(f),
                'loads': self._load_enhanced_load_data(f),
                'lines': self._load_enhanced_line_data(f),
                'transformers': self._load_enhanced_transformer_data(f),
                'shunts': self._load_shunt_data(f),
                'y_matrix': self._load_y_matrix(f) if 'y_matrix' in f else None
            }
            
            # Validate and fix power injection data
            data = self._fix_power_injection_data(data)
            
            return data
    
    def _load_enhanced_bus_data(self, f: h5py.File) -> Dict:
        """Load enhanced bus data with proper power injection handling"""
        print("   Loading bus data...")
        
        if 'detailed_system_data/buses' not in f:
            print("   ⚠️  No bus data found in H5 file")
            return {}
        
        buses_group = f['detailed_system_data/buses']
        
        # Basic bus data
        bus_data = {
            'names': self._safe_load_dataset(buses_group, 'names', []),
            'voltages_pu': self._safe_load_dataset(buses_group, 'voltages_pu', []),
            'voltage_angles_deg': self._safe_load_dataset(buses_group, 'voltage_angles_deg', []),
            'base_voltages_kV': self._safe_load_dataset(buses_group, 'base_voltages_kV', []),
            'in_service': self._safe_load_dataset(buses_group, 'in_service', []),
        }
        
        # Try to load power injection data (might be NaN or missing)
        active_inj = self._safe_load_dataset(buses_group, 'active_injection_MW', None)
        reactive_inj = self._safe_load_dataset(buses_group, 'reactive_injection_MVAR', None)
        
        # Check if power injection data is valid
        if active_inj is not None and not np.all(np.isnan(active_inj)):
            bus_data['active_injection_MW'] = active_inj
            bus_data['reactive_injection_MVAR'] = reactive_inj
            print("   Found valid power injection data")
        else:
            print("   Power injection data is missing or invalid - will synthesize")
            bus_data['active_injection_MW'] = None
            bus_data['reactive_injection_MVAR'] = None
        
        return bus_data
    
    def _load_enhanced_generator_data(self, f: h5py.File) -> Dict:
        """Load generator data with power outputs"""
        print("   Loading generator data...")
        
        if 'detailed_system_data/generators' not in f:
            print("   No generator data found")
            return {}
        
        gen_group = f['detailed_system_data/generators']
        
        gen_data = {
            'names': self._safe_load_dataset(gen_group, 'names', []),
            'buses': self._safe_load_dataset(gen_group, 'buses', []),
            'active_power_MW': self._safe_load_dataset(gen_group, 'active_power_MW', []),
            'reactive_power_MVAR': self._safe_load_dataset(gen_group, 'reactive_power_MVAR', []),
            'voltage_setpoint_pu': self._safe_load_dataset(gen_group, 'voltage_setpoint_pu', []),
            'reactive_limits_min_MVAR': self._safe_load_dataset(gen_group, 'reactive_limits_min_MVAR', []),
            'reactive_limits_max_MVAR': self._safe_load_dataset(gen_group, 'reactive_limits_max_MVAR', []),
            'in_service': self._safe_load_dataset(gen_group, 'in_service', []),
        }
        
        return gen_data
    
    def _load_enhanced_load_data(self, f: h5py.File) -> Dict:
        """Load load data with power demands"""
        print("   Loading load data...")
        
        if 'detailed_system_data/loads' not in f:
            print("   No load data found")
            return {}
        
        load_group = f['detailed_system_data/loads']
        
        load_data = {
            'names': self._safe_load_dataset(load_group, 'names', []),
            'buses': self._safe_load_dataset(load_group, 'buses', []),
            'active_power_MW': self._safe_load_dataset(load_group, 'active_power_MW', []),
            'reactive_power_MVAR': self._safe_load_dataset(load_group, 'reactive_power_MVAR', []),
            'in_service': self._safe_load_dataset(load_group, 'in_service', []),
        }
        
        return load_data
    
    def _load_enhanced_line_data(self, f: h5py.File) -> Dict:
        """Load line data with proper impedance units"""
        print("   Loading line data...")
        
        if 'detailed_system_data/lines' not in f:
            return {}
        
        line_group = f['detailed_system_data/lines']
        
        line_data = {
            'names': self._safe_load_dataset(line_group, 'names', []),
            'from_buses': self._safe_load_dataset(line_group, 'from_buses', []),
            'to_buses': self._safe_load_dataset(line_group, 'to_buses', []),
            'R_ohm': self._safe_load_dataset(line_group, 'R_ohm', []),
            'X_ohm': self._safe_load_dataset(line_group, 'X_ohm', []),
            'B_uS': self._safe_load_dataset(line_group, 'B_uS', []),
            'rating_MVA': self._safe_load_dataset(line_group, 'rating_MVA', []),
            'length_km': self._safe_load_dataset(line_group, 'length_km', []),
            'in_service': self._safe_load_dataset(line_group, 'in_service', []),
        }
        
        return line_data
    
    def _load_enhanced_transformer_data(self, f: h5py.File) -> Dict:
        """Load transformer data"""
        print("   Loading transformer data...")
        
        if 'detailed_system_data/transformers' not in f:
            return {}
        
        trafo_group = f['detailed_system_data/transformers']
        
        trafo_data = {
            'names': self._safe_load_dataset(trafo_group, 'names', []),
            'from_buses': self._safe_load_dataset(trafo_group, 'from_buses', []),
            'to_buses': self._safe_load_dataset(trafo_group, 'to_buses', []),
            'R_ohm': self._safe_load_dataset(trafo_group, 'R_ohm', []),
            'X_ohm': self._safe_load_dataset(trafo_group, 'X_ohm', []),
            'rating_MVA': self._safe_load_dataset(trafo_group, 'rating_MVA', []),
            'tap_ratio': self._safe_load_dataset(trafo_group, 'tap_ratio', []),
            'in_service': self._safe_load_dataset(trafo_group, 'in_service', []),
        }
        
        return trafo_data
    
    def _fix_power_injection_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix missing or invalid power injection data by synthesizing from generators and loads"""
        print("   Fixing power injection data...")
        
        buses = data.get('buses', {})
        generators = data.get('generators', {})
        loads = data.get('loads', {})
        
        bus_names = buses.get('names', [])
        if len(bus_names) == 0:
            print("   No bus data available")
            return data
        
        n_buses = len(bus_names)
        
        # Initialize power injection arrays
        P_injection = np.zeros(n_buses)
        Q_injection = np.zeros(n_buses)
        
        # Create bus name to index mapping
        bus_to_index = {name: i for i, name in enumerate(bus_names)}
        
        # Add generator contributions (positive injection)
        gen_names = generators.get('names', [])
        if len(gen_names) > 0:
            gen_buses = generators.get('buses', [])
            gen_P = generators.get('active_power_MW', [])
            gen_Q = generators.get('reactive_power_MVAR', [])
            
            for i, bus_name in enumerate(gen_buses):
                if bus_name in bus_to_index and i < len(gen_P) and i < len(gen_Q):
                    bus_idx = bus_to_index[bus_name]
                    if not np.isnan(gen_P[i]):
                        P_injection[bus_idx] += gen_P[i]
                    if not np.isnan(gen_Q[i]):
                        Q_injection[bus_idx] += gen_Q[i]
            
            print(f"   Added {len(gen_buses)} generator contributions")
        
        # Add load contributions (negative injection)
        load_names = loads.get('names', [])
        if len(load_names) > 0:
            load_buses = loads.get('buses', [])
            load_P = loads.get('active_power_MW', [])
            load_Q = loads.get('reactive_power_MVAR', [])
            
            for i, bus_name in enumerate(load_buses):
                if bus_name in bus_to_index and i < len(load_P) and i < len(load_Q):
                    bus_idx = bus_to_index[bus_name]
                    if not np.isnan(load_P[i]):
                        P_injection[bus_idx] -= load_P[i]  # Negative for load
                    if not np.isnan(load_Q[i]):
                        Q_injection[bus_idx] -= load_Q[i]  # Negative for load
            
            print(f"   Added {len(load_buses)} load contributions")
        
        # Update bus data with synthesized power injections
        buses['active_injection_MW'] = P_injection
        buses['reactive_injection_MVAR'] = Q_injection
        
        print(f"   Power injection summary:")
        print(f"      - Total generation: {np.sum(P_injection[P_injection > 0]):.1f} MW")
        print(f"      - Total load: {abs(np.sum(P_injection[P_injection < 0])):.1f} MW")
        print(f"      - Net imbalance: {np.sum(P_injection):.1f} MW")
        
        return data
    
    def _safe_load_dataset(self, group, name: str, default=None):
        """Safely load dataset from HDF5 group"""
        try:
            if name in group:
                data = group[name][()]
                if isinstance(data, bytes):
                    return data.decode('utf-8')
                elif isinstance(data, np.ndarray) and data.dtype.kind in ['S', 'U']:
                    return [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in data]
                return data
            else:
                return default
        except Exception as e:
            print(f"   Error loading {name}: {e}")
            return default
    
    def _load_scenario_info(self, f: h5py.File) -> Dict:
        """Load scenario information"""
        if 'scenario_info' in f:
            return {
                'scenario_id': self._safe_load_dataset(f['scenario_info'], 'scenario_id', 0),
                'description': self._safe_load_dataset(f['scenario_info'], 'description', ''),
                'timestamp': self._safe_load_dataset(f['scenario_info'], 'timestamp', ''),
            }
        return {}
    
    def _load_shunt_data(self, f: h5py.File) -> Dict:
        """Load shunt data"""
        if 'detailed_system_data/shunts' not in f:
            return {}
        
        shunt_group = f['detailed_system_data/shunts']
        return {
            'names': self._safe_load_dataset(shunt_group, 'names', []),
            'buses': self._safe_load_dataset(shunt_group, 'buses', []),
            'reactive_power_MVAR': self._safe_load_dataset(shunt_group, 'reactive_power_MVAR', []),
            'in_service': self._safe_load_dataset(shunt_group, 'in_service', []),
        }
    
    def _load_y_matrix(self, f: h5py.File) -> Optional[np.ndarray]:
        """Load Y-matrix if available"""
        try:
            if 'y_matrix' in f and 'matrix_data' in f['y_matrix']:
                return f['y_matrix']['matrix_data'][()]
        except Exception as e:
            print(f"   Error loading Y-matrix: {e}")
        return None


def test_enhanced_loader():
    """Test the enhanced data loader"""
    h5_file = "data/scenario_0.h5"
    
    if not os.path.exists(h5_file):
        print(f"H5 file not found: {h5_file}")
        return
    
    print("Testing Enhanced H5 Data Loader")
    print("=" * 50)
    
    loader = EnhancedH5DataLoader(h5_file)
    data = loader.load_complete_system_data()
    
    print(f"\nData Loading Results:")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"   - {key}: {len(value)} fields")
            if 'names' in value:
                print(f"     -> {len(value['names'])} items")
        elif value is not None:
            print(f"   - {key}: Available")
        else:
            print(f"   - {key}: None")
    
    # Check power injection data specifically
    buses = data.get('buses', {})
    if buses.get('active_injection_MW') is not None:
        P_inj = buses['active_injection_MW']
        Q_inj = buses['reactive_injection_MVAR']
        print(f"\nPower Injection Validation:")
        print(f"   - P injection range: {np.min(P_inj):.2f} to {np.max(P_inj):.2f} MW")
        print(f"   - Q injection range: {np.min(Q_inj):.2f} to {np.max(Q_inj):.2f} MVAR")
        print(f"   - Non-zero P injections: {np.count_nonzero(P_inj)}/{len(P_inj)}")
        print(f"   - System P balance: {np.sum(P_inj):.2f} MW")
        
        if np.all(P_inj == 0):
            print(f"   WARNING: All power injections are zero - need actual PowerFactory data")
        else:
            print(f"   SUCCESS: Power injections look reasonable")
    
    return data


if __name__ == "__main__":
    test_enhanced_loader()