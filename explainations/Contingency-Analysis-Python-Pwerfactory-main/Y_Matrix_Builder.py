# Y_Matrix_Builder.py - 2025-08-02
"""
Y-Matrix Builder Module for Post-Contingency Analysis
Compatible with existing H5 file structure from Contingency_Executor.py

This module:
1. Reads existing H5 files from contingency analysis
2. Extracts impedance data for all network elements
3. Builds the bus admittance matrix (Y-matrix) post-contingency
4. Stores Y-matrix results in the same H5 file structure
5. Updates analysis module status

Features:
- Reads impedance data from H5 files (no PowerFactory connection needed)
- Handles lines, transformers, generators, loads, and shunts
- Constructs full Y-matrix with proper bus indexing
- Calculates matrix properties (eigenvalues, condition number)
- Stores results in HDF5 format for efficient access
"""

import sys, os, h5py, numpy as np
import time
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import warnings

# ── Configuration ──────────────────────────────────────────────────────────
H5_DIR = os.path.join(os.getcwd(), "contingency_scenarios")
SBASE_MVA = 100.0  # Base power for per-unit calculations

# Y-matrix calculation parameters
SMALL_IMPEDANCE_THRESHOLD = 1e-8  # Minimum impedance to avoid numerical issues
DEFAULT_GENERATOR_R_PU = 0.01     # Default generator resistance in p.u.
DEFAULT_LOAD_PF = 0.9             # Default load power factor for impedance calculation

print(f"🔧 Y-MATRIX BUILDER MODULE")
print("="*60)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 H5 Directory: {H5_DIR}")
print(f"⚙️ Base Power: {SBASE_MVA} MVA")
print()

# Helper functions
def safe_complex_division(numerator, denominator, default=0.0):
    """Safely perform complex division with fallback"""
    if abs(denominator) < SMALL_IMPEDANCE_THRESHOLD:
        return complex(default, default)
    return numerator / denominator

def impedance_to_admittance(R_ohm, X_ohm):
    """Convert resistance and reactance to admittance"""
    Z = complex(R_ohm, X_ohm)
    if abs(Z) < SMALL_IMPEDANCE_THRESHOLD:
        return complex(0, 0)
    return 1.0 / Z

def read_scenario_h5_data(h5_path):
    """Read all necessary data from H5 file for Y-matrix construction"""
    
    print(f"      📖 Reading H5 data from: {os.path.basename(h5_path)}")
    
    scenario_data = {
        'buses': {},
        'lines': {},
        'transformers': {},
        'generators': {},
        'loads': {},
        'shunts': {},
        'scenario_info': {}
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Read scenario metadata
            if 'scenario_metadata' in f:
                meta = f['scenario_metadata']
                scenario_data['scenario_info'] = {
                    'scenario_id': meta['scenario_id'][()],
                    'contingency_type': meta['contingency_type'][()].decode(),
                    'description': meta['description'][()].decode()
                }
            
            # Read detailed system data (this contains the impedance information)
            if 'detailed_system_data' in f:
                detailed = f['detailed_system_data']
                
                if 'error' in detailed:
                    print(f"         ❌ Detailed system data has error: {detailed['error'][()].decode()}")
                    return None
                
                # Read bus data
                if 'buses' in detailed:
                    buses = detailed['buses']
                    bus_names = [name.decode() if isinstance(name, bytes) else name 
                               for name in buses['names'][:]]
                    
                    scenario_data['buses'] = {
                        'names': bus_names,
                        'voltages_pu': buses['voltages_pu'][:],
                        'voltage_angles_deg': buses['voltage_angles_deg'][:],
                        'base_voltages_kV': buses['base_voltages_kV'][:],
                        'in_service': buses['in_service'][:]
                    }
                
                # Read line data with impedances
                if 'lines' in detailed:
                    lines = detailed['lines']
                    line_names = [name.decode() if isinstance(name, bytes) else name 
                                for name in lines['names'][:]]
                    from_buses = [bus.decode() if isinstance(bus, bytes) else bus 
                                for bus in lines['from_buses'][:]]
                    to_buses = [bus.decode() if isinstance(bus, bytes) else bus 
                              for bus in lines['to_buses'][:]]
                    
                    scenario_data['lines'] = {
                        'names': line_names,
                        'from_buses': from_buses,
                        'to_buses': to_buses,
                        'R_ohm': lines['R_ohm'][:],
                        'X_ohm': lines['X_ohm'][:],
                        'B_uS': lines['B_uS'][:],
                        'impedance_source': [src.decode() if isinstance(src, bytes) else src 
                                           for src in lines['impedance_source'][:]]
                    }
                
                # Read transformer data with impedances
                if 'transformers' in detailed:
                    transformers = detailed['transformers']
                    trafo_names = [name.decode() if isinstance(name, bytes) else name 
                                 for name in transformers['names'][:]]
                    from_buses = [bus.decode() if isinstance(bus, bytes) else bus 
                                for bus in transformers['from_buses'][:]]
                    to_buses = [bus.decode() if isinstance(bus, bytes) else bus 
                              for bus in transformers['to_buses'][:]]
                    
                    scenario_data['transformers'] = {
                        'names': trafo_names,
                        'from_buses': from_buses,
                        'to_buses': to_buses,
                        'R_ohm': transformers['R_ohm'][:],
                        'X_ohm': transformers['X_ohm'][:],
                        'tap_ratio': transformers['tap_ratio'][:],
                        'phase_shift_deg': transformers['phase_shift_deg'][:],
                        'impedance_source': [src.decode() if isinstance(src, bytes) else src 
                                           for src in transformers['impedance_source'][:]]
                    }
                
                # Read generator data with impedances
                if 'generators' in detailed:
                    generators = detailed['generators']
                    gen_names = [name.decode() if isinstance(name, bytes) else name 
                               for name in generators['names'][:]]
                    gen_buses = [bus.decode() if isinstance(bus, bytes) else bus 
                               for bus in generators['buses'][:]]
                    
                    scenario_data['generators'] = {
                        'names': gen_names,
                        'buses': gen_buses,
                        'active_power_MW': generators['active_power_MW'][:],
                        'reactive_power_MVAR': generators['reactive_power_MVAR'][:],
                        'xd_pu': generators['xd_pu'][:],
                        'xq_pu': generators['xq_pu'][:],
                        'S_rated_MVA': generators['S_rated_MVA'][:],
                        'V_rated_kV': generators['V_rated_kV'][:]
                    }
                
                # Read load data with impedances
                if 'loads' in detailed:
                    loads = detailed['loads']
                    load_names = [name.decode() if isinstance(name, bytes) else name 
                                for name in loads['names'][:]]
                    load_buses = [bus.decode() if isinstance(bus, bytes) else bus 
                                for bus in loads['buses'][:]]
                    
                    scenario_data['loads'] = {
                        'names': load_names,
                        'buses': load_buses,
                        'active_power_MW': loads['active_power_MW'][:],
                        'reactive_power_MVAR': loads['reactive_power_MVAR'][:],
                        'Z_equivalent_ohm': loads['Z_equivalent_ohm'][:],
                        'R_equivalent_ohm': loads['R_equivalent_ohm'][:],
                        'X_equivalent_ohm': loads['X_equivalent_ohm'][:],
                        'bus_voltage_pu': loads['bus_voltage_pu'][:]
                    }
                
                # Read shunt data with impedances
                if 'shunts' in detailed:
                    shunts = detailed['shunts']
                    shunt_names = [name.decode() if isinstance(name, bytes) else name 
                                 for name in shunts['names'][:]]
                    shunt_buses = [bus.decode() if isinstance(bus, bytes) else bus 
                                 for bus in shunts['buses'][:]]
                    
                    scenario_data['shunts'] = {
                        'names': shunt_names,
                        'buses': shunt_buses,
                        'reactive_power_MVAR': shunts['reactive_power_MVAR'][:],
                        'X_shunt_ohm': shunts['X_shunt_ohm'][:],
                        'capacitive': shunts['capacitive'][:]
                    }
            
            else:
                print(f"         ❌ No detailed system data found in H5 file")
                return None
            
        print(f"         ✅ Successfully read H5 data")
        print(f"         📊 Buses: {len(scenario_data['buses']['names'])}")
        print(f"         📊 Lines: {len(scenario_data['lines']['names'])}")
        print(f"         📊 Transformers: {len(scenario_data['transformers']['names'])}")
        print(f"         📊 Generators: {len(scenario_data['generators']['names'])}")
        print(f"         📊 Loads: {len(scenario_data['loads']['names'])}")
        print(f"         📊 Shunts: {len(scenario_data['shunts']['names'])}")
        
        return scenario_data
        
    except Exception as e:
        print(f"         ❌ Error reading H5 data: {e}")
        return None

def create_bus_index_mapping(bus_names):
    """Create mapping from bus names to matrix indices"""
    return {bus_name: i for i, bus_name in enumerate(bus_names)}

def build_y_matrix(scenario_data):
    """Build the Y-matrix from scenario impedance data"""
    
    print(f"      🔧 Building Y-matrix...")
    
    # Get bus information
    bus_names = scenario_data['buses']['names']
    bus_voltages_pu = scenario_data['buses']['voltages_pu']
    bus_base_voltages_kV = scenario_data['buses']['base_voltages_kV']
    n_buses = len(bus_names)
    
    if n_buses == 0:
        print(f"         ❌ No buses found in scenario data")
        return None
    
    # Create bus index mapping
    bus_index = create_bus_index_mapping(bus_names)
    
    # Initialize Y-matrix as complex sparse matrix
    Y_matrix = sp.lil_matrix((n_buses, n_buses), dtype=complex)
    
    print(f"         📊 Matrix size: {n_buses} x {n_buses}")
    
    # Track elements added for debugging
    elements_added = {
        'lines': 0,
        'transformers': 0,
        'generators': 0,
        'loads': 0,
        'shunts': 0,
        'skipped': 0
    }
    
    # Add line admittances
    print(f"         ⚡ Processing lines...")
    lines = scenario_data['lines']
    for i, line_name in enumerate(lines['names']):
        from_bus = lines['from_buses'][i]
        to_bus = lines['to_buses'][i]
        R_ohm = lines['R_ohm'][i]
        X_ohm = lines['X_ohm'][i]
        B_uS = lines['B_uS'][i]
        
        # Skip if buses not found or impedance invalid
        if from_bus not in bus_index or to_bus not in bus_index:
            elements_added['skipped'] += 1
            continue
            
        if np.isnan(R_ohm) or np.isnan(X_ohm):
            elements_added['skipped'] += 1
            continue
        
        # Calculate series admittance
        Y_series = impedance_to_admittance(R_ohm, X_ohm)
        
        # Calculate shunt admittance (charging)
        Y_shunt = complex(0, B_uS * 1e-6) if not np.isnan(B_uS) else complex(0, 0)
        
        # Get bus indices
        from_idx = bus_index[from_bus]
        to_idx = bus_index[to_bus]
        
        # Add to Y-matrix
        # Off-diagonal terms (negative series admittance)
        Y_matrix[from_idx, to_idx] -= Y_series
        Y_matrix[to_idx, from_idx] -= Y_series
        
        # Diagonal terms (positive series + half shunt)
        Y_matrix[from_idx, from_idx] += Y_series + Y_shunt / 2
        Y_matrix[to_idx, to_idx] += Y_series + Y_shunt / 2
        
        elements_added['lines'] += 1
    
    # Add transformer admittances
    print(f"         🔄 Processing transformers...")
    transformers = scenario_data['transformers']
    for i, trafo_name in enumerate(transformers['names']):
        from_bus = transformers['from_buses'][i]
        to_bus = transformers['to_buses'][i]
        R_ohm = transformers['R_ohm'][i]
        X_ohm = transformers['X_ohm'][i]
        tap_ratio = transformers['tap_ratio'][i]
        phase_shift_deg = transformers['phase_shift_deg'][i]
        
        # Skip if buses not found or impedance invalid
        if from_bus not in bus_index or to_bus not in bus_index:
            elements_added['skipped'] += 1
            continue
            
        if np.isnan(R_ohm) or np.isnan(X_ohm):
            elements_added['skipped'] += 1
            continue
        
        # Calculate transformer admittance
        Z_trafo = complex(R_ohm, X_ohm)
        Y_trafo = safe_complex_division(1.0, Z_trafo)
        
        # Handle tap ratio and phase shift
        if np.isnan(tap_ratio):
            tap_ratio = 1.0
        if np.isnan(phase_shift_deg):
            phase_shift_deg = 0.0
        
        # Complex tap ratio with phase shift
        tap_complex = tap_ratio * np.exp(1j * np.radians(phase_shift_deg))
        
        # Get bus indices
        from_idx = bus_index[from_bus]
        to_idx = bus_index[to_bus]
        
        # Transformer admittance matrix elements
        Y11 = Y_trafo / (tap_complex * np.conj(tap_complex))  # From side
        Y12 = -Y_trafo / np.conj(tap_complex)                 # Off-diagonal
        Y21 = -Y_trafo / tap_complex                          # Off-diagonal
        Y22 = Y_trafo                                         # To side
        
        # Add to Y-matrix
        Y_matrix[from_idx, from_idx] += Y11
        Y_matrix[from_idx, to_idx] += Y12
        Y_matrix[to_idx, from_idx] += Y21
        Y_matrix[to_idx, to_idx] += Y22
        
        elements_added['transformers'] += 1
    
    # Add generator admittances (diagonal elements only)
    print(f"         🔋 Processing generators...")
    generators = scenario_data['generators']
    for i, gen_name in enumerate(generators['names']):
        gen_bus = generators['buses'][i]
        xd_pu = generators['xd_pu'][i]
        S_rated_MVA = generators['S_rated_MVA'][i]
        V_rated_kV = generators['V_rated_kV'][i]
        P_MW = generators['active_power_MW'][i]
        Q_MVAR = generators['reactive_power_MVAR'][i]
        
        # Skip if bus not found
        if gen_bus not in bus_index:
            elements_added['skipped'] += 1
            continue
        
        # Skip if xd_pu is invalid
        if np.isnan(xd_pu) or xd_pu <= 0:
            elements_added['skipped'] += 1
            continue
        
        # Method 1: Use S_rated and V_rated if available
        if not (np.isnan(S_rated_MVA) or np.isnan(V_rated_kV)) and S_rated_MVA > 0 and V_rated_kV > 0:
            # Standard method with rated values
            Z_base_gen = (V_rated_kV ** 2) / S_rated_MVA
            R_gen_ohm = DEFAULT_GENERATOR_R_PU * Z_base_gen
            X_gen_ohm = xd_pu * Z_base_gen
            
        else:
            # Method 2: Estimate using generator output and bus voltage (FALLBACK)
            gen_idx = bus_index[gen_bus]
            V_bus_pu = bus_voltages_pu[gen_idx]
            V_base_kV = bus_base_voltages_kV[gen_idx]
            
            if np.isnan(V_bus_pu) or np.isnan(V_base_kV) or V_bus_pu <= 0.1 or V_base_kV <= 0:
                elements_added['skipped'] += 1
                continue
            
            # Estimate generator MVA base from actual output (with safety factor)
            S_apparent_MVA = np.sqrt(P_MW**2 + Q_MVAR**2)
            if S_apparent_MVA < 1.0:  # Very small or zero output
                S_apparent_MVA = 100.0  # Default 100 MVA for small generators
            
            # Use actual bus voltage as base (typical for generators)
            V_actual_kV = V_bus_pu * V_base_kV
            
            # Estimate impedance base using apparent power and bus voltage
            S_base_estimate = max(S_apparent_MVA * 1.2, 50.0)  # 20% margin, min 50 MVA
            Z_base_gen = (V_actual_kV ** 2) / S_base_estimate
            
            R_gen_ohm = DEFAULT_GENERATOR_R_PU * Z_base_gen
            X_gen_ohm = xd_pu * Z_base_gen
            
            print(f"            ⚡ {gen_name}: Using estimated base (S≈{S_base_estimate:.0f}MVA, V≈{V_actual_kV:.1f}kV)")
        
        # Generator admittance
        Y_gen = impedance_to_admittance(R_gen_ohm, X_gen_ohm)
        
        # Add to diagonal (generator is connected to ground through its impedance)
        gen_idx = bus_index[gen_bus]
        Y_matrix[gen_idx, gen_idx] += Y_gen
        
        elements_added['generators'] += 1
    
    # Add load admittances (diagonal elements only)
    print(f"         🏠 Processing loads...")
    loads = scenario_data['loads']
    for i, load_name in enumerate(loads['names']):
        load_bus = loads['buses'][i]
        P_MW = loads['active_power_MW'][i]
        Q_MVAR = loads['reactive_power_MVAR'][i]
        V_bus_pu = loads['bus_voltage_pu'][i]
        
        # Try to use pre-calculated equivalent impedance
        R_eq_ohm = loads['R_equivalent_ohm'][i]
        X_eq_ohm = loads['X_equivalent_ohm'][i]
        
        # Skip if bus not found
        if load_bus not in bus_index:
            elements_added['skipped'] += 1
            continue
        
        # Use pre-calculated impedance if available
        if not (np.isnan(R_eq_ohm) or np.isnan(X_eq_ohm)):
            Y_load = impedance_to_admittance(R_eq_ohm, X_eq_ohm)
        else:
            # Calculate from P, Q, and V if pre-calculated values not available
            if np.isnan(P_MW) or np.isnan(Q_MVAR) or np.isnan(V_bus_pu):
                elements_added['skipped'] += 1
                continue
                
            if abs(P_MW) < 1e-6 and abs(Q_MVAR) < 1e-6:  # No load
                elements_added['skipped'] += 1
                continue
            
            # Get bus base voltage
            load_idx = bus_index[load_bus]
            V_base_kV = bus_base_voltages_kV[load_idx]
            
            if np.isnan(V_base_kV) or V_base_kV <= 0:
                elements_added['skipped'] += 1
                continue
            
            # Calculate actual voltage
            V_actual_kV = V_bus_pu * V_base_kV
            
            # Calculate equivalent impedance: Z = V²/(P - jQ)
            S_complex = complex(P_MW, -Q_MVAR)  # Negative Q for load convention
            Z_load = (V_actual_kV ** 2) / S_complex
            
            Y_load = safe_complex_division(1.0, Z_load)
        
        # Add to diagonal
        load_idx = bus_index[load_bus]
        Y_matrix[load_idx, load_idx] += Y_load
        
        elements_added['loads'] += 1
    
    # Add shunt admittances (diagonal elements only)
    print(f"         🔧 Processing shunts...")
    shunts = scenario_data['shunts']
    for i, shunt_name in enumerate(shunts['names']):
        shunt_bus = shunts['buses'][i]
        Q_MVAR = shunts['reactive_power_MVAR'][i]
        X_shunt_ohm = shunts['X_shunt_ohm'][i]
        is_capacitive = shunts['capacitive'][i]
        
        # Skip if bus not found
        if shunt_bus not in bus_index:
            elements_added['skipped'] += 1
            continue
        
        # Method 1: Use pre-calculated X_shunt_ohm if available
        if not np.isnan(X_shunt_ohm) and abs(X_shunt_ohm) >= SMALL_IMPEDANCE_THRESHOLD:
            # Use pre-calculated shunt reactance
            if is_capacitive:
                Y_shunt = complex(0, -1.0 / X_shunt_ohm)  # Capacitive (negative susceptance)
            else:
                Y_shunt = complex(0, 1.0 / X_shunt_ohm)   # Inductive (positive susceptance)
        
        # Method 2: Calculate from Q_MVAR and bus voltage (FALLBACK)
        elif not np.isnan(Q_MVAR) and abs(Q_MVAR) > 0.001:
            shunt_idx = bus_index[shunt_bus]
            V_bus_pu = bus_voltages_pu[shunt_idx]
            V_base_kV = bus_base_voltages_kV[shunt_idx]
            
            if np.isnan(V_bus_pu) or np.isnan(V_base_kV) or V_bus_pu <= 0.1 or V_base_kV <= 0:
                elements_added['skipped'] += 1
                continue
            
            # Calculate shunt reactance from Q = V²/X
            V_actual_kV = V_bus_pu * V_base_kV
            X_shunt_calc = (V_actual_kV ** 2) / abs(Q_MVAR)
            
            # Shunt admittance (Q > 0 = inductive, Q < 0 = capacitive)
            if Q_MVAR > 0:  # Inductive (consuming reactive power)
                Y_shunt = complex(0, 1.0 / X_shunt_calc)
            else:  # Capacitive (supplying reactive power)
                Y_shunt = complex(0, -1.0 / X_shunt_calc)
            
            print(f"            🔧 {shunt_name}: Using Q-based calculation (Q={Q_MVAR:.1f}MVAR, X={X_shunt_calc:.1f}Ω)")
            
        else:
            # No valid data for shunt
            elements_added['skipped'] += 1
            continue
        
        # Add to diagonal
        shunt_idx = bus_index[shunt_bus]
        Y_matrix[shunt_idx, shunt_idx] += Y_shunt
        
        elements_added['shunts'] += 1
    
    # Convert to CSR format for efficiency
    Y_matrix = Y_matrix.tocsr()
    
    # Summary
    total_added = sum(elements_added[key] for key in elements_added if key != 'skipped')
    print(f"         ✅ Y-matrix construction complete")
    print(f"         📊 Elements added: {total_added} ({elements_added['lines']} lines, "
          f"{elements_added['transformers']} transformers, {elements_added['generators']} generators, "
          f"{elements_added['loads']} loads, {elements_added['shunts']} shunts)")
    print(f"         ⚠️ Elements skipped: {elements_added['skipped']}")
    
    # Calculate matrix properties
    print(f"         🔬 Calculating matrix properties...")
    matrix_properties = calculate_matrix_properties(Y_matrix, bus_names)
    
    y_matrix_data = {
        'Y_matrix': Y_matrix,
        'bus_names': bus_names,
        'bus_index_mapping': bus_index,
        'matrix_size': n_buses,
        'elements_added': elements_added,
        'matrix_properties': matrix_properties,
        'construction_timestamp': datetime.now().isoformat()
    }
    
    return y_matrix_data

def calculate_matrix_properties(Y_matrix, bus_names):
    """Calculate important properties of the Y-matrix"""
    
    n = Y_matrix.shape[0]
    
    properties = {
        'matrix_size': n,
        'nnz': Y_matrix.nnz,  # Number of non-zero elements
        'density': Y_matrix.nnz / (n * n),
        'is_symmetric': False,
        'is_hermitian': False,
        'condition_number': np.nan,
        'smallest_eigenvalue_magnitude': np.nan,
        'largest_eigenvalue_magnitude': np.nan,
        'rank_estimate': np.nan
    }
    
    try:
        # Check if matrix is symmetric (for real part) and Hermitian
        Y_dense = Y_matrix.toarray()
        
        # Check Hermitian property (Y = Y^H)
        Y_hermitian_diff = np.max(np.abs(Y_dense - Y_dense.conj().T))
        properties['is_hermitian'] = Y_hermitian_diff < 1e-10
        
        # Check symmetric property (for real part)
        Y_real = Y_dense.real
        Y_symmetric_diff = np.max(np.abs(Y_real - Y_real.T))
        properties['is_symmetric'] = Y_symmetric_diff < 1e-10
        
        # Calculate eigenvalues (only a few for large matrices)
        if n <= 1000:  # Full eigenvalue calculation for smaller matrices
            eigenvals = np.linalg.eigvals(Y_dense)
            eigenval_magnitudes = np.abs(eigenvals)
            
            properties['smallest_eigenvalue_magnitude'] = np.min(eigenval_magnitudes)
            properties['largest_eigenvalue_magnitude'] = np.max(eigenval_magnitudes)
            
            # Condition number
            nonzero_eigenvals = eigenval_magnitudes[eigenval_magnitudes > 1e-12]
            if len(nonzero_eigenvals) > 0:
                properties['condition_number'] = np.max(nonzero_eigenvals) / np.min(nonzero_eigenvals)
            
            # Rank estimate
            properties['rank_estimate'] = np.sum(eigenval_magnitudes > 1e-10)
            
        else:  # Use sparse eigenvalue calculation for larger matrices
            try:
                # Calculate a few largest and smallest eigenvalues
                k = min(6, n-2)  # Number of eigenvalues to compute
                
                # Largest eigenvalues
                eigenvals_large, _ = eigs(Y_matrix, k=k, which='LM')
                # Smallest eigenvalues  
                eigenvals_small, _ = eigs(Y_matrix, k=k, which='SM')
                
                eigenval_magnitudes = np.abs(np.concatenate([eigenvals_large, eigenvals_small]))
                
                properties['smallest_eigenvalue_magnitude'] = np.min(eigenval_magnitudes)
                properties['largest_eigenvalue_magnitude'] = np.max(eigenval_magnitudes)
                
                # Approximate condition number
                nonzero_eigenvals = eigenval_magnitudes[eigenval_magnitudes > 1e-12]
                if len(nonzero_eigenvals) > 0:
                    properties['condition_number'] = np.max(nonzero_eigenvals) / np.min(nonzero_eigenvals)
                    
            except Exception as e:
                print(f"            ⚠️ Sparse eigenvalue calculation failed: {e}")
        
        # Additional matrix diagnostics
        properties['max_diagonal_real'] = np.max(Y_dense.diagonal().real)
        properties['min_diagonal_real'] = np.min(Y_dense.diagonal().real)
        properties['max_diagonal_imag'] = np.max(Y_dense.diagonal().imag)
        properties['min_diagonal_imag'] = np.min(Y_dense.diagonal().imag)
        
        # Check for potential issues
        zero_diagonal_count = np.sum(np.abs(Y_dense.diagonal()) < 1e-12)
        properties['zero_diagonal_elements'] = zero_diagonal_count
        
        # Matrix norm
        properties['frobenius_norm'] = np.linalg.norm(Y_dense, 'fro')
        
    except Exception as e:
        print(f"            ⚠️ Error calculating matrix properties: {e}")
    
    return properties

def save_y_matrix_to_h5(h5_path, y_matrix_data, construction_time):
    """Save Y-matrix results to H5 file"""
    
    print(f"      💾 Saving Y-matrix to H5 file...")
    
    try:
        with h5py.File(h5_path, 'a') as f:
            # Remove existing y_matrix group if it exists
            if 'y_matrix' in f:
                del f['y_matrix']
            
            # Create Y-matrix group
            y_group = f.create_group('y_matrix')
            
            # Add metadata
            y_group.create_dataset('construction_timestamp', 
                                 data=y_matrix_data['construction_timestamp'].encode())
            y_group.create_dataset('construction_time_seconds', data=construction_time)
            y_group.create_dataset('matrix_size', data=y_matrix_data['matrix_size'])
            
            # Save bus information
            bus_group = y_group.create_group('bus_information')
            bus_group.create_dataset('bus_names', 
                                   data=[name.encode() for name in y_matrix_data['bus_names']])
            bus_group.create_dataset('num_buses', data=len(y_matrix_data['bus_names']))
            
            # Save Y-matrix in sparse format
            Y_matrix = y_matrix_data['Y_matrix']
            matrix_group = y_group.create_group('admittance_matrix')
            
            # Store sparse matrix components
            matrix_group.create_dataset('data_real', data=Y_matrix.data.real)
            matrix_group.create_dataset('data_imag', data=Y_matrix.data.imag)
            matrix_group.create_dataset('indices', data=Y_matrix.indices)
            matrix_group.create_dataset('indptr', data=Y_matrix.indptr)
            matrix_group.create_dataset('shape', data=Y_matrix.shape)
            matrix_group.create_dataset('format', data=b'csr')  # Compressed Sparse Row format
            
            # Also store as dense matrix for smaller systems (for easier access)
            if y_matrix_data['matrix_size'] <= 100:
                Y_dense = Y_matrix.toarray()
                matrix_group.create_dataset('dense_real', data=Y_dense.real)
                matrix_group.create_dataset('dense_imag', data=Y_dense.imag)
            
            # Save construction statistics
            stats_group = y_group.create_group('construction_statistics')
            elements_added = y_matrix_data['elements_added']
            for key, value in elements_added.items():
                stats_group.create_dataset(key, data=value)
            
            # Save matrix properties
            props_group = y_group.create_group('matrix_properties')
            properties = y_matrix_data['matrix_properties']
            for key, value in properties.items():
                if isinstance(value, (bool, np.bool_)):
                    props_group.create_dataset(key, data=value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    props_group.create_dataset(key, data=value)
                elif isinstance(value, complex):
                    props_group.create_dataset(f'{key}_real', data=value.real)
                    props_group.create_dataset(f'{key}_imag', data=value.imag)
                else:
                    # Handle other types as strings
                    props_group.create_dataset(key, data=str(value).encode())
            
            # Update analysis modules status
            if 'analysis_modules' in f:
                modules_group = f['analysis_modules']
                # Update Y-matrix status
                if 'y_matrix_building' in modules_group:
                    del modules_group['y_matrix_building']
                modules_group.create_dataset('y_matrix_building', data=b'completed')
                
                # Remove pending placeholder if it exists
                if 'y_matrix_pending' in modules_group:
                    del modules_group['y_matrix_pending']
            
            print(f"         ✅ Y-matrix saved successfully")
            print(f"         📊 Matrix size: {y_matrix_data['matrix_size']} x {y_matrix_data['matrix_size']}")
            print(f"         📊 Non-zero elements: {Y_matrix.nnz}")
            print(f"         📊 Matrix density: {properties.get('density', 0):.4f}")
            
    except Exception as e:
        print(f"         ❌ Error saving Y-matrix: {e}")

def analyze_scenario_y_matrix(h5_path):
    """Perform complete Y-matrix construction for a single scenario"""
    
    scenario_name = os.path.basename(h5_path)
    print(f"\n🎯 BUILDING Y-MATRIX FOR SCENARIO: {scenario_name}")
    
    start_time = time.time()
    
    # Step 1: Read scenario data from H5 file
    print(f"   📖 Reading scenario data...")
    scenario_data = read_scenario_h5_data(h5_path)
    if not scenario_data:
        print(f"   ❌ Failed to read scenario data")
        return False
    
    scenario_info = scenario_data['scenario_info']
    print(f"      📋 Scenario {scenario_info['scenario_id']}: {scenario_info['contingency_type']}")
    print(f"      📋 Description: {scenario_info['description']}")
    
    # Step 2: Build Y-matrix from impedance data
    print(f"   🔧 Constructing Y-matrix...")
    y_matrix_data = build_y_matrix(scenario_data)
    
    if not y_matrix_data:
        print(f"   ❌ Failed to build Y-matrix")
        return False
    
    # Step 3: Save Y-matrix to H5 file
    construction_time = time.time() - start_time
    save_y_matrix_to_h5(h5_path, y_matrix_data, construction_time)
    
    # Step 4: Summary
    properties = y_matrix_data['matrix_properties']
    elements_added = y_matrix_data['elements_added']
    
    print(f"   ✅ Y-matrix construction completed")
    print(f"      📊 Matrix size: {y_matrix_data['matrix_size']} x {y_matrix_data['matrix_size']}")
    print(f"      📊 Elements: {elements_added['lines']}L + {elements_added['transformers']}T + "
          f"{elements_added['generators']}G + {elements_added['loads']}Ld + {elements_added['shunts']}S")
    print(f"      📊 Non-zero elements: {properties.get('nnz', 0)}")
    print(f"      📊 Density: {properties.get('density', 0):.4f}")
    print(f"      📊 Condition number: {properties.get('condition_number', np.nan):.2e}")
    print(f"      ⏱️ Construction time: {construction_time:.2f} seconds")
    
    # Warnings for potential issues
    if properties.get('zero_diagonal_elements', 0) > 0:
        print(f"      ⚠️ Warning: {properties['zero_diagonal_elements']} buses have zero diagonal elements")
    
    if properties.get('condition_number', 0) > 1e12:
        print(f"      ⚠️ Warning: Matrix is ill-conditioned (condition number > 1e12)")
    
    return True

def validate_y_matrix(y_matrix_data):
    """Validate the constructed Y-matrix for common issues"""
    
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'properties': {}
    }
    
    try:
        Y_matrix = y_matrix_data['Y_matrix']
        properties = y_matrix_data['matrix_properties']
        
        # Check matrix size
        if y_matrix_data['matrix_size'] <= 0:
            validation_results['errors'].append("Matrix has zero or negative size")
            validation_results['is_valid'] = False
        
        # Check for zero diagonal elements (islanded buses)
        zero_diag = properties.get('zero_diagonal_elements', 0)
        if zero_diag > 0:
            validation_results['warnings'].append(f"{zero_diag} buses have zero diagonal elements (possible islanding)")
        
        # Check condition number
        cond_num = properties.get('condition_number', np.nan)
        if not np.isnan(cond_num):
            if cond_num > 1e12:
                validation_results['warnings'].append(f"Matrix is ill-conditioned (κ = {cond_num:.2e})")
            elif cond_num > 1e8:
                validation_results['warnings'].append(f"Matrix conditioning is poor (κ = {cond_num:.2e})")
        
        # Check matrix density
        density = properties.get('density', 0)
        if density < 0.01:
            validation_results['warnings'].append(f"Matrix is very sparse (density = {density:.4f})")
        elif density > 0.5:
            validation_results['warnings'].append(f"Matrix is unusually dense (density = {density:.4f})")
        
        # Check if matrix is Hermitian (should be for admittance matrices)
        if not properties.get('is_hermitian', False):
            validation_results['warnings'].append("Matrix is not Hermitian (asymmetric admittances)")
        
        # Check eigenvalue properties
        min_eig = properties.get('smallest_eigenvalue_magnitude', np.nan)
        if not np.isnan(min_eig) and min_eig < 1e-10:
            validation_results['warnings'].append(f"Matrix has very small eigenvalues (min = {min_eig:.2e})")
        
        # Summary properties
        validation_results['properties'] = {
            'matrix_size': y_matrix_data['matrix_size'],
            'nnz': properties.get('nnz', 0),
            'density': density,
            'condition_number': cond_num,
            'is_hermitian': properties.get('is_hermitian', False),
            'zero_diagonal_count': zero_diag
        }
        
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {e}")
        validation_results['is_valid'] = False
    
    return validation_results

def load_y_matrix_from_h5(h5_path):
    """Load Y-matrix from H5 file (utility function for other modules)"""
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'y_matrix' not in f:
                return None
            
            y_group = f['y_matrix']
            
            # Load bus information
            bus_names = [name.decode() if isinstance(name, bytes) else name 
                        for name in y_group['bus_information']['bus_names'][:]]
            
            # Load sparse matrix
            matrix_group = y_group['admittance_matrix']
            data_real = matrix_group['data_real'][:]
            data_imag = matrix_group['data_imag'][:]
            data_complex = data_real + 1j * data_imag
            indices = matrix_group['indices'][:]
            indptr = matrix_group['indptr'][:]
            shape = tuple(matrix_group['shape'][:])
            
            # Reconstruct sparse matrix
            Y_matrix = sp.csr_matrix((data_complex, indices, indptr), shape=shape)
            
            # Load metadata
            construction_time = y_group['construction_time_seconds'][()]
            timestamp = y_group['construction_timestamp'][()].decode()
            
            return {
                'Y_matrix': Y_matrix,
                'bus_names': bus_names,
                'construction_time': construction_time,
                'timestamp': timestamp
            }
            
    except Exception as e:
        print(f"Error loading Y-matrix from {h5_path}: {e}")
        return None

def main():
    """Main function to build Y-matrices for all scenarios"""
    
    # Find all H5 files
    h5_files = [os.path.join(H5_DIR, f) for f in os.listdir(H5_DIR) if f.endswith('.h5')]
    h5_files.sort()
    
    if not h5_files:
        print(f"❌ No H5 files found in {H5_DIR}")
        print(f"💡 Make sure you've run the contingency executor and data collector first")
        return
    
    print(f"📊 Found {len(h5_files)} H5 files: {', '.join(os.path.basename(f) for f in h5_files)}")
    
    successful_constructions = 0
    failed_constructions = 0
    total_construction_time = 0
    
    # Process each scenario
    for h5_path in h5_files:
        try:
            success = analyze_scenario_y_matrix(h5_path)
            if success:
                successful_constructions += 1
            else:
                failed_constructions += 1
        except Exception as e:
            print(f"\n❌ Error building Y-matrix for {os.path.basename(h5_path)}: {e}")
            failed_constructions += 1
    
    # Final summary
    print(f"\n🎉 Y-MATRIX CONSTRUCTION COMPLETE!")
    print("="*60)
    print(f"📊 FINAL SUMMARY:")
    print(f"   🎯 Total scenarios processed: {len(h5_files)}")
    print(f"   ✅ Successful constructions: {successful_constructions}")
    print(f"   ❌ Failed constructions: {failed_constructions}")
    print(f"   📈 Success rate: {successful_constructions/len(h5_files)*100:.1f}%")
    print(f"   📁 H5 files updated: {successful_constructions}")
    
    print(f"\n🔄 MODULAR CONTINGENCY ANALYSIS STATUS:")
    print(f"   1. ✅ Module 1: Contingency Generator - Complete")
    print(f"   2. ✅ Module 2: Contingency Executor - Complete")
    print(f"   3. ✅ Module 3: Load Flow Data Collector - Complete")
    print(f"   4. ✅ Module 4: Voltage Sensitivity Analysis - Complete")
    print(f"   5. ✅ Module 5: Y-Matrix Builder - Complete")
    
    print(f"\n💡 Each H5 file now contains:")
    print(f"   • Scenario metadata and disconnection actions")
    print(f"   • Load flow results (voltages, angles, convergence)")
    print(f"   • Comprehensive power flow data (MW/MVAR for all elements)")
    print(f"   • Detailed system data with impedances")
    print(f"   • Voltage sensitivity analysis results")
    print(f"   • ✨ Y-matrix (admittance matrix) with properties")
    
    if successful_constructions > 0:
        print(f"\n📋 Y-MATRIX DATA STRUCTURE:")
        print(f"   📁 y_matrix/")
        print(f"      📊 construction_metadata (timestamp, time, matrix_size)")
        print(f"      📁 bus_information/")
        print(f"         📊 bus_names, num_buses")
        print(f"      📁 admittance_matrix/")
        print(f"         📊 sparse format (data_real, data_imag, indices, indptr)")
        print(f"         📊 dense format (for matrices ≤ 100x100)")
        print(f"      📁 construction_statistics/")
        print(f"         📊 elements_added (lines, transformers, generators, loads, shunts)")
        print(f"      📁 matrix_properties/")
        print(f"         📊 size, density, condition_number, eigenvalue_info")
        print(f"         📊 symmetry_properties, numerical_diagnostics")

def create_y_matrix_summary_report():
    """Create a summary report of Y-matrix construction results"""
    
    summary_file = os.path.join(H5_DIR, "y_matrix_construction_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("Y-MATRIX CONSTRUCTION SUMMARY REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Module: Y-Matrix Builder\n")
        f.write(f"Base Power: {SBASE_MVA} MVA\n\n")
        
        # Analyze each H5 file
        h5_files = [os.path.join(H5_DIR, f) for f in os.listdir(H5_DIR) if f.endswith('.h5')]
        h5_files.sort()
        
        f.write(f"Scenarios Analyzed: {len(h5_files)}\n")
        f.write("-" * 40 + "\n")
        
        total_y_matrices = 0
        total_avg_density = 0
        total_avg_condition = 0
        valid_condition_count = 0
        
        for h5_path in h5_files:
            h5_file = os.path.basename(h5_path)
            
            try:
                with h5py.File(h5_path, 'r') as hf:
                    scenario_id = hf['scenario_metadata']['scenario_id'][()]
                    contingency_type = hf['scenario_metadata']['contingency_type'][()].decode()
                    description = hf['scenario_metadata']['description'][()].decode()
                    
                    f.write(f"\n{h5_file}:\n")
                    f.write(f"  Scenario ID: {scenario_id}\n")
                    f.write(f"  Type: {contingency_type}\n")
                    f.write(f"  Description: {description}\n")
                    
                    if 'y_matrix' in hf:
                        total_y_matrices += 1
                        y_matrix = hf['y_matrix']
                        
                        construction_time = y_matrix['construction_time_seconds'][()]
                        matrix_size = y_matrix['matrix_size'][()]
                        
                        f.write(f"  Y-Matrix Status: ✅ Completed\n")
                        f.write(f"  Construction Time: {construction_time:.2f} seconds\n")
                        f.write(f"  Matrix Size: {matrix_size} x {matrix_size}\n")
                        
                        # Matrix properties
                        if 'matrix_properties' in y_matrix:
                            props = y_matrix['matrix_properties']
                            
                            nnz = props['nnz'][()]
                            density = props['density'][()]
                            total_avg_density += density
                            
                            f.write(f"  Non-zero Elements: {nnz}\n")
                            f.write(f"  Density: {density:.4f}\n")
                            
                            if 'condition_number' in props:
                                cond_num = props['condition_number'][()]
                                if not np.isnan(cond_num):
                                    f.write(f"  Condition Number: {cond_num:.2e}\n")
                                    total_avg_condition += cond_num
                                    valid_condition_count += 1
                                else:
                                    f.write(f"  Condition Number: Not calculated\n")
                            
                            if 'is_hermitian' in props:
                                is_hermitian = props['is_hermitian'][()]
                                f.write(f"  Hermitian: {is_hermitian}\n")
                            
                            if 'zero_diagonal_elements' in props:
                                zero_diag = props['zero_diagonal_elements'][()]
                                if zero_diag > 0:
                                    f.write(f"  ⚠️ Zero diagonal elements: {zero_diag}\n")
                        
                        # Construction statistics
                        if 'construction_statistics' in y_matrix:
                            stats = y_matrix['construction_statistics']
                            lines = stats['lines'][()]
                            transformers = stats['transformers'][()]
                            generators = stats['generators'][()]
                            loads = stats['loads'][()]
                            shunts = stats['shunts'][()]
                            skipped = stats['skipped'][()]
                            
                            f.write(f"  Elements Added: {lines}L + {transformers}T + {generators}G + {loads}Ld + {shunts}S\n")
                            if skipped > 0:
                                f.write(f"  Elements Skipped: {skipped}\n")
                    else:
                        f.write(f"  Y-Matrix Status: ❌ Not completed\n")
                        
            except Exception as e:
                f.write(f"\n{h5_file}: Error reading file - {e}\n")
        
        # Overall statistics
        f.write(f"\nOverall Statistics:\n")
        f.write(f"-" * 20 + "\n")
        f.write(f"Total Y-matrices constructed: {total_y_matrices}/{len(h5_files)}\n")
        f.write(f"Success rate: {total_y_matrices/len(h5_files)*100:.1f}%\n")
        
        if total_y_matrices > 0:
            avg_density = total_avg_density / total_y_matrices
            f.write(f"Average matrix density: {avg_density:.4f}\n")
            
            if valid_condition_count > 0:
                avg_condition = total_avg_condition / valid_condition_count
                f.write(f"Average condition number: {avg_condition:.2e}\n")
        
        f.write(f"\nY-Matrix Construction Parameters:\n")
        f.write(f"- Small impedance threshold: {SMALL_IMPEDANCE_THRESHOLD}\n")
        f.write(f"- Default generator resistance: {DEFAULT_GENERATOR_R_PU} p.u.\n")
        f.write(f"- Default load power factor: {DEFAULT_LOAD_PF}\n")
        
        f.write(f"\nY-Matrix Elements:\n")
        f.write(f"- Lines: Series admittance + charging susceptance\n")
        f.write(f"- Transformers: Series admittance with tap ratio and phase shift\n")
        f.write(f"- Generators: Diagonal admittance based on xd reactance\n")
        f.write(f"- Loads: Diagonal admittance based on equivalent impedance\n")
        f.write(f"- Shunts: Diagonal susceptance (capacitive/inductive)\n")
    
    print(f"📄 Y-matrix summary report saved: {os.path.basename(summary_file)}")

def test_y_matrix_construction():
    """Test function for standalone module testing"""
    
    print(f"🧪 TESTING Y-MATRIX CONSTRUCTION MODULE")
    print("="*60)
    
    # Find a test H5 file
    h5_files = [os.path.join(H5_DIR, f) for f in os.listdir(H5_DIR) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"❌ No H5 files found for testing")
        return False
    
    # Test with first file
    test_h5_path = h5_files[0]
    print(f"🔬 Testing with: {os.path.basename(test_h5_path)}")
    
    # Test data reading
    scenario_data = read_scenario_h5_data(test_h5_path)
    if not scenario_data:
        print(f"❌ Failed to read test data")
        return False
    
    print(f"✅ Data reading successful")
    
    # Test Y-matrix construction
    y_matrix_data = build_y_matrix(scenario_data)
    if not y_matrix_data:
        print(f"❌ Failed to build Y-matrix")
        return False
    
    print(f"✅ Y-matrix construction successful")
    
    # Test validation
    validation = validate_y_matrix(y_matrix_data)
    print(f"✅ Y-matrix validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    
    if validation['warnings']:
        print(f"⚠️ Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings'][:3]:  # Show first 3 warnings
            print(f"   - {warning}")
    
    # Test loading (if saved)
    try:
        save_y_matrix_to_h5(test_h5_path, y_matrix_data, 1.0)
        loaded_data = load_y_matrix_from_h5(test_h5_path)
        if loaded_data:
            print(f"✅ Y-matrix save/load successful")
        else:
            print(f"❌ Y-matrix save/load failed")
    except Exception as e:
        print(f"❌ Y-matrix save/load error: {e}")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Y-Matrix Builder for Contingency Analysis')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--single', type=str, help='Process single H5 file')
    args = parser.parse_args()
    
    if args.test:
        test_y_matrix_construction()
    elif args.single:
        h5_path = os.path.join(H5_DIR, args.single)
        if os.path.exists(h5_path):
            analyze_scenario_y_matrix(h5_path)
        else:
            print(f"❌ File not found: {h5_path}")
    else:
        main()
        create_y_matrix_summary_report()