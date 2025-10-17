"""
Simple Load Flow Solver for Diagnosis

Let's create the most basic single-phase solver to identify where the NaN values come from.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from data.h5_loader import H5DataLoader


def create_simple_solver():
    """Create a very simple solver for diagnosis"""
    print("🔬 Simple Load Flow Diagnosis")
    print("=" * 40)
    
    # Load data
    loader = H5DataLoader('data/scenario_0.h5')
    data = loader.load_all_data()
    
    # Basic system setup
    bus_names = data['buses']['names']
    n_buses = len(bus_names)
    bus_to_idx = {name: i for i, name in enumerate(bus_names)}
    
    print(f"System: {n_buses} buses")
    
    # Build Y matrix step by step
    print("\\n1️⃣ Building Y-matrix...")
    Y = np.zeros((n_buses, n_buses), dtype=complex)
    
    # Add lines
    line_count = 0
    for i, line_name in enumerate(data['lines']['names']):
        from_bus = data['lines']['from_buses'][i]
        to_bus = data['lines']['to_buses'][i]
        
        if from_bus in bus_to_idx and to_bus in bus_to_idx:
            from_idx = bus_to_idx[from_bus]
            to_idx = bus_to_idx[to_bus]
            
            R = data['lines']['R_ohm'][i]
            X = data['lines']['X_ohm'][i]
            Z = R + 1j * X
            Y_line = 1 / Z
            
            # Add to Y matrix
            Y[from_idx, to_idx] -= Y_line
            Y[to_idx, from_idx] -= Y_line
            Y[from_idx, from_idx] += Y_line
            Y[to_idx, to_idx] += Y_line
            
            line_count += 1
    
    # Add transformers
    xfmr_count = 0
    for i, xfmr_name in enumerate(data['transformers']['names']):
        from_bus = data['transformers']['from_buses'][i]
        to_bus = data['transformers']['to_buses'][i]
        
        if from_bus in bus_to_idx and to_bus in bus_to_idx:
            from_idx = bus_to_idx[from_bus]
            to_idx = bus_to_idx[to_bus]
            
            R = data['transformers']['R_ohm'][i]
            X = data['transformers']['X_ohm'][i]
            Z = R + 1j * X
            Y_xfmr = 1 / Z
            
            Y[from_idx, to_idx] -= Y_xfmr
            Y[to_idx, from_idx] -= Y_xfmr
            Y[from_idx, from_idx] += Y_xfmr
            Y[to_idx, to_idx] += Y_xfmr
            
            xfmr_count += 1
    
    print(f"   Added {line_count} lines, {xfmr_count} transformers")
    
    # Check Y matrix properties
    y_diag = np.diag(Y)
    y_condition = np.linalg.cond(Y)
    y_rank = np.linalg.matrix_rank(Y)
    
    print(f"   Y-matrix diagonal range: {np.abs(y_diag).min():.6f} - {np.abs(y_diag).max():.6f}")
    print(f"   Y-matrix condition number: {y_condition:.2e}")
    print(f"   Y-matrix rank: {y_rank} / {n_buses}")
    
    # Check for NaN or infinite values
    y_nan_count = np.sum(np.isnan(Y))
    y_inf_count = np.sum(np.isinf(Y))
    print(f"   NaN elements: {y_nan_count}, Inf elements: {y_inf_count}")
    
    if y_nan_count > 0 or y_inf_count > 0:
        print("   ❌ Y-matrix has NaN/Inf values!")
        return False
    
    # Simple power injection test
    print("\\n2️⃣ Testing power injections...")
    P_inj = np.zeros(n_buses)
    Q_inj = np.zeros(n_buses)
    
    # Add generation
    for i, gen_name in enumerate(data['generators']['names']):
        bus_name = data['generators']['buses'][i]
        if bus_name in bus_to_idx:
            bus_idx = bus_to_idx[bus_name]
            P_inj[bus_idx] += data['generators']['active_power_MW'][i] / 100.0
            Q_inj[bus_idx] += data['generators']['reactive_power_MVAR'][i] / 100.0
    
    # Subtract loads  
    for i, load_name in enumerate(data['loads']['names']):
        bus_name = data['loads']['buses'][i]
        if bus_name in bus_to_idx:
            bus_idx = bus_to_idx[bus_name]
            P_inj[bus_idx] -= data['loads']['active_power_MW'][i] / 100.0
            Q_inj[bus_idx] -= data['loads']['reactive_power_MVAR'][i] / 100.0
    
    total_p = np.sum(P_inj)
    total_q = np.sum(Q_inj)
    print(f"   Total P injection: {total_p:.3f} pu ({total_p * 100:.1f} MW)")
    print(f"   Total Q injection: {total_q:.3f} pu ({total_q * 100:.1f} MVAR)")
    
    non_zero_buses = np.sum(np.abs(P_inj) > 0.001)
    print(f"   Buses with power: {non_zero_buses}")
    
    # Simple power flow calculation test
    print("\\n3️⃣ Testing power flow calculation...")
    V_test = np.ones(n_buses, dtype=complex)  # Flat start
    
    # Calculate power injections
    P_calc = np.zeros(n_buses)
    Q_calc = np.zeros(n_buses)
    
    V_mag = np.abs(V_test)
    theta = np.angle(V_test)
    G = Y.real
    B = Y.imag
    
    for i in range(n_buses):
        for j in range(n_buses):
            theta_ij = theta[i] - theta[j]
            cos_term = np.cos(theta_ij)
            sin_term = np.sin(theta_ij)
            
            P_calc[i] += V_mag[i] * V_mag[j] * (G[i,j] * cos_term + B[i,j] * sin_term)
            Q_calc[i] += V_mag[i] * V_mag[j] * (G[i,j] * sin_term - B[i,j] * cos_term)
    
    # Check for NaN in calculations
    p_calc_nan = np.sum(np.isnan(P_calc))
    q_calc_nan = np.sum(np.isnan(Q_calc))
    
    print(f"   P_calc NaN count: {p_calc_nan}")
    print(f"   Q_calc NaN count: {q_calc_nan}")
    print(f"   P_calc range: {P_calc.min():.6f} - {P_calc.max():.6f}")
    print(f"   Q_calc range: {Q_calc.min():.6f} - {Q_calc.max():.6f}")
    
    if p_calc_nan > 0 or q_calc_nan > 0:
        print("   ❌ Power calculation produces NaN!")
        return False
    
    # Calculate mismatches
    delta_P = P_inj - P_calc
    delta_Q = Q_inj - Q_calc
    
    max_p_mismatch = np.max(np.abs(delta_P))
    max_q_mismatch = np.max(np.abs(delta_Q))
    
    print(f"   Max P mismatch: {max_p_mismatch:.6f} pu ({max_p_mismatch * 100:.1f} MW)")
    print(f"   Max Q mismatch: {max_q_mismatch:.6f} pu ({max_q_mismatch * 100:.1f} MVAR)")
    
    if np.isnan(max_p_mismatch) or np.isnan(max_q_mismatch):
        print("   ❌ Mismatch calculation produces NaN!")
        return False
    
    print("\\n✅ Basic calculations are working!")
    print("   The NaN issue must be in the Jacobian or voltage update logic.")
    
    return True


if __name__ == "__main__":
    success = create_simple_solver()
    if success:
        print("\\n🎯 Next step: Debug Jacobian matrix construction")
    else:
        print("\\n❌ Found the source of NaN values")