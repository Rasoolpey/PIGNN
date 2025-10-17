"""
Single-Phase Load Flow Solver

This matches PowerFactory's single-phase positive sequence modeling approach
instead of the full three-phase solver. This should converge properly.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
from typing import Dict, Tuple, Optional, List

sys.path.append(str(Path(__file__).parent))

from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder


class SinglePhaseLoadFlowSolver:
    """Single-phase Newton-Raphson load flow solver matching PowerFactory approach"""
    
    def __init__(self, bus_data: Dict, gen_data: Dict, load_data: Dict, line_data: Dict, 
                 transformer_data: Dict = None, base_mva: float = 100.0, tolerance: float = 1e-6, max_iterations: int = 50):
        
        self.base_mva = base_mva
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # Process network data
        self.bus_names = bus_data['names']
        self.n_buses = len(self.bus_names)
        
        # Create bus name to index mapping
        self.bus_to_idx = {name: i for i, name in enumerate(self.bus_names)}
        
        # Bus classification
        self.pq_buses = []  # Load buses
        self.pv_buses = []  # Generator buses 
        self.slack_bus = 0   # Reference bus (Bus 01)
        
        # Power injections (positive = generation, negative = load)
        self.P_specified = np.zeros(self.n_buses)
        self.Q_specified = np.zeros(self.n_buses)
        
        # Voltage setpoints for PV buses
        self.V_setpoints = np.ones(self.n_buses)  
        
        # Build system matrices
        self._process_generators(gen_data)
        self._process_loads(load_data)
        self._build_y_matrix(line_data, transformer_data)
        self._classify_buses()
        
    def _process_generators(self, gen_data: Dict):
        """Process generator data"""
        if 'names' not in gen_data:
            return
            
        for i, gen_name in enumerate(gen_data['names']):
            bus_name = gen_data['buses'][i]
            if bus_name in self.bus_to_idx:
                bus_idx = self.bus_to_idx[bus_name]
                
                # Add generation (positive injection)
                self.P_specified[bus_idx] += gen_data['active_power_MW'][i] / self.base_mva
                self.Q_specified[bus_idx] += gen_data['reactive_power_MVAR'][i] / self.base_mva
                
                # Set voltage setpoint
                if 'voltage_setpoint_pu' in gen_data:
                    self.V_setpoints[bus_idx] = gen_data['voltage_setpoint_pu'][i]
                elif 'terminal_voltage_pu' in gen_data:
                    self.V_setpoints[bus_idx] = gen_data['terminal_voltage_pu'][i]
    
    def _process_loads(self, load_data: Dict):
        """Process load data"""
        if 'names' not in load_data:
            return
            
        for i, load_name in enumerate(load_data['names']):
            bus_name = load_data['buses'][i]
            if bus_name in self.bus_to_idx:
                bus_idx = self.bus_to_idx[bus_name]
                
                # Subtract load (negative injection)
                self.P_specified[bus_idx] -= load_data['active_power_MW'][i] / self.base_mva
                self.Q_specified[bus_idx] -= load_data['reactive_power_MVAR'][i] / self.base_mva
    
    def _build_y_matrix(self, line_data: Dict, transformer_data: Dict = None):
        """Build admittance matrix from line and transformer data"""
        self.Y = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        
        # Process lines
        if 'names' in line_data:
            for i in range(len(line_data['names'])):
                from_bus = line_data['from_buses'][i]
                to_bus = line_data['to_buses'][i]
                
                if from_bus in self.bus_to_idx and to_bus in self.bus_to_idx:
                    from_idx = self.bus_to_idx[from_bus]
                    to_idx = self.bus_to_idx[to_bus]
                    
                    # Calculate admittance
                    R = line_data['R_ohm'][i]
                    X = line_data['X_ohm'][i]
                    Z = R + 1j * X
                    Y_line = 1 / Z if abs(Z) > 1e-10 else 0
                    
                    # Add to Y matrix
                    self.Y[from_idx, to_idx] -= Y_line
                    self.Y[to_idx, from_idx] -= Y_line
                    self.Y[from_idx, from_idx] += Y_line
                    self.Y[to_idx, to_idx] += Y_line
        
        # Process transformers
        if transformer_data and 'names' in transformer_data:
            for i in range(len(transformer_data['names'])):
                from_bus = transformer_data['from_buses'][i]
                to_bus = transformer_data['to_buses'][i]
                
                if from_bus in self.bus_to_idx and to_bus in self.bus_to_idx:
                    from_idx = self.bus_to_idx[from_bus]
                    to_idx = self.bus_to_idx[to_bus]
                    
                    # Calculate transformer admittance
                    R = transformer_data['R_ohm'][i]
                    X = transformer_data['X_ohm'][i] 
                    Z = R + 1j * X
                    Y_xfmr = 1 / Z if abs(Z) > 1e-10 else 0
                    
                    # Add to Y matrix
                    self.Y[from_idx, to_idx] -= Y_xfmr
                    self.Y[to_idx, from_idx] -= Y_xfmr
                    self.Y[from_idx, from_idx] += Y_xfmr
                    self.Y[to_idx, to_idx] += Y_xfmr
    
    def _classify_buses(self):
        """Classify buses based on power injections and generation"""
        for i, bus_name in enumerate(self.bus_names):
            if i == self.slack_bus:
                continue  # Skip slack bus
            elif abs(self.P_specified[i]) > 1e-6 and self.V_setpoints[i] != 1.0:
                # Generator bus with voltage control
                self.pv_buses.append(i)
            else:
                # Load bus or bus with only load
                self.pq_buses.append(i)
        
        print(f"Bus classification:")
        print(f"  Slack bus: {self.bus_names[self.slack_bus]} (idx {self.slack_bus})")
        print(f"  PV buses: {len(self.pv_buses)} - {[self.bus_names[i] for i in self.pv_buses[:5]]}")
        print(f"  PQ buses: {len(self.pq_buses)} - {[self.bus_names[i] for i in self.pq_buses[:5]]}")
    
    def solve(self, verbose: bool = True) -> Dict:
        """Solve single-phase load flow"""
        if verbose:
            print("🔧 Single-Phase Newton-Raphson Load Flow")
            print(f"   System: {self.n_buses} buses")
            print(f"   Total generation: {np.sum(self.P_specified[self.P_specified > 0]) * self.base_mva:.1f} MW")
            print(f"   Total load: {-np.sum(self.P_specified[self.P_specified < 0]) * self.base_mva:.1f} MW")
        
        # Initialize voltages
        V = np.ones(self.n_buses, dtype=complex)
        V[self.slack_bus] = 1.0 + 0j  # Slack bus reference
        
        # Set PV bus voltage magnitudes  
        for i in self.pv_buses:
            V[i] = complex(self.V_setpoints[i], 0.0)
        
        converged = False
        iteration = 0
        
        if verbose:
            print("\\n📊 Iteration Progress:")
            print("   Iter |  Max ΔP (MW) |  Max ΔQ (MVAR) |  Max Mismatch")
            print("   -----|---------------|-----------------|-------------")
        
        for iteration in range(self.max_iterations):
            # Calculate power injections
            P_calc, Q_calc = self._calculate_power_injections(V)
            
            # Calculate mismatches
            delta_P = self.P_specified - P_calc
            delta_Q = self.Q_specified - Q_calc
            
            # Only consider mismatches for appropriate buses
            # P mismatches for all buses except slack
            # Q mismatches only for PQ buses
            
            mismatch_indices = []
            mismatch_values = []
            
            # P mismatches (all buses except slack)
            for i in range(self.n_buses):
                if i != self.slack_bus:
                    mismatch_indices.append(('P', i))
                    mismatch_values.append(delta_P[i])
            
            # Q mismatches (only PQ buses)
            for i in self.pq_buses:
                mismatch_indices.append(('Q', i))
                mismatch_values.append(delta_Q[i])
            
            mismatch_vector = np.array(mismatch_values)
            max_mismatch = np.max(np.abs(mismatch_vector))
            
            if verbose:
                max_p_mismatch = np.max(np.abs(delta_P)) * self.base_mva
                max_q_mismatch = np.max(np.abs(delta_Q)) * self.base_mva
                print(f"   {iteration:4d} | {max_p_mismatch:12.3f} | {max_q_mismatch:14.3f} | {max_mismatch:12.3e}")
            
            if max_mismatch < self.tolerance:
                converged = True
                break
                
            # Build Jacobian matrix
            J = self._build_jacobian(V, mismatch_indices)
            
            # Solve for corrections
            try:
                corrections = np.linalg.solve(J, mismatch_vector)
            except np.linalg.LinAlgError as e:
                if verbose:
                    print(f"   ❌ Jacobian singular: {e}")
                break
            
            # Update voltages
            V = self._update_voltages(V, corrections, mismatch_indices)
        
        # Calculate final results
        P_final, Q_final = self._calculate_power_injections(V)
        total_losses = np.sum(V * np.conj(self.Y @ V)).real * self.base_mva
        
        if verbose:
            if converged:
                print(f"   ✅ Converged in {iteration + 1} iterations")
            else:
                print(f"   ❌ Failed to converge after {self.max_iterations} iterations")
            print(f"   📈 Voltage range: {np.abs(V).min():.3f} - {np.abs(V).max():.3f} pu")
            print(f"   📈 Total losses: {total_losses:.1f} MW")
        
        return {
            'converged': converged,
            'iterations': iteration + 1,
            'voltages': V,
            'voltage_magnitudes': np.abs(V),
            'voltage_angles_deg': np.degrees(np.angle(V)),
            'active_power': P_final * self.base_mva,
            'reactive_power': Q_final * self.base_mva,
            'total_losses_mw': total_losses,
            'bus_names': self.bus_names
        }
    
    def _calculate_power_injections(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bus power injections"""
        P = np.zeros(self.n_buses)
        Q = np.zeros(self.n_buses)
        
        V_mag = np.abs(V)
        theta = np.angle(V)
        G = self.Y.real
        B = self.Y.imag
        
        for i in range(self.n_buses):
            for j in range(self.n_buses):
                theta_ij = theta[i] - theta[j]
                P[i] += V_mag[i] * V_mag[j] * (G[i,j] * np.cos(theta_ij) + B[i,j] * np.sin(theta_ij))
                Q[i] += V_mag[i] * V_mag[j] * (G[i,j] * np.sin(theta_ij) - B[i,j] * np.cos(theta_ij))
        
        return P, Q
    
    def _build_jacobian(self, V: np.ndarray, mismatch_indices: List) -> np.ndarray:
        """Build Jacobian matrix"""
        n_eq = len(mismatch_indices)
        
        # Variable indices: theta for all buses except slack, V for PQ buses
        theta_vars = [i for i in range(self.n_buses) if i != self.slack_bus]
        v_vars = self.pq_buses.copy()
        n_vars = len(theta_vars) + len(v_vars)
        
        if n_eq == 0 or n_vars == 0:
            return np.eye(max(1, min(n_eq, n_vars)))
        
        J = np.zeros((n_eq, n_vars))
        V_mag = np.abs(V)
        theta = np.angle(V)
        G = self.Y.real
        B = self.Y.imag
        
        for eq_idx, (power_type, bus_i) in enumerate(mismatch_indices):
            # Derivatives with respect to angles
            for var_idx, bus_j in enumerate(theta_vars):
                if power_type == 'P':
                    if bus_i == bus_j:
                        # ∂P_i/∂θ_i
                        sum_val = sum(V_mag[bus_i] * V_mag[k] * (G[bus_i,k] * np.sin(theta[bus_i] - theta[k]) - 
                                                               B[bus_i,k] * np.cos(theta[bus_i] - theta[k])) 
                                    for k in range(self.n_buses) if k != bus_i)
                        J[eq_idx, var_idx] = -sum_val
                    else:
                        # ∂P_i/∂θ_j
                        theta_ij = theta[bus_i] - theta[bus_j]
                        J[eq_idx, var_idx] = V_mag[bus_i] * V_mag[bus_j] * (G[bus_i,bus_j] * np.sin(theta_ij) - 
                                                                           B[bus_i,bus_j] * np.cos(theta_ij))
                elif power_type == 'Q':
                    if bus_i == bus_j:
                        # ∂Q_i/∂θ_i
                        sum_val = sum(V_mag[bus_i] * V_mag[k] * (G[bus_i,k] * np.cos(theta[bus_i] - theta[k]) + 
                                                               B[bus_i,k] * np.sin(theta[bus_i] - theta[k])) 
                                    for k in range(self.n_buses) if k != bus_i)
                        J[eq_idx, var_idx] = sum_val
                    else:
                        # ∂Q_i/∂θ_j
                        theta_ij = theta[bus_i] - theta[bus_j]
                        J[eq_idx, var_idx] = -V_mag[bus_i] * V_mag[bus_j] * (G[bus_i,bus_j] * np.cos(theta_ij) + 
                                                                             B[bus_i,bus_j] * np.sin(theta_ij))
            
            # Derivatives with respect to voltage magnitudes (only for PQ buses)
            for var_idx, bus_j in enumerate(v_vars):
                global_var_idx = len(theta_vars) + var_idx
                if global_var_idx < n_vars:
                    if power_type == 'P':
                        if bus_i == bus_j:
                            # ∂P_i/∂V_i
                            sum_val = sum(V_mag[k] * (G[bus_i,k] * np.cos(theta[bus_i] - theta[k]) + 
                                                    B[bus_i,k] * np.sin(theta[bus_i] - theta[k])) 
                                        for k in range(self.n_buses))
                            J[eq_idx, global_var_idx] = sum_val
                        else:
                            # ∂P_i/∂V_j
                            theta_ij = theta[bus_i] - theta[bus_j]
                            J[eq_idx, global_var_idx] = V_mag[bus_i] * (G[bus_i,bus_j] * np.cos(theta_ij) + 
                                                                       B[bus_i,bus_j] * np.sin(theta_ij))
                    elif power_type == 'Q':
                        if bus_i == bus_j:
                            # ∂Q_i/∂V_i
                            sum_val = sum(V_mag[k] * (G[bus_i,k] * np.sin(theta[bus_i] - theta[k]) - 
                                                    B[bus_i,k] * np.cos(theta[bus_i] - theta[k])) 
                                        for k in range(self.n_buses))
                            J[eq_idx, global_var_idx] = sum_val
                        else:
                            # ∂Q_i/∂V_j
                            theta_ij = theta[bus_i] - theta[bus_j]
                            J[eq_idx, global_var_idx] = V_mag[bus_i] * (G[bus_i,bus_j] * np.sin(theta_ij) - 
                                                                       B[bus_i,bus_j] * np.cos(theta_ij))
        
        return J
    
    def _update_voltages(self, V: np.ndarray, corrections: np.ndarray, mismatch_indices: List) -> np.ndarray:
        """Update voltage vector with corrections"""
        V_new = V.copy()
        
        theta_vars = [i for i in range(self.n_buses) if i != self.slack_bus]
        v_vars = self.pq_buses.copy()
        
        # Apply angle corrections
        for i, bus_idx in enumerate(theta_vars):
            if i < len(corrections):
                current_mag = abs(V_new[bus_idx])
                current_angle = np.angle(V_new[bus_idx])
                new_angle = current_angle + corrections[i]
                V_new[bus_idx] = current_mag * np.exp(1j * new_angle)
        
        # Apply voltage magnitude corrections
        v_start_idx = len(theta_vars)
        for i, bus_idx in enumerate(v_vars):
            corr_idx = v_start_idx + i
            if corr_idx < len(corrections):
                current_mag = abs(V_new[bus_idx])
                current_angle = np.angle(V_new[bus_idx])
                new_mag = current_mag + corrections[corr_idx]
                new_mag = max(0.5, min(1.5, new_mag))  # Voltage limits
                V_new[bus_idx] = new_mag * np.exp(1j * current_angle)
        
        # Maintain PV bus voltage magnitudes
        for bus_idx in self.pv_buses:
            current_angle = np.angle(V_new[bus_idx])
            V_new[bus_idx] = self.V_setpoints[bus_idx] * np.exp(1j * current_angle)
        
        # Maintain slack bus voltage
        V_new[self.slack_bus] = 1.0 + 0j
        
        return V_new


def test_single_phase_solver():
    """Test the single-phase load flow solver"""
    print("🔋 Testing Single-Phase Load Flow Solver")
    print("=" * 50)
    
    # Load PowerFactory data
    loader = H5DataLoader('data/scenario_0.h5')
    data = loader.load_all_data()
    
    # Create single-phase solver
    solver = SinglePhaseLoadFlowSolver(
        bus_data=data['buses'],
        gen_data=data['generators'], 
        load_data=data['loads'],
        line_data=data['lines'],
        transformer_data=data['transformers'],
        base_mva=100.0
    )
    
    # Solve
    results = solver.solve(verbose=True)
    
    # Compare with PowerFactory reference
    print("\\n📊 Comparison with PowerFactory:")
    with h5py.File('data/scenario_0.h5', 'r') as f:
        pf_voltages = f['load_flow_results/bus_data/bus_voltages_pu'][:]
        pf_angles = f['load_flow_results/bus_data/bus_angles_deg'][:]
        
    print("   Bus Name    | Solver V  | PF V     | Solver θ | PF θ     | Error")
    print("   ------------|-----------|----------|----------|----------|-------")
    
    max_v_error = 0
    max_angle_error = 0
    
    for i in range(min(10, len(results['bus_names']))):  # Show first 10 buses
        bus_name = results['bus_names'][i]
        solver_v = results['voltage_magnitudes'][i]
        solver_angle = results['voltage_angles_deg'][i]
        pf_v = pf_voltages[i] 
        pf_angle = pf_angles[i]
        
        v_error = abs(solver_v - pf_v)
        angle_error = abs(solver_angle - pf_angle)
        
        max_v_error = max(max_v_error, v_error)
        max_angle_error = max(max_angle_error, angle_error)
        
        print(f"   {bus_name:<11} | {solver_v:8.4f}  | {pf_v:8.4f} | {solver_angle:7.2f}°  | {pf_angle:7.2f}° | {v_error:.4f}")
    
    print(f"\\n📈 Maximum errors:")
    print(f"   Voltage magnitude: {max_v_error:.4f} pu")
    print(f"   Voltage angle: {max_angle_error:.2f}°")
    
    success = results['converged'] and max_v_error < 0.01 and max_angle_error < 5.0
    return success


if __name__ == "__main__":
    success = test_single_phase_solver()
    print(f"\\nTest result: {'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}")