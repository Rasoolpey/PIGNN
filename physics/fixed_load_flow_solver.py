"""
Fixed Newton-Raphson Load Flow Solver
This addresses the numerical issues identified in our diagnostic analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.enhanced_h5_loader import EnhancedH5DataLoader
from core.graph_base import PowerGridGraph


class FixedNewtonRaphsonSolver:
    """
    Fixed Newton-Raphson load flow solver with proper numerical conditioning
    and enhanced power injection handling.
    """
    
    def __init__(self, 
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 verbose: bool = True):
        """
        Initialize the fixed Newton-Raphson solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for power mismatch
            verbose: Enable detailed output
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Bus type constants
        self.SLACK_BUS = 3
        self.PV_BUS = 2
        self.PQ_BUS = 1
        
        # Solver state
        self.iteration_count = 0
        self.converged = False
        self.power_mismatch_history = []
        self.damping_factor = 1.0
        
    def solve_load_flow(self, graph: PowerGridGraph, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve load flow using Newton-Raphson method with enhanced data.
        
        Args:
            graph: Power grid graph
            data: Complete system data from enhanced loader
            
        Returns:
            Dictionary with solution results
        """
        print("Starting Fixed Newton-Raphson Load Flow Solution")
        print("=" * 60)
        
        # Initialize solver state
        self._reset_solver_state()
        
        # Prepare system data
        solver_data = self._prepare_solver_data(graph, data)
        
        if not self._validate_solver_data(solver_data):
            return self._create_failure_result("Data validation failed")
        
        # Get Y-matrix
        Y = solver_data['Y_matrix']
        n_buses = Y.shape[0]
        
        print(f"System Information:")
        print(f"   - Buses: {n_buses}")
        print(f"   - Y-matrix condition: {np.linalg.cond(Y.toarray() if hasattr(Y, 'toarray') else Y):.2e}")
        
        # Initialize voltage solution
        voltage_solution = self._initialize_voltage_solution(solver_data)
        
        # Main Newton-Raphson iteration loop
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            
            if self.verbose:
                print(f"\nIteration {self.iteration_count}")
            
            # Calculate power mismatches
            delta_P, delta_Q, max_mismatch = self._calculate_power_mismatches(
                voltage_solution, solver_data)
            
            self.power_mismatch_history.append(max_mismatch)
            
            if self.verbose:
                print(f"   - Max power mismatch: {max_mismatch:.6f} pu")
                print(f"   - Damping factor: {self.damping_factor:.3f}")
            
            # Check convergence
            if max_mismatch < self.tolerance:
                self.converged = True
                print(f"Converged in {self.iteration_count} iterations!")
                break
            
            # Build Jacobian matrix
            J = self._build_jacobian_matrix(voltage_solution, solver_data)
            
            # Check Jacobian conditioning
            J_cond = np.linalg.cond(J)
            if J_cond > 1e12:
                print(f"WARNING: Jacobian poorly conditioned: {J_cond:.2e}")
                return self._create_failure_result("Jacobian singular")
            
            # Solve linear system: J * dx = -mismatch
            mismatch_vector = self._build_mismatch_vector(delta_P, delta_Q, solver_data)
            
            try:
                dx = np.linalg.solve(J, -mismatch_vector)
            except np.linalg.LinAlgError:
                print("Failed to solve Jacobian system")
                return self._create_failure_result("Jacobian solution failed")
            
            # Update voltage solution with adaptive damping
            voltage_solution = self._update_voltage_solution(
                voltage_solution, dx, solver_data)
            
            # Adaptive damping adjustment
            self._adjust_damping_factor(iteration)
        
        # Create solution result
        if self.converged:
            return self._create_success_result(voltage_solution, solver_data, data)
        else:
            return self._create_failure_result(f"Failed to converge in {self.max_iterations} iterations")
    
    def _prepare_solver_data(self, graph: PowerGridGraph, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare all data needed for Newton-Raphson solution"""
        buses = data.get('buses', {})
        generators = data.get('generators', {})
        
        # Get Y-matrix from graph
        Y_matrix = graph.get_admittance_matrix()
        
        # Prepare bus data
        bus_names = buses.get('names', [])
        n_buses = len(bus_names)
        
        # Power injections (should be fixed by enhanced loader)
        P_injection = np.array(buses.get('active_injection_MW', np.zeros(n_buses)))
        Q_injection = np.array(buses.get('reactive_injection_MVAR', np.zeros(n_buses)))
        
        # Convert MW/MVAR to pu (assuming 100 MVA base)
        S_base = 100.0  # MVA
        P_injection_pu = P_injection / S_base
        Q_injection_pu = Q_injection / S_base
        
        # Determine bus types
        bus_types = self._determine_bus_types(data)
        
        # Voltage setpoints
        V_setpoint = np.array(buses.get('voltages_pu', np.ones(n_buses)))
        
        # Identify unknown variables
        pq_buses = np.where(bus_types == self.PQ_BUS)[0]
        pv_buses = np.where(bus_types == self.PV_BUS)[0]
        slack_buses = np.where(bus_types == self.SLACK_BUS)[0]
        
        solver_data = {
            'Y_matrix': Y_matrix,
            'P_injection_pu': P_injection_pu,
            'Q_injection_pu': Q_injection_pu,
            'bus_types': bus_types,
            'V_setpoint': V_setpoint,
            'pq_buses': pq_buses,
            'pv_buses': pv_buses,
            'slack_buses': slack_buses,
            'n_buses': n_buses,
            'bus_names': bus_names,
            'S_base': S_base
        }
        
        return solver_data
    
    def _determine_bus_types(self, data: Dict[str, Any]) -> np.ndarray:
        """Determine bus types from generator and load data"""
        buses = data.get('buses', {})
        generators = data.get('generators', {})
        
        bus_names = buses.get('names', [])
        n_buses = len(bus_names)
        
        # Default all buses to PQ
        bus_types = np.full(n_buses, self.PQ_BUS)
        
        # Create bus name to index mapping
        bus_to_index = {name: i for i, name in enumerate(bus_names)}
        
        # Mark generator buses as PV
        gen_names = generators.get('names', [])
        if len(gen_names) > 0:
            gen_buses = generators.get('buses', [])
            for bus_name in gen_buses:
                if bus_name in bus_to_index:
                    bus_idx = bus_to_index[bus_name]
                    bus_types[bus_idx] = self.PV_BUS
        
        # Set first generator bus as slack (or first bus if no generators)
        if len(np.where(bus_types == self.PV_BUS)[0]) > 0:
            slack_idx = np.where(bus_types == self.PV_BUS)[0][0]
        else:
            slack_idx = 0
        
        bus_types[slack_idx] = self.SLACK_BUS
        
        return bus_types
    
    def _validate_solver_data(self, solver_data: Dict[str, Any]) -> bool:
        """Validate solver data for numerical issues"""
        try:
            # Check Y-matrix
            Y = solver_data['Y_matrix']
            if Y is None or Y.shape[0] == 0:
                print("Y-matrix is empty or None")
                return False
            
            # Check power injections
            P_inj = solver_data['P_injection_pu']
            Q_inj = solver_data['Q_injection_pu']
            
            if np.any(np.isnan(P_inj)) or np.any(np.isnan(Q_inj)):
                print("NaN values in power injections")
                return False
            
            # Check for at least one slack bus
            slack_buses = solver_data['slack_buses']
            if len(slack_buses) == 0:
                print("No slack bus defined")
                return False
            
            print("Solver data validation passed")
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def _initialize_voltage_solution(self, solver_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Initialize voltage solution"""
        n_buses = solver_data['n_buses']
        V_setpoint = solver_data['V_setpoint']
        
        # Initialize with flat start but use voltage setpoints for magnitude
        voltage_solution = {
            'V_mag': V_setpoint.copy(),  # Use setpoints for initial voltage magnitudes
            'V_angle': np.zeros(n_buses)  # Start with zero angles
        }
        
        print(f"Initialized voltage solution:")
        print(f"   - Voltage range: {np.min(voltage_solution['V_mag']):.3f} to {np.max(voltage_solution['V_mag']):.3f} pu")
        
        return voltage_solution
    
    def _calculate_power_mismatches(self, voltage_solution: Dict[str, np.ndarray], 
                                    solver_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculate power mismatches for Newton-Raphson"""
        V_mag = voltage_solution['V_mag']
        V_angle = voltage_solution['V_angle']
        Y = solver_data['Y_matrix']
        P_inj = solver_data['P_injection_pu']
        Q_inj = solver_data['Q_injection_pu']
        pq_buses = solver_data['pq_buses']
        pv_buses = solver_data['pv_buses']
        
        # Convert Y to dense if sparse
        if hasattr(Y, 'toarray'):
            Y = Y.toarray()
        
        # Calculate computed power injections
        P_calc = np.zeros_like(P_inj)
        Q_calc = np.zeros_like(Q_inj)
        
        for i in range(len(V_mag)):
            for j in range(len(V_mag)):
                Y_mag = abs(Y[i, j])
                Y_angle = np.angle(Y[i, j])
                
                angle_diff = V_angle[i] - V_angle[j] - Y_angle
                
                P_calc[i] += V_mag[i] * V_mag[j] * Y_mag * np.cos(angle_diff)
                Q_calc[i] -= V_mag[i] * V_mag[j] * Y_mag * np.sin(angle_diff)
        
        # Calculate mismatches (only for PQ and PV buses)
        delta_P = np.zeros_like(P_inj)
        delta_Q = np.zeros_like(Q_inj)
        
        # P mismatch for PQ and PV buses
        active_buses = np.concatenate([pq_buses, pv_buses])
        delta_P[active_buses] = P_inj[active_buses] - P_calc[active_buses]
        
        # Q mismatch only for PQ buses
        delta_Q[pq_buses] = Q_inj[pq_buses] - Q_calc[pq_buses]
        
        # Maximum mismatch for convergence check
        max_P_mismatch = np.max(np.abs(delta_P[active_buses])) if len(active_buses) > 0 else 0
        max_Q_mismatch = np.max(np.abs(delta_Q[pq_buses])) if len(pq_buses) > 0 else 0
        max_mismatch = max(max_P_mismatch, max_Q_mismatch)
        
        return delta_P, delta_Q, max_mismatch
    
    def _build_jacobian_matrix(self, voltage_solution: Dict[str, np.ndarray], 
                               solver_data: Dict[str, Any]) -> np.ndarray:
        """Build Jacobian matrix for Newton-Raphson"""
        V_mag = voltage_solution['V_mag']
        V_angle = voltage_solution['V_angle']
        Y = solver_data['Y_matrix']
        pq_buses = solver_data['pq_buses']
        pv_buses = solver_data['pv_buses']
        
        # Convert Y to dense if sparse
        if hasattr(Y, 'toarray'):
            Y = Y.toarray()
        
        n_buses = len(V_mag)
        active_buses = np.concatenate([pq_buses, pv_buses])
        n_active = len(active_buses)
        n_pq = len(pq_buses)
        
        # Calculate power injections for Jacobian derivatives
        P_calc = np.zeros(n_buses)
        Q_calc = np.zeros(n_buses)
        
        for i in range(n_buses):
            for j in range(n_buses):
                Y_mag = abs(Y[i, j])
                Y_angle = np.angle(Y[i, j])
                angle_diff = V_angle[i] - V_angle[j] - Y_angle
                
                P_calc[i] += V_mag[i] * V_mag[j] * Y_mag * np.cos(angle_diff)
                Q_calc[i] -= V_mag[i] * V_mag[j] * Y_mag * np.sin(angle_diff)
        
        # Build Jacobian submatrices
        # J11: ∂P/∂θ (for active buses vs active buses)
        J11 = np.zeros((n_active, n_active))
        for idx_i, i in enumerate(active_buses):
            for idx_j, j in enumerate(active_buses):
                if i == j:
                    # Diagonal element
                    J11[idx_i, idx_j] = -Q_calc[i] - V_mag[i]**2 * Y[i, i].imag
                else:
                    # Off-diagonal element
                    Y_mag = abs(Y[i, j])
                    Y_angle = np.angle(Y[i, j])
                    angle_diff = V_angle[i] - V_angle[j] - Y_angle
                    J11[idx_i, idx_j] = V_mag[i] * V_mag[j] * Y_mag * np.sin(angle_diff)
        
        # J12: ∂P/∂V (for active buses vs PQ buses)
        J12 = np.zeros((n_active, n_pq))
        for idx_i, i in enumerate(active_buses):
            for idx_j, j in enumerate(pq_buses):
                if i == j:
                    # Diagonal element
                    J12[idx_i, idx_j] = P_calc[i]/V_mag[i] + V_mag[i] * Y[i, i].real
                else:
                    # Off-diagonal element
                    Y_mag = abs(Y[i, j])
                    Y_angle = np.angle(Y[i, j])
                    angle_diff = V_angle[i] - V_angle[j] - Y_angle
                    J12[idx_i, idx_j] = V_mag[i] * Y_mag * np.cos(angle_diff)
        
        # J21: ∂Q/∂θ (for PQ buses vs active buses)
        J21 = np.zeros((n_pq, n_active))
        for idx_i, i in enumerate(pq_buses):
            for idx_j, j in enumerate(active_buses):
                if i == j:
                    # Diagonal element
                    J21[idx_i, idx_j] = P_calc[i] - V_mag[i]**2 * Y[i, i].real
                else:
                    # Off-diagonal element
                    Y_mag = abs(Y[i, j])
                    Y_angle = np.angle(Y[i, j])
                    angle_diff = V_angle[i] - V_angle[j] - Y_angle
                    J21[idx_i, idx_j] = -V_mag[i] * V_mag[j] * Y_mag * np.cos(angle_diff)
        
        # J22: ∂Q/∂V (for PQ buses vs PQ buses)
        J22 = np.zeros((n_pq, n_pq))
        for idx_i, i in enumerate(pq_buses):
            for idx_j, j in enumerate(pq_buses):
                if i == j:
                    # Diagonal element
                    J22[idx_i, idx_j] = Q_calc[i]/V_mag[i] - V_mag[i] * Y[i, i].imag
                else:
                    # Off-diagonal element
                    Y_mag = abs(Y[i, j])
                    Y_angle = np.angle(Y[i, j])
                    angle_diff = V_angle[i] - V_angle[j] - Y_angle
                    J22[idx_i, idx_j] = V_mag[i] * Y_mag * np.sin(angle_diff)
        
        # Assemble full Jacobian
        if n_pq > 0:
            J = np.block([[J11, J12],
                          [J21, J22]])
        else:
            J = J11
        
        return J
    
    def _build_mismatch_vector(self, delta_P: np.ndarray, delta_Q: np.ndarray, 
                               solver_data: Dict[str, Any]) -> np.ndarray:
        """Build mismatch vector for Newton-Raphson"""
        pq_buses = solver_data['pq_buses']
        pv_buses = solver_data['pv_buses']
        
        active_buses = np.concatenate([pq_buses, pv_buses])
        
        # P mismatches for active buses
        mismatch = delta_P[active_buses].tolist()
        
        # Q mismatches for PQ buses
        if len(pq_buses) > 0:
            mismatch.extend(delta_Q[pq_buses].tolist())
        
        return np.array(mismatch)
    
    def _update_voltage_solution(self, voltage_solution: Dict[str, np.ndarray], 
                                 dx: np.ndarray, solver_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Update voltage solution with Newton-Raphson correction"""
        pq_buses = solver_data['pq_buses']
        pv_buses = solver_data['pv_buses']
        
        active_buses = np.concatenate([pq_buses, pv_buses])
        n_active = len(active_buses)
        n_pq = len(pq_buses)
        
        # Update angles for active buses
        d_angles = dx[:n_active]
        voltage_solution['V_angle'][active_buses] += self.damping_factor * d_angles
        
        # Update voltage magnitudes for PQ buses
        if n_pq > 0:
            d_voltages = dx[n_active:]
            voltage_solution['V_mag'][pq_buses] += self.damping_factor * d_voltages
        
        # Ensure voltage magnitudes stay within reasonable bounds
        voltage_solution['V_mag'] = np.clip(voltage_solution['V_mag'], 0.5, 1.5)
        
        return voltage_solution
    
    def _adjust_damping_factor(self, iteration: int):
        """Adjust damping factor for better convergence"""
        if iteration > 0:
            # Reduce damping if mismatch is not improving
            if len(self.power_mismatch_history) >= 2:
                if self.power_mismatch_history[-1] > self.power_mismatch_history[-2]:
                    self.damping_factor *= 0.8
                elif iteration > 10:
                    self.damping_factor = max(0.5, self.damping_factor)
                else:
                    self.damping_factor = min(1.0, self.damping_factor * 1.05)
    
    def _reset_solver_state(self):
        """Reset solver state for new solution"""
        self.iteration_count = 0
        self.converged = False
        self.power_mismatch_history = []
        self.damping_factor = 1.0
    
    def _create_success_result(self, voltage_solution: Dict[str, np.ndarray], 
                               solver_data: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create success result dictionary"""
        return {
            'converged': True,
            'iterations': self.iteration_count,
            'voltage_magnitudes_pu': voltage_solution['V_mag'],
            'voltage_angles_deg': np.degrees(voltage_solution['V_angle']),
            'power_mismatch_history': self.power_mismatch_history,
            'final_mismatch': self.power_mismatch_history[-1] if self.power_mismatch_history else 0,
            'bus_names': solver_data['bus_names'],
            'bus_types': solver_data['bus_types'],
            'solver_data': solver_data,
            'original_data': data
        }
    
    def _create_failure_result(self, reason: str) -> Dict[str, Any]:
        """Create failure result dictionary"""
        return {
            'converged': False,
            'iterations': self.iteration_count,
            'failure_reason': reason,
            'power_mismatch_history': self.power_mismatch_history,
            'final_mismatch': self.power_mismatch_history[-1] if self.power_mismatch_history else float('inf')
        }


def test_fixed_solver():
    """Test the fixed Newton-Raphson solver"""
    print("Testing Fixed Newton-Raphson Load Flow Solver")
    print("=" * 60)
    
    # Load enhanced data
    h5_file = "data/scenario_0.h5"
    
    if not os.path.exists(h5_file):
        print(f"H5 file not found: {h5_file}")
        return
    
    # Load data with enhanced loader
    loader = EnhancedH5DataLoader(h5_file)
    data = loader.load_complete_system_data()
    
    # Create power grid graph
    from data.graph_builder import GraphBuilder
    builder = GraphBuilder()
    graph = builder.build_graph_from_h5_data(data)
    
    # Create and run solver
    solver = FixedNewtonRaphsonSolver(max_iterations=50, tolerance=1e-6, verbose=True)
    result = solver.solve_load_flow(graph, data)
    
    # Display results
    print(f"\nSolution Results:")
    print(f"   - Converged: {result['converged']}")
    if result['converged']:
        print(f"   - Iterations: {result['iterations']}")
        print(f"   - Final mismatch: {result['final_mismatch']:.2e} pu")
        
        V_mag = result['voltage_magnitudes_pu']
        V_ang = result['voltage_angles_deg']
        
        print(f"   - Voltage range: {np.min(V_mag):.3f} to {np.max(V_mag):.3f} pu")
        print(f"   - Angle range: {np.min(V_ang):.2f}° to {np.max(V_ang):.2f}°")
        
        # Show some bus results
        bus_names = result.get('bus_names', [])
        if len(bus_names) > 0:
            print(f"\n   Sample Bus Results:")
            for i in range(min(5, len(bus_names))):
                print(f"      - {bus_names[i]}: {V_mag[i]:.4f} angle {V_ang[i]:.2f}° pu")
    else:
        print(f"   - Failure reason: {result.get('failure_reason', 'Unknown')}")
    
    return result


if __name__ == "__main__":
    test_fixed_solver()