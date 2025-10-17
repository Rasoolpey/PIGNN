"""
Enhanced Load Flow Solver with Better Convergence

This creates an improved version of the load flow solver with better numerical
stability and convergence characteristics.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from physics.load_flow_solver import ThreePhaseLoadFlowSolver, LoadFlowResults, BusType
from core.graph_base import PowerGridGraph, PhaseType


class EnhancedLoadFlowSolver(ThreePhaseLoadFlowSolver):
    """Enhanced Newton-Raphson load flow solver with better convergence"""
    
    def __init__(self, graph: PowerGridGraph, **kwargs):
        # Initialize with more conservative defaults
        kwargs.setdefault('tolerance', 1e-4)  # Relaxed tolerance
        kwargs.setdefault('max_iterations', 100) 
        kwargs.setdefault('acceleration_factor', 0.5)  # More conservative
        
        super().__init__(graph, **kwargs)
        
        # Enhanced convergence parameters
        self.voltage_limits = (0.8, 1.2)  # Tighter voltage limits
        self.max_angle_change = np.pi / 6  # Limit angle changes
        self.damping_threshold = 5  # Start damping after this many iterations
        self.line_search_steps = 5  # Line search for step size
        
    def solve(self, verbose: bool = True) -> LoadFlowResults:
        """Solve with enhanced convergence techniques"""
        if verbose:
            print("🔧 Enhanced Three-Phase Load Flow Analysis")
            print(f"   • System size: {self.Y_matrix.shape[0]} nodes")
            print(f"   • Enhanced stability features enabled")
        
        # Initialize with better starting point
        V = self._initialize_voltages_enhanced()
        
        converged = False
        iteration = 0
        max_mismatch = float('inf')
        previous_mismatch = float('inf')
        stagnation_count = 0
        
        if verbose:
            print("\\n📊 Iteration Progress:")
            print("   Iter |  Max ΔP (MW) |  Max ΔQ (MVAR) |  Max Mismatch | Step Size")
            print("   -----|---------------|-----------------|---------------|----------")
        
        for iteration in range(self.max_iterations):
            # Calculate power mismatches
            P_calc, Q_calc = self._calculate_power_injections(V)
            P_spec, Q_spec = self._get_specified_powers()
            delta_P, delta_Q = self._calculate_mismatches(P_calc, Q_calc, P_spec, Q_spec)
            
            # Check convergence
            max_mismatch = max(np.max(np.abs(delta_P)), np.max(np.abs(delta_Q)))
            
            if verbose and iteration % 5 == 0:  # Print every 5th iteration
                max_delta_p = np.max(np.abs(delta_P)) * self.graph.base_mva
                max_delta_q = np.max(np.abs(delta_Q)) * self.graph.base_mva
                print(f"   {iteration:4d} | {max_delta_p:12.3f} | {max_delta_q:14.3f} | "
                      f"{max_mismatch:12.3e} | {self.acceleration_factor:8.3f}")
            
            if max_mismatch < self.tolerance:
                converged = True
                break
            
            # Check for stagnation
            if abs(max_mismatch - previous_mismatch) / max(previous_mismatch, 1e-6) < 0.01:
                stagnation_count += 1
            else:
                stagnation_count = 0
                
            if stagnation_count > 10:
                if verbose:
                    print("   ⚠️  Stagnation detected - applying perturbation")
                V = self._apply_perturbation(V)
                stagnation_count = 0
            
            # Build Jacobian with regularization
            J = self._build_jacobian_enhanced(V)
            mismatch = self._prepare_mismatch_vector(delta_P, delta_Q)
            
            if len(mismatch) == 0:
                break
                
            # Solve with multiple fallback methods
            delta_x = self._solve_linear_system(J, mismatch)
            if delta_x is None:
                break
                
            # Line search for optimal step size
            optimal_step = self._line_search(V, delta_x, P_spec, Q_spec)
            
            # Update voltages with optimal step
            previous_mismatch = max_mismatch
            V = self._update_voltages_enhanced(V, delta_x, optimal_step)
        
        # Calculate final results
        P_final, Q_final = self._calculate_power_injections(V)
        total_losses_mw, total_losses_mvar = self._calculate_losses(V)
        
        if verbose:
            if converged:
                print(f"   ✅ Converged in {iteration + 1} iterations")
            else:
                print(f"   ⚠️  Reached max iterations ({self.max_iterations})")
            print(f"   📈 Final mismatch: {max_mismatch:.2e}")
            print(f"   📈 System losses: {total_losses_mw:.1f} MW")
        
        return LoadFlowResults(
            converged=converged,
            iterations=iteration + 1,
            max_mismatch=max_mismatch,
            voltages=V.copy(),
            voltage_magnitudes=np.abs(V),
            voltage_angles=np.angle(V),
            active_power=P_final * self.graph.base_mva,
            reactive_power=Q_final * self.graph.base_mva,
            total_losses_mw=total_losses_mw,
            total_losses_mvar=total_losses_mvar,
            node_mapping=self.y_builder.node_to_index.copy()
        )
    
    def _initialize_voltages_enhanced(self) -> np.ndarray:
        """Initialize with better starting voltages"""
        n_nodes = len(self.y_builder.node_to_index)
        V = np.ones(n_nodes, dtype=complex)
        
        # Set generator voltages closer to their targets
        for node_id in self.pv_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    node = self.graph.nodes[node_id][phase]
                    
                    # Use specified voltage magnitude for PV buses
                    v_target = getattr(node, 'voltage_setpoint_pu', 1.0)
                    V[idx] = complex(v_target, 0.0)
        
        # Set slack bus voltages
        for node_id in self.slack_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    node = self.graph.nodes[node_id][phase]
                    V[idx] = node.voltage_pu
        
        return V
    
    def _build_jacobian_enhanced(self, V: np.ndarray) -> np.ndarray:
        """Build Jacobian with enhanced numerical stability"""
        J = self._build_jacobian(V)
        
        # Add regularization for numerical stability
        if J.shape[0] > 0 and J.shape[1] > 0:
            # Regularize diagonal elements
            min_diagonal = 1e-8
            for i in range(min(J.shape[0], J.shape[1])):
                if abs(J[i, i]) < min_diagonal:
                    J[i, i] += min_diagonal * np.sign(J[i, i]) if J[i, i] != 0 else min_diagonal
        
        return J
    
    def _solve_linear_system(self, J: np.ndarray, mismatch: np.ndarray) -> Optional[np.ndarray]:
        """Solve linear system with multiple fallback methods"""
        try:
            # Primary method: Direct solve
            return np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            try:
                # Fallback 1: Pseudo-inverse
                return np.linalg.pinv(J) @ mismatch
            except:
                try:
                    # Fallback 2: SVD-based solve
                    U, s, Vt = np.linalg.svd(J)
                    # Filter small singular values
                    s_inv = np.where(s > 1e-10, 1/s, 0)
                    return Vt.T @ np.diag(s_inv) @ U.T @ mismatch
                except:
                    return None
    
    def _line_search(self, V: np.ndarray, delta_x: np.ndarray, P_spec: np.ndarray, Q_spec: np.ndarray) -> float:
        """Find optimal step size using line search"""
        best_step = 1.0
        best_mismatch = float('inf')
        
        # Try different step sizes
        for step in [1.0, 0.8, 0.6, 0.4, 0.2]:
            V_trial = self._update_voltages_enhanced(V, delta_x, step)
            
            # Calculate trial mismatch
            P_calc, Q_calc = self._calculate_power_injections(V_trial)
            delta_P, delta_Q = self._calculate_mismatches(P_calc, Q_calc, P_spec, Q_spec)
            trial_mismatch = max(np.max(np.abs(delta_P)), np.max(np.abs(delta_Q)))
            
            if trial_mismatch < best_mismatch:
                best_mismatch = trial_mismatch
                best_step = step
        
        return best_step
    
    def _update_voltages_enhanced(self, V: np.ndarray, delta_x: np.ndarray, step_size: float = 1.0) -> np.ndarray:
        """Update voltages with enhanced stability"""
        V_new = V.copy()
        
        if len(delta_x) == 0:
            return V_new
        
        # Get variable indices
        pq_indices = []
        pv_indices = []
        
        for node_id in self.pq_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    pq_indices.append(idx)
        
        for node_id in self.pv_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    pv_indices.append(idx)
        
        all_variable_indices = pq_indices + pv_indices
        n_theta_vars = len(all_variable_indices)
        
        # Apply angle corrections with limits
        if len(delta_x) >= n_theta_vars:
            delta_theta = delta_x[:n_theta_vars]
            
            for i, idx in enumerate(all_variable_indices):
                if i < len(delta_theta) and idx < len(V_new):
                    # Limit angle changes
                    angle_change = step_size * delta_theta[i]
                    angle_change = np.clip(angle_change, -self.max_angle_change, self.max_angle_change)
                    
                    current_mag = abs(V_new[idx])
                    current_angle = np.angle(V_new[idx])
                    new_angle = current_angle + angle_change
                    V_new[idx] = current_mag * np.exp(1j * new_angle)
        
        # Apply voltage magnitude corrections with strict limits
        if len(delta_x) >= n_theta_vars + len(pq_indices):
            delta_v = delta_x[n_theta_vars:n_theta_vars + len(pq_indices)]
            
            for i, idx in enumerate(pq_indices):
                if i < len(delta_v) and idx < len(V_new):
                    current_mag = abs(V_new[idx])
                    current_angle = np.angle(V_new[idx])
                    
                    # Apply voltage change with limits
                    v_change = step_size * delta_v[i]
                    new_mag = current_mag + v_change
                    
                    # Enforce voltage limits
                    new_mag = np.clip(new_mag, self.voltage_limits[0], self.voltage_limits[1])
                    V_new[idx] = new_mag * np.exp(1j * current_angle)
        
        # Maintain PV and slack bus constraints
        self._enforce_bus_constraints(V_new)
        
        return V_new
    
    def _enforce_bus_constraints(self, V: np.ndarray):
        """Enforce PV and slack bus voltage constraints"""
        # Maintain PV bus voltage magnitudes
        for node_id in self.pv_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    if idx < len(V):
                        node = self.graph.nodes[node_id][phase]
                        V_setpoint = getattr(node, 'voltage_setpoint_pu', abs(node.voltage_pu))
                        
                        current_angle = np.angle(V[idx])
                        V[idx] = V_setpoint * np.exp(1j * current_angle)
        
        # Maintain slack bus voltages
        for node_id in self.slack_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    if idx < len(V):
                        node = self.graph.nodes[node_id][phase]
                        V[idx] = node.voltage_pu
    
    def _apply_perturbation(self, V: np.ndarray) -> np.ndarray:
        """Apply small random perturbation to break stagnation"""
        V_new = V.copy()
        
        # Add small random perturbations to PQ bus voltages
        for node_id in self.pq_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    if idx < len(V_new):
                        # Small random angle perturbation
                        current_mag = abs(V_new[idx])
                        current_angle = np.angle(V_new[idx])
                        
                        perturbation = (np.random.random() - 0.5) * 0.01  # ±0.01 radians
                        V_new[idx] = current_mag * np.exp(1j * (current_angle + perturbation))
        
        return V_new