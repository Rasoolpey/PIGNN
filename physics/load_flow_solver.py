"""
Three-phase load flow solver using Newton-Raphson method.
Works with the physics-informed graph structure.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_base import PowerGridGraph, PhaseType
from physics.impedance_matrix import AdmittanceMatrixBuilder


class BusType(Enum):
    """Bus types for load flow analysis"""
    PQ = 'PQ'      # Load bus - P and Q specified
    PV = 'PV'      # Generator bus - P and V specified
    SLACK = 'SLACK'  # Slack bus - V and angle specified


@dataclass
class LoadFlowResults:
    """Results from load flow analysis"""
    converged: bool
    iterations: int
    max_mismatch: float
    
    # Solution vectors (3N x 1 for three-phase)
    voltages: np.ndarray  # Complex voltages
    voltage_magnitudes: np.ndarray  # |V|
    voltage_angles: np.ndarray  # ∠V in radians
    
    # Power injections
    active_power: np.ndarray  # P in MW
    reactive_power: np.ndarray  # Q in MVAR
    
    # System losses
    total_losses_mw: float
    total_losses_mvar: float
    
    # Node mapping
    node_mapping: Dict[str, int]
    
    def get_bus_results(self, bus_id: str) -> Dict[str, Dict[str, float]]:
        """Get results for a specific bus (all three phases)"""
        results = {}
        
        for phase in PhaseType:
            node_id = f"{bus_id}_{phase.value}"
            if node_id in self.node_mapping:
                idx = self.node_mapping[node_id]
                results[phase.value] = {
                    'voltage_pu': abs(self.voltages[idx]),
                    'angle_deg': np.degrees(np.angle(self.voltages[idx])),
                    'voltage_real': self.voltages[idx].real,
                    'voltage_imag': self.voltages[idx].imag,
                    'active_power_mw': self.active_power[idx],
                    'reactive_power_mvar': self.reactive_power[idx]
                }
        
        return results


class ThreePhaseLoadFlowSolver:
    """
    Newton-Raphson load flow solver for three-phase unbalanced systems.
    Handles the full 3N x 3N admittance matrix with inter-phase coupling.
    """
    
    def __init__(self, 
                 graph: PowerGridGraph,
                 tolerance: float = 1e-6,
                 max_iterations: int = 50,
                 acceleration_factor: float = 1.0):
        """
        Initialize the load flow solver.
        
        Args:
            graph: PowerGridGraph object
            tolerance: Convergence tolerance for power mismatches
            max_iterations: Maximum number of iterations
            acceleration_factor: Newton step acceleration (1.0 = normal)
        """
        self.graph = graph
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.acceleration_factor = acceleration_factor
        
        # Build admittance matrix
        self.y_builder = AdmittanceMatrixBuilder(graph)
        self.Y_matrix = None
        
        # Bus classification
        self.bus_types = {}
        self.pq_buses = []
        self.pv_buses = []
        self.slack_buses = []
        
        # Adaptive control parameters
        self.min_acceleration = 0.1
        self.max_acceleration = 1.0
        self.divergence_count = 0
        
        # Initialize
        self._setup_system()
    
    def _setup_system(self):
        """Setup the system matrices and bus classifications"""
        # Build Y-matrix
        self.Y_matrix = self.y_builder.build_y_matrix(use_sparse=False)
        
        # Classify buses based on node types
        self._classify_buses()
        
        # Validate system setup
        self._validate_system()
    
    def _classify_buses(self):
        """Classify buses based on their control modes"""
        for node_id, phase_nodes in self.graph.nodes.items():
            # Get the first phase to determine bus type
            phase_a_node = phase_nodes[PhaseType.A]
            
            if hasattr(phase_a_node, 'control_mode'):
                if phase_a_node.control_mode == 'slack':
                    self.bus_types[node_id] = BusType.SLACK
                    self.slack_buses.append(node_id)
                elif phase_a_node.control_mode == 'PV':
                    self.bus_types[node_id] = BusType.PV
                    self.pv_buses.append(node_id)
                else:
                    self.bus_types[node_id] = BusType.PQ
                    self.pq_buses.append(node_id)
            else:
                # Default classification based on node type
                if phase_a_node.node_type == 'generator':
                    # Check if it has voltage control
                    if hasattr(phase_a_node, 'voltage_setpoint_pu'):
                        self.bus_types[node_id] = BusType.PV
                        self.pv_buses.append(node_id)
                    else:
                        self.bus_types[node_id] = BusType.PQ
                        self.pq_buses.append(node_id)
                elif phase_a_node.node_type == 'bus':
                    # Check if it's marked as slack
                    if phase_a_node.properties.get('is_slack', False):
                        self.bus_types[node_id] = BusType.SLACK
                        self.slack_buses.append(node_id)
                    else:
                        self.bus_types[node_id] = BusType.PQ
                        self.pq_buses.append(node_id)
                else:
                    self.bus_types[node_id] = BusType.PQ
                    self.pq_buses.append(node_id)
    
    def _validate_system(self):
        """Validate the system setup"""
        if len(self.slack_buses) == 0:
            # Automatically assign the first bus as slack
            if self.graph.nodes:
                first_bus = list(self.graph.nodes.keys())[0]
                self.bus_types[first_bus] = BusType.SLACK
                self.slack_buses.append(first_bus)
                if first_bus in self.pq_buses:
                    self.pq_buses.remove(first_bus)
                if first_bus in self.pv_buses:
                    self.pv_buses.remove(first_bus)
                warnings.warn(f"No slack bus found. Assigned bus '{first_bus}' as slack.")
        
        if len(self.slack_buses) > 1:
            warnings.warn(f"Multiple slack buses found: {self.slack_buses}. Using first one.")
            # Keep only the first slack bus
            for bus in self.slack_buses[1:]:
                self.bus_types[bus] = BusType.PQ
                self.pq_buses.append(bus)
            self.slack_buses = self.slack_buses[:1]
    
    def solve(self, verbose: bool = True) -> LoadFlowResults:
        """
        Solve the three-phase load flow problem using PyPSA's approach.
        
        Args:
            verbose: Print iteration information
        
        Returns:
            LoadFlowResults object
        """
        if verbose:
            print("[>>] Starting Three-Phase Load Flow Analysis (PyPSA Method)")
            print(f"   * Total buses: {len(self.graph.nodes)}")
            print(f"   * PQ buses: {len(self.pq_buses)}")
            print(f"   * PV buses: {len(self.pv_buses)}")
            print(f"   * Slack buses: {len(self.slack_buses)}")
            print(f"   * System size: {self.Y_matrix.shape[0]} nodes")
        
        # Initialize voltages (all buses)
        V = self._initialize_voltages()
        
        # Get specified powers
        S_spec = self._get_specified_powers()
        
        # Get bus indices
        pq_idx, pv_idx, pvpq_idx, slack_idx = self._get_bus_indices()
        
        # Build initial state vector (unknowns only)
        # state = [θ_pvpq, |V|_pq]
        state = np.r_[
            np.angle(V)[pvpq_idx],  # Angles for PV and PQ buses
            np.abs(V)[pq_idx]        # Magnitudes for PQ buses only
        ]
        
        if verbose:
            n_pvpq = len(pvpq_idx)
            n_pq = len(pq_idx)
            print(f"   • State variables: {len(state)} ({n_pvpq} angles + {n_pq} magnitudes)")
            print(f"   • Equations: {n_pvpq} P + {n_pq} Q = {n_pvpq + n_pq}")
            print("\n[>>] Iteration Progress:")
            print("   Iter |  Max dP (MW) |  Max dQ (MVAR) |  Max |F|")
            print("   -----|---------------|-----------------|-------------")
        
        # Newton-Raphson iterations
        converged = False
        iteration = 0
        
        for iteration in range(self.max_iterations):
            # Reconstruct full voltage vector from state
            V = self._update_voltage_from_state(V, state)
            
            # Calculate mismatch F = f(state)
            F = self._calculate_power_mismatch(V, S_spec)
            
            # Check convergence
            max_mismatch = np.max(np.abs(F))
            
            if verbose:
                # Separate P and Q mismatches for display
                n_pvpq = len(pvpq_idx)
                max_delta_p = np.max(np.abs(F[:n_pvpq])) * self.graph.base_mva
                max_delta_q = np.max(np.abs(F[n_pvpq:])) * self.graph.base_mva if len(F) > n_pvpq else 0.0
                print(f"   {iteration:4d} | {max_delta_p:12.6f} | {max_delta_q:14.6f} | {max_mismatch:12.6f}")
            
            if max_mismatch < self.tolerance:
                converged = True
                break
            
            # Build Jacobian matrix J = dF/dx
            J = self._build_jacobian_pypsa(V)
            
            # Solve linear system: J * Δx = F
            try:
                if sp.issparse(J):
                    delta_x = sp.linalg.spsolve(J, F)
                else:
                    delta_x = np.linalg.solve(J, F)
            except (np.linalg.LinAlgError, RuntimeError) as e:
                if verbose:
                    print(f"   ❌ Singular Jacobian matrix! {e}")
                break
            
            # Update state: x_new = x_old - Δx
            state = state - delta_x
        
        # Final voltage reconstruction
        V = self._update_voltage_from_state(V, state)
        
        # Calculate final power injections
        P_final, Q_final = self._calculate_power_injections(V)
        
        # Calculate losses
        total_losses_mw, total_losses_mvar = self._calculate_losses(V)
        
        if verbose:
            if converged:
                print(f"   [OK] Converged in {iteration + 1} iterations")
            else:
                print(f"   ❌ Failed to converge after {self.max_iterations} iterations")
            print(f"   [>>] Total system losses: {total_losses_mw:.3f} MW, {total_losses_mvar:.3f} MVAR")
        
        # Create results object
        results = LoadFlowResults(
            converged=converged,
            iterations=iteration + 1,
            max_mismatch=max_mismatch,
            voltages=V.copy(),
            voltage_magnitudes=np.abs(V),
            voltage_angles=np.angle(V),
            active_power=P_final * self.graph.base_mva,  # Convert to MW
            reactive_power=Q_final * self.graph.base_mva,  # Convert to MVAR
            total_losses_mw=total_losses_mw,
            total_losses_mvar=total_losses_mvar,
            node_mapping=self.y_builder.node_to_index.copy()
        )
        
        return results
    
    def _initialize_voltages(self) -> np.ndarray:
        """Initialize voltage vector"""
        n_nodes = len(self.y_builder.node_to_index)
        V = np.ones(n_nodes, dtype=complex)
        
        # Set initial voltages from graph data
        for node_id, phase_nodes in self.graph.nodes.items():
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    node = phase_nodes[phase]
                    V[idx] = node.voltage_pu
        
        return V
    
    def _calculate_power_injections(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power injections at all nodes using proper three-phase formulation"""
        n = len(V)
        P = np.zeros(n)
        Q = np.zeros(n)
        
        # Extract Y matrix components
        G = self.Y_matrix.real
        B = self.Y_matrix.imag
        
        # Extract voltage magnitudes and angles
        V_mag = np.abs(V)
        theta = np.angle(V)
        
        # Calculate power injections: P_i + jQ_i = V_i * conj(I_i) = V_i * conj(sum(Y_ij * V_j))
        for i in range(n):
            if V_mag[i] < 1e-10:  # Avoid division by zero
                continue
                
            P_sum = 0.0
            Q_sum = 0.0
            
            for j in range(n):
                if V_mag[j] < 1e-10:
                    continue
                    
                # Angle difference
                theta_ij = theta[i] - theta[j]
                
                # Power contributions
                P_sum += V_mag[j] * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
                Q_sum += V_mag[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))
            
            P[i] = V_mag[i] * P_sum
            Q[i] = V_mag[i] * Q_sum
        
        return P, Q
    
    def _get_specified_powers(self) -> np.ndarray:
        """
        Get specified complex power S = P + jQ for all buses.
        Following PyPSA's approach: use complex power formulation.
        """
        n_nodes = len(self.y_builder.node_to_index)
        S_spec = np.zeros(n_nodes, dtype=complex)
        
        for node_id, phase_nodes in self.graph.nodes.items():
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    node = phase_nodes[phase]
                    
                    # Get power injection from node properties (in per-unit)
                    P_pu = node.properties.get('P_injection_pu', 0.0)
                    Q_pu = node.properties.get('Q_injection_pu', 0.0)
                    S_spec[idx] = P_pu + 1j * Q_pu
        
        return S_spec
    
    def _get_bus_indices(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Get phase node indices for different bus types.
        Returns: (pq_indices, pv_indices, pvpq_indices, slack_indices)
        
        Following PyPSA: indices are for PHASE nodes (e.g., 'Bus01_a'), not buses.
        """
        pq_indices = []
        pv_indices = []
        slack_indices = []
        
        # PQ bus indices
        for node_id in self.pq_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    pq_indices.append(idx)
        
        # PV bus indices  
        for node_id in self.pv_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    pv_indices.append(idx)
        
        # Slack bus indices
        for node_id in self.slack_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    slack_indices.append(idx)
        
        # PVPQ = PV + PQ (all non-slack buses)
        pvpq_indices = pv_indices + pq_indices
        
        return pq_indices, pv_indices, pvpq_indices, slack_indices
    
    def _calculate_power_mismatch(self, V: np.ndarray, S_spec: np.ndarray) -> np.ndarray:
        """
        Calculate power mismatch following PyPSA's approach.
        
        Mismatch = V * conj(Y * V) - S_specified
        
        Build F vector as:
        F = [real(mismatch)[non-slack],     # P mismatches (skip slack)
             imag(mismatch)[pq_only]]        # Q mismatches (PQ buses only)
        """
        # Calculate injected power: S = V * conj(I) = V * conj(Y * V)
        if sp.issparse(self.Y_matrix):
            I = self.Y_matrix @ V
        else:
            I = self.Y_matrix @ V
        
        S_calc = V * np.conj(I)
        
        # Complex power mismatch
        mismatch = S_calc - S_spec
        
        # Get bus indices
        pq_idx, pv_idx, pvpq_idx, slack_idx = self._get_bus_indices()
        
        # Build mismatch vector F
        # P mismatches: all non-slack buses (PV + PQ)
        F_P = np.real(mismatch)[pvpq_idx]
        
        # Q mismatches: only PQ buses
        F_Q = np.imag(mismatch)[pq_idx]
        
        # Concatenate
        F = np.r_[F_P, F_Q]
        
        return F
    
    def _build_jacobian_pypsa(self, V: np.ndarray) -> sp.csr_matrix:
        """
        Build Jacobian matrix following PyPSA's elegant formulation.
        
        Uses complex matrix operations to compute:
        dS/dθ and dS/d|V|
        
        Then extracts real and imaginary parts for P and Q equations.
        """
        n = len(V)
        index = np.arange(n)
        
        # Calculate current injection
        if sp.issparse(self.Y_matrix):
            I = self.Y_matrix @ V
        else:
            I = self.Y_matrix @ V
        
        # Create diagonal matrices (convert to sparse for efficiency)
        V_diag = sp.csr_matrix((V, (index, index)), shape=(n, n))
        
        # Avoid division by zero in V_norm_diag
        V_abs = np.abs(V)
        V_abs[V_abs < 1e-10] = 1e-10  # Prevent division by zero
        V_norm_diag = sp.csr_matrix((V / V_abs, (index, index)), shape=(n, n))
        
        I_diag = sp.csr_matrix((I, (index, index)), shape=(n, n))
        
        # Convert Y to sparse if not already
        if not sp.issparse(self.Y_matrix):
            Y_sparse = sp.csr_matrix(self.Y_matrix)
        else:
            Y_sparse = self.Y_matrix
        
        # Compute derivatives using PyPSA's formulation
        # dS/dθ = j * V_diag * conj(I_diag - Y * V_diag)
        dS_dVa = 1j * V_diag @ np.conj(I_diag - Y_sparse @ V_diag)
        
        # dS/d|V| = V_norm_diag * conj(I_diag) + V_diag * conj(Y * V_norm_diag)
        dS_dVm = V_norm_diag @ np.conj(I_diag) + V_diag @ np.conj(Y_sparse @ V_norm_diag)
        
        # Get bus indices
        pq_idx, pv_idx, pvpq_idx, slack_idx = self._get_bus_indices()
        
        # Extract submatrices - use sparse matrix slicing
        # J00: ∂P/∂θ for PV+PQ buses (rows: pvpq, cols: pvpq)
        J00 = dS_dVa.real.tocsr()[pvpq_idx, :][:, pvpq_idx]
        
        # J01: ∂P/∂|V| for PV+PQ buses vs PQ buses (rows: pvpq, cols: pq)
        J01 = dS_dVm.real.tocsr()[pvpq_idx, :][:, pq_idx]
        
        # J10: ∂Q/∂θ for PQ buses (rows: pq, cols: pvpq)
        J10 = dS_dVa.imag.tocsr()[pq_idx, :][:, pvpq_idx]
        
        # J11: ∂Q/∂|V| for PQ buses (rows: pq, cols: pq)
        J11 = dS_dVm.imag.tocsr()[pq_idx, :][:, pq_idx]
        
        # Assemble full Jacobian
        # J = [[J00  J01]
        #      [J10  J11]]
        J = sp.vstack([
            sp.hstack([J00, J01]),
            sp.hstack([J10, J11])
        ], format='csr')
        
        return J
    
    def _update_voltage_from_state(self, V: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Reconstruct full voltage vector from state vector.
        
        State vector contains:
        - state[0:n_pvpq]: angles for PV and PQ buses
        - state[n_pvpq:]: magnitudes for PQ buses only
        
        Updates V in place for non-slack buses.
        """
        V_new = V.copy()
        
        pq_idx, pv_idx, pvpq_idx, slack_idx = self._get_bus_indices()
        
        n_pvpq = len(pvpq_idx)
        n_pq = len(pq_idx)
        
        # Update angles for PV and PQ buses
        theta = state[:n_pvpq]
        for i, idx in enumerate(pvpq_idx):
            V_mag = np.abs(V_new[idx])
            V_new[idx] = V_mag * np.exp(1j * theta[i])
        
        # Update magnitudes for PQ buses only  
        if n_pq > 0:
            v_mag = state[n_pvpq:n_pvpq + n_pq]
            for i, idx in enumerate(pq_idx):
                theta_i = np.angle(V_new[idx])
                V_new[idx] = v_mag[i] * np.exp(1j * theta_i)
        
        # Maintain PV bus voltage magnitudes at setpoint
        for node_id in self.pv_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    node = self.graph.nodes[node_id][phase]
                    v_setpoint = getattr(node, 'voltage_setpoint_pu', 1.0)
                    theta_i = np.angle(V_new[idx])
                    V_new[idx] = v_setpoint * np.exp(1j * theta_i)
        
        return V_new
    
    def _calculate_mismatches(self, P_calc: np.ndarray, Q_calc: np.ndarray, 
                            P_spec: np.ndarray, Q_spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power mismatches"""
        # Only calculate mismatches for PQ and PV buses
        n_nodes = len(self.y_builder.node_to_index)
        delta_P = np.zeros(n_nodes)
        delta_Q = np.zeros(n_nodes)
        
        for node_id in self.pq_buses + self.pv_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    delta_P[idx] = P_spec[idx] - P_calc[idx]
        
        # Q mismatches only for PQ buses
        for node_id in self.pq_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    delta_Q[idx] = Q_spec[idx] - Q_calc[idx]
        
        return delta_P, delta_Q
    
    def _build_jacobian(self, V: np.ndarray) -> np.ndarray:
        """Build proper Newton-Raphson Jacobian matrix for three-phase system"""
        n = len(V)
        
        # Extract voltage magnitudes and angles
        V_mag = np.abs(V)
        theta = np.angle(V)
        
        # Extract Y matrix components
        G = self.Y_matrix.real
        B = self.Y_matrix.imag
        
        # Identify variable indices (exclude slack buses)
        pq_indices = []  # PQ buses: theta and V are variables
        pv_indices = []  # PV buses: only theta is variable
        
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
        
        # Total equations: P equations for PQ+PV buses, Q equations for PQ buses only
        n_p_equations = len(pq_indices) + len(pv_indices)
        n_q_equations = len(pq_indices)
        n_equations = n_p_equations + n_q_equations
        
        # Total variables: theta for PQ+PV buses, V for PQ buses only  
        n_theta_vars = len(pq_indices) + len(pv_indices)
        n_v_vars = len(pq_indices)
        n_variables = n_theta_vars + n_v_vars
        
        if n_equations == 0 or n_variables == 0:
            return np.eye(1)
        
        # Initialize Jacobian
        J = np.zeros((n_equations, n_variables))
        
        # Variable mapping
        all_variable_indices = pq_indices + pv_indices  # For theta variables
        
        # Fill Jacobian submatrices
        # J = [∂P/∂θ   ∂P/∂V]
        #     [∂Q/∂θ   ∂Q/∂V]
        
        # ===== ∂P/∂θ block (upper left) =====
        for eq_idx, i in enumerate(all_variable_indices):
            if eq_idx >= n_p_equations:
                break
                
            for var_idx, j in enumerate(all_variable_indices):
                if var_idx >= n_theta_vars:
                    break
                    
                if i == j:
                    # Diagonal element ∂P_i/∂θ_i
                    sum_term = 0.0
                    for k in range(n):
                        if k != i and V_mag[k] > 1e-10:
                            theta_ik = theta[i] - theta[k]
                            sum_term += V_mag[k] * (G[i,k] * np.sin(theta_ik) - B[i,k] * np.cos(theta_ik))
                    J[eq_idx, var_idx] = -V_mag[i] * sum_term
                else:
                    # Off-diagonal element ∂P_i/∂θ_j
                    if V_mag[i] > 1e-10 and V_mag[j] > 1e-10:
                        theta_ij = theta[i] - theta[j]
                        J[eq_idx, var_idx] = V_mag[i] * V_mag[j] * (G[i,j] * np.sin(theta_ij) - B[i,j] * np.cos(theta_ij))
        
        # ===== ∂P/∂V block (upper right) =====
        for eq_idx, i in enumerate(all_variable_indices):
            if eq_idx >= n_p_equations:
                break
                
            for var_idx, j in enumerate(pq_indices):  # Only PQ buses have V as variable
                j_idx = pq_indices[var_idx]
                global_var_idx = n_theta_vars + var_idx
                
                if global_var_idx >= n_variables:
                    break
                    
                if i == j_idx:
                    # Diagonal element ∂P_i/∂V_i
                    sum_term = 0.0
                    for k in range(n):
                        if V_mag[k] > 1e-10:
                            theta_ik = theta[i] - theta[k]
                            sum_term += V_mag[k] * (G[i,k] * np.cos(theta_ik) + B[i,k] * np.sin(theta_ik))
                    J[eq_idx, global_var_idx] = sum_term
                else:
                    # Off-diagonal element ∂P_i/∂V_j
                    if V_mag[i] > 1e-10 and V_mag[j_idx] > 1e-10:
                        theta_ij = theta[i] - theta[j_idx]
                        J[eq_idx, global_var_idx] = V_mag[i] * (G[i,j_idx] * np.cos(theta_ij) + B[i,j_idx] * np.sin(theta_ij))
        
        # ===== ∂Q/∂θ block (lower left) =====
        for eq_idx, i in enumerate(pq_indices):  # Only PQ buses have Q equations
            global_eq_idx = n_p_equations + eq_idx
            
            if global_eq_idx >= n_equations:
                break
                
            for var_idx, j in enumerate(all_variable_indices):
                if var_idx >= n_theta_vars:
                    break
                    
                if i == j:
                    # Diagonal element ∂Q_i/∂θ_i
                    sum_term = 0.0
                    for k in range(n):
                        if k != i and V_mag[k] > 1e-10:
                            theta_ik = theta[i] - theta[k]
                            sum_term += V_mag[k] * (G[i,k] * np.cos(theta_ik) + B[i,k] * np.sin(theta_ik))
                    J[global_eq_idx, var_idx] = V_mag[i] * sum_term
                else:
                    # Off-diagonal element ∂Q_i/∂θ_j
                    if V_mag[i] > 1e-10 and V_mag[j] > 1e-10:
                        theta_ij = theta[i] - theta[j]
                        J[global_eq_idx, var_idx] = -V_mag[i] * V_mag[j] * (G[i,j] * np.cos(theta_ij) + B[i,j] * np.sin(theta_ij))
        
        # ===== ∂Q/∂V block (lower right) =====
        for eq_idx, i in enumerate(pq_indices):  # Only PQ buses have Q equations
            global_eq_idx = n_p_equations + eq_idx
            
            if global_eq_idx >= n_equations:
                break
                
            for var_idx, j in enumerate(pq_indices):  # Only PQ buses have V as variable
                j_idx = pq_indices[var_idx]
                global_var_idx = n_theta_vars + var_idx
                
                if global_var_idx >= n_variables:
                    break
                    
                if i == j_idx:
                    # Diagonal element ∂Q_i/∂V_i
                    sum_term = 0.0
                    for k in range(n):
                        if V_mag[k] > 1e-10:
                            theta_ik = theta[i] - theta[k]
                            sum_term += V_mag[k] * (G[i,k] * np.sin(theta_ik) - B[i,k] * np.cos(theta_ik))
                    J[global_eq_idx, global_var_idx] = sum_term
                else:
                    # Off-diagonal element ∂Q_i/∂V_j
                    if V_mag[i] > 1e-10 and V_mag[j_idx] > 1e-10:
                        theta_ij = theta[i] - theta[j_idx]
                        J[global_eq_idx, global_var_idx] = V_mag[i] * (G[i,j_idx] * np.sin(theta_ij) - B[i,j_idx] * np.cos(theta_ij))
        
        # Add regularization for numerical stability
        regularization = 1e-12
        for i in range(min(n_equations, n_variables)):
            if abs(J[i, i]) < regularization:
                J[i, i] += regularization
        
        return J
    
    def _prepare_mismatch_vector(self, delta_P: np.ndarray, delta_Q: np.ndarray) -> np.ndarray:
        """Prepare mismatch vector for Jacobian solution - must match Jacobian structure"""
        mismatches = []
        
        # Get indices for PQ and PV buses
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
        
        # Add P mismatches for all PQ and PV buses (first block)
        for idx in all_variable_indices:
            if idx < len(delta_P):
                mismatches.append(delta_P[idx])
        
        # Add Q mismatches for PQ buses only (second block)
        for idx in pq_indices:
            if idx < len(delta_Q):
                mismatches.append(delta_Q[idx])
        
        return np.array(mismatches) if mismatches else np.array([0.0])
    
    def _update_voltages(self, V: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
        """Update voltage vector with Newton-Raphson corrections"""
        V_new = V.copy()
        
        if len(delta_x) == 0:
            return V_new
        
        # Get indices for PQ and PV buses
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
        n_v_vars = len(pq_indices)
        
        # Apply angle corrections (first part of delta_x)
        if len(delta_x) >= n_theta_vars:
            delta_theta = delta_x[:n_theta_vars]
            
            for i, idx in enumerate(all_variable_indices):
                if i < len(delta_theta) and idx < len(V_new):
                    # Apply angle correction with acceleration factor
                    current_mag = abs(V_new[idx])
                    current_angle = np.angle(V_new[idx])
                    new_angle = current_angle + self.acceleration_factor * delta_theta[i]
                    V_new[idx] = current_mag * np.exp(1j * new_angle)
        
        # Apply voltage magnitude corrections (second part of delta_x, only for PQ buses)
        if len(delta_x) >= n_theta_vars + n_v_vars:
            delta_v = delta_x[n_theta_vars:n_theta_vars + n_v_vars]
            
            for i, idx in enumerate(pq_indices):
                if i < len(delta_v) and idx < len(V_new):
                    # Apply voltage magnitude correction
                    current_mag = abs(V_new[idx])
                    current_angle = np.angle(V_new[idx])
                    new_mag = current_mag + self.acceleration_factor * delta_v[i]
                    
                    # Apply voltage limits for stability
                    new_mag = max(0.5, min(1.5, new_mag))  # Keep between 0.5 and 1.5 pu
                    V_new[idx] = new_mag * np.exp(1j * current_angle)
        
        # Maintain PV bus voltage magnitudes
        for node_id in self.pv_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    if idx < len(V_new):
                        # Get specified voltage magnitude
                        node = self.graph.nodes[node_id][phase]
                        V_setpoint = getattr(node, 'voltage_setpoint_pu', abs(node.voltage_pu))
                        
                        # Keep magnitude fixed, only angle can change
                        current_angle = np.angle(V_new[idx])
                        V_new[idx] = V_setpoint * np.exp(1j * current_angle)
        
        # Maintain slack bus voltages (both magnitude and angle)
        for node_id in self.slack_buses:
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in self.y_builder.node_to_index:
                    idx = self.y_builder.node_to_index[node_phase_id]
                    if idx < len(V_new):
                        # Keep slack bus voltage fixed
                        node = self.graph.nodes[node_id][phase]
                        V_new[idx] = node.voltage_pu
        
        return V_new
    
    def _calculate_losses(self, V: np.ndarray) -> Tuple[float, float]:
        """Calculate total system losses"""
        I = self.Y_matrix @ V
        S_loss = np.sum(V * np.conj(I)).real  # Only real part for losses
        
        # Convert to MW/MVAR
        P_loss_mw = S_loss * self.graph.base_mva
        Q_loss_mvar = 0.0  # Simplified - reactive losses from line charging, etc.
        
        return P_loss_mw, Q_loss_mvar
    
    def print_system_summary(self):
        """Print summary of the system setup"""
        print("📋 Three-Phase Load Flow System Summary")
        print("=" * 50)
        print(f"Base MVA: {self.graph.base_mva}")
        print(f"Frequency: {self.graph.frequency_hz} Hz")
        print(f"Total nodes: {len(self.y_builder.node_to_index)}")
        print(f"Total buses: {len(self.graph.nodes)}")
        print(f"Total edges: {len(self.graph.edges)}")
        print(f"Y-matrix size: {self.Y_matrix.shape}")
        print(f"Y-matrix sparsity: {np.count_nonzero(self.Y_matrix) / self.Y_matrix.size * 100:.1f}%")
        print()
        print("Bus Classification:")
        print(f"  • PQ buses: {len(self.pq_buses)} - {self.pq_buses}")
        print(f"  • PV buses: {len(self.pv_buses)} - {self.pv_buses}")
        print(f"  • Slack buses: {len(self.slack_buses)} - {self.slack_buses}")


def run_load_flow_from_h5(h5_path: str, 
                          tolerance: float = 1e-6,
                          max_iterations: int = 50,
                          save_to_h5: bool = True) -> LoadFlowResults:
    """
    Convenient wrapper to run load flow directly from Graph_model.h5
    
    This function:
    1. Loads data from Graph_model.h5
    2. Builds the power grid graph
    3. Runs the three-phase load flow solver
    4. Optionally saves results back to Graph_model.h5 (in steady_state/power_flow_results)
    
    Args:
        h5_path: Path to Graph_model.h5 file
        tolerance: Convergence tolerance
        max_iterations: Maximum solver iterations
        save_to_h5: If True, saves results to steady_state/power_flow_results group
        
    Returns:
        LoadFlowResults object with converged solution
        
    Example:
        >>> results = run_load_flow_from_h5('graph_model/Graph_model.h5')
        >>> print(f"Converged: {results.converged}")
        >>> print(f"Voltages: {results.voltage_magnitudes}")
    """
    import h5py
    from data.h5_loader import H5DataLoader
    from data.graph_builder import GraphBuilder
    
    # Step 1: Load data from H5
    loader = H5DataLoader(h5_path)
    data = loader.load_all_data()
    
    # Step 2: Build graph
    builder = GraphBuilder()
    graph = builder.build_from_h5_data(data)
    
    # Step 3: Run load flow
    solver = ThreePhaseLoadFlowSolver(graph, tolerance=tolerance, max_iterations=max_iterations)
    results = solver.solve()
    
    # Step 4: Save results to H5 if requested
    if save_to_h5 and results.converged:
        with h5py.File(h5_path, 'a') as f:
            # Create steady_state group if it doesn't exist
            if 'steady_state' not in f:
                f.create_group('steady_state')
            
            # Create or overwrite power_flow_results
            if 'steady_state/power_flow_results' in f:
                del f['steady_state/power_flow_results']
            
            pf_group = f.create_group('steady_state/power_flow_results')
            
            # Save voltage results (extract phase A only for single-phase representation)
            phase_a_indices = [i for i in range(0, len(results.voltages), 3)]  # Every 3rd index
            V_mag_phase_a = results.voltage_magnitudes[phase_a_indices]
            V_ang_phase_a = np.rad2deg(results.voltage_angles[phase_a_indices])
            
            pf_group.create_dataset('bus_voltages_pu', data=V_mag_phase_a)
            pf_group.create_dataset('bus_angles_deg', data=V_ang_phase_a)
            
            # Save generator power outputs (sum across three phases)
            num_buses = len(V_mag_phase_a)
            gen_P_MW = np.zeros(num_buses)
            gen_Q_MVAR = np.zeros(num_buses)
            
            # Aggregate generation by bus (P > 0 means generation)
            for i, idx in enumerate(phase_a_indices):
                # Sum power injection across all three phases for this bus
                P_total = 0.0
                Q_total = 0.0
                for phase_offset in range(3):
                    if idx + phase_offset < len(results.active_power):
                        P_total += results.active_power[idx + phase_offset]
                        Q_total += results.reactive_power[idx + phase_offset]
                
                # Only store positive (generation)
                if P_total > 0:
                    gen_P_MW[i] = P_total
                    gen_Q_MVAR[i] = Q_total
            
            pf_group.create_dataset('gen_P_MW', data=gen_P_MW)
            pf_group.create_dataset('gen_Q_MVAR', data=gen_Q_MVAR)
            
            # Save system totals
            pf_group.attrs['total_generation_MW'] = np.sum(gen_P_MW)
            pf_group.attrs['total_generation_MVAR'] = np.sum(gen_Q_MVAR)
            pf_group.attrs['total_losses_MW'] = results.total_losses_mw
            pf_group.attrs['total_losses_MVAR'] = results.total_losses_mvar
            pf_group.attrs['converged'] = results.converged
            pf_group.attrs['iterations'] = results.iterations
            pf_group.attrs['max_mismatch'] = results.max_mismatch
    
    return results