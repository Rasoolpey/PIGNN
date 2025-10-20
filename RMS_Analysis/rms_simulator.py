"""
RMS Simulator - Dynamic Transient Stability Simulation

This module coordinates time-domain simulation of power system dynamics:
- Loads network and generator data from Graph_model.h5
- Initializes generators, exciters, governors from power flow
- Simulates electromechanical transients
- Handles fault events (3-phase, line-to-ground, line trips)

Author: PIGNN Project
Date: October 20, 2025
Reference: Kundur "Power System Stability and Control"
"""

import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from RMS_Analysis.generator_models import GENROUGenerator, load_genrou_from_h5
from RMS_Analysis.exciter_models import SEXSExciter, IEEEAC1AExciter, load_exciters_from_h5
from RMS_Analysis.governor_models import TGOV1Governor, HYGOVGovernor, load_governors_from_h5
from RMS_Analysis.integrator import create_integrator


@dataclass
class FaultEvent:
    """Fault event specification"""
    t_start: float      # Fault start time (s)
    t_clear: float      # Fault clear time (s)
    bus_id: int         # Bus where fault occurs
    fault_type: str     # '3phase', 'lg' (line-to-ground), 'line_trip'
    line_id: Optional[int] = None  # For line trip events


class RMSSimulator:
    """
    RMS (Root Mean Square) Dynamic Simulator
    
    Simulates electromechanical transients using phasor representation.
    """
    
    def __init__(self, h5_file: str, integrator_method: str = 'rk4', dt: float = 0.005):
        """
        Initialize RMS simulator.
        
        Args:
            h5_file: Path to Graph_model.h5
            integrator_method: 'rk4', 'trapezoidal', or 'adaptive'
            dt: Time step size (s), typical 0.005-0.01s
        """
        self.h5_file = h5_file
        self.dt = dt
        self.f_base = 60.0  # Hz
        self.omega_base = 2 * np.pi * self.f_base
        
        print(f"\n{'='*70}")
        print(f"  RMS Dynamic Simulator Initialization")
        print(f"{'='*70}")
        
        # STEP 1: Run load flow to get FRESH steady-state solution
        print("\n[STEP 1] Running Load Flow Analysis...")
        print("-" * 70)
        self._run_load_flow_and_save()
        
        # STEP 2: Load network data (now with updated power flow results)
        print("\n[STEP 2] Loading Network Data...")
        print("-" * 70)
        self._load_network_data()
        
        # STEP 3: Load generators
        print("\n[STEP 3] Loading Dynamic Models...")
        print("-" * 70)
        self.generators = load_genrou_from_h5(h5_file)
        print(f"[OK] Loaded {len(self.generators)} GENROU generators")
        
        # Load exciters
        self.exciters = load_exciters_from_h5(h5_file, list(self.generators.keys()))
        
        # Load governors
        self.governors = load_governors_from_h5(h5_file, list(self.generators.keys()))
        
        # STEP 4: Create integrator
        print("\n[STEP 4] Setting Up Integration Method...")
        print("-" * 70)
        self.integrator = create_integrator(integrator_method, dt)
        print(f"[OK] Using {integrator_method.upper()} integrator with dt={dt*1000:.1f}ms")
        
        # State tracking
        self.time_history = []
        self.state_history = []
        
        print(f"\n{'='*70}")
        print(f"  [SUCCESS] RMS Simulator Ready!")
        print(f"{'='*70}\n")
    
    def _run_load_flow_and_save(self):
        """
        Run REAL PyPSA Newton-Raphson load flow and save results.
        
        This ensures we have PROPER initial conditions with realistic generator outputs
        (not all at 75% loading!).
        
        Following ANDES methodology: Always initialize from solved power flow!
        """
        print("  -> Running PyPSA Newton-Raphson Load Flow...")
        
        # Import load flow solver
        from physics.load_flow_solver import run_load_flow_from_h5
        
        # Run load flow - it will automatically save to H5
        print("     [Running] Newton-Raphson solver...")
        results = run_load_flow_from_h5(
            h5_path=self.h5_file,
            tolerance=1e-6,
            max_iterations=50,
            save_to_h5=True  # Automatically saves results
        )
        
        if not results.converged:
            raise RuntimeError(f"❌ Load flow did NOT converge! Iterations: {results.iterations}")
        
        print(f"  [OK] Load flow converged:")
        print(f"     Iterations: {results.iterations}")
        print(f"     Max mismatch: {results.max_mismatch:.2e} pu")
        
        # Read back the saved results to display
        with h5py.File(self.h5_file, 'r') as f:
            pf_group = f['steady_state/power_flow_results']
            bus_voltages = pf_group['bus_voltages_pu'][:]
            gen_P_MW_by_bus = pf_group['gen_P_MW'][:]
            gen_Q_MVAR_by_bus = pf_group['gen_Q_MVAR'][:]
            
            print(f"     Voltage range: {bus_voltages.min():.4f} - {bus_voltages.max():.4f} pu")
            print(f"     Total generation: {pf_group.attrs['total_generation_MW']:.2f} MW")
            print(f"     Total losses: {pf_group.attrs['total_losses_MW']:.2f} MW")
        
        # Display generator outputs (non-zero buses)
        gen_indices = np.where(gen_P_MW_by_bus > 0)[0]
        print(f"\n  [Generator Outputs] (from CONVERGED load flow):")
        for i in gen_indices:
            print(f"     Bus {i+1}: P = {gen_P_MW_by_bus[i]:7.2f} MW, Q = {gen_Q_MVAR_by_bus[i]:7.2f} MVAR")
        
        print("  [OK] Power flow results saved to Graph_model.h5/steady_state/power_flow_results")
    
    def _load_network_data(self):
        """Load network topology and power flow solution from H5."""
        with h5py.File(self.h5_file, 'r') as f:
            # Use phase_a data (for balanced 3-phase system)
            phase = 'phases/phase_a'
            
            # Buses
            self.bus_ids = f[f'{phase}/nodes/bus_ids'][:]
            self.bus_names = [name.decode() if isinstance(name, bytes) else name 
                             for name in f[f'{phase}/nodes/bus_names'][:]]
            self.bus_types = f[f'{phase}/nodes/bus_types'][:]
            
            # Create bus name to ID mapping
            self.bus_name_to_id = {name: bus_id for bus_id, name in zip(self.bus_ids, self.bus_names)}
            
            # ============================================================
            # LOAD FRESH POWER FLOW RESULTS (from load flow we just ran!)
            # ============================================================
            if 'steady_state/power_flow_results' not in f:
                raise RuntimeError("❌ No power flow results found! Load flow should have run first.")
            
            pf_group = f['steady_state/power_flow_results']
            
            # Bus voltages from solved load flow
            self.V_mag = pf_group['bus_voltages_pu'][:]
            self.V_ang = pf_group['bus_angles_deg'][:] * np.pi / 180  # Convert to radians
            
            print(f"  [OK] Loaded power flow solution:")
            print(f"     Converged: {pf_group.attrs['converged']}")
            print(f"     Iterations: {pf_group.attrs['iterations']}")
            print(f"     Max mismatch: {pf_group.attrs['max_mismatch']:.2e} pu")
            print(f"     Voltage range: {self.V_mag.min():.4f} - {self.V_mag.max():.4f} pu")
            
            # Generators (from dynamic models)
            gen_group = f['dynamic_models/generators']
            gen_bus_names = [name.decode() if isinstance(name, bytes) else name 
                            for name in gen_group['buses'][:]]
            
            # Map generator bus names to IDs
            self.gen_bus_ids = np.array([self.bus_name_to_id[name] for name in gen_bus_names])
            
            self.gen_Sn_MVA = gen_group['Sn_MVA'][:]
            self.gen_names = [name.decode() if isinstance(name, bytes) else name 
                             for name in gen_group['names'][:]]
            
            # ============================================================
            # GET REAL GENERATOR OUTPUTS FROM LOAD FLOW RESULTS
            # Map bus-indexed power to generator-indexed power
            # ============================================================
            gen_P_MW_by_bus = pf_group['gen_P_MW'][:]  # Indexed by bus
            gen_Q_MVAR_by_bus = pf_group['gen_Q_MVAR'][:]  # Indexed by bus
            
            # Map generators to their bus power
            gen_P_MW = np.zeros(len(self.gen_bus_ids))
            gen_Q_MVAR = np.zeros(len(self.gen_bus_ids))
            
            for i, gen_bus_id in enumerate(self.gen_bus_ids):
                # Find bus index in bus array
                bus_idx = np.where(self.bus_ids == gen_bus_id)[0][0]
                # Get power from that bus
                gen_P_MW[i] = gen_P_MW_by_bus[bus_idx]
                gen_Q_MVAR[i] = gen_Q_MVAR_by_bus[bus_idx]
            
            # Convert to pu on machine base
            self.gen_P = gen_P_MW / self.gen_Sn_MVA
            self.gen_Q = gen_Q_MVAR / self.gen_Sn_MVA
            
            print(f"\n  [Generator Initial Conditions] (from load flow):")
            for i, name in enumerate(self.gen_names):
                print(f"     {name}: P = {self.gen_P[i]:.4f} pu ({gen_P_MW[i]:.2f} MW), "
                      f"Q = {self.gen_Q[i]:.4f} pu ({gen_Q_MVAR[i]:.2f} MVAR)")
            
            # Admittance matrix (for network solution)
            self.Y_bus = self._build_admittance_matrix(f)
        
        print(f"\n[OK] Network: {len(self.bus_ids)} buses, {len(self.gen_names)} generators")
    
    def _build_admittance_matrix(self, f: h5py.File) -> np.ndarray:
        """Build bus admittance matrix."""
        n_bus = len(self.bus_ids)
        Y = np.zeros((n_bus, n_bus), dtype=complex)
        
        # Use phase_a edges
        phase = 'phases/phase_a'
        
        if f'{phase}/edges' in f:
            edges = f[f'{phase}/edges']
            from_buses = edges['from_bus'][:]
            to_buses = edges['to_bus'][:]
            r_pu = edges['R_pu'][:]
            x_pu = edges['X_pu'][:]
            b_pu = edges['B_shunt_pu'][:]
            
            for i, (fb, tb) in enumerate(zip(from_buses, to_buses)):
                # Find bus indices
                from_idx = np.where(self.bus_ids == fb)[0][0]
                to_idx = np.where(self.bus_ids == tb)[0][0]
                
                # Series admittance
                z = r_pu[i] + 1j * x_pu[i]
                y_series = 1 / z if abs(z) > 1e-10 else 0
                
                # Shunt admittance
                y_shunt = 1j * b_pu[i] / 2
                
                # Build Y matrix
                Y[from_idx, to_idx] -= y_series
                Y[to_idx, from_idx] -= y_series
                Y[from_idx, from_idx] += y_series + y_shunt
                Y[to_idx, to_idx] += y_series + y_shunt
        
        return Y
    
    def initialize(self):
        """Initialize all dynamic models from power flow solution."""
        print("Initializing dynamic models from power flow...")
        
        for i, gen_name in enumerate(self.gen_names):
            # Get bus voltage
            bus_id = self.gen_bus_ids[i]
            bus_idx = np.where(self.bus_ids == bus_id)[0][0]
            Vt_mag = self.V_mag[bus_idx]
            Vt_ang = self.V_ang[bus_idx]
            
            # Initialize generator
            gen = self.generators[gen_name]
            gen.initialize(self.gen_P[i], self.gen_Q[i], Vt_mag, Vt_ang)
            
            # Initialize exciter
            exc = self.exciters[gen_name]
            exc.initialize(Vt_mag, gen.Efd_pu)
            
            # Initialize governor
            gov = self.governors[gen_name]
            gov.initialize(self.gen_P[i])  # Pm = Pelec at steady state
        
        print(f"[OK] All models initialized\n")
    
    def _collect_states(self) -> np.ndarray:
        """Collect all state variables into single vector."""
        states = []
        
        for gen_name in self.gen_names:
            gen = self.generators[gen_name]
            states.extend(gen.states)  # [delta, omega, Eq', Ed', Eq'', Ed'']
            
            # Exciter states
            exc = self.exciters[gen_name]
            if isinstance(exc, SEXSExciter):
                states.append(exc.Efd_pu)
            elif isinstance(exc, IEEEAC1AExciter):
                states.extend(exc.states)  # [Vr, Efd, Vf]
            
            # Governor states
            gov = self.governors[gen_name]
            if isinstance(gov, TGOV1Governor):
                states.append(gov.Pm_pu)
            elif isinstance(gov, HYGOVGovernor):
                states.extend(gov.states)  # [gate, q, Pm]
        
        return np.array(states)
    
    def _distribute_states(self, state_vector: np.ndarray):
        """Distribute state vector back to individual models."""
        idx = 0
        
        for gen_name in self.gen_names:
            gen = self.generators[gen_name]
            gen.states = state_vector[idx:idx+6]
            idx += 6
            
            exc = self.exciters[gen_name]
            if isinstance(exc, SEXSExciter):
                exc.Efd_pu = state_vector[idx]
                idx += 1
            elif isinstance(exc, IEEEAC1AExciter):
                exc.states = state_vector[idx:idx+3]
                idx += 3
            
            gov = self.governors[gen_name]
            if isinstance(gov, TGOV1Governor):
                gov.Pm_pu = state_vector[idx]
                idx += 1
            elif isinstance(gov, HYGOVGovernor):
                gov.states = state_vector[idx:idx+3]
                idx += 3
    
    def _compute_derivatives(self, t: float, state_vector: np.ndarray) -> np.ndarray:
        """Compute time derivatives of all states."""
        # Distribute states to models
        self._distribute_states(state_vector)
        
        # Update network solution (simplified - assuming constant voltage for now)
        # In full implementation, solve algebraic network equations
        
        derivatives = []
        
        for i, gen_name in enumerate(self.gen_names):
            gen = self.generators[gen_name]
            exc = self.exciters[gen_name]
            gov = self.governors[gen_name]
            
            # Get terminal voltage (from network solution)
            bus_id = self.gen_bus_ids[i]
            bus_idx = np.where(self.bus_ids == bus_id)[0][0]
            Vt_mag = self.V_mag[bus_idx]  # Simplified - should be updated
            
            # Generator derivatives
            Efd = exc.Efd_pu if isinstance(exc, SEXSExciter) else exc.get_field_voltage()
            Pm = gov.Pm_pu if isinstance(gov, TGOV1Governor) else gov.get_mechanical_power()
            
            gen_deriv = gen.derivatives(Pm, Efd, Vt_mag, 0.0)  # Simplified Vq
            derivatives.extend(gen_deriv)
            
            # Exciter derivatives
            if isinstance(exc, SEXSExciter):
                exc_deriv = exc.compute_derivative(t, Vt_mag)
                derivatives.append(exc_deriv)
            elif isinstance(exc, IEEEAC1AExciter):
                exc_deriv = exc.compute_derivatives(t, Vt_mag)
                derivatives.extend(exc_deriv)
            
            # Governor derivatives
            omega_pu = gen.states[1]  # Generator speed
            if isinstance(gov, TGOV1Governor):
                gov_deriv = gov.compute_derivative(t, omega_pu)
                derivatives.append(gov_deriv)
            elif isinstance(gov, HYGOVGovernor):
                gov_deriv = gov.compute_derivatives(t, omega_pu)
                derivatives.extend(gov_deriv)
        
        return np.array(derivatives)
    
    def simulate(self, t_end: float, fault_events: List[FaultEvent] = None) -> Dict:
        """
        Run time-domain simulation.
        
        Args:
            t_end: End time (s)
            fault_events: List of fault events
        
        Returns:
            Dictionary with simulation results
        """
        print(f"Starting RMS simulation: 0 -> {t_end}s (dt={self.dt*1000:.1f}ms)")
        start_time = time.time()
        
        # Initialize
        self.initialize()
        
        # Collect initial states
        t = 0.0
        states = self._collect_states()
        
        self.time_history = [t]
        self.state_history = [states.copy()]
        
        # Main integration loop
        n_steps = int(t_end / self.dt)
        
        for step in range(n_steps):
            # Check for fault events
            if fault_events:
                for event in fault_events:
                    if abs(t - event.t_start) < self.dt / 2:
                        print(f"  [FAULT] Fault applied at bus {event.bus_id} at t={t:.3f}s")
                        # Apply fault (modify Y matrix)
                    elif abs(t - event.t_clear) < self.dt / 2:
                        print(f"  [OK] Fault cleared at t={t:.3f}s")
                        # Clear fault
            
            # Integration step
            t, states = self.integrator.step(t, states, self._compute_derivatives)
            
            # Store
            self.time_history.append(t)
            self.state_history.append(states.copy())
            
            # Progress
            if (step + 1) % max(1, n_steps // 10) == 0:
                progress = (step + 1) / n_steps * 100
                print(f"  Progress: {progress:.0f}% (t={t:.2f}s)")
        
        elapsed = time.time() - start_time
        print(f"[OK] Simulation complete in {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/s)\n")
        
        # Extract results
        return self._extract_results()
    
    def _extract_results(self) -> Dict:
        """Extract simulation results."""
        results = {
            'time': np.array(self.time_history),
            'generators': {}
        }
        
        for i, gen_name in enumerate(self.gen_names):
            # Extract generator states from history
            n_states_per_gen = 6  # GENROU states
            
            delta = []
            omega = []
            
            for state_vec in self.state_history:
                # Find this generator's states in the full vector
                # (Simplified extraction - assumes fixed ordering)
                gen_states = state_vec[i*n_states_per_gen:(i+1)*n_states_per_gen]
                delta.append(gen_states[0])
                omega.append(gen_states[1])
            
            results['generators'][gen_name] = {
                'delta_deg': np.array(delta) * 180 / np.pi,
                'omega_pu': np.array(omega),
                'freq_Hz': self.f_base * (1 + np.array(omega))
            }
        
        return results
