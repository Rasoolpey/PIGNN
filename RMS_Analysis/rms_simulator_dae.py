"""
RMS Simulator - DAE VERSION with Implicit Trapezoid Solver

Complete rewrite using DAE framework for stability and accuracy.

Key changes from old version:
1. Uses DAESystem infrastructure (dae_system.py)
2. Implicit trapezoid solver instead of RK4
3. Proper handling of differential + algebraic equations
4. Network equations will be added (currently using fixed bus voltages from load flow)

Author: PIGNN Project
Date: October 20, 2025
Status: UNDER DEVELOPMENT - Transitioning to full DAE implementation
"""

import numpy as np
import h5py
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from RMS_Analysis.generator_models import GENROUGenerator, GENROUParameters
from RMS_Analysis.exciter_models import SEXSExciter, IEEEAC1AExciter, SEXSParameters, IEEEAC1AParameters
from RMS_Analysis.governor_models import TGOV1Governor, HYGOVGovernor, TGOV1Parameters, HYGOVParameters
from RMS_Analysis.dae_system import DAESystem
from RMS_Analysis.dae_integrator import ImplicitTrapezoidDAE


@dataclass
class FaultEvent:
    """Fault event specification."""
    bus_id: int
    t_start: float
    t_clear: float
    fault_type: str = "3phase"  # 3phase, line-to-ground, etc.


class RMSSimulator:
    """
    RMS (Root Mean Square) Dynamic Simulator using DAE framework.
    
    Features:
    - GENROU 6th-order generator model
    - SEXS/IEEEAC1A exciter models
    - TGOV1/HYGOV governor models
    - Implicit trapezoid DAE solver (stable for stiff systems)
    - Load flow initialization (real PyPSA Newton-Raphson)
    """
    
    def __init__(self, h5_file: str, dt: float = 0.005, f_base: float = 60.0):
        """
        Initialize RMS simulator.
        
        Args:
            h5_file: Path to H5 file with graph model
            dt: Time step (s) - smaller for implicit solver
            f_base: Base frequency (Hz)
        """
        self.h5_file = h5_file
        self.dt = dt
        self.f_base = f_base
        self.ws = 2 * np.pi * f_base  # Synchronous speed (rad/s)
        self.base_MVA = 100.0  # System base MVA
        
        # Component models
        self.generators = {}
        self.exciters = {}
        self.governors = {}
        
        # Network data
        self.bus_ids = None
        self.gen_bus_ids = None
        self.gen_names = []
        
        # Load flow results
        self.V_mag = None
        self.V_angle = None
        
        # DAE system
        self.dae = None
        self.integrator = None
        
        # Simulation time
        self.t = 0.0
        
        # Simulation history
        self.time_history = []
        self.state_history = []
        
        print(f"[OK] RMS Simulator initialized (dt={dt*1000:.1f}ms, f={f_base}Hz)")
    
    def initialize(self):
        """Initialize simulator: run load flow, create models, set initial conditions."""
        print("\n" + "="*70)
        print("RMS SIMULATOR INITIALIZATION")
        print("="*70)
        
        # Step 1: Run load flow (real PyPSA solver)
        print("\n[1/5] Running load flow (PyPSA Newton-Raphson)...")
        self._run_load_flow_and_save()
        
        # Step 2: Load network data
        print("\n[2/5] Loading network data...")
        self._load_network_data()
        
        # Step 3: Create component models
        print("\n[3/5] Creating component models...")
        self._create_models()
        
        # Step 4: Initialize component models from load flow
        print("\n[4/5] Initializing generator models...")
        self._initialize_models()
        
        # Step 5: Setup DAE system
        print("\n[5/5] Setting up DAE system...")
        self._setup_dae_system()
        
        print("\n" + "="*70)
        print("[OK] INITIALIZATION COMPLETE")
        print("="*70)
    
    def _run_load_flow_and_save(self):
        """Run REAL PyPSA Newton-Raphson load flow and save to H5."""
        from physics.load_flow_solver import run_load_flow_from_h5
        
        print("  Running PyPSA Newton-Raphson...")
        results = run_load_flow_from_h5(
            h5_path=self.h5_file,
            tolerance=1e-6,
            max_iterations=50,
            save_to_h5=True  # Automatically saves to H5
        )
        
        if not results.converged:
            raise RuntimeError(f"❌ Load flow did NOT converge after {results.iterations} iterations!")
        
        print(f"  [OK] Load flow converged:")
        print(f"     Iterations: {results.iterations}")
        print(f"     Max mismatch: {results.max_mismatch:.2e} pu")
        print(f"     System losses: {results.total_losses_mw:.2f} MW")
    
    def _load_network_data(self):
        """Load network topology and load flow results from H5."""
        with h5py.File(self.h5_file, 'r') as f:
            # Generator data (from dynamic_models/generators)
            gen_grp = f['dynamic_models/generators']
            self.gen_names = [name.decode('utf-8') for name in gen_grp['names'][:]]
            gen_bus_names = [name.decode('utf-8') for name in gen_grp['buses'][:]]
            
            # Get unique bus IDs from generator buses
            # For now, use generator buses as our bus list (simplified)
            unique_buses = sorted(set(gen_bus_names))
            self.bus_ids = np.array([int(b.split()[1]) for b in unique_buses])  # Extract bus numbers
            self.gen_bus_ids = np.array([int(b.split()[1]) for b in gen_bus_names])
            
            # Load flow results (saved by run_load_flow_from_h5)
            pf_grp = f['steady_state/power_flow_results']
            
            # Bus voltages and angles (39 buses total) - STORE FOR ALL BUSES
            self.V_loadflow_pu = pf_grp['bus_voltages_pu'][:]
            self.theta_loadflow_rad = pf_grp['bus_angles_deg'][:] * np.pi / 180
            
            # Generator outputs (bus-indexed array with 39 entries)
            gen_P_MW_by_bus = pf_grp['gen_P_MW'][:]
            gen_Q_MVAr_by_bus = pf_grp['gen_Q_MVAR'][:]
            
            # Map generator data: extract only the generator buses
            num_gens = len(self.gen_names)
            self.V_mag = np.zeros(num_gens)
            self.V_angle = np.zeros(num_gens)
            self.gen_P_MW = np.zeros(num_gens)
            self.gen_Q_MVAr = np.zeros(num_gens)
            
            for i, gen_bus_id in enumerate(self.gen_bus_ids):
                # Find this bus in the full bus array
                # Bus IDs are like 30, 31, 32, ... 39 (for IEEE 39)
                # Array index is bus_id - 1 (assuming 1-indexed bus IDs)
                bus_array_idx = gen_bus_id - 1
                
                self.V_mag[i] = self.V_loadflow_pu[bus_array_idx]
                self.V_angle[i] = self.theta_loadflow_rad[bus_array_idx]
                self.gen_P_MW[i] = gen_P_MW_by_bus[bus_array_idx]
                self.gen_Q_MVAr[i] = gen_Q_MVAr_by_bus[bus_array_idx]
        
        print(f"  Loaded {len(self.bus_ids)} buses, {len(self.gen_names)} generators")
        print(f"  Generator outputs: {self.gen_P_MW.min():.0f} - {self.gen_P_MW.max():.0f} MW")
        
        # Build network admittance matrix
        self._build_admittance_matrix()
    
    def _build_admittance_matrix(self):
        """Build FULL Y-matrix for ALL 39 buses (no Kron reduction)."""
        with h5py.File(self.h5_file, 'r') as f:
            # Load branch data from phase_a (positive sequence)
            edges = f['phases/phase_a/edges']
            from_bus = edges['from_bus'][:]
            to_bus = edges['to_bus'][:]
            R_pu = edges['R_pu'][:]
            X_pu = edges['X_pu'][:]
            B_shunt_pu = edges['B_shunt_pu'][:]
            in_service = edges['in_service'][:]
            
            # Total number of buses in system
            self.n_buses = 39
            
            # Build FULL Y-matrix (39x39) - NO REDUCTION
            self.Y_matrix = np.zeros((self.n_buses, self.n_buses), dtype=complex)
            
            for i in range(len(from_bus)):
                if not in_service[i]:
                    continue
                
                fb = from_bus[i]
                tb = to_bus[i]
                
                # Series admittance
                Z = R_pu[i] + 1j * X_pu[i]
                if abs(Z) < 1e-10:
                    Y_series = 1e10  # Very large (short circuit)
                else:
                    Y_series = 1.0 / Z
                
                # Shunt admittance (split half at each end)
                Y_shunt = 1j * B_shunt_pu[i] / 2.0
                
                # Add to Y-matrix (bus indices are 0-based in arrays)
                self.Y_matrix[fb, fb] += Y_series + Y_shunt
                self.Y_matrix[tb, tb] += Y_series + Y_shunt
                self.Y_matrix[fb, tb] -= Y_series
                self.Y_matrix[tb, fb] -= Y_series
            
            # Store magnitude and angle for easier computation
            self.Y_mag = np.abs(self.Y_matrix)
            self.Y_ang = np.angle(self.Y_matrix)
            
            # Create mapping: which buses have generators
            self.gen_bus_mask = np.zeros(self.n_buses, dtype=bool)
            self.gen_bus_indices = self.gen_bus_ids - 1  # Convert to 0-based
            self.gen_bus_mask[self.gen_bus_indices] = True
            
            # Load data (for buses without generators)
            pf_grp = f['steady_state/power_flow_results']
            gen_P_MW_all = pf_grp['gen_P_MW'][:]
            gen_Q_MVAr_all = pf_grp['gen_Q_MVAR'][:]
            
            # Net load at each bus (positive = load, negative = generation)
            self.bus_P_load_MW = -gen_P_MW_all  # Negative because gen_P is injection
            self.bus_Q_load_MVAr = -gen_Q_MVAr_all
            
            # For generator buses, we'll handle power separately in the DAE
            # Zero out loads at generator buses (generators will provide power)
            self.bus_P_load_MW[self.gen_bus_indices] = 0.0
            self.bus_Q_load_MVAr[self.gen_bus_indices] = 0.0
            
            print(f"  Built FULL Y-matrix: {self.n_buses}x{self.n_buses} (NO Kron reduction)")
            print(f"  Generator buses: {len(self.gen_bus_indices)}")
            print(f"  Load buses: {self.n_buses - len(self.gen_bus_indices)}")
            print(f"  Max |Y_diag|: {np.max(np.abs(np.diag(self.Y_matrix))):.4f}")
            print(f"  Max |Y_off|: {np.max(np.abs(self.Y_matrix - np.diag(np.diag(self.Y_matrix)))):.4f}")
    
    def _create_models(self):
        """Create generator, exciter, governor models from H5 data."""
        with h5py.File(self.h5_file, 'r') as f:
            gen_grp = f['dynamic_models/generators']
            exc_grp = f['dynamic_models/exciters']
            gov_grp = f['dynamic_models/governors']
            
            for i, gen_name in enumerate(self.gen_names):
                # Generator model (GENROU)
                gen_params = GENROUParameters(
                    gen_name=gen_name,
                    Sn_MVA=gen_grp['Sn_MVA'][i],
                    H_s=gen_grp['H_s'][i],
                    D_pu=gen_grp['D_pu'][i],
                    xd_pu=gen_grp['xd_pu'][i],
                    xq_pu=gen_grp['xq_pu'][i],
                    xd_prime_pu=gen_grp['xd_prime_pu'][i],
                    xq_prime_pu=gen_grp['xq_prime_pu'][i],
                    xd_double_pu=gen_grp['xd_double_prime_pu'][i],
                    xq_double_pu=gen_grp['xq_double_prime_pu'][i],
                    xl_pu=gen_grp['xl_pu'][i],
                    ra_pu=gen_grp['ra_pu'][i],
                    Td0_prime_s=gen_grp['Td0_prime_s'][i],
                    Tq0_prime_s=gen_grp['Tq0_prime_s'][i],
                    Td0_double_prime_s=gen_grp['Td0_double_prime_s'][i],
                    Tq0_double_prime_s=gen_grp['Tq0_double_prime_s'][i],
                    S10=gen_grp['S10'][i],
                    S12=gen_grp['S12'][i]
                )
                self.generators[gen_name] = GENROUGenerator(gen_params)
                
                # Exciter model
                exc_model = exc_grp['model_type'][i].decode('utf-8')
                if exc_model == 'SEXS':
                    exc_params = SEXSParameters(
                        Ta_s=0.1,  # Default AVR time constant
                        Te_s=exc_grp['Te_s'][i],
                        Efd_min=exc_grp['Efd_min'][i],
                        Efd_max=exc_grp['Efd_max'][i]
                    )
                    self.exciters[gen_name] = SEXSExciter(exc_params)
                elif exc_model == 'IEEEAC1A':
                    exc_params = IEEEAC1AParameters(
                        Ka=200.0,  # Default regulator gain
                        Ta_s=0.02,  # Default regulator time constant
                        Ke=exc_grp['Ke'][i],
                        Te_s=exc_grp['Te_s'][i],
                        Kf=exc_grp['Kf'][i],
                        Tf_s=exc_grp['Tf_s'][i],
                        Efd_min=exc_grp['Efd_min'][i],
                        Efd_max=exc_grp['Efd_max'][i],
                        Vr_min=-5.0,  # Default limits
                        Vr_max=5.0
                    )
                    self.exciters[gen_name] = IEEEAC1AExciter(exc_params)
                
                # Governor model
                gov_model = gov_grp['model_type'][i].decode('utf-8')
                if gov_model == 'TGOV1':
                    gov_params = TGOV1Parameters(
                        R_pu=gov_grp['R_pu'][i],
                        Dt=gov_grp['Dt_pu'][i],
                        T1_s=0.5,  # Default lag time constant
                        T2_s=3.0,  # Default lead time constant  
                        T3_s=gov_grp['Tg_s'][i],
                        Pmin_pu=0.0,
                        Pmax_pu=1.2
                    )
                    self.governors[gen_name] = TGOV1Governor(gov_params)
                elif gov_model == 'HYGOV':
                    gov_params = HYGOVParameters(
                        R_pu=gov_grp['R_pu'][i],
                        r_pu=0.05,  # Default temporary droop
                        Tr_s=5.0,  # Default reset time
                        Tf_s=0.05,  # Default filter time
                        Tg_s=gov_grp['Tg_s'][i],
                        Tw_s=1.0,  # Default water time constant
                        At=1.2,  # Default turbine gain
                        Dturb=0.5,  # Default turbine damping
                        qnl_pu=0.08,  # Default no-load flow
                        Pmin_pu=0.0,  # Default limits
                        Pmax_pu=1.2
                    )
                    self.governors[gen_name] = HYGOVGovernor(gov_params)
        
        print(f"  Created {len(self.generators)} generators")
        print(f"  Created {len(self.exciters)} exciters")
        print(f"  Created {len(self.governors)} governors")
    
    def _initialize_models(self):
        """Initialize all models from load flow results."""
        print("\n  Initializing from load flow:")
        
        for i, gen_name in enumerate(self.gen_names):
            # Get load flow data for this generator
            # V_mag and V_angle are already indexed by generator (not bus!)
            Vt_pu = self.V_mag[i]
            theta_rad = self.V_angle[i]
            
            # Generator rated power
            Sn_MVA = self.generators[gen_name].params.Sn_MVA
            
            # Convert MW/MVAr to pu
            P_pu = self.gen_P_MW[i] / Sn_MVA
            Q_pu = self.gen_Q_MVAr[i] / Sn_MVA
            
            # Initialize generator (uses ANDES complex phasor method)
            self.generators[gen_name].initialize(P_pu, Q_pu, Vt_pu, theta_rad)
            
            # Initialize exciter (use generator's computed Efd from algebraic[4])
            Efd_init = self.generators[gen_name].algebraic[4]  # Efd from generator init
            self.exciters[gen_name].initialize(Vt_pu, Efd_init)
            
            # Initialize governor (Pm = Pe at steady state)
            Pm_init = P_pu  # At steady state, Pm = Pe (no damping term)
            self.governors[gen_name].initialize(Pm_init)
            
            # Print summary for first 3 generators
            if i < 3:
                print(f"    {gen_name}: P={P_pu:.3f}pu ({self.gen_P_MW[i]:.0f}MW), Q={Q_pu:.3f}pu, "
                      f"Vt={Vt_pu:.4f}pu, delta={np.rad2deg(self.generators[gen_name].states[0]):.2f}°")
        
        if len(self.gen_names) > 3:
            print(f"    ... ({len(self.gen_names)-3} more generators)")
    
    def _setup_dae_system(self):
        """Setup DAE system with differential and algebraic states."""
        self.dae = DAESystem()
        
        # Count states
        n_diff = 0  # Differential states
        
        for gen_name in self.gen_names:
            # Generator: 6 differential states
            n_diff += 6
            
            # Exciter
            exc = self.exciters[gen_name]
            if isinstance(exc, SEXSExciter):
                n_diff += 1  # Efd
            elif isinstance(exc, IEEEAC1AExciter):
                n_diff += 3  # Vr, Efd, Vf
            
            # Governor
            gov = self.governors[gen_name]
            if isinstance(gov, TGOV1Governor):
                n_diff += 1  # Pm
            elif isinstance(gov, HYGOVGovernor):
                n_diff += 3  # gate, q, Pm
        
        # Algebraic states: ALL 39 buses × 4 states [Vd, Vq, Id, Iq]
        n_alg = self.n_buses * 4  # 39 buses × 4 = 156 states
        
        print(f"  DAE system dimensions (FULL NETWORK):")
        print(f"    Differential states (x): {n_diff}")
        print(f"    Algebraic states (y):    {n_alg} (ALL 39 buses × 4)")
        print(f"    Total states:            {n_diff + n_alg}")
        
        # Initialize DAE system
        self.dae.n = n_diff
        self.dae.m = n_alg
        
        # Collect initial states
        x0 = self._collect_differential_states()
        y0 = self._collect_algebraic_states()
        
        self.dae.x = x0
        self.dae.y = y0
        
        # Setup time constants
        self.dae.Tf = self._collect_time_constants()
        self.dae.Teye = np.diag(self.dae.Tf)
        
        # Create DAE integrator (implicit trapezoid)
        self.integrator = ImplicitTrapezoidDAE(
            dt=self.dt,
            max_iter=15,
            tol=0.5  # Relaxed tolerance for large system (initial g~0.35)
        )
        
        print(f"  Integrator: Implicit Trapezoid (dt={self.dt*1000:.1f}ms, tol=1e-4)")
    
    def _collect_differential_states(self) -> np.ndarray:
        """Collect all differential states into single vector."""
        states = []
        
        for gen_name in self.gen_names:
            # Generator states (6)
            gen = self.generators[gen_name]
            states.extend(gen.states)
            
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
    
    def _collect_algebraic_states(self) -> np.ndarray:
        """
        Collect algebraic states for ALL 39 buses: [Vd, Vq, Id, Iq] per bus.
        
        For generator buses: Use generator terminal voltages and currents
        For load buses: Use load flow voltages, zero currents (constant power load)
        
        Returns:
            y: Array of algebraic states [Vd0, Vq0, Id0, Iq0, Vd1, Vq1, Id1, Iq1, ...]
                Shape: (39 buses * 4 states/bus,) = (156,)
        """
        y = np.zeros(self.n_buses * 4)
        
        # Load flow voltages (for initialization)
        V_lf = self.V_loadflow_pu
        theta_lf = self.theta_loadflow_rad
        
        for bus_idx in range(self.n_buses):
            base_idx = bus_idx * 4
            
            # Check if this bus has a generator
            gen_idx = np.where(self.gen_bus_indices == bus_idx)[0]
            
            if len(gen_idx) > 0:
                # Generator bus: transform from rotor frame to network frame
                gen_name = self.gen_names[gen_idx[0]]
                gen = self.generators[gen_name]
                delta = gen.states[0]  # Rotor angle
                
                # Get from algebraic array (IN ROTOR FRAME): [Id, Iq, Vd, Vq, Efd, Pm]
                Id_rotor = gen.algebraic[0]
                Iq_rotor = gen.algebraic[1]
                Vd_rotor = gen.algebraic[2]
                Vq_rotor = gen.algebraic[3]
                
                # Transform to NETWORK frame for algebraic states
                cos_delta = np.cos(delta)
                sin_delta = np.sin(delta)
                
                # Inverse Park: rotor → network
                y[base_idx + 0] = Vd_rotor * cos_delta - Vq_rotor * sin_delta  # Vd
                y[base_idx + 1] = Vd_rotor * sin_delta + Vq_rotor * cos_delta  # Vq
                y[base_idx + 2] = Id_rotor * cos_delta - Iq_rotor * sin_delta  # Id
                y[base_idx + 3] = Id_rotor * sin_delta + Iq_rotor * cos_delta  # Iq
            else:
                # Load bus: use load flow voltage in dq frame
                # Reference angle = 0 (global reference frame)
                V_mag = V_lf[bus_idx]
                theta = theta_lf[bus_idx]
                
                # Convert to dq (d-axis aligned with global reference)
                y[base_idx + 0] = V_mag * np.cos(theta)  # Vd
                y[base_idx + 1] = V_mag * np.sin(theta)  # Vq
                
                # Currents: initialize to zero, will compute from Y-matrix below
                y[base_idx + 2] = 0.0  # Id
                y[base_idx + 3] = 0.0  # Iq
        
        # Second pass: compute load bus currents from Y-matrix
        self._initialize_load_currents(y)
        
        return y
    
    def _initialize_load_currents(self, y: np.ndarray):
        """Compute load bus currents from Y-matrix after voltages are initialized."""
        for bus_idx in range(self.n_buses):  # Use n_buses, not len(bus_ids)!
            # Skip generator buses
            if bus_idx in self.gen_bus_indices:
                continue
            
            # Compute network current for this load bus
            Id_net, Iq_net = self._compute_network_current(bus_idx, y)
            
            # Set algebraic states to match network
            base_idx = bus_idx * 4
            y[base_idx + 2] = Id_net
            y[base_idx + 3] = Iq_net
    
    def _collect_time_constants(self) -> np.ndarray:
        """Collect time constants for all differential states."""
        Tf = []
        
        for gen_name in self.gen_names:
            gen = self.generators[gen_name]
            exc = self.exciters[gen_name]
            gov = self.governors[gen_name]
            
            # Generator time constants
            Tf.extend(gen.get_time_constants())
            
            # Exciter time constants
            if isinstance(exc, SEXSExciter):
                Tf.append(exc.params.Te_s)
            elif isinstance(exc, IEEEAC1AExciter):
                Tf.extend([exc.params.Te_s, exc.params.Te_s, exc.params.Tf_s])
            
            # Governor time constants
            if isinstance(gov, TGOV1Governor):
                Tf.append(gov.params.T3_s)  # Valve positioner time constant
            elif isinstance(gov, HYGOVGovernor):
                Tf.extend([gov.params.Tg_s, gov.params.Tw_s, 1.0])
        
        return np.array(Tf)
    
    def _update_dae_equations(self, dae: DAESystem, t: float):
        """
        Update DAE equations f(x,y) and g(x,y).
        
        This is called by the integrator at each iteration.
        """
        # Distribute states back to models
        self._distribute_differential_states(dae.x)
        self._distribute_algebraic_states(dae.y)
        
        # Compute differential equations
        f = []
        offset = 0  # Track position in dae.x
        
        for i, gen_name in enumerate(self.gen_names):
            gen = self.generators[gen_name]
            exc = self.exciters[gen_name]
            gov = self.governors[gen_name]
            
            # Get generator bus index
            bus_idx = self.gen_bus_indices[i]
            base_y = bus_idx * 4
            
            # Get terminal currents and voltages from algebraic states (NETWORK frame)
            # y = [Vd0, Vq0, Id0, Iq0, Vd1, Vq1, Id1, Iq1, ...]
            Vd_network = dae.y[base_y + 0]
            Vq_network = dae.y[base_y + 1]
            Id_network = dae.y[base_y + 2]
            Iq_network = dae.y[base_y + 3]
            Vt_mag = np.sqrt(Vd_network**2 + Vq_network**2)
            
            # Transform currents from NETWORK frame to ROTOR frame for derivatives
            delta = gen.states[0]
            cos_delta = np.cos(delta)
            sin_delta = np.sin(delta)
            Id_rotor = Id_network * cos_delta + Iq_network * sin_delta
            Iq_rotor = -Id_network * sin_delta + Iq_network * cos_delta
            
            # Skip generator states (6 states)
            gen_states_idx = offset
            offset += 6
            
            # Get Efd from exciter states
            if isinstance(exc, SEXSExciter):
                # SEXS: x = [Efd] (1 state)
                Efd = dae.x[offset]
                offset += 1
            elif isinstance(exc, IEEEAC1AExciter):
                # IEEEAC1A: x = [Vr, Efd, Vf] (3 states)
                Efd = dae.x[offset + 1]  # Efd is second state
                offset += 3
            else:
                Efd = exc.Efd_pu  # Fallback
            
            # Get Pm from governor states
            if isinstance(gov, TGOV1Governor):
                # TGOV1: x = [Pm] (1 state)
                Pm = dae.x[offset]
                offset += 1
            elif isinstance(gov, HYGOVGovernor):
                # HYGOV: x = [gate, q, Pm] (3 states)
                Pm = dae.x[offset + 2]  # Pm is third state
                offset += 3
            else:
                Pm = gov.Pm_pu  # Fallback
            
            # Generator derivatives with ROTOR-FRAME CURRENTS
            gen_deriv = gen.derivatives(Pm, Efd, Id_rotor, Iq_rotor)
            f.extend(gen_deriv)
            
            # Exciter derivatives
            if isinstance(exc, SEXSExciter):
                exc_deriv = exc.compute_derivative(t, Vt_mag)
                f.append(exc_deriv)
            elif isinstance(exc, IEEEAC1AExciter):
                exc_deriv = exc.compute_derivatives(t, Vt_mag)
                f.extend(exc_deriv)
            
            # Governor derivatives
            omega = gen.states[1]  # Speed deviation (already in rad/s)
            omega_pu = omega / (2 * np.pi * 60.0)  # Normalized frequency deviation
            if isinstance(gov, TGOV1Governor):
                gov_deriv = gov.compute_derivative(t, omega_pu)
                f.append(gov_deriv)
            elif isinstance(gov, HYGOVGovernor):
                gov_deriv = gov.compute_derivatives(t, omega_pu)
                f.extend(gov_deriv)
        
        dae.f = np.array(f)
        
        # ============================================================
        # Algebraic equations: FULL NETWORK Current Balance at ALL 39 buses
        # ============================================================
        # For each bus k (0-38):
        #   I_net[k] = sum_j( Y[k,j] * V[j] ) = I_gen[k] - I_load[k]
        # 
        # Split into dq components (4 equations per bus):
        #   g[4*k + 0] = Vd[k] equation
        #   g[4*k + 1] = Vq[k] equation  
        #   g[4*k + 2] = Id[k] - Id_net[k] = 0  (current balance d-axis)
        #   g[4*k + 3] = Iq[k] - Iq_net[k] = 0  (current balance q-axis)
        
        g = np.zeros(self.n_buses * 4)
        
        for bus_idx in range(self.n_buses):
            base_idx = bus_idx * 4
            
            # Get bus voltage and current from algebraic states
            Vd = dae.y[base_idx + 0]
            Vq = dae.y[base_idx + 1]
            Id = dae.y[base_idx + 2]
            Iq = dae.y[base_idx + 3]
            
            V_mag = np.sqrt(Vd**2 + Vq**2)
            theta = np.arctan2(Vq, Vd)
            
            # Check if this bus has a generator
            gen_idx = np.where(self.gen_bus_indices == bus_idx)[0]
            
            if len(gen_idx) > 0:
                # ========== GENERATOR BUS ==========
                gen_name = self.gen_names[gen_idx[0]]
                gen = self.generators[gen_name]
                delta = gen.states[0]  # Rotor angle
                
                # Generator stores algebraic variables in ROTOR frame
                # Get from generator: [Id_rotor, Iq_rotor, Vd_rotor, Vq_rotor, Efd, Pm]
                Id_rotor = gen.algebraic[0]
                Iq_rotor = gen.algebraic[1]
                Vd_rotor = gen.algebraic[2]
                Vq_rotor = gen.algebraic[3]
                
                # Transform from ROTOR frame to NETWORK frame
                # Inverse Park transformation: multiply by e^(+j*delta)
                cos_delta = np.cos(delta)
                sin_delta = np.sin(delta)
                
                # Rotor → Network transformation
                Id_network = Id_rotor * cos_delta - Iq_rotor * sin_delta
                Iq_network = Id_rotor * sin_delta + Iq_rotor * cos_delta
                Vd_network = Vd_rotor * cos_delta - Vq_rotor * sin_delta
                Vq_network = Vd_rotor * sin_delta + Vq_rotor * cos_delta
                
                # Generator bus equations (SIMPLE INTERFACE FORMULATION):
                # The generator model internally computes Id, Iq, Vd, Vq using circuit equations.
                # We simply copy these to the network algebraic states.
                # This creates an "interface" where generator dictates its terminal conditions.
                
                # Voltage: Bus voltage = Generator's computed terminal voltage
                g[base_idx + 0] = Vd - Vd_network
                g[base_idx + 1] = Vq - Vq_network
                
                # Current: Bus current = Generator's computed current
                g[base_idx + 2] = Id - Id_network
                g[base_idx + 3] = Iq - Iq_network
                
            else:
                # ========== LOAD BUS ==========
                # For load buses: fix voltage at load flow, enforce network current balance
                V_lf = self.V_loadflow_pu[bus_idx]
                theta_lf = self.theta_loadflow_rad[bus_idx]
                
                # Fix voltage at load flow values
                g[base_idx + 0] = Vd - V_lf * np.cos(theta_lf)
                g[base_idx + 1] = Vq - V_lf * np.sin(theta_lf)
                
                # Network current from Y-matrix
                Id_net, Iq_net = self._compute_network_current(bus_idx, dae.y)
                
                # Current balance: network injection = load consumption
                # This enforces I_net = I_load through the network equations
                g[base_idx + 2] = Id - Id_net
                g[base_idx + 3] = Iq - Iq_net
        
        dae.g = g
    
    def _compute_network_current(self, bus_idx: int, y_states: np.ndarray) -> tuple:
        """
        Compute network current injection at bus_idx using Y-matrix.
        
        I_net[k] = sum_j( Y[k,j] * V[j] )
        
        Args:
            bus_idx: Bus index (0-38)
            y_states: Algebraic state vector [Vd0, Vq0, Id0, Iq0, Vd1, Vq1, ...]
        
        Returns:
            (Id_net, Iq_net): Current injection in dq frame
        """
        I_complex = 0.0 + 0.0j
        
        for j in range(self.n_buses):
            # Get voltage at bus j
            base_j = j * 4
            Vd_j = y_states[base_j + 0]
            Vq_j = y_states[base_j + 1]
            V_j_complex = Vd_j + 1j * Vq_j
            
            # Current contribution from bus j
            Y_kj = self.Y_matrix[bus_idx, j]
            I_complex += Y_kj * V_j_complex
        
        # Split into dq components
        Id_net = I_complex.real
        Iq_net = I_complex.imag
        
        return Id_net, Iq_net
    
    def _update_dae_jacobians(self, dae: DAESystem, t: float):
        """
        Update Jacobian matrices: fx, fy, gx, gy.
        
        This is called by the integrator when building Ac matrix.
        """
        # TODO: Implement analytical Jacobians
        # For now: use finite differences (SLOW but works)
        
        eps = 1e-7
        n, m = dae.n, dae.m
        
        # Save current f, g
        self._update_dae_equations(dae, t)
        f0 = dae.f.copy()
        g0 = dae.g.copy()
        
        # ∂f/∂x
        fx = np.zeros((n, n))
        for i in range(n):
            x_save = dae.x[i]
            dae.x[i] += eps
            self._update_dae_equations(dae, t)
            fx[:, i] = (dae.f - f0) / eps
            dae.x[i] = x_save
        
        # ∂f/∂y
        fy = np.zeros((n, m))
        self._update_dae_equations(dae, t)  # Restore
        for i in range(m):
            y_save = dae.y[i]
            dae.y[i] += eps
            self._update_dae_equations(dae, t)
            fy[:, i] = (dae.f - f0) / eps
            dae.y[i] = y_save
        
        # ∂g/∂x
        gx = np.zeros((m, n))
        self._update_dae_equations(dae, t)  # Restore
        for i in range(n):
            x_save = dae.x[i]
            dae.x[i] += eps
            self._update_dae_equations(dae, t)
            gx[:, i] = (dae.g - g0) / eps
            dae.x[i] = x_save
        
        # ∂g/∂y
        gy = np.zeros((m, m))
        self._update_dae_equations(dae, t)  # Restore
        for i in range(m):
            y_save = dae.y[i]
            dae.y[i] += eps
            self._update_dae_equations(dae, t)
            gy[:, i] = (dae.g - g0) / eps
            dae.y[i] = y_save
        
        # Restore
        self._update_dae_equations(dae, t)
        
        # Store in DAE system (convert to sparse)
        from scipy import sparse
        dae.fx = sparse.csr_matrix(fx)
        dae.fy = sparse.csr_matrix(fy)
        dae.gx = sparse.csr_matrix(gx)
        dae.gy = sparse.csr_matrix(gy)
    
    def _distribute_differential_states(self, x: np.ndarray):
        """Distribute differential state vector to component models."""
        idx = 0
        
        for gen_name in self.gen_names:
            gen = self.generators[gen_name]
            gen.states = x[idx:idx+6]
            idx += 6
            
            exc = self.exciters[gen_name]
            if isinstance(exc, SEXSExciter):
                exc.Efd_pu = x[idx]
                idx += 1
            elif isinstance(exc, IEEEAC1AExciter):
                exc.states = x[idx:idx+3]
                idx += 3
            
            gov = self.governors[gen_name]
            if isinstance(gov, TGOV1Governor):
                gov.Pm_pu = x[idx]
                idx += 1
            elif isinstance(gov, HYGOVGovernor):
                gov.states = x[idx:idx+3]
                idx += 3
    
    def _distribute_algebraic_states(self, y: np.ndarray):
        """
        Distribute algebraic state vector to generators.
        
        y = [Vd0, Vq0, Id0, Iq0, Vd1, Vq1, Id1, Iq1, ...] for ALL 39 buses
        
        For generator buses: update generator's Vd, Vq, Id, Iq
        For load buses: just network states (used in equations)
        """
        for gen_idx in range(len(self.gen_names)):
            bus_idx = self.gen_bus_indices[gen_idx]
            base_idx = bus_idx * 4
            
            gen_name = self.gen_names[gen_idx]
            gen = self.generators[gen_name]
            gen.Vd_pu = y[base_idx + 0]
            gen.Vq_pu = y[base_idx + 1]
            gen.Id_pu = y[base_idx + 2]
            gen.Iq_pu = y[base_idx + 3]
    
    def step(self) -> Tuple[bool, int, float]:
        """
        Take one time step forward.
        
        Returns:
            (converged, iterations, residual)
        """
        if self.dae is None:
            raise RuntimeError("Must call initialize() before step()")
        
        converged, iters, resid = self.integrator.step(
            self.t, self.dae,
            self._update_dae_equations,
            self._update_dae_jacobians,
            predictor='constant'
        )
        
        if converged:
            self.t += self.dt
        
        return converged, iters, resid
    
    def simulate(self, t_end: float, fault_events: List[FaultEvent] = None) -> Dict:
        """
        Run time-domain simulation using DAE framework.
        
        Args:
            t_end: End time (s)
            fault_events: List of fault events (not implemented yet)
        
        Returns:
            Dictionary with simulation results
        """
        print(f"\nStarting RMS DAE simulation: 0 -> {t_end}s (dt={self.dt*1000:.1f}ms)")
        print(f"Using Implicit Trapezoid solver\n")
        start_time = time.time()
        
        # Initialize if not done
        if self.dae is None:
            self.initialize()
        
        # Reset history
        t = 0.0
        self.time_history = [t]
        self.state_history = [self.dae.x.copy()]
        
        # Main integration loop
        n_steps = int(t_end / self.dt)
        print(f"Running {n_steps} time steps...\n")
        
        for step in range(n_steps):
            # Integration step using implicit trapezoid
            # Use 'constant' predictor to avoid Euler step with large initial f residuals
            converged, iters, resid = self.integrator.step(
                t, self.dae,
                self._update_dae_equations,
                self._update_dae_jacobians,
                predictor='constant'  # Skip Euler predictor, start Newton from x₀
            )
            
            if not converged:
                print(f"\n❌ WARNING: Step {step+1} did not converge!")
                print(f"   Stopping simulation at t={t:.3f}s")
                break
            
            # Update time
            t += self.dt
            
            # Store
            self.time_history.append(t)
            self.state_history.append(self.dae.x.copy())
            
            # Progress
            if (step + 1) % max(1, n_steps // 20) == 0:
                progress = (step + 1) / n_steps * 100
                print(f"  Progress: {progress:.0f}% (t={t:.2f}s, iters={iters}, ||q||={resid:.2e})")
        
        elapsed = time.time() - start_time
        print(f"\n[OK] Simulation complete in {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/s)")
        
        # Print solver statistics
        self.integrator.print_statistics()
        
        # Extract results
        return self._extract_results()
    
    def _extract_results(self) -> Dict:
        """Extract simulation results from history."""
        results = {
            'time': np.array(self.time_history),
            'generators': {}
        }
        
        for i, gen_name in enumerate(self.gen_names):
            # Extract this generator's states from full state vector
            # Generator states are first 6 per generator
            gen_idx_start = i * 6  # Simplified - assumes all generators come first
            
            delta = []
            omega = []
            
            for x_vec in self.state_history:
                gen_states = x_vec[gen_idx_start:gen_idx_start+6]
                delta.append(gen_states[0])
                omega.append(gen_states[1])
            
            delta_arr = np.array(delta)
            omega_arr = np.array(omega)
            
            results['generators'][gen_name] = {
                'delta_deg': delta_arr * 180 / np.pi,
                'omega_rad_s': omega_arr,
                'omega_pu': (omega_arr - self.ws) / self.ws,
                'freq_Hz': self.f_base + (omega_arr - self.ws) / (2*np.pi)
            }
        
        return results


if __name__ == "__main__":
    print(__doc__)
    print("\n✅ RMS Simulator (DAE version) loaded successfully!")
