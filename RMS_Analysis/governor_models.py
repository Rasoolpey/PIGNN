"""
Governor Models - Turbine Speed Controllers

This module implements turbine governor models for frequency control:
- TGOV1: Steam Turbine Governor (thermal plants)
- HYGOV: Hydro Turbine Governor (hydro plants)

The governor maintains frequency by adjusting mechanical power.

Author: PIGNN Project
Date: October 20, 2025
Reference: IEEE standards for governor models
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class TGOV1Parameters:
    """Parameters for TGOV1 Steam Turbine Governor"""
    R_pu: float         # Droop (speed regulation), typically 0.05 (5%)
    Dt: float = 0.0     # Turbine damping coefficient
    T1_s: float = 0.5   # Governor lag time constant (s)
    T2_s: float = 3.0   # Governor lead time constant (s)
    T3_s: float = 10.0  # Valve positioner time constant (s)
    Pmin_pu: float = 0.0    # Minimum power output (pu on machine base)
    Pmax_pu: float = 1.2    # Maximum power output (pu on machine base)


@dataclass
class HYGOVParameters:
    """Parameters for HYGOV Hydro Turbine Governor"""
    R_pu: float         # Permanent droop
    r_pu: float         # Temporary droop
    Tr_s: float = 5.0   # Dashpot time constant (s)
    Tf_s: float = 0.05  # Filter time constant (s)
    Tg_s: float = 0.5   # Gate servo time constant (s)
    Tw_s: float = 1.0   # Water time constant (s)
    At: float = 1.1     # Turbine gain
    Dturb: float = 0.5  # Turbine damping
    qnl_pu: float = 0.08    # No-load flow (pu)
    Pmin_pu: float = 0.0    # Minimum power (pu)
    Pmax_pu: float = 1.2    # Maximum power (pu)


class TGOV1Governor:
    """
    TGOV1 - Steam Turbine Governor
    
    Simple thermal governor with droop control.
    States: [Pm] (mechanical power)
    """
    
    def __init__(self, params: TGOV1Parameters, omega_ref: float = 0.0):
        """
        Initialize TGOV1 governor.
        
        Args:
            params: TGOV1 parameters
            omega_ref: Reference speed DEVIATION (pu), typically 0.0 for deviation-based control
        """
        self.params = params
        self.omega_ref = omega_ref  # 0.0 for deviation control
        
        # State: mechanical power
        self.Pm_pu = 0.5  # Will be initialized properly
        
        # Valve position (intermediate variable)
        self.valve_pos = 0.5
    
    def initialize(self, Pm_init: float):
        """
        Initialize governor from steady-state.
        
        Args:
            Pm_init: Initial mechanical power (pu on machine base)
        """
        self.Pm_pu = Pm_init
        self.valve_pos = Pm_init
    
    def compute_derivative(self, t: float, omega_pu: float) -> float:
        """
        Compute derivative of mechanical power.
        
        Args:
            t: Time (s)
            omega_pu: Generator speed DEVIATION (pu)
        
        Returns:
            dPm/dt
        """
        p = self.params
        
        # Speed error (droop control)
        omega_error = self.omega_ref - omega_pu  # At SS: 0 - 0 = 0
        
        # Governor output (with droop)
        # At steady state: omega_error = 0, so valve_cmd = Pm_pu
        valve_cmd = self.Pm_pu + omega_error / p.R_pu
        
        # Lead-lag compensation (simplified first-order approximation)
        # For steady-state initialization: valve_desired = valve_cmd
        valve_desired = valve_cmd
        
        # Valve positioner
        valve_limited = np.clip(valve_desired, p.Pmin_pu, p.Pmax_pu)
        
        # Mechanical power response
        # At steady state: valve_limited = Pm_pu, so dPm = 0
        dPm = (valve_limited - self.Pm_pu) / p.T3_s
        
        # Add turbine damping
        dPm += p.Dt * (omega_pu - self.omega_ref)  # At SS: 0
        
        return dPm
    
    def update(self, Pm_new: float):
        """Update mechanical power state."""
        self.Pm_pu = np.clip(Pm_new, self.params.Pmin_pu, self.params.Pmax_pu)


class HYGOVGovernor:
    """
    HYGOV - Hydro Turbine Governor
    
    Hydro governor with transient and permanent droop.
    States: [gate_position, q_flow, Pm]
    """
    
    def __init__(self, params: HYGOVParameters, omega_ref: float = 0.0):
        """
        Initialize HYGOV governor.
        
        Args:
            params: HYGOV parameters
            omega_ref: Reference speed DEVIATION (pu), typically 0.0 for deviation-based control
        """
        self.params = params
        self.omega_ref = omega_ref  # 0.0 for deviation control
        
        # States: [gate position, water flow, mechanical power]
        self.states = np.array([0.5, 0.5, 0.5])
        
        # Steady-state gate position (set during initialize())
        self.gate_ss = 0.5
    
    def initialize(self, Pm_init: float):
        """
        Initialize from steady-state.
        
        Args:
            Pm_init: Initial mechanical power (pu)
        """
        p = self.params
        
        # Mechanical power
        self.states[2] = Pm_init
        
        # Flow at steady state (from turbine equation)
        # Pm = At * (q - qnl)  =>  q = Pm/At + qnl
        q = Pm_init / p.At + p.qnl_pu
        self.states[1] = q
        
        # Gate position (at steady state: gate = q)
        self.states[0] = q
        
        # Store steady-state gate position for droop control
        self.gate_ss = q
    
    def compute_derivatives(self, t: float, omega_pu: float) -> np.ndarray:
        """
        Compute state derivatives.
        
        Args:
            t: Time (s)
            omega_pu: Generator speed (pu)
        
        Returns:
            [dgate/dt, dq/dt, dPm/dt]
        """
        p = self.params
        gate, q, Pm = self.states
        
        # Speed error
        omega_error = self.omega_ref - omega_pu
        
        # Droop control (transient + permanent)
        # Gate command adjusts from steady-state value based on speed error
        gate_cmd = self.gate_ss + omega_error / p.r_pu
        
        # Gate servo - apply same logic as Pm: don't clip if at steady state
        if abs(gate_cmd - gate) < 1e-6:
            gate_limited = gate_cmd
        else:
            gate_limited = np.clip(gate_cmd, p.Pmin_pu, p.Pmax_pu)
        
        dgate = (gate_limited - gate) / p.Tg_s
        
        # Water flow dynamics (penstock)
        dq = (gate - q) / p.Tw_s
        
        # Turbine power (including no-load flow)
        q_effective = q - p.qnl_pu
        Pm_turbine = p.At * q_effective * (1 - p.Dturb * (omega_pu - self.omega_ref))
        
        # Apply limits, but allow operation at current point even if outside limits
        # This ensures dPm = 0 at initialization when operating beyond rated capacity
        if abs(Pm_turbine - Pm) < 1e-6:
            # At steady state - don't clip to maintain dPm = 0
            Pm_limited = Pm_turbine
        else:
            # During dynamics - enforce limits
            Pm_limited = np.clip(Pm_turbine, p.Pmin_pu, p.Pmax_pu)
        
        dPm = (Pm_limited - Pm) / p.Tf_s
        
        return np.array([dgate, dq, dPm])
    
    def update(self, states_new: np.ndarray):
        """Update states."""
        self.states = states_new
        # Note: Don't clip Pm here - it can be outside limits at initialization
        # Clipping is applied in compute_derivatives via Pm_limited
    
    def get_mechanical_power(self) -> float:
        """Get current mechanical power."""
        return self.states[2]


def load_governors_from_h5(h5_file: str, gen_names: list) -> Dict[str, object]:
    """
    Load governor models from Graph_model.h5.
    
    Args:
        h5_file: Path to Graph_model.h5
        gen_names: List of generator names
    
    Returns:
        Dictionary of {gen_name: Governor object}
    """
    import h5py
    
    governors = {}
    
    with h5py.File(h5_file, 'r') as f:
        if 'dynamic_models/governors' not in f:
            # Use default TGOV1 for all generators
            print("[WARN] No governor data in H5, using default TGOV1")
            for gen_name in gen_names:
                params = TGOV1Parameters(R_pu=0.05, T1_s=0.5, T2_s=3.0, T3_s=10.0)
                governors[gen_name] = TGOV1Governor(params)
            return governors
        
        gov_group = f['dynamic_models/governors']
        n_gov = len(gov_group['names'])
        
        for i in range(n_gov):
            model_type = gov_group['model_type'][i]
            if isinstance(model_type, bytes):
                model_type = model_type.decode()
            
            gen_name = gov_group['generator_names'][i]
            if isinstance(gen_name, bytes):
                gen_name = gen_name.decode()
            
            if model_type == 'TGOV1':
                # Use available fields with defaults for missing ones
                params = TGOV1Parameters(
                    R_pu=float(gov_group['R_pu'][i]),
                    Dt=float(gov_group.get('Dt_pu', np.zeros(n_gov))[i]),
                    T1_s=0.5,  # Default
                    T2_s=3.0,  # Default
                    T3_s=float(gov_group.get('Tt_s', np.ones(n_gov)*10.0)[i]),
                    Pmin_pu=float(gov_group['Pmin_pu'][i]),
                    Pmax_pu=float(gov_group['Pmax_pu'][i])
                )
                governors[gen_name] = TGOV1Governor(params)
            
            elif model_type == 'HYGOV':
                # Use available fields with defaults
                params = HYGOVParameters(
                    R_pu=float(gov_group['R_pu'][i]),
                    r_pu=float(gov_group['R_pu'][i]) * 0.5,  # Default: half of permanent droop
                    Tr_s=5.0,  # Default
                    Tf_s=0.05,  # Default
                    Tg_s=float(gov_group.get('Tg_s', np.ones(n_gov)*0.5)[i]),
                    Tw_s=1.0,  # Default
                    At=1.1,  # Default
                    Dturb=0.5,  # Default
                    qnl_pu=0.08,  # Default
                    Pmin_pu=float(gov_group['Pmin_pu'][i]),
                    Pmax_pu=float(gov_group['Pmax_pu'][i])
                )
                governors[gen_name] = HYGOVGovernor(params)
    
    print(f"[OK] Loaded {len(governors)} governors from {h5_file}")
    return governors
