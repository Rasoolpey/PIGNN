"""
Exciter Models - Automatic Voltage Regulators (AVR)

This module implements excitation system models for controlling generator field voltage:
- SEXS: Simplified Excitation System
- IEEEAC1A: IEEE Type AC1A Excitation System

The exciter maintains terminal voltage by adjusting field current.

Author: PIGNN Project
Date: October 20, 2025
Reference: IEEE Std 421.5-2016
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class SEXSParameters:
    """Parameters for SEXS (Simplified Excitation System)"""
    Ta_s: float         # AVR time constant (s)
    Tb_s: float = 0.0   # AVR lag time constant (s)
    K: float = 100.0    # AVR gain
    Te_s: float = 0.1   # Exciter time constant (s)
    Efd_min: float = 0.0    # Minimum field voltage (pu)
    Efd_max: float = 5.0    # Maximum field voltage (pu)


@dataclass
class IEEEAC1AParameters:
    """Parameters for IEEE Type AC1A Excitation System"""
    Ka: float           # Regulator gain
    Ta_s: float         # Regulator time constant (s)
    Ke: float           # Exciter constant
    Te_s: float         # Exciter time constant (s)
    Kf: float           # Stabilizer gain
    Tf_s: float         # Stabilizer time constant (s)
    Efd_min: float      # Minimum field voltage (pu)
    Efd_max: float      # Maximum field voltage (pu)
    Vr_min: float       # Minimum regulator output (pu)
    Vr_max: float       # Maximum regulator output (pu)


class SEXSExciter:
    """
    SEXS - Simplified Excitation System
    
    A simple first-order exciter model suitable for preliminary studies.
    State: Efd (field voltage)
    """
    
    def __init__(self, params: SEXSParameters, Vref_pu: float = 1.0):
        """
        Initialize SEXS exciter.
        
        Args:
            params: SEXS parameters
            Vref_pu: Voltage reference setpoint (pu)
        """
        self.params = params
        self.Vref_pu = Vref_pu
        
        # State: field voltage
        self.Efd_pu = 1.0
        
        # Regulator output (before limits)
        self.Vr_pu = 1.0
    
    def initialize(self, Vt_pu: float, Efd_init: float):
        """
        Initialize exciter from steady-state conditions.
        
        Args:
            Vt_pu: Terminal voltage (pu)
            Efd_init: Initial field voltage (pu)
        """
        self.Efd_pu = Efd_init
        
        # Set Vref to maintain current voltage
        self.Vref_pu = Vt_pu
        
        # Regulator output
        self.Vr_pu = Efd_init
    
    def compute_derivative(self, t: float, Vt_pu: float) -> float:
        """
        Compute derivative of field voltage.
        
        Args:
            t: Time (s)
            Vt_pu: Terminal voltage (pu)
        
        Returns:
            dEfd/dt
        """
        p = self.params
        
        # Voltage error
        Ve = self.Vref_pu - Vt_pu
        
        # Regulator output (with gain and lead-lag)
        if p.Tb_s > 0:
            # Simplified lead-lag (using current values)
            Vr = p.K * Ve * (1 + p.Ta_s / p.Tb_s)
        else:
            Vr = p.K * Ve
        
        # Apply limits
        Vr = np.clip(Vr, p.Efd_min, p.Efd_max)
        self.Vr_pu = Vr
        
        # Exciter dynamics
        dEfd = (Vr - self.Efd_pu) / p.Te_s
        
        return dEfd
    
    def update(self, Efd_new: float):
        """Update field voltage state."""
        self.Efd_pu = np.clip(Efd_new, self.params.Efd_min, self.params.Efd_max)


class IEEEAC1AExciter:
    """
    IEEE Type AC1A Excitation System
    
    A detailed AC exciter model with voltage regulator, exciter, and stabilizer.
    States: [Vr, Efd, Vf]
    """
    
    def __init__(self, params: IEEEAC1AParameters, Vref_pu: float = 1.0):
        """
        Initialize IEEEAC1A exciter.
        
        Args:
            params: IEEEAC1A parameters
            Vref_pu: Voltage reference setpoint (pu)
        """
        self.params = params
        self.Vref_pu = Vref_pu
        
        # States: [Vr (regulator), Efd (field voltage), Vf (stabilizer)]
        self.states = np.array([1.0, 1.0, 0.0])
    
    def initialize(self, Vt_pu: float, Efd_init: float):
        """
        Initialize exciter from steady-state.
        
        Args:
            Vt_pu: Terminal voltage (pu)
            Efd_init: Initial field voltage (pu)
        """
        p = self.params
        
        # Set field voltage
        self.states[1] = Efd_init
        
        # Regulator output to maintain Efd
        Vr = Efd_init * (1 + p.Ke)
        self.states[0] = Vr
        
        # Stabilizer feedback
        self.states[2] = p.Kf * Efd_init / p.Tf_s
        
        # Set Vref to maintain current voltage
        self.Vref_pu = Vt_pu + self.states[2]
    
    def compute_derivatives(self, t: float, Vt_pu: float) -> np.ndarray:
        """
        Compute state derivatives.
        
        Args:
            t: Time (s)
            Vt_pu: Terminal voltage (pu)
        
        Returns:
            [dVr/dt, dEfd/dt, dVf/dt]
        """
        p = self.params
        Vr, Efd, Vf = self.states
        
        # Voltage error (with stabilizer feedback)
        Ve = self.Vref_pu - Vt_pu - Vf
        
        # Regulator
        Vr_input = p.Ka * Ve
        Vr_limited = np.clip(Vr_input, p.Vr_min, p.Vr_max)
        dVr = (Vr_limited - Vr) / p.Ta_s
        
        # Exciter (AC rotating type)
        Efd_input = Vr / (1 + p.Ke * Efd)
        Efd_limited = np.clip(Efd_input, p.Efd_min, p.Efd_max)
        dEfd = (Efd_limited - Efd) / p.Te_s
        
        # Stabilizer (rate feedback)
        dVf = (p.Kf * Efd / p.Tf_s - Vf) / p.Tf_s
        
        return np.array([dVr, dEfd, dVf])
    
    def update(self, states_new: np.ndarray):
        """Update states."""
        p = self.params
        self.states = states_new
        # Apply limits
        self.states[1] = np.clip(self.states[1], p.Efd_min, p.Efd_max)
    
    def get_field_voltage(self) -> float:
        """Get current field voltage."""
        return self.states[1]


def load_exciters_from_h5(h5_file: str, gen_names: list) -> Dict[str, object]:
    """
    Load exciter models from Graph_model.h5.
    
    Args:
        h5_file: Path to Graph_model.h5
        gen_names: List of generator names
    
    Returns:
        Dictionary of {gen_name: Exciter object}
    """
    import h5py
    
    exciters = {}
    
    with h5py.File(h5_file, 'r') as f:
        if 'dynamic_models/exciters' not in f:
            # Use default SEXS for all generators
            print("[WARN] No exciter data in H5, using default SEXS")
            for gen_name in gen_names:
                params = SEXSParameters(Ta_s=0.5, Te_s=0.1, K=100.0)
                exciters[gen_name] = SEXSExciter(params)
            return exciters
        
        exc_group = f['dynamic_models/exciters']
        n_exc = len(exc_group['names'])
        
        for i in range(n_exc):
            model_type = exc_group['model_type'][i]
            if isinstance(model_type, bytes):
                model_type = model_type.decode()
            
            gen_name = exc_group['generator_names'][i]
            if isinstance(gen_name, bytes):
                gen_name = gen_name.decode()
            
            if model_type == 'SEXS':
                params = SEXSParameters(
                    Ta_s=float(exc_group['Ta_s'][i]),
                    Te_s=float(exc_group['Te_s'][i]),
                    K=float(exc_group.get('Ka', [100.0]*n_exc)[i]),
                    Efd_min=float(exc_group['Efd_min'][i]),
                    Efd_max=float(exc_group['Efd_max'][i])
                )
                exciters[gen_name] = SEXSExciter(params)
            
            elif model_type == 'IEEEAC1A':
                params = IEEEAC1AParameters(
                    Ka=float(exc_group['Ka'][i]),
                    Ta_s=float(exc_group['Ta_s'][i]),
                    Ke=float(exc_group['Ke'][i]),
                    Te_s=float(exc_group['Te_s'][i]),
                    Kf=float(exc_group['Kf'][i]),
                    Tf_s=float(exc_group['Tf_s'][i]),
                    Efd_min=float(exc_group['Efd_min'][i]),
                    Efd_max=float(exc_group['Efd_max'][i]),
                    Vr_min=float(exc_group['Vr_min'][i]),
                    Vr_max=float(exc_group['Vr_max'][i])
                )
                exciters[gen_name] = IEEEAC1AExciter(params)
    
    print(f"[OK] Loaded {len(exciters)} exciters from {h5_file}")
    return exciters
