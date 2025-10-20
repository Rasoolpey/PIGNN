"""
Generator Models - GENROU Synchronous Machine

This module implements the GENROU (round-rotor) synchronous generator model:
- 6th order model with swing equations
- Electrical transient dynamics (field and damper windings)
- Magnetic saturation effects
- Loads REAL PowerFactory parameters from Graph_model.h5

Author: PIGNN Project
Date: October 20, 2025
Reference: IEEE Std 421.5-2016
"""

import numpy as np
import h5py
from dataclasses import dataclass
from typing import Dict


@dataclass
class GENROUParameters:
    """Parameters for GENROU generator model"""
    gen_name: str
    Sn_MVA: float       # Machine rated power (MVA)
    H_s: float          # Inertia constant (s)
    D_pu: float         # Damping coefficient (pu)
    
    # Reactances (pu on machine base)
    xd_pu: float        # d-axis synchronous reactance
    xq_pu: float        # q-axis synchronous reactance
    xd_prime_pu: float  # d-axis transient reactance
    xq_prime_pu: float  # q-axis transient reactance
    xd_double_pu: float # d-axis subtransient reactance
    xq_double_pu: float # q-axis subtransient reactance
    xl_pu: float        # Leakage reactance
    ra_pu: float        # Armature resistance
    
    # Time constants (s)
    Td0_prime_s: float      # d-axis open-circuit transient time constant
    Tq0_prime_s: float      # q-axis open-circuit transient time constant
    Td0_double_prime_s: float   # d-axis open-circuit subtransient time constant
    Tq0_double_prime_s: float   # q-axis open-circuit subtransient time constant
    
    # Saturation
    S10: float = 0.0    # Saturation factor at 1.0 pu
    S12: float = 0.0    # Saturation factor at 1.2 pu


class GENROUGenerator:
    """
    GENROU - Round Rotor Generator Model
    
    6th order model with:
    - Swing equation (rotor angle and speed)
    - Field winding transients
    - Damper winding subtransients
    - Magnetic saturation
    
    States: [delta, omega, Eq', Ed', Eq'', Ed'']
    """
    
    def __init__(self, params: GENROUParameters):
        """
        Initialize GENROU generator.
        
        Args:
            params: Generator parameters
        """
        self.params = params
        
        # State vector: [delta, omega, Eq', Ed', Eq'', Ed'']
        self.states = np.zeros(6)
        
        # Algebraic variables: [Id, Iq, Vd, Vq, Efd, Pmech]
        self.algebraic = np.zeros(6)
        
        # Compute saturation coefficients
        if params.S10 > 0 and params.S12 > 0:
            # Exponential saturation: Sat(E) = B*(E-A)^2
            self.sat_A = 1.0
            self.sat_B = np.log(params.S12 / params.S10) / (1.2 - 1.0)**2
        else:
            self.sat_A = 0.0
            self.sat_B = 0.0
    
    def initialize(self, P_pu: float, Q_pu: float, Vt_pu: float, theta_rad: float):
        """
        Initialize generator from power flow solution using ANDES methodology.
        
        Reference: ANDES andes/models/synchronous/genbase.py
        
        Steps:
        1. Use complex phasor: V = Vt * exp(j*theta)
        2. Compute current: I = (P - jQ) / conj(V)
        3. Compute internal voltage: E' = V + (Ra + jXd') * I
        4. Rotor angle: delta = angle(E')
        5. Transform to dq frame using delta
        
        Args:
            P_pu: Active power output (pu on machine base)
            Q_pu: Reactive power output (pu on machine base)
            Vt_pu: Terminal voltage magnitude (pu)
            theta_rad: Voltage angle at terminal (rad)
        """
        p = self.params
        
        # ============================================================
        # STEP 1: Complex phasor representation (network reference frame)
        # ============================================================
        V_phasor = Vt_pu * np.exp(1j * theta_rad)
        
        # ============================================================
        # STEP 2: Compute current from power balance (generator convention)
        # S = P + jQ = V * conj(I)  =>  I = conj(S / V)
        # ============================================================
        if abs(V_phasor) > 1e-6:
            S_complex = P_pu + 1j * Q_pu
            I_phasor = np.conj(S_complex / V_phasor)
        else:
            # Zero voltage - use small current
            I_phasor = 0.01 + 1j * 0.01
        
        # ============================================================
        # STEP 3: Compute transient internal voltage E' (behind Xd')
        # E' = V + (Ra + jXd') * I
        # ============================================================
        Z_prime = p.ra_pu + 1j * p.xd_prime_pu
        E_prime_phasor = V_phasor + Z_prime * I_phasor
        
        # ============================================================
        # STEP 4: Rotor angle is the angle of E' (CORRECT METHOD!)
        # ============================================================
        delta = np.angle(E_prime_phasor)
        
        # ============================================================
        # STEP 5: Transform to dq frame (rotor reference frame)
        # Rotation: V_dq = V_phasor * exp(-j*delta)
        # ============================================================
        V_dq = V_phasor * np.exp(-1j * delta)
        I_dq = I_phasor * np.exp(-1j * delta)
        E_prime_dq = E_prime_phasor * np.exp(-1j * delta)
        
        # Extract dq components
        Vd = V_dq.imag  # d-axis
        Vq = V_dq.real  # q-axis
        Id = I_dq.imag
        Iq = I_dq.real
        Eq_prime = E_prime_dq.real  # E' is aligned with q-axis
        Ed_prime = E_prime_dq.imag
        
        # ============================================================
        # STEP 6: Compute subtransient voltages (behind Xd'', Xq'')
        # ============================================================
        Eq_double = Vq + p.ra_pu * Iq + p.xd_double_pu * Id
        Ed_double = Vd + p.ra_pu * Id - p.xq_double_pu * Iq
        
        # ============================================================
        # STEP 7: Field voltage (no saturation initially)
        # Efd = Eq' + (Xd - Xd') * Id
        # ============================================================
        Efd = Eq_prime + (p.xd_pu - p.xd_prime_pu) * Id
        
        # ============================================================
        # STEP 8: Initialize state vector
        # ============================================================
        self.states = np.array([
            delta,          # Rotor angle (rad) - from angle(E')
            0.0,            # Speed deviation (synchronous initially)
            Eq_prime,       # q-axis transient voltage
            Ed_prime,       # d-axis transient voltage
            Eq_double,      # q-axis subtransient voltage
            Ed_double       # d-axis subtransient voltage
        ])
        
        # Initial algebraic variables
        self.algebraic = np.array([Id, Iq, Vd, Vq, Efd, P_pu])
        
        # Store terminal conditions
        self.Vt_pu = Vt_pu
        self.P_pu = P_pu
        self.Q_pu = Q_pu
        
        # Verify power balance (should be exact at initialization)
        Pe_check = Vd * Id + Vq * Iq
        Qe_check = Vq * Id - Vd * Iq
        
        print(f"[OK] {p.gen_name}: delta={np.degrees(delta):.2f}°, "
              f"P={P_pu:.4f} pu (check: {Pe_check:.4f}), "
              f"Q={Q_pu:.4f} pu (check: {Qe_check:.4f})")
    
    @property
    def Efd_pu(self) -> float:
        """Get current field voltage from algebraic variables."""
        if hasattr(self, 'algebraic') and len(self.algebraic) > 4:
            return self.algebraic[4]
        return 1.0  # Default
    
    def compute_saturation(self, Eqprime: float) -> float:
        """
        Compute magnetic saturation function.
        
        Args:
            Eqprime: q-axis transient voltage (pu)
        
        Returns:
            Saturation factor Sat(Eq')
        """
        if self.sat_B == 0:
            return 0.0
        
        E = abs(Eqprime)
        if E <= self.sat_A:
            return 0.0
        else:
            return self.sat_B * (E - self.sat_A)**2
    
    def derivatives(self, Pm_pu: float, Efd_pu: float, Vd: float, Vq: float) -> np.ndarray:
        """
        Compute state derivatives.
        
        Args:
            Pm_pu: Mechanical power (pu on machine base)
            Efd_pu: Field voltage (pu)
            Vd: d-axis terminal voltage (pu)
            Vq: q-axis terminal voltage (pu)
        
        Returns:
            Array of derivatives [d(delta)/dt, d(omega)/dt, ...]
        """
        p = self.params
        
        # Extract states
        delta = self.states[0]
        omega = self.states[1]  # Speed deviation from synchronous
        Eq_prime = self.states[2]
        Ed_prime = self.states[3]
        Eq_double = self.states[4]
        Ed_double = self.states[5]
        
        # Compute saturation
        Sat = self.compute_saturation(Eq_prime)
        
        # Currents (simplified - should solve network equations)
        # For now, use simplified relations
        Id = (Eq_double - Vq) / p.xd_double_pu
        Iq = (Vd - Ed_double) / p.xq_double_pu
        
        # Electrical power
        Pe = Vd * Id + Vq * Iq
        
        # Swing equation
        omega_base = 2 * np.pi * 60.0  # rad/s for 60 Hz
        d_delta = omega * omega_base
        d_omega = (Pm_pu - Pe - p.D_pu * omega) / (2 * p.H_s)
        
        # Field winding (transient)
        d_Eq_prime = (Efd_pu - Eq_prime - (p.xd_pu - p.xd_prime_pu - Sat) * Id) / p.Td0_prime_s
        
        # Damper winding d-axis (transient)
        d_Ed_prime = (-Ed_prime + (p.xq_pu - p.xq_prime_pu) * Iq) / p.Tq0_prime_s
        
        # Subtransient dynamics
        d_Eq_double = (Eq_prime - Eq_double - (p.xd_prime_pu - p.xd_double_pu) * Id) / p.Td0_double_prime_s
        d_Ed_double = (Ed_prime - Ed_double + (p.xq_prime_pu - p.xq_double_pu) * Iq) / p.Tq0_double_prime_s
        
        # Update algebraic variables
        self.algebraic = np.array([Id, Iq, Vd, Vq, Efd_pu, Pm_pu])
        
        return np.array([d_delta, d_omega, d_Eq_prime, d_Ed_prime, d_Eq_double, d_Ed_double])
    
    def compute_electrical_power(self) -> float:
        """Compute electrical power output."""
        Id, Iq, Vd, Vq = self.algebraic[0:4]
        return Vd * Id + Vq * Iq
    
    def compute_terminal_voltage(self) -> float:
        """Compute terminal voltage magnitude."""
        Vd, Vq = self.algebraic[2:4]
        return np.sqrt(Vd**2 + Vq**2)
    
    def update_outputs(self):
        """Update output power and current."""
        Id, Iq = self.algebraic[0:2]
        self.P_pu = self.compute_electrical_power()
        self.Q_pu = self.algebraic[3] * Id - self.algebraic[2] * Iq  # Vq*Id - Vd*Iq


def load_genrou_from_h5(h5_file: str) -> Dict[str, GENROUGenerator]:
    """
    Load GENROU generator models from Graph_model.h5.
    
    Args:
        h5_file: Path to Graph_model.h5
    
    Returns:
        Dictionary of {gen_name: GENROUGenerator object}
    """
    generators = {}
    
    with h5py.File(h5_file, 'r') as f:
        if 'dynamic_models/generators' not in f:
            print("[WARN] No generator data in H5 file")
            return generators
        
        gen_group = f['dynamic_models/generators']
        n_gen = len(gen_group['names'])
        
        for i in range(n_gen):
            # Decode name
            gen_name = gen_group['names'][i]
            if isinstance(gen_name, bytes):
                gen_name = gen_name.decode()
            
            # Create parameters
            params = GENROUParameters(
                gen_name=gen_name,
                Sn_MVA=float(gen_group['Sn_MVA'][i]),
                H_s=float(gen_group['H_s'][i]),
                D_pu=float(gen_group['D_pu'][i]),
                xd_pu=float(gen_group['xd_pu'][i]),
                xq_pu=float(gen_group['xq_pu'][i]),
                xd_prime_pu=float(gen_group['xd_prime_pu'][i]),
                xq_prime_pu=float(gen_group['xq_prime_pu'][i]),
                xd_double_pu=float(gen_group['xd_double_prime_pu'][i]),
                xq_double_pu=float(gen_group['xq_double_prime_pu'][i]),
                xl_pu=float(gen_group['xl_pu'][i]),
                ra_pu=float(gen_group['ra_pu'][i]),
                Td0_prime_s=float(gen_group['Td0_prime_s'][i]),
                Tq0_prime_s=float(gen_group['Tq0_prime_s'][i]),
                Td0_double_prime_s=float(gen_group['Td0_double_prime_s'][i]),
                Tq0_double_prime_s=float(gen_group['Tq0_double_prime_s'][i]),
                S10=float(gen_group['S10'][i]),
                S12=float(gen_group['S12'][i])
            )
            
            # Create generator
            generators[gen_name] = GENROUGenerator(params)
    
    print(f"[OK] Loaded {len(generators)} GENROU generators from {h5_file}")
    return generators
