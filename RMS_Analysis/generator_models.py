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
        Initialize generator from power flow solution using ANDES GENROU methodology.
        
        Reference: ANDES andes/models/synchronous/genrou.py (lines 88-155)
        Key fix: Uses xd" (subtransient) for equivalent circuit and includes saturation.
        
        Steps (matching ANDES exactly):
        1. Complex phasor calculations (V, I, Is)
        2. Subtransient flux psi" and saturation Se0
        3. Rotor angle delta from geometry
        4. Transform to dq frame
        5. Compute transient/subtransient voltages with saturation
        
        Args:
            P_pu: Active power output (pu on machine base)
            Q_pu: Reactive power output (pu on machine base)
            Vt_pu: Terminal voltage magnitude (pu)
            theta_rad: Voltage angle at terminal (rad)
        """
        p = self.params
        
        # ============================================================
        # STEP 1: Complex phasor representation (ANDES genrou.py lines 88-97)
        # ============================================================
        _V = Vt_pu * np.exp(1j * theta_rad)  # Bus voltage phasor
        _S = P_pu - 1j * Q_pu                # Complex power (gen convention)
        
        # **KEY FIX**: Use xd" (subtransient), NOT xd' (transient)!
        _Zs = p.ra_pu + 1j * p.xd_double_pu  # Equivalent impedance
        
        # Terminal current
        if abs(_V) > 1e-6:
            _It = _S / np.conj(_V)
        else:
            _It = 0.01 + 1j * 0.01
        
        # Equivalent current source behind subtransient impedance
        _Is = _It + _V / _Zs
        
        # ============================================================
        # STEP 2: Subtransient flux linkage and saturation (lines 98-103)
        # ============================================================
        psi20 = _Is * _Zs  # ψ" in stator reference frame
        psi20_abs = abs(psi20)
        psi20_arg = np.angle(psi20)
        
        # Saturation calculation (quadratic model)
        # Se0 = B*(|ψ"| - A)^2 / |ψ"| if |ψ"| >= A, else 0
        SAT_A = self.sat_A  # Typically 1.0
        SAT_B = self.sat_B  # From S10, S12 parameters
        
        if psi20_abs >= SAT_A:
            Se0 = (psi20_abs - SAT_A)**2 * SAT_B / psi20_abs
        else:
            Se0 = 0.0
        
        # ============================================================
        # STEP 3: Rotor angle from geometry (lines 104-109)
        # ============================================================
        # Saturation correction factors
        gqd = (p.xq_pu - p.xl_pu) / (p.xd_pu - p.xl_pu)  # Ratio of q/d unsaturated reactances
        
        _a = psi20_abs * (1 + Se0 * gqd)
        _b = abs(_It) * (p.xq_double_pu - p.xq_pu)  # Note: xd" = xq" for round rotor
        
        # Angle between psi" and It
        _It_arg = np.angle(_It)
        _psi20_It_arg = psi20_arg - _It_arg
        
        # Solve for delta using geometry
        numerator = _b * np.cos(_psi20_It_arg)
        denominator = _b * np.sin(_psi20_It_arg) - _a
        
        if abs(denominator) > 1e-6:
            delta0 = np.arctan(numerator / denominator) + psi20_arg
        else:
            # Fallback to simple estimate
            delta0 = psi20_arg
        
        # ============================================================
        # STEP 4: Park transformation to dq frame (lines 110-122)
        # ============================================================
        # Transformation: multiply by e^(-j*delta)
        _Tdq = np.cos(delta0) - 1j * np.sin(delta0)
        
        psi20_dq = psi20 * _Tdq
        It_dq = np.conj(_It * _Tdq)
        
        # Extract dq components
        psi2d0 = psi20_dq.real         # d-axis subtransient flux
        psi2q0 = -psi20_dq.imag        # q-axis subtransient flux
        
        Id0 = It_dq.imag               # d-axis current
        Iq0 = It_dq.real               # q-axis current
        
        # ============================================================
        # STEP 5: Terminal voltage in dq frame (lines 123-124)
        # ============================================================
        vd0 = psi2q0 + p.xq_double_pu * Iq0 - p.ra_pu * Id0
        vq0 = psi2d0 - p.xd_double_pu * Id0 - p.ra_pu * Iq0
        
        # ============================================================
        # STEP 6: Field voltage and mechanical torque (lines 125-127)
        # ============================================================
        vf0 = (Se0 + 1) * psi2d0 + (p.xd_pu - p.xd_double_pu) * Id0
        tm0 = (vq0 + p.ra_pu * Iq0) * Iq0 + (vd0 + p.ra_pu * Id0) * Id0
        
        # Flux linkages for initial electric torque
        psid0 = p.ra_pu * Iq0 + vq0
        psiq0 = -(p.ra_pu * Id0 + vd0)
        
        # ============================================================
        # STEP 7: Transient voltages (lines 137-140) **WITH SATURATION**
        # ============================================================
        e1q0 = Id0 * (-p.xd_pu + p.xd_prime_pu) - Se0 * psi2d0 + vf0
        e1d0 = Iq0 * (p.xq_pu - p.xq_prime_pu) - Se0 * gqd * psi2q0
        
        # ============================================================
        # STEP 8: Subtransient voltages 
        # Must match derivative equations for zero residual:
        #   dEq"/dt = (Eq' - Eq" - (xd' - xd")*Id) / Td0"
        #   dEd"/dt = (Ed' - Ed" + (xq' - xq")*Iq) / Tq0"
        # At steady state (derivatives = 0):
        #   Eq" = Eq' - (xd' - xd")*Id
        #   Ed" = Ed' + (xq' - xq")*Iq
        # ============================================================
        e2q0 = e1q0 - (p.xd_prime_pu - p.xd_double_pu) * Id0
        e2d0 = e1d0 + (p.xq_prime_pu - p.xq_double_pu) * Iq0
        
        # ============================================================
        # STEP 9: Initialize state vector
        # ============================================================
        self.states = np.array([
            delta0,    # Rotor angle (rad) - from ANDES geometry
            0.0,       # Speed deviation (synchronous initially)
            e1q0,      # Eq' (q-axis transient voltage) - with saturation
            e1d0,      # Ed' (d-axis transient voltage) - NOT ZERO!
            e2q0,      # Eq" (q-axis subtransient voltage)
            e2d0       # Ed" (d-axis subtransient voltage)
        ])
        
        # Safety: If Tq0_prime = 0 (round rotor), Ed' = 0 (no damper winding)
        # This overrides the calculation above for true round rotors
        if p.Tq0_prime_s < 1e-6:
            # Round rotor: Ed' fixed at zero (no q-axis transient)
            self.states[3] = 0.0
        
        # Initial algebraic variables (IN ROTOR FRAME)
        # Transformation to network frame happens in algebraic equations
        self.algebraic = np.array([Id0, Iq0, vd0, vq0, vf0, tm0])
        
        # Store terminal conditions
        self.Vt_pu = Vt_pu
        self.P_pu = P_pu
        self.Q_pu = Q_pu
        
        # Store initialization parameters for exciter/governor
        p.Efd_init = vf0
        p.Pm_init = tm0
        
        # Verify power balance (in ROTOR frame - should be exact at initialization)
        Pe_check = vd0 * Id0 + vq0 * Iq0
        Qe_check = vq0 * Id0 - vd0 * Iq0
        
        print(f"[OK] {p.gen_name}: delta={np.degrees(delta0):.2f}°, "
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
    
    def derivatives(self, Pm_pu: float, Efd_pu: float, Id: float, Iq: float) -> np.ndarray:
        """
        Compute state derivatives with ROTOR-FRAME currents.
        
        Args:
            Pm_pu: Mechanical power (pu on machine base)
            Efd_pu: Field voltage (pu)
            Id: d-axis current (pu) - IN ROTOR FRAME
            Iq: q-axis current (pu) - IN ROTOR FRAME
        
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
        
        # Compute terminal voltages in ROTOR frame (from internal EMFs and rotor currents)
        # Include stator resistance Ra
        Vd_rotor = Ed_double - p.ra_pu * Id - p.xq_double_pu * Iq
        Vq_rotor = Eq_double - p.ra_pu * Iq - p.xd_double_pu * Id
        
        # Electrical power (in ROTOR frame)
        Pe = Vd_rotor * Id + Vq_rotor * Iq
        
        # Swing equation
        omega_base = 2 * np.pi * 60.0  # rad/s for 60 Hz
        d_delta = omega * omega_base
        d_omega = (Pm_pu - Pe - p.D_pu * omega) / (2 * p.H_s)
        
        # Field winding (transient)
        d_Eq_prime = (Efd_pu - Eq_prime - (p.xd_pu - p.xd_prime_pu - Sat) * Id) / p.Td0_prime_s
        
        # Damper winding d-axis (transient)
        # Safety: If Tq0_prime = 0 (round rotor), Ed' = 0 (no damper winding)
        if p.Tq0_prime_s > 1e-6:
            d_Ed_prime = (-Ed_prime + (p.xq_pu - p.xq_prime_pu) * Iq) / p.Tq0_prime_s
        else:
            # Round rotor: Ed' fixed at zero (no q-axis transient)
            d_Ed_prime = 0.0
        
        # Subtransient dynamics
        d_Eq_double = (Eq_prime - Eq_double - (p.xd_prime_pu - p.xd_double_pu) * Id) / p.Td0_double_prime_s
        d_Ed_double = (Ed_prime - Ed_double + (p.xq_prime_pu - p.xq_double_pu) * Iq) / p.Tq0_double_prime_s
        
        # Update algebraic variables (IN ROTOR FRAME)
        self.algebraic = np.array([Id, Iq, Vd_rotor, Vq_rotor, Efd_pu, Pm_pu])
        
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
    
    def get_time_constants(self) -> np.ndarray:
        """
        Get time constants for all 6 generator states.
        
        Returns:
            Array of time constants [T_delta, T_omega, T_Eq', T_Ed', T_Eq'', T_Ed'']
        """
        p = self.params
        
        # Safety: If Tq0' = 0 (round rotor), use large time constant (Ed' = constant)
        Tq0_prime = p.Tq0_prime_s if p.Tq0_prime_s > 1e-6 else 1e6
        
        return np.array([
            2 * p.H_s,  # Delta (rotor angle) - inertial time constant
            1.0,  # Omega (speed deviation) - normalized
            p.Td0_prime_s,  # Eq' (d-axis transient voltage)
            Tq0_prime,  # Ed' (q-axis transient voltage) - large if Tq0'=0
            p.Td0_double_prime_s,  # Eq'' (d-axis subtransient voltage)
            p.Tq0_double_prime_s   # Ed'' (q-axis subtransient voltage)
        ])


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
