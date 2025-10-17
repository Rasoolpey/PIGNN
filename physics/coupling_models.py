"""
Physics-based coupling calculations for three-phase components.
Computes mutual impedances based on geometry and component type.
"""

import numpy as np
from typing import Tuple, Optional


class CouplingCalculator:
    """
    Calculate coupling matrices for different component types.
    """
    
    # Physical constants
    MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    EPSILON_0 = 8.854e-12    # Permittivity of free space (F/m)
    
    @staticmethod
    def calculate_line_coupling(
        R_self: float,
        L_self: float,
        phase_spacing_m: float,
        conductor_GMR_m: float,
        frequency_hz: float = 60.0,
        include_ground_return: bool = True
    ) -> np.ndarray:
        """
        Calculate 3x3 impedance matrix for transmission line using Carson's equations.
        
        Args:
            R_self: Self resistance per phase (Ohm)
            L_self: Self inductance per phase (H)
            phase_spacing_m: Distance between phases (meters)
            conductor_GMR_m: Geometric mean radius (meters)
            frequency_hz: System frequency
            include_ground_return: Include ground return impedance
        
        Returns:
            3x3 complex impedance matrix
        """
        omega = 2 * np.pi * frequency_hz
        
        # Self impedance
        X_self = omega * L_self
        Z_self = complex(R_self, X_self)
        
        # Mutual inductance (Carson's formula approximation)
        M_mutual = (CouplingCalculator.MU_0 / (2 * np.pi)) * \
                   np.log(phase_spacing_m / conductor_GMR_m)
        X_mutual = omega * M_mutual
        
        # Ground return resistance (simplified)
        R_ground = 0.0
        if include_ground_return:
            # Simplified ground return resistance
            R_ground = omega * CouplingCalculator.MU_0 / (8 * np.pi)
        
        Z_mutual = complex(R_ground, X_mutual)
        
        # Build 3x3 matrix
        # Assuming symmetric spacing (can be extended)
        Z_matrix = np.array([
            [Z_self, Z_mutual, Z_mutual],
            [Z_mutual, Z_self, Z_mutual],
            [Z_mutual, Z_mutual, Z_self]
        ], dtype=complex)
        
        return Z_matrix
    
    @staticmethod
    def calculate_transformer_coupling(
        Z_leakage: complex,
        winding_type: str,
        mutual_coupling_factor: float = 0.05
    ) -> np.ndarray:
        """
        Calculate 3x3 impedance matrix for transformer.
        
        Args:
            Z_leakage: Leakage impedance per phase
            winding_type: 'YY', 'YD', 'DY', 'DD'
            mutual_coupling_factor: Coupling between phases (typically 0.05)
        
        Returns:
            3x3 complex impedance matrix
        """
        # Base matrix with self impedances
        Z_matrix = np.eye(3, dtype=complex) * Z_leakage
        
        # Add mutual coupling
        Z_mutual = Z_leakage * mutual_coupling_factor
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    Z_matrix[i, j] = Z_mutual
        
        return Z_matrix
    
    @staticmethod
    def calculate_generator_coupling(
        xd: float,
        xq: float,
        ra: float,
        base_impedance: float,
        frequency_hz: float = 60.0
    ) -> np.ndarray:
        """
        Calculate 3x3 impedance matrix for generator.
        
        Args:
            xd: d-axis reactance (pu)
            xq: q-axis reactance (pu)
            ra: Armature resistance (pu)
            base_impedance: Base impedance (Ohm)
            frequency_hz: System frequency
        
        Returns:
            3x3 complex impedance matrix in Ohms
        """
        # For balanced generator, assume symmetric
        # In reality, would need Park transformation
        Z_self = complex(ra, (xd + xq) / 2) * base_impedance
        
        # Small mutual coupling in stator
        Z_mutual = Z_self * 0.1
        
        Z_matrix = np.array([
            [Z_self, Z_mutual, Z_mutual],
            [Z_mutual, Z_self, Z_mutual],
            [Z_mutual, Z_mutual, Z_self]
        ], dtype=complex)
        
        return Z_matrix
    
    @staticmethod
    def get_transformer_connection_matrix(winding_type: str) -> np.ndarray:
        """
        Get connection matrix for transformer windings.
        
        Args:
            winding_type: 'YY', 'YD', 'DY', 'DD'
        
        Returns:
            3x3 connection matrix
        """
        if winding_type == 'YY':
            # Wye-wye: direct connection
            return np.eye(3)
        
        elif winding_type == 'YD' or winding_type == 'Yd':
            # Wye-delta: 30 degree phase shift
            return np.array([
                [1, -1, 0],
                [0, 1, -1],
                [-1, 0, 1]
            ]) / np.sqrt(3)
        
        elif winding_type == 'DY' or winding_type == 'Dy':
            # Delta-wye: -30 degree phase shift
            return np.array([
                [1, 0, -1],
                [-1, 1, 0],
                [0, -1, 1]
            ]) / np.sqrt(3)
        
        elif winding_type == 'DD' or winding_type == 'Dd':
            # Delta-delta: no phase shift
            return np.array([
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 2]
            ]) / 3
        
        else:
            # Default to YY
            return np.eye(3)
    
    @staticmethod
    def estimate_line_coupling_from_impedance(
        R_total: float,
        X_total: float,
        mutual_factor: float = 0.3
    ) -> np.ndarray:
        """
        Estimate coupling matrix when only total impedance is known.
        
        Args:
            R_total: Total resistance (Ohm)
            X_total: Total reactance (Ohm)
            mutual_factor: Mutual impedance as fraction of self (0.2-0.4 typical)
        
        Returns:
            3x3 complex impedance matrix
        """
        Z_self = complex(R_total, X_total)
        Z_mutual = Z_self * mutual_factor
        
        Z_matrix = np.array([
            [Z_self, Z_mutual, Z_mutual],
            [Z_mutual, Z_self, Z_mutual],
            [Z_mutual, Z_mutual, Z_self]
        ], dtype=complex)
        
        return Z_matrix