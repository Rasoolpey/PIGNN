"""
Specialized edge types for power system components.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .graph_base import Edge, PhaseType


@dataclass
class TransmissionLine(Edge):
    """Transmission line with pi-model parameters"""
    
    def __init__(self, **kwargs):
        super().__init__(edge_type='line', **kwargs)
        
        # Line parameters (per phase)
        self.R_ohm: float = 0.01
        self.X_ohm: float = 0.1
        self.B_total_uS: float = 0.0  # Total shunt susceptance
        
        # Line ratings
        self.rating_MVA: float = 100.0
        self.emergency_rating_MVA: float = 120.0
        
        # Length
        self.length_km: float = 1.0
    
    @property
    def series_impedance(self) -> complex:
        """Get series impedance"""
        return complex(self.R_ohm, self.X_ohm)
    
    @property
    def shunt_admittance_half(self) -> complex:
        """Get half of shunt admittance (for pi-model)"""
        return complex(0, self.B_total_uS * 1e-6 / 2)
    
    def get_ABCD_parameters(self) -> Tuple[complex, complex, complex, complex]:
        """Get ABCD parameters for the line"""
        Z = self.series_impedance
        Y = 2 * self.shunt_admittance_half  # Total shunt
        
        # For medium line pi-model
        A = 1 + (Y * Z) / 2
        B = Z
        C = Y * (1 + (Y * Z) / 4)
        D = A
        
        return A, B, C, D
    
    def calculate_losses(self, I_amps: float) -> Tuple[float, float]:
        """Calculate line losses given current magnitude"""
        P_loss_MW = 3 * (I_amps ** 2) * self.R_ohm * 1e-6
        Q_loss_MVAR = 3 * (I_amps ** 2) * self.X_ohm * 1e-6
        return P_loss_MW, Q_loss_MVAR


@dataclass
class Transformer(Edge):
    """Two-winding transformer"""
    
    def __init__(self, **kwargs):
        super().__init__(edge_type='transformer', **kwargs)
        
        # Transformer parameters
        self.rating_MVA: float = 100.0
        self.V_primary_kV: float = 138.0
        self.V_secondary_kV: float = 69.0
        self.R_pu: float = 0.01
        self.X_pu: float = 0.1
        
        # Tap changer
        self.tap_ratio: float = 1.0
        self.tap_position: int = 0
        self.tap_min: float = 0.9
        self.tap_max: float = 1.1
        self.tap_step: float = 0.00625
        
        # Winding configuration
        self.winding_config: str = 'YY'  # YY, YD, DD, etc.
        self.phase_shift_deg: float = 0.0
        
        # Vector group (for phase shift)
        self.vector_group: str = 'Yy0'  # e.g., Dyn11, Yy0
    
    def get_base_impedance(self) -> float:
        """Calculate base impedance on primary side"""
        return (self.V_primary_kV ** 2) / self.rating_MVA
    
    @property
    def series_impedance_ohm(self) -> complex:
        """Get series impedance in ohms (primary side)"""
        Z_base = self.get_base_impedance()
        return complex(self.R_pu, self.X_pu) * Z_base
    
    def get_turns_ratio_complex(self) -> complex:
        """Get complex turns ratio including phase shift"""
        magnitude = self.tap_ratio * (self.V_secondary_kV / self.V_primary_kV)
        phase_rad = np.deg2rad(self.phase_shift_deg)
        return magnitude * np.exp(1j * phase_rad)
    
    def get_equivalent_pi_model(self) -> dict:
        """Get equivalent pi-model parameters"""
        # Simplified model - neglecting magnetizing branch
        n = self.get_turns_ratio_complex()
        Z = complex(self.R_pu, self.X_pu)
        
        # Referred to primary side
        return {
            'Y_series': 1 / (Z / self.tap_ratio),
            'Y_shunt_primary': 0 + 0j,  # Neglecting magnetizing
            'Y_shunt_secondary': 0 + 0j,
            'turns_ratio': n
        }
    
    def adjust_tap(self, direction: int) -> bool:
        """Adjust tap position (+1 or -1)"""
        new_position = self.tap_position + direction
        new_ratio = 1.0 + new_position * self.tap_step
        
        if self.tap_min <= new_ratio <= self.tap_max:
            self.tap_position = new_position
            self.tap_ratio = new_ratio
            return True
        return False


def create_edge(edge_id: str, edge_type: str, phase: PhaseType,
                from_node_id: str, to_node_id: str, **kwargs) -> Edge:
    """Factory function to create appropriate edge type"""
    
    if edge_type == 'line':
        edge = TransmissionLine(
            id=f"{edge_id}_{phase.value}",
            parent_id=edge_id,
            phase=phase,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            **kwargs
        )
    elif edge_type == 'transformer':
        edge = Transformer(
            id=f"{edge_id}_{phase.value}",
            parent_id=edge_id,
            phase=phase,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            **kwargs
        )
    else:
        # Default to base Edge
        edge = Edge(
            id=f"{edge_id}_{phase.value}",
            parent_id=edge_id,
            phase=phase,
            edge_type=edge_type,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            **kwargs
        )
    
    return edge