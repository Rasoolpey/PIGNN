"""
Specialized node types for power system components.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from .graph_base import Node, PhaseType


@dataclass
class Generator(Node):
    """Generator node with machine parameters"""
    
    def __init__(self, **kwargs):
        super().__init__(node_type='generator', **kwargs)
        
        # Generator parameters
        self.P_nominal_MW: float = 0.0
        self.Q_min_MVAR: float = -100.0
        self.Q_max_MVAR: float = 100.0
        self.P_actual_MW: float = 0.0
        self.Q_actual_MVAR: float = 0.0
        
        # Machine parameters
        self.xd_pu: float = 1.0
        self.xq_pu: float = 0.8
        self.xd_prime_pu: float = 0.3
        self.ra_pu: float = 0.003
        self.H_s: float = 5.0
        self.D_pu: float = 0.0
        
        # Control mode
        self.control_mode: str = 'PV'  # PV or slack
        self.voltage_setpoint_pu: float = 1.0
    
    def get_internal_voltage(self) -> complex:
        """Calculate internal voltage behind transient reactance"""
        # Simplified calculation
        I_terminal = self.properties.get('terminal_current', 0 + 0j)
        E_internal = self.voltage_pu + 1j * self.xd_prime_pu * I_terminal
        return E_internal
    
    def check_reactive_limits(self) -> bool:
        """Check if reactive power is within limits"""
        return self.Q_min_MVAR <= self.Q_actual_MVAR <= self.Q_max_MVAR


@dataclass
class Load(Node):
    """Load node with power demand"""
    
    def __init__(self, **kwargs):
        super().__init__(node_type='load', **kwargs)
        
        # Load parameters
        self.P_MW: float = 0.0
        self.Q_MVAR: float = 0.0
        self.load_model: str = 'constant_power'  # constant_power, constant_impedance, constant_current
        
        # Voltage dependency (ZIP model)
        self.ZIP_coefficients = {
            'P': [1.0, 0.0, 0.0],  # [constant_P, constant_I, constant_Z]
            'Q': [1.0, 0.0, 0.0]
        }
        
        # Power factor
        self.pf_nominal: float = 0.95
    
    def get_power_at_voltage(self, V_pu: float) -> Tuple[float, float]:
        """Calculate actual power at given voltage using ZIP model"""
        # ZIP model: P = P0 * (a + b*V + c*V^2)
        P_coeff = self.ZIP_coefficients['P']
        Q_coeff = self.ZIP_coefficients['Q']
        
        P_actual = self.P_MW * (P_coeff[0] + P_coeff[1]*V_pu + P_coeff[2]*V_pu**2)
        Q_actual = self.Q_MVAR * (Q_coeff[0] + Q_coeff[1]*V_pu + Q_coeff[2]*V_pu**2)
        
        return P_actual, Q_actual
    
    def get_equivalent_impedance(self) -> complex:
        """Calculate equivalent impedance for constant Z model"""
        if abs(self.P_MW) < 1e-6 and abs(self.Q_MVAR) < 1e-6:
            return float('inf') + 0j
        
        S_MVA = complex(self.P_MW, self.Q_MVAR)
        V_kV = abs(self.voltage_pu) * self.voltage_base_kv
        Z_ohm = (V_kV ** 2) / np.conj(S_MVA)
        return Z_ohm


@dataclass
class Bus(Node):
    """Simple bus node (connection point)"""
    
    def __init__(self, **kwargs):
        super().__init__(node_type='bus', **kwargs)
        
        # Bus type for power flow
        self.bus_type: str = 'PQ'  # PQ, PV, or slack
        
        # Shunt admittance (if any)
        self.shunt_G_pu: float = 0.0
        self.shunt_B_pu: float = 0.0
    
    def get_shunt_injection(self) -> complex:
        """Calculate shunt current injection"""
        Y_shunt = complex(self.shunt_G_pu, self.shunt_B_pu)
        I_shunt = Y_shunt * self.voltage_pu
        return I_shunt


def create_node(node_id: str, node_type: str, phase: PhaseType, **kwargs) -> Node:
    """Factory function to create appropriate node type"""
    
    if node_type == 'generator':
        node = Generator(
            id=f"{node_id}_{phase.value}",
            parent_id=node_id,
            phase=phase,
            **kwargs
        )
    elif node_type == 'load':
        node = Load(
            id=f"{node_id}_{phase.value}",
            parent_id=node_id,
            phase=phase,
            **kwargs
        )
    elif node_type == 'bus':
        node = Bus(
            id=f"{node_id}_{phase.value}",
            parent_id=node_id,
            phase=phase,
            **kwargs
        )
    else:
        # Default to base Node
        node = Node(
            id=f"{node_id}_{phase.value}",
            parent_id=node_id,
            phase=phase,
            node_type=node_type,
            **kwargs
        )
    
    return node