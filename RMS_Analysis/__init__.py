"""
RMS_Analysis - Root Mean Square Dynamic Simulation Module

This module provides time-domain electromechanical simulation for power systems,
including transient stability analysis using REAL PowerFactory parameters.

Components:
- generator_models: GENROU synchronous generator model
- exciter_models: SEXS, IEEEAC1A automatic voltage regulators
- governor_models: TGOV1, HYGOV turbine-governor models
- integrator: Time-domain numerical integration (RK4, trapezoidal)
- rms_simulator: Main simulation coordinator

Author: PIGNN Project
Date: October 20, 2025
"""

from .generator_models import GENROUGenerator
from .exciter_models import SEXSExciter, IEEEAC1AExciter
from .governor_models import TGOV1Governor, HYGOVGovernor
from .integrator import RK4Integrator, TrapezoidalIntegrator
from .rms_simulator import RMSSimulator

__all__ = [
    'GENROUGenerator',
    'SEXSExciter',
    'IEEEAC1AExciter',
    'TGOV1Governor',
    'HYGOVGovernor',
    'RK4Integrator',
    'TrapezoidalIntegrator',
    'RMSSimulator'
]

__version__ = '1.0.0'
