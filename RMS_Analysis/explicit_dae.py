"""
Simple Explicit DAE Integrator for debugging.

Uses explicit Euler for differential equations:
  x_{n+1} = x_n + h * f(x_n, y_n)

Algebraic variables stay fixed from initialization.
This is NOT accurate but will show if dynamics are working.
"""

import numpy as np
from typing import Callable, Tuple


class ExplicitEulerDAE:
    """Simple explicit Euler for DAE (for debugging only)."""
    
    def __init__(self, dt: float):
        self.dt = dt
        self.total_steps = 0
        self.total_failures = 0
        
    def step(self, t: float, dae, update_equations: Callable, 
             update_jacobians: Callable = None, predictor: str = 'euler') -> Tuple[bool, int, float]:
        """Take one explicit Euler step."""
        h = self.dt
        
        # Evaluate equations at current time
        update_equations(dae, t)
        f_curr = dae.f.copy()
        g_curr = dae.g.copy()
        
        # Update differential states: x_{n+1} = x_n + h * f_n
        dae.x += h * f_curr
        
        # Algebraic states stay fixed (this is the approximation!)
        # In a real solver, we would solve g(x_{n+1}, y_{n+1}) = 0
        
        # Compute residual for reporting
        norm_q = max(np.max(np.abs(f_curr)), np.max(np.abs(g_curr)))
        
        self.total_steps += 1
        return True, 1, norm_q
    
    def print_statistics(self):
        """Print statistics."""
        print("\n" + "="*70)
        print("INTEGRATION STATISTICS (Explicit Euler - DEBUG MODE)")
        print("="*70)
        print(f"Total steps:        {self.total_steps}")
        print(f"Failed steps:       {self.total_failures}")
        print("NOTE: This is explicit Euler (not implicit trapezoid)")
        print("      Algebraic constraints NOT enforced during integration!")
        print("="*70)
    
    def reset_statistics(self):
        self.total_steps = 0
        self.total_failures = 0
