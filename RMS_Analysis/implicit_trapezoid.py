"""
Implicit Trapezoid DAE Solver

Following ANDES architecture for semi-explicit DAE systems:
    T * dx/dt = f(x, y, t)   -- Differential equations
    0 = g(x, y, t)            -- Algebraic equations

Implicit trapezoidal method:
    T * (x[n+1] - x[n]) / h = 0.5 * (f[n+1] + f[n])
    0 = g(x[n+1], y[n+1], t[n+1])

Solved via Newton-Raphson on augmented system:
    Ac * [Δx; Δy] = -q
    
where:
    Ac = [[ Teye - 0.5*h*fx,  scale*gx^T  ],
          [ -0.5*h*fy^T,      scale*gy     ]]
          
    q[:n] = Tf*(x-x0) - 0.5*h*(f+f0)   # Differential residual
    q[n:] = scale*g                     # Algebraic residual

Author: PIGNN Project
Date: October 20, 2025
Reference: ANDES andes/routines/daeint.py
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from typing import Callable, Tuple, Optional
import warnings


class ImplicitTrapezoidSolver:
    """
    Implicit trapezoidal DAE integrator with Newton-Raphson iterations.
    
    Designed for semi-explicit DAE systems following ANDES architecture.
    """
    
    def __init__(self, 
                 dt: float,
                 tol: float = 1e-6,
                 max_iter: int = 15,
                 g_scale: float = 0.0,
                 damping: float = 1.0,
                 verbose: bool = False):
        """
        Initialize implicit trapezoid solver.
        
        Args:
            dt: Time step size (s)
            tol: Newton iteration convergence tolerance
            max_iter: Maximum Newton iterations per time step
            g_scale: Algebraic equation scaling (0=no scale, >0=scale by h)
            damping: Newton damping factor (1.0=full step, <1.0=damped)
            verbose: Print iteration details
        """
        self.dt = dt
        self.tol = tol
        self.max_iter = max_iter
        self.g_scale = g_scale
        self.damping = damping
        self.verbose = verbose
        
        # Statistics
        self.total_steps = 0
        self.total_newton_iters = 0
        self.total_failures = 0
        self.max_newton_iters_per_step = 0
        
    def step(self, 
             t: float,
             dae,  # DAESystem object
             update_equations: Callable,
             update_jacobians: Callable,
             predictor: str = 'euler') -> Tuple[bool, int, float]:
        """
        Take one implicit trapezoid step.
        
        Args:
            t: Current time
            dae: DAESystem object with states x, y and equations f, g
            update_equations: Function to recompute f and g from current x, y
            update_jacobians: Function to recompute fx, fy, gx, gy
            predictor: Initial guess method ('euler', 'constant', 'none')
            
        Returns:
            (converged, num_iters, residual_norm)
        """
        h = self.dt
        
        # Save previous time step values
        x0 = dae.x.copy()
        y0 = dae.y.copy()
        
        # Evaluate at t=n
        update_equations(dae, t)
        f0 = dae.f.copy()
        g0 = dae.g.copy()
        
        # Predictor step (initial guess for x[n+1], y[n+1])
        if predictor == 'euler':
            # Explicit Euler: x[n+1] = x[n] + h * f[n] / T
            dae.x = x0 + h * f0 / dae.Tf
            # Keep y constant for now
        elif predictor == 'constant':
            # Just keep x[n], y[n]
            pass
        # else: no prediction, use current values
        
        # Newton-Raphson iterations
        converged = False
        for iter_num in range(self.max_iter):
            # Evaluate equations at current iterate
            update_equations(dae, t + h)
            
            # Compute residual q
            q = dae.calc_residual_q(h, x0, f0, config_g_scale=self.g_scale)
            residual_norm = np.linalg.norm(q)
            
            if self.verbose:
                print(f"    Newton iter {iter_num+1}: ||q|| = {residual_norm:.6e}")
            
            # Check convergence
            if residual_norm < self.tol:
                converged = True
                break
            
            # Check for divergence (residual growing too large)
            if residual_norm > 1e6:
                if self.verbose:
                    print(f"    [DIVERGED] Residual > 1e6")
                break
            
            # Update Jacobians
            update_jacobians(dae, t + h)
            
            # Build augmented Jacobian
            Ac = dae.build_jacobian_ac(h, config_g_scale=self.g_scale)
            
            # Solve for increment: Ac * inc = -q
            try:
                inc = spsolve(Ac, -q)
            except Exception as e:
                if self.verbose:
                    print(f"    [FAILED] Linear solve error: {e}")
                converged = False
                break
            
            # Apply increment with damping
            dae.update_states(self.damping * inc)
            
            # Check for NaN
            if np.any(np.isnan(dae.x)) or np.any(np.isnan(dae.y)):
                if self.verbose:
                    print(f"    [NaN DETECTED] States became NaN")
                converged = False
                break
        
        # Update statistics
        self.total_steps += 1
        self.total_newton_iters += (iter_num + 1)
        self.max_newton_iters_per_step = max(self.max_newton_iters_per_step, iter_num + 1)
        
        if not converged:
            self.total_failures += 1
            if self.verbose:
                print(f"    [WARNING] Newton did not converge in {self.max_iter} iterations")
            
            # Restore previous values on failure
            dae.x[:] = x0
            dae.y[:] = y0
            update_equations(dae, t)
        
        return converged, iter_num + 1, residual_norm
    
    def get_statistics(self) -> dict:
        """Get solver statistics."""
        avg_iters = self.total_newton_iters / self.total_steps if self.total_steps > 0 else 0
        failure_rate = 100 * self.total_failures / self.total_steps if self.total_steps > 0 else 0
        
        return {
            'total_steps': self.total_steps,
            'total_newton_iters': self.total_newton_iters,
            'avg_newton_iters': avg_iters,
            'max_newton_iters': self.max_newton_iters_per_step,
            'total_failures': self.total_failures,
            'failure_rate_percent': failure_rate
        }
    
    def print_statistics(self):
        """Print solver statistics."""
        stats = self.get_statistics()
        print("\n" + "="*70)
        print("IMPLICIT TRAPEZOID SOLVER STATISTICS")
        print("="*70)
        print(f"Total time steps:        {stats['total_steps']}")
        print(f"Total Newton iterations: {stats['total_newton_iters']}")
        print(f"Avg Newton iters/step:   {stats['avg_newton_iters']:.2f}")
        print(f"Max Newton iters/step:   {stats['max_newton_iters']}")
        print(f"Failed steps:            {stats['total_failures']}")
        print(f"Failure rate:            {stats['failure_rate_percent']:.2f}%")
        print("="*70)


def create_integrator(method: str = 'trapezoidal', dt: float = 0.01, **kwargs):
    """
    Factory function to create integrator.
    
    Args:
        method: 'trapezoidal', 'rk4', or 'adaptive'
        dt: Time step size
        **kwargs: Additional integrator-specific parameters
        
    Returns:
        Integrator object
    """
    if method.lower() in ['trapezoidal', 'trap', 'implicit']:
        return ImplicitTrapezoidSolver(
            dt=dt,
            tol=kwargs.get('tol', 1e-6),
            max_iter=kwargs.get('max_iter', 15),
            g_scale=kwargs.get('g_scale', 0.0),
            damping=kwargs.get('damping', 1.0),
            verbose=kwargs.get('verbose', False)
        )
    elif method.lower() == 'rk4':
        # Keep old RK4 for backwards compatibility
        from RMS_Analysis.integrator import RK4Integrator
        return RK4Integrator(dt)
    else:
        raise ValueError(f"Unknown integration method: {method}")


if __name__ == "__main__":
    print(__doc__)
    print("\n✅ Implicit Trapezoid Solver module loaded successfully!")
    print("\nUsage:")
    print("  from implicit_trapezoid import create_integrator")
    print("  solver = create_integrator('trapezoidal', dt=0.01, tol=1e-6)")
    print("  converged, iters, resid = solver.step(t, dae, update_eqs, update_jac)")
