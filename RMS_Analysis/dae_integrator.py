"""
DAE Integrator for RMS Simulation

Implements implicit trapezoid method for Differential-Algebraic Equations (DAE):
  Teye * dx/dt = f(x, y, t)  (differential equations)
            0  = g(x, y, t)  (algebraic equations)

Following ANDES architecture with augmented Jacobian approach.

Author: PIGNN Project
Date: October 21, 2025
"""

import numpy as np
from typing import Callable, Tuple
import time


class ImplicitTrapezoidDAE:
    """
    Implicit Trapezoid Integrator for DAE Systems
    
    Solves DAE system using implicit trapezoid rule with Newton-Raphson.
    
    The implicit trapezoid discretization:
        Teye * (x_{n+1} - x_n) = (h/2) * [f(x_n, y_n) + f(x_{n+1}, y_{n+1})]
        0 = g(x_{n+1}, y_{n+1})
    
    Rearranged as nonlinear system q = 0:
        q_x = Teye * (x_{n+1} - x_n) - (h/2) * [f_n + f_{n+1}]
        q_y = g_{n+1}
    
    Newton iteration:
        Ac * [Δx; Δy] = -[q_x; q_y]
    
    where Ac is the augmented Jacobian:
        Ac = [[Teye - (h/2)*fx,  -(h/2)*fy],
              [gx,                gy       ]]
    """
    
    def __init__(self, dt: float, max_iter: int = 15, tol: float = 1e-4):
        """
        Initialize DAE integrator.
        
        Args:
            dt: Time step size (s)
            max_iter: Maximum Newton iterations per step
            tol: Convergence tolerance for ||q||
        """
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        
        # Statistics
        self.total_steps = 0
        self.total_iters = 0
        self.total_failures = 0
        self.max_resid_seen = 0.0
        
    def step(self, t: float, dae, 
             update_equations: Callable,
             update_jacobians: Callable,
             predictor: str = 'constant') -> Tuple[bool, int, float]:
        """
        Take one implicit trapezoid step for DAE system.
        
        Args:
            t: Current time
            dae: DAESystem object with states x, y and equations f, g
            update_equations: Function(dae, t) that updates f, g
            update_jacobians: Function(dae, t) that updates fx, fy, gx, gy
            predictor: 'constant' (x_new = x_old) or 'euler' (x_new = x + h*f)
        
        Returns:
            (converged, iterations, final_residual)
        """
        h = self.dt
        n, m = dae.n, dae.m  # n = differential states, m = algebraic states
        
        # Save old states and equations
        x_old = dae.x.copy()
        y_old = dae.y.copy()
        
        # Evaluate at old time
        update_equations(dae, t)
        f_old = dae.f.copy()
        # g_old not needed (algebraic constraint at new time only)
        
        # Predictor: initial guess for x_{n+1}, y_{n+1}
        if predictor == 'euler':
            # Explicit Euler predictor
            dae.x = x_old + h * f_old
            # Keep y_old as guess for algebraic states
        elif predictor == 'constant':
            # Keep x_old, y_old (better for stiff systems)
            dae.x = x_old.copy()
            dae.y = y_old.copy()
        elif predictor == 'semi-explicit':
            # Semi-explicit: update x with Euler, keep y constant
            # This is essentially a semi-explicit DAE solver
            dae.x = x_old + h * f_old
            dae.y = y_old.copy()
        else:
            raise ValueError(f"Unknown predictor: {predictor}")
        
        # Newton-Raphson iteration
        converged = False
        t_new = t + h
        
        for iteration in range(self.max_iter):
            # Update equations at new time
            update_equations(dae, t_new)
            f_new = dae.f.copy()
            g_new = dae.g.copy()
            
            # Compute residual q
            q_x = dae.Teye @ (dae.x - x_old) - (h/2) * (f_old + f_new)
            q_y = g_new
            q = np.concatenate([q_x, q_y])
            
            # Check convergence
            norm_q = np.linalg.norm(q, np.inf)  # Max norm (like ANDES)
            
            if iteration == 0:
                initial_resid = norm_q
            
            # Update Jacobians (always do at least one iteration)
            update_jacobians(dae, t_new)
            
            # Build augmented Jacobian Ac
            # Ac = [[Teye - (h/2)*fx,  h*gx^T    ],
            #       [-(h/2)*fy^T,       h*gy     ]]
            Ac = dae.build_jacobian_ac(h, config_g_scale=1.0)
            
            # Solve: Ac * [Δx; Δy] = -q
            # Use least-squares solve to handle potential rank deficiency
            try:
                # First try direct solve (faster if matrix is well-conditioned)
                Ac_dense = Ac.toarray() if hasattr(Ac, 'toarray') else Ac
                delta = np.linalg.solve(Ac_dense, -q)
            except np.linalg.LinAlgError as e:
                # If singular, use least-squares (pseudo-inverse)
                print(f"[WARN] Singular Jacobian at t={t_new:.6f}s, using least-squares solve")
                Ac_dense = Ac.toarray() if hasattr(Ac, 'toarray') else Ac
                delta, residuals, rank, s = np.linalg.lstsq(Ac_dense, -q, rcond=None)
                print(f"   Matrix rank: {rank}/{Ac.shape[0]}, residual: {np.linalg.norm(residuals):.6e}")
                if rank < Ac.shape[0] - 30:  # Too rank deficient
                    print(f"[ERR] Matrix too rank deficient (rank={rank}/{Ac.shape[0]})")
                    converged = False
                    break
            
            # Update states
            dae.x += delta[:n]
            dae.y += delta[n:]
            
            # Check convergence AFTER update
            if norm_q < self.tol:
                converged = True
                self.total_iters += iteration + 1
                break
            
            # Prevent runaway
            if norm_q > 1e6:
                print(f"[ERR] Newton solver: residual exploding (||q||={norm_q:.2e})")
                converged = False
                break
        
        # Statistics
        self.total_steps += 1
        if not converged:
            self.total_failures += 1
            self.total_iters += self.max_iter
            
        self.max_resid_seen = max(self.max_resid_seen, norm_q)
        
        # If failed, revert states
        if not converged:
            dae.x = x_old
            dae.y = y_old
        
        return converged, iteration + 1, norm_q
    
    def print_statistics(self):
        """Print integration statistics."""
        print("\n" + "="*70)
        print("INTEGRATION STATISTICS")
        print("="*70)
        print(f"Total steps:        {self.total_steps}")
        print(f"Failed steps:       {self.total_failures} ({self.total_failures/max(1,self.total_steps)*100:.1f}%)")
        print(f"Total iterations:   {self.total_iters}")
        print(f"Avg iters/step:     {self.total_iters/max(1,self.total_steps):.2f}")
        print(f"Max residual seen:  {self.max_resid_seen:.6e}")
        print("="*70)
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_steps = 0
        self.total_iters = 0
        self.total_failures = 0
        self.max_resid_seen = 0.0
