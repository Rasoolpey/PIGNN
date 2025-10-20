"""
Time Integrators for RMS Simulation

This module implements numerical integration methods for solving ODEs:
- RK4Integrator: 4th-order Runge-Kutta (explicit, accurate)
- TrapezoidalIntegrator: Trapezoidal rule (implicit, stable)

Author: PIGNN Project
Date: October 20, 2025
"""

import numpy as np
from typing import Callable, Tuple


class RK4Integrator:
    """
    4th-Order Runge-Kutta Integrator
    
    Explicit method with good accuracy for smooth systems.
    Error: O(h^5) per step
    """
    
    def __init__(self, dt: float):
        """
        Initialize RK4 integrator.
        
        Args:
            dt: Time step size (s)
        """
        self.dt = dt
    
    def step(self, t: float, y: np.ndarray, 
             derivative_func: Callable[[float, np.ndarray], np.ndarray]) -> Tuple[float, np.ndarray]:
        """
        Take one RK4 integration step.
        
        Args:
            t: Current time
            y: Current state vector
            derivative_func: Function computing dy/dt = f(t, y)
        
        Returns:
            (t_new, y_new) after one time step
        """
        dt = self.dt
        
        # RK4 coefficients
        k1 = derivative_func(t, y)
        k2 = derivative_func(t + dt/2, y + dt/2 * k1)
        k3 = derivative_func(t + dt/2, y + dt/2 * k2)
        k4 = derivative_func(t + dt, y + dt * k3)
        
        # Update
        y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_new = t + dt
        
        return t_new, y_new


class TrapezoidalIntegrator:
    """
    Trapezoidal Rule Integrator (Implicit)
    
    Implicit method with excellent stability (A-stable).
    Requires iteration for nonlinear systems.
    """
    
    def __init__(self, dt: float, max_iter: int = 10, tol: float = 1e-6):
        """
        Initialize trapezoidal integrator.
        
        Args:
            dt: Time step size (s)
            max_iter: Maximum Newton iterations
            tol: Convergence tolerance
        """
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
    
    def step(self, t: float, y: np.ndarray,
             derivative_func: Callable[[float, np.ndarray], np.ndarray]) -> Tuple[float, np.ndarray]:
        """
        Take one trapezoidal integration step.
        
        Solves: y_new = y_old + (dt/2) * (f(t, y) + f(t+dt, y_new))
        
        Args:
            t: Current time
            y: Current state vector
            derivative_func: Function computing dy/dt = f(t, y)
        
        Returns:
            (t_new, y_new) after one time step
        """
        dt = self.dt
        t_new = t + dt
        
        # Initial guess (explicit Euler)
        f_old = derivative_func(t, y)
        y_new = y + dt * f_old
        
        # Newton iteration
        for iteration in range(self.max_iter):
            # Evaluate at new time
            f_new = derivative_func(t_new, y_new)
            
            # Residual: R = y_new - y_old - (dt/2)*(f_old + f_new)
            residual = y_new - y - (dt/2) * (f_old + f_new)
            
            # Check convergence
            if np.linalg.norm(residual) < self.tol:
                break
            
            # Numerical jacobian (simplified - using finite differences)
            # For production, supply analytical jacobian
            epsilon = 1e-6
            J_approx = np.eye(len(y))
            for i in range(len(y)):
                y_perturb = y_new.copy()
                y_perturb[i] += epsilon
                f_perturb = derivative_func(t_new, y_perturb)
                J_approx[:, i] = (f_perturb - f_new) / epsilon
            
            # System matrix: I - (dt/2)*J
            A = np.eye(len(y)) - (dt/2) * J_approx
            
            # Newton update
            try:
                delta = np.linalg.solve(A, -residual)
                y_new = y_new + delta
            except np.linalg.LinAlgError:
                # Singular matrix - fall back to previous guess
                print(f"⚠️  Trapezoidal: singular matrix at t={t_new:.3f}s, using previous guess")
                break
        
        return t_new, y_new


class AdaptiveRK4Integrator:
    """
    Adaptive RK4 with error control (optional advanced feature)
    
    Adjusts time step based on local truncation error.
    """
    
    def __init__(self, dt_initial: float = 0.01, dt_min: float = 1e-6, 
                 dt_max: float = 0.1, tol: float = 1e-5):
        """
        Initialize adaptive RK4.
        
        Args:
            dt_initial: Initial time step
            dt_min: Minimum allowed time step
            dt_max: Maximum allowed time step
            tol: Error tolerance
        """
        self.dt = dt_initial
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tol = tol
        
        # Statistics
        self.n_accepted = 0
        self.n_rejected = 0
    
    def step(self, t: float, y: np.ndarray,
             derivative_func: Callable[[float, np.ndarray], np.ndarray]) -> Tuple[float, np.ndarray]:
        """
        Take one adaptive step with error control.
        
        Uses embedded RK methods to estimate error and adjust step size.
        
        Returns:
            (t_new, y_new)
        """
        while True:
            dt = self.dt
            
            # Full step
            k1 = derivative_func(t, y)
            k2 = derivative_func(t + dt/2, y + dt/2 * k1)
            k3 = derivative_func(t + dt/2, y + dt/2 * k2)
            k4 = derivative_func(t + dt, y + dt * k3)
            y_full = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Two half steps
            k1h = derivative_func(t, y)
            k2h = derivative_func(t + dt/4, y + dt/4 * k1h)
            k3h = derivative_func(t + dt/4, y + dt/4 * k2h)
            k4h = derivative_func(t + dt/2, y + dt/2 * k3h)
            y_half = y + (dt/2 / 6.0) * (k1h + 2*k2h + 2*k3h + k4h)
            
            k1h2 = derivative_func(t + dt/2, y_half)
            k2h2 = derivative_func(t + 3*dt/4, y_half + dt/4 * k1h2)
            k3h2 = derivative_func(t + 3*dt/4, y_half + dt/4 * k2h2)
            k4h2 = derivative_func(t + dt, y_half + dt/2 * k3h2)
            y_two_half = y_half + (dt/2 / 6.0) * (k1h2 + 2*k2h2 + 2*k3h2 + k4h2)
            
            # Error estimate
            error = np.linalg.norm(y_two_half - y_full) / (1 + np.linalg.norm(y))
            
            # Accept or reject
            if error < self.tol or self.dt <= self.dt_min:
                # Accept step
                self.n_accepted += 1
                
                # Adjust step size for next step
                if error > 0:
                    self.dt = min(self.dt * min(2.0, 0.9 * (self.tol / error)**0.2), self.dt_max)
                
                return t + dt, y_two_half  # Use more accurate solution
            else:
                # Reject and retry with smaller step
                self.n_rejected += 1
                self.dt = max(self.dt * max(0.5, 0.9 * (self.tol / error)**0.25), self.dt_min)


def create_integrator(method: str = 'rk4', dt: float = 0.01, **kwargs):
    """
    Factory function to create integrator.
    
    Args:
        method: 'rk4', 'trapezoidal', or 'adaptive'
        dt: Time step size
        **kwargs: Additional parameters
    
    Returns:
        Integrator object
    """
    if method.lower() == 'rk4':
        return RK4Integrator(dt)
    elif method.lower() == 'trapezoidal':
        return TrapezoidalIntegrator(dt, **kwargs)
    elif method.lower() == 'adaptive':
        return AdaptiveRK4Integrator(dt, **kwargs)
    else:
        raise ValueError(f"Unknown integration method: {method}")
