"""
DAE System Infrastructure for RMS Dynamic Simulation.

Following ANDES architecture for differential-algebraic equation (DAE) systems:
    T * dx/dt = f(x, y, t)   -- Differential equations
    0 = g(x, y, t)            -- Algebraic equations

Where:
    x: differential states (generator rotor angles, speeds, flux linkages, etc.)
    y: algebraic states (bus voltages and angles)
    f: right-hand side of differential equations
    g: algebraic equation residuals (power balance at buses)
    T: time constant diagonal matrix (Teye)

Jacobians:
    fx: ∂f/∂x
    fy: ∂f/∂y  
    gx: ∂g/∂x
    gy: ∂g/∂y (network admittance-based Jacobian)
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DAESystem:
    """
    Differential-Algebraic Equation (DAE) system for RMS simulation.
    
    Stores all DAE variables, equations, and Jacobian matrices following
    the ANDES framework structure.
    
    Attributes
    ----------
    n : int
        Number of differential variables (states)
    m : int
        Number of algebraic variables
    x : np.ndarray
        Differential state variables [n]
    y : np.ndarray
        Algebraic variables [m]
    f : np.ndarray
        RHS of differential equations [n]
    g : np.ndarray
        Algebraic equation residuals [m]
    Tf : np.ndarray
        Time constants for differential equations [n]
    fx, fy, gx, gy : scipy.sparse matrices
        Jacobian matrices
    """
    
    def __init__(self):
        """Initialize empty DAE system."""
        # Sizes
        self.n = 0  # Number of differential states
        self.m = 0  # Number of algebraic states
        
        # State vectors
        self.x = np.array([])  # Differential states
        self.y = np.array([])  # Algebraic states
        
        # Equation RHS
        self.f = np.array([])  # Differential equation RHS
        self.g = np.array([])  # Algebraic equation residuals
        
        # Time constants (diagonal matrix stored as vector)
        self.Tf = np.array([])  # Time constant array for f
        
        # Jacobian matrices (sparse)
        self.fx = None  # ∂f/∂x [n x n]
        self.fy = None  # ∂f/∂y [n x m]
        self.gx = None  # ∂g/∂x [m x n]
        self.gy = None  # ∂g/∂y [m x m]
        
        # Mass matrix (Teye in ANDES)
        self.Teye = None  # Identity scaled by Tf [n x n]
        
        # Variable names for debugging
        self.x_names = []
        self.y_names = []
        
        # Addressing: maps from component to DAE indices
        self._x_addr = {}  # {component_id: slice or array of indices in x}
        self._y_addr = {}  # {component_id: slice or array of indices in y}
        
        # Time
        self.t = 0.0
        
        logger.info("DAE system initialized")
    
    def allocate_arrays(self, n: int, m: int):
        """
        Allocate arrays for n differential and m algebraic variables.
        
        Parameters
        ----------
        n : int
            Number of differential states
        m : int
            Number of algebraic states
        """
        self.n = n
        self.m = m
        
        # Allocate state vectors
        self.x = np.zeros(n)
        self.y = np.zeros(m)
        
        # Allocate equation vectors
        self.f = np.zeros(n)
        self.g = np.zeros(m)
        
        # Allocate time constants (default to 1.0)
        self.Tf = np.ones(n)
        
        # Create identity matrix scaled by Tf
        self._update_Teye()
        
        # Initialize Jacobians as zero sparse matrices
        self.fx = sparse.csr_matrix((n, n))
        self.fy = sparse.csr_matrix((n, m))
        self.gx = sparse.csr_matrix((m, n))
        self.gy = sparse.csr_matrix((m, m))
        
        logger.info(f"Allocated DAE arrays: {n} differential, {m} algebraic variables")
    
    def _update_Teye(self):
        """Update the Teye matrix (diagonal matrix of time constants)."""
        if self.n > 0:
            self.Teye = sparse.diags(self.Tf, format='csr')
    
    def request_address(self, component_id: str, n_diff: int, n_alg: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Request address space for a component's variables.
        
        Parameters
        ----------
        component_id : str
            Unique identifier for the component
        n_diff : int
            Number of differential states needed
        n_alg : int
            Number of algebraic states needed
            
        Returns
        -------
        x_addr : np.ndarray
            Indices in x array allocated to this component
        y_addr : np.ndarray
            Indices in y array allocated to this component
        """
        # Allocate differential states
        x_start = self.n
        x_end = x_start + n_diff
        x_addr = np.arange(x_start, x_end)
        self._x_addr[component_id] = x_addr
        self.n = x_end
        
        # Allocate algebraic states
        y_start = self.m
        y_end = y_start + n_alg
        y_addr = np.arange(y_start, y_end)
        self._y_addr[component_id] = y_addr
        self.m = y_end
        
        logger.debug(f"Allocated addresses for {component_id}: x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
        
        return x_addr, y_addr
    
    def get_address(self, component_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the address space for a component.
        
        Parameters
        ----------
        component_id : str
            Component identifier
            
        Returns
        -------
        x_addr : np.ndarray
            Indices in x array
        y_addr : np.ndarray
            Indices in y array
        """
        x_addr = self._x_addr.get(component_id, np.array([]))
        y_addr = self._y_addr.get(component_id, np.array([]))
        return x_addr, y_addr
    
    def clear_fg(self):
        """Reset equation arrays to zero."""
        self.f[:] = 0
        self.g[:] = 0
    
    def clear_xy(self):
        """Reset variable arrays to zero."""
        self.x[:] = 0
        self.y[:] = 0
    
    def set_time_constant(self, indices: np.ndarray, values: np.ndarray):
        """
        Set time constants for differential equations.
        
        Parameters
        ----------
        indices : np.ndarray
            Indices in the Tf array
        values : np.ndarray
            Time constant values
        """
        self.Tf[indices] = values
        self._update_Teye()
    
    def build_jacobian_ac(self, h: float, config_g_scale: float = 1.0) -> sparse.csr_matrix:
        """
        Build the augmented Jacobian matrix Ac for implicit trapezoidal method.
        
        Following ANDES, the augmented system for Newton's method is:
            Ac * [Δx; Δy] = -[q_diff; q_alg]
        
        Where:
            Ac = [[ Teye - 0.5*h*fx,  -0.5*h*gx^T ],
                  [ -0.5*h*fy,         gys        ]]
        
        Note: In our convention:
            - fx, fy: ∂f/∂x [n x n], ∂f/∂y [n x m]
            - gx, gy: ∂g/∂x [m x n], ∂g/∂y [m x m]
            - gx^T is used in top-right block to match dimensions
        
        Parameters
        ----------
        h : float
            Time step size
        config_g_scale : float
            Scaling factor for algebraic equations (default 1.0)
            
        Returns
        -------
        Ac : scipy.sparse.csr_matrix
            Augmented Jacobian matrix [(n+m) x (n+m)]
        """
        # Scale algebraic Jacobians if needed (ANDES applies this to entire gx, gy)
        if config_g_scale > 0:
            scale = config_g_scale * h
        else:
            scale = 1.0
        
        # Build Ac = [[A11, A12], [A21, A22]]
        # A11 = Teye - 0.5*h*fx [n x n]
        A11 = self.Teye - 0.5 * h * self.fx
        
        # A12 = scale*gx^T [n x m]
        A12 = scale * self.gx.T
        
        # A21 = -0.5*h*fy^T [m x n] (transpose fy to get correct dimensions)
        A21 = -0.5 * h * self.fy.T
        
        # A22 = scale*gy [m x m]
        A22 = scale * self.gy
        
        # Construct block matrix
        Ac = sparse.bmat([[A11, A12],
                          [A21, A22]], format='csr')
        
        return Ac
    
    def calc_residual_q(self, h: float, x0: np.ndarray, f0: np.ndarray, 
                        config_g_scale: float = 1.0) -> np.ndarray:
        """
        Calculate the residual vector q for implicit trapezoidal method.
        
        Following ANDES:
            q[:n] = Tf * (x - x0) - 0.5*h*(f + f0)   # Algebraized differential eqs
            q[n:] = g_scale * h * g                   # Scaled algebraic residuals
        
        Parameters
        ----------
        h : float
            Time step size
        x0 : np.ndarray
            Previous time step differential states
        f0 : np.ndarray
            Previous time step differential equation RHS
        config_g_scale : float
            Scaling factor for algebraic equations
            
        Returns
        -------
        q : np.ndarray
            Residual vector [n+m]
        """
        q = np.zeros(self.n + self.m)
        
        # Differential part: Tf * (x - x0) - 0.5*h*(f + f0)
        q[:self.n] = self.Tf * (self.x - x0) - 0.5 * h * (self.f + f0)
        
        # Algebraic part: g_scale * h * g
        if config_g_scale > 0:
            q[self.n:] = config_g_scale * h * self.g
        else:
            q[self.n:] = self.g
        
        return q
    
    def update_states(self, inc: np.ndarray):
        """
        Update states with Newton-Raphson increment.
        
        Parameters
        ----------
        inc : np.ndarray
            Increment vector [n+m]
        """
        self.x -= inc[:self.n]
        self.y -= inc[self.n:]
    
    def get_state_dict(self) -> Dict[str, np.ndarray]:
        """
        Get current state as a dictionary.
        
        Returns
        -------
        state : dict
            Dictionary with keys 'x', 'y', 'f', 'g'
        """
        return {
            'x': self.x.copy(),
            'y': self.y.copy(),
            'f': self.f.copy(),
            'g': self.g.copy(),
            't': self.t
        }
    
    def set_state_dict(self, state: Dict[str, np.ndarray]):
        """
        Restore state from dictionary.
        
        Parameters
        ----------
        state : dict
            Dictionary with keys 'x', 'y', 'f', 'g', 't'
        """
        self.x[:] = state['x']
        self.y[:] = state['y']
        self.f[:] = state['f']
        self.g[:] = state['g']
        self.t = state['t']
    
    def __repr__(self):
        return (f"DAESystem(n={self.n} differential, m={self.m} algebraic, "
                f"t={self.t:.4f}s)")
