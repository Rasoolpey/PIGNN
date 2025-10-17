"""
Error metrics calculation for comparing solver results with DIgSILENT reference data.

This module provides comprehensive error analysis including:
- Voltage magnitude and angle errors
- Power flow errors 
- Loss comparison
- Statistical summaries and convergence analysis
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.load_flow_solver import LoadFlowResults


@dataclass
class ComparisonResults:
    """
    Container for comparison results between solver and DIgSILENT outputs.
    """
    # Voltage comparison
    voltage_mag_error_pu: Optional[np.ndarray] = None
    voltage_mag_error_percent: Optional[np.ndarray] = None
    voltage_angle_error_deg: Optional[np.ndarray] = None
    
    # Flow comparison  
    p_flow_error_mw: Optional[np.ndarray] = None
    q_flow_error_mvar: Optional[np.ndarray] = None
    p_flow_error_percent: Optional[np.ndarray] = None
    q_flow_error_percent: Optional[np.ndarray] = None
    
    # Loss comparison
    p_loss_error_mw: Optional[np.ndarray] = None
    q_loss_error_mvar: Optional[np.ndarray] = None
    
    # Summary statistics
    max_voltage_error_pu: float = 0.0
    max_voltage_error_percent: float = 0.0
    max_angle_error_deg: float = 0.0
    rms_voltage_error_pu: float = 0.0
    rms_angle_error_deg: float = 0.0
    
    max_p_flow_error_percent: float = 0.0
    max_q_flow_error_percent: float = 0.0
    rms_p_flow_error_percent: float = 0.0
    rms_q_flow_error_percent: float = 0.0
    
    # Bus and branch mappings
    bus_names: Optional[List[str]] = None
    branch_names: Optional[List[str]] = None
    
    # Convergence comparison
    solver_converged: bool = True
    solver_iterations: int = 0
    digsilent_converged: bool = True
    
    def summary(self) -> Dict[str, float]:
        """Return summary statistics as dictionary."""
        return {
            'max_voltage_error_pu': self.max_voltage_error_pu,
            'max_voltage_error_percent': self.max_voltage_error_percent,
            'max_angle_error_deg': self.max_angle_error_deg,
            'rms_voltage_error_pu': self.rms_voltage_error_pu,
            'rms_angle_error_deg': self.rms_angle_error_deg,
            'max_p_flow_error_percent': self.max_p_flow_error_percent,
            'max_q_flow_error_percent': self.max_q_flow_error_percent,
            'rms_p_flow_error_percent': self.rms_p_flow_error_percent,
            'rms_q_flow_error_percent': self.rms_q_flow_error_percent,
            'solver_converged': self.solver_converged,
            'solver_iterations': self.solver_iterations
        }


class ErrorCalculator:
    """
    Calculate error metrics between solver results and DIgSILENT reference data.
    """
    
    def __init__(self, tolerance_voltage_pu: float = 0.001, tolerance_angle_deg: float = 0.1):
        """
        Initialize error calculator.
        
        Args:
            tolerance_voltage_pu: Voltage magnitude tolerance in per-unit
            tolerance_angle_deg: Voltage angle tolerance in degrees
        """
        self.tolerance_voltage_pu = tolerance_voltage_pu
        self.tolerance_angle_deg = tolerance_angle_deg
    
    def compare_voltages(self, 
                        solver_voltages: Dict[str, complex],
                        digsilent_voltages: Dict[str, Any],
                        base_voltage_kv: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compare voltage results between solver and DIgSILENT.
        
        Args:
            solver_voltages: Dictionary mapping bus names to complex voltages (pu)
            digsilent_voltages: Dictionary with DIgSILENT voltage data
            base_voltage_kv: Base voltages for each bus (optional)
            
        Returns:
            Tuple of (magnitude_errors_pu, angle_errors_deg, common_buses)
        """
        # Find common buses
        solver_buses = set(solver_voltages.keys())
        digsilent_buses = set(digsilent_voltages['bus_names'])
        common_buses = list(solver_buses.intersection(digsilent_buses))
        
        if not common_buses:
            raise ValueError("No common buses found between solver and DIgSILENT results")
        
        # Create mappings
        digsilent_bus_idx = {bus: i for i, bus in enumerate(digsilent_voltages['bus_names'])}
        
        mag_errors = []
        angle_errors = []
        
        for bus in common_buses:
            # Get solver results
            solver_v = solver_voltages[bus]
            solver_mag = abs(solver_v)
            solver_angle = np.degrees(np.angle(solver_v))
            
            # Get DIgSILENT results
            ds_idx = digsilent_bus_idx[bus]
            ds_mag = digsilent_voltages['v_magnitude_pu'][ds_idx]
            ds_angle = digsilent_voltages['v_angle_deg'][ds_idx]
            
            # Calculate errors
            mag_error = solver_mag - ds_mag
            angle_error = solver_angle - ds_angle
            
            # Handle angle wrap-around
            angle_error = self._normalize_angle_error(angle_error)
            
            mag_errors.append(mag_error)
            angle_errors.append(angle_error)
        
        return np.array(mag_errors), np.array(angle_errors), common_buses
    
    def compare_flows(self,
                     solver_flows: Dict[Tuple[str, str], Dict[str, float]],
                     digsilent_flows: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compare power flow results between solver and DIgSILENT.
        
        Args:
            solver_flows: Dictionary mapping (from_bus, to_bus) tuples to flow data
            digsilent_flows: Dictionary with DIgSILENT flow data
            
        Returns:
            Tuple of (p_flow_errors_mw, q_flow_errors_mvar, common_branches)
        """
        # Create DIgSILENT branch mapping
        digsilent_branches = {}
        for i, (from_bus, to_bus) in enumerate(zip(digsilent_flows['from_buses'], 
                                                  digsilent_flows['to_buses'])):
            key = (from_bus, to_bus)
            digsilent_branches[key] = i
            # Also add reverse direction
            reverse_key = (to_bus, from_bus)
            if reverse_key not in digsilent_branches:
                digsilent_branches[reverse_key] = i
        
        # Find common branches
        solver_branches = set(solver_flows.keys())
        digsilent_branch_keys = set(digsilent_branches.keys())
        common_branches = list(solver_branches.intersection(digsilent_branch_keys))
        
        if not common_branches:
            raise ValueError("No common branches found between solver and DIgSILENT results")
        
        p_errors = []
        q_errors = []
        branch_names = []
        
        for branch in common_branches:
            # Get solver results (assumed in MW/MVAr)
            solver_data = solver_flows[branch]
            solver_p = solver_data.get('P_MW', solver_data.get('p_flow', 0.0))
            solver_q = solver_data.get('Q_MVAr', solver_data.get('q_flow', 0.0))
            
            # Get DIgSILENT results
            ds_idx = digsilent_branches[branch]
            ds_p = digsilent_flows['p_flow_mw'][ds_idx]
            ds_q = digsilent_flows['q_flow_mvar'][ds_idx]
            
            # Check if we need to reverse direction for proper comparison
            if branch != (digsilent_flows['from_buses'][ds_idx], 
                         digsilent_flows['to_buses'][ds_idx]):
                # Reverse direction
                ds_p = -ds_p
                ds_q = -ds_q
            
            # Calculate errors
            p_error = solver_p - ds_p
            q_error = solver_q - ds_q
            
            p_errors.append(p_error)
            q_errors.append(q_error)
            branch_names.append(f"{branch[0]}-{branch[1]}")
        
        return np.array(p_errors), np.array(q_errors), branch_names
    
    def calculate_comprehensive_comparison(self,
                                        solver_results: LoadFlowResults,
                                        digsilent_data: Dict[str, Any]) -> ComparisonResults:
        """
        Perform comprehensive comparison between solver and DIgSILENT results.
        
        Args:
            solver_results: LoadFlowResults from the custom solver
            digsilent_data: Parsed DIgSILENT data dictionary
            
        Returns:
            ComparisonResults with all error metrics
        """
        results = ComparisonResults()
        
        # Extract convergence info
        results.solver_converged = solver_results.converged
        results.solver_iterations = getattr(solver_results, 'iterations', 0)
        
        # Voltage comparison
        if (hasattr(solver_results, 'voltages') and 
            solver_results.voltages is not None and 
            'bus_names' in digsilent_data):
            try:
                # Convert voltage array to dictionary if needed
                if isinstance(solver_results.voltages, np.ndarray):
                    # Create dummy bus names if node_mapping not available
                    if hasattr(solver_results, 'node_mapping') and solver_results.node_mapping:
                        voltage_dict = {bus: solver_results.voltages[idx] 
                                      for bus, idx in solver_results.node_mapping.items()}
                    else:
                        voltage_dict = {f"Bus{i+1}": solver_results.voltages[i] 
                                      for i in range(len(solver_results.voltages))}
                else:
                    voltage_dict = solver_results.voltages
                
                mag_errors, angle_errors, common_buses = self.compare_voltages(
                    voltage_dict, digsilent_data)
                
                results.voltage_mag_error_pu = mag_errors
                results.voltage_angle_error_deg = angle_errors
                results.bus_names = common_buses
                
                # Calculate percentage errors
                ds_bus_idx = {bus: i for i, bus in enumerate(digsilent_data['bus_names'])}
                ds_mags = np.array([digsilent_data['v_magnitude_pu'][ds_bus_idx[bus]] 
                                  for bus in common_buses])
                results.voltage_mag_error_percent = (mag_errors / ds_mags) * 100
                
                # Summary statistics
                results.max_voltage_error_pu = np.max(np.abs(mag_errors))
                results.max_voltage_error_percent = np.max(np.abs(results.voltage_mag_error_percent))
                results.max_angle_error_deg = np.max(np.abs(angle_errors))
                results.rms_voltage_error_pu = np.sqrt(np.mean(mag_errors**2))
                results.rms_angle_error_deg = np.sqrt(np.mean(angle_errors**2))
                
            except Exception as e:
                print(f"Warning: Voltage comparison failed: {e}")
        
        # Flow comparison
        if (hasattr(solver_results, 'branch_flows') and 
            'branch_from_buses' in digsilent_data):
            try:
                p_errors, q_errors, branch_names = self.compare_flows(
                    solver_results.branch_flows, 
                    {
                        'from_buses': digsilent_data['branch_from_buses'],
                        'to_buses': digsilent_data['branch_to_buses'], 
                        'p_flow_mw': digsilent_data['branch_p_flow_mw'],
                        'q_flow_mvar': digsilent_data['branch_q_flow_mvar']
                    })
                
                results.p_flow_error_mw = p_errors
                results.q_flow_error_mvar = q_errors
                results.branch_names = branch_names
                
                # Calculate percentage errors (avoid division by zero)
                ds_p = np.array([digsilent_data['branch_p_flow_mw'][
                    digsilent_data['branch_from_buses'].index(name.split('-')[0])] 
                    for name in branch_names])
                ds_q = np.array([digsilent_data['branch_q_flow_mvar'][
                    digsilent_data['branch_from_buses'].index(name.split('-')[0])]
                    for name in branch_names])
                
                # Avoid division by very small numbers
                p_mask = np.abs(ds_p) > 0.01  # 0.01 MW threshold
                q_mask = np.abs(ds_q) > 0.01  # 0.01 MVAr threshold
                
                results.p_flow_error_percent = np.zeros_like(p_errors)
                results.q_flow_error_percent = np.zeros_like(q_errors)
                
                if np.any(p_mask):
                    results.p_flow_error_percent[p_mask] = (p_errors[p_mask] / ds_p[p_mask]) * 100
                if np.any(q_mask):
                    results.q_flow_error_percent[q_mask] = (q_errors[q_mask] / ds_q[q_mask]) * 100
                
                # Summary statistics
                results.max_p_flow_error_percent = np.max(np.abs(results.p_flow_error_percent))
                results.max_q_flow_error_percent = np.max(np.abs(results.q_flow_error_percent))
                results.rms_p_flow_error_percent = np.sqrt(np.mean(results.p_flow_error_percent**2))
                results.rms_q_flow_error_percent = np.sqrt(np.mean(results.q_flow_error_percent**2))
                
            except Exception as e:
                print(f"Warning: Flow comparison failed: {e}")
        
        return results
    
    def _normalize_angle_error(self, angle_error: float) -> float:
        """Normalize angle error to [-180, 180] degrees."""
        while angle_error > 180:
            angle_error -= 360
        while angle_error < -180:
            angle_error += 360
        return angle_error
    
    def assess_convergence_quality(self, comparison: ComparisonResults) -> Dict[str, Any]:
        """
        Assess the quality of convergence based on error magnitudes.
        
        Returns:
            Dictionary with quality assessment
        """
        assessment = {
            'voltage_quality': 'unknown',
            'flow_quality': 'unknown',
            'overall_quality': 'unknown',
            'issues': []
        }
        
        # Voltage quality assessment
        if comparison.max_voltage_error_pu is not None:
            if comparison.max_voltage_error_pu < self.tolerance_voltage_pu:
                assessment['voltage_quality'] = 'excellent'
            elif comparison.max_voltage_error_pu < 2 * self.tolerance_voltage_pu:
                assessment['voltage_quality'] = 'good'
            elif comparison.max_voltage_error_pu < 5 * self.tolerance_voltage_pu:
                assessment['voltage_quality'] = 'acceptable'
            else:
                assessment['voltage_quality'] = 'poor'
                assessment['issues'].append('High voltage magnitude errors')
        
        # Angle quality assessment
        if comparison.max_angle_error_deg is not None:
            if comparison.max_angle_error_deg > self.tolerance_angle_deg:
                assessment['issues'].append('High voltage angle errors')
        
        # Flow quality assessment
        if comparison.max_p_flow_error_percent is not None:
            if comparison.max_p_flow_error_percent < 1.0:  # < 1%
                assessment['flow_quality'] = 'excellent'
            elif comparison.max_p_flow_error_percent < 5.0:  # < 5%
                assessment['flow_quality'] = 'good'
            elif comparison.max_p_flow_error_percent < 10.0:  # < 10%
                assessment['flow_quality'] = 'acceptable'
            else:
                assessment['flow_quality'] = 'poor'
                assessment['issues'].append('High power flow errors')
        
        # Overall assessment
        if not comparison.solver_converged:
            assessment['overall_quality'] = 'failed'
            assessment['issues'].append('Solver did not converge')
        elif 'poor' in [assessment['voltage_quality'], assessment['flow_quality']]:
            assessment['overall_quality'] = 'poor'
        elif 'acceptable' in [assessment['voltage_quality'], assessment['flow_quality']]:
            assessment['overall_quality'] = 'acceptable'
        elif 'good' in [assessment['voltage_quality'], assessment['flow_quality']]:
            assessment['overall_quality'] = 'good'
        elif assessment['voltage_quality'] == 'excellent' and assessment['flow_quality'] == 'excellent':
            assessment['overall_quality'] = 'excellent'
        else:
            assessment['overall_quality'] = 'good'
        
        return assessment