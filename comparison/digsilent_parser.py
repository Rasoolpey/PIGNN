"""
DIgSILENT PowerFactory output parser for CSV/Excel exports.

This module handles parsing of DIgSILENT exported data including:
- Bus voltages (magnitude, angle)
- Branch flows (P, Q, losses)
- Generator outputs
- Load values
"""

import os
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


class DIgSILENTParser:
    """
    Parser for DIgSILENT PowerFactory exported data.
    
    Supports common export formats:
    - CSV files with bus voltage data
    - CSV files with branch flow data  
    - Excel files with multiple sheets
    """
    
    def __init__(self, use_pandas: bool = False):
        """
        Initialize parser.
        
        Args:
            use_pandas: If True, use pandas for CSV parsing (requires pandas installation)
                       If False, use built-in csv module
        """
        self.use_pandas = use_pandas
        if use_pandas:
            try:
                import pandas as pd
                self.pd = pd
            except ImportError:
                print("Warning: pandas not available, falling back to csv module")
                self.use_pandas = False
        
        if not self.use_pandas:
            import csv
            self.csv = csv
    
    def parse_bus_voltages(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse bus voltage data from DIgSILENT export.
        
        Expected format:
        Bus Name, V_mag (p.u.), V_angle (deg), V_mag (kV)
        
        Args:
            file_path: Path to CSV file with bus voltage data
            
        Returns:
            Dictionary with bus voltage data:
            {
                'bus_names': List[str],
                'v_magnitude_pu': np.ndarray,
                'v_angle_deg': np.ndarray,
                'v_magnitude_kv': np.ndarray
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.use_pandas:
            return self._parse_bus_voltages_pandas(file_path)
        else:
            return self._parse_bus_voltages_csv(file_path)
    
    def parse_branch_flows(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse branch flow data from DIgSILENT export.
        
        Expected format:
        From Bus, To Bus, P_flow (MW), Q_flow (MVAr), P_loss (MW), Q_loss (MVAr)
        
        Args:
            file_path: Path to CSV file with branch flow data
            
        Returns:
            Dictionary with branch flow data:
            {
                'from_buses': List[str],
                'to_buses': List[str], 
                'p_flow_mw': np.ndarray,
                'q_flow_mvar': np.ndarray,
                'p_loss_mw': np.ndarray,
                'q_loss_mvar': np.ndarray
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if self.use_pandas:
            return self._parse_branch_flows_pandas(file_path)
        else:
            return self._parse_branch_flows_csv(file_path)
    
    def _parse_bus_voltages_pandas(self, file_path: Path) -> Dict[str, Any]:
        """Parse bus voltages using pandas."""
        df = self.pd.read_csv(file_path)
        
        # Handle different possible column names
        bus_col = self._find_column(df, ['bus', 'name', 'bus_name', 'node'])
        vmag_pu_col = self._find_column(df, ['v_mag_pu', 'vmag_pu', 'voltage_pu', 'v_pu'])
        vang_col = self._find_column(df, ['v_angle', 'angle', 'v_ang', 'phase'])
        vmag_kv_col = self._find_column(df, ['v_mag_kv', 'vmag_kv', 'voltage_kv', 'v_kv'])
        
        return {
            'bus_names': df[bus_col].tolist(),
            'v_magnitude_pu': df[vmag_pu_col].values,
            'v_angle_deg': df[vang_col].values,
            'v_magnitude_kv': df[vmag_kv_col].values if vmag_kv_col else None
        }
    
    def _parse_bus_voltages_csv(self, file_path: Path) -> Dict[str, Any]:
        """Parse bus voltages using built-in csv module."""
        bus_names = []
        v_mag_pu = []
        v_angle = []
        v_mag_kv = []
        
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = self.csv.DictReader(csvfile)
            
            for row in reader:
                # Try to find the right columns (case-insensitive)
                row_lower = {k.lower(): v for k, v in row.items()}
                
                bus_name = self._extract_value(row_lower, ['bus', 'name', 'bus_name', 'node', 'bus name'])
                vmag_pu = self._extract_value(row_lower, ['v_mag_pu', 'vmag_pu', 'voltage_pu', 'v_pu', 'v_mag (p.u.)'])
                vang = self._extract_value(row_lower, ['v_angle', 'angle', 'v_ang', 'phase', 'v_angle (deg)'])
                vmag_kv = self._extract_value(row_lower, ['v_mag_kv', 'vmag_kv', 'voltage_kv', 'v_kv', 'v_mag (kv)'])
                
                if bus_name and vmag_pu is not None and vang is not None:
                    bus_names.append(bus_name)
                    v_mag_pu.append(float(vmag_pu))
                    v_angle.append(float(vang))
                    v_mag_kv.append(float(vmag_kv) if vmag_kv is not None else None)
        
        return {
            'bus_names': bus_names,
            'v_magnitude_pu': np.array(v_mag_pu),
            'v_angle_deg': np.array(v_angle),
            'v_magnitude_kv': np.array(v_mag_kv) if any(v is not None for v in v_mag_kv) else None
        }
    
    def _parse_branch_flows_pandas(self, file_path: Path) -> Dict[str, Any]:
        """Parse branch flows using pandas."""
        df = self.pd.read_csv(file_path)
        
        from_col = self._find_column(df, ['from', 'from_bus', 'bus_from', 'from_node'])
        to_col = self._find_column(df, ['to', 'to_bus', 'bus_to', 'to_node'])
        p_flow_col = self._find_column(df, ['p_flow', 'p', 'active_power', 'p_mw'])
        q_flow_col = self._find_column(df, ['q_flow', 'q', 'reactive_power', 'q_mvar'])
        p_loss_col = self._find_column(df, ['p_loss', 'loss_p', 'active_loss', 'ploss'])
        q_loss_col = self._find_column(df, ['q_loss', 'loss_q', 'reactive_loss', 'qloss'])
        
        return {
            'from_buses': df[from_col].tolist(),
            'to_buses': df[to_col].tolist(),
            'p_flow_mw': df[p_flow_col].values,
            'q_flow_mvar': df[q_flow_col].values,
            'p_loss_mw': df[p_loss_col].values if p_loss_col else np.zeros(len(df)),
            'q_loss_mvar': df[q_loss_col].values if q_loss_col else np.zeros(len(df))
        }
    
    def _parse_branch_flows_csv(self, file_path: Path) -> Dict[str, Any]:
        """Parse branch flows using built-in csv module."""
        from_buses = []
        to_buses = []
        p_flows = []
        q_flows = []
        p_losses = []
        q_losses = []
        
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = self.csv.DictReader(csvfile)
            
            for row in reader:
                row_lower = {k.lower(): v for k, v in row.items()}
                
                from_bus = self._extract_value(row_lower, ['from', 'from_bus', 'bus_from', 'from_node', 'from bus'])
                to_bus = self._extract_value(row_lower, ['to', 'to_bus', 'bus_to', 'to_node', 'to bus'])
                p_flow = self._extract_value(row_lower, ['p_flow', 'p', 'active_power', 'p_mw', 'p_flow (mw)'])
                q_flow = self._extract_value(row_lower, ['q_flow', 'q', 'reactive_power', 'q_mvar', 'q_flow (mvar)'])
                p_loss = self._extract_value(row_lower, ['p_loss', 'loss_p', 'active_loss', 'ploss', 'p_loss (mw)'])
                q_loss = self._extract_value(row_lower, ['q_loss', 'loss_q', 'reactive_loss', 'qloss', 'q_loss (mvar)'])
                
                if from_bus and to_bus and p_flow is not None and q_flow is not None:
                    from_buses.append(from_bus)
                    to_buses.append(to_bus)
                    p_flows.append(float(p_flow))
                    q_flows.append(float(q_flow))
                    p_losses.append(float(p_loss) if p_loss is not None else 0.0)
                    q_losses.append(float(q_loss) if q_loss is not None else 0.0)
        
        return {
            'from_buses': from_buses,
            'to_buses': to_buses,
            'p_flow_mw': np.array(p_flows),
            'q_flow_mvar': np.array(q_flows),
            'p_loss_mw': np.array(p_losses),
            'q_loss_mvar': np.array(q_losses)
        }
    
    def _find_column(self, df, possible_names: List[str]) -> str:
        """Find column name from possible alternatives (case-insensitive)."""
        df_columns_lower = [col.lower() for col in df.columns]
        
        for name in possible_names:
            if name.lower() in df_columns_lower:
                idx = df_columns_lower.index(name.lower())
                return df.columns[idx]
        
        raise ValueError(f"Could not find column matching any of: {possible_names}")
    
    def _extract_value(self, row_dict: Dict[str, str], possible_keys: List[str]) -> Optional[str]:
        """Extract value from row dictionary using possible key names."""
        for key in possible_keys:
            if key in row_dict:
                return row_dict[key]
        return None
    
    def parse_scenario_file(self, scenario_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a complete scenario from DIgSILENT exports.
        
        Assumes the scenario directory contains:
        - bus_voltages.csv
        - branch_flows.csv
        
        Args:
            scenario_path: Path to directory containing scenario files
            
        Returns:
            Combined scenario data dictionary
        """
        scenario_path = Path(scenario_path)
        
        result = {}
        
        # Parse bus voltages if available
        bus_voltage_file = scenario_path / "bus_voltages.csv"
        if bus_voltage_file.exists():
            result.update(self.parse_bus_voltages(bus_voltage_file))
        
        # Parse branch flows if available
        branch_flow_file = scenario_path / "branch_flows.csv"
        if branch_flow_file.exists():
            branch_data = self.parse_branch_flows(branch_flow_file)
            result.update({f"branch_{k}": v for k, v in branch_data.items()})
        
        return result