"""
Batch comparison utilities for running contingency analysis and comparing with DIgSILENT results.

This module provides tools to:
- Run multiple contingency scenarios
- Compare results with DIgSILENT reference data
- Generate summary reports
- Store results for later analysis
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder
from physics.load_flow_solver import ThreePhaseLoadFlowSolver, LoadFlowResults
from comparison.digsilent_parser import DIgSILENTParser
from comparison.error_metrics import ErrorCalculator, ComparisonResults


class BatchComparator:
    """
    Batch processor for contingency analysis comparison with DIgSILENT results.
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "Contingency Analysis/contingency_out",
                 use_pandas: bool = False):
        """
        Initialize batch comparator.
        
        Args:
            output_dir: Directory to store comparison results
            use_pandas: Whether to use pandas for DIgSILENT parsing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = DIgSILENTParser(use_pandas=use_pandas)
        self.error_calc = ErrorCalculator()
        
        # Initialize components
        self.loader = H5DataLoader()
        self.graph_builder = GraphBuilder()
        self.solver = ThreePhaseLoadFlowSolver()
        
        # Results storage
        self.results = {}
        self.summary_stats = {}
    
    def run_scenario_comparison(self, 
                               scenario_file: Union[str, Path],
                               digsilent_data_dir: Union[str, Path],
                               scenario_name: Optional[str] = None) -> ComparisonResults:
        """
        Run comparison for a single scenario.
        
        Args:
            scenario_file: Path to HDF5 scenario file
            digsilent_data_dir: Directory containing DIgSILENT reference data
            scenario_name: Optional name for the scenario
            
        Returns:
            ComparisonResults object
        """
        scenario_file = Path(scenario_file)
        digsilent_data_dir = Path(digsilent_data_dir)
        
        if scenario_name is None:
            scenario_name = scenario_file.stem
        
        print(f"Processing scenario: {scenario_name}")
        
        try:
            # Load and build graph from H5 file
            raw_data = self.loader.load_scenario(scenario_file)
            graphs = self.graph_builder.build_three_phase_graphs(raw_data)
            
            # Run load flow solver
            print(f"  Running load flow solver...")
            solver_results = self.solver.solve_three_phase(graphs)
            
            # Load DIgSILENT reference data
            print(f"  Loading DIgSILENT reference data...")
            digsilent_data = self.parser.parse_scenario_file(digsilent_data_dir)
            
            # Perform comparison
            print(f"  Computing error metrics...")
            comparison = self.error_calc.calculate_comprehensive_comparison(
                solver_results, digsilent_data)
            
            # Store results
            self.results[scenario_name] = {
                'scenario_file': str(scenario_file),
                'comparison': comparison,
                'solver_results': solver_results,
                'digsilent_data': digsilent_data,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"  Completed. Max voltage error: {comparison.max_voltage_error_pu:.6f} pu")
            
            return comparison
            
        except Exception as e:
            print(f"  Error processing scenario {scenario_name}: {e}")
            # Store error info
            self.results[scenario_name] = {
                'scenario_file': str(scenario_file),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise
    
    def run_batch_comparison(self,
                           scenario_dir: Union[str, Path],
                           digsilent_dir: Union[str, Path],
                           scenario_pattern: str = "scenario_*.h5",
                           max_scenarios: Optional[int] = None) -> Dict[str, ComparisonResults]:
        """
        Run batch comparison for multiple scenarios.
        
        Args:
            scenario_dir: Directory containing HDF5 scenario files
            digsilent_dir: Directory containing DIgSILENT reference data
            scenario_pattern: File pattern for scenario files
            max_scenarios: Maximum number of scenarios to process (None for all)
            
        Returns:
            Dictionary mapping scenario names to ComparisonResults
        """
        scenario_dir = Path(scenario_dir)
        digsilent_dir = Path(digsilent_dir)
        
        # Find scenario files
        scenario_files = list(scenario_dir.glob(scenario_pattern))
        
        if max_scenarios:
            scenario_files = scenario_files[:max_scenarios]
        
        print(f"Found {len(scenario_files)} scenario files to process")
        
        comparisons = {}
        
        for i, scenario_file in enumerate(scenario_files):
            scenario_name = scenario_file.stem
            
            print(f"\nProcessing {i+1}/{len(scenario_files)}: {scenario_name}")
            
            try:
                # Assume DIgSILENT data is in subdirectory with same name
                scenario_digsilent_dir = digsilent_dir / scenario_name
                if not scenario_digsilent_dir.exists():
                    print(f"  Warning: DIgSILENT data not found at {scenario_digsilent_dir}")
                    continue
                
                comparison = self.run_scenario_comparison(
                    scenario_file, scenario_digsilent_dir, scenario_name)
                
                comparisons[scenario_name] = comparison
                
            except Exception as e:
                print(f"  Failed to process {scenario_name}: {e}")
                continue
        
        # Generate summary statistics
        self._generate_summary_stats(comparisons)
        
        # Save results
        self._save_batch_results()
        
        print(f"\nBatch comparison completed. Processed {len(comparisons)} scenarios successfully.")
        
        return comparisons
    
    def run_contingency_analysis(self,
                               base_scenario_file: Union[str, Path],
                               contingency_scenarios_dir: Union[str, Path],
                               digsilent_contingency_dir: Union[str, Path]) -> Dict[str, ComparisonResults]:
        """
        Run contingency analysis comparing N-1 scenarios with DIgSILENT.
        
        Args:
            base_scenario_file: Base case HDF5 file
            contingency_scenarios_dir: Directory with contingency scenario files
            digsilent_contingency_dir: Directory with DIgSILENT contingency results
            
        Returns:
            Dictionary mapping contingency names to ComparisonResults
        """
        print("Running contingency analysis...")
        
        # First run base case
        print("Processing base case...")
        base_comparison = self.run_scenario_comparison(
            base_scenario_file, 
            digsilent_contingency_dir / "base_case",
            "base_case")
        
        # Then run all contingency scenarios
        contingency_results = self.run_batch_comparison(
            contingency_scenarios_dir,
            digsilent_contingency_dir,
            scenario_pattern="scenario_*.h5")
        
        # Combine results
        all_results = {"base_case": base_comparison}
        all_results.update(contingency_results)
        
        # Generate contingency-specific analysis
        self._analyze_contingency_impacts(all_results)
        
        return all_results
    
    def _generate_summary_stats(self, comparisons: Dict[str, ComparisonResults]):
        """Generate summary statistics across all comparisons."""
        if not comparisons:
            return
        
        # Collect all error metrics
        voltage_errors = []
        angle_errors = []
        p_flow_errors = []
        q_flow_errors = []
        
        converged_count = 0
        total_iterations = []
        
        for name, comp in comparisons.items():
            if comp.voltage_mag_error_pu is not None:
                voltage_errors.extend(comp.voltage_mag_error_pu)
            if comp.voltage_angle_error_deg is not None:
                angle_errors.extend(comp.voltage_angle_error_deg)
            if comp.p_flow_error_percent is not None:
                p_flow_errors.extend(comp.p_flow_error_percent)
            if comp.q_flow_error_percent is not None:
                q_flow_errors.extend(comp.q_flow_error_percent)
            
            if comp.solver_converged:
                converged_count += 1
            
            total_iterations.append(comp.solver_iterations)
        
        # Calculate aggregate statistics
        self.summary_stats = {
            'total_scenarios': len(comparisons),
            'converged_scenarios': converged_count,
            'convergence_rate': converged_count / len(comparisons),
            'avg_iterations': np.mean(total_iterations),
            'max_iterations': np.max(total_iterations),
            
            'voltage_error_stats': {
                'max_abs': np.max(np.abs(voltage_errors)) if voltage_errors else 0,
                'rms': np.sqrt(np.mean(np.array(voltage_errors)**2)) if voltage_errors else 0,
                'mean': np.mean(voltage_errors) if voltage_errors else 0,
                'std': np.std(voltage_errors) if voltage_errors else 0
            },
            
            'angle_error_stats': {
                'max_abs': np.max(np.abs(angle_errors)) if angle_errors else 0,
                'rms': np.sqrt(np.mean(np.array(angle_errors)**2)) if angle_errors else 0,
                'mean': np.mean(angle_errors) if angle_errors else 0,
                'std': np.std(angle_errors) if angle_errors else 0
            },
            
            'p_flow_error_stats': {
                'max_abs': np.max(np.abs(p_flow_errors)) if p_flow_errors else 0,
                'rms': np.sqrt(np.mean(np.array(p_flow_errors)**2)) if p_flow_errors else 0,
                'mean': np.mean(p_flow_errors) if p_flow_errors else 0,
                'std': np.std(p_flow_errors) if p_flow_errors else 0
            }
        }
    
    def _analyze_contingency_impacts(self, all_results: Dict[str, ComparisonResults]):
        """Analyze the impact of different contingencies."""
        if "base_case" not in all_results:
            return
        
        base_case = all_results["base_case"]
        contingency_analysis = {
            'base_case_summary': base_case.summary(),
            'contingency_impacts': {}
        }
        
        for name, result in all_results.items():
            if name == "base_case":
                continue
            
            # Compare with base case
            impact = {
                'voltage_error_increase': (result.max_voltage_error_pu - 
                                         base_case.max_voltage_error_pu),
                'angle_error_increase': (result.max_angle_error_deg - 
                                       base_case.max_angle_error_deg),
                'converged': result.solver_converged,
                'iteration_change': result.solver_iterations - base_case.solver_iterations
            }
            
            contingency_analysis['contingency_impacts'][name] = impact
        
        # Save contingency analysis
        with open(self.output_dir / "contingency_impact_analysis.json", 'w') as f:
            json.dump(contingency_analysis, f, indent=2, default=str)
    
    def _save_batch_results(self):
        """Save batch results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary statistics
        if self.summary_stats:
            summary_file = self.output_dir / f"batch_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(self.summary_stats, f, indent=2, default=str)
        
        # Save detailed results for each scenario
        for scenario_name, result_data in self.results.items():
            scenario_file = self.output_dir / f"comparison_{scenario_name}_{timestamp}.json"
            
            # Prepare data for JSON serialization
            save_data = {
                'scenario_file': result_data['scenario_file'],
                'timestamp': result_data['timestamp']
            }
            
            if 'comparison' in result_data:
                save_data['comparison_summary'] = result_data['comparison'].summary()
            
            if 'error' in result_data:
                save_data['error'] = result_data['error']
            
            with open(scenario_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
    
    def get_worst_scenarios(self, metric: str = 'max_voltage_error_pu', n: int = 5) -> List[str]:
        """
        Get the scenarios with worst performance for a given metric.
        
        Args:
            metric: Metric to sort by
            n: Number of scenarios to return
            
        Returns:
            List of scenario names sorted by worst performance
        """
        scenario_metrics = []
        
        for name, result_data in self.results.items():
            if 'comparison' in result_data:
                comp = result_data['comparison']
                value = getattr(comp, metric, 0)
                scenario_metrics.append((name, value))
        
        # Sort by metric value (descending for errors)
        scenario_metrics.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in scenario_metrics[:n]]
    
    def export_csv_summary(self, output_file: Optional[Union[str, Path]] = None):
        """
        Export summary of all comparisons to CSV format.
        
        Args:
            output_file: Output CSV file path (optional)
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"comparison_summary_{timestamp}.csv"
        
        # Prepare CSV data
        csv_data = []
        headers = ['scenario_name', 'converged', 'iterations', 'max_voltage_error_pu',
                  'max_angle_error_deg', 'rms_voltage_error_pu', 'rms_angle_error_deg',
                  'max_p_flow_error_percent', 'max_q_flow_error_percent']
        
        for name, result_data in self.results.items():
            if 'comparison' not in result_data:
                continue
            
            comp = result_data['comparison']
            row = [
                name,
                comp.solver_converged,
                comp.solver_iterations,
                comp.max_voltage_error_pu,
                comp.max_angle_error_deg,
                comp.rms_voltage_error_pu,
                comp.rms_angle_error_deg,
                comp.max_p_flow_error_percent,
                comp.max_q_flow_error_percent
            ]
            csv_data.append(row)
        
        # Write CSV
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)
        
        print(f"Summary exported to: {output_file}")