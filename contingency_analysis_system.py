"""
Contingency Analysis System for PIGNN Project

This module implements contingency analysis by:
1. Reading contingency definitions from CSV
2. Applying contingencies to the base case network
3. Running load flow analysis
4. Comparing results with DIgSILENT reference data (H5 files)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder
from physics.load_flow_solver import ThreePhaseLoadFlowSolver
from comparison.error_metrics import ErrorCalculator, ComparisonResults
from visualization.powerfactory_comparison import PowerFactoryComparator


@dataclass
class ContingencyScenario:
    """Represents a single contingency scenario"""
    scenario_id: int
    contingency_type: str  # N-1, N-2, etc.
    outage_type: str       # line, generator, transformer
    description: str
    severity: str
    element1_name: str     # e.g., "Line 16 - 19"
    element1_type: str     # line, generator, etc.
    element1_class: str    # ElmLne, ElmGen, etc.
    notes: str


class ContingencyAnalyzer:
    """
    Main contingency analysis engine that processes contingency scenarios
    and compares results with DIgSILENT reference data.
    """
    
    def __init__(self, 
                 base_scenario_file: Union[str, Path],
                 contingency_csv_file: Union[str, Path],
                 digsilent_scenarios_dir: Union[str, Path],
                 output_dir: Union[str, Path] = "Contingency Analysis/analysis_results"):
        """
        Initialize contingency analyzer.
        
        Args:
            base_scenario_file: Path to base case H5 file (scenario_0.h5)
            contingency_csv_file: Path to contingency definitions CSV
            digsilent_scenarios_dir: Directory with DIgSILENT H5 results
            output_dir: Directory to store analysis results
        """
        self.base_scenario_file = Path(base_scenario_file)
        self.contingency_csv_file = Path(contingency_csv_file)
        self.digsilent_scenarios_dir = Path(digsilent_scenarios_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base case
        print("Loading base case...")
        self.base_loader = H5DataLoader(str(self.base_scenario_file))
        self.base_data = self.base_loader.load_all_data()
        self.graph_builder = GraphBuilder()
        
        # Load contingency scenarios
        print("Loading contingency scenarios...")
        self.contingency_scenarios = self._load_contingency_scenarios()
        
        # Initialize comparison utilities
        self.error_calculator = ErrorCalculator()
        self.pf_comparator = PowerFactoryComparator()
        
        # Results storage
        self.results = {}
        
    def _load_contingency_scenarios(self) -> List[ContingencyScenario]:
        """Load contingency scenarios from CSV file."""
        df = pd.read_csv(self.contingency_csv_file)
        
        scenarios = []
        for _, row in df.iterrows():
            scenario = ContingencyScenario(
                scenario_id=row['scenario_id'],
                contingency_type=row['contingency_type'],
                outage_type=row['outage_type'],
                description=row['description'],
                severity=row['severity'],
                element1_name=row['element1_name'],
                element1_type=row['element1_type'],
                element1_class=row['element1_class'],
                notes=row['notes']
            )
            scenarios.append(scenario)
        
        print(f"Loaded {len(scenarios)} contingency scenarios")
        return scenarios
    
    def apply_contingency(self, base_data: Dict[str, Any], scenario: ContingencyScenario) -> Dict[str, Any]:
        """
        Apply a contingency to the base case data.
        
        Args:
            base_data: Base case network data
            scenario: Contingency scenario to apply
            
        Returns:
            Modified network data with contingency applied
        """
        # Create a copy of the base data
        import copy
        contingency_data = copy.deepcopy(base_data)
        
        if scenario.contingency_type == "BASE":
            # No modifications needed for base case
            return contingency_data
        
        elif scenario.contingency_type == "N-1" and scenario.outage_type == "line":
            # Parse line name (e.g., "Line 16 - 19" -> from_bus=16, to_bus=19)
            from_bus, to_bus = self._parse_line_name(scenario.element1_name)
            
            # Remove the line from the network data
            contingency_data = self._remove_line(contingency_data, from_bus, to_bus)
            
        elif scenario.contingency_type == "N-1" and scenario.outage_type == "generator":
            # Remove generator
            gen_name = scenario.element1_name
            contingency_data = self._remove_generator(contingency_data, gen_name)
            
        else:
            raise NotImplementedError(f"Contingency type {scenario.contingency_type} "
                                    f"with outage type {scenario.outage_type} not implemented")
        
        return contingency_data
    
    def _parse_line_name(self, line_name: str) -> Tuple[str, str]:
        """
        Parse line name to extract from and to bus numbers.
        
        Examples:
        - "Line 16 - 19" -> ("16", "19")
        - "Line 05 - 06" -> ("05", "06")
        """
        # Remove "Line " prefix and split by " - "
        line_part = line_name.replace("Line ", "")
        parts = line_part.split(" - ")
        
        if len(parts) != 2:
            raise ValueError(f"Cannot parse line name: {line_name}")
        
        from_bus = parts[0].strip()
        to_bus = parts[1].strip()
        
        return from_bus, to_bus
    
    def _remove_line(self, data: Dict[str, Any], from_bus: str, to_bus: str) -> Dict[str, Any]:
        """
        Remove a transmission line from the network data.
        
        This modifies the 'lines' section of the H5 data to mark the line as out of service.
        """
        if 'lines' not in data:
            print("Warning: No lines data found in network")
            return data
        
        lines_data = data['lines']
        
        # Look for the line in different possible formats
        line_found = False
        
        # Check if we have bus names or IDs to match
        if 'from_buses' in lines_data and 'to_buses' in lines_data:
            from_buses = lines_data['from_buses']
            to_buses = lines_data['to_buses']
            
            for i, (f_bus, t_bus) in enumerate(zip(from_buses, to_buses)):
                # Extract bus numbers from the bus names (e.g., "Bus 01" -> "01")
                if isinstance(f_bus, str) and f_bus.startswith('Bus '):
                    f_bus_num = f_bus.replace('Bus ', '')
                else:
                    f_bus_num = str(f_bus).zfill(2)
                
                if isinstance(t_bus, str) and t_bus.startswith('Bus '):
                    t_bus_num = t_bus.replace('Bus ', '')
                else:
                    t_bus_num = str(t_bus).zfill(2)
                
                # Check both directions (contingency uses numbers like "16", "19")
                if ((f_bus_num == from_bus.zfill(2) and t_bus_num == to_bus.zfill(2)) or
                    (f_bus_num == to_bus.zfill(2) and t_bus_num == from_bus.zfill(2))):
                    
                    # Mark line as out of service (your data uses 'in_service')
                    if 'in_service' in lines_data:
                        lines_data['in_service'][i] = False
                    elif 'status' in lines_data:
                        lines_data['status'][i] = 0  # 0 = out of service
                    else:
                        # Add in_service array if it doesn't exist
                        if 'in_service' not in lines_data:
                            lines_data['in_service'] = np.ones(len(from_buses), dtype=bool)
                        lines_data['in_service'][i] = False
                    
                    line_found = True
                    print(f"Removed line {from_bus} - {to_bus} (index {i})")
                    break
        
        if not line_found:
            print(f"Warning: Line {from_bus} - {to_bus} not found in network data")
        
        return data
    
    def _remove_generator(self, data: Dict[str, Any], gen_name: str) -> Dict[str, Any]:
        """Remove a generator from the network data."""
        # This would be implemented similar to _remove_line
        # but for generators. Implementation depends on your H5 data structure.
        print(f"Warning: Generator removal not implemented yet: {gen_name}")
        return data
    
    def run_scenario_analysis(self, scenario_id: int) -> ComparisonResults:
        """
        Run analysis for a single contingency scenario.
        
        Args:
            scenario_id: ID of the scenario to analyze
            
        Returns:
            ComparisonResults object with error metrics
        """
        # Find the scenario
        scenario = None
        for s in self.contingency_scenarios:
            if s.scenario_id == scenario_id:
                scenario = s
                break
        
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        print(f"\n--- Analyzing Scenario {scenario_id} ---")
        print(f"Description: {scenario.description}")
        
        # Apply contingency to base case
        print("Applying contingency...")
        contingency_data = self.apply_contingency(self.base_data, scenario)
        
        # Build graph with contingency applied
        print("Building network graph...")
        contingency_graph = self.graph_builder.build_from_h5_data(contingency_data)
        
        # Run load flow solver
        print("Running load flow solver...")
        solver = ThreePhaseLoadFlowSolver(contingency_graph)
        solver_results = solver.solve()
        
        print(f"Solver converged: {solver_results.converged}")
        print(f"Iterations: {getattr(solver_results, 'iterations', solver_results.iterations if hasattr(solver_results, 'iterations') else 'N/A')}")
        
        # Load DIgSILENT reference results
        digsilent_file = self.digsilent_scenarios_dir / f"scenario_{scenario_id}.h5"
        if not digsilent_file.exists():
            print(f"Warning: DIgSILENT reference file not found: {digsilent_file}")
            return self._create_empty_comparison_result(solver_results)
        
        print("Loading DIgSILENT reference results...")
        digsilent_loader = H5DataLoader(str(digsilent_file))
        digsilent_data = digsilent_loader.load_all_data()
        
        # Perform comparison
        print("Computing error metrics...")
        comparison = self.error_calculator.calculate_comprehensive_comparison(
            solver_results, digsilent_data)
        
        # Store results
        self.results[scenario_id] = {
            'scenario': scenario,
            'solver_results': solver_results,
            'digsilent_data': digsilent_data,
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Max voltage error: {comparison.max_voltage_error_pu:.6f} pu")
        print(f"Max angle error: {comparison.max_angle_error_deg:.4f} degrees")
        
        return comparison
    
    def run_batch_analysis(self, scenario_ids: Optional[List[int]] = None, 
                          max_scenarios: Optional[int] = None) -> Dict[int, ComparisonResults]:
        """
        Run analysis for multiple scenarios.
        
        Args:
            scenario_ids: Specific scenario IDs to analyze (None for all)
            max_scenarios: Maximum number of scenarios to process
            
        Returns:
            Dictionary mapping scenario IDs to ComparisonResults
        """
        if scenario_ids is None:
            scenario_ids = [s.scenario_id for s in self.contingency_scenarios]
        
        if max_scenarios:
            scenario_ids = scenario_ids[:max_scenarios]
        
        print(f"\n=== Running Batch Analysis for {len(scenario_ids)} scenarios ===")
        
        batch_results = {}
        
        for i, scenario_id in enumerate(scenario_ids):
            print(f"\n[{i+1}/{len(scenario_ids)}] Processing Scenario {scenario_id}")
            
            try:
                comparison = self.run_scenario_analysis(scenario_id)
                batch_results[scenario_id] = comparison
                
            except Exception as e:
                print(f"Error processing scenario {scenario_id}: {e}")
                continue
        
        # Generate summary report
        self._generate_batch_summary(batch_results)
        
        return batch_results
    
    def _create_empty_comparison_result(self, solver_results) -> ComparisonResults:
        """Create empty comparison result when DIgSILENT data is not available."""
        return ComparisonResults(
            solver_converged=solver_results.converged,
            solver_iterations=getattr(solver_results, 'iterations', 0)
        )
    
    def _generate_batch_summary(self, batch_results: Dict[int, ComparisonResults]):
        """Generate summary report for batch analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare summary data
        summary_data = []
        for scenario_id, comparison in batch_results.items():
            scenario = next(s for s in self.contingency_scenarios if s.scenario_id == scenario_id)
            
            summary_data.append({
                'scenario_id': scenario_id,
                'description': scenario.description,
                'contingency_type': scenario.contingency_type,
                'outage_type': scenario.outage_type,
                'element_name': scenario.element1_name,
                'converged': comparison.solver_converged,
                'iterations': comparison.solver_iterations,
                'max_voltage_error_pu': comparison.max_voltage_error_pu or 0.0,
                'max_angle_error_deg': comparison.max_angle_error_deg or 0.0,
                'rms_voltage_error_pu': comparison.rms_voltage_error_pu or 0.0,
                'rms_angle_error_deg': comparison.rms_angle_error_deg or 0.0
            })
        
        # Save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"contingency_analysis_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nBatch analysis completed!")
        print(f"Summary saved to: {summary_file}")
        print(f"Processed {len(batch_results)} scenarios successfully")
        
        # Print quick statistics
        converged_count = sum(1 for r in batch_results.values() if r.solver_converged)
        print(f"Convergence rate: {converged_count}/{len(batch_results)} ({converged_count/len(batch_results)*100:.1f}%)")
        
        if any(r.max_voltage_error_pu for r in batch_results.values()):
            max_errors = [r.max_voltage_error_pu for r in batch_results.values() if r.max_voltage_error_pu]
            print(f"Voltage error range: {min(max_errors):.6f} to {max(max_errors):.6f} pu")

    def create_powerfactory_comparison_plots(self, scenario_id: int, show_plots: bool = True) -> Dict[str, Any]:
        """Create detailed PowerFactory comparison visualization plots.
        
        Args:
            scenario_id: ID of the scenario to analyze and visualize
            show_plots: Whether to display plots interactively
            
        Returns:
            Dictionary containing the matplotlib figures and analysis results
        """
        print(f"\n🎨 Creating PowerFactory Comparison Plots for Scenario {scenario_id}")
        print("=" * 60)
        
        # Run the scenario analysis to get solver results
        scenario = self.contingency_scenarios[scenario_id]
        
        try:
            # Apply contingency and run solver
            print(f"🔧 Analyzing scenario: {scenario.description}")
            modified_data = self.apply_contingency(self.base_data, scenario)
            
            # Build graph and run load flow
            graph = self.graph_builder.build_from_h5_data(modified_data)
            solver = ThreePhaseLoadFlowSolver(graph)
            solver_results = solver.solve()
            
            # Get PowerFactory H5 file path for this scenario
            pf_h5_path = self.digsilent_scenarios_dir / f"scenario_{scenario_id}.h5"
            
            if not pf_h5_path.exists():
                print(f"   ⚠️  PowerFactory file not found: {pf_h5_path}")
                return {}
            
            # Prepare scenario info
            scenario_info = {
                'scenario_id': scenario_id,
                'description': scenario.description,
                'contingency_type': scenario.contingency_type,
                'outage_type': scenario.outage_type
            }
            
            # Create comprehensive comparison plots
            figures = self.pf_comparator.create_comprehensive_comparison(
                solver_results=solver_results,
                powerfactory_h5_path=str(pf_h5_path),
                scenario_info=scenario_info,
                save_plots=True
            )
            
            # Show plots if requested
            if show_plots and figures:
                import matplotlib.pyplot as plt
                for plot_name, fig in figures.items():
                    plt.figure(fig.number)
                    plt.show()
                
                # Keep plots open
                print("\n📊 Comparison plots created and displayed!")
                print("   Close the plot windows to continue...")
                
            return {
                'figures': figures,
                'solver_results': solver_results,
                'scenario_info': scenario_info,
                'pf_file_path': str(pf_h5_path)
            }
            
        except Exception as e:
            print(f"   ❌ Error creating comparison plots: {e}")
            return {}

    def run_multiple_scenario_comparisons(self, scenario_ids: List[int], 
                                        max_scenarios: int = 5,
                                        show_plots: bool = True) -> Dict[int, Dict[str, Any]]:
        """Create PowerFactory comparison plots for multiple scenarios.
        
        Args:
            scenario_ids: List of scenario IDs to compare
            max_scenarios: Maximum number of scenarios to process (to avoid overwhelming)
            show_plots: Whether to display plots interactively
            
        Returns:
            Dictionary mapping scenario IDs to their comparison results
        """
        print(f"\n🎨 Creating Multiple PowerFactory Comparisons")
        print(f"   Scenarios to process: {scenario_ids[:max_scenarios]}")
        print("=" * 60)
        
        comparison_results = {}
        
        for i, scenario_id in enumerate(scenario_ids[:max_scenarios]):
            print(f"\n[{i+1}/{min(len(scenario_ids), max_scenarios)}] Processing Scenario {scenario_id}")
            
            try:
                result = self.create_powerfactory_comparison_plots(
                    scenario_id=scenario_id,
                    show_plots=show_plots
                )
                
                if result:
                    comparison_results[scenario_id] = result
                    print(f"   ✅ Scenario {scenario_id} comparison completed")
                else:
                    print(f"   ❌ Scenario {scenario_id} comparison failed")
                    
            except Exception as e:
                print(f"   ❌ Error in scenario {scenario_id}: {e}")
                continue
        
        # Generate summary report
        if comparison_results:
            report_path = self.pf_comparator.output_dir / f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            summary_data = [
                {
                    'scenario_info': result['scenario_info'],
                    'errors': {}  # Error metrics would be calculated here
                }
                for result in comparison_results.values()
            ]
            
            summary_report = self.pf_comparator.create_summary_report(
                comparison_results=summary_data,
                save_path=str(report_path)
            )
            
            print(f"\n📄 Summary report created: {report_path}")
        
        return comparison_results


def main():
    """Example usage of the contingency analyzer."""
    # Define paths (adjust these to your actual file locations)
    base_scenario = "data/scenario_0.h5"  # Your base case
    contingency_csv = "Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv"
    digsilent_dir = "Contingency Analysis/contingency_scenarios"
    
    # Check if files exist
    if not Path(contingency_csv).exists():
        print(f"Contingency CSV not found: {contingency_csv}")
        return
    
    if not Path(base_scenario).exists():
        print(f"Base scenario not found: {base_scenario}")
        return
    
    # Initialize analyzer
    analyzer = ContingencyAnalyzer(
        base_scenario_file=base_scenario,
        contingency_csv_file=contingency_csv,
        digsilent_scenarios_dir=digsilent_dir
    )
    
    # Example 1: Analyze a specific scenario
    print("\n=== Single Scenario Analysis ===")
    try:
        result = analyzer.run_scenario_analysis(scenario_id=1)
        print(f"Analysis completed for scenario 1")
    except Exception as e:
        print(f"Error in single scenario analysis: {e}")
    
    # Example 2: Batch analysis of first 5 scenarios
    print("\n=== Batch Analysis ===")
    try:
        batch_results = analyzer.run_batch_analysis(
            scenario_ids=[1, 2, 3, 4, 5]  # First 5 scenarios
        )
        print(f"Batch analysis completed for {len(batch_results)} scenarios")
    except Exception as e:
        print(f"Error in batch analysis: {e}")


if __name__ == "__main__":
    main()