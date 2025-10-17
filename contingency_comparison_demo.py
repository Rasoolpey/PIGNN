"""
Demonstration script for DIgSILENT comparison utilities.

This script shows how to use the comparison package to:
1. Parse DIgSILENT exported data
2. Run load flow solver on scenarios
3. Compare results and compute error metrics
4. Generate reports

Usage:
    python contingency_comparison_demo.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from comparison import DIgSILENTParser, ErrorCalculator, BatchComparator, ComparisonReportGenerator
from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder
from physics.load_flow_solver import ThreePhaseLoadFlowSolver


def demo_single_scenario_comparison():
    """Demonstrate comparison for a single scenario."""
    print("=== Single Scenario Comparison Demo ===")
    
    # Paths (adjust these to your actual data locations)
    scenario_file = "data/scenario_0.h5"
    
    # Check if scenario file exists
    if not Path(scenario_file).exists():
        print(f"Warning: Scenario file {scenario_file} not found.")
        print("Please ensure you have scenario data available.")
        return
    
    print(f"Processing scenario: {scenario_file}")
    
    try:
        # Load and solve scenario
        loader = H5DataLoader()
        graph_builder = GraphBuilder()
        solver = ThreePhaseLoadFlowSolver()
        
        print("Loading scenario data...")
        raw_data = loader.load_scenario(scenario_file)
        
        print("Building graphs...")
        graphs = graph_builder.build_three_phase_graphs(raw_data)
        
        print("Running load flow solver...")
        solver_results = solver.solve_three_phase(graphs)
        
        print(f"Solver converged: {solver_results.converged}")
        print(f"Solver iterations: {getattr(solver_results, 'iterations', 'N/A')}")
        
        # For demo purposes, create mock DIgSILENT data
        # In practice, this would be loaded from actual DIgSILENT exports
        print("Creating mock DIgSILENT reference data...")
        mock_digsilent_data = create_mock_digsilent_data(solver_results)
        
        # Perform comparison
        print("Computing error metrics...")
        error_calc = ErrorCalculator()
        comparison = error_calc.calculate_comprehensive_comparison(
            solver_results, mock_digsilent_data)
        
        # Print results
        print("\n=== Comparison Results ===")
        print(f"Max voltage error: {comparison.max_voltage_error_pu:.6f} pu")
        print(f"Max angle error: {comparison.max_angle_error_deg:.4f} degrees")
        print(f"RMS voltage error: {comparison.rms_voltage_error_pu:.6f} pu")
        
        # Generate report
        print("Generating report...")
        report_gen = ComparisonReportGenerator()
        report_file = report_gen.generate_scenario_report("demo_scenario", comparison)
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error in single scenario comparison: {e}")
        import traceback
        traceback.print_exc()


def demo_digsilent_parser():
    """Demonstrate DIgSILENT parser functionality."""
    print("\\n=== DIgSILENT Parser Demo ===")
    
    # Create sample DIgSILENT CSV data for demonstration
    sample_dir = Path("sample_digsilent_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample bus voltage CSV
    bus_voltage_file = sample_dir / "bus_voltages.csv"
    with open(bus_voltage_file, 'w') as f:
        f.write("Bus Name,V_mag (p.u.),V_angle (deg),V_mag (kV)\\n")
        f.write("Bus1,1.0000,0.0000,11.0\\n")
        f.write("Bus2,0.9950,-2.3456,10.945\\n")
        f.write("Bus3,0.9875,-5.6789,10.8625\\n")
    
    # Create sample branch flow CSV
    branch_flow_file = sample_dir / "branch_flows.csv"
    with open(branch_flow_file, 'w') as f:
        f.write("From Bus,To Bus,P_flow (MW),Q_flow (MVAr),P_loss (MW),Q_loss (MVAr)\\n")
        f.write("Bus1,Bus2,50.25,15.30,0.125,0.080\\n")
        f.write("Bus2,Bus3,25.10,8.75,0.095,0.060\\n")
    
    try:
        # Parse the sample data
        parser = DIgSILENTParser(use_pandas=False)
        
        print("Parsing bus voltage data...")
        voltage_data = parser.parse_bus_voltages(bus_voltage_file)
        print(f"Found {len(voltage_data['bus_names'])} buses")
        print(f"Bus names: {voltage_data['bus_names']}")
        print(f"Voltage magnitudes: {voltage_data['v_magnitude_pu']}")
        
        print("\\nParsing branch flow data...")
        flow_data = parser.parse_branch_flows(branch_flow_file)
        print(f"Found {len(flow_data['from_buses'])} branches")
        print(f"P flows: {flow_data['p_flow_mw']}")
        print(f"Q flows: {flow_data['q_flow_mvar']}")
        
    except Exception as e:
        print(f"Error in DIgSILENT parser demo: {e}")
    
    finally:
        # Clean up sample files
        import shutil
        if sample_dir.exists():
            shutil.rmtree(sample_dir)


def demo_batch_comparison():
    """Demonstrate batch comparison functionality."""
    print("\\n=== Batch Comparison Demo ===")
    
    # Check if contingency scenarios exist
    scenarios_dir = Path("Contingency Analysis/contingency_scenarios")
    if not scenarios_dir.exists() or not list(scenarios_dir.glob("scenario_*.h5")):
        print("Warning: No contingency scenarios found.")
        print("Skipping batch comparison demo.")
        return
    
    try:
        # Initialize batch comparator
        batch_comparator = BatchComparator(use_pandas=False)
        
        # For demo, we'll process just a few scenarios
        scenario_files = list(scenarios_dir.glob("scenario_*.h5"))[:3]
        print(f"Found {len(scenario_files)} scenarios to process")
        
        # Create mock DIgSILENT directory structure
        mock_digsilent_dir = Path("mock_digsilent_results")
        mock_digsilent_dir.mkdir(exist_ok=True)
        
        for scenario_file in scenario_files:
            scenario_name = scenario_file.stem
            scenario_ds_dir = mock_digsilent_dir / scenario_name
            scenario_ds_dir.mkdir(exist_ok=True)
            
            # Create minimal mock data for this scenario
            create_mock_digsilent_files(scenario_ds_dir)
        
        print("Running batch comparison...")
        comparisons = batch_comparator.run_batch_comparison(
            scenarios_dir, mock_digsilent_dir, max_scenarios=3)
        
        print(f"\\nCompleted comparison of {len(comparisons)} scenarios")
        
        # Generate batch report
        report_gen = ComparisonReportGenerator()
        report_file = report_gen.generate_batch_report(
            comparisons, batch_comparator.summary_stats)
        print(f"Batch report saved to: {report_file}")
        
        # Export CSV summary
        batch_comparator.export_csv_summary()
        print("CSV summary exported")
        
    except Exception as e:
        print(f"Error in batch comparison demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up mock data
        import shutil
        mock_dir = Path("mock_digsilent_results")
        if mock_dir.exists():
            shutil.rmtree(mock_dir)


def create_mock_digsilent_data(solver_results) -> dict:
    """Create mock DIgSILENT data based on solver results with small perturbations."""
    # Extract some sample data from solver results
    bus_names = []
    v_magnitudes = []
    v_angles = []
    
    if hasattr(solver_results, 'voltages') and solver_results.voltages:
        for bus_name, voltage in list(solver_results.voltages.items())[:10]:  # First 10 buses
            bus_names.append(bus_name)
            
            # Add small random perturbations to create realistic comparison
            v_mag_perturbation = np.random.normal(0, 0.001)  # Small voltage error
            v_angle_perturbation = np.random.normal(0, 0.1)  # Small angle error
            
            v_magnitudes.append(abs(voltage) + v_mag_perturbation)
            v_angles.append(np.degrees(np.angle(voltage)) + v_angle_perturbation)
    else:
        # Fallback: create some sample data
        bus_names = ["Bus1", "Bus2", "Bus3"]
        v_magnitudes = [1.000, 0.995, 0.988]
        v_angles = [0.0, -2.3, -5.7]
    
    return {
        'bus_names': bus_names,
        'v_magnitude_pu': np.array(v_magnitudes),
        'v_angle_deg': np.array(v_angles),
        'v_magnitude_kv': np.array(v_magnitudes) * 11.0  # Assume 11 kV base
    }


def create_mock_digsilent_files(output_dir: Path):
    """Create mock DIgSILENT CSV files for testing."""
    # Bus voltages file
    bus_file = output_dir / "bus_voltages.csv"
    with open(bus_file, 'w') as f:
        f.write("Bus Name,V_mag (p.u.),V_angle (deg),V_mag (kV)\\n")
        f.write("Bus1,1.0005,0.0123,11.0055\\n")
        f.write("Bus2,0.9948,-2.3567,10.9428\\n")
        f.write("Bus3,0.9871,-5.6912,10.8581\\n")
    
    # Branch flows file
    flow_file = output_dir / "branch_flows.csv"
    with open(flow_file, 'w') as f:
        f.write("From Bus,To Bus,P_flow (MW),Q_flow (MVAr),P_loss (MW),Q_loss (MVAr)\\n")
        f.write("Bus1,Bus2,50.12,15.28,0.123,0.078\\n")
        f.write("Bus2,Bus3,25.05,8.72,0.092,0.058\\n")


def main():
    """Main demonstration function."""
    print("DIgSILENT Comparison Utilities Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demo_digsilent_parser()
    demo_single_scenario_comparison()
    demo_batch_comparison()
    
    print("\\n" + "=" * 50)
    print("Demo completed!")
    print("\\nNext steps:")
    print("1. Prepare actual DIgSILENT exported data")
    print("2. Update file paths in the demo to point to your data")
    print("3. Run contingency analysis with real scenarios")
    print("4. Generate comparison reports for your study")


if __name__ == "__main__":
    main()