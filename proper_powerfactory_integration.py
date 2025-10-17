"""
Proper PowerFactory Integration for Your Solver

This script shows how to properly integrate the PowerFactory comparison 
with your actual solver, using the correct data structures and methods.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from contingency_analysis_system import ContingencyAnalyzer
from visualization.powerfactory_comparison import PowerFactoryComparator


def extract_solver_results_properly(solver_output, graph):
    """Extract solver results in the format expected by the comparison system.
    
    This function converts your LoadFlowResults object to match what 
    the PowerFactory comparison expects.
    """
    try:
        # Get bus information from graph
        bus_names = list(graph.nodes.keys())
        n_buses = len(bus_names)
        
        # Extract voltage results from LoadFlowResults object
        if hasattr(solver_output, 'voltage_magnitudes') and solver_output.voltage_magnitudes is not None:
            v_magnitudes = solver_output.voltage_magnitudes
        else:
            print("   ⚠️  No voltage magnitudes found, using defaults")
            v_magnitudes = np.ones(n_buses)
        
        if hasattr(solver_output, 'voltage_angles') and solver_output.voltage_angles is not None:
            v_angles = solver_output.voltage_angles
        else:
            print("   ⚠️  No voltage angles found, using defaults")
            v_angles = np.zeros(n_buses)
        
        # Extract power injection results (use as proxy for line flows)
        if hasattr(solver_output, 'active_power') and solver_output.active_power is not None:
            line_flows_active = solver_output.active_power
        else:
            print("   ⚠️  No active power found, using defaults")
            line_flows_active = np.zeros(n_buses)
        
        if hasattr(solver_output, 'reactive_power') and solver_output.reactive_power is not None:
            line_flows_reactive = solver_output.reactive_power
        else:
            print("   ⚠️  No reactive power found, using defaults")
            line_flows_reactive = np.zeros(n_buses)
        
        # For generators, we need to extract from the bus data (approximate)
        # This is a simplification - in practice you'd need generator-specific data
        gen_active = line_flows_active[:10] if len(line_flows_active) >= 10 else np.zeros(10)
        gen_reactive = line_flows_reactive[:10] if len(line_flows_reactive) >= 10 else np.zeros(10)
        
        # Check solver convergence
        converged = getattr(solver_output, 'converged', False)
        iterations = getattr(solver_output, 'iterations', 0)
        max_mismatch = getattr(solver_output, 'max_mismatch', float('nan'))
        
        # Ensure arrays have correct dimensions
        if len(v_magnitudes) != n_buses:
            print(f"   ⚠️  Voltage magnitude array size mismatch: {len(v_magnitudes)} vs {n_buses}")
            v_magnitudes = np.ones(n_buses)
        
        if len(v_angles) != n_buses:
            print(f"   ⚠️  Voltage angle array size mismatch: {len(v_angles)} vs {n_buses}")
            v_angles = np.zeros(n_buses)
        
        # Handle NaN values (common when solver doesn't converge)
        if np.any(np.isnan(v_magnitudes)):
            print("   ⚠️  NaN values found in voltage magnitudes, replacing with 1.0 pu")
            v_magnitudes = np.where(np.isnan(v_magnitudes), 1.0, v_magnitudes)
        
        if np.any(np.isnan(v_angles)):
            print("   ⚠️  NaN values found in voltage angles, replacing with 0.0 rad")
            v_angles = np.where(np.isnan(v_angles), 0.0, v_angles)
        
        # Format for comparison system
        formatted_results = {
            'voltages': {
                'magnitude': v_magnitudes,
                'angle': v_angles
            },
            'line_flows': {
                'active_power': line_flows_active,
                'reactive_power': line_flows_reactive
            },
            'generators': {
                'active_power': gen_active,
                'reactive_power': gen_reactive
            },
            'convergence': {
                'converged': converged,
                'iterations': iterations,
                'max_mismatch': max_mismatch
            }
        }
        
        print(f"   📊 Extracted: {len(v_magnitudes)} bus voltages, convergence: {converged}")
        if not converged:
            print(f"   📊 Max mismatch: {max_mismatch}, iterations: {iterations}")
        
        return formatted_results
    
    except Exception as e:
        print(f"   ❌ Error extracting solver results: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_proper_comparison_with_your_solver(scenario_id: int = 2):
    """Create PowerFactory comparison using your actual solver (when it works)."""
    print(f"\\n🔧 Proper PowerFactory Integration - Scenario {scenario_id}")
    print("=" * 60)
    
    # Initialize your contingency analysis system
    try:
        print("📊 Initializing contingency analyzer...")
        analyzer = ContingencyAnalyzer(
            base_scenario_file="Contingency Analysis/contingency_scenarios/scenario_1.h5",
            contingency_csv_file="Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv",
            digsilent_scenarios_dir="Contingency Analysis/contingency_scenarios"
        )
        
        scenario = analyzer.contingency_scenarios[scenario_id]
        print(f"📋 Analyzing: {scenario.description}")
        
        # Apply contingency to base case
        print("🔧 Applying contingency...")
        modified_data = analyzer.apply_contingency(analyzer.base_data, scenario)
        
        # Build network graph
        print("🌐 Building network graph...")
        graph = analyzer.graph_builder.build_from_h5_data(modified_data)
        
        # Run your solver
        print("⚡ Running load flow solver...")
        from physics.load_flow_solver import ThreePhaseLoadFlowSolver
        solver = ThreePhaseLoadFlowSolver(graph)
        
        try:
            raw_solver_output = solver.solve()
            solver_status = "Converged" if getattr(raw_solver_output, 'converged', False) else "Did not converge"
            print(f"   📊 Solver status: {solver_status}")
            
            # Extract results in proper format
            solver_results = extract_solver_results_properly(raw_solver_output, graph)
            
            if solver_results:
                # Create PowerFactory comparison
                print("📊 Creating PowerFactory comparison...")
                comparator = PowerFactoryComparator(figure_size=(16, 12))
                
                pf_h5_path = analyzer.digsilent_scenarios_dir / f"scenario_{scenario_id}.h5"
                
                scenario_info = {
                    'scenario_id': scenario_id,
                    'description': f"{scenario.description} (Your Solver)",
                    'solver_status': solver_status,
                    'contingency_type': scenario.contingency_type
                }
                
                # Create comparison plots
                figures = comparator.create_comprehensive_comparison(
                    solver_results=solver_results,
                    powerfactory_h5_path=str(pf_h5_path),
                    scenario_info=scenario_info,
                    save_plots=True
                )
                
                if figures:
                    print("   ✅ PowerFactory comparison completed!")
                    return figures, solver_results
                else:
                    print("   ❌ Failed to create comparison plots")
                    return None, solver_results
            else:
                print("   ❌ Could not extract solver results")
                return None, None
                
        except Exception as e:
            print(f"   ❌ Solver failed: {e}")
            
            # Create comparison anyway with dummy data to show the format
            print("   📊 Creating comparison with default values...")
            default_results = create_default_solver_results(graph)
            
            comparator = PowerFactoryComparator(figure_size=(16, 12))
            pf_h5_path = analyzer.digsilent_scenarios_dir / f"scenario_{scenario_id}.h5"
            
            scenario_info = {
                'scenario_id': scenario_id,
                'description': f"{scenario.description} (Solver Failed)",
                'solver_status': "Failed - using defaults",
                'contingency_type': scenario.contingency_type
            }
            
            figures = comparator.create_comprehensive_comparison(
                solver_results=default_results,
                powerfactory_h5_path=str(pf_h5_path),
                scenario_info=scenario_info,
                save_plots=True
            )
            
            return figures, default_results
    
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_default_solver_results(graph):
    """Create reasonable default results when solver fails."""
    bus_names = list(graph.nodes())
    n_buses = len(bus_names)
    
    # Use flat start conditions
    default_results = {
        'voltages': {
            'magnitude': np.ones(n_buses),  # 1.0 pu flat start
            'angle': np.zeros(n_buses)      # 0 degrees flat start
        },
        'line_flows': {
            'active_power': np.zeros(n_buses),
            'reactive_power': np.zeros(n_buses)
        },
        'generators': {
            'active_power': np.zeros(10),
            'reactive_power': np.zeros(10)
        },
        'convergence': {
            'converged': False,
            'iterations': 0
        }
    }
    
    print(f"   📊 Created default results for {n_buses} buses")
    return default_results


def test_multiple_scenarios_with_your_solver():
    """Test the integration with multiple scenarios."""
    print("\\n🔬 Testing Multiple Scenarios with Your Solver")
    print("=" * 50)
    
    # Test scenarios (start with ones likely to converge)
    test_scenarios = [0, 2, 3]  # Base case + some line outages
    
    results_summary = []
    
    for scenario_id in test_scenarios:
        try:
            figures, solver_results = create_proper_comparison_with_your_solver(scenario_id)
            
            if figures and solver_results:
                convergence = solver_results.get('convergence', {}).get('converged', False)
                results_summary.append({
                    'scenario_id': scenario_id,
                    'success': True,
                    'converged': convergence,
                    'plots_created': len(figures)
                })
                print(f"   ✅ Scenario {scenario_id}: Success")
            else:
                results_summary.append({
                    'scenario_id': scenario_id,
                    'success': False,
                    'converged': False,
                    'plots_created': 0
                })
                print(f"   ❌ Scenario {scenario_id}: Failed")
                
        except Exception as e:
            print(f"   ❌ Scenario {scenario_id}: Error - {e}")
            results_summary.append({
                'scenario_id': scenario_id,
                'success': False,
                'converged': False,
                'plots_created': 0
            })
    
    # Summary
    successful = sum(1 for r in results_summary if r['success'])
    converged = sum(1 for r in results_summary if r['converged'])
    
    print(f"\\n📊 Summary:")
    print(f"   Scenarios tested: {len(results_summary)}")
    print(f"   Successful comparisons: {successful}")
    print(f"   Solver convergences: {converged}")
    
    return results_summary


if __name__ == "__main__":
    print("🔧 Proper PowerFactory Integration for Your Solver")
    print("=" * 60)
    
    # Single scenario test
    figures, results = create_proper_comparison_with_your_solver(scenario_id=2)
    
    # Multiple scenario test
    summary = test_multiple_scenarios_with_your_solver()
    
    print("\\n🎯 Integration Guide:")
    print("1. Fix your solver convergence issues first")
    print("2. Modify extract_solver_results_properly() to match your solver's output format") 
    print("3. Run comparisons on converged cases")
    print("4. Use the error metrics to validate and improve your solver")
    print("\\n📁 Check comparison_plots/ for the latest results!")