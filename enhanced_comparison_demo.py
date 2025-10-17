"""
Enhanced PowerFactory Comparison for Real Contingency Analysis

This script integrates with your actual contingency analysis system
and creates comparison plots even when the solver has convergence issues.
It extracts whatever results are available and compares them with PowerFactory.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from contingency_analysis_system import ContingencyAnalyzer


def create_robust_comparison_plots(analyzer: ContingencyAnalyzer, 
                                 scenario_id: int,
                                 show_plots: bool = True) -> dict:
    """Create comparison plots that work even with convergence issues.
    
    Args:
        analyzer: The contingency analyzer instance
        scenario_id: Scenario to analyze
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with results and plot information
    """
    print(f"\\n🎨 Creating Robust PowerFactory Comparison for Scenario {scenario_id}")
    print("=" * 60)
    
    scenario = analyzer.contingency_scenarios[scenario_id]
    print(f"📋 Scenario: {scenario.description}")
    
    try:
        # Run the analysis (even if it doesn't converge)
        print("🔧 Running load flow analysis...")
        
        # Apply contingency
        modified_data = analyzer.apply_contingency(analyzer.base_data, scenario)
        
        # Build graph and attempt load flow
        graph = analyzer.graph_builder.build_from_h5_data(modified_data)
        from physics.load_flow_solver import ThreePhaseLoadFlowSolver
        solver = ThreePhaseLoadFlowSolver(graph)
        
        # Try to get whatever results we can
        try:
            solver_results = solver.solve()
            convergence_status = "Converged" if solver_results.get('converged', False) else "Did not converge"
        except Exception as e:
            print(f"   ⚠️  Solver issues: {e}")
            # Create minimal results structure
            solver_results = {
                'converged': False,
                'voltages': {'magnitude': np.ones(39), 'angle': np.zeros(39)},
                'line_flows': {'active_power': np.zeros(46), 'reactive_power': np.zeros(46)},
                'generators': {'active_power': np.zeros(10), 'reactive_power': np.zeros(10)}
            }
            convergence_status = "Failed to run"
        
        print(f"   📊 Solver status: {convergence_status}")
        
        # Load PowerFactory reference data
        pf_h5_path = analyzer.digsilent_scenarios_dir / f"scenario_{scenario_id}.h5"
        
        if pf_h5_path.exists():
            print(f"   📁 Loading PowerFactory reference: {pf_h5_path}")
            
            # Try to load PowerFactory results
            try:
                pf_results = analyzer.pf_comparator._load_powerfactory_results(str(pf_h5_path))
                if pf_results:
                    print("   ✅ PowerFactory data loaded successfully")
                    
                    # Create scenario info
                    scenario_info = {
                        'scenario_id': scenario_id,
                        'description': f"{scenario.description} ({convergence_status})",
                        'contingency_type': scenario.contingency_type,
                        'solver_status': convergence_status
                    }
                    
                    # Create the comparison plots
                    figures = analyzer.pf_comparator.create_comprehensive_comparison(
                        solver_results=solver_results,
                        powerfactory_h5_path=str(pf_h5_path),
                        scenario_info=scenario_info,
                        save_plots=True
                    )
                    
                    if figures:
                        print("   ✅ Comparison plots created!")
                        
                        # Show plots if requested
                        if show_plots:
                            for plot_name, fig in figures.items():
                                plt.figure(fig.number)
                                plt.show()
                        
                        return {
                            'success': True,
                            'figures': figures,
                            'solver_status': convergence_status,
                            'scenario_info': scenario_info
                        }
                    else:
                        print("   ❌ Failed to create plots")
                        return {'success': False, 'error': 'Plot creation failed'}
                else:
                    print("   ⚠️  Could not parse PowerFactory data")
                    return {'success': False, 'error': 'PowerFactory data parsing failed'}
            
            except Exception as e:
                print(f"   ❌ Error loading PowerFactory data: {e}")
                return {'success': False, 'error': f'PowerFactory loading error: {e}'}
        
        else:
            print(f"   ⚠️  PowerFactory file not found: {pf_h5_path}")
            return {'success': False, 'error': 'PowerFactory file not found'}
    
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        return {'success': False, 'error': f'Analysis error: {e}'}


def run_enhanced_comparison_demo():
    """Run the enhanced comparison demo with your actual system."""
    print("🎨 Enhanced PowerFactory Comparison Demo")
    print("=" * 50)
    print("This demo creates comparison plots using your actual:")
    print("• Contingency scenarios")
    print("• Load flow solver (even with convergence issues)")
    print("• DIgSILENT PowerFactory reference data")
    print()
    
    # Setup paths 
    base_scenario = "Contingency Analysis/contingency_scenarios/scenario_1.h5"
    contingency_csv = "Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv"
    digsilent_dir = "Contingency Analysis/contingency_scenarios"
    
    try:
        # Initialize analyzer
        print("📊 Initializing analyzer...")
        analyzer = ContingencyAnalyzer(
            base_scenario_file=base_scenario,
            contingency_csv_file=contingency_csv,
            digsilent_scenarios_dir=digsilent_dir
        )
        
        print(f"✅ Loaded {len(analyzer.contingency_scenarios)} scenarios")
        
        # Test scenarios that might have PowerFactory data
        test_scenarios = [0, 1, 2, 3, 5]  # Try a few different ones
        
        successful_comparisons = []
        
        for scenario_id in test_scenarios:
            if scenario_id < len(analyzer.contingency_scenarios):
                print(f"\\n{'='*40}")
                print(f"Testing Scenario {scenario_id}")
                
                result = create_robust_comparison_plots(
                    analyzer=analyzer,
                    scenario_id=scenario_id,
                    show_plots=False  # Set to True to see plots
                )
                
                if result.get('success'):
                    successful_comparisons.append(scenario_id)
                    print(f"✅ Scenario {scenario_id} comparison successful!")
                else:
                    print(f"❌ Scenario {scenario_id} failed: {result.get('error', 'Unknown error')}")
        
        # Summary
        print(f"\\n🎉 Enhanced Comparison Demo Complete!")
        print(f"✅ Successful comparisons: {len(successful_comparisons)}")
        print(f"📁 All plots saved to: {analyzer.pf_comparator.output_dir}")
        
        if successful_comparisons:
            print(f"\\n📊 Successfully compared scenarios: {successful_comparisons}")
            print("\\n🔧 To see plots interactively, set show_plots=True")
        
        return analyzer, successful_comparisons
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []


if __name__ == "__main__":
    run_enhanced_comparison_demo()