"""
Working PowerFactory Comparison Demo

This creates actual comparison plots using your real PowerFactory data
with the correct DIgSILENT H5 file structure.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from visualization.powerfactory_comparison import PowerFactoryComparator


def create_working_powerfactory_comparison():
    """Create comparison plots with actual PowerFactory data."""
    print("🎨 Working PowerFactory Comparison Demo")
    print("=" * 50)
    print("Using your actual DIgSILENT PowerFactory H5 files!")
    print()
    
    # Create sample solver results (representing what your solver would produce)
    # These should eventually come from your actual solver
    n_buses = 39
    n_gens = 10
    
    # Realistic solver results for IEEE 39-bus system
    solver_results = {
        'voltages': {
            'magnitude': 1.0 + 0.02 * np.random.randn(n_buses),  # Around 1.0 pu ± 2%
            'angle': 0.05 * np.random.randn(n_buses)  # Small angles in radians
        },
        'line_flows': {
            'active_power': 10 + 20 * np.random.randn(n_buses),  # Bus injections as proxy
            'reactive_power': 5 * np.random.randn(n_buses)  
        },
        'generators': {
            'active_power': 50 + 100 * np.random.rand(n_gens),  # 50-150 MW range
            'reactive_power': 20 * np.random.randn(n_gens)  # ±20 MVAR
        },
        'converged': False,  # Matches your current solver status
        'iterations': 1
    }
    
    # Test with multiple scenarios
    scenarios_to_test = [
        {'id': 1, 'desc': 'Line outage: Line 16 - 19'},
        {'id': 2, 'desc': 'Line outage: Line 05 - 06'}, 
        {'id': 3, 'desc': 'Line outage: Line 03 - 18'}
    ]
    
    comparator = PowerFactoryComparator(figure_size=(16, 12))
    successful_comparisons = []
    
    for scenario in scenarios_to_test:
        scenario_id = scenario['id']
        pf_h5_path = f"Contingency Analysis/contingency_scenarios/scenario_{scenario_id}.h5"
        
        if not Path(pf_h5_path).exists():
            print(f"❌ PowerFactory file not found: {pf_h5_path}")
            continue
        
        print(f"\\n📊 Testing Scenario {scenario_id}: {scenario['desc']}")
        
        try:
            # Test loading PowerFactory results
            pf_results = comparator._load_powerfactory_results(pf_h5_path)
            
            if pf_results:
                print("   ✅ PowerFactory data loaded successfully!")
                
                # Create scenario info
                scenario_info = {
                    'scenario_id': scenario_id,
                    'description': scenario['desc'],
                    'solver_status': 'Did not converge (test data)'
                }
                
                # Create comparison plots
                figures = comparator.create_comprehensive_comparison(
                    solver_results=solver_results,
                    powerfactory_h5_path=pf_h5_path,
                    scenario_info=scenario_info,
                    save_plots=True
                )
                
                if figures:
                    successful_comparisons.append(scenario_id)
                    print(f"   ✅ Comparison plots created for Scenario {scenario_id}")
                    
                    # Show what was created
                    for plot_type in figures.keys():
                        print(f"      • {plot_type.replace('_', ' ').title()} Comparison")
                else:
                    print(f"   ❌ Failed to create plots for Scenario {scenario_id}")
            else:
                print("   ❌ Could not load PowerFactory data")
                
        except Exception as e:
            print(f"   ❌ Error processing Scenario {scenario_id}: {e}")
    
    # Summary
    print(f"\\n🎉 Working PowerFactory Comparison Complete!")
    print(f"✅ Successful comparisons: {len(successful_comparisons)} out of {len(scenarios_to_test)}")
    print(f"📁 All plots saved to: {comparator.output_dir}")
    
    if successful_comparisons:
        print(f"\\n📊 Successfully compared scenarios: {successful_comparisons}")
        print("\\n🎯 What these plots show:")
        print("   • Green lines/points = Your solver results")
        print("   • Red squares/lines = PowerFactory reference results")  
        print("   • Error bars = Differences between the two")
        print("   • Statistics boxes = Max and mean error values")
        
        print("\\n🔧 Next steps:")
        print("   1. Fix your solver convergence issues")
        print("   2. Replace the sample solver_results with actual solver output")
        print("   3. Run comparisons on converged cases") 
        print("   4. Use error metrics to validate your solver accuracy")
    
    return successful_comparisons


if __name__ == "__main__":
    create_working_powerfactory_comparison()