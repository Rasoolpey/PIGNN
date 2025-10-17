"""
PowerFactory Comparison Demo

This script demonstrates how to create detailed comparison visualizations
between your solver results and DIgSILENT PowerFactory results.

Features:
- Voltage magnitude and angle comparisons
- Line flow (active/reactive power) comparisons  
- Generation output comparisons
- Error analysis with statistical metrics
- Automatic plot saving and report generation
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from contingency_analysis_system import ContingencyAnalyzer


def demo_powerfactory_comparison():
    """Demonstrate PowerFactory comparison visualization system."""
    print("🎨 PowerFactory vs Solver Comparison Demo")
    print("=" * 50)
    print("This demo creates detailed comparison plots between:")
    print("• Your three-phase load flow solver results")  
    print("• DIgSILENT PowerFactory reference results")
    print()
    
    # Setup paths (same as your working contingency system)
    base_scenario = "Contingency Analysis/contingency_scenarios/scenario_1.h5"  # Using working file
    contingency_csv = "Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv"
    digsilent_dir = "Contingency Analysis/contingency_scenarios"
    
    # Check if files exist
    if not Path(contingency_csv).exists():
        print(f"❌ Contingency CSV not found: {contingency_csv}")
        return
    
    if not Path(base_scenario).exists():
        print(f"❌ Base scenario H5 not found: {base_scenario}")
        return
        
    if not Path(digsilent_dir).exists():
        print(f"❌ DIgSILENT results directory not found: {digsilent_dir}")
        return
    
    try:
        # Initialize the analyzer
        print("📊 Initializing Contingency Analyzer...")
        analyzer = ContingencyAnalyzer(
            base_scenario_file=base_scenario,
            contingency_csv_file=contingency_csv,
            digsilent_scenarios_dir=digsilent_dir
        )
        
        print(f"✅ Loaded {len(analyzer.contingency_scenarios)} contingency scenarios")
        
        # Show available scenarios
        print("\\n📋 Available scenarios:")
        for i, scenario in enumerate(analyzer.contingency_scenarios[:10]):  # Show first 10
            print(f"  {i}: {scenario.description}")
        if len(analyzer.contingency_scenarios) > 10:
            print(f"  ... and {len(analyzer.contingency_scenarios) - 10} more scenarios")
        
        # Create comparison plots for selected scenarios
        print("\\n🎨 Creating PowerFactory Comparison Plots...")
        
        # Test scenarios - pick ones that might converge well
        test_scenarios = [0, 2, 3, 5, 7]  # Base case + some line outages
        available_scenarios = [i for i in test_scenarios if i < len(analyzer.contingency_scenarios)]
        
        if not available_scenarios:
            print("❌ No test scenarios available")
            return
        
        print(f"Testing scenarios: {available_scenarios}")
        
        # Option 1: Single scenario detailed comparison
        print("\\n🔬 Option 1: Single Scenario Detailed Analysis")
        scenario_to_test = available_scenarios[1]  # Pick a line outage scenario
        print(f"Analyzing scenario {scenario_to_test}: {analyzer.contingency_scenarios[scenario_to_test].description}")
        
        single_result = analyzer.create_powerfactory_comparison_plots(
            scenario_id=scenario_to_test,
            show_plots=False  # Set to True to display plots interactively
        )
        
        if single_result and single_result.get('figures'):
            print("✅ Single scenario comparison plots created!")
            print(f"   📁 Plots saved to: {analyzer.pf_comparator.output_dir}")
            
            # Show what was created
            figures = single_result['figures']
            print("   📊 Generated plots:")
            for plot_type in figures.keys():
                print(f"      • {plot_type.replace('_', ' ').title()} Comparison")
        else:
            print("❌ Single scenario comparison failed")
        
        # Option 2: Multiple scenarios batch comparison  
        print("\\n🔬 Option 2: Multiple Scenarios Batch Analysis")
        print(f"Comparing {len(available_scenarios[:3])} scenarios...")
        
        batch_results = analyzer.run_multiple_scenario_comparisons(
            scenario_ids=available_scenarios[:3],  # Limit to 3 for demo
            max_scenarios=3,
            show_plots=False  # Set to True to display plots interactively
        )
        
        if batch_results:
            print(f"✅ Batch comparison completed for {len(batch_results)} scenarios!")
            print(f"   📁 All plots saved to: {analyzer.pf_comparator.output_dir}")
            
            # Summary of what was created
            total_plots = sum(len(result.get('figures', {})) for result in batch_results.values())
            print(f"   📊 Total plots generated: {total_plots}")
            
            print("\\n📋 Scenarios processed:")
            for scenario_id, result in batch_results.items():
                status = "✅" if result.get('figures') else "❌"
                desc = result.get('scenario_info', {}).get('description', 'Unknown')
                print(f"      {status} Scenario {scenario_id}: {desc}")
        else:
            print("❌ Batch comparison failed")
        
        # Summary and next steps
        print("\\n🎉 PowerFactory Comparison Demo Completed!")
        print("\\n📊 What was created:")
        print("   1. Individual voltage comparison plots (magnitude, angle, errors)")
        print("   2. Line flow comparison plots (active power, reactive power, errors)")  
        print("   3. Generation comparison plots (active power, reactive power, errors)")
        print("   4. Detailed error analysis with statistical metrics")
        print("   5. Summary reports with all comparison statistics")
        
        print(f"\\n📁 Find all results in: {analyzer.pf_comparator.output_dir}")
        
        print("\\n🔧 To customize:")
        print("   • Set show_plots=True to display plots interactively")
        print("   • Modify test_scenarios list to analyze different contingencies")
        print("   • Adjust figure_size in PowerFactoryComparator for different plot sizes")
        print("   • Edit color scheme in PowerFactoryComparator.colors")
        
        return analyzer, batch_results
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def interactive_comparison_demo():
    """Interactive version that shows plots on screen."""
    print("🎨 Interactive PowerFactory Comparison Demo")
    print("=" * 50)
    
    analyzer, results = demo_powerfactory_comparison()
    
    if analyzer and results:
        print("\\n🖥️  Starting Interactive Mode...")
        print("This will show comparison plots on screen.")
        
        # Pick one good scenario to show interactively
        if results:
            scenario_id = list(results.keys())[0]
            print(f"\\nDisplaying interactive plots for Scenario {scenario_id}...")
            
            interactive_result = analyzer.create_powerfactory_comparison_plots(
                scenario_id=scenario_id,
                show_plots=True  # This will display plots
            )
            
            print("\\n📊 Interactive plots displayed!")
            print("Close the plot windows when you're done viewing them.")


if __name__ == "__main__":
    # Run the demo
    print("Choose demo mode:")
    print("1. Standard demo (saves plots to files)")
    print("2. Interactive demo (shows plots on screen)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "2":
            interactive_comparison_demo()
        else:
            demo_powerfactory_comparison()
    except KeyboardInterrupt:
        print("\\n\\nDemo cancelled by user.")
    except Exception as e:
        print(f"\\nDemo error: {e}")