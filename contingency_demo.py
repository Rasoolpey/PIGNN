"""
Demo script for the proper contingency analysis system that works with your specific setup.

This demonstrates how to:
1. Read your contingency CSV file
2. Apply specific contingencies (like disconnecting Line 16-19) 
3. Compare with your DIgSILENT H5 results
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from contingency_analysis_system import ContingencyAnalyzer


def demo_contingency_analysis():
    """Demonstrate the contingency analysis system."""
    print("=== PIGNN Contingency Analysis Demo ===")
    print("This demo uses YOUR actual contingency data and setup.")
    
    # Your actual file paths - using scenario_1.h5 as base since data/scenario_0.h5 has access issues
    base_scenario = "Contingency Analysis/contingency_scenarios/scenario_1.h5"
    contingency_csv = "Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv"
    digsilent_dir = "Contingency Analysis/contingency_scenarios"
    
    # Check if files exist
    if not Path(contingency_csv).exists():
        print(f"❌ Contingency CSV not found: {contingency_csv}")
        print("Please ensure the file exists.")
        return
    
    if not Path(digsilent_dir).exists():
        print(f"❌ DIgSILENT scenarios directory not found: {digsilent_dir}")
        print("Please ensure the directory exists.")
        return
    
    try:
        # Initialize the analyzer with your data
        print("\\n📊 Initializing Contingency Analyzer...")
        analyzer = ContingencyAnalyzer(
            base_scenario_file=base_scenario,
            contingency_csv_file=contingency_csv,
            digsilent_scenarios_dir=digsilent_dir
        )
        
        # Show some information about loaded scenarios
        print(f"\\n📋 Loaded {len(analyzer.contingency_scenarios)} contingency scenarios")
        
        # Show first few scenarios
        print("\\nFirst 5 scenarios:")
        for scenario in analyzer.contingency_scenarios[:5]:
            print(f"  {scenario.scenario_id}: {scenario.description}")
        
        # Example 1: Analyze the specific scenario you mentioned (scenario 1)
        print("\\n🔬 Analyzing Scenario 1 (Line 16-19 outage)...")
        try:
            result_1 = analyzer.run_scenario_analysis(scenario_id=1)
            print("✅ Scenario 1 analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error analyzing scenario 1: {e}")
        
        # Example 2: Analyze a few more scenarios
        print("\\n🔬 Analyzing multiple scenarios...")
        try:
            # Test scenarios 1-3 (you can adjust this list)
            test_scenarios = [1, 2, 3]
            batch_results = analyzer.run_batch_analysis(scenario_ids=test_scenarios)
            
            print(f"✅ Batch analysis completed for {len(batch_results)} scenarios!")
            
            # Show summary
            print("\\n📈 Quick Summary:")
            for scenario_id, comparison in batch_results.items():
                scenario_desc = next(s.description for s in analyzer.contingency_scenarios 
                                   if s.scenario_id == scenario_id)
                converged = "✅" if comparison.solver_converged else "❌"
                print(f"  Scenario {scenario_id}: {converged} {scenario_desc[:50]}...")
                if comparison.max_voltage_error_pu:
                    print(f"    Max voltage error: {comparison.max_voltage_error_pu:.6f} pu")
            
        except Exception as e:
            print(f"❌ Error in batch analysis: {e}")
        
        print("\\n🎉 Demo completed!")
        print("\\nNext steps:")
        print("1. Review the generated analysis results")
        print("2. Check the CSV summary file in 'Contingency Analysis/analysis_results/'")
        print("3. Adjust the scenario list to analyze more contingencies")
        print("4. Use the comparison results to validate your solver against DIgSILENT")
        
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_contingency_analysis()