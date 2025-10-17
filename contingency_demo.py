#!/usr/bin/env python3
"""
Contingency Analysis Demo for PIGNN Project

This demo demonstrates comprehensive contingency analysis:
1. Loads contingency scenarios from CSV and H5 files
2. Runs load flow analysis for each contingency
3. Compares results with PowerFactory reference
4. Generates detailed comparison plots and reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent))

from physics.powerfactory_solver import create_powerfactory_based_results
from visualization.powerfactory_detailed_comparison import create_powerfactory_comparisons
import h5py


def run_contingency_demo():
    """Run comprehensive contingency analysis demonstration"""
    
    print("🚨 PIGNN Contingency Analysis Demo")
    print("=" * 50)
    
    # Load contingency scenarios list
    contingency_csv = "Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv"
    scenarios_dir = Path("Contingency Analysis/contingency_scenarios")
    output_dir = Path("Contingency Analysis/contingency_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(contingency_csv).exists():
        print(f"❌ Contingency CSV not found: {contingency_csv}")
        return
    
    # Load scenario list
    print("📋 Loading contingency scenarios...")
    try:
        df = pd.read_csv(contingency_csv)
        print(f"✅ Found {len(df)} contingency scenarios")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Select demonstration scenarios (including user's favorites)
    demo_scenarios = [0, 2, 5, 20, 77, 150, 196]  # Representative scenarios
    
    print(f"\n🎯 Running detailed PowerFactory comparisons for {len(demo_scenarios)} scenarios...")
    print("   This will generate 3 comparison plots per scenario:")
    print("   1. Busbar Voltages (PowerFactory vs Our Solver)")
    print("   2. Line Power Flows (PowerFactory vs Our Solver)")
    print("   3. Generator Power (PowerFactory vs Our Solver)")
    print()
    
    # Process each demo scenario
    for scenario_id in demo_scenarios:
        h5_file = scenarios_dir / f"scenario_{scenario_id}.h5"
        
        if h5_file.exists():
            print(f"🔍 Processing Scenario {scenario_id}...")
            try:
                # Generate the 3 essential comparison plots
                create_powerfactory_comparisons(scenario_id, str(h5_file), output_dir)
                print(f"✅ Completed comparison plots for Scenario {scenario_id}")
                
            except Exception as e:
                print(f"❌ Error processing Scenario {scenario_id}: {e}")
        else:
            print(f"⚠️  Scenario {scenario_id} H5 file not found: {h5_file}")
    
    print(f"\n📊 Comparison Analysis Complete!")
    print(f"📁 All plots saved to: {output_dir}")
    print("\n🎯 Summary of Generated Plots:")
    print("   • Voltage Comparisons: Side-by-side busbar voltage analysis")
    print("   • Line Flow Comparisons: Active/reactive power flow analysis")  
    print("   • Generator Comparisons: Generator power output analysis")
    print("\nEach plot shows PowerFactory results vs Our Solver results side-by-side!")
    
    return output_dir
    
    # Analyze representative scenarios
    scenarios_to_analyze = [0, 2, 5, 20, 77, 150, 196]  # Representative selection
    
    results_summary = []
    
    for scenario_id in scenarios_to_analyze:
        h5_path = scenarios_dir / f"scenario_{scenario_id}.h5"
        
        print(f"\n🔍 Analyzing Scenario {scenario_id}")
        print("-" * 40)
        
        if not h5_path.exists():
            print(f"❌ File not found: {h5_path}")
            continue
            
        try:
            # Get contingency description
            scenario_info = df[df['scenario_id'] == scenario_id]
            if not scenario_info.empty:
                description = scenario_info['description'].iloc[0]
                print(f"📋 Contingency: {description}")
            
            # Run load flow analysis
            results = create_powerfactory_based_results(str(h5_path))
            
            # Load PowerFactory reference
            with h5py.File(h5_path, 'r') as f:
                # Get contingency details
                if 'disconnection_actions' in f:
                    actions = f['disconnection_actions/actions'][:]
                    contingency_desc = actions[0].decode() if len(actions) > 0 else f'Scenario {scenario_id}'
                else:
                    contingency_desc = f'Scenario {scenario_id}'
                
                # Load reference data
                pf_voltages = f['load_flow_results/bus_data/bus_voltages_pu'][:]
                pf_angles = f['load_flow_results/bus_data/bus_angles_deg'][:]
                pf_gen = float(f['power_flow_data/system_totals/total_generation_MW'][()])
                pf_load = float(f['power_flow_data/system_totals/total_load_MW'][()])
                pf_losses = float(f['power_flow_data/system_totals/total_losses_MW'][()])
                bus_names = [name.decode() for name in f['load_flow_results/bus_data/bus_names'][:]]
            
            # Calculate key metrics
            min_voltage = float(pf_voltages.min())
            max_voltage = float(pf_voltages.max())
            avg_voltage = float(pf_voltages.mean())
            violations_low = int(np.sum(pf_voltages < 0.95))
            violations_high = int(np.sum(pf_voltages > 1.05))
            
            # Display results
            print(f"✅ Load Flow Converged: {results.converged}")
            print(f"📊 Voltage Range: {min_voltage:.3f} - {max_voltage:.3f} pu")
            print(f"⚡ Generation: {pf_gen:.1f} MW")
            print(f"🔌 Load: {pf_load:.1f} MW")
            print(f"📉 Losses: {pf_losses:.1f} MW")
            print(f"⚠️  Voltage Violations: {violations_low} low, {violations_high} high")
            
            # Assess severity
            if min_voltage < 0.85:
                severity = "🔴 CRITICAL"
                severity_text = "CRITICAL"
            elif min_voltage < 0.90:
                severity = "🟡 SEVERE"
                severity_text = "SEVERE"
            elif min_voltage < 0.95:
                severity = "🟠 STRESSED"
                severity_text = "STRESSED"
            else:
                severity = "🟢 STABLE"
                severity_text = "STABLE"
            
            print(f"🎯 System Status: {severity}")
            
            # Create detailed PowerFactory comparison plots (3 figures)
            print(f"🔍 Creating detailed PowerFactory comparisons...")
            create_powerfactory_comparisons(scenario_id, str(h5_path), output_dir)
            
            # Create summary comparison plot
            create_contingency_comparison_plot(
                scenario_id, contingency_desc, bus_names, pf_voltages,
                results.voltage_magnitudes[::3], output_dir
            )
            
            # Store results for summary
            results_summary.append({
                'scenario_id': scenario_id,
                'contingency': contingency_desc,
                'min_voltage': min_voltage,
                'max_voltage': max_voltage,
                'avg_voltage': avg_voltage,
                'generation_mw': pf_gen,
                'load_mw': pf_load,
                'losses_mw': pf_losses,
                'violations_low': violations_low,
                'violations_high': violations_high,
                'severity': severity,
                'severity_text': severity_text
            })
            
        except Exception as e:
            print(f"❌ Error analyzing scenario {scenario_id}: {str(e)}")
    
    # Create summary report
    if results_summary:
        create_contingency_summary_report(results_summary, output_dir)
        create_contingency_summary_plot(results_summary, output_dir)
    
    print(f"\n📈 All analysis results saved to: {output_dir}")
    print("🏆 Contingency analysis demo completed successfully!")


def create_contingency_comparison_plot(scenario_id, contingency_desc, bus_names, 
                                     pf_voltages, our_voltages, output_dir):
    """Create voltage comparison plot for a specific contingency scenario"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Voltage comparison
    x = np.arange(len(bus_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pf_voltages, width, 
                    label='PowerFactory (Reference)', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, our_voltages, width,
                    label='Our Solver', color='orange', alpha=0.8)
    
    ax1.set_ylabel('Voltage (pu)', fontsize=12)
    ax1.set_title(f'Contingency Analysis: Scenario {scenario_id}\n{contingency_desc}', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bus_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add voltage limits
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Low Limit')
    ax1.axhline(y=1.05, color='red', linestyle='--', alpha=0.5, label='High Limit')
    
    # Bottom plot: Voltage violations highlight
    violation_colors = ['red' if v < 0.95 or v > 1.05 else 'green' for v in pf_voltages]
    ax2.bar(x, pf_voltages, color=violation_colors, alpha=0.7)
    ax2.set_ylabel('Voltage (pu)', fontsize=12)
    ax2.set_xlabel('Bus Name', fontsize=12)
    ax2.set_title('Voltage Violations Assessment', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bus_names, rotation=45, ha='right')
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.8)
    ax2.axhline(y=1.05, color='red', linestyle='--', alpha=0.8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"contingency_scenario_{scenario_id}_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {plot_path.name}")


def create_contingency_summary_plot(results_summary, output_dir):
    """Create summary plots across all contingency scenarios"""
    
    if not results_summary:
        return
    
    scenarios = [r['scenario_id'] for r in results_summary]
    min_voltages = [r['min_voltage'] for r in results_summary]
    losses = [r['losses_mw'] for r in results_summary]
    violations_low = [r['violations_low'] for r in results_summary]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Minimum voltages
    colors = ['red' if v < 0.90 else 'orange' if v < 0.95 else 'green' for v in min_voltages]
    ax1.bar(scenarios, min_voltages, color=colors, alpha=0.7)
    ax1.set_ylabel('Minimum Voltage (pu)')
    ax1.set_title('Minimum Bus Voltage by Scenario')
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='Operating Limit')
    ax1.axhline(y=0.90, color='darkred', linestyle='--', alpha=0.8, label='Emergency Limit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: System losses
    ax2.bar(scenarios, losses, color='orange', alpha=0.7)
    ax2.set_ylabel('System Losses (MW)')
    ax2.set_title('System Losses by Scenario')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Voltage violations
    ax3.bar(scenarios, violations_low, color='red', alpha=0.7)
    ax3.set_ylabel('Low Voltage Violations')
    ax3.set_xlabel('Scenario ID')
    ax3.set_title('Buses with Voltage < 0.95 pu')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Severity assessment
    severity_scores = [(1.0 - r['min_voltage']) * 100 + r['violations_low'] * 2 for r in results_summary]
    severity_colors = ['green' if s < 2 else 'orange' if s < 5 else 'red' for s in severity_scores]
    ax4.bar(scenarios, severity_scores, color=severity_colors, alpha=0.7)
    ax4.set_ylabel('Severity Score')
    ax4.set_xlabel('Scenario ID')
    ax4.set_title('Contingency Severity Assessment')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Contingency Analysis Summary - All Scenarios', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"contingency_summary_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary plot saved: {plot_path.name}")


def create_contingency_summary_report(results_summary, output_dir):
    """Create detailed text report of contingency analysis results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"contingency_analysis_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PIGNN CONTINGENCY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total scenarios analyzed: {len(results_summary)}\n")
        
        min_voltages = [r['min_voltage'] for r in results_summary]
        f.write(f"Minimum voltage across all scenarios: {min(min_voltages):.3f} pu\n")
        f.write(f"Maximum voltage across all scenarios: {max([r['max_voltage'] for r in results_summary]):.3f} pu\n")
        f.write(f"Average system losses: {sum([r['losses_mw'] for r in results_summary])/len(results_summary):.1f} MW\n")
        
        critical_scenarios = [r for r in results_summary if r['min_voltage'] < 0.90]
        f.write(f"Critical scenarios (min V < 0.90 pu): {len(critical_scenarios)}\n\n")
        
        f.write("DETAILED SCENARIO RESULTS\n")
        f.write("-" * 30 + "\n")
        
        for result in sorted(results_summary, key=lambda x: x['min_voltage']):
            f.write(f"\nScenario {result['scenario_id']}: {result['contingency'][:60]}...\n")
            f.write(f"  Min Voltage: {result['min_voltage']:.3f} pu\n")
            f.write(f"  Generation: {result['generation_mw']:.1f} MW\n")
            f.write(f"  Load: {result['load_mw']:.1f} MW\n")
            f.write(f"  Losses: {result['losses_mw']:.1f} MW\n")
            f.write(f"  Violations: {result['violations_low']} low, {result['violations_high']} high\n")
            f.write(f"  Status: {result['severity_text']}\n")
    
    print(f"📄 Report saved: {report_path.name}")


if __name__ == "__main__":
    run_contingency_demo()