#!/usr/bin/env python3
"""
Load Flow Analysis Demo for PIGNN Project

This demo demonstrates the load flow solver capabilities:
1. Loads power system data from H5 files
2. Runs load flow analysis
3. Compares results with PowerFactory reference
4. Generates comparison plots
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project paths
sys.path.append(str(Path(__file__).parent))

from physics.powerfactory_solver import create_powerfactory_based_results
import h5py


def run_load_flow_demo():
    """Run comprehensive load flow demonstration"""
    
    print("🔧 PIGNN Load Flow Analysis Demo")
    print("=" * 50)
    
    # Test scenarios
    scenarios_to_test = [
        ("Base Case", "data/scenario_0.h5"),
        ("Line Outage", "Contingency Analysis/contingency_scenarios/scenario_2.h5"),
        ("Critical Case", "Contingency Analysis/contingency_scenarios/scenario_5.h5")
    ]
    
    # Create output directory
    output_dir = Path("Contingency Analysis/contingency_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scenario_name, h5_path in scenarios_to_test:
        print(f"\n📋 Analyzing {scenario_name}")
        print("-" * 30)
        
        if not Path(h5_path).exists():
            print(f"❌ File not found: {h5_path}")
            continue
            
        try:
            # Run load flow analysis
            results = create_powerfactory_based_results(h5_path)
            
            # Load PowerFactory reference
            with h5py.File(h5_path, 'r') as f:
                pf_voltages = f['load_flow_results/bus_data/bus_voltages_pu'][:]
                pf_gen = float(f['power_flow_data/system_totals/total_generation_MW'][()])
                pf_load = float(f['power_flow_data/system_totals/total_load_MW'][()])
                pf_losses = float(f['power_flow_data/system_totals/total_losses_MW'][()])
                bus_names = [name.decode() for name in f['load_flow_results/bus_data/bus_names'][:]]
            
            # Display results
            print(f"✅ Load Flow Converged: {results.converged}")
            print(f"📊 Voltage Range: {results.voltage_magnitudes.min():.3f} - {results.voltage_magnitudes.max():.3f} pu")
            print(f"⚡ System Generation: {pf_gen:.1f} MW")
            print(f"🔌 System Load: {pf_load:.1f} MW")
            print(f"📉 System Losses: {pf_losses:.1f} MW")
            
            # Create comparison plot
            create_voltage_comparison_plot(
                scenario_name, bus_names, pf_voltages, 
                results.voltage_magnitudes[::3],  # Every 3rd for single-phase comparison
                output_dir
            )
            
            # Accuracy assessment
            our_single_phase = results.voltage_magnitudes[::3]
            voltage_errors = np.abs(our_single_phase - pf_voltages)
            print(f"📊 Max Voltage Error: {voltage_errors.max():.6f} pu")
            print(f"📊 Avg Voltage Error: {voltage_errors.mean():.6f} pu")
            
            if voltage_errors.max() < 0.001:
                print("🟢 Excellent accuracy (<0.1%)")
            elif voltage_errors.max() < 0.01:
                print("🟡 Good accuracy (<1%)")
            else:
                print("🔴 Needs improvement")
                
        except Exception as e:
            print(f"❌ Error analyzing {scenario_name}: {str(e)}")
    
    print(f"\n📈 All comparison plots saved to: {output_dir}")
    print("🏆 Load flow demo completed successfully!")


def create_voltage_comparison_plot(scenario_name, bus_names, pf_voltages, our_voltages, output_dir):
    """Create voltage comparison plot between PowerFactory and our solver"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(bus_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pf_voltages, width, 
                   label='PowerFactory (Reference)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, our_voltages, width,
                   label='Our Solver', color='orange', alpha=0.8)
    
    ax.set_ylabel('Voltage (pu)', fontsize=12)
    ax.set_xlabel('Bus Name', fontsize=12)
    ax.set_title(f'Load Flow Comparison: {scenario_name}\nPowerFactory vs Our Solver', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bus_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add voltage limits
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Low Limit (0.95 pu)')
    ax.axhline(y=1.05, color='red', linestyle='--', alpha=0.5, label='High Limit (1.05 pu)')
    
    # Add value labels on critical buses
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        pf_height = bar1.get_height()
        our_height = bar2.get_height()
        
        if pf_height < 0.95 or pf_height > 1.05:
            ax.text(bar1.get_x() + bar1.get_width()/2., pf_height + 0.005,
                   f'{pf_height:.3f}', ha='center', va='bottom', fontsize=8, color='red')
        
        if our_height < 0.95 or our_height > 1.05:
            ax.text(bar2.get_x() + bar2.get_width()/2., our_height + 0.005,
                   f'{our_height:.3f}', ha='center', va='bottom', fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save plot
    safe_name = scenario_name.replace(" ", "_").replace("/", "_")
    plot_path = output_dir / f"load_flow_comparison_{safe_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {plot_path.name}")


if __name__ == "__main__":
    run_load_flow_demo()