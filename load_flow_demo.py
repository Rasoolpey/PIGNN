"""
Load Flow Demo - Updated for Graph_model.h5
Demonstrates load flow solver reading from and writing to Graph_model.h5

Date: 2025-10-20
Data Source: graph_model/Graph_model.h5 (SINGLE SOURCE OF TRUTH)
Output: Updates steady_state group in Graph_model.h5
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Add project paths
sys.path.append(str(Path(__file__).parent))

from physics.load_flow_solver import run_load_flow_from_h5


def run_load_flow_demo():
    """Run load flow demonstration using Graph_model.h5"""
    
    print("🔧 PIGNN Load Flow Analysis Demo")
    print("=" * 60)
    print("📊 Data Source: graph_model/Graph_model.h5")
    print("=" * 60)
    
    h5_path = "graph_model/Graph_model.h5"
    
    if not Path(h5_path).exists():
        print(f"❌ Graph_model.h5 not found at: {h5_path}")
        return
    
    # Step 1: Load initial data
    print("\n📋 Step 1: Loading system data from Graph_model.h5...")
    
    with h5py.File(h5_path, 'r') as f:
        num_buses = f['metadata'].attrs['num_buses']
        num_generators = f['metadata'].attrs['num_generators']
        num_loads = f['metadata'].attrs['num_loads']
        
        bus_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f['phases/phase_a/nodes/bus_names'][:]]
        
        P_load = f['phases/phase_a/nodes/P_load_MW'][:]
        Q_load = f['phases/phase_a/nodes/Q_load_MVAR'][:]
        
        print(f"   ✓ Buses: {num_buses}")
        print(f"   ✓ Generators: {num_generators}")
        print(f"   ✓ Loads: {num_loads}")
        print(f"   ✓ Total Load: {np.sum(P_load):.1f} MW, {np.sum(Q_load):.1f} MVAR")
    
    # Step 2: Run load flow solver
    print("\n📋 Step 2: Running load flow solver...")
    
    try:
        # This will run load flow AND save results to Graph_model.h5
        results = run_load_flow_from_h5(h5_path, save_to_h5=True)
        
        if results.converged:
            print(f"   ✅ Load Flow Converged in {results.iterations} iterations")
            print(f"   📊 Voltage Range: {results.voltage_magnitudes.min():.4f} - {results.voltage_magnitudes.max():.4f} pu")
            print(f"   📊 Angle Range: {np.rad2deg(results.voltage_angles).min():.2f}° - {np.rad2deg(results.voltage_angles).max():.2f}°")
            
            # Calculate totals
            total_gen = np.sum(results.active_power[results.active_power > 0])
            total_load = abs(np.sum(results.active_power[results.active_power < 0]))
            
            print(f"   ⚡ Total Generation: {total_gen:.1f} MW")
            print(f"   🔌 Total Load: {total_load:.1f} MW")
            print(f"   📉 Total Losses: {results.total_losses_mw:.1f} MW")
            print(f"   ✅ Power Balance: {total_gen:.1f} = {total_load:.1f} + {results.total_losses_mw:.1f} MW")
        else:
            print(f"   ❌ Load Flow did NOT converge!")
            return
            
    except Exception as e:
        print(f"   ❌ Error running load flow: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Verify data was written to Graph_model.h5
    print("\n📋 Step 3: Verifying load flow results in Graph_model.h5...")
    
    with h5py.File(h5_path, 'r') as f:
        if 'steady_state' in f:
            if 'power_flow_results' in f['steady_state']:
                pf_group = f['steady_state/power_flow_results']
                
                stored_V = pf_group['bus_voltages_pu'][:]
                stored_angles = pf_group['bus_angles_deg'][:]
                stored_P_gen = pf_group['gen_P_MW'][:]
                stored_Q_gen = pf_group['gen_Q_MVAR'][:]
                
                print(f"   ✅ Found steady_state/power_flow_results group")
                print(f"   ✓ Voltages: {len(stored_V)} values")
                print(f"   ✓ Angles: {len(stored_angles)} values")
                print(f"   ✓ Gen P: {len(stored_P_gen)} values (Total: {np.sum(stored_P_gen):.1f} MW)")
                print(f"   ✓ Gen Q: {len(stored_Q_gen)} values (Total: {np.sum(stored_Q_gen):.1f} MVAR)")
                
                # Verify match with solver results
                voltage_match = np.allclose(stored_V, results.voltage_magnitudes[::3], rtol=1e-4)  # Compare phase A only
                angle_match = np.allclose(stored_angles, np.rad2deg(results.voltage_angles)[::3], rtol=1e-4)
                
                if voltage_match and angle_match:
                    print(f"   ✅ Stored data MATCHES solver results!")
                else:
                    print(f"   ⚠️ Stored data does NOT match solver results")
            else:
                print(f"   ❌ No power_flow_results found in steady_state group")
        else:
            print(f"   ❌ No steady_state group found in Graph_model.h5")
    
    # Step 4: Create comparison plots
    print("\n📋 Step 4: Creating visualization plots...")
    
    output_dir = Path("Contingency Analysis/contingency_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_voltage_comparison_plot(
        "Base Case - Graph_model.h5",
        bus_names,
        results.voltage_magnitudes[::3],  # Phase A only
        results.voltage_angles[::3],
        output_dir
    )
    
    print(f"\n📈 All plots saved to: {output_dir}/")
    print("🏆 Load flow demo completed successfully!")
    print("\n" + "=" * 60)
    print("✅ SUMMARY:")
    print(f"   - Load flow converged: {results.converged}")
    print(f"   - Results saved to: graph_model/Graph_model.h5 (steady_state group)")
    print(f"   - Plots saved to: {output_dir}/")
    print(f"   - Data ready for RMS simulation!")
    print("=" * 60)


def create_voltage_comparison_plot(scenario_name, bus_names, voltages, angles_rad, output_dir):
    """Create voltage and angle plots"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.arange(len(bus_names))
    
    # Voltage plot
    ax1.bar(x, voltages, color='steelblue', alpha=0.8, label='Voltage Magnitude')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Nominal')
    ax1.axhline(y=0.95, color='red', linestyle=':', linewidth=1.5, label='±5% limits')
    ax1.axhline(y=1.05, color='red', linestyle=':', linewidth=1.5)
    ax1.set_ylabel('Voltage (pu)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Load Flow Results: {scenario_name}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bus_names, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Angle plot
    angles_deg = np.rad2deg(angles_rad)
    ax2.bar(x, angles_deg, color='darkred', alpha=0.8, label='Voltage Angle')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Bus Name', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bus_names, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    safe_name = scenario_name.replace(" ", "_").replace("/", "_").replace(".", "_")
    plot_path = output_dir / f"load_flow_{safe_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Plot saved: {plot_path.name}")


if __name__ == "__main__":
    run_load_flow_demo()
