"""
Visualize RMS simulation results with REAL load flow initialization
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
h5_path = Path("graph_model/Graph_model.h5")
plot_dir = Path("RMS_Analysis/rms_plots")

print("="*70)
print("RMS SIMULATION RESULTS ANALYSIS")
print("="*70)

# Load power flow results
print("\n1. LOAD FLOW RESULTS (Initial Conditions)")
print("-"*70)

with h5py.File(h5_path, 'r') as f:
    pf = f['steady_state/power_flow_results']
    
    bus_V = pf['bus_voltages_pu'][:]
    bus_ang = pf['bus_angles_deg'][:]
    gen_P_bus = pf['gen_P_MW'][:]
    gen_Q_bus = pf['gen_Q_MVAR'][:]
    
    print(f"Load Flow Convergence:")
    print(f"  Converged: {pf.attrs['converged']}")
    print(f"  Iterations: {pf.attrs['iterations']}")
    print(f"  Max mismatch: {pf.attrs['max_mismatch']:.2e}")
    print(f"\nVoltage Profile:")
    print(f"  Min: {bus_V.min():.4f} pu at bus {bus_V.argmin()+1}")
    print(f"  Max: {bus_V.max():.4f} pu at bus {bus_V.argmax()+1}")
    print(f"  Mean: {bus_V.mean():.4f} pu")
    print(f"\nAngle Spread:")
    print(f"  Min: {bus_ang.min():.2f}° at bus {bus_ang.argmin()+1}")
    print(f"  Max: {bus_ang.max():.2f}° at bus {bus_ang.argmax()+1}")
    print(f"  Range: {bus_ang.max() - bus_ang.min():.2f}°")
    
    # Generator outputs
    gen_buses = np.where(gen_P_bus > 0)[0]
    print(f"\nGenerator Outputs ({len(gen_buses)} generators):")
    total_P = 0
    total_Q = 0
    for idx in gen_buses:
        print(f"  Bus {idx+1:2d}: P = {gen_P_bus[idx]:6.1f} MW, Q = {gen_Q_bus[idx]:6.1f} MVAR")
        total_P += gen_P_bus[idx]
        total_Q += gen_Q_bus[idx]
    
    print(f"\nTotal Generation:")
    print(f"  Active Power: {total_P:.1f} MW")
    print(f"  Reactive Power: {total_Q:.1f} MVAR")
    print(f"  Losses: {pf.attrs['total_losses_MW']:.1f} MW")

# Create visualization
print("\n2. CREATING VISUALIZATIONS")
print("-"*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Bus Voltage Profile
ax1 = plt.subplot(2, 3, 1)
bus_nums = np.arange(1, len(bus_V)+1)
colors = ['red' if v < 0.95 else 'orange' if v < 0.98 else 'green' for v in bus_V]
ax1.bar(bus_nums, bus_V, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0.95, color='r', linestyle='--', label='0.95 pu limit')
ax1.axhline(y=1.05, color='r', linestyle='--', label='1.05 pu limit')
ax1.set_xlabel('Bus Number', fontweight='bold')
ax1.set_ylabel('Voltage (pu)', fontweight='bold')
ax1.set_title('Bus Voltage Profile (Post Load Flow)', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0.9, 1.1])

# Plot 2: Bus Angle Profile
ax2 = plt.subplot(2, 3, 2)
ax2.plot(bus_nums, bus_ang, 'b-o', markersize=4, linewidth=1.5)
ax2.fill_between(bus_nums, bus_ang, alpha=0.3)
ax2.set_xlabel('Bus Number', fontweight='bold')
ax2.set_ylabel('Angle (degrees)', fontweight='bold')
ax2.set_title('Bus Angle Profile', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Generator Active Power
ax3 = plt.subplot(2, 3, 3)
gen_labels = [f'B{idx+1}' for idx in gen_buses]
gen_P_vals = [gen_P_bus[idx] for idx in gen_buses]
colors_gen = plt.cm.viridis(np.linspace(0, 1, len(gen_buses)))
bars = ax3.barh(gen_labels, gen_P_vals, color=colors_gen, edgecolor='black')
ax3.set_xlabel('Active Power (MW)', fontweight='bold')
ax3.set_title('Generator Output Distribution', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, gen_P_vals)):
    ax3.text(val + 20, bar.get_y() + bar.get_height()/2, 
             f'{val:.0f} MW', 
             va='center', fontsize=9, fontweight='bold')

# Plot 4: Load Flow convergence stats
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
stats_text = f"""
LOAD FLOW RESULTS
{'='*40}

Convergence:
  Status: {'✅ CONVERGED' if pf.attrs['converged'] else '❌ FAILED'}
  Iterations: {pf.attrs['iterations']}
  Max Mismatch: {pf.attrs['max_mismatch']:.2e} pu

System Summary:
  Buses: {len(bus_V)}
  Generators: {len(gen_buses)}
  Total Generation: {total_P:.1f} MW
  Total Losses: {pf.attrs['total_losses_MW']:.1f} MW
  Loss %: {100*pf.attrs['total_losses_MW']/total_P:.2f}%

Voltage Statistics:
  Min: {bus_V.min():.4f} pu (Bus {bus_V.argmin()+1})
  Max: {bus_V.max():.4f} pu (Bus {bus_V.argmax()+1})
  Mean: {bus_V.mean():.4f} pu
  Std Dev: {bus_V.std():.4f} pu

Angle Statistics:
  Min: {bus_ang.min():.2f}° (Bus {bus_ang.argmin()+1})
  Max: {bus_ang.max():.2f}° (Bus {bus_ang.argmax()+1})
  Spread: {bus_ang.max()-bus_ang.min():.2f}°
"""
ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 5: Generator Loading (pu)
ax5 = plt.subplot(2, 3, 5)
# Load generator rated powers
with h5py.File(h5_path, 'r') as f:
    gen_Sn = f['dynamic_models/generators/Sn_MVA'][:]
    gen_names = [n.decode() if isinstance(n, bytes) else n for n in f['dynamic_models/generators/names'][:]]
    gen_bus_names = [n.decode() if isinstance(n, bytes) else n for n in f['dynamic_models/generators/buses'][:]]

# Map generator powers
gen_P_actual = []
gen_loading = []
for i, bus_name in enumerate(gen_bus_names):
    # Find bus index (simplified - assumes bus name matches)
    bus_num = int(bus_name.split()[-1])  # Extract bus number from "Bus XX"
    bus_idx = bus_num - 1
    
    P_MW = gen_P_bus[bus_idx] if bus_idx < len(gen_P_bus) else 0
    loading_pu = P_MW / gen_Sn[i] if gen_Sn[i] > 0 else 0
    
    gen_P_actual.append(P_MW)
    gen_loading.append(loading_pu)

# Plot loading
x = np.arange(len(gen_names))
colors_loading = ['red' if l > 0.9 else 'orange' if l > 0.75 else 'green' for l in gen_loading]
bars = ax5.bar(x, gen_loading, color=colors_loading, edgecolor='black', alpha=0.7)
ax5.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Rated capacity')
ax5.set_xlabel('Generator', fontweight='bold')
ax5.set_ylabel('Loading (pu)', fontweight='bold')
ax5.set_title('Generator Loading (P/Prated)', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels([f'G{i+1}' for i in range(len(gen_names))], rotation=45)
ax5.grid(True, alpha=0.3, axis='y')
ax5.legend()

# Plot 6: P-Q diagram
ax6 = plt.subplot(2, 3, 6)
gen_P_scatter = [gen_P_bus[idx] for idx in gen_buses]
gen_Q_scatter = [gen_Q_bus[idx] for idx in gen_buses]
sc = ax6.scatter(gen_P_scatter, gen_Q_scatter, s=200, c=range(len(gen_buses)), 
                 cmap='viridis', edgecolors='black', linewidth=2, alpha=0.7)
for i, idx in enumerate(gen_buses):
    ax6.annotate(f'B{idx+1}', (gen_P_scatter[i], gen_Q_scatter[i]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

ax6.set_xlabel('Active Power P (MW)', fontweight='bold')
ax6.set_ylabel('Reactive Power Q (MVAR)', fontweight='bold')
ax6.set_title('Generator P-Q Operating Points', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax6.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('loadflow_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: loadflow_analysis.png")

# Show existing RMS plots
print("\n3. EXISTING RMS PLOTS")
print("-"*70)
if plot_dir.exists():
    plots = list(plot_dir.glob("*.png"))
    if plots:
        for plot in plots:
            print(f"  📊 {plot.name}")
        print(f"\nTotal plots: {len(plots)}")
    else:
        print("  No plots found")
else:
    print("  Plot directory not found")

plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
