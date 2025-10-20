"""
Visualization Demo - Updated for Graph_model.h5
Creates visualization of the IEEE 39-bus power system from the master Graph_model.h5 file.

Date: 2025-10-20
Data Source: graph_model/Graph_model.h5 (SINGLE SOURCE OF TRUTH)
"""

import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import numpy as np


def main():
    """Main visualization demo workflow"""
    
    # Path to Graph_model.h5 (single source of truth)
    h5_path = "graph_model/Graph_model.h5"
    
    print("🔄 Three-Phase Power Grid Visualization Demo")
    print("=" * 60)
    print(f"📊 Data Source: {h5_path}")
    print("=" * 60)
    
    # Load data from Graph_model.h5
    print("\n📊 Loading IEEE39 system data from Graph_model.h5...")
    
    with h5py.File(h5_path, 'r') as f:
        # Read metadata
        num_buses = f['metadata'].attrs['num_buses']
        num_generators = f['metadata'].attrs['num_generators']
        num_loads = f['metadata'].attrs['num_loads']
        num_lines = f['metadata'].attrs['num_lines']
        num_transformers = f['metadata'].attrs['num_transformers']
        
        print(f"   ✓ Buses: {num_buses}")
        print(f"   ✓ Lines: {num_lines}")
        print(f"   ✓ Transformers: {num_transformers}")
        print(f"   ✓ Generators: {num_generators}")
        print(f"   ✓ Loads: {num_loads}")
        
        # Read bus data
        bus_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f['phases/phase_a/nodes/bus_names'][:]]
        bus_voltages = f['phases/phase_a/nodes/voltages_pu'][:]
        bus_angles = f['phases/phase_a/nodes/angles_deg'][:]
        P_load = f['phases/phase_a/nodes/P_load_MW'][:]
        Q_load = f['phases/phase_a/nodes/Q_load_MVAR'][:]
        P_gen = f['phases/phase_a/nodes/P_generation_MW'][:]
        
        # Read edge data
        edge_types = f['phases/phase_a/edges/element_type'][:]
        from_buses = f['phases/phase_a/edges/from_bus'][:]
        to_buses = f['phases/phase_a/edges/to_bus'][:]
        
        print(f"\n📊 System Summary:")
        print(f"   Total Load: {np.sum(P_load):.1f} MW, {np.sum(Q_load):.1f} MVAR")
        print(f"   Total Generation: {np.sum(P_gen):.1f} MW")
        print(f"   Total Edges: {len(edge_types)}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Create output directory
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Voltage Profile Plot
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    bus_indices = np.arange(num_buses)
    ax1.plot(bus_indices, bus_voltages, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Nominal (1.0 pu)')
    ax1.axhline(y=0.95, color='orange', linestyle=':', linewidth=1.5, label='±5% limits')
    ax1.axhline(y=1.05, color='orange', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Bus Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Voltage (pu)', fontsize=12, fontweight='bold')
    ax1.set_title('IEEE 39-Bus System - Voltage Profile\n(from Graph_model.h5)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    plt.tight_layout()
    
    voltage_plot_path = output_dir / "voltage_profile.png"
    fig1.savefig(voltage_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {voltage_plot_path}")
    plt.close(fig1)
    
    # 2. Power Distribution Plot
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Active Power
    ax2a.bar(bus_indices, P_gen, alpha=0.7, color='green', label='Generation', width=0.4)
    ax2a.bar(bus_indices, -P_load, alpha=0.7, color='red', label='Load', width=0.4)
    ax2a.set_xlabel('Bus Index', fontsize=12, fontweight='bold')
    ax2a.set_ylabel('Active Power (MW)', fontsize=12, fontweight='bold')
    ax2a.set_title('Active Power Distribution (Generation vs Load)', fontsize=13, fontweight='bold')
    ax2a.grid(True, alpha=0.3)
    ax2a.legend(fontsize=10)
    ax2a.axhline(y=0, color='black', linewidth=1)
    
    # Reactive Power
    ax2b.bar(bus_indices, -Q_load, alpha=0.7, color='purple', label='Reactive Load', width=0.6)
    ax2b.set_xlabel('Bus Index', fontsize=12, fontweight='bold')
    ax2b.set_ylabel('Reactive Power (MVAR)', fontsize=12, fontweight='bold')
    ax2b.set_title('Reactive Power Distribution', fontsize=13, fontweight='bold')
    ax2b.grid(True, alpha=0.3)
    ax2b.legend(fontsize=10)
    ax2b.axhline(y=0, color='black', linewidth=1)
    
    plt.tight_layout()
    power_plot_path = output_dir / "power_distribution.png"
    fig2.savefig(power_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {power_plot_path}")
    plt.close(fig2)
    
    # 3. Angle Distribution Plot
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(bus_indices, bus_angles, 'o-', linewidth=2, markersize=6, color='darkred')
    ax3.set_xlabel('Bus Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Voltage Angle (degrees)', fontsize=12, fontweight='bold')
    ax3.set_title('IEEE 39-Bus System - Voltage Angle Profile\n(from Graph_model.h5)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    
    angle_plot_path = output_dir / "angle_profile.png"
    fig3.savefig(angle_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {angle_plot_path}")
    plt.close(fig3)
    
    print("\n✅ Visualization complete!")
    print(f"📁 All plots saved to: {output_dir}/")
    print(f"   - voltage_profile.png")
    print(f"   - power_distribution.png")
    print(f"   - angle_profile.png")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
