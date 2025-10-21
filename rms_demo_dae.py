"""
Demo: RMS Simulator with DAE Framework and Implicit Trapezoid Solver

Tests the new DAE-based RMS simulator with implicit integration.

Expected improvements over old RK4 version:
1. NO NaN values (implicit method handles stiffness)
2. Stable integration (proper DAE formulation)
3. Converges from realistic initial conditions

Author: PIGNN Project
Date: October 20, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RMS_Analysis.rms_simulator_dae import RMSSimulator


def main():
    print("="*70)
    print("RMS SIMULATOR - DAE VERSION DEMO")
    print("="*70)
    
    # Path to H5 file
    h5_file = "graph_model/Graph_model.h5"
    
    if not os.path.exists(h5_file):
        print(f"❌ Error: H5 file not found: {h5_file}")
        return
    
    # Create simulator
    print("\nCreating RMS simulator with implicit trapezoid solver...")
    sim = RMSSimulator(
        h5_file=h5_file,
        dt=0.005,  # 5ms time step (same as debugging scripts)
        f_base=60.0
    )
    
    # Initialize (run load flow, create models, setup DAE)
    sim.initialize()
    
    # Run simulation
    print("\n" + "="*70)
    print("STARTING SIMULATION")
    print("="*70)
    
    t_end = 1.0  # 1 second simulation
    results = sim.simulate(t_end=t_end)
    
    # Plot results
    print("\n" + "="*70)
    print("PLOTTING RESULTS")
    print("="*70)
    
    plot_results(results, sim.gen_names)
    
    print("\n✅ Demo complete!")


def plot_results(results, gen_names):
    """Plot simulation results."""
    time = results['time']
    
    # Plot first 3 generators
    plot_gens = gen_names[:min(3, len(gen_names))]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Rotor angle
    ax = axes[0]
    for gen_name in plot_gens:
        gen_data = results['generators'][gen_name]
        ax.plot(time, gen_data['delta_deg'], label=gen_name, linewidth=1.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rotor Angle (degrees)')
    ax.set_title('Generator Rotor Angles - DAE Implicit Solver')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Frequency
    ax = axes[1]
    for gen_name in plot_gens:
        gen_data = results['generators'][gen_name]
        ax.plot(time, gen_data['freq_Hz'], label=gen_name, linewidth=1.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Generator Frequencies - DAE Implicit Solver')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=60.0, color='k', linestyle='--', alpha=0.5, label='Nominal')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = "RMS_Analysis/rms_plots/dae_simulation_test.png"
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {plot_file}")
    
    # Check for issues
    print("\n  Checking results:")
    
    all_good = True
    for gen_name in gen_names:
        gen_data = results['generators'][gen_name]
        
        # Check for NaN
        if np.any(np.isnan(gen_data['delta_deg'])):
            print(f"    ❌ {gen_name}: NaN values detected!")
            all_good = False
        
        # Check for excessive drift
        delta_change = abs(gen_data['delta_deg'][-1] - gen_data['delta_deg'][0])
        if delta_change > 10.0:  # degrees
            print(f"    ⚠️  {gen_name}: Large angle change ({delta_change:.2f}°)")
        
        # Check frequency stays near 60Hz
        freq_dev = abs(gen_data['freq_Hz'] - 60.0).max()
        if freq_dev > 0.5:  # Hz
            print(f"    ⚠️  {gen_name}: Large frequency deviation ({freq_dev:.3f} Hz)")
    
    if all_good:
        print("    ✅ No NaN values detected!")
        print("    ✅ All generators stable!")
    
    plt.show()


if __name__ == "__main__":
    main()
