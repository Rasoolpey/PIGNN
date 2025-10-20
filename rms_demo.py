"""
RMS Simulation Demo

This script demonstrates dynamic transient stability simulation:
1. Loads IEEE 39-bus system from Graph_model.h5
2. Runs fault scenarios (3-phase fault)
3. Plots rotor angles, frequencies, voltages
4. Analyzes stability

Author: PIGNN Project
Date: October 20, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from RMS_Analysis.rms_simulator import RMSSimulator, FaultEvent


def plot_rotor_angles(results: dict, title: str = "Rotor Angles"):
    """Plot rotor angles for all generators."""
    plt.figure(figsize=(12, 6))
    
    time = results['time']
    
    for gen_name, gen_data in results['generators'].items():
        delta = gen_data['delta_deg']
        plt.plot(time, delta, label=gen_name, linewidth=1.5)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Rotor Angle (deg)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_frequencies(results: dict, title: str = "Generator Frequencies"):
    """Plot frequencies for all generators."""
    plt.figure(figsize=(12, 6))
    
    time = results['time']
    
    for gen_name, gen_data in results['generators'].items():
        freq = gen_data['freq_Hz']
        plt.plot(time, freq, label=gen_name, linewidth=1.5)
    
    plt.axhline(y=60.0, color='k', linestyle='--', alpha=0.5, label='Nominal (60 Hz)')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim([59.0, 61.0])  # Typical range
    plt.tight_layout()


def plot_rotor_angle_differences(results: dict, title: str = "Rotor Angle Differences"):
    """Plot rotor angle differences (relative to COI)."""
    plt.figure(figsize=(12, 6))
    
    time = results['time']
    gen_names = list(results['generators'].keys())
    
    # Compute Center of Inertia (COI) angle
    delta_coi = np.zeros(len(time))
    for gen_name in gen_names:
        delta = results['generators'][gen_name]['delta_deg']
        delta_coi += delta
    delta_coi /= len(gen_names)
    
    # Plot differences
    for gen_name in gen_names:
        delta = results['generators'][gen_name]['delta_deg']
        delta_diff = delta - delta_coi
        plt.plot(time, delta_diff, label=gen_name, linewidth=1.5)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle Difference from COI (deg)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def analyze_stability(results: dict) -> dict:
    """
    Analyze transient stability from simulation results.
    
    Returns:
        Dictionary with stability metrics
    """
    metrics = {}
    
    for gen_name, gen_data in results['generators'].items():
        freq = gen_data['freq_Hz']
        delta = gen_data['delta_deg']
        
        # Peak frequency deviation
        freq_dev = freq - 60.0
        max_freq_dev = np.max(np.abs(freq_dev))
        
        # Peak angle swing
        delta_swing = np.max(delta) - np.min(delta)
        
        # Damping (check if oscillations decay)
        # Simple check: compare first peak to last peak
        peaks_indices = []
        for i in range(1, len(freq_dev)-1):
            if abs(freq_dev[i]) > abs(freq_dev[i-1]) and abs(freq_dev[i]) > abs(freq_dev[i+1]):
                peaks_indices.append(i)
        
        damping_ratio = 0.0
        if len(peaks_indices) >= 2:
            first_peak = abs(freq_dev[peaks_indices[0]])
            last_peak = abs(freq_dev[peaks_indices[-1]])
            if first_peak > 0:
                damping_ratio = (first_peak - last_peak) / first_peak
        
        # Stability criteria (simplified)
        is_stable = (max_freq_dev < 0.5) and (delta_swing < 180)  # Less than 180 deg swing
        
        metrics[gen_name] = {
            'max_freq_deviation_Hz': max_freq_dev,
            'angle_swing_deg': delta_swing,
            'damping_ratio': damping_ratio,
            'stable': is_stable
        }
    
    return metrics


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("  RMS Dynamic Simulation Demo - IEEE 39-Bus System")
    print("="*70 + "\n")
    
    # Paths
    base_dir = Path(__file__).parent
    h5_file = base_dir / "graph_model" / "Graph_model.h5"
    
    if not h5_file.exists():
        print(f"❌ Graph_model.h5 not found at {h5_file}")
        print("   Run graph_exporter_demo.py first to generate it.")
        return
    
    # Create output directory
    output_dir = base_dir / "RMS_Analysis" / "rms_plots"
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # Scenario 1: No fault (verify initialization)
    # ========================================================================
    print("\n" + "-"*70)
    print("Scenario 1: Steady-State (No Fault)")
    print("-"*70)
    
    sim = RMSSimulator(str(h5_file), integrator_method='rk4', dt=0.01)
    results = sim.simulate(t_end=2.0)
    
    plot_frequencies(results, "Scenario 1: Frequencies (No Fault)")
    plt.savefig(output_dir / "scenario1_frequencies.png", dpi=150)
    plt.close()
    
    # ========================================================================
    # Scenario 2: 3-phase fault at bus 16
    # ========================================================================
    print("\n" + "-"*70)
    print("Scenario 2: 3-Phase Fault at Bus 16")
    print("-"*70)
    
    sim = RMSSimulator(str(h5_file), integrator_method='rk4', dt=0.005)
    
    # Fault event: starts at 1.0s, cleared at 1.1s
    fault = FaultEvent(
        t_start=1.0,
        t_clear=1.1,
        bus_id=16,
        fault_type='3phase'
    )
    
    results = sim.simulate(t_end=10.0, fault_events=[fault])
    
    # Plot results
    plot_rotor_angles(results, "Scenario 2: Rotor Angles (3-Phase Fault)")
    plt.savefig(output_dir / "scenario2_rotor_angles.png", dpi=150)
    plt.close()
    
    plot_frequencies(results, "Scenario 2: Frequencies (3-Phase Fault)")
    plt.savefig(output_dir / "scenario2_frequencies.png", dpi=150)
    plt.close()
    
    plot_rotor_angle_differences(results, "Scenario 2: Angle Differences (3-Phase Fault)")
    plt.savefig(output_dir / "scenario2_angle_differences.png", dpi=150)
    plt.close()
    
    # Stability analysis
    print("\n" + "-"*70)
    print("Stability Analysis")
    print("-"*70)
    
    metrics = analyze_stability(results)
    
    all_stable = True
    for gen_name, m in metrics.items():
        status = "[STABLE]" if m['stable'] else "[UNSTABLE]"
        print(f"{gen_name:20s}: {status}")
        print(f"  Max Freq Dev: {m['max_freq_deviation_Hz']:.3f} Hz")
        print(f"  Angle Swing:  {m['angle_swing_deg']:.1f} deg")
        print(f"  Damping:      {m['damping_ratio']*100:.1f}%")
        
        all_stable = all_stable and m['stable']
    
    print("\n" + "="*70)
    if all_stable:
        print("  [OK] SYSTEM STABLE - All generators maintained synchronism")
    else:
        print("  [WARN] SYSTEM UNSTABLE - Some generators lost synchronism")
    print("="*70 + "\n")
    
    print(f"[>] Plots saved to: {output_dir}")
    print("\nDone! [OK]")


if __name__ == "__main__":
    main()
