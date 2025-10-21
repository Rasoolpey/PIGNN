"""Test 1-second RMS simulation to observe system dynamics."""

from RMS_Analysis.rms_simulator_dae import RMSSimulator
import numpy as np
import matplotlib.pyplot as plt
import time as pytime

print("="*70)
print("RMS DAE SIMULATION - 1 SECOND TEST")
print("="*70)

# Create simulator
sim = RMSSimulator(h5_file='graph_model/Graph_model.h5', dt=0.005)
sim.initialize()

print("\nInitial residuals:")
sim._update_dae_equations(sim.dae, 0.0)
print(f"  Differential: max|f| = {np.max(np.abs(sim.dae.f)):.6e}")
print(f"  Algebraic:    max|g| = {np.max(np.abs(sim.dae.g)):.6e}")

# Check initial generator states
print("\nInitial generator states:")
for gen_name in ['G 01', 'G 02', 'G 05']:
    gen = sim.generators[gen_name]
    delta_deg = np.rad2deg(gen.states[0])
    omega_pu = gen.states[1] / sim.ws
    print(f"  {gen_name}: delta={delta_deg:6.2f}°, omega={omega_pu:+.6f} pu")

# Run 1-second simulation (200 steps at 5ms)
print("\n" + "="*70)
print("Running 1.0s simulation (200 steps)...")
print("="*70)

start_time = pytime.time()
results = sim.simulate(t_end=1.0)
elapsed = pytime.time() - start_time

print(f"\n[OK] Simulation completed in {elapsed:.2f}s ({200/elapsed:.0f} steps/s)")
print(f"Generated {len(results['time'])} time points")

# Analyze final states
print("\nFinal generator states:")
for gen_name in ['G 01', 'G 02', 'G 05']:
    delta_final = results['generators'][gen_name]['delta_deg'][-1]
    omega_final = results['generators'][gen_name]['omega_pu'][-1]
    freq_final = results['generators'][gen_name]['freq_Hz'][-1]
    print(f"  {gen_name}: delta={delta_final:6.2f}°, omega={omega_final:+.6f} pu, f={freq_final:.6f} Hz")

# Check if system is stable (frequency should stay at 60 Hz)
print("\nStability check:")
for gen_name in sim.gen_names:
    freq = results['generators'][gen_name]['freq_Hz']
    freq_dev = np.max(np.abs(freq - 60.0))
    if freq_dev < 0.01:
        status = "[OK]"
    elif freq_dev < 0.1:
        status = "[WARN]"
    else:
        status = "[ERROR]"
    print(f"  {status} {gen_name}: max freq deviation = {freq_dev:.6f} Hz")

# Create comprehensive plots
fig = plt.figure(figsize=(14, 10))

# Plot 1: Rotor angles (all generators)
ax1 = plt.subplot(3, 2, 1)
for gen_name in sim.gen_names:
    ax1.plot(results['time'], results['generators'][gen_name]['delta_deg'], 
             label=gen_name, linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Rotor Angle (deg)')
ax1.legend(ncol=2, fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_title('Rotor Angles - All Generators')

# Plot 2: Rotor angles (selected generators)
ax2 = plt.subplot(3, 2, 2)
for gen_name in ['G 01', 'G 02', 'G 05', 'G 10']:
    ax2.plot(results['time'], results['generators'][gen_name]['delta_deg'], 
             label=gen_name, linewidth=2)
ax2.set_ylabel('Rotor Angle (deg)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Rotor Angles - Selected Generators')

# Plot 3: Frequencies (all generators)
ax3 = plt.subplot(3, 2, 3)
for gen_name in sim.gen_names:
    ax3.plot(results['time'], results['generators'][gen_name]['freq_Hz'], 
             linewidth=1.5, alpha=0.8)
ax3.axhline(60.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='60 Hz')
ax3.set_ylabel('Frequency (Hz)')
ax3.grid(True, alpha=0.3)
ax3.set_title('Frequencies - All Generators')
ax3.set_ylim([59.95, 60.05])

# Plot 4: Frequency deviations (selected)
ax4 = plt.subplot(3, 2, 4)
for gen_name in ['G 01', 'G 02', 'G 05', 'G 10']:
    freq_dev = (results['generators'][gen_name]['freq_Hz'] - 60.0) * 1000  # mHz
    ax4.plot(results['time'], freq_dev, label=gen_name, linewidth=2)
ax4.axhline(0, color='k', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_ylabel('Frequency Deviation (mHz)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_title('Frequency Deviations - Selected Generators')

# Plot 5: Speed deviations (all generators)
ax5 = plt.subplot(3, 2, 5)
for gen_name in sim.gen_names:
    omega_pct = results['generators'][gen_name]['omega_pu'] * 100
    ax5.plot(results['time'], omega_pct, linewidth=1.5, alpha=0.8)
ax5.axhline(0.0, color='k', linestyle='--', linewidth=2, alpha=0.5)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Speed Deviation (%)')
ax5.grid(True, alpha=0.3)
ax5.set_title('Speed Deviations - All Generators')

# Plot 6: Angle differences (coherency check)
ax6 = plt.subplot(3, 2, 6)
# Plot relative angles to slack bus (G 02)
ref_angle = results['generators']['G 02']['delta_deg']
for gen_name in ['G 01', 'G 05', 'G 10']:
    angle_diff = results['generators'][gen_name]['delta_deg'] - ref_angle
    ax6.plot(results['time'], angle_diff, label=f"{gen_name} - G 02", linewidth=2)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Angle Difference (deg)')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_title('Relative Rotor Angles (vs G 02)')

plt.suptitle('RMS DAE Simulation - IEEE 39 Bus System (1.0s)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/rms_dae_1sec.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] Plot saved to: plots/rms_dae_1sec.png")

# Statistical analysis
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

for gen_name in ['G 01', 'G 02', 'G 05']:
    delta = results['generators'][gen_name]['delta_deg']
    omega = results['generators'][gen_name]['omega_pu']
    freq = results['generators'][gen_name]['freq_Hz']
    
    print(f"\n{gen_name}:")
    print(f"  Delta:  mean={np.mean(delta):6.2f}°, std={np.std(delta):.4f}°, range=[{np.min(delta):.2f}, {np.max(delta):.2f}]")
    print(f"  Omega:  mean={np.mean(omega):+.6f} pu, std={np.std(omega):.6f} pu")
    print(f"  Freq:   mean={np.mean(freq):.6f} Hz, std={np.std(freq):.6f} Hz")

print("\n" + "="*70)
print("[OK] ANALYSIS COMPLETE")
print("="*70)

plt.show()
