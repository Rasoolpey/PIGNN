"""Test short RMS simulation with DAE solver."""

from RMS_Analysis.rms_simulator_dae import RMSSimulator
import numpy as np
import matplotlib.pyplot as plt

# Create simulator
sim = RMSSimulator(h5_file='graph_model/Graph_model.h5', dt=0.005)
sim.initialize()

print("\nInitial residuals:")
sim._update_dae_equations(sim.dae, 0.0)
print(f"  Differential: max|f| = {np.max(np.abs(sim.dae.f)):.6e}")
print(f"  Algebraic:    max|g| = {np.max(np.abs(sim.dae.g)):.6e}")
print()

# Run short simulation (0.1 seconds = 20 steps at 5ms)
print("Running 0.1s simulation (20 steps)...")
results = sim.simulate(t_end=0.1)

print("\n[OK] Simulation completed successfully!")
print(f"Generated {len(results['time'])} time points")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot rotor angles
ax = axes[0]
for gen_name in ['G 01', 'G 02', 'G 05']:
    ax.plot(results['time'], results['generators'][gen_name]['delta_deg'], 
            label=gen_name, linewidth=2)
ax.set_ylabel('Rotor Angle (deg)')
ax.legend()
ax.grid(True)
ax.set_title('RMS DAE Simulation - IEEE 39 Bus System')

# Plot frequencies
ax = axes[1]
for gen_name in ['G 01', 'G 02', 'G 05']:
    ax.plot(results['time'], results['generators'][gen_name]['freq_Hz'], 
            label=gen_name, linewidth=2)
ax.set_ylabel('Frequency (Hz)')
ax.axhline(60.0, color='k', linestyle='--', alpha=0.3)
ax.legend()
ax.grid(True)

# Plot speed deviations
ax = axes[2]
for gen_name in ['G 01', 'G 02', 'G 05']:
    ax.plot(results['time'], results['generators'][gen_name]['omega_pu']*100, 
            label=gen_name, linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed Deviation (%)')
ax.axhline(0.0, color='k', linestyle='--', alpha=0.3)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('plots/rms_dae_test.png', dpi=150)
print(f"\n[OK] Plot saved to: plots/rms_dae_test.png")

plt.show()
