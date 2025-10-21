"""Test full coupled DAE system with 39 buses."""

from RMS_Analysis.rms_simulator_dae import RMSSimulator
import numpy as np

print("="*70)
print("TESTING FULL COUPLED DAE (39 buses, NO Kron reduction)")
print("="*70)

# Initialize
sim = RMSSimulator('graph_model/Graph_model.h5', dt=0.005)
sim.initialize()

print(f"\n{'='*70}")
print(f"DAE SYSTEM: {sim.dae.n} diff + {sim.dae.m} alg = {sim.dae.n + sim.dae.m} total")
print(f"{'='*70}")

# Check initial residuals
sim._update_dae_equations(sim.dae, 0.0)
print(f"\nInitial residuals:")
print(f"  ||f|| = {np.linalg.norm(sim.dae.f):.3e} (differential)")
print(f"  ||g|| = {np.linalg.norm(sim.dae.g):.3e} (algebraic)")

# Show first few residuals
print(f"\nFirst 6 differential residuals (Gen 1):")
for i in range(6):
    print(f"  f[{i}] = {sim.dae.f[i]:+.3e}")

print(f"\nFirst 12 algebraic residuals (Bus 30, 31, 32):")
for i in range(12):
    bus = 30 + i//4
    state = ['Vd', 'Vq', 'Id', 'Iq'][i%4]
    print(f"  g[{i:3d}] (Bus {bus:2d} {state}): {sim.dae.g[i]:+.3e}")

# Try simulation
print(f"\n{'='*70}")
print("RUNNING SIMULATION: 0 -> 0.1s")
print(f"{'='*70}\n")

try:
    results = sim.simulate(t_end=0.1)
    
    time = results['time']
    print(f"\n✅ SUCCESS!")
    print(f"   Simulated {len(time)} steps")
    print(f"   Final time: {time[-1]:.3f}s")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
