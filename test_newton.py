"""Test Newton convergence with current residuals."""

from RMS_Analysis.rms_simulator_dae import RMSSimulator
import numpy as np

sim = RMSSimulator(h5_file='graph_model/Graph_model.h5', dt=0.005)
sim.initialize()

print("Initial residuals:")
sim._update_dae_equations(sim.dae, 0.0)
print(f"  Differential: max|f| = {np.max(np.abs(sim.dae.f)):.6e}")
print(f"  Algebraic:    max|g| = {np.max(np.abs(sim.dae.g)):.6e}")
print()

print("Attempting first time step...")
try:
    converged, iters, resid = sim.step()
    if converged:
        print(f"[OK] [SUCCESS] Time step completed!")
        print(f"   Time: {sim.t:.6f} s")
        print(f"   Newton iterations: {iters}")
        print(f"   Final residual: {resid:.6e}")
    else:
        print(f"[FAILED] Newton did not converge")
        print(f"   Iterations: {iters}")
        print(f"   Final residual: {resid:.6e}")
except Exception as e:
    print(f"[EXCEPTION] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
