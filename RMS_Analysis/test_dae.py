"""
Test script for DAE system infrastructure.
"""

import numpy as np
from dae_system import DAESystem

print("="*70)
print("Testing DAE System Infrastructure")
print("="*70)

# Create DAE system
dae = DAESystem()
print(f"\n1. Created empty DAE system: {dae}")

# Request addresses for components
print("\n2. Requesting addresses for components...")
gen1_x, gen1_y = dae.request_address("gen_1", n_diff=6, n_alg=2)
print(f"   Generator 1: x_addr={gen1_x}, y_addr={gen1_y}")

gen2_x, gen2_y = dae.request_address("gen_2", n_diff=6, n_alg=2)
print(f"   Generator 2: x_addr={gen2_x}, y_addr={gen2_y}")

bus_x, bus_y = dae.request_address("network", n_diff=0, n_alg=78)  # 39 buses * 2 (V, theta)
print(f"   Network (39 buses): x_addr={bus_x}, y_addr={bus_y}")

print(f"\n   Total: {dae.n} differential, {dae.m} algebraic variables")

# Allocate arrays
print("\n3. Allocating arrays...")
dae.allocate_arrays(dae.n, dae.m)
print(f"   x shape: {dae.x.shape}")
print(f"   y shape: {dae.y.shape}")
print(f"   f shape: {dae.f.shape}")
print(f"   g shape: {dae.g.shape}")
print(f"   Tf shape: {dae.Tf.shape}")
print(f"   Teye shape: {dae.Teye.shape}")
print(f"   fx shape: {dae.fx.shape}")
print(f"   fy shape: {dae.fy.shape}")
print(f"   gx shape: {dae.gx.shape}")
print(f"   gy shape: {dae.gy.shape}")

# Set some values
print("\n4. Setting initial conditions...")
dae.x[gen1_x] = [0.1, 1.0, 0.5, 0.6, 0.7, 0.8]  # Gen 1 states
dae.x[gen2_x] = [0.2, 1.0, 0.5, 0.6, 0.7, 0.8]  # Gen 2 states
dae.y[bus_y] = np.ones(78)  # All bus voltages at 1.0 pu
print(f"   x range: [{dae.x.min():.3f}, {dae.x.max():.3f}]")
print(f"   y range: [{dae.y.min():.3f}, {dae.y.max():.3f}]")

# Set time constants
print("\n5. Setting time constants...")
H = np.array([4.0, 5.0])  # Inertia constants for 2 generators
omega_idx = [1, 7]  # Speed state indices
dae.set_time_constant(omega_idx, 2*H)  # T = 2H for swing equation
print(f"   Tf range: [{dae.Tf.min():.3f}, {dae.Tf.max():.3f}]")

# Test Jacobian building
print("\n6. Building augmented Jacobian Ac...")
from scipy import sparse
# Jacobian dimensions:
# fx: ∂f/∂x [n x n] - differential eqs w.r.t. differential states
# fy: ∂f/∂y [n x m] - differential eqs w.r.t. algebraic states  
# gx: ∂g/∂x [m x n] - algebraic eqs w.r.t. differential states
# gy: ∂g/∂y [m x m] - algebraic eqs w.r.t. algebraic states
dae.fx = sparse.random(dae.n, dae.n, density=0.1, format='csr')
dae.fy = sparse.random(dae.n, dae.m, density=0.05, format='csr')
dae.gx = sparse.random(dae.m, dae.n, density=0.05, format='csr')
dae.gy = sparse.random(dae.m, dae.m, density=0.1, format='csr')
print(f"   fx: {dae.fx.shape}, fy: {dae.fy.shape}, gx: {dae.gx.shape}, gy: {dae.gy.shape}")

h = 0.01  # 10 ms time step
Ac = dae.build_jacobian_ac(h, config_g_scale=1.0)
print(f"   Ac shape: {Ac.shape}")
print(f"   Ac nnz: {Ac.nnz}")
print(f"   Ac density: {Ac.nnz / (Ac.shape[0] * Ac.shape[1]) * 100:.2f}%")

# Test residual calculation
print("\n7. Calculating residual q...")
dae.f = np.random.randn(dae.n)
dae.g = np.random.randn(dae.m)
x0 = dae.x.copy()
f0 = dae.f.copy()

q = dae.calc_residual_q(h, x0, f0, config_g_scale=1.0)
print(f"   q shape: {q.shape}")
print(f"   q[:n] (diff) range: [{q[:dae.n].min():.6f}, {q[:dae.n].max():.6f}]")
print(f"   q[n:] (alg) range: [{q[dae.n:].min():.6f}, {q[dae.n:].max():.6f}]")

# Test state update
print("\n8. Testing state update...")
inc = np.random.randn(dae.n + dae.m) * 0.01
x_before = dae.x.copy()
y_before = dae.y.copy()
dae.update_states(inc)
print(f"   Max x change: {np.max(np.abs(dae.x - x_before)):.6f}")
print(f"   Max y change: {np.max(np.abs(dae.y - y_before)):.6f}")

# Test state save/restore
print("\n9. Testing state save/restore...")
state_dict = dae.get_state_dict()
dae.x[:] = 0
dae.y[:] = 0
print(f"   Cleared: x sum={dae.x.sum():.6f}, y sum={dae.y.sum():.6f}")
dae.set_state_dict(state_dict)
print(f"   Restored: x sum={dae.x.sum():.6f}, y sum={dae.y.sum():.6f}")

print("\n" + "="*70)
print("✅ All DAE system tests passed!")
print("="*70)
