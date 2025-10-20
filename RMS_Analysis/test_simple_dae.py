"""
Simplest possible DAE test: Single generator with fixed voltage.

NOW USING CORRECT INITIALIZATION from power flow!

This tests ONLY the DAE solver mathematics, not network physics.

System:
- Differential: delta, omega (rotor angle and speed)
- Algebraic: V, theta (terminal voltage - FIXED for simplicity)

Equations:
- d(delta)/dt = omega_base * (omega - 1)
- 2H * d(omega)/dt = Pm - Pe - D*(omega - 1)
- g[0]: V = V_fixed  (algebraic constraint)
- g[1]: theta = theta_fixed (algebraic constraint)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dae_system import DAESystem

print("="*70)
print("SIMPLE DAE TEST: Swing Equation with CORRECT Initialization")
print("="*70)

# System parameters
omega_base = 2 * np.pi * 60  # rad/s
H = 4.0  # Inertia (seconds)
D = 0.0  # Damping = 0 for pure steady state test
Pm_target = 0.75  # Mechanical power (pu) - target
V_fixed = 1.0  # Terminal voltage (pu)
theta_fixed = 0.0  # Terminal angle (rad)
Xd_prime = 0.3  # d-axis transient reactance
Ra = 0.003  # Armature resistance

# ============================================================
# CORRECT INITIALIZATION (following ANDES and our new code!)
# ============================================================
print("\n1. Computing Steady-State Initial Conditions...")
print("-" * 70)

# Use complex phasor method
P_gen = Pm_target
Q_gen = 0.3  # Reactive power (pu)

V_phasor = V_fixed * np.exp(1j * theta_fixed)
S_complex = P_gen + 1j * Q_gen
I_phasor = np.conj(S_complex / V_phasor)

# Internal voltage
Z_prime = Ra + 1j * Xd_prime
E_prime_phasor = V_phasor + Z_prime * I_phasor

# Rotor angle (CORRECT!)
delta_ss = np.angle(E_prime_phasor)

# Transform to dq
V_dq = V_phasor * np.exp(-1j * delta_ss)
I_dq = I_phasor * np.exp(-1j * delta_ss)

Vd = V_dq.imag
Vq = V_dq.real
Id = I_dq.imag
Iq = I_dq.real

# Electrical power
Pe_ss = Vd * Id + Vq * Iq

print(f"Steady-State Solution:")
print(f"  delta = {delta_ss:.4f} rad ({np.degrees(delta_ss):.2f}°)")
print(f"  omega = 1.0000 pu (synchronous)")
print(f"  P_gen = {P_gen:.4f} pu")
print(f"  Pe_elec = {Pe_ss:.4f} pu")
print(f"  Power balance error: {abs(P_gen - Pe_ss):.2e}")
print(f"  Vd = {Vd:.4f}, Vq = {Vq:.4f}")
print(f"  Id = {Id:.4f}, Iq = {Iq:.4f}")

# Create DAE
dae = DAESystem()
gen_x, gen_y = dae.request_address("gen", n_diff=2, n_alg=2)
dae.allocate_arrays(dae.n, dae.m)

# Time constants
dae.Tf[0] = 1.0   # delta equation
dae.Tf[1] = 2*H   # swing equation
dae._update_Teye()

print(f"\n1. System: n={dae.n} differential, m={dae.m} algebraic")
print(f"   Tf = {dae.Tf}")

# Initial conditions (near equilibrium)
delta_0 = 0.4  # rad (~23 degrees)
omega_0 = 1.0  # pu (synchronous)
dae.x[0] = delta_0
dae.x[1] = omega_0
dae.y[0] = V_fixed
dae.y[1] = theta_fixed

# Create DAE
dae = DAESystem()
gen_x, gen_y = dae.request_address("gen", n_diff=2, n_alg=2)
dae.allocate_arrays(dae.n, dae.m)

# Time constants
dae.Tf[0] = 1.0   # delta equation
dae.Tf[1] = 2*H   # swing equation
dae._update_Teye()

print(f"\n2. DAE System Setup:")
print(f"   n={dae.n} differential, m={dae.m} algebraic")
print(f"   Tf = {dae.Tf}")

# Initial conditions (FROM STEADY STATE!)
dae.x[0] = delta_ss
dae.x[1] = 1.0  # omega = 1.0 pu (synchronous)
dae.y[0] = V_fixed
dae.y[1] = theta_fixed

print(f"\n3. Initial State:")
print(f"   delta = {dae.x[0]:.4f} rad ({np.degrees(dae.x[0]):.1f}°)")
print(f"   omega = {dae.x[1]:.4f} pu")
print(f"   V = {dae.y[0]:.4f} pu")
print(f"   theta = {dae.y[1]:.4f} rad")

# Define equations
def update_equations(dae):
    delta = dae.x[0]
    omega = dae.x[1]
    V = dae.y[0]
    theta = dae.y[1]
    
    # Electrical power (using dq values)
    # Need to recompute dq from current delta
    V_phasor_now = V * np.exp(1j * theta)
    V_dq_now = V_phasor_now * np.exp(-1j * delta)
    
    # Use steady-state current (assumes fixed current for this test)
    I_phasor_now = I_phasor  # Keep current constant
    I_dq_now = I_phasor_now * np.exp(-1j * delta)
    
    Vd_now = V_dq_now.imag
    Vq_now = V_dq_now.real
    Id_now = I_dq_now.imag
    Iq_now = I_dq_now.real
    
    Pe = Vd_now * Id_now + Vq_now * Iq_now
    
    # Differential equations
    dae.f[0] = omega_base * (omega - 1.0)
    dae.f[1] = Pm_target - Pe - D * (omega - 1.0)
    
    # Algebraic constraints
    dae.g[0] = V - V_fixed
    dae.g[1] = theta - theta_fixed
    
    return Pe

def update_jacobians(dae):
    delta = dae.x[0]
    V = dae.y[0]
    theta = dae.y[1]
    
    # For this test, use numerical derivatives (simplified)
    # In real code, we'd compute analytical derivatives of Pe w.r.t. delta
    
    # Approximate derivatives (since Pe depends on delta through rotation)
    eps = 1e-6
    Pe_0 = Pe_ss  # Use steady-state value as approximation
    dPe_ddelta = -0.5  # Approximate linearization
    
    # fx = ∂f/∂x [2×2]
    dae.fx = sparse.csr_matrix([
        [0.0, omega_base],
        [dPe_ddelta, -D]
    ], shape=(2, 2))
    
    # fy = ∂f/∂y [2×2] - Pe doesn't depend on V, theta strongly in this test
    dae.fy = sparse.csr_matrix([
        [0.0, 0.0],
        [0.0, 0.0]
    ], shape=(2, 2))
    
    # gx = ∂g/∂x [2×2]
    dae.gx = sparse.csr_matrix((2, 2))  # All zeros
    
    # gy = ∂g/∂y [2×2]
    dae.gy = sparse.csr_matrix([
        [1.0, 0.0],  # ∂g0/∂V = 1
        [0.0, 1.0]   # ∂g1/∂theta = 1
    ], shape=(2, 2))

# Evaluate initial
Pe = update_equations(dae)
update_jacobians(dae)

print(f"\n3. Initial Equations:")
print(f"   f[0] (d(delta)/dt) = {dae.f[0]:.6f}")
print(f"   f[1] (swing) = {dae.f[1]:.6f}")
print(f"   g[0] (V constraint) = {dae.g[0]:.6f}")
print(f"   g[1] (theta constraint) = {dae.g[1]:.6f}")
print(f"   Pe = {Pe:.6f} pu")

# Time step
h = 0.001  # 1 ms (smaller step)
x0 = dae.x.copy()
f0 = dae.f.copy()

# Build Jacobian
g_scale = 0.0  # Don't scale algebraic equations
print(f"\n4. Building Augmented Jacobian (h={h}s, g_scale={g_scale})...")
Ac = dae.build_jacobian_ac(h, config_g_scale=g_scale)
print(f"   Ac shape: {Ac.shape}, nnz: {Ac.nnz}")
print(f"   Ac matrix:")
print(Ac.toarray())
print(f"   Condition number: {np.linalg.cond(Ac.toarray()):.2e}")

# Compute residual
q = dae.calc_residual_q(h, x0, f0, config_g_scale=g_scale)
print(f"\n5. Residual:")
print(f"   q = {q}")

# Solve
print(f"\n6. Solving Ac * inc = -q...")
inc = spsolve(Ac, -q)
print(f"   inc = {inc}")

# Update
dae.update_states(inc)
print(f"\n7. Updated State:")
print(f"   delta = {dae.x[0]:.6f} rad ({np.degrees(dae.x[0]):.2f}°)")
print(f"   omega = {dae.x[1]:.6f} pu")
print(f"   V = {dae.y[0]:.6f} pu")
print(f"   theta = {dae.y[1]:.6f} rad")

# Re-evaluate
Pe_new = update_equations(dae)
print(f"\n8. Updated Equations:")
print(f"   f[0] = {dae.f[0]:.6f}")
print(f"   f[1] = {dae.f[1]:.6f}")
print(f"   g[0] = {dae.g[0]:.6f}")
print(f"   g[1] = {dae.g[1]:.6f}")
print(f"   Pe = {Pe_new:.6f} pu")

# Physics check
print(f"\n9. Physics Validation:")
print(f"   Pm = {Pm_target:.4f} pu")
print(f"   Pe = {Pe_new:.4f} pu")
print(f"   Damping = {D * (dae.x[1] - 1):.4f} pu")
print(f"   Acceleration = {(Pm_target - Pe_new - D*(dae.x[1]-1))/(2*H):.6f} pu/s")

# Do multiple Newton iterations to converge
# The key: x0 and f0 are from the PREVIOUS TIME STEP (not current iteration)
# Within Newton, we iterate on the CURRENT time step states
print(f"\n10. Newton Iterations to Convergence:")

# Prediction: use explicit Euler for initial guess
dae.x[0] = x0[0] + h * f0[0]  # delta prediction
dae.x[1] = x0[1] + h * f0[1] / (2*H)  # omega prediction

alpha = 0.5  # Relaxation factor (damped Newton)
for i in range(20):
    # Re-evaluate equations with current states
    update_equations(dae)
    update_jacobians(dae)
    
    # Build Jacobian
    Ac = dae.build_jacobian_ac(h, config_g_scale=g_scale)
    
    # Compute residual - x0/f0 remain FIXED from previous timestep
    q = dae.calc_residual_q(h, x0, f0, config_g_scale=g_scale)
    
    residual_norm = np.linalg.norm(q)
    inc_delta = -q[0]
    inc_omega = -q[1]
    print(f"   Iter {i+1}: ||q|| = {residual_norm:.6e}, delta={dae.x[0]:.6f}, omega={dae.x[1]:.6f}")
    
    if residual_norm < 1e-6:
        print(f"   ✅ Converged in {i+1} iterations!")
        break
    
    # Solve and update with relaxation
    inc = spsolve(Ac, -q)
    dae.update_states(alpha * inc)  # Damped step

print("\n" + "="*70)
