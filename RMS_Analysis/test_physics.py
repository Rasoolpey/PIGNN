"""
Physical validation of DAE system with simple swing equation.

We'll test with a 2-bus system with 1 generator to verify:
1. The Jacobian structure is correct
2. The solver converges to the right answer
3. The physics makes sense

System:
- Bus 1: Generator (swing bus)
- Bus 2: Load  
- Line connecting them

States:
- x = [delta, omega] (generator rotor angle and speed)
- y = [V1, theta1, V2, theta2] (bus voltages and angles)

Equations:
- Swing: 2H*d(omega)/dt = Pm - Pe - D*omega
- Angle: d(delta)/dt = omega_base * (omega - 1)
- Network: P_gen - P_load - P_flow = 0 (at each bus)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dae_system import DAESystem

print("="*70)
print("Physical Validation Test: Simple Swing Equation")
print("="*70)

# System parameters
omega_base = 2 * np.pi * 60  # rad/s
H = 4.0  # Inertia constant (seconds)
D = 2.0  # Damping
Pm = 0.8  # Mechanical power (pu)
Pload = 0.7  # Load power (pu)

# Create DAE system
dae = DAESystem()

# Allocate addresses
# Generator: 2 differential states (delta, omega)
gen_x, gen_y = dae.request_address("gen", n_diff=2, n_alg=0)
# Network: 4 algebraic states (V1, theta1, V2, theta2)
net_x, net_y = dae.request_address("network", n_diff=0, n_alg=4)

print(f"\n1. System Structure:")
print(f"   Differential states (x): {dae.n}")
print(f"     - delta (rotor angle): index {gen_x[0]}")
print(f"     - omega (rotor speed): index {gen_x[1]}")
print(f"   Algebraic states (y): {dae.m}")
print(f"     - V1: index {net_y[0]}")
print(f"     - theta1: index {net_y[1]}")
print(f"     - V2: index {net_y[2]}")
print(f"     - theta2: index {net_y[3]}")

# Allocate arrays
dae.allocate_arrays(dae.n, dae.m)

# Set time constants
dae.Tf[gen_x[0]] = 1.0  # d(delta)/dt equation
dae.Tf[gen_x[1]] = 2*H  # Swing equation: 2H * d(omega)/dt
dae._update_Teye()

print(f"\n2. Time Constants:")
print(f"   Tf = {dae.Tf}")
print(f"   Teye diagonal = {dae.Teye.diagonal()}")

# Initial conditions (near steady state)
dae.x[gen_x[0]] = 0.2  # delta = 0.2 rad (~11 degrees)
dae.x[gen_x[1]] = 1.0  # omega = 1.0 pu (synchronous)
dae.y[net_y[0]] = 1.0  # V1 = 1.0 pu
dae.y[net_y[1]] = 0.0  # theta1 = 0 (reference)
dae.y[net_y[2]] = 0.95 # V2 = 0.95 pu
dae.y[net_y[3]] = -0.1 # theta2 = -0.1 rad

print(f"\n3. Initial Conditions:")
print(f"   delta = {dae.x[gen_x[0]]:.4f} rad ({np.degrees(dae.x[gen_x[0]]):.2f}°)")
print(f"   omega = {dae.x[gen_x[1]]:.4f} pu")
print(f"   V1 = {dae.y[net_y[0]]:.4f} pu, theta1 = {dae.y[net_y[1]]:.4f} rad")
print(f"   V2 = {dae.y[net_y[2]]:.4f} pu, theta2 = {dae.y[net_y[3]]:.4f} rad")

# Define equations
def update_equations(dae, delta, omega, V1, theta1, V2, theta2):
    """
    Update f and g vectors with current states.
    """
    # Calculate electrical power from generator
    # Pe = V1 * V2 * Y * sin(delta + theta1 - theta2)
    Y = 10.0  # Line admittance (pu)
    Pe = V1 * V2 * Y * np.sin(delta + theta1 - theta2)
    
    # Differential equations
    # f[0]: d(delta)/dt = omega_base * (omega - 1)
    dae.f[0] = omega_base * (omega - 1.0)
    
    # f[1]: 2H * d(omega)/dt = Pm - Pe - D*(omega - 1)
    dae.f[1] = Pm - Pe - D * (omega - 1.0)
    
    # Algebraic equations (power balance)
    # g[0]: P_gen - P_flow1 = 0
    P_flow1 = V1 * V2 * Y * np.sin(theta1 - theta2)
    dae.g[0] = Pe - P_flow1
    
    # g[1]: Fix theta1 = 0 (reference bus)
    dae.g[1] = theta1
    
    # g[2]: -P_load + P_flow2 = 0
    P_flow2 = -P_flow1  # Conservation
    dae.g[2] = -Pload + P_flow2
    
    # g[3]: Fix V2 = 0.95 (PV bus or controlled)
    dae.g[3] = V2 - 0.95

# Compute Jacobians analytically
def update_jacobians(dae, delta, omega, V1, theta1, V2, theta2):
    """
    Compute analytical Jacobians.
    """
    Y = 10.0
    
    # fx = ∂f/∂x [n×n] = [2×2]
    # ∂f/∂[delta, omega]
    dPe_ddelta = V1 * V2 * Y * np.cos(delta + theta1 - theta2)
    
    fx_data = [
        0.0,                  # ∂f0/∂delta = 0
        omega_base,           # ∂f0/∂omega = omega_base
        -dPe_ddelta,          # ∂f1/∂delta = -dPe/ddelta
        -D                    # ∂f1/∂omega = -D
    ]
    fx_row = [0, 0, 1, 1]
    fx_col = [0, 1, 0, 1]
    dae.fx = sparse.csr_matrix((fx_data, (fx_row, fx_col)), shape=(dae.n, dae.n))
    
    # fy = ∂f/∂y [n×m] = [2×4]
    # ∂f/∂[V1, theta1, V2, theta2]
    dPe_dV1 = V2 * Y * np.sin(delta + theta1 - theta2)
    dPe_dtheta1 = V1 * V2 * Y * np.cos(delta + theta1 - theta2)
    dPe_dV2 = V1 * Y * np.sin(delta + theta1 - theta2)
    dPe_dtheta2 = -V1 * V2 * Y * np.cos(delta + theta1 - theta2)
    
    fy_data = [
        -dPe_dV1,      # ∂f1/∂V1
        -dPe_dtheta1,  # ∂f1/∂theta1
        -dPe_dV2,      # ∂f1/∂V2
        -dPe_dtheta2   # ∂f1/∂theta2
    ]
    fy_row = [1, 1, 1, 1]
    fy_col = [0, 1, 2, 3]
    dae.fy = sparse.csr_matrix((fy_data, (fy_row, fy_col)), shape=(dae.n, dae.m))
    
    # gx = ∂g/∂x [m×n] = [4×2]
    # ∂g/∂[delta, omega]
    gx_data = [
        dPe_ddelta,    # ∂g0/∂delta = dPe/ddelta
    ]
    gx_row = [0]
    gx_col = [0]
    dae.gx = sparse.csr_matrix((gx_data, (gx_row, gx_col)), shape=(dae.m, dae.n))
    
    # gy = ∂g/∂y [m×m] = [4×4]
    # ∂g/∂[V1, theta1, V2, theta2]
    dP1_dV1 = V2 * Y * np.sin(theta1 - theta2)
    dP1_dtheta1 = V1 * V2 * Y * np.cos(theta1 - theta2)
    dP1_dV2 = V1 * Y * np.sin(theta1 - theta2)
    dP1_dtheta2 = -V1 * V2 * Y * np.cos(theta1 - theta2)
    
    gy_data = [
        -dP1_dV1,      # ∂g0/∂V1
        -dP1_dtheta1,  # ∂g0/∂theta1
        -dP1_dV2,      # ∂g0/∂V2
        -dP1_dtheta2,  # ∂g0/∂theta2
        1.0,           # ∂g1/∂theta1 = 1 (fix theta1=0)
        dP1_dV1,       # ∂g2/∂V1 = -∂g0/∂V1
        dP1_dtheta1,   # ∂g2/∂theta1
        dP1_dV2,       # ∂g2/∂V2
        dP1_dtheta2,   # ∂g2/∂theta2
        1.0            # ∂g3/∂V2 = 1 (fix V2=0.95)
    ]
    gy_row = [0, 0, 0, 0, 1, 2, 2, 2, 2, 3]
    gy_col = [0, 1, 2, 3, 1, 0, 1, 2, 3, 2]
    dae.gy = sparse.csr_matrix((gy_data, (gy_row, gy_col)), shape=(dae.m, dae.m))

# Update with initial conditions
update_equations(dae, dae.x[0], dae.x[1], dae.y[0], dae.y[1], dae.y[2], dae.y[3])
update_jacobians(dae, dae.x[0], dae.x[1], dae.y[0], dae.y[1], dae.y[2], dae.y[3])

print(f"\n4. Initial Equations:")
print(f"   f[0] (d(delta)/dt) = {dae.f[0]:.6f}")
print(f"   f[1] (swing eq) = {dae.f[1]:.6f}")
print(f"   g[0] (P balance bus 1) = {dae.g[0]:.6f}")
print(f"   g[2] (P balance bus 2) = {dae.g[2]:.6f}")

# Test trapezoidal step
h = 0.01  # 10 ms
x0 = dae.x.copy()
f0 = dae.f.copy()

print(f"\n5. Building Augmented Jacobian for h={h}s...")
Ac = dae.build_jacobian_ac(h, config_g_scale=0.0)  # No scaling
print(f"   Ac shape: {Ac.shape}")
print(f"   Ac nnz: {Ac.nnz}")

# Check individual Jacobian blocks
print(f"\n   Individual Jacobians:")
print(f"   fx shape: {dae.fx.shape}, nnz: {dae.fx.nnz}")
print(f"   fy shape: {dae.fy.shape}, nnz: {dae.fy.nnz}")
print(f"   gx shape: {dae.gx.shape}, nnz: {dae.gx.nnz}")
print(f"   gy shape: {dae.gy.shape}, nnz: {dae.gy.nnz}")
print(f"\n   Ac as dense matrix:")
print(Ac.toarray())
print(f"\n   Condition number: {np.linalg.cond(Ac.toarray())}")

# Compute residual
q = dae.calc_residual_q(h, x0, f0, config_g_scale=0.0)
print(f"\n6. Residual q:")
print(f"   q[:n] (differential): {q[:dae.n]}")
print(f"   q[n:] (algebraic): {q[dae.n:]}")

# Solve for increment
print(f"\n7. Solving Ac * inc = -q...")
inc = spsolve(Ac, -q)
print(f"   inc[:n] (Δx): {inc[:dae.n]}")
print(f"   inc[n:] (Δy): {inc[dae.n:]}")

# Update states
dae.update_states(inc)
print(f"\n8. Updated States:")
print(f"   delta = {dae.x[0]:.6f} rad ({np.degrees(dae.x[0]):.2f}°)")
print(f"   omega = {dae.x[1]:.6f} pu")
print(f"   V1 = {dae.y[0]:.6f} pu, theta1 = {dae.y[1]:.6f} rad")
print(f"   V2 = {dae.y[2]:.6f} pu, theta2 = {dae.y[3]:.6f} rad")

# Re-evaluate equations to check convergence
update_equations(dae, dae.x[0], dae.x[1], dae.y[0], dae.y[1], dae.y[2], dae.y[3])
print(f"\n9. Updated Equations (should be smaller):")
print(f"   f[0] = {dae.f[0]:.6f}")
print(f"   f[1] = {dae.f[1]:.6f}")
print(f"   g[0] = {dae.g[0]:.6f}")
print(f"   g[2] = {dae.g[2]:.6f}")

# Physics check
print(f"\n10. Physics Validation:")
Pe = dae.y[0] * dae.y[2] * 10.0 * np.sin(dae.x[0] + dae.y[1] - dae.y[3])
print(f"   Mechanical power Pm = {Pm:.4f} pu")
print(f"   Electrical power Pe = {Pe:.4f} pu")
print(f"   Power balance (Pm - Pe - D*(omega-1)) = {Pm - Pe - D*(dae.x[1]-1):.6f}")
print(f"   Speed deviation (omega - 1) = {dae.x[1] - 1:.6f} pu")

if abs(Pm - Pe) < 0.1:
    print(f"\n✅ Physics check PASSED: Pe ≈ Pm (balanced)")
else:
    print(f"\n⚠️ Physics check: Large power imbalance!")

print("\n" + "="*70)
