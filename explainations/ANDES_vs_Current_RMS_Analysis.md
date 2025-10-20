# ANDES vs Current RMS Implementation - Analysis & Improvements

## Problem Identified

Your current RMS simulation shows:
- All generators initialized at **P=0.000 pu** (WRONG!)
- NaN values in simulation results
- Unacceptable stability analysis

## Root Cause Analysis

### 1. **Power Flow Initialization Issue**
**Current Code** (`rms_simulator.py` line 116-121):
```python
# Match generators to buses
self.gen_P = []
self.gen_Q = []
for i, bus_id in enumerate(self.gen_bus_ids):
    bus_idx = np.where(self.bus_ids == bus_id)[0][0]
    # Convert to pu on machine base
    self.gen_P.append(P_gen[bus_idx] / self.gen_Sn_MVA[i])
    self.gen_Q.append(Q_gen[bus_idx] / self.gen_Sn_MVA[i])
```

**Problem**: Uses `P_generation_MW` from nodes which is ZERO if generators aren't producing

**ANDES Approach**: Uses **solved power flow results** (from PFlow routine)
```python
system.dae.y[:len(system.PFlow.y_sol)] = system.PFlow.y_sol
```

### 2. **Differential-Algebraic Equation (DAE) System**

**Current Implementation**: Treats as pure ODE
- Generators: Differential equations only
- Network: **Simplified to constant voltage** (WRONG!)
- No coupling between generator dynamics and network

**ANDES Implementation**: Full DAE system
- **Differential equations** (`x`): Generator states (delta, omega, Eq', Ed', etc.)
- **Algebraic equations** (`y`): Network voltages, currents
- **Coupled solution**: Solves both simultaneously using Newton-Raphson

### 3. **Network Solution**

**Current Code** (rms_simulator.py line 291):
```python
# Get terminal voltage (from network solution)
Vt_mag = self.V_mag[bus_idx]  # Simplified - should be updated
```

**Problem**: Uses **STATIC** power flow voltage, doesn't update during dynamics

**ANDES**: Solves network algebraic equations at each time step
- Builds complete Jacobian: `[[Teye - h*fx, gx], [-h*fy, gy]]`
- Updates bus voltages based on generator injections
- Enforces power balance: `P_gen - P_load - P_flow = 0`

## Key ANDES Features Missing in Current Implementation

### 1. **Proper Initialization from Power Flow**

ANDES initializes from **solved power flow**:
```python
# From genrou.py (lines 80-150)
self._S = ConstService(v_str='p0 - 1j * q0')  # Complex power from PFlow
self._V = ConstService(v_str='v * exp(1j * a)')  # Bus voltage from PFlow
self._It = ConstService(v_str='_S / conj(_V)')  # Terminal current

# Calculate internal voltage
self._Zs = ConstService(v_str='ra + 1j * xd2')
self._Is = ConstService(v_str='_It + _V / _Zs')
self.psi20 = ConstService(v_str='_Is * _Zs')  # Flux linkage

# Calculate rotor angle
self.delta0 = ConstService(...)  # From phasor diagram
```

**What we need**: 
- Load **REAL power flow results** (not zero generation)
- Calculate initial rotor angle from terminal conditions
- Initialize internal voltages correctly

### 2. **DAE Solver with Network Equations**

ANDES solves at each time step:

**Differential Equations** (ITM - Implicit Trapezoid):
```
Tf*(x - x0) - h/2*(f + f0) = 0
```

**Algebraic Equations** (Network):
```
g(x, y) = 0   # Power balance, voltage equations
```

**Combined Jacobian**:
```
Ac = [[Teye - h/2*fx,  gx  ],
      [    -h/2*fy,     gy  ]]
```

**Newton-Raphson Iteration**:
```
Ac * [Δx; Δy] = -[qg_diff; qg_alg]
x_new = x_old - Δx
y_new = y_old - Δy
```

### 3. **Generator-Network Coupling**

ANDES generators inject current into network:
```python
# From genrou.py
self.Id = Algeb(...)  # d-axis current (algebraic variable)
self.Iq = Algeb(...)  # q-axis current (algebraic variable)

# Network sees this as injection
I_injection = Id + 1j*Iq  # in dq frame
I_abc = rotate_to_abc(I_injection, delta)  # Transform to network frame
```

Current implementation: **No injection** - network frozen!

## Recommended Improvements

### **Priority 1: Fix Power Flow Initialization** ⭐⭐⭐
**Impact**: CRITICAL - Without this, all results are meaningless

**What to do**:
1. Run power flow **FIRST** using your existing `load_flow_solver.py`
2. Store results in Graph_model.h5 under `steady_state/power_flow_results`
3. Load REAL P, Q, V, theta for initialization

**Code changes needed**:
```python
# In graph_exporter_demo.py - ADD THIS
from physics.load_flow_solver import run_load_flow

# After building graph
print("Running power flow...")
results = run_load_flow(...)  # Your existing solver

# Save to H5
with h5py.File('graph_model/Graph_model.h5', 'a') as f:
    if 'steady_state/power_flow_results' in f:
        del f['steady_state/power_flow_results']
    
    pf_group = f.create_group('steady_state/power_flow_results')
    pf_group.create_dataset('bus_voltages_pu', data=results['V_mag'])
    pf_group.create_dataset('bus_angles_deg', data=results['V_ang_deg'])
    pf_group.create_dataset('gen_P_MW', data=results['gen_P'])
    pf_group.create_dataset('gen_Q_MVAR', data=results['gen_Q'])
```

```python
# In rms_simulator.py - LOAD POWER FLOW RESULTS
with h5py.File(self.h5_file, 'r') as f:
    if 'steady_state/power_flow_results' not in f:
        raise ValueError("No power flow results in H5! Run load flow first.")
    
    pf = f['steady_state/power_flow_results']
    self.V_mag = pf['bus_voltages_pu'][:]
    self.V_ang = pf['bus_angles_deg'][:] * np.pi / 180
    
    # Get REAL generator outputs
    gen_P_MW = pf['gen_P_MW'][:]
    gen_Q_MVAR = pf['gen_Q_MVAR'][:]
    
    # Convert to pu on machine base
    self.gen_P = gen_P_MW / self.gen_Sn_MVA
    self.gen_Q = gen_Q_MVAR / self.gen_Sn_MVA
```

### **Priority 2: Implement Network Algebraic Equations** ⭐⭐
**Impact**: HIGH - Enables dynamic voltage response

**What to do**:
1. Add bus voltage as **algebraic variables** (`y`)
2. Formulate power balance equations (`g = 0`)
3. Update voltages at each time step

**Conceptual approach**:
```python
# Algebraic equations for each bus
def network_equations(self, V_bus, delta_gen, Id_gen, Iq_gen):
    """
    g(y) = 0 where y = bus voltages
    
    For each bus:
        P_gen - P_load - sum(P_flow) = 0
        Q_gen - Q_load - sum(Q_flow) = 0
    """
    g = np.zeros(2 * n_bus)
    
    for i in range(n_bus):
        # Generator injection (if present)
        if bus_has_gen[i]:
            P_inj = Vd*Id + Vq*Iq  # Rotate from dq to network frame
            Q_inj = Vq*Id - Vd*Iq
        
        # Load
        P_load = ...
        Q_load = ...
        
        # Flow to other buses
        P_flow = sum(V[i] * V[j] * (G[i,j]*cos(θ[i]-θ[j]) + B[i,j]*sin(θ[i]-θ[j])))
        Q_flow = sum(V[i] * V[j] * (G[i,j]*sin(θ[i]-θ[j]) - B[i,j]*cos(θ[i]-θ[j])))
        
        g[2*i] = P_inj - P_load - P_flow
        g[2*i+1] = Q_inj - Q_load - Q_flow
    
    return g
```

### **Priority 3: Use DAE Integrator** ⭐
**Impact**: MEDIUM - Improves accuracy and convergence

Replace simple RK4 with **implicit trapezoidal + Newton-Raphson**:

```python
def dae_step(self, t, x, y):
    """
    Solve DAE system:
        Tf*(x-x0) - h/2*(f(x,y) + f0) = 0  # Differential
        g(x, y) = 0                        # Algebraic
    """
    
    for iter in range(max_iter):
        # Evaluate f and g
        f = self.compute_f(x, y)  # Generator dynamics
        g = self.compute_g(x, y)  # Network equations
        
        # Build Jacobian
        Ac = [[Teye - h/2*fx,  gx  ],
              [   -h/2*fy,     gy  ]]
        
        # Residuals
        qx = Tf*(x - x0) - h/2*(f + f0)
        qy = g
        
        # Solve
        [Δx, Δy] = solve(Ac, -[qx, qy])
        
        # Update
        x -= Δx
        y -= Δy
        
        # Check convergence
        if max(abs(Δx), abs(Δy)) < tol:
            break
    
    return x, y
```

## Implementation Plan

### **Phase 1: Get REAL Initialization** (1-2 hours)
1. ✅ Run power flow using existing solver
2. ✅ Save results to Graph_model.h5
3. ✅ Load REAL P, Q, V, theta in RMS simulator
4. ✅ Verify generators initialize with correct power output

**Expected output**: Initialization shows P=0.5-1.0 pu (not zero!)

### **Phase 2: Add Basic Network Updates** (2-3 hours)
1. ⏳ Create algebraic variable array `y` for bus voltages
2. ⏳ Implement simplified network equations
3. ⏳ Update voltages based on generator power changes

**Expected output**: Voltages change during fault (not constant)

### **Phase 3: Full DAE Solver** (3-4 hours)
1. ⏳ Implement implicit trapezoid with Newton-Raphson
2. ⏳ Build complete Jacobian
3. ⏳ Couple generator dynamics with network

**Expected output**: Stable, accurate simulation matching PowerFactory

## Quick Wins (What You Can Do NOW)

### 1. **Verify Power Flow Data Exists**
```python
import h5py

with h5py.File('graph_model/Graph_model.h5', 'r') as f:
    if 'steady_state/power_flow_results' in f:
        pf = f['steady_state/power_flow_results']
        print("Power flow exists!")
        print(f"Gen P: {pf['gen_P_MW'][:]}")
    else:
        print("NO POWER FLOW - Need to run load_flow_demo.py first!")
```

### 2. **Check What Load Flow Demo Generates**
```bash
python load_flow_demo.py
```

Then check if it saves results to H5 file.

### 3. **Use Existing Load Flow Results**
If `phases/phase_a/nodes/P_generation_MW` is zero, but load flow ran successfully, the results might be elsewhere in the H5 file.

## Summary

**Why Current Results are Unacceptable**:
1. ❌ Generators start at P=0 (no power flow initialization)
2. ❌ Network voltages frozen (no algebraic equations)
3. ❌ No generator-network coupling
4. ❌ Using ODE solver for DAE system

**What ANDES Does Right**:
1. ✅ Initializes from solved power flow
2. ✅ Solves full DAE system (differential + algebraic)
3. ✅ Updates network voltages at each step
4. ✅ Uses implicit integration with Newton-Raphson

**Your Next Step**: 
**Fix initialization FIRST** - Without correct initial conditions, nothing else matters!

Would you like me to start with Phase 1 (power flow initialization)?
