# PyPSA Load Flow Implementation - Key Learnings

## Overview
PyPSA uses a clean, efficient Newton-Raphson approach for AC power flow. Here are the key insights from their implementation that we should adopt.

## Core Concepts

### 1. **Bus Classification**
- **Slack Bus**: V_mag and V_ang are fixed (reference bus)
- **PV Buses**: V_mag and P are fixed, V_ang and Q are variables
- **PQ Buses**: P and Q are fixed, V_mag and V_ang are variables

### 2. **Power Flow Equation**
The fundamental equation is:
```python
mismatch = V * conj(Y * V) - S_specified
```

Where:
- `V`: Complex voltage vector (V_mag * exp(1j * V_ang))
- `Y`: Admittance matrix
- `S_specified`: Specified complex power (P + 1j*Q)

### 3. **Mismatch Vector Construction**
PyPSA builds the mismatch vector (F) as:
```python
if distribute_slack:
    F = r_[real(mismatch)[:],          # All P mismatches
           imag(mismatch)[1+len(pvs):]]  # Q mismatches for PQ buses only
else:
    F = r_[real(mismatch)[1:],          # P mismatches (skip slack)
           imag(mismatch)[1+len(pvs):]]  # Q mismatches for PQ buses only
```

**Key Points:**
- Skip slack bus P equation (index [0] or [1] depending on distribute_slack)
- Skip all PV bus Q equations (indices [1] to [1+len(pvs)])
- Only include PQ bus Q equations

### 4. **State Vector (Guess)**
The state vector contains only the **unknown variables**:
```python
guess = r_[
    v_ang[pvpqs],  # Angles for PV and PQ buses (NOT slack)
    v_mag[pqs]      # Magnitudes for PQ buses only (NOT PV, NOT slack)
]
```

**Dimensions:**
- P equations: len(PV) + len(PQ) = len(pvpqs)
- Q equations: len(PQ) only
- θ variables: len(PV) + len(PQ) = len(pvpqs)  
- |V| variables: len(PQ) only

This ensures: **#equations == #variables**

### 5. **Jacobian Structure**
```python
J = [[ ∂P/∂θ    ∂P/∂|V|   ]     # Real power mismatches
     [ ∂Q/∂θ    ∂Q/∂|V|   ]]    # Reactive power mismatches
```

**Dimensions:**
- J00 (∂P/∂θ): [len(pvpqs), len(pvpqs)]
- J01 (∂P/∂|V|): [len(pvpqs), len(pqs)]
- J10 (∂Q/∂θ): [len(pqs), len(pvpqs)]  
- J11 (∂Q/∂|V|): [len(pqs), len(pqs)]

### 6. **Jacobian Calculation**
PyPSA uses elegant matrix formulation:

```python
# Compute current injection
I = Y * V

# Make diagonal matrices
V_diag = diag(V)
V_norm_diag = diag(V / |V|)
I_diag = diag(I)

# Derivatives of complex power S = V * conj(I)
dS_dVa = 1j * V_diag * conj(I_diag - Y * V_diag)
dS_dVm = V_norm_diag * conj(I_diag) + V_diag * conj(Y * V_norm_diag)

# Extract submatrices
J00 = real(dS_dVa)[1:, 1:]                      # Skip slack for P/θ
J01 = real(dS_dVm)[1:, 1+len(pvs):]            # Skip slack and PVs for P/|V|
J10 = imag(dS_dVa)[1+len(pvs):, 1:]            # Skip slack and PVs for Q/θ
J11 = imag(dS_dVm)[1+len(pvs):, 1+len(pvs):]   # Skip slack and PVs for Q/|V|
```

### 7. **Newton-Raphson Update**
```python
while not converged and iter < max_iter:
    # 1. Calculate mismatch
    F = f(guess)
    
    # 2. Build Jacobian
    J = dfdx(guess)
    
    # 3. Solve linear system: J * Δx = F
    delta_x = spsolve(J, F)
    
    # 4. Update state
    guess = guess - delta_x
    
    # 5. Check convergence
    converged = norm(F, inf) < tolerance
```

### 8. **Voltage Update Logic**
```python
# Update angles for PV and PQ buses
v_ang[pvpqs] = guess[:len(pvpqs)]

# Update magnitudes for PQ buses only
v_mag[pqs] = guess[len(pvpqs):]

# PV bus magnitudes remain at setpoint (e.g., 1.0 pu)
# Slack bus V and angle remain fixed
```

## Key Differences from Current Implementation

### ❌ **Current Issues:**
1. **Mismatch includes ALL buses** - should skip slack P and PV Q equations
2. **State vector might include fixed variables** - should only have unknowns
3. **Jacobian includes rows/columns for fixed values** - wrong dimensions
4. **Power specified as `P_injection_pu`** - should be `S = P + 1j*Q`

### ✅ **PyPSA Approach:**
1. **Mismatch excludes fixed equations** - mathematically correct
2. **State vector = unknowns only** - ensures square Jacobian
3. **Jacobian dimensions match** - solvable system
4. **Complex power formulation** - elegant and efficient

## Implementation Recommendations

### Step 1: Fix Power Specification
Instead of:
```python
node.properties['P_injection_pu'] = P_MW / S_base
node.properties['Q_injection_pu'] = Q_MW / S_base
```

Use:
```python
# For load flow, we need SPECIFIED powers
node.properties['P_specified_pu'] = P_MW / S_base  # Positive for gen, negative for load
node.properties['Q_specified_pu'] = Q_MW / S_base
```

### Step 2: Build Specified Power Vector
```python
def _get_specified_powers(self) -> np.ndarray:
    """Get S = P + 1j*Q for all buses"""
    n_nodes = len(self.y_builder.node_to_index)
    S = np.zeros(n_nodes, dtype=complex)
    
    for bus_name, phases in self.graph.nodes.items():
        for phase in PhaseType:
            node_phase_id = f"{bus_name}_{phase.value}"
            if node_phase_id in self.y_builder.node_to_index:
                idx = self.y_builder.node_to_index[node_phase_id]
                node = phases[phase]
                
                P = node.properties.get('P_specified_pu', 0.0)
                Q = node.properties.get('Q_specified_pu', 0.0)
                S[idx] = P + 1j*Q
    
    return S
```

### Step 3: Implement PyPSA-style Mismatch
```python
def _calculate_mismatch(self, V: np.ndarray, S_spec: np.ndarray) -> np.ndarray:
    """Calculate power mismatch F"""
    # Complex power mismatch
    mismatch = V * np.conj(self.Y_matrix * V) - S_spec
    
    # Build F vector (skip appropriate buses)
    F = np.r_[
        np.real(mismatch)[1:],                    # P: skip slack (index 0)
        np.imag(mismatch)[1 + len(self.pv_buses):]  # Q: skip slack and PV buses
    ]
    
    return F
```

### Step 4: Implement PyPSA-style Jacobian
```python
def _build_jacobian(self, V: np.ndarray) -> csr_matrix:
    """Build Jacobian using PyPSA formulation"""
    n = len(V)
    index = np.r_[:n]
    
    # Current injection
    I = self.Y_matrix * V
    
    # Diagonal matrices
    V_diag = csr_matrix((V, (index, index)))
    V_norm_diag = csr_matrix((V / np.abs(V), (index, index)))
    I_diag = csr_matrix((I, (index, index)))
    
    # Derivatives
    dS_dVa = 1j * V_diag @ np.conj(I_diag - self.Y_matrix @ V_diag)
    dS_dVm = V_norm_diag @ np.conj(I_diag) + V_diag @ np.conj(self.Y_matrix @ V_norm_diag)
    
    # Extract blocks (skip slack at index 0, PV buses for Q)
    n_pv = len(self.pv_buses)
    n_pq = len(self.pq_buses)
    
    J00 = dS_dVa[1:, 1:].real                  # ∂P/∂θ
    J01 = dS_dVm[1:, 1+n_pv:].real            # ∂P/∂|V|
    J10 = dS_dVa[1+n_pv:, 1:].imag            # ∂Q/∂θ  
    J11 = dS_dVm[1+n_pv:, 1+n_pv:].imag       # ∂Q/∂|V|
    
    # Assemble full Jacobian
    J = svstack([
        shstack([J00, J01]),
        shstack([J10, J11])
    ], format='csr')
    
    return J
```

### Step 5: Newton-Raphson with Correct Dimensions
```python
def solve(self):
    # Initialize
    V = self._initialize_voltages()  # All buses
    S_spec = self._get_specified_powers()
    
    # Build state vector (unknowns only)
    guess = np.r_[
        np.angle(V)[self.pvpq_indices],  # θ for PV and PQ buses
        np.abs(V)[self.pq_indices]        # |V| for PQ buses only
    ]
    
    for iteration in range(self.max_iterations):
        # Calculate mismatch
        F = self._calculate_mismatch(V, S_spec)
        
        # Check convergence
        if np.max(np.abs(F)) < self.tolerance:
            converged = True
            break
        
        # Build Jacobian
        J = self._build_jacobian(V)
        
        # Solve: J * Δx = F
        delta_x = spsolve(J, F)
        
        # Update state
        guess -= delta_x
        
        # Reconstruct V from guess
        V = self._reconstruct_voltage(guess)
    
    return LoadFlowResults(...)
```

## Summary

PyPSA's implementation is mathematically rigorous and computationally efficient because:

1. **Correct equation count** - only unknown variables in the system
2. **Sparse matrix operations** - efficient for large systems  
3. **Complex power formulation** - elegant S = V * conj(I)
4. **Proper bus type handling** - different equations for PQ/PV/Slack
5. **Clean separation** - mismatch calculation vs Jacobian vs update

By adopting this approach, your load flow solver will be robust, converge properly, and handle all bus types correctly!
