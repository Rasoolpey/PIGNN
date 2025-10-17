# Three-Phase Line and Transformer Models with Coupling

## Part 1: Why Coupling Matters

### 1.1 What is Phase Coupling?

**Coupling** means that current/voltage in one phase affects the other phases.

**Where it comes from:**
- **Magnetic coupling:** Current in phase A creates a magnetic field that induces voltage in phases B and C
- **Capacitive coupling:** Electric field between phase A and B conductors
- **Common impedance:** Shared neutral or ground return path

**When we can ignore it (balanced systems):**
- All three phases have equal magnitude, 120° apart
- Symmetric network configuration
- Can use positive-sequence equivalent (single-phase model)

**When we MUST include it (unbalanced systems):**
- Single line-to-ground faults
- Unbalanced loads
- Untransposed lines (phases not swapped periodically)
- Asymmetric winding connections

---

## Part 2: Three-Phase Transmission Line Model

### 2.1 Basic Three-Phase Line Equations

For a line connecting bus $i$ to bus $j$:

$$\begin{bmatrix} V_a^i \\ V_b^i \\ V_c^i \end{bmatrix} - \begin{bmatrix} V_a^j \\ V_b^j \\ V_c^j \end{bmatrix} = \begin{bmatrix} Z_{aa} & Z_{ab} & Z_{ac} \\ Z_{ba} & Z_{bb} & Z_{bc} \\ Z_{ca} & Z_{cb} & Z_{cc} \end{bmatrix} \begin{bmatrix} I_a \\ I_b \\ I_c \end{bmatrix}$$

Or in compact form:
$$\mathbf{V}^i - \mathbf{V}^j = \mathbf{Z}_{line} \mathbf{I}$$

**Key point:** The off-diagonal terms ($Z_{ab}, Z_{ac}, Z_{bc}$) are the mutual impedances that couple the phases!

### 2.2 Impedance Matrix Components

**Self impedance** (diagonal terms):
$$Z_{aa} = Z_{bb} = Z_{cc} = R + j\omega L_{self}$$

Where:
- $R$ = resistance of conductor
- $L_{self}$ = self-inductance of phase conductor

**Mutual impedance** (off-diagonal):
$$Z_{ab} = Z_{ba} = R_{ground} + j\omega M_{ab}$$

Where:
- $R_{ground}$ = ground return resistance (if present)
- $M_{ab}$ = mutual inductance between phases A and B

### 2.3 Calculating Mutual Inductance

**Carson's equations** (for overhead lines):

$$M_{ab} = \frac{\mu_0}{2\pi} \ln\left(\frac{D_{ab}}{GMR}\right)$$

Where:
- $D_{ab}$ = geometric distance between phase A and B conductors
- $GMR$ = geometric mean radius of conductor
- $\mu_0$ = permeability of free space

**For typical overhead lines:**
```
Configuration:  a     b     c    (horizontal)
               
Self impedance: Z_self = R + j0.4 Ω/km (typical)
Mutual impedance: Z_mutual ≈ j0.1 to j0.3 Ω/km
```

**Physical interpretation:** 
- Closer phases → larger mutual impedance
- Larger conductors → smaller self-impedance

### 2.4 Complete Line Model (Pi-Model with Coupling)

Including shunt capacitance:

$$\begin{bmatrix} I_a^i \\ I_b^i \\ I_c^i \end{bmatrix} = \mathbf{Y}_{series} \left( \begin{bmatrix} V_a^i \\ V_b^i \\ V_c^i \end{bmatrix} - \begin{bmatrix} V_a^j \\ V_b^j \\ V_c^j \end{bmatrix} \right) + \mathbf{Y}_{shunt} \begin{bmatrix} V_a^i \\ V_b^i \\ V_c^i \end{bmatrix}$$

Where:
- $\mathbf{Y}_{series} = \mathbf{Z}_{line}^{-1}$ (3×3 matrix)
- $\mathbf{Y}_{shunt} = j\omega \mathbf{C}_{line}$ (3×3 capacitance matrix)

**Shunt capacitance matrix:**
$$\mathbf{C}_{line} = \begin{bmatrix} C_{aa} & -C_{ab} & -C_{ac} \\ -C_{ab} & C_{bb} & -C_{bc} \\ -C_{ac} & -C_{bc} & C_{cc} \end{bmatrix}$$

Note: Off-diagonal terms are negative (capacitive coupling)

### 2.5 Symmetrical Components Perspective

The impedance matrix can be decomposed:
$$\mathbf{Z}_{abc} = \mathbf{A} \mathbf{Z}_{012} \mathbf{A}^{-1}$$

Where $\mathbf{A}$ is the symmetrical components transformation:
$$\mathbf{A} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & a^2 & a \\ 1 & a & a^2 \end{bmatrix}, \quad a = e^{j2\pi/3}$$

And:
$$\mathbf{Z}_{012} = \begin{bmatrix} Z_0 & 0 & 0 \\ 0 & Z_1 & 0 \\ 0 & 0 & Z_2 \end{bmatrix}$$

Where:
- $Z_0$ = zero-sequence impedance (ground return path)
- $Z_1$ = positive-sequence impedance (balanced operation)
- $Z_2$ = negative-sequence impedance (usually $Z_2 \approx Z_1$)

**Key relationships:**
$$Z_0 = Z_{self} + 2Z_{mutual}$$
$$Z_1 = Z_2 = Z_{self} - Z_{mutual}$$

**Physical meaning:**
- Zero-sequence sees all mutual coupling (all phases in phase)
- Positive/negative sequence sees difference (phases cancel coupling)

### 2.6 Dynamic Line Model

For electromagnetic transients (sub-cycle phenomena):

$$\mathbf{L}_{line} \frac{d\mathbf{I}}{dt} = \mathbf{V}^i - \mathbf{V}^j - \mathbf{R}_{line} \mathbf{I}$$

Where:
$$\mathbf{L}_{line} = \begin{bmatrix} L_{aa} & L_{ab} & L_{ac} \\ L_{ba} & L_{bb} & L_{bc} \\ L_{ca} & L_{cb} & L_{cc} \end{bmatrix}$$

**This is a coupled differential equation!** Current in phase A affects the voltage change in phases B and C.

### 2.7 Example: Numerical Values

**Typical 230 kV overhead line (per km):**

$$\mathbf{Z}_{line} = \begin{bmatrix} 
0.1 + j0.4 & 0.05 + j0.15 & 0.05 + j0.1 \\
0.05 + j0.15 & 0.1 + j0.4 & 0.05 + j0.15 \\
0.05 + j0.1 & 0.05 + j0.15 & 0.1 + j0.4
\end{bmatrix} \, \Omega/\text{km}$$

Notice:
- Diagonal (self) is larger than off-diagonal (mutual)
- Matrix is symmetric: $Z_{ab} = Z_{ba}$
- Not perfectly symmetric because phase spacing may vary

---

## Part 3: Three-Phase Transformer Model

### 3.1 Why Transformers are Complex

Transformers couple phases in **fundamentally different ways** depending on winding configuration:

**Winding types:**
- **Wye (Y):** Phases connected to common neutral
- **Delta (Δ):** Phases connected in a triangle
- **Grounded vs. Ungrounded:** Affects zero-sequence behavior

**Common configurations:**
1. Y-Y (both wye)
2. Y-Δ (wye primary, delta secondary)
3. Δ-Δ (both delta)
4. Zig-zag (special for grounding)

### 3.2 Ideal Transformer Equations

For an ideal transformer (no losses, no leakage):

**Voltage relationship:**
$$\mathbf{V}_{secondary} = n \mathbf{T} \mathbf{V}_{primary}$$

**Current relationship:**
$$\mathbf{I}_{primary} = n \mathbf{T}^T \mathbf{I}_{secondary}$$

Where:
- $n$ = turns ratio
- $\mathbf{T}$ = connection matrix (depends on winding configuration)

### 3.3 Y-Y Transformer (Grounded)

**Connection matrix:**
$$\mathbf{T}_{YY} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Impedance model:**
$$\mathbf{Z}_{YY} = \begin{bmatrix} 
Z_t & Z_{m} & Z_{m} \\
Z_{m} & Z_t & Z_{m} \\
Z_{m} & Z_{m} & Z_t
\end{bmatrix}$$

Where:
- $Z_t = R + j\omega L_{leakage}$ (leakage impedance)
- $Z_m$ = mutual impedance between windings (usually small, ~0.1 $Z_t$)

**Key characteristic:** 
- Zero-sequence current can flow (if both neutrals grounded)
- No phase shift between primary and secondary
- Simple 1:1 phase relationship

### 3.4 Y-Δ Transformer

**Connection matrix:**
$$\mathbf{T}_{Y\Delta} = \begin{bmatrix} 
1 & -1 & 0 \\
0 & 1 & -1 \\
-1 & 0 & 1
\end{bmatrix}$$

**This introduces a 30° phase shift!**

**Voltage relationship (per unit):**
$$\begin{bmatrix} V_{ab} \\ V_{bc} \\ V_{ca} \end{bmatrix}_{sec} = n \begin{bmatrix} 
1 & -1 & 0 \\
0 & 1 & -1 \\
-1 & 0 & 1
\end{bmatrix} \begin{bmatrix} V_a \\ V_b \\ V_c \end{bmatrix}_{prim}$$

**Impedance model:**
Same as Y-Y, but zero-sequence is **blocked** (cannot pass through delta)!

$$Z_{0,Y\Delta} = \infty$$

**Key characteristics:**
- 30° phase shift between primary and secondary
- Zero-sequence current cannot pass through
- Used to isolate ground faults
- Common in transmission/distribution interface

### 3.5 Δ-Δ Transformer

**Connection matrix:**
$$\mathbf{T}_{\Delta\Delta} = \begin{bmatrix} 
1 & 0 & -1 \\
-1 & 1 & 0 \\
0 & -1 & 1
\end{bmatrix} \begin{bmatrix} 
1 & -1 & 0 \\
0 & 1 & -1 \\
-1 & 0 & 1
\end{bmatrix}^{-1}$$

**Key characteristics:**
- No phase shift (0° if configured standard)
- Zero-sequence blocked on both sides
- No neutral connection possible
- Circulating currents can flow in delta

### 3.6 Complete Transformer Model with Coupling

**General form (including magnetizing branch):**

$$\begin{bmatrix} V_a^p \\ V_b^p \\ V_c^p \\ V_a^s \\ V_b^s \\ V_c^s \end{bmatrix} = \begin{bmatrix} 
\mathbf{Z}_p & \mathbf{Z}_m \\
\mathbf{Z}_m^T & \mathbf{Z}_s
\end{bmatrix} \begin{bmatrix} I_a^p \\ I_b^p \\ I_c^p \\ I_a^s \\ I_b^s \\ I_c^s \end{bmatrix}$$

Where:
- $\mathbf{Z}_p$ = primary leakage impedance (3×3)
- $\mathbf{Z}_s$ = secondary leakage impedance (3×3)
- $\mathbf{Z}_m$ = mutual impedance between primary and secondary (3×3)

**For Y-Y transformer:**
$$\mathbf{Z}_p = \mathbf{Z}_s = \begin{bmatrix} 
R_l + j\omega L_l & j\omega M_{ab} & j\omega M_{ac} \\
j\omega M_{ab} & R_l + j\omega L_l & j\omega M_{bc} \\
j\omega M_{ac} & j\omega M_{bc} & R_l + j\omega L_l
\end{bmatrix}$$

$$\mathbf{Z}_m = n \begin{bmatrix} 
j\omega M & 0 & 0 \\
0 & j\omega M & 0 \\
0 & 0 & j\omega M
\end{bmatrix}$$

Where $M$ = magnetizing inductance

### 3.7 Practical Simplification

For most power flow and stability studies, use **simplified model**:

**Step 1:** Convert to per-unit on common base
**Step 2:** Model as series impedance:

$$\mathbf{Z}_{transformer} = n^2 \mathbf{Z}_p + \mathbf{Z}_s$$

**Step 3:** Add ideal transformer to account for phase shift:

$$\mathbf{I}_{primary} = n \mathbf{T}^T \mathbf{I}_{secondary}$$

### 3.8 Zero-Sequence Behavior (Critical!)

This is where winding configuration matters most:

| Configuration | Zero-Sequence Path | $Z_0$ |
|--------------|-------------------|-------|
| Y(grounded)-Y(grounded) | Passes through | Finite |
| Y(grounded)-Δ | Blocked by delta | ∞ (delta side) |
| Δ-Δ | Blocked both sides | ∞ (both sides) |
| Y(ungrounded)-Y(grounded) | Blocked by ungrounded Y | ∞ (ungrounded side) |

**Why this matters:**
- Ground faults involve zero-sequence current
- Wrong model → incorrect fault current calculations
- Can affect protection coordination

---

## Part 4: Graph Representation

### 4.1 How to Represent on Graph

**Option 1: Edge with 3×3 Impedance Matrix**

```
Node i (3 phases)  ---[Z_line (3×3)]---  Node j (3 phases)
    [V_abc^i]                                [V_abc^j]
```

Each edge stores a 3×3 complex matrix.

**Option 2: Expanded Graph (9 edges)**

```
Node i_a  --[Z_aa]-- Node j_a
          --[Z_ab]-- Node j_b
          --[Z_ac]-- Node j_c

Node i_b  --[Z_ba]-- Node j_a
          --[Z_bb]-- Node j_b
          --[Z_bc]-- Node j_c

Node i_c  --[Z_ca]-- Node j_a
          --[Z_cb]-- Node j_b
          --[Z_cc]-- Node j_c
```

Explicitly shows all coupling as separate edges!

**Recommendation:** Use Option 1 (cleaner, more compact)

### 4.2 Node States in Unbalanced System

Each node has 3-phase voltage and current:

$$\mathbf{x}_i = \begin{bmatrix} V_a^i \\ V_b^i \\ V_c^i \\ \theta_a^i \\ \theta_b^i \\ \theta_c^i \end{bmatrix}$$

Or in complex form:
$$\mathbf{x}_i = \begin{bmatrix} \tilde{V}_a^i \\ \tilde{V}_b^i \\ \tilde{V}_c^i \end{bmatrix}, \quad \tilde{V}_\phi = V_\phi e^{j\theta_\phi}$$

### 4.3 Edge Equations with Coupling

For edge $e$ connecting nodes $i$ and $j$:

$$\mathbf{L}_{e} \frac{d\mathbf{I}_e}{dt} = \mathbf{V}^i - \mathbf{V}^j - \mathbf{R}_e \mathbf{I}_e$$

Where:
$$\mathbf{I}_e = \begin{bmatrix} I_{e,a} \\ I_{e,b} \\ I_{e,c} \end{bmatrix}, \quad \mathbf{L}_e = \begin{bmatrix} L_{aa} & L_{ab} & L_{ac} \\ L_{ba} & L_{bb} & L_{bc} \\ L_{ca} & L_{cb} & L_{cc} \end{bmatrix}$$

**This is a coupled ODE!** Solving for $\frac{dI_a}{dt}$ requires inverting $\mathbf{L}_e$.

### 4.4 Transformer as Special Edge

For Y-Δ transformer edge:

$$\mathbf{V}^{primary} = \mathbf{Z}_{transformer} \mathbf{I}^{primary}$$
$$\mathbf{V}^{secondary} = n \mathbf{T}_{Y\Delta} \mathbf{V}^{primary}$$
$$\mathbf{I}^{primary} = n \mathbf{T}_{Y\Delta}^T \mathbf{I}^{secondary}$$

Store both $\mathbf{Z}_{transformer}$ and $\mathbf{T}$ as edge attributes.

---

## Part 5: Practical Implementation

### 5.1 Data Structure for Coupled Edge

```python
class ThreePhaseLine:
    from_node: int
    to_node: int
    
    # Full 3×3 impedance matrices
    Z_series: np.ndarray  # shape (3, 3), complex
    Y_shunt: np.ndarray   # shape (3, 3), complex
    
    # For dynamics
    L_matrix: np.ndarray  # shape (3, 3), real
    R_matrix: np.ndarray  # shape (3, 3), real
    
    # Current state (if dynamic model)
    I_abc: np.ndarray  # shape (3,), complex
    
    def voltage_drop(self, V_i, V_j):
        """Calculate voltage drop considering coupling"""
        return self.Z_series @ self.I_abc
    
    def dynamics(self, V_i, V_j):
        """dI/dt = L^{-1} (V_i - V_j - R*I)"""
        L_inv = np.linalg.inv(self.L_matrix)
        V_drop = V_i - V_j - self.R_matrix @ self.I_abc
        return L_inv @ V_drop
```

### 5.2 Transformer Edge Class

```python
class ThreePhaseTransformer:
    from_node: int
    to_node: int
    
    # Configuration
    config: str  # 'YY', 'YD', 'DY', 'DD'
    turns_ratio: float
    
    # Impedance
    Z_leakage: np.ndarray  # shape (3, 3)
    
    # Connection matrix
    T_matrix: np.ndarray  # shape (3, 3)
    
    # Zero-sequence handling
    allows_zero_seq: bool
    
    def get_connection_matrix(self):
        if self.config == 'YY':
            return np.eye(3)
        elif self.config == 'YD':
            return np.array([[1, -1, 0],
                           [0, 1, -1],
                           [-1, 0, 1]])
        # ... other configurations
    
    def apply_coupling(self, V_primary, I_secondary):
        """Apply voltage and current transformations"""
        n = self.turns_ratio
        T = self.T_matrix
        
        V_secondary = n * T @ V_primary
        I_primary = n * T.T @ I_secondary
        
        return V_secondary, I_primary
```

### 5.3 Building the System

```python
def assemble_unbalanced_system(grid):
    n_nodes = len(grid.nodes)
    n_phases = 3
    
    # State vector: [V_a1, V_b1, V_c1, ..., V_an, V_bn, V_cn]
    V = np.zeros(n_nodes * n_phases, dtype=complex)
    
    # Current vector for all edges
    I_edges = []
    
    def dynamics(t, state):
        # Unpack state
        V = state[:n_nodes*n_phases]
        I_all = state[n_nodes*n_phases:]
        
        dVdt = np.zeros_like(V)
        dIdt = []
        
        # For each node
        for i, node in enumerate(grid.nodes):
            idx = slice(i*n_phases, (i+1)*n_phases)
            V_i = V[idx]
            
            # Current injection from neighbors (KCL)
            I_injection = np.zeros(3, dtype=complex)
            
            for edge in node.connected_edges:
                if edge.from_node == i:
                    I_injection -= edge.I_abc
                else:
                    I_injection += edge.I_abc
            
            # Node dynamics (generator, load, etc.)
            dVdt[idx] = node.dynamics(V_i, I_injection)
        
        # For each edge
        for edge in grid.edges:
            i, j = edge.from_node, edge.to_node
            V_i = V[i*n_phases:(i+1)*n_phases]
            V_j = V[j*n_phases:(j+1)*n_phases]
            
            # Edge dynamics with coupling
            dI = edge.dynamics(V_i, V_j)
            dIdt.append(dI)
        
        return np.concatenate([dVdt, np.concatenate(dIdt)])
    
    return dynamics
```

### 5.4 Computing with Coupling

**Example: Current through coupled line**

```python
def compute_line_current_coupled(V_i, V_j, Z_matrix):
    """
    V_i, V_j: complex arrays of shape (3,)
    Z_matrix: complex array of shape (3, 3)
    
    Returns: I_abc (current in each phase)
    """
    # Invert impedance to get admittance
    Y_matrix = np.linalg.inv(Z_matrix)
    
    # Current = Y * (V_i - V_j)
    # This automatically includes coupling!
    V_diff = V_i - V_j
    I_abc = Y_matrix @ V_diff
    
    return I_abc

# Example usage
V_i = np.array([1.0 + 0j, -0.5 - 0.866j, -0.5 + 0.866j])  # Balanced
V_j = np.array([0.95 + 0j, -0.48 - 0.83j, -0.47 + 0.83j])  # Slightly unbalanced

Z_line = np.array([
    [0.1 + 0.4j, 0.05 + 0.15j, 0.05 + 0.1j],
    [0.05 + 0.15j, 0.1 + 0.4j, 0.05 + 0.15j],
    [0.05 + 0.1j, 0.05 + 0.15j, 0.1 + 0.4j]
])

I_abc = compute_line_current_coupled(V_i, V_j, Z_line)
print(f"Phase A current: {I_abc[0]}")
print(f"Phase B current: {I_abc[1]}")
print(f"Phase C current: {I_abc[2]}")
```

---

## Part 6: Key Takeaways

### 6.1 When to Use Full Coupling Model

**Must use full 3×3 matrices when:**
- ✅ Studying unbalanced faults (SLG, LL, LLG)
- ✅ Untransposed transmission lines
- ✅ Unbalanced loads (single-phase, unequal)
- ✅ Y-Δ transformers with ground faults
- ✅ Harmonic analysis
- ✅ Sub-synchronous resonance studies

**Can use simplified (balanced) model when:**
- ✅ All loads are balanced three-phase
- ✅ Only interested in positive sequence dynamics
- ✅ Planning studies at steady-state
- ✅ Well-transposed lines

### 6.2 Computational Complexity

**Balanced system:**
- State size: $n$ nodes
- Admittance matrix: $n \times n$

**Unbalanced system:**
- State size: $3n$ nodes (one per phase)
- Admittance matrix: $3n \times 3n$
- **3× memory, 9× computation per matrix operation**

### 6.3 Common Mistakes to Avoid

❌ **Assuming transformer phase shift is 0°**
   → Check if Y-Δ introduces 30° shift

❌ **Forgetting zero-sequence is blocked by delta**
   → Can give wrong fault currents

❌ **Using single-phase model for SLG faults**
   → Must use three-phase model

❌ **Not properly grounding neutrals**
   → Zero-sequence current has no return path

❌ **Ignoring mutual coupling in parallel lines**
   → Can be 30-50% of self impedance!

### 6.4 Validation Checklist

✅ Check symmetry: $Z_{ab} = Z_{ba}$
✅ Verify positive/negative sequence impedances are similar
✅ Zero-sequence impedance should be larger (more coupling)
✅ Phase shift correct for Y-Δ transformers
✅ Power balance satisfied (sum of three phases)
✅ Compare to established software (PSCAD, EMTP)

---

## Part 7: Practical Example

### 7.1 10 km Line with Typical Values

```python
# Impedance matrix (Ω)
Z_line = 10 * np.array([  # 10 km line
    [0.15 + 0.45j, 0.05 + 0.15j, 0.05 + 0.12j],
    [0.05 + 0.15j, 0.15 + 0.45j, 0.05 + 0.15j],
    [0.05 + 0.12j, 0.05 + 0.15j, 0.15 + 0.45j]
])

# Scenario: Balanced voltages, but phase C has higher current (unbalanced load)
V_send = np.array([230e3/np.sqrt(3) * np.exp(1j*0),
                   230e3/np.sqrt(3) * np.exp(1j*-2*np.pi/3),
                   230e3/np.sqrt(3) * np.exp(1j*2*np.pi/3)])

I_load = np.array([1000 * np.exp(1j*-np.pi/6),  # Phase A
                   1000 * np.exp(1j*(-np.pi/6 - 2*np.pi/3)),  # Phase B
                   1500 * np.exp(1j*(-np.pi/6 + 2*np.pi/3))])  # Phase C (50% higher!)

# Receiving end voltage (affected by coupling!)
V_receive = V_send - Z_line @ I_load

print("Voltage unbalance:")
print(f"Phase A: {abs(V_receive[0])/230e3*np.sqrt(3):.3f} pu")
print(f"Phase B: {abs(V_receive[1])/230e3*np.sqrt(3):.3f} pu")
print(f"Phase C: {abs(V_receive[2])/230e3*np.sqrt(3):.3f} pu")

# Without coupling (diagonal Z only), unbalance would be worse!
```

---

## Summary

**Lines:**
- Use 3×3 impedance matrix: $\mathbf{Z} = \mathbf{R} + j\omega\mathbf{L}$
- Off-diagonal = mutual coupling between phases
- Zero-sequence impedance = $Z_{self} + 2Z_{mutual}$

**Transformers:**
- Configuration determines coupling: Y-Y allows zero-seq, Y-Δ blocks it
- Phase shift from connection matrix $\mathbf{T}$
- Store both impedance and connection matrices

**Graph representation:**
- Each edge has 3×3 complex matrix
- Each node has 3-phase voltage vector
- Equations naturally couple through matrix multiplication

**Implementation:**
- Invert impedance matrices to get admittance
- Use matrix operations for coupled equations
- Validate using symmetrical components

This gives you the foundation for accurate unbalanced system modeling!