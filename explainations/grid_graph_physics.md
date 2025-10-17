# Electrical Grid Graph Formulation with Physics Equations

## Part 1: Graph Structure Fundamentals

### 1.1 Basic Graph Definition

An electrical grid can be represented as a **directed graph**:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{X}, \mathcal{U})$$

Where:
- **$\mathcal{V}$** = Set of nodes (vertices)
- **$\mathcal{E}$** = Set of edges
- **$\mathcal{X}$** = State vectors associated with nodes and edges
- **$\mathcal{U}$** = Input/control vectors

### 1.2 Node Types in Power Grid

**Type 1: Generator Buses**
- Synchronous generators
- Inverter-based generators (solar, wind, battery)
- Have voltage magnitude and active power control

**Type 2: Load Buses (PQ buses)**
- Constant or dynamic loads
- Specified active (P) and reactive (Q) power

**Type 3: Slack/Reference Bus**
- Voltage magnitude and angle are fixed
- Balances system power
- Usually only one per system

**Type 4: PV Buses**
- Voltage and active power controlled
- Reactive power adjusts to maintain voltage

### 1.3 Edge Types in Power Grid

**Type 1: Transmission Lines**
- Pi-model representation
- Have series impedance and shunt admittance

**Type 2: Transformers**
- Similar to lines but with turns ratio
- May have tap changers (controllable)

**Type 3: Switches/Breakers**
- Binary on/off state
- Topology changes

### 1.4 Incidence and Adjacency Matrices

**Incidence Matrix** $\mathbf{M} \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{E}|}$:
- $M_{ij} = +1$ if edge $j$ leaves node $i$
- $M_{ij} = -1$ if edge $j$ enters node $i$  
- $M_{ij} = 0$ otherwise

**Adjacency Matrix** $\mathbf{A} \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$:
- $A_{ij} = 1$ if there's an edge from node $i$ to node $j$
- $A_{ij} = 0$ otherwise

**Weighted Adjacency** $\mathbf{A}_w$:
- $A_{w,ij} = Y_{ij}$ (admittance) if connected
- Captures electrical coupling strength

---

## Part 2: Three-Phase Extension

### 2.1 Multi-Graph Structure

For three-phase systems:

$$\mathcal{G}_{3\phi} = \{\mathcal{G}_a, \mathcal{G}_b, \mathcal{G}_c, \mathcal{E}_{coupling}\}$$

Where:
- $\mathcal{G}_a, \mathcal{G}_b, \mathcal{G}_c$ = Individual phase graphs
- $\mathcal{E}_{coupling}$ = Inter-phase coupling edges

**Important:** Most components couple all three phases:
- Transformers (especially Y-Δ connections)
- Mutual inductance between parallel lines
- Unbalanced loads
- Ground connections

### 2.2 State Vector Organization

For a system with $n$ nodes:

$$\mathbf{X}_{total} = \begin{bmatrix} \mathbf{X}_a \\ \mathbf{X}_b \\ \mathbf{X}_c \end{bmatrix} \in \mathbb{R}^{3n}$$

Or as a block matrix:
$$\mathbf{X}_{phases} = \begin{bmatrix} \mathbf{x}_1^a & \mathbf{x}_1^b & \mathbf{x}_1^c \\ \mathbf{x}_2^a & \mathbf{x}_2^b & \mathbf{x}_2^c \\ \vdots & \vdots & \vdots \\ \mathbf{x}_n^a & \mathbf{x}_n^b & \mathbf{x}_n^c \end{bmatrix}$$

### 2.3 Coupling Matrix

$$\mathbf{C}_{coupling} = \begin{bmatrix} 
\mathbf{Y}_{aa} & \mathbf{Y}_{ab} & \mathbf{Y}_{ac} \\
\mathbf{Y}_{ba} & \mathbf{Y}_{bb} & \mathbf{Y}_{bc} \\
\mathbf{Y}_{ca} & \mathbf{Y}_{cb} & \mathbf{Y}_{cc}
\end{bmatrix}$$

Where $\mathbf{Y}_{\phi\psi}$ captures mutual coupling between phases $\phi$ and $\psi$.

---

## Part 3: Physics Equations on Graphs

### 3.1 General Form - Node Dynamics

For each node $i$, the dynamics can be written as:

$$\frac{d\mathbf{x}_i}{dt} = \mathbf{f}_i(\mathbf{x}_i, \{\mathbf{x}_j : j \in \mathcal{N}(i)\}, \mathbf{u}_i)$$

Where $\mathcal{N}(i)$ is the set of neighbors of node $i$.

**Key Insight:** Node dynamics depend on:
1. Its own state $\mathbf{x}_i$
2. States of connected neighbors
3. External inputs $\mathbf{u}_i$

### 3.2 General Form - Edge Dynamics

For each edge $e = (i,j)$:

$$\frac{d\mathbf{x}_e}{dt} = \mathbf{f}_e(\mathbf{x}_e, \mathbf{x}_i, \mathbf{x}_j)$$

**Key Insight:** Edge dynamics depend on the states of both terminal nodes.

### 3.3 Power Flow Equations (Algebraic Constraints)

At each node $i$, power balance must hold:

$$P_i = V_i \sum_{j \in \mathcal{N}(i)} V_j (G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij})$$

$$Q_i = V_i \sum_{j \in \mathcal{N}(i)} V_j (G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij})$$

Where:
- $\theta_{ij} = \theta_i - \theta_j$ (angle difference)
- $G_{ij}, B_{ij}$ are conductance and susceptance
- Sum is over all neighbors (captured by graph structure!)

**In matrix form:**
$$\mathbf{P} = \text{diag}(\mathbf{V}) \mathbf{Y} \mathbf{V} \odot \cos(\boldsymbol{\theta})$$

---

## Part 4: Component Models on Graphs

### 4.1 Synchronous Generator (Classical Model)

**Node state:** $\mathbf{x}_i = [\delta_i, \omega_i]^T$

Where:
- $\delta_i$ = rotor angle (rad)
- $\omega_i$ = rotor speed (rad/s)

**Dynamics:**
$$\frac{d\delta_i}{dt} = \omega_i - \omega_0$$

$$\frac{d\omega_i}{dt} = \frac{1}{2H_i}(P_{m,i} - P_{e,i} - D_i(\omega_i - \omega_0))$$

**Electrical Power (Graph Interaction):**
$$P_{e,i} = E_i' V_i B_{ii}'\sin(\delta_i - \theta_i) + E_i' \sum_{j \in \mathcal{N}(i)} V_j B_{ij}' \sin(\delta_i - \theta_j)$$

**Key Observation:** The sum over neighbors $\mathcal{N}(i)$ is where the graph structure enters!

### 4.2 Transmission Line (Pi-Model)

**Edge connecting nodes $i$ and $j$:**

**Parameters:**
- Series impedance: $Z_{ij} = R_{ij} + jX_{ij}$
- Shunt admittance: $Y_{sh} = jB_{sh}/2$ at each end

**Current flow (Kirchhoff's Law):**
$$I_{ij} = Y_{ij}(V_i - V_j) + Y_{sh}V_i$$
$$I_{ji} = Y_{ij}(V_j - V_i) + Y_{sh}V_j$$

Where $Y_{ij} = 1/Z_{ij}$

**Power flow on edge:**
$$S_{ij} = V_i I_{ij}^* = P_{ij} + jQ_{ij}$$

**For dynamic models (RL line):**
$$L_{ij}\frac{dI_{ij}}{dt} = V_i - V_j - R_{ij}I_{ij}$$

### 4.3 Load Model (Voltage-Dependent)

**Static load at node $i$:**
$$P_{L,i} = P_{L,i}^0 \left(\frac{V_i}{V_i^0}\right)^{\alpha_p}$$
$$Q_{L,i} = Q_{L,i}^0 \left(\frac{V_i}{V_i^0}\right)^{\alpha_q}$$

**Dynamic load (induction motor):**
$$\frac{d\omega_{m,i}}{dt} = \frac{1}{2H_m}(T_e - T_L)$$

Where torque depends on neighbors through voltage.

### 4.4 Inverter-Based Resource

**Node state:** $\mathbf{x}_i = [i_{d,i}, i_{q,i}, v_{dc,i}, \phi_i]^T$

**Current dynamics (dq-frame):**
$$L_f \frac{di_{d,i}}{dt} = -R_f i_{d,i} + \omega L_f i_{q,i} + v_{d,i}^{ref} - v_{d,i}$$
$$L_f \frac{di_{q,i}}{dt} = -R_f i_{q,i} - \omega L_f i_{d,i} + v_{q,i}^{ref} - v_{q,i}$$

**Connection to neighbors:** Through grid voltage $v_{d,i}, v_{q,i}$ which depends on adjacent nodes.

---

## Part 5: Complete System Formulation

### 5.1 Differential-Algebraic Equations (DAE)

The complete system is a DAE system:

**Differential equations (dynamics):**
$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, \mathbf{y}, \mathbf{u})$$

**Algebraic equations (network constraints):**
$$\mathbf{0} = \mathbf{g}(\mathbf{x}, \mathbf{y}, \mathbf{u})$$

Where:
- $\mathbf{x}$ = differential states (rotor angles, speeds, currents, etc.)
- $\mathbf{y}$ = algebraic states (bus voltages, angles)
- $\mathbf{u}$ = inputs/controls

### 5.2 Network Equations in Graph Form

**Kirchhoff's Current Law (KCL) at each node:**
$$\sum_{j \in \mathcal{N}(i)} I_{ij} + I_{gen,i} - I_{load,i} = 0$$

**In matrix form using incidence matrix:**
$$\mathbf{M} \mathbf{I}_{edges} + \mathbf{I}_{nodes} = \mathbf{0}$$

**Kirchhoff's Voltage Law (KVL) for each loop:**
$$\sum_{e \in loop} V_e = 0$$

### 5.3 Admittance Matrix from Graph

The bus admittance matrix $\mathbf{Y}_{bus}$ can be constructed from the graph:

$$Y_{bus,ii} = \sum_{j \in \mathcal{N}(i)} Y_{ij} + Y_{sh,i}$$

$$Y_{bus,ij} = -Y_{ij} \text{ if } j \in \mathcal{N}(i), \text{ else } 0$$

**Graph interpretation:** 
- Diagonal = self-admittance (sum of all connected edges)
- Off-diagonal = negative of edge admittance
- Sparsity pattern = graph adjacency!

---

## Part 6: Numerical Examples

### 6.1 Simple 3-Bus System

**Graph structure:**
```
    Bus 1 (Gen) ---Line 1---> Bus 2 (Load)
                                 |
                              Line 2
                                 |
                                 v
                             Bus 3 (Load)
```

**Adjacency matrix:**
$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

**Incidence matrix:**
$$\mathbf{M} = \begin{bmatrix} 
+1 & 0 \\
-1 & +1 \\
0 & -1
\end{bmatrix}$$
(Columns = edges, Rows = nodes)

**State vectors:**
- Node 1: $\mathbf{x}_1 = [\delta_1, \omega_1]$ (generator)
- Node 2: $\mathbf{x}_2 = [V_2, \theta_2]$ (load bus)
- Node 3: $\mathbf{x}_3 = [V_3, \theta_3]$ (load bus)

**Dynamics:**

Generator (Node 1):
$$\frac{d\delta_1}{dt} = \omega_1 - \omega_0$$
$$\frac{d\omega_1}{dt} = \frac{1}{2H_1}(P_m - E_1'V_2 B_{12}\sin(\delta_1-\theta_2))$$

Network (Nodes 2, 3):
$$P_2 = V_2[G_{22}V_2 + G_{21}V_1\cos(\theta_2-\theta_1) + G_{23}V_3\cos(\theta_2-\theta_3)]$$

The neighbor relationships $\mathcal{N}(1) = \{2\}$, $\mathcal{N}(2) = \{1,3\}$, $\mathcal{N}(3) = \{2\}$ are explicit in the equations!

### 6.2 Handling Three-Phase

**For balanced system:**
- All three phases have identical topology
- Can solve single-phase equivalent
- Multiply powers by 3

**For unbalanced system:**
- Need full three-phase representation
- State vector: $\mathbf{X} = [\mathbf{X}_a, \mathbf{X}_b, \mathbf{X}_c]^T$
- Coupling matrix captures phase interactions

**Example - Three-phase line:**
$$\begin{bmatrix} V_a^i \\ V_b^i \\ V_c^i \end{bmatrix} - \begin{bmatrix} V_a^j \\ V_b^j \\ V_c^j \end{bmatrix} = \begin{bmatrix} Z_{aa} & Z_{ab} & Z_{ac} \\ Z_{ba} & Z_{bb} & Z_{bc} \\ Z_{ca} & Z_{cb} & Z_{cc} \end{bmatrix} \begin{bmatrix} I_a \\ I_b \\ I_c \end{bmatrix}$$

The coupling matrix shows how phases interact!

---

## Part 7: Implementation Strategy

### 7.1 Data Structures

**Node class:**
```python
class Node:
    id: int
    type: str  # 'generator', 'load', 'bus'
    state: np.ndarray  # [δ, ω, ...] or [V, θ, ...]
    neighbors: List[int]
    parameters: Dict  # H, D, X'd, etc.
```

**Edge class:**
```python
class Edge:
    id: int
    from_node: int
    to_node: int
    type: str  # 'line', 'transformer', 'switch'
    state: np.ndarray  # [I, φ, ...] if dynamic
    parameters: Dict  # R, X, B, etc.
```

**Graph class:**
```python
class PowerGrid:
    nodes: List[Node]
    edges: List[Edge]
    adjacency_matrix: np.ndarray
    incidence_matrix: np.ndarray
    admittance_matrix: np.ndarray
```

### 7.2 Computing Neighbor Interactions

For node $i$, compute power from neighbors:

```python
def compute_power_injection(node_i, grid):
    P_e = 0
    V_i = node_i.state[0]  # voltage magnitude
    theta_i = node_i.state[1]  # angle
    
    for j in node_i.neighbors:
        node_j = grid.nodes[j]
        V_j = node_j.state[0]
        theta_j = node_j.state[1]
        
        Y_ij = grid.admittance_matrix[node_i.id, node_j.id]
        G_ij = Y_ij.real
        B_ij = Y_ij.imag
        
        P_e += V_i * V_j * (G_ij*cos(theta_i - theta_j) 
                           + B_ij*sin(theta_i - theta_j))
    
    return P_e
```

### 7.3 Building System Equations

**Step 1:** Initialize graph structure
**Step 2:** For each node, write its dynamics using neighbor info
**Step 3:** For each edge, write its dynamics using terminal nodes
**Step 4:** Assemble into system DAE

```python
def assemble_system(grid):
    def dynamics(t, X, Y):
        dXdt = np.zeros_like(X)
        
        # For each node
        for i, node in enumerate(grid.nodes):
            if node.type == 'generator':
                dXdt[i] = generator_dynamics(node, grid)
            elif node.type == 'load':
                dXdt[i] = load_dynamics(node, grid)
        
        # For each edge
        for e, edge in enumerate(grid.edges):
            if edge.type == 'line' and edge.is_dynamic:
                dXdt[...] = line_dynamics(edge, grid)
        
        return dXdt
    
    def algebraic(X, Y):
        # Power balance at each bus
        g = power_balance(X, Y, grid)
        return g
    
    return dynamics, algebraic
```

---

## Part 8: Special Considerations

### 8.1 Topology Changes

When a switch opens or line trips:
- Update adjacency matrix $\mathbf{A}$
- Update incidence matrix $\mathbf{M}$
- Recompute admittance matrix $\mathbf{Y}_{bus}$
- Neighbor lists change!

**Implementation:** Keep graph structure flexible, allow dynamic updates.

### 8.2 Coordinate Frames

**Common Operating Point (COE):**
- Angles relative to reference bus

**Synchronous Reference Frame (SRF):**
- Rotating at system frequency $\omega_0$
- Used for generator dynamics

**Local dq-frames:**
- Each inverter has its own rotating frame
- Need transformation matrices

**Graph implication:** Need to track reference frame for each node.

### 8.3 Different Time Scales

Different components have different time constants:
- **Electromechanical (seconds):** Generator swings
- **Electromagnetic (milliseconds):** Stator transients, line dynamics
- **Electronic (microseconds):** Inverter switching

**Multi-rate simulation:** Use different time steps for different parts of graph.

---

## Part 9: Validation Checklist

Before adding learning terms, verify physics model:

✅ **Conservation laws:**
- Total active power in = out
- Total reactive power balanced

✅ **Steady-state:**
- Power flow solution matches
- Voltages within reasonable range

✅ **Dynamic response:**
- Generator swing curves look reasonable
- Damped oscillations
- Stable equilibrium

✅ **Frequency response:**
- Correct modal frequencies
- Proper damping ratios

✅ **Benchmark comparison:**
- Compare to commercial tools (PSS/E, PowerWorld, PSCAD)
- Use IEEE test cases

---

## Part 10: Summary

### Key Takeaways

**1. Graph = Natural Representation**
- Nodes = buses/generators/loads
- Edges = lines/transformers
- Adjacency captures connectivity

**2. Physics on Graphs**
- Node dynamics depend on neighbors: $\frac{dx_i}{dt} = f(x_i, \{x_j : j \in \mathcal{N}(i)\})$
- Edge dynamics couple terminal nodes
- Network equations enforce KCL/KVL

**3. Three-Phase Extension**
- Three coupled graphs
- Coupling matrix for phase interactions
- Can be balanced (simplified) or unbalanced (full)

**4. DAE Structure**
- Differential equations for dynamics
- Algebraic equations for network
- Solved simultaneously

**5. Implementation Ready**
- Clear data structures
- Systematic assembly
- Neighbor interactions explicit

### Next Steps

Once this physics foundation is solid:
1. ✅ Validate against known benchmarks
2. ✅ Identify where model errors occur
3. → Add learnable correction terms at those locations
4. → Train to minimize error
5. → Validate hybrid model

---

## References for Physics Models

**Books:**
1. Kundur - "Power System Stability and Control" (comprehensive)
2. Sauer & Pai - "Power System Dynamics and Stability" (good for DAEs)
3. Milano - "Power System Modelling and Scripting" (implementation focus)

**Standards:**
1. IEEE Std 1110 - Guide for Synchronous Generator Modeling
2. IEEE Std 421.5 - Excitation System Models

**Graph Theory:**
1. Dörfler & Bullo - "Kron Reduction of Graphs with Applications to Electrical Networks"
2. Bollobás - "Modern Graph Theory"