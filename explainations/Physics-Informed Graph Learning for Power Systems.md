
**A Hybrid Approach with Learnable Correction Terms**

---

## 🎯 Core Research Vision

Develop a **physics-informed graph neural network** framework that:

1. Starts with known mathematical models of electrical grids
2. Adds learnable correction terms to capture unmodeled dynamics
3. Maintains physical structure and guarantees
4. Enables multi-fidelity modeling for different study types
5. Can dynamically adapt as the system changes

### Key Innovation

> Rather than replacing physics with pure ML, we **augment** accurate physics models with minimal learnable terms to capture:
> 
> - Parasitic components often neglected
> - Sub-electromagnetic transients
> - Model mismatch and uncertainties
> - Time-varying system characteristics

---

## 📊 Problem Motivation

### Why Existing Models Fall Short

**Mathematical Models:**

- Often neglect parasitic effects (stray capacitance, skin effect, etc.)
- Simplified for computational tractability
- Fixed parameters don't capture aging/degradation
- Sub-electromagnetic transients require extremely detailed models

**Pure ML Models:**

- Lack physical interpretability
- Poor generalization outside training data
- No stability guarantees
- Cannot leverage centuries of power systems knowledge

**Our Hybrid Approach:**

- Keeps physics as the backbone (stability, interpretability, generalization)
- Learns only the "correction" needed for accuracy
- Much more data-efficient than pure ML
- Maintains physical constraints

---

## 🏗️ Mathematical Framework

### 1. Grid Representation as Multi-Phase Graph

#### Three-Phase Graph Structure

```
G = (G_a, G_b, G_c, E_coupling)
```

Where:

- **G_a, G_b, G_c**: Graphs for phases A, B, C
- **E_coupling**: Inter-phase coupling edges (transformers, mutual inductances)

#### Graph Components

**Nodes (V):**

- Generators (synchronous machines, inverters)
- Loads (dynamic, static)
- Buses (PQ, PV, slack)
- Storage devices

**Edges (E):**

- Transmission lines
- Transformers
- Reactors/Capacitors
- Switches (topology changes)

**Node State Vector:**

```
x_i = [V_i, θ_i, ω_i, I_d,i, I_q,i, ...]^T
```

**Edge State Vector:**

```
x_e = [I_e, φ_e, ...]^T
```

### 2. Hybrid Dynamics Formulation

#### Core Equation

```
dx/dt = F_physics(x, A, u) + G_θ(x, A, u)
        └─────────────┘     └───────────┘
         Known physics      Learnable correction
```

Where:

- **x**: System state vector
- **A**: Adjacency/incidence matrix
- **u**: Control inputs
- **θ**: Learnable parameters

#### Structured Learnable Terms

**Option 1: Additive Node/Edge Corrections**

```
dx_i/dt = f_physics,i(x_i, neighbors) + g_θ,i(x_i, neighbors)
```

**Option 2: Multiplicative Relative Corrections**

```
dx/dt = f_physics(x) · (1 + α_θ(x))
```

- Ensures correction → 0 as base model improves
- Physically interpretable as "model error factor"

**Option 3: Hierarchical Multi-Order**

```
g_θ(x) = g_1(x) + ε·g_2(x) + ε²·g_3(x) + ...
```

- Natural multi-fidelity structure
- Low-order for simple studies, high-order for detailed

#### Structure Preservation

**Port-Hamiltonian Structure:**

```
dx/dt = [J(x) - R(x)]∇H(x) + g(x)u + Δ_θ(x)
```

Constraints on Δ_θ to ensure:

- **Energy conservation**: ∇H^T Δ_θ ≤ 0 (passivity)
- **Power balance**: Σ P_i = 0
- **Physical bounds**: |x| ≤ x_max

---

## 🧮 Mathematical Theory Foundations

### 3. Universal Approximation Theory

#### Why This Proves Globality

**Theorem 1: Universal Approximation on Manifolds (Kidger & Lyons, 2020)**

> Neural ODEs can approximate any continuous flow on a compact manifold to arbitrary accuracy.

**Application to Power Systems:**

- Grid dynamics evolve on manifolds (e.g., rotor angles on T^n)
- If base physics + learnable term = Neural ODE
- Then: Can represent ANY grid dynamics

#### Kolmogorov-Arnold Representation

**Classical Theorem:** Any multivariate continuous function f: [0,1]^n → R can be written as:

```
f(x_1,...,x_n) = Σ Φ_q(Σ φ_q,p(x_p))
```

**Modern Interpretation (KAN):**

- Replace fixed basis with learnable univariate functions
- More efficient than MLPs for certain function classes
- Natural for control systems (composition of simple nonlinearities)

**For Grid Modeling:**

```
g_θ(x) = Σ Φ_q(Σ φ_q,p(x_p))
```

where φ and Φ are learnable splines/polynomials

### 4. Graph Neural Network Theory

#### Universal Approximation on Graphs

**Key Result:** A GNN with sufficient depth can approximate any permutation-invariant function on graphs.

**For Power Grids:**

```
G_θ(X, A) = Σ_{k=1}^K A^k σ(W_k X + b_k)
```

Where:

- k = order of neighborhood (1 = immediate neighbors)
- As K → ∞, can represent any graph function
- Automatically respects grid topology

#### Message Passing Interpretation

```
x_i^(t+1) = UPDATE(x_i^(t), AGGREGATE({x_j^(t) : j ∈ N(i)}))
```

Physical meaning:

- **AGGREGATE**: Collect measurements from neighbors (Kirchhoff's laws)
- **UPDATE**: Apply local dynamics with neighbor influence

### 5. Stability and Boundedness Proofs

#### Lyapunov Stability

**Goal:** Prove learned system remains stable

**Approach:**

1. Construct Lyapunov function: V(x) (e.g., total energy)
2. Prove: dV/dt ≤ 0 along trajectories
3. Constraint on G_θ: Ensure dV/dt|_{G_θ} ≤ 0

**Implementation:**

- Use energy as Lyapunov function
- Add soft constraint during training
- Or: Project G_θ onto passive subspace

#### Input-to-State Stability (ISS)

**Ensures:** Bounded inputs → Bounded states

**Theorem:** If physics model is ISS and ||G_θ|| is bounded, then hybrid model is ISS.

**Practical Check:**

- Lipschitz bound on G_θ
- Use techniques from robust control

---

## 🔍 Multi-Fidelity Modeling

### 6. Reduced-Order Models from High-Fidelity

#### Theoretical Foundation

**Idea:** Full model lives on low-dimensional manifold M ⊂ R^n

**Methods:**

**A. Proper Orthogonal Decomposition (POD)**

```
x(t) ≈ Σ_{i=1}^r a_i(t)φ_i
```

- φ_i: Dominant spatial modes
- r << n: Reduced dimension
- a_i: Temporal coefficients

**B. Dynamic Mode Decomposition (DMD)**

```
x(t) ≈ Σ_{i=1}^r b_i φ_i e^{λ_i t}
```

- φ_i: Dynamic modes
- λ_i: Growth rates and frequencies
- Natural for oscillatory systems (power grids!)

**C. Autoencoder Approach**

```
Encoder: x → z (low-dim)
Learned dynamics: dz/dt = f_θ(z)
Decoder: z → x̂
```

#### Multi-Fidelity Hierarchy

**Level 0 (Fastest):** Steady-state power flow

- Algebraic equations only
- For planning studies

**Level 1:** Electromechanical transients

- 1-100 Hz dynamics
- Rotor angles, frequencies
- For stability studies

**Level 2:** Electromagnetic transients

- 100 Hz - 10 kHz
- Includes stator transients
- For control design

**Level 3:** Sub-electromagnetic

- > 10 kHz
    
- Parasitics, switching transients
- For EMI/detailed hardware studies

**Key Innovation:** Single learned model → Project to any level!

---

## 🎓 Methodology for Paper

### Phase 1: Problem Formulation

#### 1.1 Define System Classes

- Synchronous generator grids
- Inverter-based grids
- Hybrid grids
- Each has different "hard to model" aspects

#### 1.2 Identify Model Deficiencies

- Compare high-fidelity simulation vs. measured data
- Quantify error: ε(t) = x_measured(t) - x_model(t)
- Analyze frequency content of error
- **This motivates the learnable term!**

### Phase 2: Architecture Design

#### 2.1 Graph Construction

- Define node types and features
- Define edge types and features
- Specify coupling structure (inter-phase)

#### 2.2 Physics Model Selection

- Choose appropriate fidelity for base model
- Document all assumptions and neglected effects
- This sets the "baseline" that learning will improve

#### 2.3 Learnable Term Design

- Choose structure (additive vs. multiplicative)
- Select approximator (MLP, KAN, GNN, implicit layer)
- Define where corrections are applied (nodes, edges, both)

#### 2.4 Constraint Implementation

- Energy conservation
- Power balance
- Physical bounds
- Stability certificates

### Phase 3: Theoretical Guarantees

#### 3.1 Universal Approximation Proof

**Theorem Statement:**

> Given grid dynamics satisfying [regularity conditions], there exists θ* such that: ||x_true(t) - x_hybrid(t; θ*)|| < ε for all t ∈ [0, T]

**Proof Outline:**

1. Show dynamics are Lipschitz on compact domain
2. Apply universal approximation on manifolds
3. Construct covering argument
4. Provide explicit bound on network size

#### 3.2 Stability Analysis

- Prove Lyapunov stability of learned system
- Derive ISS bounds
- Show preservation under reduced-order projection

#### 3.3 Generalization Bounds

- PAC learning framework
- Sample complexity for grid dynamics
- Dependence on graph size/topology

### Phase 4: Learning Algorithm

#### 4.1 Training Data Generation

**Scenario Coverage:**

- Normal operation
- Faults (short circuits, line trips)
- Large disturbances
- Topology changes
- Various loading conditions

**Data Sources:**

- High-fidelity simulation (EMTP, PSCAD)
- Field measurements (PMU data)
- Hardware-in-the-loop tests

#### 4.2 Loss Function Design

```
L = L_prediction + λ_physics L_physics + λ_stability L_stability
```

Where:

- **L_prediction**: State trajectory error
- **L_physics**: Violation of known constraints
- **L_stability**: Lyapunov condition violation

#### 4.3 Training Strategy

- Curriculum learning: Easy → Hard scenarios
- Physics-based initialization
- Regularization for generalization
- Multi-fidelity training

### Phase 5: Validation & Case Studies

#### 5.1 Benchmark Systems

- IEEE test cases (9-bus, 39-bus, 118-bus)
- Real grid data (if available)
- Inverter-dominant microgrid

#### 5.2 Validation Metrics

- **Accuracy**: RMSE, MAE vs. ground truth
- **Speed**: Computational time vs. high-fidelity
- **Stability**: Check eigenvalues, Lyapunov
- **Generalization**: Performance on unseen scenarios

#### 5.3 Ablation Studies

- Physics-only vs. Learned vs. Hybrid
- Different learnable term structures
- Effect of data quantity/quality
- Network size vs. performance

### Phase 6: Multi-Fidelity Demonstration

#### 6.1 Model Reduction

- Extract reduced models at each fidelity level
- Validate against appropriate benchmarks
- Show accuracy-speed tradeoffs

#### 6.2 Cross-Fidelity Consistency

- Ensure Level 1 model is consistent limit of Level 2
- Validate energy conservation across scales

---

## 🔧 Implementation Considerations

### 7. Practical Aspects

#### Choice of Learnable Approximator

**Option A: Kolmogorov-Arnold Networks (KAN)**

- **Pros:** Efficient for low-dimensional, interpretable
- **Cons:** Scaling to high-dim unclear
- **Best for:** Component-level corrections (inverter model)

**Option B: Deep Implicit Layers**

- **Pros:** Natural for equilibrium constraints, guaranteed bounds
- **Cons:** Training more complex
- **Best for:** Steady-state or quasi-static problems

**Option C: Graph Neural Networks**

- **Pros:** Respects topology, scales to large grids
- **Cons:** May need many layers for global information
- **Best for:** Full grid modeling

**Option D: Hybrid**

- GNN for graph structure
- KAN for node/edge dynamics
- **Recommended approach!**

#### Handling Discrete Events

Power systems have discrete events:

- Switching
- Protection actions
- Topology changes

**Approach:** Hybrid dynamical systems framework

```
Continuous: dx/dt = f(x, q)
Discrete: q^+ = δ(x, q, event)
```

Where q = discrete mode

#### Computational Efficiency

**Online vs. Offline:**

- **Offline:** Full learned model for scenario generation
- **Online:** Reduced model for real-time control

**Parallelization:**

- GNNs naturally parallel (per-node computation)
- Can distribute over GPU for large grids

---

## 📝 Paper Structure Outline

### Suggested Paper Organization

**Title:** _Physics-Informed Graph Neural Networks with Learnable Corrections for Multi-Fidelity Power System Modeling_

**Abstract**

- Problem: Need for accurate yet computationally efficient grid models
- Gap: Physics models miss details, pure ML lacks guarantees
- Solution: Hybrid approach with theoretically justified learnable corrections
- Results: Achieves X% accuracy with Y% speedup

**1. Introduction**

- Motivation: Renewable integration, grid complexity
- Challenge: Accuracy vs. speed tradeoff
- Related work: Physics-informed ML, GNNs for power systems
- Contributions

**2. Background & Theory**

- 2.1 Power System Dynamics
- 2.2 Graph Representation of Grids
- 2.3 Universal Approximation Theory
- 2.4 Multi-Fidelity Modeling

**3. Methodology**

- 3.1 Problem Formulation
- 3.2 Hybrid Model Architecture
- 3.3 Learnable Term Design
- 3.4 Constraint Enforcement

**4. Theoretical Analysis**

- 4.1 Universal Approximation Proof
- 4.2 Stability Guarantees
- 4.3 Reduced-Order Consistency

**5. Implementation**

- 5.1 Training Algorithm
- 5.2 Data Generation
- 5.3 Network Architecture Details

**6. Case Studies**

- 6.1 Benchmark System Results
- 6.2 Ablation Studies
- 6.3 Multi-Fidelity Validation

**7. Discussion**

- Interpretability of learned terms
- Limitations and future work
- Broader impact

**8. Conclusion**

---

## 🎯 Next Steps & Discussion Topics

### Priority 1: Architecture Details

- [ ] Exact GNN structure for grid
- [ ] KAN vs. MLP for correction terms
- [ ] How to handle three-phase coupling

### Priority 2: Theoretical Proofs

- [ ] Complete universal approximation proof
- [ ] Stability certificate derivation
- [ ] Sample complexity bounds

### Priority 3: Implementation

- [ ] Code framework (PyTorch Geometric?)
- [ ] Data generation pipeline
- [ ] Training infrastructure

### Priority 4: Validation

- [ ] Select benchmark systems
- [ ] Define metrics
- [ ] Baseline comparisons

---

## 📚 Key References to Study

### Universal Approximation

1. Hornik et al. (1989) - Multilayer feedforward networks are universal approximators
2. Kidger & Lyons (2020) - Universal approximation with neural ODEs on manifolds
3. Liu et al. (2024) - Kolmogorov-Arnold Networks

### Graph Neural Networks

1. Battaglia et al. (2018) - Relational inductive biases, deep learning, and graph networks
2. Sanchez-Gonzalez et al. (2020) - Learning to simulate complex physics with graph networks
3. Pfaff et al. (2021) - Learning mesh-based simulation with graph networks

### Physics-Informed ML

1. Raissi et al. (2019) - Physics-informed neural networks (PINNs)
2. Chen et al. (2018) - Neural ordinary differential equations
3. Cranmer et al. (2020) - Lagrangian neural networks

### Power Systems

1. Kundur (1994) - Power System Stability and Control
2. Milano (2010) - Power System Modelling and Scripting
3. Dörfler & Bullo (2013) - Kron reduction of graphs with applications to electrical networks

### Multi-Fidelity

1. Peherstorfer et al. (2018) - Survey of multifidelity methods
2. Meng & Karniadakis (2020) - A composite neural network for multifidelity modeling

---

## 💡 Research Questions to Address

### Fundamental

1. What is the minimal complexity of G_θ needed for a given accuracy?
2. How does performance scale with grid size?
3. What training data coverage is sufficient?

### Practical

1. How to handle topology changes in real-time?
2. Can we guarantee real-time performance for control?
3. How to update model online as grid changes?

### Theoretical

1. Can we derive closed-form bounds on approximation error?
2. What is the sample complexity as function of grid size?
3. Under what conditions can we guarantee stability?

---

## 🔄 Living Document Notes

**Last Updated:** [Date]

**Current Focus:** [Section being developed]

**Open Questions:** [List]

**Recent Insights:** [Notes]

---

_This document is a working research framework. Each section can be expanded into detailed subsections as we progress through the research._