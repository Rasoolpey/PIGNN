# Power Grid Graph Storage and Learnable Dynamics Specification

**Version:** 1.0  
**Date:** 2025-01-19  
**Purpose:** Complete specification for HDF5-based power grid graph storage and Port-Hamiltonian Kolmogorov-Arnold Network (PH-KAN) learnable dynamic objects

---

## ⚡ ANDES RMS Simulation Requirements

### Current Data Status (✅ Complete)

**Extracted from PowerFactory** (`data/composite_model_out/39_Bus_New_England_System_COMPOSITE_EXTRACTED.h5`):

| Parameter Category | Status | Details |
|-------------------|--------|---------|
| Generator Dynamics | ✅ 100% | H, D, Xd, Xq, X'd, X"d, X'q, X"q, Td0', Tq0', Td0", Tq0", Ra, Xl |
| Machine Base Values | ✅ 100% | Sn_MVA, Un_kV for all 10 generators |
| Voltage Setpoints | ✅ 100% | Vset_pu from power flow |
| Control Systems | ✅ 90% | 9/10 AVR, 9/10 GOV, 9/10 PSS (G01 missing) |
| Operating Points | ✅ 100% | P_MW, Q_MVAR, Vt_pu, theta_rad, omega_pu |
| Network Topology | ✅ 100% | 39 buses, 46 branches, admittance matrix |

### Next Step: H5 → ANDES Conversion

**Required ANDES Models:**

1. **Bus** (39 buses):
   - `idx`, `name`, `Vn` (kV), `v0` (pu), `a0` (rad)

2. **Line** (46 branches):
   - `bus1`, `bus2`, `r` (pu), `x` (pu), `b` (pu), `rate_a` (MVA)

3. **PQ Load** (19 loads):
   - `bus`, `p0` (pu), `q0` (pu)

4. **GENROU** (10 generators - Round Rotor Model):
   - `bus`, `Sn` (MVA), `Vn` (kV), `fn` (Hz)
   - Mechanical: `M` (= 2*H), `D`
   - Steady-state: `xd`, `xq`, `ra`
   - Transient: `xd1` (X'd), `xq1` (X'q), `Td01` (Td0'), `Tq01` (Tq0')
   - Subtransient: `xd2` (X"d), `xq2` (X"q), `Td02` (Td0"), `Tq02` (Tq0")
   - Other: `xl` (leakage)

5. **EXDC1** (9 AVR systems - DC Exciter):
   - `syn` (generator idx), `Ka`, `Ta`, `Ke`, `Te`, `Kf`, `Tf`, `Vrmax`, `Vrmin`

6. **TGOV1** (9 Governor systems - Simple Turbine Governor):
   - `syn` (generator idx), `R` (droop), `T1`, `T2`, `T3`, `Pmax`, `Pmin`

7. **STAB1** or **ST2CUT** (9 PSS systems):
   - `syn` (generator idx), `Kw` (gain), `T1`, `T2`, `T3`, `T4`

### Parameter Mapping Guide

| H5 Parameter | ANDES GENROU | Notes |
|--------------|--------------|-------|
| `H_s` | `M` = 2*H | ANDES uses M (momentum) |
| `Xd_prime` | `xd1` | Transient reactance |
| `Xd_double` | `xd2` | Subtransient reactance |
| `Td0_prime` | `Td01` | d-axis transient time constant |
| `Td0_double` | `Td02` | d-axis subtransient time constant |

**Reference:** ANDES documentation at `explainations/andes-master/docs/source/modelref.rst`

---

## Table of Contents

1. [Overview](#1-overview)
2. [HDF5 Storage Format](#2-hdf5-storage-format)
3. [Port-Hamiltonian KAN Theory](#3-port-hamiltonian-kan-theory)
4. [Learnable Dynamic Graph Objects](#4-learnable-dynamic-graph-objects)
5. [Integration Architecture](#5-integration-architecture)
6. [Training and Learning](#6-training-and-learning)
7. [Analysis Tools](#7-analysis-tools)
8. [Implementation Requirements](#8-implementation-requirements)

---

## 1. Overview

### 1.1 Purpose

This specification defines:
- **HDF5 storage format** for three-phase power grid graphs with learnable components
- **Port-Hamiltonian KAN (PH-KAN)** objects for learning unknown IBR/load dynamics
- **Integration mechanisms** between static graph and dynamic learnable objects
- **Training procedures** and stability guarantees

### 1.2 Key Features

✅ **HDF5-based storage**: Hierarchical, expandable, efficient  
✅ **Physics-informed learning**: Port-Hamiltonian structure ensures stability  
✅ **Interpretable models**: KAN allows symbolic extraction  
✅ **Modular design**: Attach/detach learnable objects to any node  
✅ **Multi-timescale**: Supports load flow → RMS → EMT simulations  
✅ **Linearization-ready**: Extract ODEs for small-signal analysis  

### 1.3 Use Cases

- **Black-box IBR modeling**: Learn control dynamics when vendor models unavailable
- **Load modeling**: Capture voltage-dependent and dynamic load behavior  
- **Aging equipment**: Adapt to changing component characteristics over time
- **Grid expansion**: Add new components with learned dynamics
- **Stability analysis**: Linearize around operating points for eigenvalue analysis
- **Impedance scanning**: Extract frequency-domain characteristics

---

## 2. HDF5 Storage Format

### 2.1 File Structure

```
power_grid_learnable.h5
│
├─ metadata/                          # File-level metadata
│  ├─ version (string)                # Format version "1.0"
│  ├─ creation_date (string)          # ISO 8601 timestamp
│  ├─ last_modified (string)          # ISO 8601 timestamp
│  ├─ grid_name (string)              # e.g., "IEEE39_unbalanced"
│  ├─ base_mva (float64)              # System base power (MVA)
│  ├─ base_frequency_hz (float64)     # Nominal frequency (Hz)
│  └─ description (string)            # Optional description
│
├─ topology/                          # Graph topology
│  ├─ num_buses (int64)               # Number of buses
│  ├─ num_phases (int64)              # Always 3 for three-phase
│  ├─ num_lines (int64)               # Number of transmission lines
│  ├─ num_transformers (int64)        # Number of transformers
│  ├─ num_generators (int64)          # Number of generators
│  ├─ num_loads (int64)               # Number of loads
│  │
│  ├─ adjacency_matrix/               # Sparse adjacency (per-phase)
│  │  ├─ data (float64, shape=(nnz,))
│  │  ├─ indices (int32, shape=(nnz,))
│  │  ├─ indptr (int32, shape=(N+1,))
│  │  └─ shape (int64, shape=(2,))    # [N, N] where N = num_buses
│  │
│  └─ edge_list/                      # Alternative representation
│     ├─ from_bus (int64, shape=(E,)) # Source bus indices
│     ├─ to_bus (int64, shape=(E,))   # Destination bus indices
│     └─ edge_type (int64, shape=(E,)) # 0=line, 1=xfmr, 2=switch
│
├─ phases/                            # Per-phase data
│  ├─ phase_a/
│  │  ├─ nodes/
│  │  │  ├─ bus_ids (int64, shape=(N,))           # Bus IDs
│  │  │  ├─ bus_names (string, shape=(N,))        # Bus names
│  │  │  ├─ bus_types (int64, shape=(N,))         # 0=PQ, 1=PV, 2=Slack
│  │  │  ├─ base_voltages_kV (float64, shape=(N,))
│  │  │  ├─ voltages_pu (float64, shape=(N,))     # Voltage magnitude
│  │  │  ├─ angles_deg (float64, shape=(N,))      # Voltage angle
│  │  │  ├─ P_injection_MW (float64, shape=(N,))  # Active power
│  │  │  ├─ Q_injection_MVAR (float64, shape=(N,))# Reactive power
│  │  │  ├─ P_generation_MW (float64, shape=(N,)) # Gen contribution
│  │  │  ├─ Q_generation_MVAR (float64, shape=(N,))
│  │  │  ├─ P_load_MW (float64, shape=(N,))       # Load contribution
│  │  │  ├─ Q_load_MVAR (float64, shape=(N,))
│  │  │  ├─ shunt_G_pu (float64, shape=(N,))      # Shunt conductance
│  │  │  ├─ shunt_B_pu (float64, shape=(N,))      # Shunt susceptance
│  │  │  └─ node_features (float64, shape=(N, F)) # Additional features
│  │  │
│  │  └─ edges/
│  │     ├─ from_bus (int64, shape=(E,))
│  │     ├─ to_bus (int64, shape=(E,))
│  │     ├─ element_id (string, shape=(E,))       # Unique identifier
│  │     ├─ element_type (int64, shape=(E,))      # 0=line, 1=xfmr
│  │     ├─ R_pu (float64, shape=(E,))            # Resistance
│  │     ├─ X_pu (float64, shape=(E,))            # Reactance
│  │     ├─ B_shunt_pu (float64, shape=(E,))      # Shunt susceptance
│  │     ├─ rating_MVA (float64, shape=(E,))      # Thermal rating
│  │     ├─ length_km (float64, shape=(E,))       # Line length
│  │     ├─ in_service (bool, shape=(E,))         # Service status
│  │     └─ edge_features (float64, shape=(E, F)) # Additional features
│  │
│  ├─ phase_b/ (same structure as phase_a)
│  └─ phase_c/ (same structure as phase_a)
│
├─ coupling/                          # Inter-phase coupling
│  ├─ node_coupling/
│  │  ├─ bus_ids (int64, shape=(N,))
│  │  ├─ coupling_matrices (complex128, shape=(N, 3, 3))  # Z or Y coupling
│  │  ├─ coupling_type (int64, shape=(N,))        # 0=none, 1=line, 2=xfmr
│  │  └─ element_ids (string, shape=(N,))         # Associated element
│  │
│  └─ edge_coupling/
│     ├─ element_ids (string, shape=(E,))
│     ├─ coupling_matrices (complex128, shape=(E, 3, 3))
│     ├─ element_type (int64, shape=(E,))         # 0=line, 1=xfmr
│     └─ mutual_impedance (complex128, shape=(E,)) # For lines
│
├─ physics/                           # Physics-based quantities
│  ├─ admittance_matrix/
│  │  ├─ Y_single_phase/              # Per-phase Y-matrix (39×39)
│  │  │  ├─ data_real (float64, shape=(nnz,))
│  │  │  ├─ data_imag (float64, shape=(nnz,))
│  │  │  ├─ indices (int32, shape=(nnz,))
│  │  │  ├─ indptr (int32, shape=(N+1,))
│  │  │  └─ shape (int64, shape=(2,))
│  │  │
│  │  └─ Y_three_phase/               # Full 3-phase Y-matrix (117×117)
│  │     ├─ data_real (float64, shape=(nnz,))
│  │     ├─ data_imag (float64, shape=(nnz,))
│  │     ├─ indices (int32, shape=(nnz,))
│  │     ├─ indptr (int32, shape=(3N+1,))
│  │     └─ shape (int64, shape=(2,))
│  │
│  ├─ power_flow_results/
│  │  ├─ converged (bool)
│  │  ├─ iterations (int64)
│  │  ├─ max_mismatch (float64)
│  │  ├─ total_generation_MW (float64)
│  │  ├─ total_load_MW (float64)
│  │  ├─ total_losses_MW (float64)
│  │  ├─ max_voltage_pu (float64)
│  │  ├─ min_voltage_pu (float64)
│  │  └─ timestamp (string)
│  │
│  └─ jacobian_matrix/                # Load flow Jacobian
│     ├─ data (float64, shape=(nnz,))
│     ├─ indices (int32, shape=(nnz,))
│     ├─ indptr (int32, shape=(2N+1,))
│     └─ operating_point (string)     # Reference to state
│
├─ learnable_objects/                 # Dynamic learnable components
│  ├─ object_registry/
│  │  ├─ object_ids (string, shape=(M,))          # Unique IDs
│  │  ├─ node_ids (string, shape=(M,))            # Attached node
│  │  ├─ object_types (string, shape=(M,))        # 'ibr', 'load', etc.
│  │  ├─ state_dims (int64, shape=(M,))           # Internal state dimension
│  │  ├─ creation_dates (string, shape=(M,))
│  │  └─ is_active (bool, shape=(M,))             # Currently in use
│  │
│  ├─ {object_id}/                    # Per-object data
│  │  ├─ metadata/
│  │  │  ├─ object_type (string)      # 'ibr', 'dynamic_load', etc.
│  │  │  ├─ component_name (string)   # e.g., "Solar_Inverter_1"
│  │  │  ├─ attached_node (string)    # Node ID
│  │  │  ├─ attached_phase (string)   # 'a', 'b', 'c', or 'abc'
│  │  │  ├─ state_dim (int64)         # Dimension of state vector
│  │  │  ├─ input_dim (int64)         # Inputs from grid
│  │  │  ├─ output_dim (int64)        # Outputs to grid
│  │  │  ├─ rated_power_MVA (float64)
│  │  │  └─ description (string)
│  │  │
│  │  ├─ ph_kan_parameters/           # Port-Hamiltonian KAN params
│  │  │  ├─ hamiltonian/              # Energy function H(x)
│  │  │  │  ├─ inner_spline_knots (float64, shape=(K, P))
│  │  │  │  ├─ inner_spline_coeffs (float64, shape=(state_dim, K, P))
│  │  │  │  ├─ outer_spline_knots (float64, shape=(K,))
│  │  │  │  └─ outer_spline_coeffs (float64, shape=(hidden_dim, K))
│  │  │  │
│  │  │  ├─ interconnection_matrix/   # J(x,u) - skew-symmetric
│  │  │  │  ├─ network_weights (float64, shape=(layers, dims))
│  │  │  │  └─ network_biases (float64, shape=(layers, dims))
│  │  │  │
│  │  │  ├─ dissipation_matrix/       # R(x,u) - positive definite
│  │  │  │  ├─ network_weights (float64, shape=(layers, dims))
│  │  │  │  └─ network_biases (float64, shape=(layers, dims))
│  │  │  │
│  │  │  ├─ input_matrix/             # g(x)
│  │  │  │  ├─ network_weights (float64, shape=(layers, dims))
│  │  │  │  └─ network_biases (float64, shape=(layers, dims))
│  │  │  │
│  │  │  ├─ output_map/               # y = h(x,u)
│  │  │  │  ├─ inner_spline_knots (float64, shape=(K, P))
│  │  │  │  ├─ inner_spline_coeffs (float64, shape=(state_dim+input_dim, K, P))
│  │  │  │  ├─ outer_spline_knots (float64, shape=(K,))
│  │  │  │  └─ outer_spline_coeffs (float64, shape=(hidden_dim, K))
│  │  │  │
│  │  │  └─ control_layer/            # For IBRs: PLL, current control
│  │  │     ├─ inner_spline_knots (float64, shape=(K, P))
│  │  │     ├─ inner_spline_coeffs (float64, shape=(state_dim+input_dim, K, P))
│  │  │     ├─ outer_spline_knots (float64, shape=(K,))
│  │  │     └─ outer_spline_coeffs (float64, shape=(hidden_dim, K))
│  │  │
│  │  ├─ physics_priors/              # Known physics constraints
│  │  │  ├─ H0_function (string)      # Initial Hamiltonian guess (symbolic)
│  │  │  ├─ known_constraints (string, shape=(C,))  # e.g., "P_max=1.5"
│  │  │  ├─ passivity_enforced (bool)
│  │  │  └─ prior_model_type (string) # 'grid_following', 'grid_forming'
│  │  │
│  │  ├─ training_history/
│  │  │  ├─ epochs (int64, shape=(T,))
│  │  │  ├─ total_loss (float64, shape=(T,))
│  │  │  ├─ trajectory_loss (float64, shape=(T,))
│  │  │  ├─ physics_loss (float64, shape=(T,))
│  │  │  ├─ passivity_loss (float64, shape=(T,))
│  │  │  ├─ learning_rate (float64, shape=(T,))
│  │  │  └─ timestamps (string, shape=(T,))
│  │  │
│  │  ├─ validation_metrics/
│  │  │  ├─ rmse_voltage (float64)
│  │  │  ├─ rmse_current (float64)
│  │  │  ├─ max_power_error_MW (float64)
│  │  │  ├─ stability_margin (float64)    # Min real part of eigenvalues
│  │  │  └─ validation_timestamp (string)
│  │  │
│  │  ├─ linearization/               # State-space matrices at operating point
│  │  │  ├─ operating_point_x (float64, shape=(state_dim,))
│  │  │  ├─ operating_point_u (float64, shape=(input_dim,))
│  │  │  ├─ A_matrix (float64, shape=(state_dim, state_dim))
│  │  │  ├─ B_matrix (float64, shape=(state_dim, input_dim))
│  │  │  ├─ C_matrix (float64, shape=(output_dim, state_dim))
│  │  │  ├─ D_matrix (float64, shape=(output_dim, input_dim))
│  │  │  ├─ eigenvalues_real (float64, shape=(state_dim,))
│  │  │  ├─ eigenvalues_imag (float64, shape=(state_dim,))
│  │  │  └─ is_stable (bool)
│  │  │
│  │  ├─ impedance_characteristics/   # Frequency-domain representation
│  │  │  ├─ frequencies_Hz (float64, shape=(F,))
│  │  │  ├─ Z_real (float64, shape=(F,))
│  │  │  ├─ Z_imag (float64, shape=(F,))
│  │  │  ├─ magnitude_dB (float64, shape=(F,))
│  │  │  └─ phase_deg (float64, shape=(F,))
│  │  │
│  │  └─ symbolic_extraction/         # Extracted symbolic model
│  │     ├─ hamiltonian_symbolic (string)      # LaTeX/SymPy format
│  │     ├─ dynamics_symbolic (string)
│  │     ├─ output_map_symbolic (string)
│  │     └─ extraction_timestamp (string)
│  │
│  └─ ... (additional learnable objects)
│
├─ neural_network/                    # Global NN state (if using graph-level learning)
│  ├─ embeddings/
│  │  ├─ node_embeddings (float64, shape=(3*N, D))  # D = embedding dim
│  │  ├─ edge_embeddings (float64, shape=(E, D))
│  │  └─ coupling_embeddings (float64, shape=(N+E, D))
│  │
│  ├─ gnn_parameters/                 # If using GNN for corrections
│  │  ├─ layer_{i}/
│  │  │  ├─ weights (float64, shape=(...))
│  │  │  └─ biases (float64, shape=(...))
│  │  └─ ...
│  │
│  └─ training_state/
│     ├─ global_step (int64)
│     ├─ best_loss (float64)
│     └─ checkpoint_path (string)
│
├─ scenarios/                         # Multiple operating scenarios
│  ├─ scenario_registry/
│  │  ├─ scenario_ids (string, shape=(S,))
│  │  ├─ descriptions (string, shape=(S,))
│  │  ├─ timestamps (string, shape=(S,))
│  │  └─ is_baseline (bool, shape=(S,))
│  │
│  ├─ scenario_{id}/
│  │  ├─ voltages_pu (float64, shape=(3*N,))
│  │  ├─ angles_deg (float64, shape=(3*N,))
│  │  ├─ P_injections_MW (float64, shape=(3*N,))
│  │  ├─ Q_injections_MVAR (float64, shape=(3*N,))
│  │  ├─ contingency_description (string)
│  │  └─ power_flow_converged (bool)
│  │
│  └─ ... (additional scenarios)
│
└─ analysis_results/                  # Cached analysis results
   ├─ small_signal_stability/
   │  ├─ eigenvalues_real (float64, shape=(N_states,))
   │  ├─ eigenvalues_imag (float64, shape=(N_states,))
   │  ├─ participation_factors (float64, shape=(N_states, N_states))
   │  ├─ damping_ratios (float64, shape=(N_modes,))
   │  ├─ oscillation_frequencies_Hz (float64, shape=(N_modes,))
   │  └─ analysis_timestamp (string)
   │
   ├─ impedance_scan/
   │  ├─ scan_points (string, shape=(K,))  # Node IDs where scanned
   │  ├─ frequencies_Hz (float64, shape=(F,))
   │  ├─ Z_matrices (complex128, shape=(K, F, 2, 2))  # dq impedance
   │  └─ stability_margins (float64, shape=(K,))
   │
   └─ contingency_analysis/
      ├─ n_minus_1/
      │  ├─ element_ids (string, shape=(E,))
      │  ├─ converged (bool, shape=(E,))
      │  ├─ max_voltage_violations (float64, shape=(E,))
      │  └─ overloaded_elements (int64, shape=(E,))
      │
      └─ n_minus_2/
         └─ ... (similar structure)
```

### 2.2 Data Types and Conventions

#### 2.2.1 Naming Conventions

- **Groups**: lowercase with underscores (e.g., `learnable_objects`)
- **Datasets**: lowercase with underscores (e.g., `voltages_pu`)
- **Object IDs**: format `{type}_{node}_{index}` (e.g., `ibr_bus30_001`)
- **Timestamps**: ISO 8601 format (e.g., `2025-01-19T14:30:00Z`)

#### 2.2.2 Units

- **Power**: MW, MVAR
- **Voltage**: per-unit (pu) or kV
- **Impedance**: per-unit (pu) or ohms
- **Frequency**: Hz
- **Angles**: degrees
- **Time**: seconds

#### 2.2.3 Array Indexing

- **Buses**: 0-indexed (0 to N-1)
- **Phases**: a=0, b=1, c=2
- **Three-phase stacking**: [phase_a_buses, phase_b_buses, phase_c_buses]
  - Example: Bus 5 phase B → index 5 + 39 = 44 in stacked array

### 2.3 Sparse Matrix Format

All admittance and Jacobian matrices stored in **CSR (Compressed Sparse Row)** format:

```
Y_sparse = (data, indices, indptr)
Y[i,j] = data[k] where k ∈ [indptr[i], indptr[i+1]) and indices[k] = j
```

### 2.4 Expandability Features

#### Adding New Learnable Objects

```python
# Pseudo-code for adding new object
with h5py.File('grid.h5', 'a') as f:
    # 1. Update registry
    registry = f['learnable_objects/object_registry']
    new_id = f"ibr_bus{node_id}_{timestamp}"
    
    # 2. Create object group
    obj_group = f.create_group(f'learnable_objects/{new_id}')
    
    # 3. Initialize structure with metadata
    # 4. Set initial parameters
```

#### Version Migration

- Store format version in `/metadata/version`
- Implement migration functions for version upgrades
- Maintain backward compatibility for reading

---

## 3. Port-Hamiltonian KAN Theory

### 3.1 Port-Hamiltonian Systems

#### 3.1.1 Mathematical Formulation

A Port-Hamiltonian system has the structure:

```
dx/dt = [J(x,u) - R(x,u)]∇H(x) + g(x)u
y = h(x, u)
```

**Where:**
- **x**: State vector (internal states of IBR/load)
- **u**: Input vector (grid interface: V_d, V_q, ω)
- **y**: Output vector (to grid: P, Q or I_d, I_q)
- **H(x)**: Hamiltonian (total energy function)
- **J(x,u)**: Interconnection matrix (skew-symmetric: J^T = -J)
- **R(x,u)**: Dissipation matrix (positive semi-definite: R ≥ 0)
- **g(x)**: Input matrix
- **h(x,u)**: Output map

#### 3.1.2 Key Properties

**Energy Balance:**
```
dH/dt = ∇H^T dx/dt 
      = ∇H^T [J - R]∇H + ∇H^T g u
      = -∇H^T R ∇H + ∇H^T g u
        └─────┬─────┘   └────┬────┘
        dissipation    power input
```

**Passivity:** If `∇H^T g u ≤ 0` (power supplied by grid), then `dH/dt ≤ 0`
→ System cannot generate energy → **Stable!**

#### 3.1.3 Why This Matters for Power Systems

1. **Guaranteed stability**: Energy dissipates over time
2. **Physical interpretability**: H represents actual energy storage
3. **Modular composition**: Multiple PH systems compose into PH system
4. **Natural for power grids**: Kirchhoff's laws have PH structure

### 3.2 Kolmogorov-Arnold Networks (KAN)

#### 3.2.1 Mathematical Representation

Classical Kolmogorov-Arnold theorem states any continuous multivariate function can be written as:

```
f(x₁, ..., xₙ) = Σ_{q=1}^{2n+1} Φ_q(Σ_{p=1}^n φ_{q,p}(x_p))
```

**Modern KAN Interpretation:**
- Replace fixed functions φ and Φ with **learnable univariate functions**
- Represent using **B-splines** or other smooth basis
- More efficient than MLPs for certain function classes

#### 3.2.2 KAN Architecture for Power Systems

```
For Hamiltonian H(x):
H(x) = Σ_{q=1}^Q Φ_q(Σ_{p=1}^{state_dim} φ_{q,p}(x_p))
```

**Advantages:**
- **Efficient**: Fewer parameters than MLP for same accuracy
- **Interpretable**: Can extract symbolic form
- **Smooth**: B-spline basis ensures smoothness
- **Physics-friendly**: Natural for energy functions

#### 3.2.3 B-Spline Basis Functions

A B-spline of order `k` with knots `t = [t_0, ..., t_m]`:

```
B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0
B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) B_{i,k-1}(x) 
           + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) B_{i+1,k-1}(x)
```

**Learnable parameters:**
- Knot positions `t_i` (can be fixed or learned)
- Spline coefficients `c_i`

Function evaluation:
```
f(x) = Σ_i c_i B_i(x)
```

### 3.3 Combining PH + KAN

#### 3.3.1 PH-KAN Architecture

```
State Dynamics:
dx/dt = [J_KAN(x,u) - R_KAN(x,u)] ∇H_KAN(x) + g_KAN(x)u

Where each component is learned:
- H_KAN(x): Energy (KAN with 1 output)
- J_KAN(x,u): Interconnection (NN constrained to be skew-symmetric)
- R_KAN(x,u): Dissipation (NN constrained to be positive definite)
- g_KAN(x): Input coupling (standard NN)

Output:
y = h_KAN(x, u)  (KAN with output_dim outputs)
```

#### 3.3.2 Constraints Implementation

**Skew-Symmetry for J:**
```python
# Method 1: Explicit construction
W = learnable_matrix(n, n)
J = (W - W.T) / 2  # Guaranteed skew-symmetric

# Method 2: Use Lie algebra parameterization
# J is always in the space of skew-symmetric matrices
```

**Positive Definiteness for R:**
```python
# Method 1: Cholesky factorization
L = learnable_lower_triangular(n, n)
R = L @ L.T  # Guaranteed PSD

# Method 2: Eigenvalue decomposition
Q = learnable_orthogonal(n, n)
λ = relu(learnable_vector(n))  # Non-negative eigenvalues
R = Q @ diag(λ) @ Q.T
```

#### 3.3.3 Training Objectives

```
Total Loss = λ₁·L_trajectory + λ₂·L_physics + λ₃·L_passivity + λ₄·L_freq

Where:
L_trajectory = ||y_pred - y_true||²  (Match observed data)
L_physics = ||power_balance_violation||²  (Kirchhoff's laws)
L_passivity = Σ relu(dH/dt)  (Penalize energy generation)
L_freq = ||Z_learned(ω) - Z_measured(ω)||²  (Match impedance if available)
```

---

## 4. Learnable Dynamic Graph Objects (LDGO)

### 4.1 LDGO Base Class Specification

#### 4.1.1 Core Attributes

```python
class LearnableDynamicGraphObject:
    """
    A modular, attachable object for learning component dynamics
    """
    # Identity
    object_id: str              # Unique identifier
    node_id: str                # Attached grid node
    component_type: str         # 'ibr', 'dynamic_load', 'storage'
    
    # Dimensions
    state_dim: int              # Internal state dimension
    input_dim: int              # Grid interface inputs (typically 3-4)
    output_dim: int             # Grid interface outputs (typically 2)
    
    # Neural Network Components
    hamiltonian_net: KANNetwork           # H(x)
    interconnection_net: SkewSymmetricNN  # J(x,u)
    dissipation_net: PositiveDefiniteNN   # R(x,u)
    input_coupling_net: StandardNN        # g(x)
    output_map_net: KANNetwork            # h(x,u)
    control_net: KANNetwork               # u_ctrl(x,u) for IBRs
    
    # Physics Priors
    physics_prior: Dict         # Known constraints/initializations
    enforce_passivity: bool     # Whether to enforce dH/dt ≤ 0
    
    # State
    current_state: Tensor       # x(t)
    operating_point: Dict       # Linearization point
```

#### 4.1.2 Required Methods

```python
def forward(t: float, x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute state derivative and output
    
    Args:
        t: current time
        x: state vector [state_dim]
        u: grid inputs [input_dim] = [V_d, V_q, ω, ...]
    
    Returns:
        dx_dt: state derivative [state_dim]
        y: output [output_dim] = [P, Q] or [I_d, I_q]
    """

def linearize(x0: Tensor, u0: Tensor) -> Tuple[Tensor, ...]:
    """
    Extract linear state-space model at operating point
    
    Returns:
        A: state matrix [state_dim × state_dim]
        B: input matrix [state_dim × input_dim]
        C: output matrix [output_dim × state_dim]
        D: feedthrough matrix [output_dim × input_dim]
    """

def compute_impedance(frequencies: List[float]) -> Tensor:
    """
    Compute frequency-domain impedance Z(jω)
    
    Uses: Z(s) = C(sI - A)^(-1)B + D
    
    Returns:
        Z: complex impedance [len(frequencies) × 2 × 2]  (dq frame)
    """

def enforce_passivity_constraint() -> float:
    """
    Compute passivity loss term
    
    Returns:
        loss: penalty for dH/dt > 0
    """

def extract_symbolic() -> Dict[str, str]:
    """
    Extract symbolic expressions from KAN networks
    
    Returns:
        expressions: Dict with 'hamiltonian', 'dynamics', 'output'
    """

def save_to_h5(h5_file: h5py.File, group_path: str):
    """Save object state to HDF5"""

def load_from_h5(h5_file: h5py.File, group_path: str):
    """Load object state from HDF5"""
```

### 4.2 Specific Component Types

#### 4.2.1 Inverter-Based Resource (IBR)

**State Vector (example):**
```
x_ibr = [
    i_d,        # d-axis current (A)
    i_q,        # q-axis current (A)
    φ_PLL,      # PLL phase (rad)
    ω_PLL,      # PLL frequency (rad/s)
    v_dc,       # DC-link voltage (V)
    ξ_P,        # Active power integral state
    ξ_Q         # Reactive power integral state
]
state_dim = 7
```

**Input Vector:**
```
u_ibr = [
    V_d,        # Grid d-axis voltage (V)
    V_q,        # Grid q-axis voltage (V)
    ω_grid      # Grid frequency (rad/s)
]
input_dim = 3
```

**Output Vector:**
```
y_ibr = [
    P,          # Active power (W)
    Q           # Reactive power (VAr)
]
output_dim = 2
```

**Physics Prior (if available):**
```python
physics_prior = {
    'H0': 'L_f/2 * (i_d^2 + i_q^2) + C_dc/2 * v_dc^2',  # Initial energy
    'rated_power_MVA': 2.0,
    'voltage_limits': (0.9, 1.1),
    'current_limit': 1.2,
    'control_type': 'grid_following'  # or 'grid_forming'
}
```

#### 4.2.2 Dynamic Load

**State Vector:**
```
x_load = [
    P_transient,    # Transient active power state
    Q_transient     # Transient reactive power state
]
state_dim = 2
```

**Input Vector:**
```
u_load = [
    V_magnitude,    # Voltage magnitude (pu)
    frequency       # Frequency (pu)
]
input_dim = 2
```

**Output Vector:**
```
y_load = [
    P,              # Active power (MW)
    Q               # Reactive power (MVAR)
]
output_dim = 2
```

**Physics Prior:**
```python
physics_prior = {
    'H0': 'P_0^2/(2*K_p) + Q_0^2/(2*K_q)',  # Quadratic energy
    'voltage_exponent_p': 1.5,  # P ∝ V^α
    'voltage_exponent_q': 2.0,  # Q ∝ V^β
    'time_constant': 0.05       # seconds
}
```

#### 4.2.3 Energy Storage System (ESS)

**State Vector:**
```
x_ess = [
    SoC,            # State of charge (0-1)
    i_d,            # d-axis current
    i_q,            # q-axis current
    ξ_power         # Power control integral state
]
state_dim = 4
```

**Additional Constraints:**
- `0 ≤ SoC ≤ 1`
- `dSoC/dt = -P/(E_capacity)`

### 4.3 Network Architecture Details

#### 4.3.1 KANNetwork Structure

```python
class KANNetwork(nn.Module):
    """
    Kolmogorov-Arnold Network for learning smooth functions
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_inner_functions: int = None,  # Default: input_dim
        spline_order: int = 3,            # Cubic splines
        num_knots: int = 10,
        grid_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        # Inner univariate functions φ_{q,p}(x_p)
        self.inner_functions = nn.ModuleList([
            BSplineFunction(
                num_knots=num_knots,
                spline_order=spline_order,
                grid_range=grid_range
            )
            for _ in range(input_dim)
        ])
        
        # Outer univariate functions Φ_q
        self.outer_functions = nn.ModuleList([
            BSplineFunction(
                num_knots=num_knots,
                spline_order=spline_order,
                grid_range=grid_range
            )
            for _ in range(output_dim)
        ])
        
        # Mixing weights
        self.mixing_weights = nn.Parameter(
            torch.randn(output_dim, hidden_dim, input_dim)
        )
```

#### 4.3.2 BSplineFunction

```python
class BSplineFunction(nn.Module):
    """
    Learnable B-spline univariate function
    """
    def __init__(
        self,
        num_knots: int,
        spline_order: int,
        grid_range: Tuple[float, float],
        learn_knots: bool = False
    ):
        # Initialize uniform knot vector
        knots = torch.linspace(grid_range[0], grid_range[1], num_knots)
        
        if learn_knots:
            self.knots = nn.Parameter(knots)
        else:
            self.register_buffer('knots', knots)
        
        # Learnable coefficients for each basis function
        num_basis = num_knots - spline_order - 1
        self.coefficients = nn.Parameter(torch.randn(num_basis) * 0.1)
        
        self.spline_order = spline_order
    
    def forward(self, x: Tensor) -> Tensor:
        # Evaluate B-spline basis at x
        basis_values = self._evaluate_basis(x)  # [batch, num_basis]
        
        # Linear combination
        output = basis_values @ self.coefficients  # [batch]
        
        return output
    
    def _evaluate_basis(self, x: Tensor) -> Tensor:
        """
        Evaluate all B-spline basis functions at x
        Uses Cox-de Boor recursion formula (differentiable)
        """
        # Implementation using torch operations for autodiff
        # ...
        
    def to_symbolic(self) -> str:
        """
        Convert to symbolic polynomial (approximate)
        """
        # For each spline segment, extract polynomial coefficients
        # Return as string in SymPy format
        pass
```

#### 4.3.3 SkewSymmetricNetwork

```python
class SkewSymmetricNetwork(nn.Module):
    """
    Neural network constrained to output skew-symmetric matrices
    J^T = -J
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        self.state_dim = state_dim
        
        # Learn upper triangular part only
        num_params = state_dim * (state_dim - 1) // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_params)
        )
    
    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        # Encode to upper triangular parameters
        xu = torch.cat([x, u], dim=-1)
        upper_params = self.encoder(xu)
        
        # Build skew-symmetric matrix
        J = torch.zeros(self.state_dim, self.state_dim)
        
        # Fill upper triangle
        triu_indices = torch.triu_indices(self.state_dim, self.state_dim, offset=1)
        J[triu_indices[0], triu_indices[1]] = upper_params
        
        # Make skew-symmetric: J = J - J^T
        J = J - J.T
        
        return J
```

#### 4.3.4 PositiveDefiniteNetwork

```python
class PositiveDefiniteNetwork(nn.Module):
    """
    Neural network constrained to output positive semi-definite matrices
    R ≥ 0 (all eigenvalues ≥ 0)
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        self.state_dim = state_dim
        
        # Learn Cholesky factor
        num_params = state_dim * (state_dim + 1) // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params)
        )
    
    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        # Encode to Cholesky parameters
        xu = torch.cat([x, u], dim=-1)
        L_params = self.encoder(xu)
        
        # Build lower triangular matrix L
        L = torch.zeros(self.state_dim, self.state_dim)
        
        # Fill lower triangle (with positive diagonal)
        tril_indices = torch.tril_indices(self.state_dim, self.state_dim)
        
        # Diagonal elements: use softplus to ensure > 0
        diag_indices = torch.arange(self.state_dim)
        L[diag_indices, diag_indices] = F.softplus(L_params[:self.state_dim])
        
        # Off-diagonal elements
        L[tril_indices[0], tril_indices[1]] = L_params[self.state_dim:]
        
        # R = L @ L^T is guaranteed PSD
        R = L @ L.T
        
        return R
```

---

## 5. Integration Architecture

### 5.1 System-Level Integration

```python
class PowerGridWithLearnableDynamics:
    """
    Complete power grid with static admittance network + learnable objects
    """
    def __init__(self, h5_path: str):
        # Load static graph structure
        self.static_graph = self._load_static_graph(h5_path)
        
        # Load learnable objects
        self.learnable_objects: Dict[str, LDGO] = {}
        self._load_learnable_objects(h5_path)
        
        # Algebraic constraints (Kirchhoff's laws)
        self.Y_matrix = self._load_admittance_matrix(h5_path)
        
        # Solver configuration
        self.solver_config = {
            'load_flow': 'newton_raphson',
            'rms_simulation': 'implicit_euler',
            'time_step_ms': 1.0
        }
```

### 5.2 DAE System Formulation

#### 5.2.1 State Vector Organization

```
x_total = [
    x_ldgo_1,      # States of learnable object 1
    x_ldgo_2,      # States of learnable object 2
    ...,
    x_ldgo_M,      # States of learnable object M
    V_1^a, θ_1^a,  # Algebraic: bus 1 phase a voltage
    V_1^b, θ_1^b,  # Algebraic: bus 1 phase b voltage
    V_1^c, θ_1^c,  # Algebraic: bus 1 phase c voltage
    ...,
    V_N^c, θ_N^c   # Algebraic: bus N phase c voltage
]
```

**Dimensions:**
- Differential states: `Σ state_dim_i` (all LDGO states)
- Algebraic states: `2 × 3 × N` (voltage magnitude & angle, 3 phases, N buses)

#### 5.2.2 DAE Equations

```
Differential equations (learnable objects):
dx_i/dt = f_i(x_i, u_i(V, θ))    for i = 1, ..., M

Algebraic equations (power flow):
P_inj(V, θ) - P_ldgo(x, V, θ) - P_load = 0
Q_inj(V, θ) - Q_ldgo(x, V, θ) - Q_load = 0
```

Where:
- `P_inj, Q_inj`: Injections from admittance network
- `P_ldgo, Q_ldgo`: Outputs from learnable objects
- `P_load, Q_load`: Static loads

### 5.3 Simulation Workflow

#### 5.3.1 Load Flow (Steady-State)

```python
def solve_load_flow(self, method='newton_raphson', tolerance=1e-6):
    """
    Solve steady-state power flow with learnable objects
    """
    # 1. Initialize voltages
    V, θ = self._initialize_voltages()
    
    # 2. Iterative solution
    for iteration in range(max_iterations):
        # 2a. Get inputs to learnable objects from current voltages
        for obj_id, ldgo in self.learnable_objects.items():
            u_i = self._compute_grid_interface(obj_id, V, θ)
            
            # Solve for steady-state: dx/dt = 0
            x_ss_i = ldgo.solve_steady_state(u_i)
            
            # Get power injections
            P_i, Q_i = ldgo.compute_output(x_ss_i, u_i)
        
        # 2b. Update power balance equations
        mismatch = self._compute_power_mismatch(V, θ, P_ldgo, Q_ldgo)
        
        # 2c. Check convergence
        if max(abs(mismatch)) < tolerance:
            break
        
        # 2d. Newton-Raphson update
        J = self._compute_jacobian(V, θ)
        ΔV, Δθ = solve(J, -mismatch)
        V += ΔV
        θ += Δθ
    
    return V, θ, converged
```

#### 5.3.2 RMS Simulation (Time-Domain)

```python
def simulate_rms(
    self,
    t_span: Tuple[float, float],
    x0: Dict[str, Tensor],
    dt: float = 0.001
):
    """
    RMS (phasor) time-domain simulation
    """
    from scipy.integrate import solve_ivp
    
    # Pack initial conditions
    x0_packed = self._pack_states(x0)
    
    def dae_residual(t, x_aug):
        """
        Compute DAE residual:
        F(dx/dt, x, V, θ, t) = 0
        """
        # Unpack
        x_diff = x_aug[:n_diff]  # Differential states
        V_θ = x_aug[n_diff:]      # Algebraic states
        
        # Differential equations
        dx_dt = []
        P_ldgo = []
        Q_ldgo = []
        
        for obj_id, ldgo in self.learnable_objects.items():
            x_i = self._extract_state(x_diff, obj_id)
            u_i = self._compute_interface(V_θ, obj_id)
            
            dx_i, y_i = ldgo.forward(t, x_i, u_i)
            
            dx_dt.append(dx_i)
            P_ldgo.append(y_i[0])
            Q_ldgo.append(y_i[1])
        
        # Algebraic equations (power balance)
        power_balance = self._power_balance_residual(V_θ, P_ldgo, Q_ldgo)
        
        return torch.cat([torch.cat(dx_dt), power_balance])
    
    # Solve DAE
    solution = solve_ivp(
        dae_residual,
        t_span,
        x0_packed,
        method='Radau',  # Implicit solver for DAEs
        dense_output=True,
        max_step=dt
    )
    
    return solution
```

#### 5.3.3 Grid Interface Computation

```python
def _compute_grid_interface(self, obj_id: str, V: Tensor, θ: Tensor) -> Tensor:
    """
    Compute inputs to learnable object from grid state
    
    For IBR at bus i, phase a:
    u = [V_d, V_q, ω]
    
    Where V_d, V_q are in synchronous reference frame
    """
    obj = self.learnable_objects[obj_id]
    node_id = obj.attached_node
    phase = obj.attached_phase
    
    # Get voltage phasor at node
    V_mag = V[node_id, phase]
    θ_angle = θ[node_id, phase]
    
    # Transform to dq frame
    ω_grid = 2 * π * 60  # rad/s
    
    if obj.component_type == 'ibr':
        # Assume PLL tracks grid phase
        θ_pll = θ_angle  # Simplified
        
        V_d = V_mag * cos(θ_angle - θ_pll)
        V_q = V_mag * sin(θ_angle - θ_pll)
        
        u = torch.tensor([V_d, V_q, ω_grid])
    
    elif obj.component_type == 'dynamic_load':
        u = torch.tensor([V_mag, ω_grid / (2*π*60)])  # [V_pu, f_pu]
    
    return u
```

---

## 6. Training and Learning

### 6.1 Training Data Requirements

#### 6.1.1 Data Sources

**Option 1: High-Fidelity Simulation**
- Use PSCAD/EMTDC or Simulink for detailed IBR model
- Generate time-series data for various scenarios
- Extract voltage, current, power trajectories

**Option 2: Hardware-in-the-Loop (HIL)**
- Real IBR connected to grid simulator
- Record measurements during disturbances
- Most accurate but expensive

**Option 3: Field Measurements**
- PMU data from actual grid
- Requires synchronization and data cleaning
- May have limited scenario coverage

#### 6.1.2 Scenario Coverage

Required scenarios for robust learning:

```python
training_scenarios = {
    'steady_state': {
        'voltage_range': (0.9, 1.1),  # pu
        'loading_levels': [0.3, 0.5, 0.7, 0.9, 1.0],
        'power_factor': [-0.9, -0.5, 0.0, 0.5, 0.9]
    },
    
    'disturbances': {
        'voltage_steps': [-0.1, -0.05, 0.05, 0.1],  # pu
        'frequency_ramps': [-0.5, -0.2, 0.2, 0.5],  # Hz/s
        'load_steps': [0.1, 0.2, 0.5],  # pu
    },
    
    'faults': {
        'fault_types': ['3phase', 'SLG', 'LL'],
        'fault_durations_ms': [50, 100, 150],
        'voltage_dips': [0.0, 0.2, 0.5, 0.8]  # Remaining voltage
    },
    
    'topology_changes': {
        'line_outages': ['line_5_6', 'line_10_11'],
        'switching_events': True
    }
}
```

#### 6.1.3 Data Format in HDF5

```
training_data.h5
├─ scenarios/
│  ├─ scenario_{id}/
│  │  ├─ metadata/
│  │  │  ├─ scenario_type (string)  # 'voltage_step', 'fault', etc.
│  │  │  ├─ description (string)
│  │  │  └─ parameters (group)      # Scenario-specific params
│  │  │
│  │  ├─ time_series/
│  │  │  ├─ time (float64, shape=(T,))
│  │  │  ├─ V_d (float64, shape=(T,))  # Grid d-axis voltage
│  │  │  ├─ V_q (float64, shape=(T,))
│  │  │  ├─ ω (float64, shape=(T,))
│  │  │  ├─ I_d (float64, shape=(T,))  # Measured currents
│  │  │  ├─ I_q (float64, shape=(T,))
│  │  │  ├─ P (float64, shape=(T,))    # Measured power
│  │  │  ├─ Q (float64, shape=(T,))
│  │  │  └─ internal_states (float64, shape=(T, state_dim))  # If available
│  │  │
│  │  └─ labels/
│  │     ├─ stable (bool)
│  │     └─ max_voltage_deviation (float64)
```

### 6.2 Loss Function Design

```python
class PHKANLoss(nn.Module):
    """
    Physics-informed loss for Port-Hamiltonian KAN training
    """
    def __init__(
        self,
        λ_trajectory: float = 1.0,
        λ_physics: float = 0.1,
        λ_passivity: float = 0.05,
        λ_frequency: float = 0.02,
        λ_regularization: float = 0.001
    ):
        self.weights = {
            'trajectory': λ_trajectory,
            'physics': λ_physics,
            'passivity': λ_passivity,
            'frequency': λ_frequency,
            'regularization': λ_regularization
        }
    
    def forward(
        self,
        ldgo: LearnableDynamicGraphObject,
        data_batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total loss
        """
        losses = {}
        
        # 1. Trajectory matching loss
        t = data_batch['time']
        x_true = data_batch['states']  # If available
        u = data_batch['inputs']       # [V_d, V_q, ω]
        y_true = data_batch['outputs']  # [P, Q]
        
        # Simulate forward
        x_pred = self._integrate(ldgo, t, u, x0=x_true[0])
        y_pred = ldgo.compute_output(x_pred, u)
        
        losses['trajectory'] = F.mse_loss(y_pred, y_true)
        
        # 2. Physics constraint loss (power balance)
        P_pred, Q_pred = y_pred[:, 0], y_pred[:, 1]
        V_d, V_q = u[:, 0], u[:, 1]
        
        # Power balance: P = V_d * I_d + V_q * I_q
        # (verify consistency if currents available)
        if 'I_d' in data_batch and 'I_q' in data_batch:
            P_from_VI = V_d * data_batch['I_d'] + V_q * data_batch['I_q']
            losses['physics'] = F.mse_loss(P_pred, P_from_VI)
        
        # 3. Passivity loss (energy dissipation)
        passivity_violations = []
        for i in range(len(t) - 1):
            x_i = x_pred[i]
            u_i = u[i]
            dx_i, _ = ldgo.forward(t[i], x_i, u_i)
            
            H_i = ldgo.hamiltonian_net(x_i)
            grad_H_i = torch.autograd.grad(H_i, x_i, create_graph=True)[0]
            dH_dt = (grad_H_i * dx_i).sum()
            
            # Penalize if dH/dt > 0 (energy creation)
            passivity_violations.append(F.relu(dH_dt))
        
        losses['passivity'] = torch.stack(passivity_violations).mean()
        
        # 4. Frequency domain loss (if impedance data available)
        if 'impedance_data' in data_batch:
            freqs = data_batch['impedance_data']['frequencies']
            Z_measured = data_batch['impedance_data']['impedance']
            
            # Compute impedance from linearized model
            A, B, C, D = ldgo.linearize(x_true[0], u[0])
            Z_predicted = self._compute_impedance(freqs, A, B, C, D)
            
            losses['frequency'] = F.mse_loss(Z_predicted, Z_measured)
        
        # 5. Regularization (prevent overfitting)
        l2_reg = sum(p.pow(2).sum() for p in ldgo.parameters())
        losses['regularization'] = l2_reg
        
        # Total loss
        total_loss = sum(
            self.weights[key] * value 
            for key, value in losses.items()
        )
        
        return total_loss, losses
```

### 6.3 Training Algorithm

```python
class PHKANTrainer:
    """
    Trainer for Port-Hamiltonian KAN objects
    """
    def __init__(
        self,
        ldgo: LearnableDynamicGraphObject,
        loss_fn: PHKANLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ):
        self.ldgo = ldgo
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.history = {
            'epoch': [],
            'total_loss': [],
            'trajectory_loss': [],
            'physics_loss': [],
            'passivity_loss': [],
            'validation_loss': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.ldgo.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Compute loss
            total_loss, loss_dict = self.loss_fn(self.ldgo, batch)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(self.ldgo.parameters(), max_norm=1.0)
            
            # Update
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validation"""
        self.ldgo.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                total_loss, _ = self.loss_fn(self.ldgo, batch)
                val_loss += total_loss.item()
        
        return val_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_path: str,
        early_stopping_patience: int = 20
    ):
        """
        Full training loop with early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Save history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(train_loss)
            self.history['validation_loss'].append(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint(save_path, epoch, val_loss)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}")
        
        # Load best model
        self._load_checkpoint(save_path)
        
        return self.history
    
    def _save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint to HDF5"""
        with h5py.File(path, 'a') as f:
            # Create checkpoint group
            ckpt_group = f.create_group(f'checkpoints/epoch_{epoch}')
            
            # Save model state
            self.ldgo.save_to_h5(f, ckpt_group.name)
            
            # Save training state
            ckpt_group.attrs['epoch'] = epoch
            ckpt_group.attrs['loss'] = loss
            ckpt_group.attrs['timestamp'] = datetime.now().isoformat()
    
    def _load_checkpoint(self, path: str):
        """Load best checkpoint"""
        with h5py.File(path, 'r') as f:
            # Find best checkpoint
            checkpoints = f['checkpoints']
            best_ckpt = min(checkpoints.keys(), 
                           key=lambda k: checkpoints[k].attrs['loss'])
            
            # Load
            self.ldgo.load_from_h5(f, f'checkpoints/{best_ckpt}')
```

### 6.4 Curriculum Learning Strategy

```python
class CurriculumTrainer(PHKANTrainer):
    """
    Curriculum learning: train on easy scenarios first, then harder
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.curriculum_stages = [
            {
                'name': 'steady_state',
                'scenarios': ['voltage_sweep', 'power_sweep'],
                'epochs': 50,
                'complexity': 1
            },
            {
                'name': 'small_disturbances',
                'scenarios': ['small_voltage_steps', 'small_frequency_ramps'],
                'epochs': 100,
                'complexity': 2
            },
            {
                'name': 'large_disturbances',
                'scenarios': ['large_voltage_steps', 'load_rejection'],
                'epochs': 150,
                'complexity': 3
            },
            {
                'name': 'faults',
                'scenarios': ['balanced_faults', 'unbalanced_faults'],
                'epochs': 200,
                'complexity': 4
            }
        ]
    
    def train_curriculum(self, data_registry: Dict[str, DataLoader]):
        """
        Train through curriculum stages
        """
        for stage in self.curriculum_stages:
            print(f"\n=== Curriculum Stage: {stage['name']} ===")
            
            # Create data loader for this stage
            stage_loader = self._create_stage_loader(
                data_registry, 
                stage['scenarios']
            )
            
            # Train for this stage
            self.train(
                train_loader=stage_loader['train'],
                val_loader=stage_loader['val'],
                num_epochs=stage['epochs'],
                save_path=f"checkpoints/stage_{stage['name']}.h5"
            )
            
            # Validate on all previous stages
            self._validate_on_all_stages(data_registry, stage['complexity'])
```

---

## 7. Analysis Tools

### 7.1 Small-Signal Stability Analysis

```python
class SmallSignalAnalyzer:
    """
    Linearize system and analyze eigenvalues
    """
    def __init__(self, grid: PowerGridWithLearnableDynamics):
        self.grid = grid
    
    def analyze_at_operating_point(
        self,
        operating_point: Dict[str, Tensor]
    ) -> Dict[str, any]:
        """
        Perform small-signal stability analysis
        
        Returns:
            analysis_results: Dict with eigenvalues, modes, etc.
        """
        # 1. Linearize each learnable object
        A_blocks = []
        B_blocks = []
        C_blocks = []
        D_blocks = []
        
        state_offset = 0
        for obj_id, ldgo in self.grid.learnable_objects.items():
            x0 = operating_point[obj_id]['state']
            u0 = operating_point[obj_id]['input']
            
            A_i, B_i, C_i, D_i = ldgo.linearize(x0, u0)
            
            A_blocks.append({
                'obj_id': obj_id,
                'offset': state_offset,
                'dim': ldgo.state_dim,
                'matrix': A_i
            })
            
            state_offset += ldgo.state_dim
        
        # 2. Linearize network equations
        V0, θ0 = operating_point['grid_voltages']
        J_network = self.grid.compute_power_flow_jacobian(V0, θ0)
        
        # 3. Assemble full system matrix
        # System: [dx/dt] = [A_ldgo    B_coupling] [x  ]
        #         [  0  ]   [C_coupling  J_network] [V,θ]
        
        n_diff = state_offset  # Total differential states
        n_alg = 2 * 3 * self.grid.num_buses  # Algebraic states
        
        # Build coupled system matrix (using implicit function theorem)
        # Effective A_eff = A - B * J_network^(-1) * C
        
        A_full = self._assemble_system_matrix(
            A_blocks, B_blocks, C_blocks, J_network
        )
        
        # 4. Compute eigenvalues
        eigenvalues, eigenvectors = torch.linalg.eig(A_full)
        
        # 5. Identify modes
        modes = self._identify_modes(eigenvalues, eigenvectors)
        
        # 6. Compute stability metrics
        metrics = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'modes': modes,
            'is_stable': torch.all(eigenvalues.real < 0),
            'stability_margin': eigenvalues.real.max(),
            'critical_modes': self._find_critical_modes(eigenvalues),
            'damping_ratios': self._compute_damping_ratios(eigenvalues),
            'oscillation_frequencies': eigenvalues.imag / (2 * np.pi)
        }
        
        return metrics
    
    def _identify_modes(
        self, 
        eigenvalues: Tensor, 
        eigenvectors: Tensor
    ) -> List[Dict]:
        """
        Classify eigenvalues into physical modes
        """
        modes = []
        
        for i, (λ, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # Extract real and imaginary parts
            σ = λ.real.item()
            ω = λ.imag.item()
            
            # Classify mode type
            if abs(ω) < 1e-6:
                mode_type = 'non_oscillatory'
                freq_hz = 0.0
            else:
                mode_type = 'oscillatory'
                freq_hz = abs(ω) / (2 * np.pi)
            
            # Damping ratio
            if abs(ω) > 1e-6:
                ζ = -σ / np.sqrt(σ**2 + ω**2)
            else:
                ζ = 1.0 if σ < 0 else -1.0
            
            # Dominant states (participation factors)
            participation = (v.abs() / v.abs().sum()).real
            dominant_states = torch.topk(participation, k=3).indices
            
            modes.append({
                'index': i,
                'eigenvalue': λ,
                'type': mode_type,
                'damping_ratio': ζ,
                'frequency_hz': freq_hz,
                'time_constant_s': -1/σ if σ < 0 else float('inf'),
                'dominant_states': dominant_states.tolist(),
                'is_stable': σ < 0
            })
        
        return modes
    
    def plot_eigenvalue_map(self, metrics: Dict, save_path: str = None):
        """
        Plot eigenvalues on complex plane
        """
        import matplotlib.pyplot as plt
        
        eigenvalues = metrics['eigenvalues']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot eigenvalues
        real = eigenvalues.real.numpy()
        imag = eigenvalues.imag.numpy()
        
        # Color by damping ratio
        damping = metrics['damping_ratios'].numpy()
        scatter = ax.scatter(real, imag, c=damping, cmap='RdYlGn', 
                           s=100, alpha=0.7, edgecolors='black')
        
        # Add stability boundary (imaginary axis)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                   label='Stability Boundary')
        
        # Add constant damping ratio curves
        for ζ in [0.05, 0.1, 0.2, 0.3]:
            θ = np.arccos(ζ)
            r = np.linspace(0, max(abs(eigenvalues)), 100)
            ax.plot(r * np.cos(θ), r * np.sin(θ), 'k--', alpha=0.3)
            ax.plot(r * np.cos(θ), -r * np.sin(θ), 'k--', alpha=0.3)
        
        ax.set_xlabel('Real Part (1/s)', fontsize=12)
        ax.set_ylabel('Imaginary Part (rad/s)', fontsize=12)
        ax.set_title('Eigenvalue Map', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.colorbar(scatter, label='Damping Ratio', ax=ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
```

### 7.2 Impedance Scanning

```python
class ImpedanceScanner:
    """
    Compute frequency-domain impedance characteristics
    """
    def __init__(self, grid: PowerGridWithLearnableDynamics):
        self.grid = grid
    
    def scan_node_impedance(
        self,
        node_id: str,
        frequencies: np.ndarray,
        operating_point: Dict[str, Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Compute impedance seen from a node
        
        Z(jω) for grid-following inverter
        Y(jω) for grid-forming inverter
        """
        results = {
            'frequencies': frequencies,
            'Z_dq': [],  # dq-frame impedance matrix
            'Z_magnitude': [],
            'Z_phase': []
        }
        
        # Get learnable object at this node
        ldgo = self.grid.learnable_objects.get(node_id)
        if ldgo is None:
            raise ValueError(f"No learnable object at node {node_id}")
        
        # Linearize at operating point
        x0 = operating_point[node_id]['state']
        u0 = operating_point[node_id]['input']
        A, B, C, D = ldgo.linearize(x0, u0)
        
        # Compute impedance for each frequency
        for f in frequencies:
            ω = 2 * np.pi * f
            s = 1j * ω
            
            # Transfer function: Z(s) = C(sI - A)^(-1)B + D
            sI_minus_A = s * torch.eye(A.shape[0]) - A
            
            try:
                inv_term = torch.linalg.inv(sI_minus_A)
                Z_s = C @ inv_term @ B + D
            except:
                # Singular matrix at this frequency
                Z_s = torch.full((C.shape[0], B.shape[1]), float('nan'))
            
            results['Z_dq'].append(Z_s.numpy())
            results['Z_magnitude'].append(np.abs(Z_s.numpy()))
            results['Z_phase'].append(np.angle(Z_s.numpy(), deg=True))
        
        results['Z_dq'] = np.array(results['Z_dq'])
        results['Z_magnitude'] = np.array(results['Z_magnitude'])
        results['Z_phase'] = np.array(results['Z_phase'])
        
        return results
    
    def plot_impedance_bode(
        self,
        scan_results: Dict[str, np.ndarray],
        save_path: str = None
    ):
        """
        Bode plot of impedance
        """
        import matplotlib.pyplot as plt
        
        freqs = scan_results['frequencies']
        Z_mag = scan_results['Z_magnitude']
        Z_phase = scan_results['Z_phase']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude plot
        ax1.semilogx(freqs, 20*np.log10(Z_mag[:, 0, 0]), 
                     label='Z_dd', linewidth=2)
        ax1.semilogx(freqs, 20*np.log10(Z_mag[:, 1, 1]), 
                     label='Z_qq', linewidth=2)
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.set_title('Impedance Bode Plot', fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend()
        
        # Phase plot
        ax2.semilogx(freqs, Z_phase[:, 0, 0], label='Z_dd', linewidth=2)
        ax2.semilogx(freqs, Z_phase[:, 1, 1], label='Z_qq', linewidth=2)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def assess_stability_margin(
        self,
        Z_source: np.ndarray,
        Z_load: np.ndarray,
        frequencies: np.ndarray
    ) -> Dict[str, float]:
        """
        Assess stability using Nyquist criterion
        
        Ratio Z_load / Z_source should not encircle (-1, 0)
        """
        # Compute impedance ratio
        ratio = Z_load / (Z_source + 1e-10)
        
        # Find minimum distance to (-1, 0)
        distances = np.abs(ratio + 1)
        min_distance = np.min(distances)
        critical_freq = frequencies[np.argmin(distances)]
        
        # Phase margin
        phase_margin = 180 + np.angle(ratio[np.argmin(distances)], deg=True)
        
        # Gain margin
        # Find frequency where phase = -180°
        phase = np.angle(ratio, deg=True)
        crossover_idx = np.argmin(np.abs(phase + 180))
        gain_margin_dB = -20 * np.log10(np.abs(ratio[crossover_idx]))
        
        return {
            'stability_margin': min_distance,
            'critical_frequency_hz': critical_freq,
            'phase_margin_deg': phase_margin,
            'gain_margin_dB': gain_margin_dB,
            'is_stable': min_distance > 0.5  # Rule of thumb
        }
```

### 7.3 Sensitivity Analysis

```python
class SensitivityAnalyzer:
    """
    Analyze sensitivity of learned models to parameters
    """
    def __init__(self, grid: PowerGridWithLearnableDynamics):
        self.grid = grid
    
    def parameter_sensitivity(
        self,
        ldgo: LearnableDynamicGraphObject,
        operating_point: Dict[str, Tensor],
        parameter_variations: Dict[str, float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute sensitivity of outputs to parameters
        
        Sensitivity: ∂y/∂θ where θ are network parameters
        """
        if parameter_variations is None:
            parameter_variations = {
                'voltage': 0.1,  # ±10%
                'frequency': 0.5,  # ±0.5 Hz
                'power': 0.2     # ±20%
            }
        
        x0 = operating_point['state']
        u0 = operating_point['input']
        
        sensitivities = {}
        
        # Compute Jacobian of output w.r.t. inputs
        u0.requires_grad_(True)
        _, y0 = ldgo.forward(0, x0, u0)
        
        # ∂y/∂u
        dy_du = torch.autograd.functional.jacobian(
            lambda u: ldgo.forward(0, x0, u)[1],
            u0
        )
        
        sensitivities['output_to_input'] = dy_du.numpy()
        
        # Vary each input
        for var_name, variation in parameter_variations.items():
            if var_name == 'voltage':
                u_varied = u0.clone()
                u_varied[0] *= (1 + variation)  # Vary V_d
                
                _, y_varied = ldgo.forward(0, x0, u_varied)
                Δy = (y_varied - y0) / (variation * u0[0])
                
                sensitivities[f'{var_name}_sensitivity'] = Δy.detach().numpy()
        
        return sensitivities
```

---

## 8. Implementation Requirements

### 8.1 Software Dependencies

```yaml
# requirements.txt
python: ">=3.9"

core:
  - torch: ">=2.0.0"
  - numpy: ">=1.24.0"
  - scipy: ">=1.10.0"
  - h5py: ">=3.8.0"

graph_learning:
  - torch-geometric: ">=2.3.0"
  - torch-scatter: ">=2.1.0"
  - torch-sparse: ">=0.6.17"

optimization:
  - optuna: ">=3.1.0"  # Hyperparameter tuning

visualization:
  - matplotlib: ">=3.7.0"
  - seaborn: ">=0.12.0"
  - plotly: ">=5.14.0"

utilities:
  - tqdm: ">=4.65.0"
  - pyyaml: ">=6.0"
  - pytest: ">=7.3.0"
```

### 8.2 Code Organization

```
power_grid_learning/
│
├── data/
│   ├── storage/
│   │   ├── h5_writer.py
│   │   ├── h5_reader.py
│   │   └── format_validator.py
│   │
│   └── loaders/
│       ├── pyg_converter.py
│       ├── batch_generator.py
│       └── scenario_loader.py
│
├── graph/
│   ├── core/
│   │   ├── power_grid_graph.py
│   │   ├── three_phase_coupling.py
│   │   └── admittance_builder.py
│   │
│   └── components/
│       ├── learnable_dynamic_object.py
│       ├── ibr_object.py
│       ├── dynamic_load_object.py
│       └── ess_object.py
│
├── neural_networks/
│   ├── kan/
│   │   ├── kan_network.py
│   │   ├── bspline_layer.py
│   │   └── symbolic_extraction.py
│   │
│   ├── ph_structure/
│   │   ├── skew_symmetric_net.py
│   │   ├── positive_definite_net.py
│   │   └── hamiltonian_net.py
│   │
│   └── models/
│       ├── ph_kan_model.py
│       └── hybrid_gnn_kan.py
│
├── training/
│   ├── losses.py
│   ├── trainer.py
│   ├── curriculum_trainer.py
│   └── validation.py
│
├── solvers/
│   ├── load_flow/
│   │   ├── newton_raphson.py
│   │   └── implicit_zbus.py
│   │
│   ├── dynamic/
│   │   ├── dae_solver.py
│   │   ├── implicit_euler.py
│   │   └── bdf_integrator.py
│   │
│   └── linearization/
│       └── jacobian_builder.py
│
├── analysis/
│   ├── small_signal_stability.py
│   ├── impedance_scanner.py
│   ├── sensitivity_analyzer.py
│   └── modal_analysis.py
│
├── utils/
│   ├── validators.py
│   ├── coordinate_transforms.py
│   ├── physics_checks.py
│   └── visualization.py
│
├── tests/
│   ├── test_h5_storage.py
│   ├── test_ph_kan.py
│   ├── test_integration.py
│   └── test_stability.py
│
└── examples/
    ├── ieee39_with_ibr.py
    ├── train_single_ibr.py
    ├── impedance_analysis.py
    └── stability_assessment.py
```

### 8.3 Configuration Management

```yaml
# config/training_config.yaml

model:
  state_dim: 7
  input_dim: 3
  output_dim: 2
  hidden_dim: 64
  
  kan:
    num_knots: 10
    spline_order: 3
    grid_range: [-1.0, 1.0]
    learn_knots: false
  
  physics:
    enforce_passivity: true
    enforce_power_balance: true
    voltage_limits: [0.9, 1.1]
    current_limit_pu: 1.2

training:
  batch_size: 32
  num_epochs: 500
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "reduce_on_plateau"
  
  loss_weights:
    trajectory: 1.0
    physics: 0.1
    passivity: 0.05
    frequency: 0.02
    regularization: 0.001
  
  early_stopping:
    patience: 20
    min_delta: 1e-6
  
  curriculum:
    enabled: true
    stages:
      - name: "steady_state"
        epochs: 50
      - name: "disturbances"
        epochs: 100
      - name: "faults"
        epochs: 200

data:
  h5_path: "data/ieee39_scenarios.h5"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  
  augmentation:
    noise_level: 0.01
    time_shift: true

analysis:
  small_signal:
    enabled: true
    frequency_range: [0.1, 100.0]  # Hz
  
  impedance_scan:
    enabled: true
    frequencies: [0.1, 1, 10, 50, 100, 500, 1000]  # Hz
```

### 8.4 Testing Requirements

```python
# tests/test_ph_kan.py

import pytest
import torch
from power_grid_learning.neural_networks.models import PHKANModel

class TestPHKAN:
    """Unit tests for Port-Hamiltonian KAN"""
    
    def test_skew_symmetry(self):
        """Test that J matrix is skew-symmetric"""
        model = PHKANModel(state_dim=7, input_dim=3, output_dim=2)
        
        x = torch.randn(10, 7)
        u = torch.randn(10, 3)
        
        J = model.interconnection_net(x, u)
        
        # Check J^T = -J
        assert torch.allclose(J, -J.transpose(-2, -1), atol=1e-6)
    
    def test_positive_definiteness(self):
        """Test that R matrix is positive definite"""
        model = PHKANModel(state_dim=7, input_dim=3, output_dim=2)
        
        x = torch.randn(10, 7)
        u = torch.randn(10, 3)
        
        R = model.dissipation_net(x, u)
        
        # Check all eigenvalues >= 0
        eigenvalues = torch.linalg.eigvalsh(R)
        assert torch.all(eigenvalues >= -1e-6)
    
    def test_energy_conservation(self):
        """Test that energy is non-increasing (passivity)"""
        model = PHKANModel(state_dim=7, input_dim=3, output_dim=2)
        model.eval()
        
        x = torch.randn(7, requires_grad=True)
        u = torch.zeros(3)  # No input power
        
        # Compute dH/dt
        dx_dt, _ = model.forward(0, x, u)
        H = model.hamiltonian_net(x)
        grad_H = torch.autograd.grad(H, x, create_graph=True)[0]
        dH_dt = (grad_H * dx_dt).sum()
        
        # Should be <= 0 (energy dissipates)
        assert dH_dt.item() <= 1e-6
    
    def test_linearization(self):
        """Test linearization produces valid state-space model"""
        model = PHKANModel(state_dim=7, input_dim=3, output_dim=2)
        
        x0 = torch.randn(7)
        u0 = torch.randn(3)
        
        A, B, C, D = model.linearize(x0, u0)
        
        # Check dimensions
        assert A.shape == (7, 7)
        assert B.shape == (7, 3)
        assert C.shape == (2, 7)
        assert D.shape == (2, 3)
        
        # Check stability (all eigenvalues have negative real part)
        eigenvalues = torch.linalg.eigvals(A)
        # Note: May not be stable before training
```

### 8.5 Performance Benchmarks

```python
# benchmarks/performance_test.py

import time
import torch
from power_grid_learning.graph import PowerGridWithLearnableDynamics

def benchmark_load_flow():
    """Benchmark load flow solver"""
    grid = PowerGridWithLearnableDynamics("data/ieee39.h5")
    
    start = time.time()
    for _ in range(100):
        grid.solve_load_flow()
    end = time.time()
    
    avg_time_ms = (end - start) / 100 * 1000
    print(f"Load flow average time: {avg_time_ms:.2f} ms")
    
    assert avg_time_ms < 50, "Load flow too slow"

def benchmark_rms_simulation():
    """Benchmark RMS simulation"""
    grid = PowerGridWithLearnableDynamics("data/ieee39.h5")
    
    start = time.time()
    grid.simulate_rms(t_span=(0, 1.0), dt=0.001)
    end = time.time()
    
    simulation_time = end - start
    print(f"1-second RMS simulation time: {simulation_time:.2f} s")
    
    assert simulation_time < 10, "RMS simulation too slow"
```

---

## 9. Usage Examples

### 9.1 Basic Workflow

```python
# examples/basic_workflow.py

from power_grid_learning.data.storage import H5Writer, H5Reader
from power_grid_learning.graph import PowerGridWithLearnableDynamics
from power_grid_learning.graph.components import IBRObject
from power_grid_learning.training import PHKANTrainer, PHKANLoss

# 1. Load grid from HDF5
grid = PowerGridWithLearnableDynamics("data/ieee39.h5")

# 2. Attach learnable IBR object
ibr = IBRObject(
    node_id="Bus_30",
    state_dim=7,
    rated_power_MVA=2.0,
    physics_prior={
        'control_type': 'grid_following',
        'voltage_limits': (0.9, 1.1)
    }
)
grid.attach_learnable_object("Bus_30", ibr)

# 3. Load training data
from power_grid_learning.data.loaders import ScenarioLoader
train_loader = ScenarioLoader(
    h5_path="data/training_scenarios.h5",
    batch_size=32,
    scenarios=['voltage_steps', 'frequency_ramps']
)

# 4. Setup training
loss_fn = PHKANLoss(
    λ_trajectory=1.0,
    λ_physics=0.1,
    λ_passivity=0.05
)

optimizer = torch.optim.Adam(ibr.parameters(), lr=0.001)

trainer = PHKANTrainer(
    ldgo=ibr,
    loss_fn=loss_fn,
    optimizer=optimizer
)

# 5. Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=200,
    save_path="checkpoints/ibr_bus30.h5"
)

# 6. Save trained model to grid HDF5
with h5py.File("data/ieee39.h5", 'a') as f:
    ibr.save_to_h5(f, "learnable_objects/ibr_bus30")

# 7. Perform analysis
from power_grid_learning.analysis import SmallSignalAnalyzer

analyzer = SmallSignalAnalyzer(grid)
operating_point = grid.get_current_operating_point()
stability_results = analyzer.analyze_at_operating_point(operating_point)

print(f"Stability margin: {stability_results['stability_margin']:.4f}")
print(f"Critical modes: {stability_results['critical_modes']}")
```

### 9.2 Advanced: Multi-IBR System

```python
# examples/multi_ibr_system.py

from power_grid_learning.graph import PowerGridWithLearnableDynamics
from power_grid_learning.graph.components import IBRObject
from power_grid_learning.training import CurriculumTrainer

# Load IEEE 39 bus system
grid = PowerGridWithLearnableDynamics("data/ieee39.h5")

# Define IBR locations and ratings
ibr_config = {
    'Bus_30': {'rated_MVA': 2.0, 'type': 'solar'},
    'Bus_33': {'rated_MVA': 1.5, 'type': 'wind'},
    'Bus_37': {'rated_MVA': 1.0, 'type': 'battery'}
}

# Attach multiple IBRs
ibr_objects = {}
for bus_id, config in ibr_config.items():
    ibr = IBRObject(
        node_id=bus_id,
        state_dim=7,
        rated_power_MVA=config['rated_MVA'],
        physics_prior={'resource_type': config['type']}
    )
    grid.attach_learnable_object(bus_id, ibr)
    ibr_objects[bus_id] = ibr

# Joint training with curriculum learning
trainer = CurriculumTrainer(
    grid=grid,
    loss_fn=PHKANLoss(),
    optimizer=torch.optim.Adam(
        [p for ibr in ibr_objects.values() for p in ibr.parameters()],
        lr=0.001
    )
)

# Train through curriculum
data_registry = load_all_scenarios("data/training_scenarios.h5")
trainer.train_curriculum(data_registry)

# Validate system stability
from power_grid_learning.analysis import ImpedanceScanner

scanner = ImpedanceScanner(grid)
frequencies = np.logspace(-1, 3, 100)  # 0.1 Hz to 1000 Hz

for bus_id in ibr_config.keys():
    impedance_results = scanner.scan_node_impedance(
        node_id=bus_id,
        frequencies=frequencies,
        operating_point=grid.get_current_operating_point()
    )
    
    scanner.plot_impedance_bode(
        impedance_results,
        save_path=f"results/impedance_{bus_id}.png"
    )
```

### 9.3 Real-Time Deployment

```python
# examples/real_time_deployment.py

import torch
from power_grid_learning.graph import PowerGridWithLearnableDynamics
from power_grid_learning.solvers import RMSSolver

class RealTimeGridSimulator:
    """
    Real-time grid simulator with learned IBR models
    """
    def __init__(self, h5_path: str, dt_ms: float = 1.0):
        self.grid = PowerGridWithLearnableDynamics(h5_path)
        self.dt = dt_ms / 1000.0  # Convert to seconds
        self.solver = RMSSolver(self.grid, dt=self.dt)
        
        # Load trained models
        self._load_trained_models(h5_path)
        
        # Initialize state
        self.current_state = self.grid.get_initial_state()
        self.time = 0.0
    
    def _load_trained_models(self, h5_path: str):
        """Load all trained learnable objects"""
        with h5py.File(h5_path, 'r') as f:
            for obj_id in f['learnable_objects/object_registry/object_ids']:
                obj_id_str = obj_id.decode() if isinstance(obj_id, bytes) else obj_id
                ldgo = self.grid.learnable_objects[obj_id_str]
                ldgo.load_from_h5(f, f'learnable_objects/{obj_id_str}')
    
    def step(self, disturbances: Dict[str, float] = None) -> Dict[str, any]:
        """
        Advance simulation by one timestep
        
        Args:
            disturbances: External disturbances (e.g., load changes, faults)
        
        Returns:
            state_dict: Current system state
        """
        # Apply disturbances
        if disturbances:
            self._apply_disturbances(disturbances)
        
        # Solve one timestep
        self.current_state = self.solver.step(
            self.time,
            self.current_state,
            self.dt
        )
        
        self.time += self.dt
        
        # Extract quantities of interest
        return self._extract_measurements()
    
    def _apply_disturbances(self, disturbances: Dict[str, float]):
        """Apply external disturbances"""
        for key, value in disturbances.items():
            if key.startswith('load_'):
                node_id = key.replace('load_', '')
                # Modify load at node
                pass
            elif key.startswith('fault_'):
                # Apply fault
                pass
    
    def _extract_measurements(self) -> Dict[str, any]:
        """Extract measurements from current state"""
        measurements = {
            'time': self.time,
            'voltages': {},
            'frequencies': {},
            'powers': {}
        }
        
        # Extract from each learnable object
        for obj_id, ldgo in self.grid.learnable_objects.items():
            x_i = self.current_state[obj_id]
            u_i = self.grid.get_grid_interface(obj_id)
            
            _, y_i = ldgo.forward(self.time, x_i, u_i)
            
            measurements['powers'][obj_id] = {
                'P_MW': y_i[0].item(),
                'Q_MVAR': y_i[1].item()
            }
        
        return measurements
    
    def run(self, duration_s: float, callback=None):
        """
        Run simulation for specified duration
        
        Args:
            duration_s: Simulation duration in seconds
            callback: Optional function called each timestep
        """
        num_steps = int(duration_s / self.dt)
        
        history = []
        
        for step in range(num_steps):
            measurements = self.step()
            history.append(measurements)
            
            if callback:
                callback(measurements)
            
            # Real-time constraint: ensure step completes in dt
            # (In actual deployment, would use timing mechanisms)
        
        return history

# Usage
simulator = RealTimeGridSimulator("data/ieee39_trained.h5", dt_ms=1.0)

# Run 10 second simulation
results = simulator.run(
    duration_s=10.0,
    callback=lambda m: print(f"t={m['time']:.3f}s, P_total={sum(p['P_MW'] for p in m['powers'].values()):.2f} MW")
)
```

---

## 10. Validation and Quality Assurance

### 10.1 Model Validation Checklist

```yaml
validation_checklist:
  
  physics_consistency:
    - name: "Energy conservation"
      test: "dH/dt ≤ 0 for all states"
      tolerance: 1e-6
      
    - name: "Power balance"
      test: "Σ P_gen - Σ P_load - P_loss = 0"
      tolerance: 1e-3  # 1 kW
      
    - name: "Voltage limits"
      test: "0.9 ≤ V ≤ 1.1 pu"
      
    - name: "Current limits"
      test: "|I| ≤ I_max"
  
  numerical_stability:
    - name: "No NaN/Inf"
      test: "All outputs finite"
      
    - name: "Lipschitz continuity"
      test: "||f(x1) - f(x2)|| ≤ L||x1 - x2||"
      
    - name: "Eigenvalue bounds"
      test: "max(Re(λ)) < 0"
  
  accuracy:
    - name: "Trajectory matching"
      test: "RMSE < 5%"
      metric: "root_mean_squared_error"
      
    - name: "Frequency response"
      test: "Impedance error < 10%"
      frequency_range: [0.1, 1000]  # Hz
      
    - name: "Generalization"
      test: "Test set accuracy within 10% of train"
  
  interpretability:
    - name: "Symbolic extraction"
      test: "Can extract readable equations"
      
    - name: "Parameter sensitivity"
      test: "Gradients finite and bounded"
```

### 10.2 Automated Testing Suite

```python
# tests/test_validation.py

import pytest
from power_grid_learning.utils import PhysicsValidator

class TestPhysicsConsistency:
    """Validate physics constraints"""
    
    @pytest.fixture
    def trained_ibr(self):
        # Load trained IBR model
        grid = PowerGridWithLearnableDynamics("data/ieee39_trained.h5")
        return grid.learnable_objects['ibr_bus30']
    
    def test_energy_dissipation(self, trained_ibr):
        """Test that energy never increases without input"""
        validator = PhysicsValidator()
        
        x0 = torch.randn(trained_ibr.state_dim)
        u0 = torch.zeros(trained_ibr.input_dim)  # No input power
        
        # Simulate for 1 second
        t_span = torch.linspace(0, 1, 100)
        trajectory = validator.simulate_trajectory(trained_ibr, t_span, x0, u0)
        
        # Check energy at each timestep
        energies = [
            trained_ibr.hamiltonian_net(trajectory[i]).item()
            for i in range(len(t_span))
        ]
        
        # Energy should decrease or stay constant
        for i in range(len(energies) - 1):
            assert energies[i+1] <= energies[i] + 1e-6, \
                f"Energy increased at t={t_span[i]}: {energies[i]} → {energies[i+1]}"
    
    def test_power_balance(self, trained_ibr):
        """Test power balance at grid interface"""
        x = torch.randn(trained_ibr.state_dim)
        u = torch.randn(trained_ibr.input_dim)
        
        dx, y = trained_ibr.forward(0, x, u)
        
        # Output power
        P_out, Q_out = y[0], y[1]
        
        # Compute internal power from states
        # (This requires knowledge of internal structure)
        # For now, just check that output is bounded
        assert abs(P_out) < trained_ibr.rated_power_MVA * 1.2
        assert abs(Q_out) < trained_ibr.rated_power_MVA * 1.2
    
    def test_stability_margins(self, trained_ibr):
        """Test that linearized system is stable"""
        x0 = torch.randn(trained_ibr.state_dim)
        u0 = torch.randn(trained_ibr.input_dim)
        
        A, B, C, D = trained_ibr.linearize(x0, u0)
        
        # Check eigenvalues
        eigenvalues = torch.linalg.eigvals(A)
        max_real_part = eigenvalues.real.max().item()
        
        assert max_real_part < 0, \
            f"Unstable system: max(Re(λ)) = {max_real_part}"
        
        # Check stability margin
        stability_margin = -max_real_part
        assert stability_margin > 0.1, \
            f"Insufficient stability margin: {stability_margin}"
```

---

## 11. Troubleshooting Guide

### 11.1 Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Training instability** | Loss oscillates or explodes | - Reduce learning rate<br>- Add gradient clipping<br>- Increase passivity loss weight<br>- Check data normalization |
| **Poor generalization** | High train accuracy, low test accuracy | - Add regularization<br>- Use curriculum learning<br>- Collect more diverse scenarios<br>- Reduce model complexity |
| **Slow convergence** | Loss plateaus early | - Increase learning rate<br>- Change optimizer (try AdamW)<br>- Adjust loss weights<br>- Check for dead neurons |
| **Physics violations** | Energy increases, power imbalance | - Increase physics loss weight<br>- Enforce constraints harder<br>- Check network architectures (J, R) |
| **Numerical instabilities** | NaN/Inf values | - Check input/output scaling<br>- Add numerical damping<br>- Use double precision<br>- Regularize matrix inversions |

### 11.2 Debugging Workflow

```python
# utils/debug_tools.py

class PHKANDebugger:
    """Debugging utilities for PH-KAN models"""
    
    @staticmethod
    def check_gradient_flow(model, x, u):
        """Check if gradients are flowing properly"""
        x.requires_grad_(True)
        u.requires_grad_(True)
        
        dx, y = model.forward(0, x, u)
        loss = y.sum()
        
        loss.backward()
        
        # Check for zero gradients
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"⚠️ No gradient for {name}")
            elif param.grad.abs().max() < 1e-8:
                print(f"⚠️ Very small gradient for {name}: {param.grad.abs().max()}")
            else:
                print(f"✓ {name}: grad range [{param.grad.min():.2e}, {param.grad.max():.2e}]")
    
    @staticmethod
    def visualize_energy_landscape(model, x_range, u_fixed):
        """Plot energy landscape H(x)"""
        import matplotlib.pyplot as plt
        
        # Create grid
        x1 = torch.linspace(x_range[0], x_range[1], 50)
        x2 = torch.linspace(x_range[0], x_range[1], 50)
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        
        # Evaluate H(x)
        H_values = torch.zeros_like(X1)
        for i in range(len(x1)):
            for j in range(len(x2)):
                x = torch.zeros(model.state_dim)
                x[0] = X1[i, j]
                x[1] = X2[i, j]
                H_values[i, j] = model.hamiltonian_net(x).item()
        
        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1.numpy(), X2.numpy(), H_values.numpy(), cmap='viridis')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('H(x)')
        ax.set_title('Energy Landscape')
        plt.show()
    
    @staticmethod
    def verify_constraints(model):
        """Verify structural constraints on J and R"""
        x = torch.randn(10, model.state_dim)
        u = torch.randn(10, model.input_dim)
        
        J = model.interconnection_net(x, u)
        R = model.dissipation_net(x, u)
        
        # Check skew-symmetry
        skew_error = (J + J.transpose(-2, -1)).abs().max()
        print(f"Skew-symmetry error: {skew_error:.2e}")
        
        # Check positive definiteness
        min_eigenvalue = torch.linalg.eigvalsh(R).min()
        print(f"Min eigenvalue of R: {min_eigenvalue:.2e}")
        
        if skew_error > 1e-5:
            print("⚠️ J is not skew-symmetric!")
        if min_eigenvalue < -1e-5:
            print("⚠️ R is not positive definite!")
```

---

## 12. Future Extensions

### 12.1 Planned Features

1. **Multi-fidelity modeling**
   - Support different fidelity levels (load flow → RMS → EMT)
   - Automatic model reduction based on time scale

2. **Online learning**
   - Adapt model parameters in real-time
   - Incremental learning from streaming data

3. **Uncertainty quantification**
   - Bayesian neural networks for confidence bounds
   - Ensemble methods for robustness

4. **Distributed learning**
   - Train on multiple GPUs/nodes
   - Federated learning for privacy

5. **Advanced architectures**
   - Attention mechanisms for long-range dependencies
   - Graph transformers for better expressiveness

### 12.2 Research Directions

- **Theoretical guarantees**: Formal proofs of stability, convergence
- **Symbolic regression**: Automatically discover governing equations
- **Transfer learning**: Leverage pre-trained models across grids
- **Explainable AI**: Better interpretability of learned dynamics

---

## 13. References and Resources

### 13.1 Key Papers

1. **Port-Hamiltonian Systems**
   - van der Schaft & Jeltsema (2014). "Port-Hamiltonian Systems Theory: An Introductory Overview"

2. **Kolmogorov-Arnold Networks**
   - Liu et al. (2024). "KAN: Kolmogorov-Arnold Networks"

3. **Physics-Informed Learning**
   - Raissi et al. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems"

4. **Power System Dynamics**
   - Kundur (1994). "Power System Stability and Control"

### 13.2 Online Resources

- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/
- HDF5 for Python: https://docs.h5py.org/
- Control tutorials: http://ctms.engin.umich.edu/

---

## Appendix A: Mathematical Notation

| Symbol | Description |
|--------|-------------|
| x | State vector |
| u | Input vector |
| y | Output vector |
| H(x) | Hamiltonian (energy function) |
| J(x,u) | Interconnection matrix (skew-symmetric) |
| R(x,u) | Dissipation matrix (positive definite) |
| g(x) | Input coupling matrix |
| θ | Neural network parameters |
| A, B, C, D | State-space matrices |
| λ | Eigenvalue |
| ω | Frequency (rad/s) |
| Z(jω) | Impedance |
| ζ | Damping ratio |

---

## Appendix B: File Format Specification

### B.1 HDF5 Attribute Standards

All HDF5 groups should include these standard attributes:

```python
group.attrs['created'] = datetime.now().isoformat()
group.attrs['modified'] = datetime.now().isoformat()
group.attrs['version'] = '1.0'
group.attrs['description'] = 'Brief description'
```

### B.2 Compression Settings

For large datasets, use compression:

```python
f.create_dataset(
    'large_array',
    data=array,
    compression='gzip',
    compression_opts=9,  # Max compression
    chunks=True  # Enable chunking
)
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-19  
**Status:** Complete specification for implementation

This specification provides the complete blueprint for implementing HDF5 storage and Port-Hamiltonian KAN learnable dynamic objects for power grid modeling with neural networks.