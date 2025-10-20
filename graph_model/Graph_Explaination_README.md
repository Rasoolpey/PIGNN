# Graph Model - H5 Storage Format for Power System Dynamics

**Version:** 2.0  
**Date:** October 19, 2025  
**Purpose:** Complete HDF5-based storage for three-phase power grid graphs with RMS dynamic parameters

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [H5 Format Specification](#h5-format-specification)
4. [Usage Guide](#usage-guide)
5. [Dynamic Models](#dynamic-models)
6. [Creating Your Own H5 Files](#creating-your-own-h5-files)

---

## Overview

This package provides a comprehensive HDF5 storage format for power system data, specifically designed for:
-  **Three-phase unbalanced power flow analysis**
-  **RMS (phasor-domain) dynamic simulation**
-  **Contingency analysis and N-1 studies**
-  **Integration with learnable dynamics (PH-KAN)**
-  **ANDES simulator compatibility**

### Key Features

- **Hierarchical Storage**: Organized group structure for efficient access
- **Complete RMS Parameters**: All generator, exciter, and governor dynamics
- **Per-Phase Data**: Full three-phase representation (phase A, B, C)
- **Sparse Matrix Support**: Efficient storage using CSR format
- **Extensible**: Easy to add new parameters and models
- **Version Control**: Format versioning for backward compatibility

### File Information

**Main File:** `Graph_model.h5`
- **Grid:** IEEE 39-bus New England Test System
- **Buses:** 39
- **Lines:** 34
- **Transformers:** 12
- **Generators:** 10 (with complete GENROU dynamics)
- **Base MVA:** 100.0
- **Frequency:** 60.0 Hz
- **Format Version:** 2.0

---

## File Structure

``
Graph_model.h5

 metadata/                                 # System information
    grid_name: "IEEE39_Enhanced"
    base_mva: 100.0
    base_frequency_hz: 60.0
    num_buses: 39
    num_phases: 3
    version: "2.0"
    creation_date: "2025-10-19"
    description: "IEEE 39-bus system with complete RMS dynamic parameters"

 topology/                                 # Graph structure
    edge_list/
        from_bus: int64[46]              # Source bus indices
        to_bus: int64[46]                # Destination bus indices
        edge_type: int64[46]             # 0=line, 1=transformer, 2=switch

 phases/                                   # Per-phase network data
    phase_a/
       nodes/                           # Bus data
          bus_ids: int64[39]
          bus_names: string[39]
          bus_types: int64[39]         # 0=PQ, 1=PV, 2=Slack
          base_voltages_kV: float64[39]
          voltages_pu: float64[39]
          angles_deg: float64[39]
          P_injection_MW: float64[39]
          Q_injection_MVAR: float64[39]
          P_generation_MW: float64[39]
          Q_generation_MVAR: float64[39]
          P_load_MW: float64[39]
          Q_load_MVAR: float64[39]
          shunt_G_pu: float64[39]
          shunt_B_pu: float64[39]
      
       edges/                           # Branch data
           from_bus: int64[46]
           to_bus: int64[46]
           element_id: string[46]
           element_type: int64[46]
           R_pu: float64[46]
           X_pu: float64[46]
           B_shunt_pu: float64[46]
           rating_MVA: float64[46]
           length_km: float64[46]
           in_service: bool[46]
   
    phase_b/                             # Same structure as phase_a
    phase_c/                             # Same structure as phase_a

 dynamic_models/                           # RMS dynamic parameters
    generators/                          # Synchronous machine models
       names: string[10]                # Generator names
       buses: string[10]                # Connected buses
       phases: string[10]               # Phase connections (abc)
       model_type: string[10]           # GENROU, GENSAL, GENCLS
      
       H_s: float64[10]                 # Inertia constant (seconds)
       D_pu: float64[10]                # Damping coefficient (pu)
      
       xd_pu: float64[10]               # d-axis synchronous reactance
       xq_pu: float64[10]               # q-axis synchronous reactance
       xd_prime_pu: float64[10]         # d-axis transient reactance
       xq_prime_pu: float64[10]         # q-axis transient reactance
       xd_double_prime_pu: float64[10]  # d-axis subtransient reactance
       xq_double_prime_pu: float64[10]  # q-axis subtransient reactance
       xl_pu: float64[10]               # Leakage reactance
       ra_pu: float64[10]               # Armature resistance
      
       Td0_prime_s: float64[10]         # d-axis transient time constant
       Tq0_prime_s: float64[10]         # q-axis transient time constant
       Td0_double_prime_s: float64[10]  # d-axis subtransient time constant
       Tq0_double_prime_s: float64[10]  # q-axis subtransient time constant
      
       S10: float64[10]                 # Saturation factor at 1.0 pu
       S12: float64[10]                 # Saturation factor at 1.2 pu
   
    exciters/                            # Automatic Voltage Regulator (AVR)
       names: string[10]                # Exciter names
       generator_names: string[10]      # Associated generators
       model_type: string[10]           # SEXS, IEEEAC1A, etc.
      
       Ka: float64[10]                  # Amplifier gain
       Ta_s: float64[10]                # Amplifier time constant
       Ke: float64[10]                  # Exciter gain
       Te_s: float64[10]                # Exciter time constant
       Kf: float64[10]                  # Stabilizer gain
       Tf_s: float64[10]                # Stabilizer time constant
       Efd_max: float64[10]             # Max field voltage (pu)
       Efd_min: float64[10]             # Min field voltage (pu)
       Vr_max: float64[10]              # Max regulator output
       Vr_min: float64[10]              # Min regulator output
   
    governors/                           # Turbine-Governor models
        names: string[10]                # Governor names
        generator_names: string[10]      # Associated generators
        model_type: string[10]           # TGOV1, HYGOV, etc.
       
        R_pu: float64[10]                # Droop (pu)
        Dt_pu: float64[10]               # Turbine damping (pu)
        Tg_s: float64[10]                # Governor time constant
        Tt_s: float64[10]                # Turbine time constant
        Pmax_pu: float64[10]             # Max turbine power (pu)
        Pmin_pu: float64[10]             # Min turbine power (pu)

 initial_conditions/                       # Dynamic state initialization
    generator_states/
        rotor_angles_rad: float64[10]    # Initial rotor angle (rad)
        rotor_speeds_pu: float64[10]     # Initial speed (pu, relative to sync)
        field_voltages_pu: float64[10]   # Initial field voltage (pu)
        mechanical_power_pu: float64[10] # Initial mechanical power (pu)

 steady_state/                             # Power flow solution
     power_flow_results/
         converged: bool                   # Convergence status
         iterations: int                   # Number of iterations
         max_mismatch: float64            # Max power mismatch (pu)
         total_generation_MW: float64      # Total generation
         total_load_MW: float64           # Total load
         total_losses_MW: float64         # Total losses
         max_voltage_pu: float64          # Maximum bus voltage
         min_voltage_pu: float64          # Minimum bus voltage
``

---

## H5 Format Specification

### Data Types

| Type | Description | Example Values |
|------|-------------|----------------|
| `float64` | 64-bit floating point | Voltages, powers, impedances |
| `int64` | 64-bit integer | Bus IDs, counts |
| `bool` | Boolean | Convergence flags, in-service status |
| `string` | Variable-length string | Names, descriptions |

### Bus Types

| Code | Type | Description |
|------|------|-------------|
| 0 | PQ | Load bus (P and Q specified) |
| 1 | PV | Generator bus (P and V specified) |
| 2 | Slack | Reference bus (V and θ specified) |
| 3 | Isolated | Disconnected bus |

### Edge Types

| Code | Type | Description |
|------|------|-------------|
| 0 | Line | Transmission line (π-model) |
| 1 | Transformer | Power transformer (two-winding) |
| 2 | Switch | Circuit breaker or disconnect |

### Generator Models

| Model | Description | Use Case |
|-------|-------------|----------|
| GENROU | Round rotor generator | Thermal units (steam, nuclear) |
| GENSAL | Salient pole generator | Hydro units |
| GENCLS | Classical model | Simplified stability studies |

### Exciter Models

| Model | Description | Parameters |
|-------|-------------|------------|
| SEXS | Simplified Excitation System | Ka, Ta, Efd limits |
| IEEEAC1A | IEEE Type AC1A | Ka, Ta, Ke, Te, Kf, Tf |

### Governor Models

| Model | Description | Application |
|-------|-------------|-------------|
| TGOV1 | Simple Steam Turbine | Thermal generation |
| HYGOV | Hydro Turbine | Hydroelectric |

---

## Usage Guide

### Reading the H5 File

``python
import h5py
import numpy as np

# Open the file
with h5py.File('graph_model/Graph_model.h5', 'r') as f:
    # Read metadata
    grid_name = f['metadata'].attrs['grid_name']
    base_mva = f['metadata'].attrs['base_mva']
    print(f"Grid: {grid_name}, Base: {base_mva} MVA")
    
    # Read topology
    from_bus = f['topology/edge_list/from_bus'][:]
    to_bus = f['topology/edge_list/to_bus'][:]
    edge_type = f['topology/edge_list/edge_type'][:]
    
    # Read phase A node data
    voltages = f['phases/phase_a/nodes/voltages_pu'][:]
    angles = f['phases/phase_a/nodes/angles_deg'][:]
    P_inj = f['phases/phase_a/nodes/P_injection_MW'][:]
    
    # Read generator dynamics
    H = f['dynamic_models/generators/H_s'][:]
    xd = f['dynamic_models/generators/xd_pu'][:]
    xd_prime = f['dynamic_models/generators/xd_prime_pu'][:]
    
    # Read initial conditions
    delta_0 = f['initial_conditions/generator_states/rotor_angles_rad'][:]
    omega_0 = f['initial_conditions/generator_states/rotor_speeds_pu'][:]
``

### Accessing Specific Data

``python
# Get all generator names
with h5py.File('graph_model/Graph_model.h5', 'r') as f:
    gen_names = [n.decode() for n in f['dynamic_models/generators/names'][:]]
    print("Generators:", gen_names)
    
    # Get generator at bus 30
    buses = [b.decode() for b in f['dynamic_models/generators/buses'][:]]
    idx = buses.index('Bus_30')
    gen_name = gen_names[idx]
    H_30 = f['dynamic_models/generators/H_s'][idx]
    print(f"{gen_name}: H = {H_30} s")
``

### Iterating Through All Phases

``python
with h5py.File('graph_model/Graph_model.h5', 'r') as f:
    for phase_name in ['phase_a', 'phase_b', 'phase_c']:
        voltages = f[f'phases/{phase_name}/nodes/voltages_pu'][:]
        print(f"{phase_name}: Avg voltage = {np.mean(voltages):.3f} pu")
``

---

## Dynamic Models

### Generator Model (GENROU)

The Round Rotor Generator model includes:

**Swing Equation:**
``
H * dω/dt = Pm - Pe - D(ω - ω)
dδ/dt = ω - ω
``

**Electrical Dynamics:**
``
Td0' * dEq'/dt = Efd - Eq' - (xd - xd')Id
Tq0' * dEd'/dt = -Ed' - (xq - xq')Iq
``

**Key Parameters:**
- **H** (s): Inertia constant - energy stored per MVA
  - Typical: 2-6 s for thermal, 2-4 s for hydro
- **D** (pu): Damping coefficient
  - Typical: 0-2 pu
- **xd, xq** (pu): Synchronous reactances
  - xd typical: 1.0-2.5 pu, xq typical: 0.6-2.4 pu
- **xd', xq'** (pu): Transient reactances
  - Typical: 0.15-0.40 pu
- **Td0', Tq0'** (s): Open-circuit time constants
  - Td0' typical: 4-10 s, Tq0' typical: 0.5-2 s

### Exciter Model (SEXS/IEEEAC1A)

Automatic Voltage Regulator controls terminal voltage:

**Control Law:**
``
Efd = Ka * (Vref - Vt + Vstab)
``

**Key Parameters:**
- **Ka**: Amplifier gain (50-400)
- **Ta** (s): Amplifier time constant (0.01-0.1 s)
- **Efd_max/min** (pu): Field voltage limits (3 to 7 pu)

### Governor Model (TGOV1/HYGOV)

Frequency control adjusts mechanical power:

**Control Law:**
``
Pm = Pref + (1/R) * (ω - ω) - Dt * (ω - ω)
``

**Key Parameters:**
- **R** (pu): Speed droop (0.03-0.08, typically 0.05)
- **Tg** (s): Governor time constant (0.1-0.5 s)
- **Tt** (s): Turbine time constant (0.2-1.0 s)
- **Pmax/Pmin** (pu): Power limits

---

## Creating Your Own H5 Files

### Using the Comprehensive Export Script

``bash
# Export from your existing scenario_0.h5
python graph_exporter_demo.py
``

This creates `graph_model/Graph_model.h5` with:
- All network data from scenario_0.h5
- Complete ANDES-compatible dynamic parameters
- Initial conditions for RMS simulation
- Power flow steady-state solution

### Using the H5 Writer Directly

``python
from graph_model import PowerGridH5Writer
import numpy as np

# Create a new H5 file
with PowerGridH5Writer('my_grid.h5', mode='w') as writer:
    
    # 1. Write metadata
    writer.write_metadata(
        grid_name='MySystem',
        base_mva=100.0,
        base_frequency_hz=60.0,
        num_buses=10,
        num_phases=3
    )
    
    # 2. Write topology
    edge_list = {
        'from_bus': np.array([0, 1, 2]),
        'to_bus': np.array([1, 2, 3]),
        'edge_type': np.array([0, 0, 1])
    }
    writer.write_topology(edge_list=edge_list)
    
    # 3. Write phase data (for each phase)
    node_data = {
        'bus_ids': np.arange(10),
        'bus_names': np.array([f'Bus_{i}'.encode() for i in range(10)], dtype='S20'),
        'voltages_pu': np.ones(10),
        'angles_deg': np.zeros(10),
        # ... more fields
    }
    edge_data = {
        'from_bus': np.array([0, 1, 2]),
        'to_bus': np.array([1, 2, 3]),
        'R_pu': np.array([0.01, 0.02, 0.005]),
        'X_pu': np.array([0.05, 0.08, 0.06]),
        # ... more fields
    }
    writer.write_phase_data('phase_a', node_data, edge_data)
    
    # 4. Write generator dynamics
    from graph_model.h5_writer import create_default_generator_parameters
    
    gen_params = create_default_generator_parameters(n_generators=3)
    writer.write_generator_dynamics(
        names=['Gen_1', 'Gen_2', 'Gen_3'],
        buses=['Bus_1', 'Bus_5', 'Bus_8'],
        phases=['abc', 'abc', 'abc'],
        model_type=['GENROU', 'GENROU', 'GENROU'],
        parameters=gen_params
    )
    
    # 5. Write exciters
    from graph_model.h5_writer import create_default_exciter_parameters
    
    exc_params = create_default_exciter_parameters(n_generators=3, model_type='SEXS')
    writer.write_exciter_models(
        names=['AVR_1', 'AVR_2', 'AVR_3'],
        generator_names=['Gen_1', 'Gen_2', 'Gen_3'],
        model_type=['SEXS', 'SEXS', 'SEXS'],
        parameters=exc_params
    )
    
    # 6. Write governors
    from graph_model.h5_writer import create_default_governor_parameters
    
    gov_params = create_default_governor_parameters(n_generators=3, model_type='TGOV1')
    writer.write_governor_models(
        names=['GOV_1', 'GOV_2', 'GOV_3'],
        generator_names=['Gen_1', 'Gen_2', 'Gen_3'],
        model_type=['TGOV1', 'TGOV1', 'TGOV1'],
        parameters=gov_params
    )
    
    # 7. Write initial conditions
    writer.write_initial_conditions(
        rotor_angles_rad=np.zeros(3),
        rotor_speeds_pu=np.ones(3),
        field_voltages_pu=np.ones(3) * 1.8,
        mechanical_power_pu=np.ones(3) * 0.8
    )
``

---

## Best Practices

### Parameter Values

1. **Always use per-unit system** with consistent base values
2. **Generator ratings** should match actual machine data sheets
3. **Time constants** from manufacturer specifications or standard ranges
4. **Saturation curves** from no-load tests

### Data Validation

``python
# Check for valid ranges
assert 2.0 <= H.all() <= 10.0, "H out of range"
assert 0.0 <= D.all() <= 5.0, "D out of range"
assert 0.1 <= xd_prime.all() <= 0.5, "xd' out of range"
``

### Performance Tips

- Use CSR sparse matrices for large grids (>100 buses)
- Store only active phases if balanced operation assumed
- Compress H5 files: `h5py.File(path, 'w', compression='gzip')`
- Index frequently accessed datasets for faster queries

---

## Future Enhancements - H5 File Updates

### Overview

The current `Graph_model.h5` file is **production-ready for RMS simulation** but can be enhanced with additional data for advanced analysis. Below is a prioritized list of features to add later.

---

### 🔴 **HIGH PRIORITY** - Immediate Value

#### 1. Populate Power Flow Results

**Location:** `/steady_state/power_flow_results/`  
**Current Status:** Group exists but is empty  
**Action Required:** Add actual power flow solution data

```python
# Add to power_flow_results group:
writer.write_power_flow_results(
    converged=True,                    # bool
    iterations=5,                      # int
    max_mismatch=1.5e-6,              # float64 (pu)
    total_generation_MW=6275.0,       # float64
    total_load_MW=6150.0,             # float64
    total_losses_MW=125.0,            # float64
    max_voltage_pu=1.05,              # float64
    min_voltage_pu=0.98               # float64
)
```

**Why Important:** Validates that steady-state solution is valid before dynamic simulation

---

#### 2. Add Metadata Counts

**Location:** `/metadata/` (as attributes)  
**Current Status:** Missing system counts  
**Action Required:** Add numerical summaries

```python
# Add as metadata attributes:
metadata.attrs['num_buses'] = 39
metadata.attrs['num_phases'] = 3
metadata.attrs['num_lines'] = 34
metadata.attrs['num_transformers'] = 12
metadata.attrs['num_generators'] = 10
metadata.attrs['num_loads'] = 19
```

**Why Important:** Quick system overview without reading all datasets

---

#### 3. Replace Default Parameters with PowerFactory Data

**Location:** `/dynamic_models/generators/`, `/dynamic_models/exciters/`, `/dynamic_models/governors/`  
**Current Status:** Default ANDES-compatible parameters (generic values)  
**Action Required:** Extract real parameters from PowerFactory models

**Generator Parameters to Update:**
```python
# From PowerFactory ComSym/ComDpl/ComDyn objects:
H_s: [3.5, 4.2, ...]           # Actual inertia constants (s)
xd_pu: [1.8, 2.1, ...]         # Real synchronous reactances
xd_prime_pu: [0.25, 0.30, ...]  # Real transient reactances
Td0_prime_s: [6.5, 8.0, ...]   # Real time constants
# ... all 18 parameters
```

**Exciter Parameters to Update:**
```python
# From PowerFactory Exc models:
Ka: [200, 150, ...]            # Real amplifier gains
Ta_s: [0.02, 0.015, ...]       # Real time constants
# ... all 10 parameters
```

**Governor Parameters to Update:**
```python
# From PowerFactory Gov models:
R_pu: [0.05, 0.04, ...]        # Real droop settings
Tg_s: [0.2, 0.3, ...]          # Real governor time constants
# ... all 6 parameters
```

**Why Important:** Accurate simulation results matching real system behavior

---

### 🟡 **MEDIUM PRIORITY** - Enhanced Analysis

#### 4. Add Sparse Admittance Matrices

**Location:** `/physics/admittance_matrix/`  
**Current Status:** Not present  
**Action Required:** Compute and store Y-matrices in CSR format

```python
# Add Y_single_phase (per-phase, 39×39):
writer.create_group('physics/admittance_matrix/Y_single_phase')
writer.write_sparse_matrix(
    group='physics/admittance_matrix/Y_single_phase',
    data_real=Y.data.real,      # float64, shape=(nnz,)
    data_imag=Y.data.imag,      # float64, shape=(nnz,)
    indices=Y.indices,           # int32, shape=(nnz,)
    indptr=Y.indptr,            # int32, shape=(40,)  # N+1
    shape=(39, 39)              # int64, shape=(2,)
)

# Add Y_three_phase (full system, 117×117):
writer.create_group('physics/admittance_matrix/Y_three_phase')
writer.write_sparse_matrix(
    group='physics/admittance_matrix/Y_three_phase',
    data_real=Y_full.data.real,
    data_imag=Y_full.data.imag,
    indices=Y_full.indices,
    indptr=Y_full.indptr,
    shape=(117, 117)            # 39 buses × 3 phases
)
```

**Why Important:** Faster power flow solving, impedance analysis, modal analysis

---

#### 5. Add Topology Adjacency Matrix

**Location:** `/topology/adjacency_matrix/`  
**Current Status:** Only edge_list format present  
**Action Required:** Add sparse CSR adjacency representation

```python
# Build adjacency matrix from edge_list:
A = build_adjacency_from_edges(from_bus, to_bus)

writer.create_group('topology/adjacency_matrix')
writer.write_sparse_matrix(
    group='topology/adjacency_matrix',
    data=A.data,                # float64, shape=(nnz,)
    indices=A.indices,          # int32, shape=(nnz,)
    indptr=A.indptr,            # int32, shape=(40,)
    shape=(39, 39)              # int64, shape=(2,)
)
```

**Why Important:** Graph algorithms, network analysis, GNN compatibility

---

#### 6. Add Contingency Scenarios

**Location:** `/scenarios/`  
**Current Status:** Not present  
**Action Required:** Store multiple operating points for N-1, N-2 analysis

```python
# Create scenario registry:
writer.create_group('scenarios/scenario_registry')
writer.write_dataset('scenarios/scenario_registry/scenario_ids', 
                     ['base_case', 'line_5_6_out', 'gen_30_out', ...])
writer.write_dataset('scenarios/scenario_registry/descriptions',
                     ['Normal operation', 'Line 5-6 outage', ...])

# For each scenario:
writer.create_group('scenarios/scenario_line_5_6_out')
writer.write_dataset('scenarios/scenario_line_5_6_out/voltages_pu',
                     voltages_3phase)  # shape=(117,) = 3*39
writer.write_dataset('scenarios/scenario_line_5_6_out/angles_deg',
                     angles_3phase)
writer.write_dataset('scenarios/scenario_line_5_6_out/P_injections_MW',
                     P_inj)
writer.write_dataset('scenarios/scenario_line_5_6_out/Q_injections_MVAR',
                     Q_inj)
writer.write_dataset('scenarios/scenario_line_5_6_out/contingency_description',
                     'Line 5-6 removed, power flow converged')
writer.write_dataset('scenarios/scenario_line_5_6_out/power_flow_converged',
                     True)
```

**Why Important:** Train neural networks, validate contingency analysis, scenario comparison

---

#### 7. Cache Analysis Results

**Location:** `/analysis_results/`  
**Current Status:** Not present  
**Action Required:** Store computed analysis to avoid re-running

**Small-Signal Stability Results:**
```python
writer.create_group('analysis_results/small_signal_stability')
writer.write_dataset('analysis_results/small_signal_stability/eigenvalues_real',
                     eig_real)  # float64, shape=(N_states,)
writer.write_dataset('analysis_results/small_signal_stability/eigenvalues_imag',
                     eig_imag)
writer.write_dataset('analysis_results/small_signal_stability/damping_ratios',
                     damping)
writer.write_dataset('analysis_results/small_signal_stability/oscillation_frequencies_Hz',
                     freqs)
writer.attrs['analysis_timestamp'] = datetime.now().isoformat()
```

**Impedance Scan Results:**
```python
writer.create_group('analysis_results/impedance_scan')
writer.write_dataset('analysis_results/impedance_scan/scan_points',
                     ['Bus_30', 'Bus_33', ...])
writer.write_dataset('analysis_results/impedance_scan/frequencies_Hz',
                     freqs)  # [0.1, 1, 10, 100, 1000]
writer.write_dataset('analysis_results/impedance_scan/Z_matrices',
                     Z_data)  # complex128, shape=(K, F, 2, 2)
```

**Contingency Analysis Results:**
```python
writer.create_group('analysis_results/contingency_analysis/n_minus_1')
writer.write_dataset('analysis_results/contingency_analysis/n_minus_1/element_ids',
                     ['Line_5_6', 'Line_10_11', ...])
writer.write_dataset('analysis_results/contingency_analysis/n_minus_1/converged',
                     [True, True, False, ...])
writer.write_dataset('analysis_results/contingency_analysis/n_minus_1/max_voltage_violations',
                     [0.02, 0.05, 0.15, ...])
writer.write_dataset('analysis_results/contingency_analysis/n_minus_1/overloaded_elements',
                     [0, 1, 3, ...])  # Count per scenario
```

**Why Important:** Performance optimization, result caching, reproducibility

---

### 🟢 **LOW PRIORITY** - Advanced Features

#### 8. Add Three-Phase Coupling Matrices

**Location:** `/coupling/`  
**Current Status:** Not present  
**Action Required:** Extract 3×3 impedance/admittance matrices for unbalanced analysis

```python
# Node coupling (e.g., transformer connections):
writer.create_group('coupling/node_coupling')
writer.write_dataset('coupling/node_coupling/bus_ids',
                     [5, 12, 18, ...])
writer.write_dataset('coupling/node_coupling/coupling_matrices',
                     Z_matrices)  # complex128, shape=(N, 3, 3)
writer.write_dataset('coupling/node_coupling/coupling_type',
                     [2, 2, 1, ...])  # 0=none, 1=line, 2=transformer
writer.write_dataset('coupling/node_coupling/element_ids',
                     ['Trafo_1', 'Trafo_2', ...])

# Edge coupling (mutual impedance between parallel lines):
writer.create_group('coupling/edge_coupling')
writer.write_dataset('coupling/edge_coupling/element_ids',
                     ['Line_5_6', 'Line_5_7'])
writer.write_dataset('coupling/edge_coupling/coupling_matrices',
                     Z_mutual)  # complex128, shape=(E, 3, 3)
writer.write_dataset('coupling/edge_coupling/mutual_impedance',
                     Z_mutual_scalar)  # complex128, shape=(E,)
```

**Why Important:** EMT simulation, unbalanced fault analysis, detailed three-phase modeling

---

#### 9. Add Neural Network Components (Future Research)

**Location:** `/learnable_objects/`, `/neural_network/`  
**Current Status:** Not applicable yet  
**Action Required:** Only needed for Stage 5+ (ML integration with PH-KAN)

**When to Add:** After implementing Port-Hamiltonian Kolmogorov-Arnold Networks for learning unknown IBR/load dynamics

**Structure Preview:**
```python
# Object registry:
writer.create_group('learnable_objects/object_registry')
writer.write_dataset('learnable_objects/object_registry/object_ids',
                     ['ibr_bus30_001', 'load_bus15_002'])

# Per-object PH-KAN parameters:
writer.create_group('learnable_objects/ibr_bus30_001')
writer.create_group('learnable_objects/ibr_bus30_001/ph_kan_parameters')
# ... spline coefficients, network weights, etc.
```

**Why Important:** Research on learning dynamics, when vendor models unavailable

---

### 📋 **Implementation Checklist**

Use this checklist when updating the H5 file:

```markdown
**Immediate Updates (Before RMS Simulation):**
- [ ] Populate /steady_state/power_flow_results/ with actual data
- [ ] Add metadata counts (num_buses, num_lines, etc.)
- [ ] Replace default dynamic parameters with PowerFactory values

**Enhanced Analysis (As Needed):**
- [ ] Add /physics/admittance_matrix/ (Y-matrix in CSR)
- [ ] Add /topology/adjacency_matrix/ (sparse adjacency)
- [ ] Create /scenarios/ for contingency cases
- [ ] Add /analysis_results/ caching groups

**Advanced Features (Future):**
- [ ] Add /coupling/ with 3×3 impedance matrices
- [ ] Implement /learnable_objects/ for ML models
- [ ] Add /neural_network/ embeddings
```

---

### 🛠️ **Helper Functions for Updates**

```python
# 1. Update power flow results after solving:
def update_power_flow_results(h5_path, pf_solution):
    with h5py.File(h5_path, 'a') as f:
        pf_group = f['steady_state/power_flow_results']
        pf_group.create_dataset('converged', data=pf_solution['converged'])
        pf_group.create_dataset('iterations', data=pf_solution['iterations'])
        pf_group.create_dataset('max_mismatch', data=pf_solution['max_mismatch'])
        pf_group.create_dataset('total_generation_MW', data=pf_solution['P_gen_total'])
        pf_group.create_dataset('total_load_MW', data=pf_solution['P_load_total'])
        pf_group.create_dataset('total_losses_MW', data=pf_solution['P_loss_total'])
        pf_group.create_dataset('max_voltage_pu', data=pf_solution['V_max'])
        pf_group.create_dataset('min_voltage_pu', data=pf_solution['V_min'])

# 2. Add scenario:
def add_contingency_scenario(h5_path, scenario_id, voltages, angles, description):
    with h5py.File(h5_path, 'a') as f:
        scenario_group = f.create_group(f'scenarios/{scenario_id}')
        scenario_group.create_dataset('voltages_pu', data=voltages)
        scenario_group.create_dataset('angles_deg', data=angles)
        scenario_group.attrs['description'] = description

# 3. Cache analysis results:
def cache_eigenvalue_analysis(h5_path, eigenvalues, modes):
    with h5py.File(h5_path, 'a') as f:
        analysis_group = f.create_group('analysis_results/small_signal_stability')
        analysis_group.create_dataset('eigenvalues_real', data=eigenvalues.real)
        analysis_group.create_dataset('eigenvalues_imag', data=eigenvalues.imag)
        # ... more results
        analysis_group.attrs['timestamp'] = datetime.now().isoformat()
```

---

### 📖 **References for Implementation**

- **H5 Writer API:** `h5_writer.py` - PowerGridH5Writer class
- **Format Specification:** `h5_format_specification.yaml`
- **Implementation Status:** `H5_IMPLEMENTATION_STATUS.md` (this directory)
- **Quick Checklist:** `H5_UPDATE_CHECKLIST.md` (this directory)
- **Full Specification:** `../Todo.md` Section 2 (HDF5 Storage Format)

---

## Next Steps: PowerFactory Integration

⚠️ **Current Status:** The H5 file contains **default parameters** for testing.

**To use real data:**

1. **Extract from PowerFactory:**
   ``python
   # Your PowerFactory extraction script
   gen_H = extract_generator_inertia_from_powerfactory()
   gen_xd = extract_generator_reactances_from_powerfactory()
   exciter_params = extract_exciter_parameters_from_powerfactory()
   ``

2. **Update H5 file** with real values using the writer
3. **Validate** against known operating points
4. **Test** with RMS simulation tools (ANDES, PowerWorld, etc.)

---

## Package Files

``
graph_model/
 __init__.py                         # Package initialization
 h5_format_specification.yaml        # Complete format spec (v2.0)
 h5_writer.py                        # PowerGridH5Writer class
 graph_exporter.py                   # GraphToH5Exporter class (WIP)
 
 Graph_model.h5                      # ✅ Main output file (production-ready)
 
 Graph_Explaination_README.md        # 📖 This file - main documentation
 H5_IMPLEMENTATION_STATUS.md         # 📊 Status vs Todo.md specification
 H5_UPDATE_CHECKLIST.md              # ✅ Quick reference for future updates
``

**Documentation Guide:**
- **Getting Started:** This file (Graph_Explaination_README.md)
- **Implementation Status:** H5_IMPLEMENTATION_STATUS.md
- **Future Updates:** H5_UPDATE_CHECKLIST.md

---

## References

1. **ANDES Documentation**: https://docs.andes.app/
2. **IEEE Standard Models**: 
   - Synchronous Machines: IEEE Std 1110-2002
   - Excitation Systems: IEEE Std 421.5-2016
   - Governors: IEEE Std 421.5-2016
3. **H5 Format Assessment**: `../explainations/H5_FORMAT_ASSESSMENT.md`
4. **Project Roadmap**: `../Todo.md`

---

**Last Updated:** October 19, 2025  
**Format Version:** 2.0  
**Maintained by:** PIGNN Project
