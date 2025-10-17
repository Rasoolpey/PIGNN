# Three-Phase Power Grid Graph Structure

## ðŸ“‹ Overview

This is a modular, physics-informed graph structure for representing unbalanced three-phase power systems. The implementation follows the concepts from your research papers on Physics-Informed Graph Learning for Power Systems.

### Key Features

âœ… **Three-Phase Representation**: Each component exists as three coupled phase planes  
âœ… **Physics-Based Coupling**: Proper modeling of mutual impedances between phases  
âœ… **Component-Specific Models**: Different coupling for lines vs transformers  
âœ… **H5 Data Integration**: Direct loading from PowerFactory simulation data  
âœ… **Modular Architecture**: Clean separation of concerns for easy debugging  
âœ… **Validation Framework**: Built-in checks for structure and physics constraints  

---

## ðŸ—ï¸ Project Structure

```
power_grid_graph/
â”‚
â”œâ”€â”€ core/                      # Core graph data structures
â”‚   â”œâ”€â”€ graph_base.py         # Base graph, nodes, edges, coupling matrices
â”‚   â”œâ”€â”€ node_types.py         # Specialized nodes (Generator, Load, Bus)
â”‚   â””â”€â”€ edge_types.py         # Specialized edges (Line, Transformer)
â”‚
â”œâ”€â”€ physics/                   # Physics calculations
â”‚   â”œâ”€â”€ coupling_models.py    # Phase coupling calculations (Carson's equations)
â”‚   â”œâ”€â”€ impedance_matrix.py   # Y-matrix construction and manipulation
â”‚   â””â”€â”€ symmetrical_components.py  # 012 transformations (not yet implemented)
â”‚
â”œâ”€â”€ data/                      # Data loading and graph building
â”‚   â”œâ”€â”€ h5_loader.py          # Load IEEE39 data from H5 files
â”‚   â””â”€â”€ graph_builder.py      # Build graph from loaded data
â”‚
â”œâ”€â”€ utils/                     # Utilities
â”‚   â””â”€â”€ validators.py         # Validation and sanity checks
â”‚
â”œâ”€â”€ visualization/             # Visualization (future)
â”‚   â””â”€â”€ graph_plotter.py      # (To be implemented)
â”‚
â””â”€â”€ __init__.py               # Package initialization
```

---

## ðŸŽ¯ Key Concepts

### 1. Three-Phase Graph Structure

Each physical component (bus, line, transformer) is represented as **three interconnected phase graphs**:

```
Phase A Graph: Nodes_a --- Edges_a --- Nodes_a
                 |            |            |
              Coupling     Coupling     Coupling
                 |            |            |
Phase B Graph: Nodes_b --- Edges_b --- Nodes_b
                 |            |            |
              Coupling     Coupling     Coupling
                 |            |            |
Phase C Graph: Nodes_c --- Edges_c --- Nodes_c
```

### 2. Coupling Matrices

**Node Coupling** (3Ã—3 matrix):
- Represents how voltage at one phase affects current injection at other phases
- Important for generators (internal stator coupling)
- Small for most buses

**Edge Coupling** (3Ã—3 matrix):
- Represents mutual impedance between phase conductors
- Lines: 20-30% mutual coupling (overhead), 5-15% (cable)
- Transformers: Depends on winding configuration (Y-Y, Y-Î”, etc.)

Example impedance matrix for a transmission line:
```
Z_line = [ Z_aa  Z_ab  Z_ac ]
         [ Z_ba  Z_bb  Z_bc ]
         [ Z_ca  Z_cb  Z_cc ]
```

Where:
- Diagonal (Z_aa, Z_bb, Z_cc): Self-impedances
- Off-diagonal (Z_ab, etc.): Mutual impedances

### 3. Component Types

**Nodes:**
- `Bus`: Connection point (zero injection)
- `Generator`: Power injection with internal impedance
- `Load`: Power consumption (constant P/Q or constant Z)

**Edges:**
- `Line`: Transmission line with series Z and shunt B
- `Transformer`: With tap ratio, phase shift, and winding configuration

---

## ðŸš€ Quick Start

### Installation

```bash
# Create directory structure
mkdir -p power_grid_graph/{core,physics,data,utils,visualization}
cd power_grid_graph

# Create __init__.py files
touch __init__.py
touch core/__init__.py
touch physics/__init__.py
touch data/__init__.py
touch utils/__init__.py
touch visualization/__init__.py
```

### Basic Usage

```python
from power_grid_graph.data.h5_loader import H5DataLoader
from power_grid_graph.data.graph_builder import GraphBuilder
from power_grid_graph.physics.impedance_matrix import AdmittanceMatrixBuilder

# 1. Load data from H5 file
loader = H5DataLoader("./data/scenario_0.h5")
data = loader.load_all_data()

# 2. Build graph
builder = GraphBuilder(base_mva=100.0, frequency_hz=60.0)
graph = builder.build_from_h5_data(data)

# 3. Build Y-matrix
y_builder = AdmittanceMatrixBuilder(graph)
Y_matrix = y_builder.build_y_matrix(use_sparse=True)

print(f"Graph has {len(graph.nodes)} three-phase components")
print(f"Y-matrix shape: {Y_matrix.shape}")
```

### Accessing Graph Elements

```python
# Get a bus node (all three phases)
bus_phases = graph.get_node("Bus_1")  # Returns dict with PhaseType keys

# Get specific phase
bus_a = graph.get_node("Bus_1", PhaseType.A)
print(f"Voltage: {abs(bus_a.voltage_pu):.4f} pu")
print(f"Angle: {np.rad2deg(bus_a.voltage_angle_rad):.2f}Â°")

# Get neighbors in a specific phase
neighbors = graph.get_neighbors("Bus_1", PhaseType.A)

# Get coupling matrix for an edge
coupling = graph.edge_couplings["Line_Bus1_Bus2_0"]
Z_matrix = coupling.matrix
print(f"Self impedance: {Z_matrix[0,0]}")
print(f"Mutual impedance: {Z_matrix[0,1]}")
```

---

## ðŸ“Š Data Flow

```
PowerFactory Simulation
         â†“
    scenario_0.h5
         â†“
   H5DataLoader
         â†“
    Dict of data
         â†“
   GraphBuilder
         â†“
  PowerGridGraph (3-phase)
         â†“
AdmittanceMatrixBuilder
         â†“
    Y-matrix (3N Ã— 3N)
         â†“
  Load Flow / Contingency Solver
```

---

## ðŸ”§ Implementation Details

### Coupling Calculation Methods

**For Lines:**
```python
# Uses Carson's equations
M_mutual = (Î¼â‚€ / 2Ï€) * ln(D_ab / GMR)
X_mutual = Ï‰ * M_mutual
Z_mutual = R_ground + j*X_mutual
```

**For Transformers:**
```python
# Based on winding configuration
Z_mutual = Z_leakage * coupling_factor (typically 0.05)

# Connection matrix for Y-Î”:
T = [[ 1, -1,  0],
     [ 0,  1, -1],
     [-1,  0,  1]]
```

### Y-Matrix Construction

The admittance matrix is built as:

1. Initialize (3N Ã— 3N) matrix, where N = number of buses
2. For each edge (iâ†’j):
   - Compute 3Ã—3 admittance block: Y_edge = inv(Z_edge_3x3)
   - Add to diagonal blocks: Y[3i:3i+3, 3i:3i+3] += Y_edge
   - Add to off-diagonal: Y[3i:3i+3, 3j:3j+3] -= Y_edge
3. Add node shunt admittances to diagonal

### State Vector Organization

```
X = [V_a1, V_b1, V_c1,  V_a2, V_b2, V_c2,  ...,  V_aN, V_bN, V_cN]
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜           â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
       Bus 1                 Bus 2                    Bus N
```

---

## âœ… Validation

The framework includes validators to check:

### Structural Validation
- All nodes exist
- All edges reference valid nodes
- Coupling matrices are 3Ã—3
- No duplicate IDs

### Physics Validation
- Voltage magnitudes in reasonable range (0.5 - 1.5 pu)
- Non-negative resistances
- Impedances not too small (numerical stability)
- Coupling matrices are symmetric (for passive elements)

### Power Balance
- Total generation vs total load
- Mismatch calculation

```python
from power_grid_graph.utils.validators import GraphValidator

validator = GraphValidator()
is_valid, errors = validator.validate_graph_structure(graph)
is_physical, warnings = validator.validate_physics_constraints(graph)
balance = validator.check_power_balance(graph)
```

---

## ðŸŽ“ Theoretical Background

This implementation follows the mathematical framework from:

### Graph Representation

- **Nodes (V)**: Buses with state vectors x_i = [V_a, V_b, V_c, Î¸_a, Î¸_b, Î¸_c]
- **Edges (E)**: Lines/transformers with impedance matrices Z_ij^(3Ã—3)
- **Coupling**: Inter-phase interactions captured in off-diagonal terms

### Physics Equations

**Node dynamics (general form):**
```
dx_i/dt = f_i(x_i, {x_j : j âˆˆ N(i)}, u_i)
```

**Power flow (algebraic):**
```
P_i = V_i Î£_{jâˆˆN(i)} V_j (G_ij cos Î¸_ij + B_ij sin Î¸_ij)
Q_i = V_i Î£_{jâˆˆN(i)} V_j (G_ij sin Î¸_ij - B_ij cos Î¸_ij)
```

**With 3-phase coupling:**
```
[I_a]   [Y_aa  Y_ab  Y_ac] [V_a]
[I_b] = [Y_ba  Y_bb  Y_bc] [V_b]
[I_c]   [Y_ca  Y_cb  Y_cc] [V_c]
```

---

## ðŸ”® Next Steps

The current implementation provides the graph structure. Next modules to develop:

1. **Load Flow Solver** (Newton-Raphson for three-phase)
2. **Contingency Analysis** (N-1, N-2 scenarios)
3. **Dynamic Simulation** (Time-domain integration)
4. **Learnable Terms** (Physics-informed neural corrections)

Each will be a separate module using this graph as foundation.

---

## ðŸ“ Notes

### Current Limitations

- Assumes three-wire system (no neutral explicitly modeled)
- Coupling matrices use simplified estimates when geometry unavailable
- No frequency-dependent line models yet
- Transformer phase shift implemented but not fully tested

### Future Enhancements

- Add neutral conductor modeling (4-wire systems)
- Implement frequency-dependent line models
- Add detailed generator dynamics (subtransient model)
- Support for distributed energy resources (DERs)
- Visualization tools for graph structure

---

## ðŸ¤ Integration with Your Workflow

This graph structure is designed to integrate with:

1. Your existing PowerFactory simulation pipeline
2. The scenario H5 files you've already generated
3. Future load flow and contingency solvers
4. The physics-informed learning framework

The modular design ensures you can:
- Debug each component independently
- Add new component types easily
- Extend with learnable terms later
- Maintain code readability

---

## ðŸ“š References

- Your research papers on Physics-Informed Graph Learning
- Carson's equations for line modeling
- Kundur, "Power System Stability and Control"
- IEEE Std 1110 - Synchronous Generator Modeling

---

*This is a living document. Update as the code evolves.*