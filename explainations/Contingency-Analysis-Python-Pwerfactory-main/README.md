# Modular Contingency Analysis System for Power Systems

A comprehensive, modular Python framework for performing contingency analysis on power systems using DIgSILENT PowerFactory. This system analyzes N-1 and N-2 contingencies, performs voltage sensitivity analysis, builds Y-matrices, and collects detailed post-contingency system data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PowerFactory](https://img.shields.io/badge/PowerFactory-2022%20SP3-green.svg)

## 🎯 Overview

This system provides a complete pipeline for power system contingency analysis with the following capabilities:

- **Automated Contingency Generation**: Creates comprehensive N-1 and N-2 scenarios
- **Load Flow Analysis**: Solves post-contingency power flows with convergence tracking
- **Voltage Sensitivity Analysis**: Calculates voltage sensitivities for all active elements
- **Y-Matrix Construction**: Builds complete admittance matrices from impedance data
- **Comprehensive Data Collection**: Gathers detailed system state information
- **HDF5 Data Storage**: Efficient, structured data storage for each scenario

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Modular Contingency Analysis                │
├─────────────────────────────────────────────────────────────────┤
│  Module 1: Contingency Generator                               │
│  ├─ Generates N-1 and N-2 scenarios                           │
│  ├─ Intelligent filtering (excludes dual generator outages)   │
│  └─ CSV output with scenario definitions                      │
├─────────────────────────────────────────────────────────────────┤
│  Module 2: Contingency Executor                               │
│  ├─ Executes each scenario in PowerFactory                    │
│  ├─ Solves load flows with convergence tracking               │
│  └─ Manages element disconnection/reconnection                │
├─────────────────────────────────────────────────────────────────┤
│  Module 3: Load Flow Data Collector                           │
│  ├─ Extracts comprehensive system impedances                  │
│  ├─ Collects bus voltages, angles, injections                 │
│  └─ Gathers line/transformer loadings and losses              │
├─────────────────────────────────────────────────────────────────┤
│  Module 4: Voltage Sensitivity Analysis                       │
│  ├─ Perturbs active power (±10 MW) and reactive power (±5 MVAR)│
│  ├─ Measures voltage changes on all system buses              │
│  └─ Calculates sensitivity matrices using finite differences  │
├─────────────────────────────────────────────────────────────────┤
│  Module 5: Y-Matrix Builder                                   │
│  ├─ Constructs bus admittance matrix from impedance data      │
│  ├─ Includes lines, transformers, generators, loads, shunts   │
│  └─ Calculates matrix properties and validation metrics       │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
contingency-analysis/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Contingency_Scenario_Generator.py      # Module 1: Generate scenarios
├── Contingency_Executor.py               # Module 2: Execute contingencies
├── Load_Flow_Data_Collector.py           # Module 3: Collect system data
├── Voltage_Sensitivity_Analysis.py       # Module 4: Sensitivity analysis
├── Y_Matrix_Builder.py                   # Module 5: Build Y-matrices
├── H5_File_inspector.py                  # Utility: Analyze results
├── contingency_out/                      # Generated scenarios (CSV)
├── contingency_scenarios/                # Analysis results (H5 files)
└── docs/                                 # Additional documentation
```

## 🚀 Quick Start

### Prerequisites

1. **DIgSILENT PowerFactory 2022 SP3** or later
2. **Python 3.8+** with the following packages:
   ```bash
   pip install numpy scipy h5py matplotlib seaborn pandas
   ```
3. **IEEE 39-Bus New England System** loaded in PowerFactory

### Basic Usage

1. **Generate Contingency Scenarios**:
   ```python
   python Contingency_Scenario_Generator.py
   ```
   - Creates N-1 and N-2 scenarios
   - Outputs CSV file with scenario definitions
   - Configurable scenario limits and percentages

2. **Execute Contingency Analysis**:
   ```python
   python Contingency_Executor.py
   ```
   - Runs all scenarios from CSV
   - Performs integrated analysis (load flow + sensitivity + Y-matrix)
   - Creates individual H5 files for each scenario

3. **Inspect Results**:
   ```python
   python H5_File_inspector.py
   ```
   - Analyzes random converged scenario
   - Creates visualizations and exports data
   - Validates result quality and realism

## 📊 Module Details

### Module 1: Contingency Scenario Generator

**File**: `Contingency_Scenario_Generator.py`

Generates comprehensive contingency scenarios for power system analysis.

**Features**:
- **N-1 Contingencies**: Single element outages (lines, transformers, generators)
- **N-2 Contingencies**: Double element outages with intelligent filtering
- **Configurable Limits**: Generate all scenarios or random subset
- **Base Case Inclusion**: Scenario 0 represents normal operation

**Configuration**:
```python
# Generate limited scenarios
GENERATE_LIMITED_SCENARIOS = True
MAX_SCENARIOS = 200
N1_PERCENTAGE = 0.30  # 30% N-1, 70% N-2

# Or generate all possible scenarios
GENERATE_LIMITED_SCENARIOS = False
```

**Output**: CSV file with columns:
- `scenario_id`: Unique identifier
- `contingency_type`: BASE, N-1, or N-2
- `description`: Human-readable scenario description
- `element1_name`, `element2_name`: Elements to disconnect
- `severity`: Estimated impact level

### Module 2: Contingency Executor

**File**: `Contingency_Executor.py`

Executes contingency scenarios with integrated analysis modules.

**Execution Sequence**:
1. Apply contingency (disconnect elements)
2. Solve load flow with convergence checking
3. Collect comprehensive power flow data
4. Collect detailed system data (impedances)
5. Perform voltage sensitivity analysis (if converged)
6. Build Y-matrix (regardless of convergence)
7. Save all data to H5 file
8. Restore elements to original state

**Features**:
- **Robust Error Handling**: Continues analysis even if load flow fails
- **Comprehensive Logging**: Detailed progress and status reporting
- **Modular Integration**: Calls all analysis modules in sequence
- **Data Validation**: Quality checks for each analysis step

### Module 3: Load Flow Data Collector

**File**: `Load_Flow_Data_Collector.py`

Collects comprehensive post-contingency system data including impedances.

**Data Collection**:
- **Bus Data**: Voltages, angles, injections, base voltages
- **Line Data**: Power flows, loadings, losses, impedances (R, X, B)
- **Transformer Data**: Power flows, loadings, tap positions, impedances
- **Generator Data**: Outputs, limits, impedances (xd, xq, reactances)
- **Load Data**: Power consumption, equivalent impedances
- **Shunt Data**: Reactive power, impedances

**Impedance Extraction**:
- **Lines**: Direct R/X values or calculated from per-km data
- **Transformers**: From type objects (uk%, ukr%) with fallback methods
- **Generators**: From type objects (xd, xq reactances)
- **Loads**: Calculated equivalent impedances from P, Q, V
- **Shunts**: Calculated from reactive power and voltage

### Module 4: Voltage Sensitivity Analysis

**File**: `Voltage_Sensitivity_Analysis.py`

Performs systematic voltage sensitivity analysis using finite difference methods.

**Methodology**:
- **Perturbation Steps**: ±10 MW for active power, ±5 MVAR for reactive power
- **Central Difference**: `dV/dP = (V+ - V-) / (2×ΔP)`
- **System-Wide Analysis**: Measures voltage changes on all buses
- **Threshold Filtering**: Stores sensitivities above 1e-6 pu/MW or pu/MVAR

**Analysis Scope**:
- **Generators**: P and Q sensitivity analysis for all active generators
- **Loads**: P and Q sensitivity analysis for all active loads
- **Coverage**: All connected buses in the post-contingency system

**Output Data**:
```
voltage_sensitivity/
├── analysis_metadata (timestamp, parameters)
├── generators/
│   └── [generator_name]/
│       ├── base_P_MW, base_Q_MVAR
│       ├── P_sensitivity/ (bus_names, sensitivities)
│       └── Q_sensitivity/ (bus_names, sensitivities)
└── loads/
    └── [load_name]/
        ├── base_P_MW, base_Q_MVAR
        ├── P_sensitivity/ (bus_names, sensitivities)
        └── Q_sensitivity/ (bus_names, sensitivities)
```

### Module 5: Y-Matrix Builder

**File**: `Y_Matrix_Builder.py`

Constructs complete bus admittance matrices from collected impedance data.

**Y-Matrix Construction**:
- **Lines**: Series admittance + charging susceptance
- **Transformers**: Series admittance with tap ratio and phase shift
- **Generators**: Diagonal admittance based on synchronous reactance
- **Loads**: Diagonal admittance from equivalent impedance
- **Shunts**: Diagonal susceptance (capacitive/inductive)

**Matrix Properties Calculated**:
- **Size and Density**: Matrix dimensions and sparsity
- **Eigenvalue Analysis**: Condition number and eigenvalue range
- **Symmetry Properties**: Hermitian property checking
- **Numerical Diagnostics**: Zero diagonal elements, conditioning

**Features**:
- **Sparse Storage**: Efficient CSR format for large matrices
- **Validation**: Comprehensive matrix property analysis
- **Error Handling**: Robust impedance processing with fallbacks
- **Documentation**: Detailed construction statistics

## 📈 Data Structure

Each contingency scenario creates an individual H5 file with the following structure:

```
scenario_X.h5
├── scenario_metadata/              # Basic scenario information
├── disconnection_actions/          # Elements taken out of service
├── load_flow_results/              # Convergence status and bus data
├── power_flow_data/                # Comprehensive MW/MVAR data
├── detailed_system_data/           # Impedances and system structure
├── voltage_sensitivity/            # Sensitivity analysis results
├── y_matrix/                       # Admittance matrix and properties
└── analysis_modules/               # Module completion status
```

### Key Data Groups

**Load Flow Results**:
- Bus voltages, angles, and injections
- Convergence status and iteration count
- Voltage statistics and violation counts

**Power Flow Data**:
- Generator outputs and limits
- Load consumption and power factors
- Line and transformer loadings
- System totals and power balance

**Detailed System Data**:
- Complete impedance matrices for all elements
- Bus connectivity and topology
- Element ratings and parameters

**Voltage Sensitivity**:
- P and Q sensitivities for all generators and loads
- Bus-by-bus sensitivity matrices
- Analysis parameters and timing

**Y-Matrix**:
- Complete bus admittance matrix (sparse format)
- Matrix properties and diagnostics
- Construction statistics and validation

## 🛠️ Configuration Options

### Scenario Generation
```python
# Contingency_Scenario_Generator.py
GENERATE_LIMITED_SCENARIOS = True   # True for subset, False for all
MAX_SCENARIOS = 200                 # Total scenarios including base case
N1_PERCENTAGE = 0.30               # Percentage of N-1 scenarios
```

### Sensitivity Analysis
```python
# Voltage_Sensitivity_Analysis.py
PERTURBATION_STEP_P_MW = 10.0      # Active power perturbation
PERTURBATION_STEP_Q_MVAR = 5.0     # Reactive power perturbation
MIN_SENSITIVITY_THRESHOLD = 1e-6    # Minimum sensitivity to store
```

### Y-Matrix Building
```python
# Y_Matrix_Builder.py
SMALL_IMPEDANCE_THRESHOLD = 1e-8   # Minimum impedance threshold
DEFAULT_GENERATOR_R_PU = 0.01      # Default generator resistance
SBASE_MVA = 100.0                  # Base power for calculations
```

### Module Control
```python
# Contingency_Executor.py
ENABLE_VOLTAGE_SENSITIVITY = True   # Enable/disable sensitivity analysis
ENABLE_Y_MATRIX_BUILDING = True    # Enable/disable Y-matrix construction
```

## 📊 Analysis Capabilities

### Voltage Sensitivity Applications
- **Voltage Stability Analysis**: Identify critical buses and controls
- **Optimal Power Flow**: Sensitivity-based optimization
- **Control Design**: Voltage regulator tuning and coordination
- **Security Assessment**: Voltage margin calculations

### Y-Matrix Applications
- **Network Analysis**: Complete system admittance representation
- **Short Circuit Studies**: Fault current calculations
- **Harmonic Analysis**: Frequency domain studies
- **State Estimation**: Measurement processing and validation

### Contingency Analysis Applications
- **Security Assessment**: N-1 and N-2 reliability evaluation
- **Operational Planning**: Contingency ranking and prioritization
- **System Strengthening**: Identify critical infrastructure
- **Emergency Response**: Pre-computed contingency solutions

## 🔍 Quality Validation

The system includes comprehensive validation features:

### Load Flow Validation
- Convergence checking with iteration limits
- Voltage range validation (0.95-1.05 pu)
- Island detection and connectivity analysis
- Power balance verification

### Sensitivity Validation
- Realistic sensitivity ranges (1e-6 to 1e-2 pu/MW)
- Generator vs load sensitivity ratios
- Threshold compliance checking
- Coverage analysis (buses analyzed vs total)

### Y-Matrix Validation
- Matrix property verification (Hermitian, conditioning)
- Density analysis (typical range 1-30% for power systems)
- Diagonal element validation (positive real parts)
- Eigenvalue analysis for numerical stability

## 🚨 Error Handling

The system includes robust error handling:

- **Contingency Execution**: Continues analysis even if load flow fails
- **Module Independence**: Each module can run independently
- **Data Validation**: Quality checks prevent invalid data storage
- **Graceful Degradation**: Partial results saved when possible
- **Comprehensive Logging**: Detailed error messages and debugging info

## 📈 Performance Characteristics

### Typical Execution Times (IEEE 39-Bus System)
- **Scenario Generation**: ~30 seconds for 200 scenarios
- **Single Contingency**: ~15-45 seconds depending on modules enabled
- **Voltage Sensitivity**: ~10-20 seconds per scenario
- **Y-Matrix Building**: ~2-5 seconds per scenario
- **Complete Analysis**: ~2-4 hours for 200 scenarios

### Memory Usage
- **H5 File Size**: ~500KB - 2MB per scenario
- **Peak Memory**: ~200-500MB during Y-matrix construction
- **Storage Scaling**: Linear with number of scenarios

## 🔧 Troubleshooting

### Common Issues

1. **PowerFactory Connection**:
   ```python
   # Ensure PowerFactory is running and project is loaded
   app = pf.GetApplication() or sys.exit("PF not running")
   ```

2. **Load Flow Convergence**:
   - Check contingency severity
   - Verify system parameters
   - Review convergence tolerances

3. **Missing Impedance Data**:
   - Verify type objects are properly defined
   - Check parameter naming conventions
   - Review fallback methods in data collector

4. **H5 File Access**:
   - Ensure proper file permissions
   - Check available disk space
   - Verify H5PY installation

### Debug Mode
Enable detailed logging by modifying the print statements or adding:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

1. **IEEE 39-Bus System**: T. Athay et al., "A Practical Method for the Direct Analysis of Transient Stability," IEEE Trans. on Power Apparatus and Systems, 1979.

2. **Power System Analysis**: Hadi Saadat, "Power System Analysis," McGraw-Hill Education, 2010.

3. **Contingency Analysis**: A.J. Wood, B.F. Wollenberg, "Power Generation, Operation, and Control," Wiley, 2012.

4. **Y-Matrix Methods**: J.J. Grainger, W.D. Stevenson, "Power System Analysis," McGraw-Hill, 1994.

## 📄 License

This project is licensed under 

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis module'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the maintainers.

---

**Note**: This system is designed for educational and research purposes. For commercial power system analysis, please ensure compliance with relevant industry standards and regulations.
