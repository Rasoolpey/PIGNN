# PIGNN Data Workflow - Simple Guide

## Overview
This document explains the complete workflow from PowerFactory to Python analysis.

## The Simple 2-Step Process

### Step 1: Extract Data from PowerFactory
**Script**: `data/data_extraction.py`  
**Input**: PowerFactory project (must be running)  
**Output**: `composite_model_out/39_Bus_New_England_System_COMPOSITE_EXTRACTED.h5`

```bash
python data/data_extraction.py
```

**What it does:**
- Connects to PowerFactory
- Runs power flow
- Extracts:
  - Buses, generators, loads
  - Transmission lines & transformers (with impedances)
  - Dynamic models (AVR, Governor)
  - Initial conditions
- Saves everything to COMPOSITE_EXTRACTED.h5

### Step 2: Convert to Graph Format
**Script**: `graph_exporter_demo.py`  
**Input**: `composite_model_out/39_Bus_New_England_System_COMPOSITE_EXTRACTED.h5`  
**Output**: `graph_model/Graph_model.h5`

```bash
python graph_exporter_demo.py
```

**What it does:**
- Reads COMPOSITE_EXTRACTED.h5
- Converts to three-phase graph format
- Organizes data by phase (a, b, c)
- Creates comprehensive Graph_model.h5

## Using the Data

Once you have `graph_model/Graph_model.h5`, you can run:

### Load Flow Analysis
```bash
python load_flow_demo.py
```

### RMS Dynamic Simulation
```bash
python rms_demo.py
```

### Contingency Analysis
```bash
python contingency_demo.py
```

### Visualization
```bash
python visualization_demo.py
```

## Troubleshooting

### Problem: Zero Impedances in Graph_model.h5

**Symptom**: Load flow doesn't converge, Y-matrix is all zeros

**Cause**: PowerFactory data extraction returned default values (0.001, 0.01)

**Solution**: The impedance values should be extracted correctly from PowerFactory. If they're still zero:

1. Check PowerFactory model has line/transformer impedances defined
2. Verify the load flow converged in PowerFactory before extraction
3. The default values in `data_extraction.py` (line 582) can be adjusted if needed

### File Locations

```
Project Root/
├── data/
│   └── data_extraction.py          # Step 1: Extract from PowerFactory
├── composite_model_out/
│   └── COMPOSITE_EXTRACTED.h5      # PowerFactory raw data
├── graph_exporter_demo.py          # Step 2: Convert to graph format
└── graph_model/
    └── Graph_model.h5              # Final data for analysis
```

## Notes

- **COMPOSITE_EXTRACTED.h5** is the raw PowerFactory export
- **Graph_model.h5** is the processed, three-phase graph format used by all analysis scripts
- You only need to run Steps 1-2 once, unless the PowerFactory model changes
- All demo scripts (load_flow, rms, contingency, etc.) use `Graph_model.h5`

---
Last Updated: October 20, 2025
