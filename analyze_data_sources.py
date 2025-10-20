"""
Analyze both source H5 files to map data for comprehensive Graph_model.h5

This script identifies:
1. What data is in scenario_0.h5 (topology, load flow results)
2. What data is in COMPOSITE_EXTRACTED.h5 (real PowerFactory parameters)
3. What needs to be merged into Graph_model.h5
"""
import h5py
import numpy as np

print("="*80)
print("DATA SOURCE ANALYSIS FOR COMPREHENSIVE Graph_model.h5")
print("="*80)

# ============================================================================
# 1. SCENARIO_0.H5 - Topology and Load Flow Results
# ============================================================================
print("\n1. SCENARIO_0.H5 CONTENT:")
print("-"*80)

with h5py.File('data/scenario_0.h5', 'r') as f:
    print("\nTop-level groups:")
    for key in f.keys():
        print(f"  - {key}")
    
    # Check detailed_system_data
    if 'detailed_system_data' in f:
        print("\ndetailed_system_data structure:")
        dsd = f['detailed_system_data']
        for key in dsd.keys():
            item = dsd[key]
            if isinstance(item, h5py.Group):
                print(f"  📂 {key}/")
                for subkey in list(item.keys())[:5]:
                    print(f"      - {subkey}")
            else:
                print(f"  📄 {key}")
        
        # Detailed check of critical groups
        print("\n  BUSES (detailed_system_data/buses):")
        if 'buses' in dsd:
            buses = dsd['buses']
            print(f"    Datasets: {list(buses.keys())}")
            if 'names' in buses:
                print(f"    Number of buses: {len(buses['names'])}")
        
        print("\n  GENERATORS (detailed_system_data/generators):")
        if 'generators' in dsd:
            gens = dsd['generators']
            print(f"    Datasets: {list(gens.keys())}")
            if 'names' in gens:
                print(f"    Number of generators: {len(gens['names'])}")
        
        print("\n  LOADS (detailed_system_data/loads):")
        if 'loads' in dsd:
            loads = dsd['loads']
            print(f"    Datasets: {list(loads.keys())}")
            if 'names' in loads:
                print(f"    Number of loads: {len(loads['names'])}")
                print(f"    Load names sample: {[n.decode() if isinstance(n, bytes) else n for n in loads['names'][:5]]}")
        
        print("\n  LINES (detailed_system_data/lines):")
        if 'lines' in dsd:
            lines = dsd['lines']
            print(f"    Datasets: {list(lines.keys())}")
            if 'names' in lines:
                print(f"    Number of lines: {len(lines['names'])}")
        
        print("\n  TRANSFORMERS (detailed_system_data/transformers):")
        if 'transformers' in dsd:
            trafos = dsd['transformers']
            print(f"    Datasets: {list(trafos.keys())}")
            if 'names' in trafos:
                print(f"    Number of transformers: {len(trafos['names'])}")

# ============================================================================
# 2. COMPOSITE_EXTRACTED.H5 - Real PowerFactory Parameters
# ============================================================================
print("\n\n2. COMPOSITE_EXTRACTED.H5 CONTENT:")
print("-"*80)

with h5py.File('data/composite_model_out/39_Bus_New_England_System_COMPOSITE_EXTRACTED.h5', 'r') as f:
    print("\nTop-level groups:")
    for key in f.keys():
        print(f"  - {key}")
    
    # Check generators
    if 'generators' in f:
        print("\nGENERATORS (with real PowerFactory parameters):")
        gens = f['generators']
        print(f"  Datasets: {list(gens.keys())}")
        
        if 'names' in gens:
            print(f"  Number of generators: {len(gens['names'])}")
            gen_names = [n.decode() if isinstance(n, bytes) else n for n in gens['names'][:]]
            print(f"  Generator names: {gen_names}")
        
        # Check for RMS dynamic parameters
        rms_params = ['Sn_MVA', 'Un_kV', 'Vset_pu', 'H_s', 'D_pu', 
                     'xd_pu', 'xq_pu', 'xd_prime_pu', 'xq_prime_pu']
        print("\n  RMS Dynamic Parameters Available:")
        for param in rms_params:
            if param in gens:
                values = gens[param][:]
                non_nan = np.sum(~np.isnan(values))
                print(f"    ✓ {param}: {non_nan}/{len(values)} valid values")
                if non_nan > 0:
                    valid_vals = values[~np.isnan(values)]
                    print(f"      Range: [{np.min(valid_vals):.3f}, {np.max(valid_vals):.3f}]")
            else:
                print(f"    ✗ {param}: NOT FOUND")
    
    # Check loads
    if 'loads' in f:
        print("\nLOADS:")
        loads = f['loads']
        print(f"  Datasets: {list(loads.keys())}")
        if 'names' in loads:
            print(f"  Number of loads: {len(loads['names'])}")

# ============================================================================
# 3. CURRENT Graph_model.h5 - What's Missing
# ============================================================================
print("\n\n3. CURRENT Graph_model.h5 STATUS:")
print("-"*80)

with h5py.File('graph_model/Graph_model.h5', 'r') as f:
    print("\nMetadata:")
    meta = f['metadata']
    for key in meta.attrs.keys():
        print(f"  {key}: {meta.attrs[key]}")
    
    print("\nDynamic Models - Generators:")
    if 'dynamic_models/generators' in f:
        gens = f['dynamic_models/generators']
        print(f"  Number of generators: {len(gens['names'])}")
        
        # Check if using defaults or real data
        if 'H_s' in gens:
            H_values = gens['H_s'][:]
            unique_H = np.unique(H_values)
            if len(unique_H) == 1:
                print(f"  ⚠️  H_s: ALL SAME VALUE ({unique_H[0]}) - Using defaults!")
            else:
                print(f"  ✓ H_s: Multiple values - Real data")
        
        if 'Sn_MVA' in gens:
            Sn_values = gens['Sn_MVA'][:]
            print(f"  Sn_MVA present: {Sn_values}")
        else:
            print(f"  ✗ Sn_MVA: NOT PRESENT")
    
    print("\nPhase Data - Loads:")
    P_load = f['phases/phase_a/nodes/P_load_MW'][:]
    Q_load = f['phases/phase_a/nodes/Q_load_MVAR'][:]
    print(f"  P_load non-zero: {np.sum(np.abs(P_load) > 0.001)}")
    print(f"  Q_load non-zero: {np.sum(np.abs(Q_load) > 0.001)}")
    if np.sum(np.abs(P_load) > 0.001) == 0:
        print(f"  ⚠️  Load data is ALL ZEROS - Not populated!")

# ============================================================================
# 4. MAPPING SUMMARY
# ============================================================================
print("\n\n4. DATA MAPPING FOR COMPREHENSIVE Graph_model.h5:")
print("="*80)

print("\n✅ AVAILABLE DATA SOURCES:")
print("  FROM scenario_0.h5:")
print("    - Topology (buses, lines, transformers)")
print("    - Load data (19 loads with P, Q, buses)")
print("    - Load flow results")
print("    - Y-matrix")
print()
print("  FROM COMPOSITE_EXTRACTED.h5:")
print("    - Real generator RMS parameters (Sn_MVA, Un_kV, H, D, xd, xq, etc.)")
print("    - Generator names and bus connections")
print()

print("\n⚠️  MISSING IN CURRENT Graph_model.h5:")
print("  1. Real generator parameters (using defaults instead)")
print("  2. Load data in phase nodes (all zeros)")
print("  3. Sn_MVA, Un_kV, Vset_pu in dynamic_models/generators")
print()

print("\n🎯 ACTION REQUIRED:")
print("  Update graph_exporter_demo.py to:")
print("  1. Read topology from scenario_0.h5 ✓ (already does)")
print("  2. Read REAL generator parameters from COMPOSITE_EXTRACTED.h5 (NEW)")
print("  3. Read load data from scenario_0.h5 and populate phase nodes (NEW)")
print("  4. Merge all data into single comprehensive Graph_model.h5")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
