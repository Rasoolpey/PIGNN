"""Check load data in scenario_0.h5"""
import h5py
import numpy as np

print("="*80)
print("CHECKING LOAD DATA IN SOURCE FILE")
print("="*80)

with h5py.File('data/scenario_0.h5', 'r') as f:
    print("\nTop-level groups in scenario_0.h5:")
    for key in f.keys():
        print(f"  - {key}")
    
    # Check for loads
    if 'loads' in f:
        print("\n✓ Found 'loads' group!")
        print("  Load datasets:")
        for key in f['loads'].keys():
            dataset = f['loads'][key]
            print(f"    - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if len(dataset) < 20:
                print(f"      Values: {dataset[:]}")
    
    # Check buses
    if 'buses' in f:
        print("\n✓ Found 'buses' group!")
        print("  Bus datasets:")
        for key in f['buses'].keys():
            dataset = f['buses'][key]
            if 'load' in key.lower() or 'p_' in key.lower() or 'q_' in key.lower():
                print(f"    - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if hasattr(dataset, '__len__') and len(dataset) < 50:
                    non_zero = np.sum(np.abs(dataset[:]) > 0.001)
                    print(f"      Non-zero values: {non_zero}/{len(dataset)}")

print("\n" + "="*80)
print("CHECKING GRAPH_MODEL.H5")
print("="*80)

with h5py.File('graph_model/Graph_model.h5', 'r') as f:
    print("\nPhase A node data:")
    P_load = f['phases/phase_a/nodes/P_load_MW'][:]
    Q_load = f['phases/phase_a/nodes/Q_load_MVAR'][:]
    P_inj = f['phases/phase_a/nodes/P_injection_MW'][:]
    Q_inj = f['phases/phase_a/nodes/Q_injection_MVAR'][:]
    
    print(f"  P_load_MW non-zero: {np.sum(np.abs(P_load) > 0.001)}")
    print(f"  Q_load_MVAR non-zero: {np.sum(np.abs(Q_load) > 0.001)}")
    print(f"  P_injection_MW non-zero: {np.sum(np.abs(P_inj) > 0.001)}")
    print(f"  Q_injection_MVAR non-zero: {np.sum(np.abs(Q_inj) > 0.001)}")
    
    if np.sum(np.abs(P_inj) > 0.001) > 0:
        print(f"\n  P_injection (negative = load):")
        neg_buses = np.where(P_inj < -0.001)[0]
        print(f"    Buses with negative injection (loads): {len(neg_buses)}")
        if len(neg_buses) > 0:
            print(f"    Bus indices: {neg_buses[:10]}...")
