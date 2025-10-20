"""Count loads in scenario_0.h5"""
import h5py
import numpy as np

print("Checking load counts in scenario_0.h5...")

with h5py.File('data/scenario_0.h5', 'r') as f:
    # Check detailed_system_data/loads
    if 'detailed_system_data/loads' in f:
        loads = f['detailed_system_data/loads']
        print("\ndetailed_system_data/loads datasets:")
        for key in loads.keys():
            ds = loads[key]
            if hasattr(ds, 'shape'):
                print(f"  - {key}: shape={ds.shape}")
        
        if 'bus_ids' in loads:
            num_loads = len(loads['bus_ids'])
            print(f"\n✅ Number of loads in scenario_0.h5: {num_loads}")
            
            if 'P_MW' in loads:
                P_loads = loads['P_MW'][:]
                print(f"   Total P_load: {np.sum(P_loads):.2f} MW")
            if 'Q_MVAR' in loads:
                Q_loads = loads['Q_MVAR'][:]
                print(f"   Total Q_load: {np.sum(Q_loads):.2f} MVAR")
