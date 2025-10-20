"""
Quick script to update Graph_model.h5 with additional metadata counts
"""
import h5py
import numpy as np

h5_path = 'graph_model/Graph_model.h5'

print(f"Updating {h5_path} with additional metadata...")

with h5py.File(h5_path, 'a') as f:
    metadata = f['metadata']
    
    # Count elements from the file structure
    if 'topology/edge_list/edge_type' in f:
        edge_types = f['topology/edge_list/edge_type'][:]
        num_lines = int(np.sum(edge_types == 0))
        num_transformers = int(np.sum(edge_types == 1))
        
        print(f"  Counted from topology:")
        print(f"    - Lines: {num_lines}")
        print(f"    - Transformers: {num_transformers}")
        
        metadata.attrs['num_lines'] = num_lines
        metadata.attrs['num_transformers'] = num_transformers
    
    if 'dynamic_models/generators/names' in f:
        num_generators = len(f['dynamic_models/generators/names'])
        print(f"    - Generators: {num_generators}")
        metadata.attrs['num_generators'] = num_generators
    
    # For IEEE 39-bus system, we know there are 19 loads from the source data
    # (The graph exporter didn't populate load data yet, but we know the count)
    num_loads = 19  # Standard IEEE 39-bus system
    print(f"    - Loads: {num_loads} (from IEEE 39-bus standard)")
    metadata.attrs['num_loads'] = num_loads
    
    print("\n  Current metadata attributes:")
    for key in metadata.attrs.keys():
        print(f"    {key}: {metadata.attrs[key]}")

print(f"\n✅ Metadata updated successfully!")
print("\n⚠️  Note: Load data (P_load_MW, Q_load_MVAR) needs to be populated")
print("   in the phase node datasets. Currently all zeros.")

