# Fix_NaN_Shunts.py - Fix the NaN shunt elements causing diagonal issues
"""
Fix the specific issue: NaN shunt elements at buses 11 and 19.
"""

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import os

def fix_nan_shunts_and_rebuild_y(h5_file_path):
    """Fix NaN shunt elements and rebuild Y matrix"""
    
    print(f"🔧 FIXING NaN SHUNT ELEMENTS")
    print(f"📁 File: {os.path.basename(h5_file_path)}")
    print("=" * 60)
    
    # Load all data
    with h5py.File(h5_file_path, 'r') as f:
        bus_names = f['bus']['name'][:]
        nb = len(bus_names)
        
        # Edge data
        edge_grp = f['edge']
        from_idx = edge_grp['from_idx'][:]
        to_idx = edge_grp['to_idx'][:]
        R_ohm = edge_grp['R_ohm'][:]
        X_ohm = edge_grp['X_ohm'][:]
        B_uS = edge_grp['B_uS'][:]
        
        # Shunt data
        shunt_grp = f['shunt']
        shunt_bus_idx = shunt_grp['bus_idx'][:]
        shunt_G_uS = shunt_grp['G_uS'][:]
        shunt_B_uS = shunt_grp['B_uS'][:]
    
    print(f"📊 Original shunt data:")
    print(f"   Shunt elements: {len(shunt_bus_idx)}")
    
    # Check shunt data for NaN
    nan_G_count = np.sum(np.isnan(shunt_G_uS))
    nan_B_count = np.sum(np.isnan(shunt_B_uS))
    print(f"   NaN G values: {nan_G_count}")
    print(f"   NaN B values: {nan_B_count}")
    
    # Show shunt details
    for i in range(len(shunt_bus_idx)):
        bus_idx = shunt_bus_idx[i]
        G_val = shunt_G_uS[i]
        B_val = shunt_B_uS[i]
        
        if bus_idx >= 0 and bus_idx < nb:
            bus_name = bus_names[bus_idx].decode() if isinstance(bus_names[bus_idx], bytes) else str(bus_names[bus_idx])
            print(f"   Shunt {i}: Bus {bus_idx} ({bus_name}) - G={G_val}, B={B_val}")
            
            if np.isnan(G_val) or np.isnan(B_val):
                print(f"      ❌ Found NaN shunt at Bus {bus_idx}")
    
    # Fix NaN shunt values
    print(f"\n🔧 FIXING NaN SHUNT VALUES...")
    shunt_G_uS_fixed = np.where(np.isnan(shunt_G_uS), 0.0, shunt_G_uS)
    shunt_B_uS_fixed = np.where(np.isnan(shunt_B_uS), 0.0, shunt_B_uS)
    
    fixes_made = np.sum(np.isnan(shunt_G_uS)) + np.sum(np.isnan(shunt_B_uS))
    print(f"✅ Fixed {fixes_made} NaN shunt values (replaced with 0.0)")
    
    # Rebuild Y matrix with fixed shunt data
    print(f"\n🎯 REBUILDING Y MATRIX WITH FIXED SHUNTS...")
    Y_full = np.zeros((nb, nb), dtype=complex)
    µS_to_S = 1e-6
    
    # Process edges
    processed_edges = 0
    for i in range(len(from_idx)):
        bus_from = from_idx[i]
        bus_to = to_idx[i]
        R = R_ohm[i]
        X = X_ohm[i]
        B_line = 0.0 if np.isnan(B_uS[i]) else B_uS[i] * µS_to_S
        
        # Skip invalid edges
        if np.isnan(R) or np.isnan(X) or (R == 0 and X == 0):
            continue
        if bus_from < 0 or bus_to < 0 or bus_from >= nb or bus_to >= nb:
            continue
        
        # Calculate admittances
        Z_series = complex(R, X)
        Y_series = 1.0 / Z_series
        Y_shunt_half = 1j * B_line / 2.0
        
        # Update Y matrix
        Y_full[bus_from, bus_to] += -Y_series
        Y_full[bus_to, bus_from] += -Y_series
        Y_full[bus_from, bus_from] += Y_series + Y_shunt_half
        Y_full[bus_to, bus_to] += Y_series + Y_shunt_half
        
        processed_edges += 1
    
    print(f"✅ Processed {processed_edges} edges")
    
    # Add FIXED shunt elements
    shunt_count = 0
    for i in range(len(shunt_bus_idx)):
        bus_i = shunt_bus_idx[i]
        if bus_i >= 0 and bus_i < nb:
            # Use FIXED shunt values
            G_sh = shunt_G_uS_fixed[i] * µS_to_S
            B_sh = shunt_B_uS_fixed[i] * µS_to_S
            Y_shunt = complex(G_sh, B_sh)
            
            Y_full[bus_i, bus_i] += Y_shunt
            shunt_count += 1
            
            bus_name = bus_names[bus_i].decode() if isinstance(bus_names[bus_i], bytes) else str(bus_names[bus_i])
            print(f"   Added shunt to Bus {bus_i} ({bus_name}): Y = {Y_shunt}")
    
    print(f"✅ Added {shunt_count} fixed shunt elements")
    
    # Check final result
    diagonal = np.diag(Y_full)
    nan_count = np.sum(np.isnan(diagonal))
    zero_count = np.sum(np.abs(diagonal) < 1e-15)
    valid_count = len(diagonal) - nan_count - zero_count
    
    print(f"\n🎯 FINAL Y MATRIX RESULTS:")
    print(f"   Total diagonal elements: {len(diagonal)}")
    print(f"   NaN elements: {nan_count}")
    print(f"   Zero elements: {zero_count}")
    print(f"   Valid elements: {valid_count}")
    
    # Show the previously problematic buses
    problem_buses = [11, 19]
    print(f"\n🔍 CHECKING PREVIOUSLY PROBLEMATIC BUSES:")
    for bus_idx in problem_buses:
        if bus_idx < len(diagonal):
            bus_name = bus_names[bus_idx].decode() if isinstance(bus_names[bus_idx], bytes) else str(bus_names[bus_idx])
            yii = diagonal[bus_idx]
            if np.isnan(yii):
                print(f"   Bus {bus_idx} ({bus_name}): Still NaN ❌")
            else:
                print(f"   Bus {bus_idx} ({bus_name}): {yii.real:+.6f} {yii.imag:+.6f}j ✅")
    
    if nan_count == 0:
        print(f"\n🎉 SUCCESS! ALL DIAGONAL ELEMENTS ARE VALID!")
        
        # Convert to sparse matrix
        Y_sparse = csr_matrix(Y_full)
        
        # Save to H5 file
        with h5py.File(h5_file_path, 'r+') as f:
            # Update admittance matrix
            if 'admittance' in f:
                del f['admittance']
            
            Y_grp = f.create_group("admittance")
            Y_grp.create_dataset("data", data=Y_sparse.data)
            Y_grp.create_dataset("indices", data=Y_sparse.indices)
            Y_grp.create_dataset("indptr", data=Y_sparse.indptr)
            Y_grp.create_dataset("shape", data=Y_sparse.shape)
            Y_grp.create_dataset("nnz", data=Y_sparse.nnz)
            
            # Also fix the shunt data in the H5 file
            shunt_grp = f['shunt']
            del shunt_grp['G_uS']
            del shunt_grp['B_uS']
            shunt_grp.create_dataset('G_uS', data=shunt_G_uS_fixed)
            shunt_grp.create_dataset('B_uS', data=shunt_B_uS_fixed)
        
        print(f"💾 UPDATED H5 FILE WITH:")
        print(f"   ✅ Fixed Y matrix (no NaN diagonal)")
        print(f"   ✅ Fixed shunt data (NaN → 0)")
        
        # Show sample of all diagonal values
        print(f"\n📋 SAMPLE DIAGONAL VALUES:")
        for i in range(min(15, len(diagonal))):
            bus_name = bus_names[i].decode() if isinstance(bus_names[i], bytes) else str(bus_names[i])
            yii = diagonal[i]
            magnitude = abs(yii)
            print(f"   Y[{i:2d},{i:2d}] ({bus_name:15s}) = {yii.real:+.6f} {yii.imag:+.6f}j  |Y|={magnitude:.6f} ✅")
        
        # Final validation
        print(f"\n✅ Y MATRIX IS NOW READY FOR GRAPH CONSTRUCTION!")
        print(f"   Shape: {Y_sparse.shape}")
        print(f"   Non-zeros: {Y_sparse.nnz}")
        print(f"   All diagonal elements valid: YES")
        
    else:
        print(f"❌ Still have {nan_count} NaN elements - additional investigation needed")
    
    return Y_full, diagonal

def main():
    h5_file = "enhanced_out/39_Bus_New_England_System_complete_enhanced.h5"
    
    if not os.path.exists(h5_file):
        print(f"❌ File not found: {h5_file}")
        return
    
    Y_matrix, diagonal = fix_nan_shunts_and_rebuild_y(h5_file)

if __name__ == "__main__":
    main()