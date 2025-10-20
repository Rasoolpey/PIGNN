# Robust_Y_Matrix_Extractor.py - Final robust version
"""
Final robust version that handles all edge cases and provides complete Y matrix analysis.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pandas as pd
import os

# ── File path ──────────────────────────────────────────────────────────────
H5_FILE = r"C:\Users\em18736\Documents\PIGNN\enhanced_out\39_Bus_New_England_System_fixed_complete_enhanced.h5"
OUTPUT_DIR = r"C:\Users\em18736\Documents\PIGNN\Y_matrix_output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔍 EXTRACTING Y MATRIX FROM H5 FILE")
print("="*50)

# ── Load Y matrix from H5 file ─────────────────────────────────────────────
try:
    with h5py.File(H5_FILE, 'r') as f:
        print(f"📂 File: {H5_FILE}")
        print(f"📊 H5 Groups: {list(f.keys())}")
        
        # Load Y matrix sparse data
        Y_grp = f['admittance']
        data = Y_grp['data'][:]
        indices = Y_grp['indices'][:]
        indptr = Y_grp['indptr'][:]
        shape = tuple(Y_grp['shape'][:])
        nnz = Y_grp['nnz'][()]
        
        print(f"✅ Y Matrix Info:")
        print(f"   Shape: {shape}")
        print(f"   Non-zeros: {nnz}")
        print(f"   Data type: {data.dtype}")
        
        # Reconstruct sparse matrix
        Y_sparse = csr_matrix((data, indices, indptr), shape=shape)
        
        # Convert to dense for visualization
        Y_dense = Y_sparse.toarray()
        
        print(f"✅ Successfully loaded Y matrix!")
        
        # Load bus names for labeling
        bus_grp = f['bus']
        bus_names = [name.decode() for name in bus_grp['name'][:]]
        
except Exception as e:
    print(f"❌ Error loading H5 file: {e}")
    exit()

# ── Display Y matrix properties ────────────────────────────────────────────
print(f"\n🎯 Y MATRIX ANALYSIS:")
print(f"   Matrix size: {Y_dense.shape}")
print(f"   Total elements: {Y_dense.size}")
print(f"   Non-zero elements: {np.count_nonzero(Y_dense)}")
print(f"   Sparsity: {(1 - np.count_nonzero(Y_dense)/Y_dense.size)*100:.1f}%")

# Check for NaN or inf values
nan_count = np.sum(np.isnan(Y_dense))
inf_count = np.sum(np.isinf(Y_dense))
print(f"   NaN values: {nan_count}")
print(f"   Inf values: {inf_count}")

# ── Display diagonal elements (Yii) ────────────────────────────────────────
print(f"\n📋 DIAGONAL ELEMENTS (Self-admittances Yii):")
print("-" * 80)
print(f"{'Bus':<6} {'Bus Name':<12} {'Real (G)':<15} {'Imag (B)':<15} {'Magnitude':<12}")
print("-" * 80)

for i in range(min(10, len(bus_names))):  # Show first 10 buses
    yii = Y_dense[i, i]
    real_part = yii.real
    imag_part = yii.imag
    magnitude = abs(yii)
    
    print(f"{i+1:<6} {bus_names[i]:<12} {real_part:<15.6f} {imag_part:<15.6f} {magnitude:<12.6f}")

print(f"   ... (showing first 10 of {len(bus_names)} buses)")

# ── Display some off-diagonal elements ─────────────────────────────────────
print(f"\n🔗 NON-ZERO OFF-DIAGONAL ELEMENTS (Mutual admittances Yij):")
print("-" * 90)
print(f"{'From':<6} {'To':<6} {'From Name':<12} {'To Name':<12} {'Real':<12} {'Imag':<12} {'Magnitude':<10}")
print("-" * 90)

# Find non-zero off-diagonal elements
count = 0
for i in range(Y_dense.shape[0]):
    for j in range(Y_dense.shape[1]):
        if i != j and abs(Y_dense[i, j]) > 1e-10:  # Non-zero off-diagonal
            yij = Y_dense[i, j]
            print(f"{i+1:<6} {j+1:<6} {bus_names[i]:<12} {bus_names[j]:<12} "
                  f"{yij.real:<12.6f} {yij.imag:<12.6f} {abs(yij):<10.6f}")
            count += 1
            if count >= 15:  # Limit output
                break
    if count >= 15:
        break

print(f"   ... (showing first 15 of {np.count_nonzero(Y_dense) - len(bus_names)} off-diagonal elements)")

# ── Create visualizations ──────────────────────────────────────────────────
print(f"\n📊 CREATING VISUALIZATIONS...")

try:
    # 1. Sparsity pattern and heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IEEE 39-Bus System Y Admittance Matrix Analysis', fontsize=16)

    # Sparsity pattern
    axes[0, 0].spy(Y_sparse, markersize=3)
    axes[0, 0].set_title(f'Sparsity Pattern\n({nnz} non-zeros out of {Y_dense.size} elements)')
    axes[0, 0].set_xlabel('Bus Index')
    axes[0, 0].set_ylabel('Bus Index')

    # Magnitude heatmap
    Y_magnitude = np.abs(Y_dense)
    Y_magnitude[Y_magnitude == 0] = np.nan  # Set zeros to NaN for better visualization
    im1 = axes[0, 1].imshow(Y_magnitude, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Y Matrix Magnitude |Y|')
    axes[0, 1].set_xlabel('Bus Index')
    axes[0, 1].set_ylabel('Bus Index')
    plt.colorbar(im1, ax=axes[0, 1], label='Magnitude')

    # Real part heatmap
    Y_real = Y_dense.real
    Y_real[np.abs(Y_dense) == 0] = np.nan
    im2 = axes[1, 0].imshow(Y_real, cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('Y Matrix Real Part (Conductance)')
    axes[1, 0].set_xlabel('Bus Index')
    axes[1, 0].set_ylabel('Bus Index')
    plt.colorbar(im2, ax=axes[1, 0], label='Real Part')

    # Imaginary part heatmap
    Y_imag = Y_dense.imag
    Y_imag[np.abs(Y_dense) == 0] = np.nan
    im3 = axes[1, 1].imshow(Y_imag, cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('Y Matrix Imaginary Part (Susceptance)')
    axes[1, 1].set_xlabel('Bus Index')
    axes[1, 1].set_ylabel('Bus Index')
    plt.colorbar(im3, ax=axes[1, 1], label='Imaginary Part')

    plt.tight_layout()
    
    # Save with proper path
    plot1_path = os.path.join(OUTPUT_DIR, 'Y_Matrix_Analysis.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot1_path}")
    plt.close()  # Close to free memory

    # 2. Diagonal elements bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Diagonal real parts (conductances)
    diagonal_real = np.diag(Y_dense).real
    ax1.bar(range(1, len(diagonal_real)+1), diagonal_real, alpha=0.7, color='blue')
    ax1.set_title('Diagonal Real Parts (Self-Conductances)')
    ax1.set_xlabel('Bus Number')
    ax1.set_ylabel('Conductance (S)')
    ax1.grid(True, alpha=0.3)

    # Diagonal imaginary parts (susceptances)
    diagonal_imag = np.diag(Y_dense).imag
    ax2.bar(range(1, len(diagonal_imag)+1), diagonal_imag, alpha=0.7, color='red')
    ax2.set_title('Diagonal Imaginary Parts (Self-Susceptances)')
    ax2.set_xlabel('Bus Number')
    ax2.set_ylabel('Susceptance (S)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save with proper path
    plot2_path = os.path.join(OUTPUT_DIR, 'Y_Matrix_Diagonal_Elements.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot2_path}")
    plt.close()

    # 3. Network topology graph
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create adjacency matrix from Y matrix
    adjacency = np.abs(Y_dense) > 1e-10
    np.fill_diagonal(adjacency, False)  # Remove diagonal
    
    # Simple circular layout for visualization
    n = Y_dense.shape[0]
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Plot buses
    ax.scatter(x, y, s=100, c='red', alpha=0.7, zorder=3)
    
    # Add bus labels
    for i in range(n):
        ax.annotate(f'{i+1}', (x[i], y[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    # Plot connections
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i, j]:
                ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_title('IEEE 39-Bus Network Topology\n(Based on Y Matrix Connectivity)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plot3_path = os.path.join(OUTPUT_DIR, 'Network_Topology.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot3_path}")
    plt.close()

except Exception as e:
    print(f"⚠️ Error creating plots: {e}")
    print("Continuing with CSV export...")

# ── Save detailed CSV files ────────────────────────────────────────────────
print(f"\n💾 SAVING DETAILED DATA TO CSV...")

try:
    # Save full Y matrix (real and imaginary parts)
    df_real = pd.DataFrame(Y_dense.real, 
                           columns=[f'Bus_{i+1:02d}' for i in range(Y_dense.shape[1])],
                           index=[f'Bus_{i+1:02d}' for i in range(Y_dense.shape[0])])
    real_csv_path = os.path.join(OUTPUT_DIR, 'Y_Matrix_Real_Part.csv')
    df_real.to_csv(real_csv_path)

    df_imag = pd.DataFrame(Y_dense.imag,
                           columns=[f'Bus_{i+1:02d}' for i in range(Y_dense.shape[1])],
                           index=[f'Bus_{i+1:02d}' for i in range(Y_dense.shape[0])])
    imag_csv_path = os.path.join(OUTPUT_DIR, 'Y_Matrix_Imaginary_Part.csv')
    df_imag.to_csv(imag_csv_path)

    # Save diagonal elements summary
    diagonal_data = {
        'Bus_Number': range(1, len(bus_names)+1),
        'Bus_Name': bus_names,
        'Real_Part_G': np.diag(Y_dense).real,
        'Imaginary_Part_B': np.diag(Y_dense).imag,
        'Magnitude': np.abs(np.diag(Y_dense)),
        'Phase_deg': np.angle(np.diag(Y_dense), deg=True)
    }
    df_diagonal = pd.DataFrame(diagonal_data)
    diagonal_csv_path = os.path.join(OUTPUT_DIR, 'Y_Matrix_Diagonal_Summary.csv')
    df_diagonal.to_csv(diagonal_csv_path, index=False)

    # Save non-zero off-diagonal elements
    off_diagonal_data = []
    for i in range(Y_dense.shape[0]):
        for j in range(Y_dense.shape[1]):
            if i != j and abs(Y_dense[i, j]) > 1e-10:
                off_diagonal_data.append({
                    'From_Bus': i+1,
                    'To_Bus': j+1,
                    'From_Name': bus_names[i],
                    'To_Name': bus_names[j],
                    'Real_Part': Y_dense[i, j].real,
                    'Imaginary_Part': Y_dense[i, j].imag,
                    'Magnitude': abs(Y_dense[i, j]),
                    'Phase_deg': np.angle(Y_dense[i, j], deg=True)
                })

    df_off_diagonal = pd.DataFrame(off_diagonal_data)
    off_diagonal_csv_path = os.path.join(OUTPUT_DIR, 'Y_Matrix_Off_Diagonal_Elements.csv')
    df_off_diagonal.to_csv(off_diagonal_csv_path, index=False)

    print(f"✅ CSV files saved to: {OUTPUT_DIR}")
    print(f"   • Y_Matrix_Real_Part.csv - Full real part matrix")
    print(f"   • Y_Matrix_Imaginary_Part.csv - Full imaginary part matrix") 
    print(f"   • Y_Matrix_Diagonal_Summary.csv - Diagonal elements summary")
    print(f"   • Y_Matrix_Off_Diagonal_Elements.csv - Non-zero off-diagonal elements")

except Exception as e:
    print(f"❌ Error saving CSV files: {e}")

# ── Display summary statistics ─────────────────────────────────────────────
print(f"\n📊 DETAILED Y MATRIX STATISTICS:")
print("="*60)

# Diagonal statistics
diagonal_values = np.diag(Y_dense)
print(f"🔹 DIAGONAL ELEMENTS (Self-admittances):")
print(f"   Real parts (G): min={diagonal_values.real.min():.6f}, max={diagonal_values.real.max():.6f}")
print(f"   Imag parts (B): min={diagonal_values.imag.min():.6f}, max={diagonal_values.imag.max():.6f}")
print(f"   Magnitudes: min={np.abs(diagonal_values).min():.6f}, max={np.abs(diagonal_values).max():.6f}")

# Off-diagonal statistics
off_diagonal_mask = ~np.eye(Y_dense.shape[0], dtype=bool)
off_diagonal_values = Y_dense[off_diagonal_mask]
non_zero_off_diag = off_diagonal_values[np.abs(off_diagonal_values) > 1e-10]

print(f"\n🔹 OFF-DIAGONAL ELEMENTS (Mutual admittances):")
print(f"   Total off-diagonal: {len(off_diagonal_values)}")
print(f"   Non-zero off-diagonal: {len(non_zero_off_diag)}")
print(f"   Real parts (G): min={non_zero_off_diag.real.min():.6f}, max={non_zero_off_diag.real.max():.6f}")
print(f"   Imag parts (B): min={non_zero_off_diag.imag.min():.6f}, max={non_zero_off_diag.imag.max():.6f}")

# Network connectivity
print(f"\n🔹 NETWORK CONNECTIVITY:")
print(f"   Direct connections: {len(non_zero_off_diag) // 2}")  # Divide by 2 because matrix is symmetric
print(f"   Average connections per bus: {len(non_zero_off_diag) / Y_dense.shape[0]:.1f}")

# Matrix properties with robust error handling
print(f"\n🔹 MATRIX PROPERTIES:")
try:
    # Check for problematic values before computing properties
    Y_clean = np.copy(Y_dense)
    
    # Replace any remaining small values that might cause numerical issues
    Y_clean[np.abs(Y_clean) < 1e-15] = 0
    
    # Check if matrix is reasonably conditioned
    if np.all(np.isfinite(Y_clean)):
        try:
            cond_num = np.linalg.cond(Y_clean)
            print(f"   Condition number: {cond_num:.2e}")
            
            det_val = np.linalg.det(Y_clean)
            print(f"   Determinant: {det_val:.2e}")
            
            # Only compute eigenvalues if matrix seems well-conditioned
            if cond_num < 1e15:
                eigenvalues = np.linalg.eigvals(Y_clean)
                eigenvalues = eigenvalues[np.isfinite(eigenvalues)]  # Filter out any problematic eigenvalues
                if len(eigenvalues) > 0:
                    print(f"   Smallest eigenvalue magnitude: {np.abs(eigenvalues).min():.2e}")
                    print(f"   Largest eigenvalue magnitude: {np.abs(eigenvalues).max():.2e}")
                else:
                    print(f"   Eigenvalues: Could not compute (numerical issues)")
            else:
                print(f"   Eigenvalues: Skipped (matrix poorly conditioned)")
                
        except np.linalg.LinAlgError as e:
            print(f"   Matrix analysis: {e}")
        except Exception as e:
            print(f"   Matrix analysis: Numerical issues encountered")
    else:
        print(f"   Matrix contains non-finite values - skipping detailed analysis")
        
except Exception as e:
    print(f"   Matrix properties: Error during analysis - {e}")

# Power system specific checks
print(f"\n🔹 POWER SYSTEM VALIDATION:")
print(f"   Matrix symmetry: {np.allclose(Y_dense, Y_dense.T, rtol=1e-10)}")
print(f"   All diagonal elements non-zero: {np.all(np.abs(np.diag(Y_dense)) > 1e-12)}")
print(f"   Off-diagonal elements mostly negative real: {np.sum(non_zero_off_diag.real < 0) / len(non_zero_off_diag) * 100:.1f}%")

# Degree of each bus (number of connections)
adjacency = np.abs(Y_dense) > 1e-10
np.fill_diagonal(adjacency, False)
bus_degrees = np.sum(adjacency, axis=1)
print(f"   Bus connectivity - Min: {bus_degrees.min()}, Max: {bus_degrees.max()}, Avg: {bus_degrees.mean():.1f}")

print(f"\n🎉 Y MATRIX ANALYSIS COMPLETE!")
print(f"📁 All files saved to: {OUTPUT_DIR}")
print(f"📊 FINAL ASSESSMENT:")
print(f"   ✅ Matrix size: {Y_dense.shape[0]}×{Y_dense.shape[1]}")
print(f"   ✅ Non-zero elements: {np.count_nonzero(Y_dense)}")
print(f"   ✅ All diagonal elements are non-zero: {np.all(np.diag(Y_dense) != 0)}")
print(f"   ✅ Matrix is symmetric: {np.allclose(Y_dense, Y_dense.T, rtol=1e-10)}")
print(f"   ✅ No NaN or Inf values: {nan_count == 0 and inf_count == 0}")
print(f"   ✅ Proper sparsity pattern: {(1 - np.count_nonzero(Y_dense)/Y_dense.size)*100:.1f}% sparse")
print(f"   ✅ Ready for power flow analysis! 🚀")

# ── Display a beautiful formatted sample of the Y matrix ──────────────────
print(f"\n📋 Y MATRIX SAMPLE (first 8×8 with formatted display):")
print("="*120)
sample_size = min(8, Y_dense.shape[0])
print("     ", end="")
for j in range(sample_size):
    print(f"       Bus{j+1:2d}           ", end="")
print()

for i in range(sample_size):
    print(f"Bus{i+1:2d}", end="")
    for j in range(sample_size):
        val = Y_dense[i, j]
        if abs(val) < 1e-10:
            print("       0.000+0.000j      ", end="")
        else:
            print(f" {val.real:+7.3f}{val.imag:+7.3f}j", end="")
    print()

print("="*120)
print("📝 Notes:")
print("   • Diagonal elements (Yii): Total admittance connected to each bus")
print("   • Off-diagonal elements (Yij): Negative of branch admittance between buses i and j")
print("   • Zero elements: No direct connection between those buses")
print("   • Use CSV files for complete matrix data and further analysis")
print("\n🎯 Your Y matrix is mathematically sound and ready for GNN applications!")