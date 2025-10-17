import h5py
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

#  Chemin vers ton fichier HDF5
fichier_hdf5 = "C:/Users/LBA100468/Documents/microgrid/39_Bus_New_England_System_fixed_complete_enhanced.h5"

with h5py.File(fichier_hdf5, "r") as f:
    data = f["admittance/data"][:]
    indices = f["admittance/indices"][:]
    indptr = f["admittance/indptr"][:]
    print("✅ Données chargées :")
    print(f"  - data   shape : {data.shape}")
    print(f"  - indices shape: {indices.shape}")
    print(f"  - indptr  shape: {indptr.shape}")

#  Construction de la matrice creuse (CSR)
n = 39  
Y_sparse = csr_matrix((data, indices, indptr), shape=(n, n))
Y_dense = Y_sparse.toarray()

#  Sauvegarde 
np.save("Y_admittance.npy", Y_dense)      # Format binaire numpy


plt.figure(figsize=(6, 5))
plt.imshow(np.abs(Y_dense), cmap="viridis")
plt.colorbar(label="|Y_ij|")
plt.title("Admittance Matrix |Y| (39×39)")
plt.xlabel("Bus j")
plt.ylabel("Bus i")
plt.grid(False)
plt.tight_layout()
plt.savefig("Y_admittance_heatmap.png", dpi=300)
plt.show()
