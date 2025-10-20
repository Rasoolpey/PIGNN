import h5py
import numpy as np

# Remplace ceci par le chemin réel vers ton fichier .h5
INPUT_H5_FILE = "C:/Users/em18736/Documents/PIGNN/enhanced_out/39_Bus_New_England_System_complete_enhanced.h5"
OUTPUT_NPY_FILE = "admittance_matrix.npy"
NUM_BUSES = 39

with h5py.File(INPUT_H5_FILE, "r") as f:
    indices = f["admittance/indices"][:]
    data = f["admittance/data"][:]

Y = np.zeros((NUM_BUSES, NUM_BUSES), dtype=np.float32)

for idx, y_value in zip(indices, data):
    i = idx // NUM_BUSES
    j = idx % NUM_BUSES
    Y[i, j] = y_value
    Y[j, i] = y_value

np.save(OUTPUT_NPY_FILE, Y)
print(f"✅ Matrice d'admittance Y enregistrée dans {OUTPUT_NPY_FILE}")
