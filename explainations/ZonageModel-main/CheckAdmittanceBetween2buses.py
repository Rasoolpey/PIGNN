import numpy as np

# Chargement de la matrice Ybus (39×39)
Ybus = np.load("Y_admittance.npy")

def get_admittance(bus_i, bus_j):
    """Affiche la connexion entre deux bus (labels de 1 à 39)."""
    # Décalage pour indices Python (0-based)
    i = bus_i - 1
    j = bus_j - 1

    if i < 0 or j < 0 or i >= Ybus.shape[0] or j >= Ybus.shape[1]:
        print("❌ Numéros de bus invalides (doivent être entre 1 et 39).")
        return

    val = Ybus[i, j]
    module = np.abs(val)
    
    if np.abs(val) < 1e-6 or np.isnan(val):
        print(f"⚠️ Aucune connexion directe entre Bus {bus_i} et Bus {bus_j}.")
    else:
        print(f"🔌 Connexion entre Bus {bus_i} et Bus {bus_j} :")
        print(f"   → Admittance complexe Y[{bus_i},{bus_j}] = {val}")
        print(f"   → Module |Y[{bus_i},{bus_j}]| = {module:.6f}")

get_admittance(17, 27)  # Bus réels 18 et 28 (au lieu de 17, 27 en 0-based)
