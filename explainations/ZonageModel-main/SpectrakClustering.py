import h5py
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
import matplotlib.patches as mpatches
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize

k = 3 # CHOOSING HOZ MANY ZONES WE WANT TO DIVIDE THE SYSTEM INTO
def build_zoned_graph(k):
    # H5 file reading
    h5_path = "C:/Users/LBA100468/Documents/microgrid/39_Bus_New_England_System_fixed_complete_enhanced.h5"
    with h5py.File(h5_path, "r") as f:
        node_features = f["features/node_features"][:]
        edge_index = np.stack([f["edge/from_idx"][:], f["edge/to_idx"][:]])
        R = f["features/edge_features"][:, 0]
        X = f["features/edge_features"][:, 1]
        Z = R + 1j * X
        Y = 1.0 / Z
        edge_weights = np.abs(Y)
        gen_bus_idx = set(f["gen/bus_idx"][:])
        load_bus_idx = set(f["load/bus_idx"][:])
        shunt_bus_idx = set(f["shunt/bus_idx"][:]) if "shunt/bus_idx" in f["shunt"] else set()

    num_nodes = node_features.shape[0]

    # ADMITTANCE MATRIX
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for idx in range(edge_index.shape[1]):
        i, j = edge_index[0, idx], edge_index[1, idx]
        adj_matrix[i, j] = edge_weights[idx]
        adj_matrix[j, i] = edge_weights[idx]

    # GEN LOAD LIST
    Pgen = np.zeros(num_nodes)
    Pload = np.zeros(num_nodes)
    bus_to_pgen = {30: 250, 31: 731.1, 32: 650, 33: 632, 34: 254, 35: 650, 36: 560, 37: 540, 38: 830, 39: 1000} #A REMPLIR A LA MAIN
    bus_to_pload = {3: 322, 4: 500, 7: 233.8, 8: 522, 12: 7.5, 15: 320, 16: 329, 18: 158, 20: 628, 21: 274,
                    23: 247.5, 24: 308.6, 25: 224, 26: 139, 27: 281, 28: 206, 29: 283.5, 31: 9.2, 39: 1104}
    for i in bus_to_pgen: Pgen[i - 1] = bus_to_pgen[i]
    for i in bus_to_pload: Pload[i - 1] = bus_to_pload[i]

    # Clustering spectral
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
    z0 = sc.fit_predict(adj_matrix).astype(float)

    # Optimize unbalanced
    def cost(z):
        z = np.clip(np.round(z).astype(int), 0, k-1)
        zone_pgen = {i: 0.0 for i in range(k)}
        zone_pload = {i: 0.0 for i in range(k)}
        for i in range(num_nodes):
            zone_pgen[z[i]] += Pgen[i]
            zone_pload[z[i]] += Pload[i]
        imbalance = np.array([zone_pgen[i] - zone_pload[i] for i in range(k)])
        return np.mean(imbalance**2)

    res = minimize(cost, z0, bounds=[(0, k-1)] * num_nodes, method='L-BFGS-B')
    zone_labels = np.clip(np.round(res.x).astype(int), 0, k-1)

    #PyG  NetworkX
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float),
        zone_label=torch.tensor(zone_labels, dtype=torch.long)
    )
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    # Zone colors
    unique_zones = sorted(np.unique(zone_labels))
    cmap = colormaps.get_cmap("Pastel1").resampled(len(unique_zones))
    zone_to_color = {z: to_rgba(cmap(i), alpha=0.85) for i, z in enumerate(unique_zones)}
    node_colors = [zone_to_color[int(z)] for z in zone_labels]

    # Nodes shape
    node_shapes_map = {}
    for i in range(data.num_nodes):
        types = []
        if i in gen_bus_idx: types.append('G')
        if i in load_bus_idx: types.append('L')
        if i in shunt_bus_idx: types.append('S')
        node_shapes_map[i] = ''.join(types) if types else 'None'

    plt.figure(figsize=(13, 11))
    plt.title("🔋 Clustering optimal des zones (déséquilibre minimisé)", fontsize=15)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[tuple(e) for e in data.edge_index.numpy().T],
        edge_color=edge_weights,
        edge_cmap=plt.cm.plasma,
        edge_vmin=min(edge_weights),
        edge_vmax=max(edge_weights),
        width=2, alpha=0.6
    )

    shape_style = {
        'G': {'marker': 's', 'size': 800, 'label': 'Generator'},
        'L': {'marker': 'd', 'size': 500, 'label': 'Load'},
        'GL': {'marker': '^', 'size': 900, 'label': 'Gen + Load'},
        'None': {'marker': 'o', 'size': 400, 'label': 'Neutral bus'}
    }

    for shape, props in shape_style.items():
        nodes = [i for i in range(data.num_nodes) if node_shapes_map[i] == shape]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=[node_colors[i] for i in nodes],
                node_size=props['size'],
                node_shape=props['marker'],
                label=props['label']
            )

    nx.draw_networkx_labels(G, pos, labels={i: str(i + 1) for i in range(data.num_nodes)}, font_size=9)

    sm = ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="|Y₍ᵢⱼ₎| (Admittance)", shrink=0.7)

    zone_patches = [mpatches.Patch(color=zone_to_color[z], label=f"Zone {z + 1}") for z in unique_zones]
    shape_patches = [
        plt.Line2D([], [], marker=v['marker'], color='w', label=v['label'], markerfacecolor='gray', markersize=10)
        for k, v in shape_style.items() if k != 'None'
    ]
    plt.legend(handles=zone_patches + shape_patches, loc='lower left', frameon=True)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("graph_optimized_zones.png", dpi=300)
    plt.show()

    # === Bilan par zone ===
    zone_pgen = {z: 0.0 for z in range(k)}
    zone_pload = {z: 0.0 for z in range(k)}

    for i in range(num_nodes):
        z = int(zone_labels[i])
        zone_pgen[z] += Pgen[i]
        zone_pload[z] += Pload[i]

    print("\n🔎 Bilan énergétique par zone :")
    for z in range(k):
        pg = zone_pgen[z]
        pl = zone_pload[z]
        net = pg - pl
        print(f"  • Zone {z+1}:  Pgen = {pg:.2f} MW,  Pload = {pl:.2f} MW,  ➤ Solde = {net:+.2f} MW")


# CHOOSING HOZ MANY ZONES WE WANT TO DIVIDE THE SYSTEM INTO
build_zoned_graph(k)  # Choisis le nombre de zones ici


