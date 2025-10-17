import h5py
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import colormaps
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize
import itertools
from copy import deepcopy

#  Vérification de la connexité des zone
def is_zone_partition_valid(zones, G_full, k):
    for z in range(k):
        nodes_z = [i for i, lab in enumerate(zones) if lab == z]
        if not nodes_z:
            return False
        sub = G_full.subgraph(nodes_z)
        if not nx.is_connected(sub):
            return False
        # Chaque bus doit avoir au moins un voisin dans sa zone
        for i in nodes_z:
            if len([j for j in G_full.neighbors(i) if zones[j] == z]) == 0:
                return False
    return True

# Raffinement par swaps (déséquilibre Pgen−Pload)
def refine_zones_by_local_swaps(zone_labels, Pgen, Pload, edge_index, edge_weights, G_full, k,
                                max_group_size=3, max_iter=10, y_threshold=0.0, verbose=True):
    num_nodes = len(zone_labels)
    G = nx.Graph()
    for idx in range(edge_index.shape[1]):
        i, j = int(edge_index[0, idx]), int(edge_index[1, idx])
        if edge_weights[idx] >= y_threshold:
            G.add_edge(i, j)

    def total_imbalance(zones):
        imb = np.zeros(k)
        for i in range(num_nodes):
            imb[zones[i]] += Pgen[i] - Pload[i]
        return np.sum(imb**2), imb

    best = deepcopy(zone_labels)
    best_cost, best_imb = total_imbalance(best)

    for it in range(max_iter):
        improved = False
        for i in range(num_nodes):
            zi = best[i]
            neigh = list(G.neighbors(i))
            neigh_zones = set(best[j] for j in neigh if best[j] != zi)
            if not neigh_zones:
                continue
            same_neigh = [j for j in neigh if best[j] == zi]
            for s in range(1, max_group_size + 1):
                for combo in itertools.combinations(same_neigh, s - 1):
                    group = [i] + list(combo)
                    for target in neigh_zones:
                        new = best.copy()
                        for n in group:
                            new[n] = target
                        if not is_zone_partition_valid(new, G_full, k):
                            continue
                        c, _ = total_imbalance(new)
                        if c < best_cost:
                            if verbose:
                                print(f"  ✅ Move {group} Z{zi+1}→Z{target+1} cost {best_cost:.2f}→{c:.2f}")
                            best, best_cost = new, c
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            if verbose:
                print(f"\n🛑 Convergence itération {it+1}")
            break
        else:
            if verbose:
                print(f"\n🔁 Itération {it+1} coût = {best_cost:.2f}")
    return best

# === Raffinement par sensibilité dV/dP ===
def refine_by_sensitivity(zone_labels, dV_dP, success_flags, G_full, k):
    labels = zone_labels.copy()
    N = len(labels)
    for i in range(N):
        if not success_flags[i]:
            zi = labels[i]
            best_z, best_score = zi, np.sum(np.abs(dV_dP[i, labels == zi]))
            for z in range(k):
                if z == zi:
                    continue
                score = np.sum(np.abs(dV_dP[i, labels == z]))
                if score < best_score:
                    tmp = labels.copy()
                    tmp[i] = z
                    if is_zone_partition_valid(tmp, G_full, k):
                        best_score, best_z = score, z
            if best_z != zi:
                print(f"  🔁 Bus {i+1} Z{zi+1}→Z{best_z+1} score {best_score:.4f}")
                labels[i] = best_z
    return labels

# Affichage des déséquilibres par zone

def print_balance(labels, Pgen, Pload, stage):
    k_local = max(labels) + 1
    pg = np.zeros(k_local)
    pl = np.zeros(k_local)
    for i, lab in enumerate(labels):
        pg[lab] += Pgen[i]
        pl[lab] += Pload[i]
    imb = pg - pl
    print(f"\n📊 {stage} :")
    for z in range(k_local):
        print(f"  Zone {z+1}: Pgen={pg[z]:.1f}, Pload={pl[z]:.1f}, Δ={imb[z]:.1f}")

# Tracé des zones avec formes de nœuds et colorbar

def plot_zones(nodes, edge_index, edge_weights, Pgen, Pload, labels, title):
    data = Data(x=torch.tensor(nodes, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_weights, dtype=torch.float))
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)


    k = max(labels) + 1
    cmap = colormaps.get_cmap('Pastel1').resampled(k)
    zone_to_color = {z: to_rgba(cmap(z), alpha=0.8) for z in range(k)}
    node_colors = [zone_to_color[lab] for lab in labels]

    node_shapes_map = {}
    for i in range(len(labels)):
        g = Pgen[i] > 0
        l = Pload[i] > 0
        if g and l:
            node_shapes_map[i] = 'GL'
        elif g:
            node_shapes_map[i] = 'G'
        elif l:
            node_shapes_map[i] = 'L'
        else:
            node_shapes_map[i] = 'None'

    shape_style = {
        'G':  {'marker': 's', 'size': 800, 'label': 'Generator'},
        'L':  {'marker': 'd', 'size': 500, 'label': 'Load'},
        'GL': {'marker': '^', 'size': 900, 'label': 'Gen + Load'},
        'None': {'marker': 'o', 'size': 400, 'label': 'Neutral bus'}
    }

    plt.figure(figsize=(12, 9))
    plt.title(title, fontsize=14)

    ew = np.array(edge_weights)
    nx.draw_networkx_edges(G, pos,
                           edgelist=[tuple(e) for e in edge_index.T],
                           edge_color=ew,
                           edge_cmap=plt.cm.plasma,
                           edge_vmin=ew.min(), edge_vmax=ew.max(),
                           width=2, alpha=0.6)

    for shape, props in shape_style.items():
        nodes_list = [i for i in range(len(labels)) if node_shapes_map[i] == shape]
        if nodes_list:
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=nodes_list,
                                   node_color=[node_colors[i] for i in nodes_list],
                                   node_size=props['size'],
                                   node_shape=props['marker'],
                                   label=props['label'])

    nx.draw_networkx_labels(G, pos,
                            labels={i: str(i+1) for i in range(len(labels))},
                            font_size=9)

    sm = ScalarMappable(cmap=plt.cm.plasma,
                        norm=plt.Normalize(vmin=ew.min(), vmax=ew.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(),
                 label="|Y₍ᵢⱼ₎| (Admittance)", shrink=0.7)

    zone_patches = [mpatches.Patch(color=zone_to_color[z], label=f"Zone {z+1}") for z in range(k)]
    shape_patches = [plt.Line2D([], [], marker=v['marker'], color='w', label=v['label'],
                                markerfacecolor='gray', markersize=10)
                     for v in shape_style.values() if v['label']!='Neutral bus']
    plt.legend(handles=zone_patches + shape_patches,
               loc='lower left', frameon=True)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"zone_plot_{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

# === Pipeline complet ===

def build_zoned_graph(k=3, conn_weight=1e4):
    # Chargement des données
    npz = np.load("voltage_sensitivity_data.npz", allow_pickle=True)
    dV_dP = npz["dV_dP"]
    success_flags = npz["success_flags"].astype(bool)

    h5_path = "C:/Users/LBA100468/Documents/microgrid/39_Bus_New_England_System_fixed_complete_enhanced.h5"
    with h5py.File(h5_path, "r") as f:
        nodes = f["features/node_features"][:]
        edge_index = np.stack([f["edge/from_idx"][:], f["edge/to_idx"][:]])
        R = f["features/edge_features"][:, 0]
        X = f["features/edge_features"][:, 1]
        Y = 1.0/(R + 1j*X)
        edge_weights = np.abs(Y)

    N = len(nodes)
    # Initialisation Pgen/Pload
    Pgen = np.zeros(N); Pload = np.zeros(N)
    bus_to_pgen = {30:250,31:731.1,32:650,33:632,34:254,35:650,36:560,37:540,38:830,39:1000}
    bus_to_pload= {3:322,4:500,7:233.8,8:522,12:7.5,15:320,16:329,18:158,20:628,21:274,
                   23:247.5,24:308.6,25:224,26:139,27:281,28:206,29:283.5,31:9.2,39:1104}
    for b,v in bus_to_pgen.items(): Pgen[b-1] = v
    for b,v in bus_to_pload.items(): Pload[b-1] = v

    # Graphe complet pour connexité
    G_full = nx.Graph()
    for idx in range(edge_index.shape[1]):
        i,j = int(edge_index[0,idx]), int(edge_index[1,idx])
        G_full.add_edge(i,j, weight=edge_weights[idx])

    # Matrice d'admittance
    adj = np.zeros((N,N))
    for idx in range(edge_index.shape[1]):
        i,j = edge_index[0,idx], edge_index[1,idx]
        adj[i,j] = adj[j,i] = edge_weights[idx]

    # Partition initiale
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
    z0 = sc.fit_predict(adj).astype(float)

    def cost(z):
        z = np.clip(np.round(z).astype(int),0,k-1)
        pg = np.zeros(k); pl = np.zeros(k)
        for i in range(N): pg[z[i]] += Pgen[i]; pl[z[i]] += Pload[i]
        imb = pg - pl
        cost_imb = np.mean(imb**2)
        pen = 0.0
        for zone in range(k):
            nodes_z = [i for i in range(N) if z[i]==zone]
            if nodes_z:
                cc = nx.number_connected_components(G_full.subgraph(nodes_z))
                if cc>1: pen += (cc-1)*conn_weight
                for i in nodes_z:
                    if len([j for j in G_full.neighbors(i) if z[j]==zone])==0:
                        pen += conn_weight
        return cost_imb + pen

    res = minimize(cost, z0, bounds=[(0,k-1)]*N, method='L-BFGS-B')
    labels_init = np.clip(np.round(res.x).astype(int),0,k-1)

    # Raffinements
    labels_swaps = refine_zones_by_local_swaps(labels_init, Pgen, Pload, edge_index, edge_weights, G_full, k)
    labels_final = refine_by_sensitivity(labels_swaps, dV_dP, success_flags, G_full, k)

    print_balance(labels_init, Pgen, Pload, "Partition initiale (Spectral+L-BFGS-B)")
    plot_zones(nodes, edge_index, edge_weights, Pgen, Pload, labels_init, "Partition initiale")

    print_balance(labels_swaps, Pgen, Pload, "Après raffinement par swaps")
    plot_zones(nodes, edge_index, edge_weights, Pgen, Pload, labels_swaps, "Après swaps")

    print_balance(labels_final, Pgen, Pload, "Partition finale raffinée (topologie)")
    plot_zones(nodes, edge_index, edge_weights, Pgen, Pload, labels_final, "Finale raffinée")

if __name__ == "__main__":
    build_zoned_graph(k=3)
