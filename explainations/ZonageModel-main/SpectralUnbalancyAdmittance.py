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
import itertools
from copy import deepcopy

def refine_zones_by_local_swaps(zone_labels, Pgen, Pload, edge_index, edge_weights, k,
                                 max_group_size=3, max_iter=10, y_threshold=0.0, verbose=True):
    num_nodes = len(zone_labels)
    G = nx.Graph()
    for idx in range(edge_index.shape[1]):
        i, j = edge_index[0, idx], edge_index[1, idx]
        if edge_weights[idx] >= y_threshold:
            G.add_edge(i, j)

    def total_imbalance(zones):
        imbalance = np.zeros(k)
        for i in range(num_nodes):
            imbalance[zones[i]] += Pgen[i] - Pload[i]
        return np.sum(imbalance ** 2)

    def is_valid(zones):
        for z in range(k):
            nodes_in_zone = [i for i in range(num_nodes) if zones[i] == z]
            if not nodes_in_zone:
                return False
            subgraph = G.subgraph(nodes_in_zone)
            if not nx.is_connected(subgraph):
                return False
        return True

    best_labels = deepcopy(zone_labels)
    best_cost = total_imbalance(best_labels)
    swap_counter = 0

    for it in range(max_iter):
        improved = False

        for i in range(num_nodes):
            zi = best_labels[i]
            neighbors = list(G.neighbors(i))
            neighbor_zones = set(best_labels[j] for j in neighbors if best_labels[j] != zi)
            if not neighbor_zones:
                continue

            group_candidates = []
            same_zone_neighbors = [j for j in neighbors if best_labels[j] == zi]
            for s in range(1, max_group_size + 1):
                for combo in itertools.combinations(same_zone_neighbors, s - 1):
                    group = [i] + list(combo)
                    group_candidates.append(group)

            for group in group_candidates:
                current_zone = best_labels[group[0]]
                for target_zone in neighbor_zones:
                    swap_counter += 1
                    new_labels = deepcopy(best_labels)
                    for node in group:
                        new_labels[node] = target_zone

                    if not is_valid(new_labels):
                        continue

                    new_cost = total_imbalance(new_labels)
                    if new_cost < best_cost:
                        if verbose:
                            print(f"  ✅ Move {group} from Z{current_zone+1} → Z{target_zone+1} | Cost ↓ {best_cost:.2f} → {new_cost:.2f}")
                        best_labels = new_labels
                        best_cost = new_cost
                        improved = True
                        break
        if not improved:
            if verbose:
                print(f"\n🛑 Convergence atteinte à l’itération {it+1}")
            break
        elif verbose:
            print(f"\n🔁 Itération {it+1} terminée — coût = {best_cost:.2f}")

    print(f"\n🔢 Nombre total de swaps testés : {swap_counter}")
    return best_labels


def refine_border_buses(zone_labels, edge_index, edge_weights, gen_bus_idx, load_bus_idx, shunt_bus_idx, k):
    num_nodes = len(zone_labels)
    labels = zone_labels.copy()
    changed = True
    iteration = 0

    while changed:
        changed = False
        iteration += 1
        print(f"\n🔄 Raffinement topologique — itération {iteration}")

        for i in range(num_nodes):
            if i in gen_bus_idx or i in load_bus_idx or i in shunt_bus_idx:
                continue  # bus critique

            zi = labels[i]
            neighbor_zones = {}
            for idx in range(edge_index.shape[1]):
                a, b = edge_index[0, idx], edge_index[1, idx]
                if i == a:
                    neighbor = b
                elif i == b:
                    neighbor = a
                else:
                    continue

                z_neighbor = labels[neighbor]
                if z_neighbor not in neighbor_zones:
                    neighbor_zones[z_neighbor] = 0.0
                neighbor_zones[z_neighbor] += edge_weights[idx]

            if len(neighbor_zones) <= 1:
                continue

            max_zone = max(neighbor_zones.items(), key=lambda x: x[1])[0]
            if max_zone != zi and neighbor_zones[max_zone] > neighbor_zones.get(zi, 0):
                print(f"  ✅ Bus {i+1} passe de Z{zi+1} → Z{max_zone+1} (liaison {neighbor_zones[max_zone]:.4f} > {neighbor_zones.get(zi,0):.4f})")
                labels[i] = max_zone
                changed = True

    print("\n✅ Raffinement topologique terminé")
    return labels


def build_zoned_graph(k):
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
    Pgen = np.zeros(num_nodes)
    Pload = np.zeros(num_nodes)
    bus_to_pgen = {30: 250, 31: 731.1, 32: 650, 33: 632, 34: 254, 35: 650, 36: 560, 37: 540, 38: 830, 39: 1000}
    bus_to_pload = {3: 322, 4: 500, 7: 233.8, 8: 522, 12: 7.5, 15: 320, 16: 329, 18: 158, 20: 628, 21: 274,
                    23: 247.5, 24: 308.6, 25: 224, 26: 139, 27: 281, 28: 206, 29: 283.5, 31: 9.2, 39: 1104}
    for i in bus_to_pgen: Pgen[i - 1] = bus_to_pgen[i]
    for i in bus_to_pload: Pload[i - 1] = bus_to_pload[i]

    adj_matrix = np.zeros((num_nodes, num_nodes))
    for idx in range(edge_index.shape[1]):
        i, j = edge_index[0, idx], edge_index[1, idx]
        adj_matrix[i, j] = edge_weights[idx]
        adj_matrix[j, i] = edge_weights[idx]

    sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
    z0 = sc.fit_predict(adj_matrix).astype(float)

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
    zone_labels_init = np.clip(np.round(res.x).astype(int), 0, k-1)

    zone_labels_refined = refine_zones_by_local_swaps(zone_labels_init, Pgen, Pload, edge_index, edge_weights, k)
    zone_labels_toporefined = refine_border_buses(zone_labels_refined, edge_index, edge_weights, gen_bus_idx, load_bus_idx, shunt_bus_idx, k)

    def print_balance(zones, label):
        print(f"\n🔎 Bilan énergétique par zone ({label}) :")
        zone_pgen = {z: 0.0 for z in range(k)}
        zone_pload = {z: 0.0 for z in range(k)}
        for i in range(num_nodes):
            z = int(zones[i])
            zone_pgen[z] += Pgen[i]
            zone_pload[z] += Pload[i]
        for z in range(k):
            pg, pl = zone_pgen[z], zone_pload[z]
            net = pg - pl
            print(f"  • Zone {z+1}:  Pgen = {pg:.2f} MW,  Pload = {pl:.2f} MW,  ➤ Solde = {net:+.2f} MW")

    def plot_zones(zones, title):
        data = Data(x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_weights, dtype=torch.float))
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G, seed=42)
        cmap = colormaps.get_cmap("Pastel1").resampled(k)
        zone_to_color = {z: to_rgba(cmap(z), alpha=0.85) for z in range(k)}
        node_colors = [zone_to_color[int(z)] for z in zones]

        node_shapes_map = {}
        for i in range(data.num_nodes):
            types = []
            if i in gen_bus_idx: types.append('G')
            if i in load_bus_idx: types.append('L')
            if i in shunt_bus_idx: types.append('S')
            node_shapes_map[i] = ''.join(types) if types else 'None'

        shape_style = {
            'G': {'marker': 's', 'size': 800, 'label': 'Generator'},
            'L': {'marker': 'd', 'size': 500, 'label': 'Load'},
            'GL': {'marker': '^', 'size': 900, 'label': 'Gen + Load'},
            'None': {'marker': 'o', 'size': 400, 'label': 'Neutral bus'}
        }

        plt.figure(figsize=(12, 9))
        plt.title(title, fontsize=14)

        edge_weights_np = edge_weights if isinstance(edge_weights, np.ndarray) else edge_weights.numpy()
        nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in edge_index.T],
                               edge_color=edge_weights_np, edge_cmap=plt.cm.plasma,
                               edge_vmin=min(edge_weights_np), edge_vmax=max(edge_weights_np),
                               width=2, alpha=0.6)

        for shape, props in shape_style.items():
            nodes = [i for i in range(data.num_nodes) if node_shapes_map[i] == shape]
            if nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                       node_color=[node_colors[i] for i in nodes],
                                       node_size=props['size'], node_shape=props['marker'],
                                       label=props['label'])

        nx.draw_networkx_labels(G, pos, labels={i: str(i + 1) for i in range(data.num_nodes)}, font_size=9)
        sm = ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(edge_weights_np), vmax=max(edge_weights_np)))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label="|Y₍ᵢⱼ₎| (Admittance)", shrink=0.7)

        zone_patches = [mpatches.Patch(color=zone_to_color[z], label=f"Zone {z + 1}") for z in range(k)]
        shape_patches = [
            plt.Line2D([], [], marker=v['marker'], color='w', label=v['label'], markerfacecolor='gray', markersize=10)
            for k, v in shape_style.items() if k != 'None'
        ]
        plt.legend(handles=zone_patches + shape_patches, loc='lower left', frameon=True)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"zone_plot_{title.replace(' ', '_')}.png", dpi=300)
        plt.show()

    print_balance(zone_labels_init, "Partition initiale (Spectral + L-BFGS-B)")
    print_balance(zone_labels_refined, "Après raffinement par swaps")
    print_balance(zone_labels_toporefined, "Après raffinement topologique")

    plot_zones(zone_labels_init, "Partition initiale (Spectral + L-BFGS-B)")
    plot_zones(zone_labels_refined, "Après raffinement par swaps")
    plot_zones(zone_labels_toporefined, "Partition finale raffinée (topologie)")

# Appel
build_zoned_graph(k=3)
