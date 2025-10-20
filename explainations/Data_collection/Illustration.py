# H5_Graph_Constructor.py - 2025-07-20
"""
Construct a graph from .h5 file with buses as nodes and connections based on Y admittance matrix.
Following the syntax and flow of First_model.py and feature_extraction.py.
"""

import sys
import os
import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from datetime import datetime
import pandas as pd

# ── Helper functions following First_model.py style ────────────────────────
def safe_read_dataset(dataset):
    """Safely read HDF5 dataset, handling both scalar and array data"""
    if dataset.shape == ():  # Scalar dataset
        value = dataset[()]
        if isinstance(value, bytes):
            return value.decode()
        return value
    else:  # Array dataset
        if dataset.dtype.kind in ['S', 'U']:  # String data
            return [item.decode() if isinstance(item, bytes) else str(item) for item in dataset[:]]
        else:
            return dataset[:]

def has(obj, attr):
    """Check if object has attribute (following feature_extraction.py style)"""
    try:
        return hasattr(obj, attr)
    except:
        return False

def get(obj, attr, default=np.nan):
    """Get attribute value with default (following feature_extraction.py style)"""
    try:
        return getattr(obj, attr, default)
    except:
        return default

def print_header(title):
    """Print formatted header (following First_model.py style)"""
    print(f"\n🔧 {title}")
    print("=" * 60)

# ── Main Graph Construction Class ─────────────────────────────────────────
class H5GraphConstructor:
    """
    Graph constructor following the style and flow of feature_extraction.py
    """
    
    def __init__(self, h5_file_path):
        """Initialize constructor with H5 file path"""
        self.h5_file_path = h5_file_path
        self.data = {}
        self.graph = None
        self.Y_matrix = None
        self.buses = []
        self.num_buses = 0
        
        print_header("H5 GRAPH CONSTRUCTOR")
        print(f"📁 File: {os.path.basename(h5_file_path)}")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"❌ H5 file not found: {h5_file_path}")
    
    def explore_h5_structure(self):
        """Explore H5 file structure to understand available data"""
        print(f"🔍 EXPLORING H5 FILE STRUCTURE")
        print("-" * 40)
        
        with h5py.File(self.h5_file_path, 'r') as f:
            def print_structure(name, obj, depth=0):
                indent = "  " * depth
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/ (Group)")
                    if depth < 2:  # Limit depth to avoid too much output
                        for key in obj.keys():
                            print_structure(key, obj[key], depth + 1)
                elif isinstance(obj, h5py.Dataset):
                    shape_str = f"{obj.shape}" if obj.shape != () else "scalar"
                    print(f"{indent}📄 {name} (Dataset: {shape_str})")
            
            for key in f.keys():
                print_structure(key, f[key])
        print()

    def load_h5_data(self):
        """Load data from H5 file following feature_extraction.py pattern"""
        print_header("LOADING H5 DATA")
        
        # First explore structure
        self.explore_h5_structure()
        
        with h5py.File(self.h5_file_path, 'r') as f:
            # Load bus data (essential for graph nodes)
            if 'bus' in f:
                self.data['bus'] = {}
                bus_grp = f['bus']
                for key in bus_grp.keys():
                    self.data['bus'][key] = safe_read_dataset(bus_grp[key])
                
                self.buses = self.data['bus']['name']
                self.num_buses = len(self.buses)
                print(f"✅ Loaded bus data: {self.num_buses} buses")
            else:
                raise ValueError("❌ No bus data found in H5 file")
            
            # Load edge data if available
            if 'edge' in f:
                self.data['edge'] = {}
                edge_grp = f['edge']
                for key in edge_grp.keys():
                    self.data['edge'][key] = safe_read_dataset(edge_grp[key])
                print(f"✅ Loaded edge data: {len(self.data['edge']['name'])} edges")
            
            # Load Y admittance matrix (sparse format)
            if 'admittance' in f:
                self.data['admittance'] = {}
                Y_grp = f['admittance']
                
                # Read sparse matrix components
                Y_data = Y_grp['data'][:]
                Y_indices = Y_grp['indices'][:]
                Y_indptr = Y_grp['indptr'][:]
                Y_shape = Y_grp['shape'][:]
                
                # Reconstruct sparse matrix
                self.Y_matrix = csr_matrix((Y_data, Y_indices, Y_indptr), shape=Y_shape)
                print(f"✅ Loaded Y admittance matrix: {Y_shape} with {Y_grp['nnz'][()]} non-zeros")
            
            # Load additional data following feature_extraction.py pattern
            for group_name in ['gen', 'load', 'shunt']:
                if group_name in f:
                    self.data[group_name] = {}
                    grp = f[group_name]
                    for key in grp.keys():
                        self.data[group_name][key] = safe_read_dataset(grp[key])
                    print(f"✅ Loaded {group_name} data")
                    
    def construct_graph_from_gen_load_connections(self):
        """Construct graph based on generator and load connections to buses"""
        print(f"🔄 Attempting graph construction from generator/load connections...")
        
        connections = []
        
        # Create a bus connectivity map based on generators and loads
        bus_connections = {}
        
        # Add generator connections
        if 'gen' in self.data and 'bus_idx' in self.data['gen']:
            gen_bus_indices = self.data['gen']['bus_idx']
            for i, bus_idx in enumerate(gen_bus_indices):
                if bus_idx not in bus_connections:
                    bus_connections[bus_idx] = {'type': 'generator', 'components': []}
                bus_connections[bus_idx]['components'].append(f"Gen_{i}")
        
        # Add load connections  
        if 'load' in self.data and 'bus_idx' in self.data['load']:
            load_bus_indices = self.data['load']['bus_idx']
            for i, bus_idx in enumerate(load_bus_indices):
                if bus_idx not in bus_connections:
                    bus_connections[bus_idx] = {'type': 'load', 'components': []}
                else:
                    bus_connections[bus_idx]['type'] = 'mixed'
                bus_connections[bus_idx]['components'].append(f"Load_{i}")
        
        print(f"📊 Found {len(bus_connections)} buses with components")
        
        # Create connections between buses with components (simplified approach)
        # This is a basic heuristic - in a real system you'd need the actual network topology
        connected_buses = list(bus_connections.keys())
        
        if len(connected_buses) > 1:
            # Connect each bus to the next one in a ring topology as a basic example
            for i in range(len(connected_buses)):
                from_bus = connected_buses[i]
                to_bus = connected_buses[(i + 1) % len(connected_buses)]
                
                connections.append({
                    'from_bus': from_bus,
                    'to_bus': to_bus,
                    'connection_type': 'component_based',
                    'weight': 1.0
                })
        
        print(f"✅ Created {len(connections)} basic connections")
        return connections
    
    def extract_connections_from_edges(self):
        """Extract bus connections from edge data (if available)"""
        connections = []
        
        if 'edge' not in self.data:
            print("⚠️ No edge data available, will use Y matrix only")
            return connections
        
        edge_data = self.data['edge']
        
        # Get connection indices
        from_indices = edge_data.get('from_idx', [])
        to_indices = edge_data.get('to_idx', [])
        R_values = edge_data.get('R_ohm', [])
        X_values = edge_data.get('X_ohm', [])
        
        print(f"🔌 Processing {len(from_indices)} edges for connections...")
        
        valid_connections = 0
        for i in range(len(from_indices)):
            from_bus = int(from_indices[i])
            to_bus = int(to_indices[i])
            R = R_values[i] if i < len(R_values) else np.nan
            X = X_values[i] if i < len(X_values) else np.nan
            
            # Check if impedance is valid (following feature_extraction.py logic)
            if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
                impedance = complex(R, X)
                admittance = 1.0 / impedance if impedance != 0 else 0
                
                connections.append({
                    'from_bus': from_bus,
                    'to_bus': to_bus,
                    'R_ohm': R,
                    'X_ohm': X,
                    'impedance': impedance,
                    'admittance': admittance
                })
                valid_connections += 1
        
        print(f"✅ Found {valid_connections} valid connections from edge data")
        return connections
    
    def extract_connections_from_Y_matrix(self):
        """Extract bus connections from Y admittance matrix"""
        connections = []
        
        if self.Y_matrix is None:
            print("⚠️ No Y admittance matrix available")
            return connections
        
        print(f"🎯 Processing Y matrix {self.Y_matrix.shape} for connections...")
        
        # Convert to COO format for easier iteration
        Y_coo = self.Y_matrix.tocoo()
        
        connection_count = 0
        for i, j, admittance in zip(Y_coo.row, Y_coo.col, Y_coo.data):
            # Only consider off-diagonal elements (connections between different buses)
            if i != j and i < j:  # Avoid duplicates by only taking upper triangle
                # Calculate impedance from admittance
                impedance = 1.0 / admittance if admittance != 0 else float('inf')
                
                connections.append({
                    'from_bus': i,
                    'to_bus': j,
                    'admittance': admittance,
                    'impedance': impedance,
                    'magnitude': abs(admittance),
                    'phase': np.angle(admittance)
                })
                connection_count += 1
        
        print(f"✅ Found {connection_count} connections from Y matrix")
        return connections
    
    def construct_graph(self):
        """Construct NetworkX graph following feature_extraction.py style"""
        print_header("CONSTRUCTING GRAPH")
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Add nodes (buses) with attributes
        for i, bus_name in enumerate(self.buses):
            node_attrs = {'name': bus_name, 'index': i}
            
            # Add additional bus attributes if available
            if 'bus' in self.data:
                bus_data = self.data['bus']
                if 'Un_kV' in bus_data:
                    node_attrs['voltage_kV'] = bus_data['Un_kV'][i]
                if 'area' in bus_data:
                    node_attrs['area'] = bus_data['area'][i]
                if 'zone' in bus_data:
                    node_attrs['zone'] = bus_data['zone'][i]
            
            self.graph.add_node(i, **node_attrs)
        
        print(f"✅ Added {self.num_buses} nodes (buses) to graph")
        
        # Get connections from Y matrix (primary method)
        connections = self.extract_connections_from_Y_matrix()
        
        # If Y matrix not available, try edge data
        if not connections:
            connections = self.extract_connections_from_edges()
            
        # If still no connections, create basic topology from generator/load data
        if not connections:
            print("⚠️ No Y matrix or edge data found. Creating basic topology from components...")
            connections = self.construct_graph_from_gen_load_connections()
        
        # Add edges (connections) to graph
        edges_added = 0
        for conn in connections:
            from_bus = conn['from_bus']
            to_bus = conn['to_bus']
            
            # Ensure bus indices are valid
            if 0 <= from_bus < self.num_buses and 0 <= to_bus < self.num_buses:
                edge_attrs = {
                    'admittance': conn.get('admittance', 0),
                    'impedance': conn.get('impedance', float('inf')),
                    'weight': conn.get('magnitude', abs(conn.get('admittance', 0)))
                }
                
                # Add impedance components if available
                if 'R_ohm' in conn:
                    edge_attrs['R_ohm'] = conn['R_ohm']
                if 'X_ohm' in conn:
                    edge_attrs['X_ohm'] = conn['X_ohm']
                
                self.graph.add_edge(from_bus, to_bus, **edge_attrs)
                edges_added += 1
        
        print(f"✅ Added {edges_added} edges (connections) to graph")
        
        # Graph statistics
        print(f"📊 Graph Statistics:")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {self.graph.number_of_edges()}")
        print(f"   Density: {nx.density(self.graph):.4f}")
        print(f"   Connected: {nx.is_connected(self.graph)}")
        
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            print(f"   Connected components: {len(components)}")
            print(f"   Largest component size: {len(max(components, key=len))}")
    
    def visualize_graph(self, save_path=None, figsize=(12, 8)):
        """Visualize the constructed graph"""
        print_header("GRAPH VISUALIZATION")
        
        if self.graph is None:
            print("❌ No graph to visualize. Run construct_graph() first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Choose layout algorithm based on graph size
        if self.graph.number_of_nodes() < 50:
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(self.graph)
        
        # Draw nodes
        node_colors = ['lightblue' if self.graph.degree(n) > 2 else 'lightcoral' 
                      for n in self.graph.nodes()]
        node_sizes = [200 + 50 * self.graph.degree(n) for n in self.graph.nodes()]
        
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, 
                              alpha=0.6, 
                              width=1.5,
                              edge_color='gray')
        
        # Draw labels (bus names or indices)
        labels = {}
        for i, node in enumerate(self.graph.nodes()):
            if i < len(self.buses):
                bus_name = self.buses[i]
                # Use short name if too long
                if len(bus_name) > 8:
                    labels[node] = f"B{node}"
                else:
                    labels[node] = bus_name
            else:
                labels[node] = f"B{node}"
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title(f"Power System Graph\n{self.num_buses} Buses, {self.graph.number_of_edges()} Connections", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Graph saved to: {save_path}")
        
        plt.show()
    
    def save_graph_data(self, output_dir="graph_output"):
        """Save graph data for further analysis"""
        print_header("SAVING GRAPH DATA")
        
        if self.graph is None:
            print("❌ No graph to save. Run construct_graph() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as GraphML (preserves attributes)
        graphml_path = os.path.join(output_dir, "power_system_graph.graphml")
        nx.write_graphml(self.graph, graphml_path)
        print(f"✅ Graph saved as GraphML: {graphml_path}")
        
        # Save as adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph)
        adj_path = os.path.join(output_dir, "adjacency_matrix.npz")
        from scipy.sparse import save_npz
        save_npz(adj_path, adj_matrix)
        print(f"✅ Adjacency matrix saved: {adj_path}")
        
        # Save node and edge lists
        nodes_df = pd.DataFrame([(n, data) for n, data in self.graph.nodes(data=True)])
        nodes_path = os.path.join(output_dir, "nodes.csv")
        nodes_df.to_csv(nodes_path, index=False)
        print(f"✅ Nodes saved: {nodes_path}")
        
        edges_df = pd.DataFrame([(u, v, data) for u, v, data in self.graph.edges(data=True)])
        edges_path = os.path.join(output_dir, "edges.csv")
        edges_df.to_csv(edges_path, index=False)
        print(f"✅ Edges saved: {edges_path}")
        
        # Save statistics
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'max_degree': max(dict(self.graph.degree()).values()),
            'min_degree': min(dict(self.graph.degree()).values())
        }
        
        stats_path = os.path.join(output_dir, "graph_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("Power System Graph Statistics\n")
            f.write("=" * 40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✅ Statistics saved: {stats_path}")
    
    def analyze_graph(self):
        """Analyze graph properties following feature_extraction.py style"""
        print_header("GRAPH ANALYSIS")
        
        if self.graph is None:
            print("❌ No graph to analyze. Run construct_graph() first.")
            return
        
        # Basic properties
        print(f"📊 Basic Properties:")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {self.graph.number_of_edges()}")
        print(f"   Density: {nx.density(self.graph):.4f}")
        print(f"   Connected: {nx.is_connected(self.graph)}")
        
        # Degree analysis
        degrees = dict(self.graph.degree())
        print(f"\n📈 Degree Analysis:")
        print(f"   Average degree: {np.mean(list(degrees.values())):.2f}")
        print(f"   Max degree: {max(degrees.values())} (Bus {max(degrees, key=degrees.get)})")
        print(f"   Min degree: {min(degrees.values())} (Bus {min(degrees, key=degrees.get)})")
        
        # Connectivity analysis
        if nx.is_connected(self.graph):
            print(f"\n🔗 Connectivity Analysis:")
            print(f"   Diameter: {nx.diameter(self.graph)}")
            print(f"   Average path length: {nx.average_shortest_path_length(self.graph):.2f}")
            print(f"   Radius: {nx.radius(self.graph)}")
        else:
            components = list(nx.connected_components(self.graph))
            print(f"\n⚠️ Graph is disconnected:")
            print(f"   Number of components: {len(components)}")
            for i, comp in enumerate(components):
                print(f"   Component {i}: {len(comp)} nodes")
        
        # Show high-degree nodes (important buses)
        high_degree_nodes = [(node, degree) for node, degree in degrees.items() if degree > np.mean(list(degrees.values()))]
        if high_degree_nodes:
            print(f"\n⭐ Important Buses (above average degree):")
            for node, degree in sorted(high_degree_nodes, key=lambda x: x[1], reverse=True)[:10]:
                bus_name = self.buses[node] if node < len(self.buses) else f"Bus_{node}"
                print(f"   {bus_name} (Bus {node}): degree {degree}")


# ── Main execution function following First_model.py style ──────────────────
def main():
    """Main function to execute graph construction"""
    print_header("H5 POWER SYSTEM GRAPH CONSTRUCTOR")
    print(f"📅 Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration (adjust path as needed)
    h5_files_to_try = [
        "enhanced_out/39_Bus_New_England_System_complete_enhanced.h5",
        # "enhanced_out/14_Bus_System_complete_enhanced.h5"  # Try static file first
    ]
    
    # Find available H5 file
    h5_file = None
    for file_path in h5_files_to_try:
        if os.path.exists(file_path):
            h5_file = file_path
            print(f"✅ Found H5 file: {file_path}")
            break
    
    if not h5_file:
        print("❌ No H5 files found. Please check the paths.")
        print("💡 Available H5 files to try:")
        for path in h5_files_to_try:
            print(f"   {path}")
        print("\n🔧 Make sure you have run the feature extraction scripts first:")
        print("   - Feature_Extraction.py (creates static files with Y matrix)")
        print("   - Full_feature_extraction_SG.py (creates enhanced files)")
        return
    
    try:
        # Initialize constructor
        constructor = H5GraphConstructor(h5_file)
        
        # Load data from H5 file
        constructor.load_h5_data()
        
        # Construct graph
        constructor.construct_graph()
        
        # Analyze graph
        constructor.analyze_graph()
        
        # Visualize graph
        output_dir = "graph_output"
        os.makedirs(output_dir, exist_ok=True)  # Create output directory
        constructor.visualize_graph(save_path=os.path.join(output_dir, "power_system_graph.png"))
        
        # Save graph data
        constructor.save_graph_data()
        
        print_header("EXECUTION COMPLETE")
        print(f"✅ Graph construction completed successfully!")
        print(f"📁 Output saved to: graph_output/")
        print(f"🎉 Graph ready for further analysis and ML applications!")
        
        return constructor
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print(f"💡 Please check your H5 file and ensure it contains bus and admittance data.")
        return None


if __name__ == "__main__":
    # Execute main function
    graph_constructor = main()
    
    # If successful, the graph_constructor object can be used for further analysis
    if graph_constructor and graph_constructor.graph:
        print(f"\n💡 You can now use the graph_constructor object for further analysis:")
        print(f"   - graph_constructor.graph: NetworkX graph object")
        print(f"   - graph_constructor.Y_matrix: Admittance matrix")
        print(f"   - graph_constructor.data: All loaded H5 data")