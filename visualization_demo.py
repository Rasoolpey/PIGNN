"""
Clean visualization demo for three-phase power grid graph.
Creates a single high-quality 3D isometric PDF visualization.
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder
from visualization.graph_plotter import ThreePhaseGraphPlotter


def main():
    """Main visualization demo workflow"""
    
    # Path to your H5 file
    h5_path = "./data/scenario_0.h5"
    
    print("🔄 Three-Phase Power Grid Visualization Demo")
    print("=" * 50)
    
    # 1. Load data from H5 file
    print("\n📊 Loading IEEE39 system data...")
    loader = H5DataLoader(h5_path)
    data = loader.load_all_data()
    
    print(f"   ✓ Buses: {len(data['buses']['names'])}")
    print(f"   ✓ Lines: {len(data['lines']['names'])}")
    print(f"   ✓ Generators: {len(data['generators']['names'])}")
    
    # 2. Build three-phase graph
    print("\n🔗 Building three-phase graph...")
    builder = GraphBuilder()
    graph = builder.build_from_h5_data(data)
    
    print(f"   ✓ Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # 3. Create visualization
    print("\n🎨 Creating 3D isometric visualization...")
    plotter = ThreePhaseGraphPlotter(graph)
    
    # Create the spectral 3D graph with isometric view
    fig, ax = plotter.plot_spectral_3d_graph(
        coupling_color_nodes=True,
        coupling_color_edges=True,
        show_node_labels=False,
        view_elevation=35,  # Isometric view
        view_azimuth=-45
    )
    
    # Enhance the title
    ax.set_title('Three-Phase Power Grid System\n'
                'Red=Phase A, Green=Phase B, Blue=Phase C', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Create output directory
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    
    # Save as PDF to preserve vector graphics
    pdf_path = output_dir / "power_grid_3d_visualization.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"   ✓ Saved high-quality PDF: {pdf_path}")
    
    # Close figure
    plt.close(fig)
    
    print("\n✅ Visualization complete!")
    print(f"📁 Output: {pdf_path}")
    print("🔧 To adjust viewing angle, modify view_elevation/view_azimuth parameters")


if __name__ == "__main__":
    main()