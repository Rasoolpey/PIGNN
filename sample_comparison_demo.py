"""
Simple PowerFactory Comparison Demo

This creates sample comparison plots to demonstrate the visualization system
even when the solver has convergence issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from visualization.powerfactory_comparison import PowerFactoryComparator


def create_sample_comparison_plots():
    """Create sample comparison plots with synthetic data to show the system."""
    print("🎨 Creating Sample PowerFactory Comparison Plots")
    print("=" * 50)
    print("This demo shows what the comparison system looks like")
    print("using sample data for the IEEE 39-bus system.")
    print()
    
    # Create sample data that represents typical power system results
    n_buses = 39
    n_lines = 46
    n_gens = 10
    
    # Sample solver results (with some realistic values)
    solver_results = {
        'voltages': {
            'magnitude': 1.0 + 0.05 * np.random.randn(n_buses),  # Around 1.0 pu
            'angle': 0.1 * np.random.randn(n_buses)  # Small angles in radians
        },
        'line_flows': {
            'active_power': 50 * np.random.randn(n_lines),  # MW
            'reactive_power': 25 * np.random.randn(n_lines)  # MVAR
        },
        'generators': {
            'active_power': 100 + 50 * np.random.rand(n_gens),  # MW
            'reactive_power': 20 * np.random.randn(n_gens)  # MVAR
        }
    }
    
    # Sample PowerFactory results (with small differences to show comparison)
    pf_results = {
        'voltages': {
            'magnitude': solver_results['voltages']['magnitude'] + 0.01 * np.random.randn(n_buses),
            'angle': solver_results['voltages']['angle'] + 0.01 * np.random.randn(n_buses),
            'bus_names': [f"Bus {i+1:02d}" for i in range(n_buses)]
        },
        'line_flows': {
            'active_power': solver_results['line_flows']['active_power'] + 5 * np.random.randn(n_lines),
            'reactive_power': solver_results['line_flows']['reactive_power'] + 2 * np.random.randn(n_lines),
            'from_buses': [f"Bus {i+1:02d}" for i in range(n_lines)],
            'to_buses': [f"Bus {(i+1)%n_buses+1:02d}" for i in range(n_lines)]
        },
        'generators': {
            'active_power': solver_results['generators']['active_power'] + 2 * np.random.randn(n_gens),
            'reactive_power': solver_results['generators']['reactive_power'] + 1 * np.random.randn(n_gens)
        }
    }
    
    # Scenario information
    scenario_info = {
        'scenario_id': 2,
        'description': 'Line outage: Line 05 - 06 (Sample Data)',
        'contingency_type': 'N-1',
        'outage_type': 'line'
    }
    
    # Create the comparator
    comparator = PowerFactoryComparator(figure_size=(16, 12))
    
    print("📊 Generating comparison plots...")
    
    # Create a mock PowerFactory H5 file for demonstration
    sample_h5_path = "sample_powerfactory_results.h5"
    create_sample_h5_file(sample_h5_path, pf_results)
    
    # Create comprehensive comparison plots
    figures = comparator.create_comprehensive_comparison(
        solver_results=solver_results,
        powerfactory_h5_path=sample_h5_path,
        scenario_info=scenario_info,
        save_plots=True
    )
    
    if figures:
        print("✅ Comparison plots created successfully!")
        print(f"   📁 Plots saved to: {comparator.output_dir}")
        
        # Show what was created
        print("   📊 Generated plots:")
        for plot_type in figures.keys():
            print(f"      • {plot_type.replace('_', ' ').title()} Comparison")
        
        # Display plots
        import matplotlib.pyplot as plt
        for plot_name, fig in figures.items():
            plt.figure(fig.number)
            plt.show()
        
        print("\n📊 Plots displayed! Close the windows when done viewing.")
    else:
        print("❌ Failed to create comparison plots")
    
    # Clean up sample file
    if Path(sample_h5_path).exists():
        Path(sample_h5_path).unlink()


def create_sample_h5_file(filepath: str, pf_results: dict):
    """Create a sample H5 file with PowerFactory-like results."""
    with h5py.File(filepath, 'w') as f:
        # Create load_flow_results group
        lf_group = f.create_group('load_flow_results')
        
        # Voltage results
        volt_group = lf_group.create_group('voltages')
        volt_group.create_dataset('magnitude', data=pf_results['voltages']['magnitude'])
        volt_group.create_dataset('angle', data=pf_results['voltages']['angle'])
        volt_group.create_dataset('bus_names', 
                                data=[name.encode('utf-8') for name in pf_results['voltages']['bus_names']])
        
        # Line flow results
        flow_group = lf_group.create_group('line_flows')
        flow_group.create_dataset('active_power', data=pf_results['line_flows']['active_power'])
        flow_group.create_dataset('reactive_power', data=pf_results['line_flows']['reactive_power'])
        flow_group.create_dataset('from_buses', 
                                data=[name.encode('utf-8') for name in pf_results['line_flows']['from_buses']])
        flow_group.create_dataset('to_buses', 
                                data=[name.encode('utf-8') for name in pf_results['line_flows']['to_buses']])
        
        # Generator results
        gen_group = lf_group.create_group('generators')
        gen_group.create_dataset('active_power', data=pf_results['generators']['active_power'])
        gen_group.create_dataset('reactive_power', data=pf_results['generators']['reactive_power'])


def demo_comparison_features():
    """Show the key features of the comparison system."""
    print("\n🎯 PowerFactory Comparison System Features")
    print("=" * 50)
    
    print("📊 VOLTAGE ANALYSIS:")
    print("   • Bus voltage magnitudes (pu) - Your Solver vs PowerFactory")
    print("   • Bus voltage angles (degrees) - Your Solver vs PowerFactory") 
    print("   • Voltage magnitude errors with max/mean statistics")
    print("   • Voltage angle errors with max/mean statistics")
    
    print("\n⚡ LINE FLOW ANALYSIS:")
    print("   • Active power flows (MW) - Your Solver vs PowerFactory")
    print("   • Reactive power flows (MVAR) - Your Solver vs PowerFactory")
    print("   • Active power flow errors with statistics")
    print("   • Reactive power flow errors with statistics")
    
    print("\n🏭 GENERATION ANALYSIS:")
    print("   • Generator active power (MW) - Your Solver vs PowerFactory")
    print("   • Generator reactive power (MVAR) - Your Solver vs PowerFactory")
    print("   • Generation active power errors with statistics")
    print("   • Generation reactive power errors with statistics")
    
    print("\n📈 VISUALIZATION FEATURES:")
    print("   • Color-coded comparison (Green=Solver, Red=PowerFactory)")
    print("   • Error bar charts with statistical summaries")
    print("   • Professional plot formatting for publications")
    print("   • Automatic plot saving (PNG, 300 DPI)")
    print("   • Comprehensive error reports")
    
    print("\n🔧 CUSTOMIZATION OPTIONS:")
    print("   • Adjustable figure sizes")
    print("   • Customizable color schemes")
    print("   • Multiple output formats")
    print("   • Batch processing for multiple scenarios")


if __name__ == "__main__":
    print("🎨 PowerFactory Comparison Visualization Demo")
    print("=" * 60)
    
    # Show features
    demo_comparison_features()
    
    # Create sample plots
    print("\n" + "="*60)
    create_sample_comparison_plots()