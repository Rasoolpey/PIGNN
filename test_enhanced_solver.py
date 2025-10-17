"""
Test Enhanced Load Flow Solver

This script tests the enhanced load flow solver with better convergence.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder
from enhanced_load_flow_solver import EnhancedLoadFlowSolver
from fix_load_flow_solver import fix_power_injections
from core.graph_base import PhaseType


def test_enhanced_solver():
    """Test the enhanced load flow solver"""
    print("🚀 Testing Enhanced Load Flow Solver")
    print("=" * 50)
    
    # Load data
    loader = H5DataLoader('data/scenario_0.h5')
    graph_data = loader.load_all_data()
    
    # Build graph
    builder = GraphBuilder(base_mva=100.0, frequency_hz=50.0)
    graph = builder.build_from_h5_data(graph_data)
    
    # Fix power injection data
    fix_power_injections(graph)
    
    print(f"📊 System: {len(graph.nodes)} buses, {len(graph.edges)} branches")
    
    # Test enhanced solver
    solver = EnhancedLoadFlowSolver(
        graph, 
        tolerance=1e-4,  # Relaxed tolerance
        max_iterations=100,
        acceleration_factor=0.3  # Conservative
    )
    
    results = solver.solve(verbose=True)
    
    # Analyze results
    print("\\n📊 Enhanced Solver Results:")
    print(f"   Converged: {results.converged}")
    print(f"   Iterations: {results.iterations}")
    print(f"   Max mismatch: {results.max_mismatch:.2e}")
    print(f"   Voltage range: {results.voltage_magnitudes.min():.3f} - {results.voltage_magnitudes.max():.3f} pu")
    print(f"   Total losses: {results.total_losses_mw:.1f} MW")
    
    # Show voltage profile
    print("\\n📊 Voltage Profile:")
    voltages_by_bus = {}
    
    for bus_id in list(graph.nodes.keys())[:10]:  # First 10 buses
        bus_results = results.get_bus_results(bus_id)
        if 'A' in bus_results:
            v_pu = bus_results['A']['voltage_pu']
            angle = bus_results['A']['angle_deg']
            voltages_by_bus[bus_id] = (v_pu, angle)
            print(f"   {bus_id}: {v_pu:.3f} pu ∠{angle:.1f}°")
    
    # Check if results are reasonable
    reasonable = (
        results.voltage_magnitudes.min() > 0.85 and 
        results.voltage_magnitudes.max() < 1.15 and
        not np.any(np.isnan(results.voltages))
    )
    
    if results.converged and reasonable:
        print("\\n✅ Enhanced solver working - voltages are reasonable!")
        return True
    elif results.converged:
        print("\\n⚠️  Solver converged but voltages look questionable")
        return False
    else:
        print("\\n❌ Solver still not converging properly")
        return False


if __name__ == "__main__":
    import numpy as np
    success = test_enhanced_solver()
    print(f"\\nOverall result: {'✅ SUCCESS' if success else '❌ NEEDS MORE WORK'}")