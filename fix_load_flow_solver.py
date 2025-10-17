"""
Fix for GraphBuilder - Add Power Injection Calculation

This script fixes the missing power injection data that causes the load flow solver
to return flat 1.0 pu voltages. The issue is that the GraphBuilder doesn't populate
the P_injection_pu and Q_injection_pu properties that the load flow solver needs.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder
from physics.load_flow_solver import ThreePhaseLoadFlowSolver
from core.graph_base import PhaseType


def fix_power_injections(graph):
    """
    Fix missing power injection data in graph nodes.
    This calculates net power injections (generation - load) for each bus.
    """
    print("🔧 Fixing power injection data...")
    
    # Calculate net power injections for each bus
    for bus_id, phases in graph.nodes.items():
        total_p_gen = 0.0  # Total generation at this bus (MW)
        total_q_gen = 0.0  # Total reactive generation (MVAR) 
        total_p_load = 0.0  # Total load at this bus (MW)
        total_q_load = 0.0  # Total reactive load (MVAR)
        
        # Sum up generation and loads at this bus
        for phase in PhaseType:
            node = phases[phase]
            
            # Count generation
            if hasattr(node, 'P_actual_MW'):
                total_p_gen += node.P_actual_MW or 0.0
            if hasattr(node, 'Q_actual_MVAR'):
                total_q_gen += node.Q_actual_MVAR or 0.0
                
            # Count loads
            if hasattr(node, 'P_MW'):
                total_p_load += node.P_MW or 0.0
            if hasattr(node, 'Q_MVAR'):
                total_q_load += node.Q_MVAR or 0.0
                
            # Also check for loads stored in properties
            if 'load_P_MW' in node.properties:
                total_p_load += node.properties['load_P_MW']
            if 'load_Q_MVAR' in node.properties:
                total_q_load += node.properties['load_Q_MVAR']
        
        # Calculate net injection (generation - load) in per unit
        net_p_mw = total_p_gen - total_p_load
        net_q_mvar = total_q_gen - total_q_load
        
        # Convert to per unit
        net_p_pu = net_p_mw / graph.base_mva
        net_q_pu = net_q_mvar / graph.base_mva
        
        # Distribute equally among phases and set injection properties
        for phase in PhaseType:
            node = phases[phase]
            node.properties['P_injection_pu'] = net_p_pu / 3.0  # Equal distribution
            node.properties['Q_injection_pu'] = net_q_pu / 3.0
            
    print(f"   ✅ Added power injection data to {len(graph.nodes)} buses")


def test_fixed_load_flow():
    """Test the load flow solver with fixed power injection data"""
    print("🧪 Testing Fixed Load Flow Solver")
    print("=" * 50)
    
    # Load data
    loader = H5DataLoader('data/scenario_0.h5')
    graph_data = loader.load_all_data()
    
    # Build graph
    builder = GraphBuilder(base_mva=100.0, frequency_hz=50.0)
    graph = builder.build_from_h5_data(graph_data)
    
    print(f"📊 System: {len(graph.nodes)} buses, {len(graph.edges)} branches")
    
    # Fix power injection data
    fix_power_injections(graph)
    
    # Verify power injections were set
    total_p_inj = 0
    total_q_inj = 0
    non_zero_buses = 0
    
    for bus_id, phases in graph.nodes.items():
        phase_a = phases[PhaseType.A]
        p_inj = phase_a.properties.get('P_injection_pu', 0.0) * 3  # Total for all phases
        q_inj = phase_a.properties.get('Q_injection_pu', 0.0) * 3
        
        if abs(p_inj) > 1e-6 or abs(q_inj) > 1e-6:
            non_zero_buses += 1
            print(f"   {bus_id}: P={p_inj*graph.base_mva:.1f} MW, Q={q_inj*graph.base_mva:.1f} MVAR")
        
        total_p_inj += p_inj
        total_q_inj += q_inj
    
    print(f"\\n📈 System totals:")
    print(f"   Total P injection: {total_p_inj * graph.base_mva:.1f} MW")
    print(f"   Total Q injection: {total_q_inj * graph.base_mva:.1f} MVAR") 
    print(f"   Buses with non-zero injections: {non_zero_buses}")
    
    # Solve load flow
    print("\\n🔄 Running Load Flow Analysis...")
    solver = ThreePhaseLoadFlowSolver(graph, tolerance=1e-6, max_iterations=50, acceleration_factor=0.8)
    
    results = solver.solve(verbose=True)
    
    # Analyze results
    print("\\n📊 Load Flow Results:")
    print(f"   Converged: {results.converged}")
    print(f"   Iterations: {results.iterations}")
    print(f"   Max mismatch: {results.max_mismatch:.2e}")
    print(f"   Voltage range: {results.voltage_magnitudes.min():.3f} - {results.voltage_magnitudes.max():.3f} pu")
    print(f"   Total losses: {results.total_losses_mw:.1f} MW")
    
    # Show some bus results
    print("\\n🏪 Sample Bus Results:")
    for i, bus_id in enumerate(list(graph.nodes.keys())[:5]):
        bus_results = results.get_bus_results(bus_id)
        if 'A' in bus_results:
            v_pu = bus_results['A']['voltage_pu']
            angle = bus_results['A']['angle_deg']
            print(f"   {bus_id}: {v_pu:.3f} pu ∠{angle:.1f}°")
    
    return results.converged and results.voltage_magnitudes.min() > 0.8


if __name__ == "__main__":
    success = test_fixed_load_flow()
    if success:
        print("\\n✅ Load flow solver is now working properly!")
    else:
        print("\\n❌ Load flow solver still has issues.")