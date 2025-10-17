"""
PowerFactory-Referenced Load Flow Solver

This solver uses PowerFactory results as reference and starting point
to debug what's wrong with the Newton-Raphson convergence.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
from typing import Dict, Tuple

sys.path.append(str(Path(__file__).parent))

from data.h5_loader import H5DataLoader
from data.graph_builder import GraphBuilder
from physics.load_flow_solver import ThreePhaseLoadFlowSolver
from fix_load_flow_solver import fix_power_injections
from core.graph_base import PhaseType


def load_powerfactory_reference(h5_path: str) -> Dict:
    """Load PowerFactory reference solution"""
    with h5py.File(h5_path, 'r') as f:
        # Load bus results
        bus_group = f['detailed_system_data/buses']
        
        bus_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                    for name in bus_group['names'][:]]
        pf_voltages = np.array(bus_group['voltages_pu'][:])
        pf_angles = np.array(bus_group['voltage_angles_deg'][:])
        pf_p_inj = np.array(bus_group['active_injection_MW'][:])
        pf_q_inj = np.array(bus_group['reactive_injection_MVAR'][:])
        
        # Load total system data
        totals_group = f['detailed_system_data/totals']
        total_gen_p = np.array(totals_group['total_generation_P_MW'][:])
        total_gen_q = np.array(totals_group['total_generation_Q_MVAR'][:])
        total_load_p = np.array(totals_group['total_load_P_MW'][:])
        total_load_q = np.array(totals_group['total_load_Q_MVAR'][:])
        
        return {
            'bus_names': bus_names,
            'voltages_pu': pf_voltages,
            'angles_deg': pf_angles,
            'p_injections_mw': pf_p_inj,
            'q_injections_mvar': pf_q_inj,
            'total_gen_p': total_gen_p,
            'total_gen_q': total_gen_q,
            'total_load_p': total_load_p,
            'total_load_q': total_load_q
        }


def compare_power_balance(graph_data: Dict, pf_ref: Dict, base_mva: float = 100.0):
    """Compare power balance between graph and PowerFactory"""
    print("🔍 Power Balance Analysis")
    print("-" * 40)
    
    # Calculate total power from graph data
    gen_data = graph_data.get('generators', {})
    load_data = graph_data.get('loads', {})
    
    if 'active_power_MW' in gen_data:
        total_gen_p = np.sum(gen_data['active_power_MW'])
        total_gen_q = np.sum(gen_data['reactive_power_MVAR'])
    else:
        total_gen_p = 0.0
        total_gen_q = 0.0
    
    if 'active_power_MW' in load_data:
        total_load_p = np.sum(load_data['active_power_MW'])
        total_load_q = np.sum(load_data['reactive_power_MVAR'])
    else:
        total_load_p = 0.0
        total_load_q = 0.0
    
    print(f"Graph Data:")
    print(f"  Generation: {total_gen_p:.1f} MW, {total_gen_q:.1f} MVAR")
    print(f"  Load: {total_load_p:.1f} MW, {total_load_q:.1f} MVAR")
    print(f"  Net: {total_gen_p - total_load_p:.1f} MW, {total_gen_q - total_load_q:.1f} MVAR")
    
    print(f"\\nPowerFactory Reference:")
    print(f"  Generation: {pf_ref['total_gen_p'][0]:.1f} MW, {pf_ref['total_gen_q'][0]:.1f} MVAR")
    print(f"  Load: {pf_ref['total_load_p'][0]:.1f} MW, {pf_ref['total_load_q'][0]:.1f} MVAR")
    net_p_pf = pf_ref['total_gen_p'][0] - pf_ref['total_load_p'][0]
    net_q_pf = pf_ref['total_gen_q'][0] - pf_ref['total_load_q'][0]
    print(f"  Net: {net_p_pf:.1f} MW, {net_q_pf:.1f} MVAR")


def set_powerfactory_injections(graph, pf_ref: Dict, base_mva: float = 100.0):
    """Set PowerFactory power injections directly in graph nodes"""
    print("\\n🔄 Setting PowerFactory Power Injections...")
    
    # Create bus name mapping
    bus_name_map = {}
    for i, bus_name in enumerate(pf_ref['bus_names']):
        bus_name_map[bus_name] = i
    
    buses_set = 0
    for bus_id, phases in graph.nodes.items():
        if bus_id in bus_name_map:
            pf_idx = bus_name_map[bus_id]
            
            # Get PowerFactory injections (MW/MVAR)
            p_inj_mw = pf_ref['p_injections_mw'][pf_idx]
            q_inj_mvar = pf_ref['q_injections_mvar'][pf_idx]
            
            # Convert to per unit
            p_inj_pu = p_inj_mw / base_mva
            q_inj_pu = q_inj_mvar / base_mva
            
            # Set for each phase (distributed equally)
            for phase in PhaseType:
                node = phases[phase]
                node.properties['P_injection_pu'] = p_inj_pu / 3.0
                node.properties['Q_injection_pu'] = q_inj_pu / 3.0
            
            buses_set += 1
    
    print(f"   ✅ Set injections for {buses_set} buses from PowerFactory data")


def test_with_powerfactory_reference():
    """Test solver using PowerFactory data as reference"""
    print("🔬 PowerFactory-Referenced Load Flow Test")
    print("=" * 50)
    
    # Load PowerFactory reference
    pf_ref = load_powerfactory_reference('data/scenario_0.h5')
    print(f"📊 PowerFactory reference: {len(pf_ref['bus_names'])} buses")
    print(f"   Voltage range: {pf_ref['voltages_pu'].min():.3f} - {pf_ref['voltages_pu'].max():.3f} pu")
    
    # Load graph data
    loader = H5DataLoader('data/scenario_0.h5')
    graph_data = loader.load_all_data()
    
    # Compare power balance
    compare_power_balance(graph_data, pf_ref)
    
    # Build graph
    builder = GraphBuilder(base_mva=100.0, frequency_hz=50.0)
    graph = builder.build_from_h5_data(graph_data)
    
    # Set PowerFactory power injections
    set_powerfactory_injections(graph, pf_ref)
    
    # Initialize solver with PowerFactory voltages as starting point
    solver = ThreePhaseLoadFlowSolver(graph, tolerance=1e-5, max_iterations=10)
    
    # Set initial voltages from PowerFactory
    print("\\n🎯 Initializing with PowerFactory voltages...")
    V_init = solver._initialize_voltages()
    
    # Update initial voltages with PowerFactory data
    bus_name_map = {}
    for i, bus_name in enumerate(pf_ref['bus_names']):
        bus_name_map[bus_name] = i
    
    for bus_id in graph.nodes.keys():
        if bus_id in bus_name_map:
            pf_idx = bus_name_map[bus_id]
            pf_v_pu = pf_ref['voltages_pu'][pf_idx]
            pf_angle_rad = np.deg2rad(pf_ref['angles_deg'][pf_idx])
            pf_voltage = pf_v_pu * np.exp(1j * pf_angle_rad)
            
            # Set for all phases
            for phase in PhaseType:
                node_id = f"{bus_id}_{phase.value}"
                if node_id in solver.y_builder.node_to_index:
                    idx = solver.y_builder.node_to_index[node_id]
                    if idx < len(V_init):
                        V_init[idx] = pf_voltage
    
    # Test power injection calculation with PowerFactory voltages
    print("\\n📊 Testing power calculations with PowerFactory voltages...")
    P_calc, Q_calc = solver._calculate_power_injections(V_init)
    P_spec, Q_spec = solver._get_specified_powers()
    delta_P, delta_Q = solver._calculate_mismatches(P_calc, Q_calc, P_spec, Q_spec)
    
    max_p_mismatch = np.max(np.abs(delta_P)) * graph.base_mva
    max_q_mismatch = np.max(np.abs(delta_Q)) * graph.base_mva
    
    print(f"   Max P mismatch: {max_p_mismatch:.3f} MW")
    print(f"   Max Q mismatch: {max_q_mismatch:.3f} MVAR")
    
    if max_p_mismatch < 1.0 and max_q_mismatch < 1.0:
        print("   ✅ PowerFactory voltages are consistent with power injections!")
        print("   🔧 This confirms the solver setup is correct")
        return True
    else:
        print("   ⚠️  Large mismatches suggest modeling differences")
        
        # Show some specific mismatches
        print("\\n   🔍 Largest mismatches:")
        for node_id in graph.pq_buses[:5]:  # Check first 5 PQ buses
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                if node_phase_id in solver.y_builder.node_to_index:
                    idx = solver.y_builder.node_to_index[node_phase_id]
                    p_miss = delta_P[idx] * graph.base_mva
                    q_miss = delta_Q[idx] * graph.base_mva
                    if abs(p_miss) > 0.1 or abs(q_miss) > 0.1:
                        print(f"     {node_phase_id}: ΔP={p_miss:.2f} MW, ΔQ={q_miss:.2f} MVAR")
                        break
        
        return False


if __name__ == "__main__":
    success = test_with_powerfactory_reference()
    print(f"\\nDiagnostic result: {'✅ CONSISTENT' if success else '❌ MODELING ISSUE'}")