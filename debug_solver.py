"""
Load Flow Solver Debugging Tool

This script helps diagnose why your three-phase load flow solver
is failing with Jacobian singularity issues.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from contingency_analysis_system import ContingencyAnalyzer
from physics.load_flow_solver import ThreePhaseLoadFlowSolver


def debug_solver_failure(scenario_id=0):
    """Debug why the load flow solver is failing."""
    print("🔧 Load Flow Solver Debugging")
    print("=" * 50)
    
    try:
        # Initialize system
        analyzer = ContingencyAnalyzer(
            base_scenario_file="Contingency Analysis/contingency_scenarios/scenario_1.h5",
            contingency_csv_file="Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv",
            digsilent_scenarios_dir="Contingency Analysis/contingency_scenarios"
        )
        
        scenario = analyzer.contingency_scenarios[scenario_id]
        print(f"📋 Debugging scenario {scenario_id}: {scenario.description}")
        
        # Apply contingency
        modified_data = analyzer.apply_contingency(analyzer.base_data, scenario)
        
        # Build graph
        graph = analyzer.graph_builder.build_from_h5_data(modified_data)
        
        # Detailed graph analysis
        print("\\n🌐 Network Analysis:")
        print(f"   Total buses: {len(graph.nodes)}")
        
        # Check bus types
        pq_buses = []
        pv_buses = []
        slack_buses = []
        
        for bus_name, bus_data in graph.nodes.items():
            if hasattr(bus_data, 'bus_type'):
                if bus_data.bus_type == 'PQ':
                    pq_buses.append(bus_name)
                elif bus_data.bus_type == 'PV':
                    pv_buses.append(bus_name)
                elif bus_data.bus_type == 'Slack':
                    slack_buses.append(bus_name)
        
        print(f"   PQ buses: {len(pq_buses)}")
        print(f"   PV buses: {len(pv_buses)}")  
        print(f"   Slack buses: {len(slack_buses)}")
        
        if len(slack_buses) == 0:
            print("   ❌ NO SLACK BUS FOUND - This will cause convergence issues!")
        
        # Check connectivity
        print("\\n🔗 Connectivity Analysis:")
        total_edges = len(graph.edges) if hasattr(graph, 'edges') else 0
        print(f"   Total lines/transformers: {total_edges}")
        
        if total_edges == 0:
            print("   ❌ NO LINES FOUND - Network is completely disconnected!")
            return
        
        # Check loads and generation
        print("\\n⚡ Power Balance Analysis:")
        total_load_p = 0
        total_load_q = 0
        total_gen_p = 0
        total_gen_q = 0
        
        for bus_name, bus_data in graph.nodes.items():
            # Check for loads
            if hasattr(bus_data, 'loads') and bus_data.loads:
                for load in bus_data.loads:
                    if hasattr(load, 'active_power_mw'):
                        total_load_p += load.active_power_mw
                    if hasattr(load, 'reactive_power_mvar'):
                        total_load_q += load.reactive_power_mvar
            
            # Check for generators
            if hasattr(bus_data, 'generators') and bus_data.generators:
                for gen in bus_data.generators:
                    if hasattr(gen, 'active_power_mw'):
                        total_gen_p += gen.active_power_mw
                    if hasattr(gen, 'reactive_power_mvar'):
                        total_gen_q += gen.reactive_power_mvar
        
        print(f"   Total load: {total_load_p:.1f} MW, {total_load_q:.1f} MVAR")
        print(f"   Total generation: {total_gen_p:.1f} MW, {total_gen_q:.1f} MVAR")
        print(f"   Power imbalance: {total_gen_p - total_load_p:.1f} MW")
        
        if abs(total_gen_p - total_load_p) > 1000:  # Large imbalance
            print("   ⚠️  LARGE POWER IMBALANCE - May cause convergence issues")
        
        if total_gen_p == 0:
            print("   ❌ NO GENERATION FOUND - System cannot supply loads!")
        
        # Test solver initialization
        print("\\n🔧 Solver Initialization Test:")
        try:
            solver = ThreePhaseLoadFlowSolver(graph)
            print("   ✅ Solver initialized successfully")
            
            # Check initial conditions before solve
            print("   📊 Checking initial conditions...")
            
            # Try to access solver internals (if possible)
            if hasattr(solver, 'voltage_vector'):
                print(f"   Initial voltage vector size: {len(solver.voltage_vector) if solver.voltage_vector is not None else 'None'}")
            
        except Exception as e:
            print(f"   ❌ Solver initialization failed: {e}")
            return
        
        # Test a minimal solve attempt
        print("\\n🚀 Minimal Solve Test:")
        try:
            # This will likely fail, but we want to see HOW it fails
            result = solver.solve()
            
            print(f"   Convergence status: {result.converged}")
            print(f"   Iterations completed: {result.iterations}")
            print(f"   Max mismatch: {result.max_mismatch}")
            
            if hasattr(result, 'voltage_magnitudes') and result.voltage_magnitudes is not None:
                v_mag = result.voltage_magnitudes
                print(f"   Voltage range: {np.min(v_mag):.3f} - {np.max(v_mag):.3f} pu")
                print(f"   Voltage std dev: {np.std(v_mag):.6f} pu")
                
                if np.std(v_mag) < 1e-6:
                    print("   ❌ FLAT VOLTAGE PROFILE - Solver not working properly!")
            
        except Exception as e:
            print(f"   ❌ Solve failed with: {e}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


def test_base_case_vs_contingency():
    """Compare base case vs contingency to see if contingency causes the issue."""
    print("\\n🆚 Base Case vs Contingency Comparison")
    print("=" * 50)
    
    try:
        analyzer = ContingencyAnalyzer(
            base_scenario_file="Contingency Analysis/contingency_scenarios/scenario_1.h5",
            contingency_csv_file="Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv",
            digsilent_scenarios_dir="Contingency Analysis/contingency_scenarios"
        )
        
        print("🔵 Testing BASE CASE (no contingency):")
        debug_solver_failure(scenario_id=0)  # Base case
        
        print("\\n🔴 Testing CONTINGENCY CASE:")
        debug_solver_failure(scenario_id=2)  # Line outage
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def check_input_data_quality():
    """Check if the input data from H5 files is reasonable."""
    print("\\n📊 Input Data Quality Check")
    print("=" * 40)
    
    try:
        analyzer = ContingencyAnalyzer(
            base_scenario_file="Contingency Analysis/contingency_scenarios/scenario_1.h5",
            contingency_csv_file="Contingency Analysis/contingency_out/contingency_scenarios_20250803_114018.csv",
            digsilent_scenarios_dir="Contingency Analysis/contingency_scenarios"
        )
        
        base_data = analyzer.base_data
        
        print("🏗️ Raw H5 Data Analysis:")
        
        # Check bus data
        if 'buses' in base_data:
            buses = base_data['buses']
            print(f"   Bus count: {len(buses.get('names', []))}")
            
            if 'voltages_pu' in buses:
                v_pu = buses['voltages_pu']
                print(f"   Bus voltage range: {np.min(v_pu):.3f} - {np.max(v_pu):.3f} pu")
        
        # Check generator data
        if 'generators' in base_data:
            gens = base_data['generators']
            if 'active_power_MW' in gens:
                gen_p = gens['active_power_MW']
                print(f"   Generator power range: {np.min(gen_p):.1f} - {np.max(gen_p):.1f} MW")
                print(f"   Total generation: {np.sum(gen_p):.1f} MW")
        
        # Check load data
        if 'loads' in base_data:
            loads = base_data['loads']
            if 'active_power_MW' in loads:
                load_p = loads['active_power_MW']
                print(f"   Load power range: {np.min(load_p):.1f} - {np.max(load_p):.1f} MW")
                print(f"   Total load: {np.sum(load_p):.1f} MW")
        
        # Check line data
        if 'lines' in base_data:
            lines = base_data['lines']
            print(f"   Line count: {len(lines.get('names', []))}")
            
            if 'in_service' in lines:
                in_service = lines['in_service']
                active_lines = np.sum(in_service)
                print(f"   Lines in service: {active_lines}/{len(in_service)}")
                
                if active_lines < len(in_service) * 0.5:
                    print("   ⚠️  Many lines out of service - may cause islands!")
    
    except Exception as e:
        print(f"❌ Data quality check failed: {e}")


if __name__ == "__main__":
    print("🔧 Load Flow Solver Diagnostic Tool")
    print("=" * 60)
    
    # Run comprehensive diagnostics
    check_input_data_quality()
    debug_solver_failure(scenario_id=0)  # Base case first
    test_base_case_vs_contingency()
    
    print("\\n🎯 Common Causes of Jacobian Singularity:")
    print("1. ❌ No slack bus assigned")
    print("2. ❌ Network islands (disconnected parts)")  
    print("3. ❌ All generators at same bus")
    print("4. ❌ Zero impedance lines")
    print("5. ❌ Numerical precision issues")
    print("6. ❌ Bad initial conditions")
    
    print("\\n🔧 Recommended Fixes:")
    print("1. ✅ Ensure at least one slack bus")
    print("2. ✅ Check network connectivity after contingencies")
    print("3. ✅ Validate impedance matrices") 
    print("4. ✅ Use better initial voltage estimates")
    print("5. ✅ Add numerical conditioning")