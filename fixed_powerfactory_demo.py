"""
Fixed PowerFactory Comparison Demo

This demonstrates the comparison system by using actual PowerFactory data
as the "solver" results to show perfect agreement, then shows how to integrate
your real solver results.
"""

import sys
from pathlib import Path
import numpy as np
import h5py

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from visualization.powerfactory_comparison import PowerFactoryComparator


def create_perfect_agreement_demo():
    """Show perfect agreement by using PowerFactory data as 'solver' results."""
    print("🎨 Fixed PowerFactory Comparison Demo")
    print("=" * 50)
    print("Step 1: Showing PERFECT agreement using PowerFactory data as 'solver'")
    print("Step 2: Then showing how to integrate your actual solver")
    print()
    
    # Test with a scenario that converged in PowerFactory
    scenario_id = 2  # Line outage: Line 05 - 06 (this one converged)
    pf_h5_path = f"Contingency Analysis/contingency_scenarios/scenario_{scenario_id}.h5"
    
    if not Path(pf_h5_path).exists():
        print(f"❌ PowerFactory file not found: {pf_h5_path}")
        return
    
    print(f"📊 Using PowerFactory data from: {pf_h5_path}")
    
    # Load PowerFactory data directly
    with h5py.File(pf_h5_path, 'r') as f:
        # Get bus data
        bus_group = f['detailed_system_data/buses']
        bus_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                    for name in bus_group['names'][:]]
        
        pf_voltages_pu = np.array(bus_group['voltages_pu'][:])
        pf_angles_rad = np.array(bus_group['voltage_angles_deg'][:]) * np.pi / 180.0
        pf_active_inj = np.array(bus_group['active_injection_MW'][:])
        pf_reactive_inj = np.array(bus_group['reactive_injection_MVAR'][:])
        
        # Get generator data
        gen_group = f['detailed_system_data/generators']
        pf_gen_active = np.array(gen_group['active_power_MW'][:])
        pf_gen_reactive = np.array(gen_group['reactive_power_MVAR'][:])
        
        print(f"   📊 Loaded {len(bus_names)} buses, {len(pf_gen_active)} generators")
        print(f"   📊 Voltage range: {pf_voltages_pu.min():.3f} - {pf_voltages_pu.max():.3f} pu")
        print(f"   📊 Generation range: {pf_gen_active.min():.1f} - {pf_gen_active.max():.1f} MW")
    
    # STEP 1: Perfect Agreement Demo
    print("\\n🎯 STEP 1: Perfect Agreement Demo")
    print("Using PowerFactory data as both 'solver' and reference...")
    
    # Use PowerFactory data as "solver results" (this should show perfect agreement)
    perfect_solver_results = {
        'voltages': {
            'magnitude': pf_voltages_pu,
            'angle': pf_angles_rad
        },
        'line_flows': {
            'active_power': pf_active_inj,
            'reactive_power': pf_reactive_inj  
        },
        'generators': {
            'active_power': pf_gen_active,
            'reactive_power': pf_gen_reactive
        }
    }
    
    comparator = PowerFactoryComparator(figure_size=(16, 12))
    
    scenario_info = {
        'scenario_id': scenario_id,
        'description': 'Perfect Agreement Demo (PF data as solver)',
        'solver_status': 'Perfect Match Test'
    }
    
    # Create comparison with perfect agreement
    figures_perfect = comparator.create_comprehensive_comparison(
        solver_results=perfect_solver_results,
        powerfactory_h5_path=pf_h5_path,
        scenario_info=scenario_info,
        save_plots=True
    )
    
    if figures_perfect:
        print("   ✅ Perfect agreement plots created!")
        print("   📊 All error bars should be nearly zero")
    
    # STEP 2: Realistic Solver Demo  
    print("\\n🎯 STEP 2: Realistic Solver Simulation")
    print("Adding small errors to simulate realistic solver results...")
    
    # Add small realistic errors to PowerFactory data to simulate solver results
    realistic_solver_results = {
        'voltages': {
            'magnitude': pf_voltages_pu + 0.001 * np.random.randn(len(pf_voltages_pu)),  # ±0.1% error
            'angle': pf_angles_rad + 0.002 * np.random.randn(len(pf_angles_rad))        # ±0.1° error
        },
        'line_flows': {
            'active_power': pf_active_inj + 0.5 * np.random.randn(len(pf_active_inj)),    # ±0.5 MW error
            'reactive_power': pf_reactive_inj + 0.2 * np.random.randn(len(pf_reactive_inj)) # ±0.2 MVAR error
        },
        'generators': {
            'active_power': pf_gen_active + 0.5 * np.random.randn(len(pf_gen_active)),      # ±0.5 MW error  
            'reactive_power': pf_gen_reactive + 0.2 * np.random.randn(len(pf_gen_reactive)) # ±0.2 MVAR error
        }
    }
    
    scenario_info_realistic = {
        'scenario_id': scenario_id,
        'description': 'Realistic Solver Demo (small errors added)',
        'solver_status': 'Converged with small errors'
    }
    
    # Create comparison with realistic errors
    figures_realistic = comparator.create_comprehensive_comparison(
        solver_results=realistic_solver_results,
        powerfactory_h5_path=pf_h5_path,
        scenario_info=scenario_info_realistic,
        save_plots=True
    )
    
    if figures_realistic:
        print("   ✅ Realistic comparison plots created!")
        print("   📊 Error bars should show small, realistic differences")
    
    # STEP 3: Show how to integrate your actual solver
    print("\\n🔧 STEP 3: Integration Guide for Your Solver")
    print("Replace the sample data with your actual solver output:")
    print("""
    # Instead of sample data, use your solver:
    from contingency_analysis_system import ContingencyAnalyzer
    
    analyzer = ContingencyAnalyzer(...)
    scenario = analyzer.contingency_scenarios[scenario_id]
    
    # Apply contingency
    modified_data = analyzer.apply_contingency(analyzer.base_data, scenario)
    
    # Run YOUR solver (once it converges)
    graph = analyzer.graph_builder.build_from_h5_data(modified_data)  
    solver = ThreePhaseLoadFlowSolver(graph)
    your_solver_results = solver.solve()
    
    # Then create comparison
    figures = comparator.create_comprehensive_comparison(
        solver_results=your_solver_results,  # <- Your actual results here
        powerfactory_h5_path=pf_h5_path,
        scenario_info=scenario_info
    )
    """)
    
    return figures_perfect, figures_realistic


def demonstrate_error_levels():
    """Show what different error levels look like in the plots."""
    print("\\n📊 Error Level Demonstration")
    print("=" * 40)
    
    # Load PowerFactory reference data
    scenario_id = 2
    pf_h5_path = f"Contingency Analysis/contingency_scenarios/scenario_{scenario_id}.h5"
    
    with h5py.File(pf_h5_path, 'r') as f:
        bus_group = f['detailed_system_data/buses']
        pf_voltages = np.array(bus_group['voltages_pu'][:])
        pf_angles = np.array(bus_group['voltage_angles_deg'][:]) * np.pi / 180.0
    
    comparator = PowerFactoryComparator(figure_size=(14, 10))
    
    # Test different error levels
    error_scenarios = [
        {'name': 'Excellent Solver', 'v_err': 0.0005, 'a_err': 0.001, 'desc': '0.05% voltage, 0.06° angle'},
        {'name': 'Good Solver', 'v_err': 0.002, 'a_err': 0.005, 'desc': '0.2% voltage, 0.3° angle'}, 
        {'name': 'Acceptable Solver', 'v_err': 0.01, 'a_err': 0.02, 'desc': '1% voltage, 1.1° angle'},
        {'name': 'Poor Solver', 'v_err': 0.05, 'a_err': 0.1, 'desc': '5% voltage, 5.7° angle'}
    ]
    
    for i, err_config in enumerate(error_scenarios):
        print(f"\\n{i+1}. {err_config['name']} ({err_config['desc']}):")
        
        # Add errors
        solver_results = {
            'voltages': {
                'magnitude': pf_voltages + err_config['v_err'] * np.random.randn(len(pf_voltages)),
                'angle': pf_angles + err_config['a_err'] * np.random.randn(len(pf_angles))
            },
            'line_flows': {'active_power': np.zeros(39), 'reactive_power': np.zeros(39)},
            'generators': {'active_power': np.zeros(10), 'reactive_power': np.zeros(10)}
        }
        
        scenario_info = {
            'scenario_id': scenario_id + 10 + i,  # Unique IDs
            'description': f'{err_config["name"]} - {err_config["desc"]}',
            'solver_status': 'Error level demo'
        }
        
        figures = comparator.create_comprehensive_comparison(
            solver_results=solver_results,
            powerfactory_h5_path=pf_h5_path, 
            scenario_info=scenario_info,
            save_plots=True
        )
        
        if figures:
            # Calculate actual errors for validation
            v_err_actual = np.mean(np.abs(solver_results['voltages']['magnitude'] - pf_voltages))
            a_err_actual = np.mean(np.abs(solver_results['voltages']['angle'] - pf_angles)) * 180/np.pi
            print(f"   📊 Actual errors: {v_err_actual:.4f} pu voltage, {a_err_actual:.3f}° angle")
            print(f"   ✅ Plots saved with '{err_config['name']}' in filename")


if __name__ == "__main__":
    print("🎨 Fixed PowerFactory Comparison System")
    print("=" * 60)
    
    # Main demonstration
    perfect_figs, realistic_figs = create_perfect_agreement_demo()
    
    # Error level examples
    demonstrate_error_levels()
    
    print("\\n🎉 Fixed Demo Complete!")
    print("\\n📁 Check the comparison_plots folder for:")
    print("   • Perfect agreement plots (should show ~zero errors)")  
    print("   • Realistic solver plots (should show small errors)")
    print("   • Different error level examples")
    print("\\n🔧 Now you can see what good vs poor solver results look like!")