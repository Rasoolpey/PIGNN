#!/usr/bin/env python3
"""
PowerFactory vs Our Solver Comparison Analysis
Shows the difference between PowerFactory results and our load flow solver
"""

import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from final_load_flow_solution import create_powerfactory_based_results
import h5py
import numpy as np
import matplotlib.pyplot as plt

def compare_powerfactory_vs_our_solver(scenario_id=2):
    """Compare PowerFactory vs our solver for a specific scenario"""
    
    print('🔍 POWERFACTORY vs OUR SOLVER - DETAILED COMPARISON')
    print('=' * 65)
    
    h5_path = f'Contingency Analysis/contingency_scenarios/scenario_{scenario_id}.h5'
    
    print(f'📋 Scenario {scenario_id}: Line 05-06 Outage Contingency')
    print()
    
    # 1. PowerFactory Results (Reference/Ground Truth)
    print('🏭 POWERFACTORY RESULTS (Reference - Ground Truth):')
    print('   This is the correct answer from professional software')
    
    with h5py.File(h5_path, 'r') as f:
        # Get contingency info
        if 'disconnection_actions' in f:
            actions = f['disconnection_actions/actions'][:]
            print(f'   🔌 Contingency: {actions[0].decode()}')
        
        # Bus voltage results
        pf_bus_names = [name.decode() for name in f['load_flow_results/bus_data/bus_names'][:]]
        pf_voltages = f['load_flow_results/bus_data/bus_voltages_pu'][:]
        pf_angles = f['load_flow_results/bus_data/bus_angles_deg'][:]
        
        # System totals
        pf_gen = float(f['power_flow_data/system_totals/total_generation_MW'][()])
        pf_load = float(f['power_flow_data/system_totals/total_load_MW'][()])
        pf_losses = float(f['power_flow_data/system_totals/total_losses_MW'][()])
        
        print(f'   📊 Bus Voltages: {pf_voltages.min():.3f} to {pf_voltages.max():.3f} pu')
        print(f'   ⚡ Generation: {pf_gen:.1f} MW')
        print(f'   🔌 Load Served: {pf_load:.1f} MW')
        print(f'   📉 System Losses: {pf_losses:.1f} MW')
        print(f'   🏛️  Total Buses: {len(pf_bus_names)}')
    
    # 2. Our Solver Results  
    print()
    print('🔧 OUR LOAD FLOW SOLVER RESULTS:')
    print('   This is what our Python implementation calculates')
    
    results = create_powerfactory_based_results(h5_path)
    
    print(f'   📊 Bus Voltages: {results.voltage_magnitudes.min():.3f} to {results.voltage_magnitudes.max():.3f} pu')
    print(f'   ⚡ Generation: {results.total_generation_mw:.1f} MW')
    print(f'   🔌 Load Served: {results.total_load_mw:.1f} MW')
    print(f'   📉 System Losses: {results.total_losses_mw:.1f} MW')
    print(f'   ✅ Converged: {results.converged}')
    print(f'   🔢 Total Nodes: {len(results.voltage_magnitudes)} (3-phase expansion)')
    
    # 3. Key Differences Explained
    print()
    print('🤔 WHAT ARE THESE TWO RESULTS?')
    print('=' * 40)
    print('🏭 PowerFactory Results:')
    print('   • Professional power system analysis software')
    print('   • Uses advanced numerical methods')
    print('   • Single-phase positive sequence model (39 buses)')
    print('   • Industry standard for utilities')
    print('   • Handles complex power system phenomena')
    print()
    print('🔧 Our Solver Results:') 
    print('   • Python-based load flow implementation')
    print('   • Currently uses PowerFactory data as reference')
    print('   • Expanded to 3-phase model (117 nodes = 39 buses × 3 phases)')
    print('   • Academic/research implementation')
    print('   • Built for graph neural network integration')
    
    # 4. Accuracy Analysis
    print()
    print('📊 ACCURACY COMPARISON:')
    print('=' * 30)
    
    # Compare single-phase equivalent (every 3rd node represents phase A)
    our_single_phase = results.voltage_magnitudes[::3]  # Take every 3rd (phase A)
    voltage_errors = np.abs(our_single_phase - pf_voltages)
    
    print(f'   Maximum voltage error: {voltage_errors.max():.6f} pu')
    print(f'   Average voltage error: {voltage_errors.mean():.6f} pu')
    print(f'   RMS voltage error: {np.sqrt(np.mean(voltage_errors**2)):.6f} pu')
    
    if voltage_errors.max() < 0.001:
        accuracy = '🟢 EXCELLENT (< 0.1%)'
    elif voltage_errors.max() < 0.01:
        accuracy = '🟡 GOOD (< 1%)'
    else:
        accuracy = '🔴 NEEDS IMPROVEMENT'
    
    print(f'   Accuracy Assessment: {accuracy}')
    
    # Power balance check
    gen_error = abs(results.total_generation_mw - pf_gen) / pf_gen * 100
    load_error = abs(results.total_load_mw - pf_load) / pf_load * 100  
    loss_error = abs(results.total_losses_mw - pf_losses) / pf_losses * 100
    
    print(f'   Generation error: {gen_error:.3f}%')
    print(f'   Load error: {load_error:.3f}%')
    print(f'   Loss error: {loss_error:.3f}%')
    
    # 5. Show critical buses comparison
    print()
    print('🏛️  CRITICAL BUSES COMPARISON:')
    print('   Bus Name    | PowerFactory | Our Solver | Error')
    print('   ------------|--------------|------------|--------')
    
    for i in range(min(10, len(pf_bus_names))):
        bus_name = pf_bus_names[i]
        pf_v = pf_voltages[i]
        our_v = our_single_phase[i]
        error = abs(our_v - pf_v)
        
        status = '✅' if error < 0.001 else '⚠️' if error < 0.01 else '❌'
        print(f'   {bus_name:<11} | {pf_v:11.4f} | {our_v:9.4f} | {error:6.4f} {status}')
    
    # 6. THE TRUTH ABOUT OUR CURRENT SOLVER
    print()
    print('🎯 THE IMPORTANT TRUTH:')
    print('=' * 25)
    print('   Currently, our solver uses PowerFactory results as input!')
    print('   This means:')
    print('   ✅ Perfect accuracy (errors are only from 3-phase expansion)')
    print('   ✅ No convergence issues') 
    print('   ⚠️  Not yet a true independent load flow solver')
    print('   🎯 Next step: Replace with actual Newton-Raphson solver')
    
    return {
        'powerfactory': {'voltages': pf_voltages, 'gen': pf_gen, 'load': pf_load, 'losses': pf_losses},
        'our_solver': {'voltages': our_single_phase, 'gen': results.total_generation_mw, 
                      'load': results.total_load_mw, 'losses': results.total_losses_mw},
        'errors': {'voltage_errors': voltage_errors, 'max_error': voltage_errors.max()}
    }

if __name__ == '__main__':
    comparison_results = compare_powerfactory_vs_our_solver()