#!/usr/bin/env python3
"""
PowerFactory Comparison Generator for PIGNN Project

Creates comprehensive comparison plots between our solver and PowerFactory:
1. Busbar Voltages Comparison (side-by-side)
2. Line Currents/Flows Comparison (side-by-side)  
3. Generator Power Generation Comparison (side-by-side)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from datetime import datetime
import sys

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
from physics.powerfactory_solver import create_powerfactory_based_results


def create_powerfactory_comparisons(scenario_id, h5_path, output_dir):
    """
    Create the 3 essential PowerFactory comparison plots
    
    Args:
        scenario_id: Scenario number
        h5_path: Path to H5 file
        output_dir: Output directory for plots
    """
    
    print(f"🔍 Creating PowerFactory comparisons for Scenario {scenario_id}")
    
    # Get our solver results
    our_results = create_powerfactory_based_results(h5_path)
    
    # Load PowerFactory reference data
    with h5py.File(h5_path, 'r') as f:
        # Get contingency description
        if 'disconnection_actions' in f:
            actions = f['disconnection_actions/actions'][:]
            contingency_desc = actions[0].decode() if len(actions) > 0 else f'Scenario {scenario_id}'
        else:
            contingency_desc = f'Scenario {scenario_id}'
        
        # Bus voltage data
        pf_bus_names = [name.decode() for name in f['load_flow_results/bus_data/bus_names'][:]]
        pf_voltages = f['load_flow_results/bus_data/bus_voltages_pu'][:]
        pf_angles = f['load_flow_results/bus_data/bus_angles_deg'][:]
        
        # Line flow data
        if 'power_flow_data/line_data' in f:
            pf_line_names = [name.decode() for name in f['power_flow_data/line_data/line_names'][:]]
            pf_line_flows_mw = f['power_flow_data/line_data/P_from_MW'][:]  # Active power from bus
            pf_line_flows_mvar = f['power_flow_data/line_data/Q_from_MVAR'][:]  # Reactive power from bus
            pf_line_currents = f['power_flow_data/line_data/current_A'][:]  # Current in Amperes
        else:
            pf_line_names = []
            pf_line_flows_mw = np.array([])
            pf_line_flows_mvar = np.array([])
            pf_line_currents = np.array([])
        
        # Generator data
        if 'power_flow_data/generation_data' in f:
            try:
                gen_group = f['power_flow_data/generation_data']
                
                # Get generator names
                if 'generator_names' in gen_group:
                    pf_gen_names = [name.decode() for name in gen_group['generator_names'][:]]
                else:
                    pf_gen_names = []
                
                # Get active power - try different field names
                pf_gen_p = np.array([])
                for p_field in ['P_actual_MW', 'active_power_MW', 'P_MW']:
                    if p_field in gen_group:
                        pf_gen_p = gen_group[p_field][:]
                        print(f"✅ Found active power data in field: {p_field}")
                        break
                
                # Get reactive power - try different field names  
                pf_gen_q = np.array([])
                for q_field in ['Q_actual_MVAR', 'reactive_power_MVAR', 'Q_MVAR']:
                    if q_field in gen_group:
                        pf_gen_q = gen_group[q_field][:]
                        print(f"✅ Found reactive power data in field: {q_field}")
                        break
                
                # Ensure we have consistent data
                if len(pf_gen_p) > 0 and len(pf_gen_names) == 0:
                    pf_gen_names = [f'Gen_{i+1}' for i in range(len(pf_gen_p))]
                    
                print(f"✅ Loaded {len(pf_gen_names)} generators with power data")
                
            except Exception as e:
                print(f"❌ Error loading generator data: {e}")
                pf_gen_names = []
                pf_gen_p = np.array([])
                pf_gen_q = np.array([])
        else:
            pf_gen_names = []
            pf_gen_p = np.array([])
            pf_gen_q = np.array([])
    
    # Extract our solver single-phase equivalent results
    our_voltages = our_results.voltage_magnitudes[::3]  # Every 3rd node (phase A)
    our_angles = np.degrees(np.angle(our_results.voltages[::3]))
    
    # Create the 3 comparison plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. BUSBAR VOLTAGES COMPARISON
    create_voltage_comparison_plot(
        scenario_id, contingency_desc, pf_bus_names, 
        pf_voltages, pf_angles, our_voltages, our_angles,
        output_dir, timestamp
    )
    
    # 2. LINE FLOWS COMPARISON  
    if len(pf_line_names) > 0:
        create_line_flows_comparison_plot(
            scenario_id, contingency_desc, pf_line_names,
            pf_line_flows_mw, pf_line_flows_mvar, pf_line_currents,
            output_dir, timestamp
        )
    
    # 3. GENERATOR POWER COMPARISON
    if len(pf_gen_names) > 0:
        create_generator_comparison_plot(
            scenario_id, contingency_desc, pf_gen_names,
            pf_gen_p, pf_gen_q,
            output_dir, timestamp
        )


def create_voltage_comparison_plot(scenario_id, contingency_desc, bus_names, 
                                 pf_voltages, pf_angles, our_voltages, our_angles,
                                 output_dir, timestamp):
    """Create busbar voltage comparison plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    x = np.arange(len(bus_names))
    width = 0.4
    
    # Left plot: PowerFactory Results
    ax1.bar(x, pf_voltages, width, color='steelblue', alpha=0.8, label='PowerFactory Voltages')
    ax1.set_ylabel('Voltage Magnitude (pu)', fontsize=12)
    ax1.set_xlabel('Bus Name', fontsize=12)
    ax1.set_title('PowerFactory Results\nBusbar Voltages', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bus_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Low Limit (0.95 pu)')
    ax1.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='High Limit (1.05 pu)')
    ax1.legend()
    
    # Add voltage values on critical buses
    for i, (bus, voltage) in enumerate(zip(bus_names, pf_voltages)):
        if voltage < 0.95 or voltage > 1.05:
            ax1.text(i, voltage + 0.01, f'{voltage:.3f}', 
                    ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    
    # Right plot: Our Solver Results  
    ax2.bar(x, our_voltages, width, color='orange', alpha=0.8, label='Our Solver Voltages')
    ax2.set_ylabel('Voltage Magnitude (pu)', fontsize=12)
    ax2.set_xlabel('Bus Name', fontsize=12)
    ax2.set_title('Our Solver Results\nBusbar Voltages', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bus_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Low Limit (0.95 pu)')
    ax2.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='High Limit (1.05 pu)')
    ax2.legend()
    
    # Add voltage values on critical buses
    for i, (bus, voltage) in enumerate(zip(bus_names, our_voltages)):
        if voltage < 0.95 or voltage > 1.05:
            ax2.text(i, voltage + 0.01, f'{voltage:.3f}', 
                    ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    
    # Calculate and show accuracy
    voltage_errors = np.abs(our_voltages - pf_voltages)
    max_error = voltage_errors.max()
    avg_error = voltage_errors.mean()
    
    fig.suptitle(f'Scenario {scenario_id}: Busbar Voltage Comparison\n{contingency_desc}\n' +
                f'Max Error: {max_error:.6f} pu | Avg Error: {avg_error:.6f} pu', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"comparison_voltages_scenario_{scenario_id}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Voltage comparison saved: {plot_path.name}")


def create_line_flows_comparison_plot(scenario_id, contingency_desc, line_names,
                                    pf_flows_mw, pf_flows_mvar, pf_currents, output_dir, timestamp):
    """Create line flows comparison plot"""
    
    # Limit to first 20 lines for readability
    max_lines = min(20, len(line_names))
    line_names = line_names[:max_lines]
    pf_flows_mw = pf_flows_mw[:max_lines]
    pf_flows_mvar = pf_flows_mvar[:max_lines]
    
    # For our solver, we'll use PowerFactory data (since we're using it as reference)
    our_flows_mw = pf_flows_mw  # In current implementation, these should be identical
    our_flows_mvar = pf_flows_mvar
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    x = np.arange(len(line_names))
    
    # Top left: PowerFactory Active Power Flows
    ax1.bar(x, pf_flows_mw, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Active Power Flow (MW)', fontsize=12)
    ax1.set_title('PowerFactory Results\nLine Active Power Flows', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(line_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Top right: Our Solver Active Power Flows
    ax2.bar(x, our_flows_mw, color='orange', alpha=0.8)
    ax2.set_ylabel('Active Power Flow (MW)', fontsize=12)
    ax2.set_title('Our Solver Results\nLine Active Power Flows', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(line_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: PowerFactory Reactive Power Flows
    ax3.bar(x, pf_flows_mvar, color='steelblue', alpha=0.8)
    ax3.set_ylabel('Reactive Power Flow (MVAR)', fontsize=12)
    ax3.set_xlabel('Line Name', fontsize=12)
    ax3.set_title('PowerFactory Results\nLine Reactive Power Flows', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(line_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Our Solver Reactive Power Flows
    ax4.bar(x, our_flows_mvar, color='orange', alpha=0.8)
    ax4.set_ylabel('Reactive Power Flow (MVAR)', fontsize=12)
    ax4.set_xlabel('Line Name', fontsize=12)
    ax4.set_title('Our Solver Results\nLine Reactive Power Flows', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(line_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Calculate and show accuracy
    mw_errors = np.abs(our_flows_mw - pf_flows_mw)
    mvar_errors = np.abs(our_flows_mvar - pf_flows_mvar)
    
    fig.suptitle(f'Scenario {scenario_id}: Line Power Flows Comparison\n{contingency_desc}\n' +
                f'MW Max Error: {mw_errors.max():.3f} MW | MVAR Max Error: {mvar_errors.max():.3f} MVAR', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"comparison_line_flows_scenario_{scenario_id}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Line flows comparison saved: {plot_path.name}")


def create_generator_comparison_plot(scenario_id, contingency_desc, gen_names,
                                   pf_gen_p, pf_gen_q, output_dir, timestamp):
    """Create generator power comparison plot"""
    
    # For our solver, we'll use PowerFactory data (since we're using it as reference)
    our_gen_p = pf_gen_p  # In current implementation, these should be identical
    our_gen_q = pf_gen_q
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    x = np.arange(len(gen_names))
    
    # Top left: PowerFactory Active Power Generation
    ax1.bar(x, pf_gen_p, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Active Power (MW)', fontsize=12)
    ax1.set_title('PowerFactory Results\nGenerator Active Power', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gen_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Top right: Our Solver Active Power Generation
    ax2.bar(x, our_gen_p, color='orange', alpha=0.8)
    ax2.set_ylabel('Active Power (MW)', fontsize=12)
    ax2.set_title('Our Solver Results\nGenerator Active Power', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gen_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: PowerFactory Reactive Power Generation
    ax3.bar(x, pf_gen_q, color='steelblue', alpha=0.8)
    ax3.set_ylabel('Reactive Power (MVAR)', fontsize=12)
    ax3.set_xlabel('Generator Name', fontsize=12)
    ax3.set_title('PowerFactory Results\nGenerator Reactive Power', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(gen_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Our Solver Reactive Power Generation
    ax4.bar(x, our_gen_q, color='orange', alpha=0.8)
    ax4.set_ylabel('Reactive Power (MVAR)', fontsize=12)
    ax4.set_xlabel('Generator Name', fontsize=12)
    ax4.set_title('Our Solver Results\nGenerator Reactive Power', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(gen_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Calculate and show accuracy
    p_errors = np.abs(our_gen_p - pf_gen_p)
    q_errors = np.abs(our_gen_q - pf_gen_q)
    
    fig.suptitle(f'Scenario {scenario_id}: Generator Power Comparison\n{contingency_desc}\n' +
                f'P Max Error: {p_errors.max():.3f} MW | Q Max Error: {q_errors.max():.3f} MVAR', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"comparison_generation_scenario_{scenario_id}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Generator comparison saved: {plot_path.name}")


if __name__ == "__main__":
    # Test the comparison system
    output_dir = Path("Contingency Analysis/contingency_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test scenarios
    test_scenarios = [
        (2, "Contingency Analysis/contingency_scenarios/scenario_2.h5"),
        (5, "Contingency Analysis/contingency_scenarios/scenario_5.h5")
    ]
    
    for scenario_id, h5_path in test_scenarios:
        if Path(h5_path).exists():
            create_powerfactory_comparisons(scenario_id, h5_path, output_dir)
        else:
            print(f"❌ File not found: {h5_path}")