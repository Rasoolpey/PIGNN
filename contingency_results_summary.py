#!/usr/bin/env python3
"""
Contingency Analysis Results Summary
Analyzes multiple contingency scenarios and provides detailed summary
"""

import numpy as np
import matplotlib.pyplot as plt
from final_load_flow_solution import create_powerfactory_based_results
import h5py
import pandas as pd
from pathlib import Path

def analyze_contingency_scenario(scenario_id):
    """Analyze a single contingency scenario"""
    h5_path = f'Contingency Analysis/contingency_scenarios/scenario_{scenario_id}.h5'
    
    try:
        # Get contingency description
        with h5py.File(h5_path, 'r') as f:
            if 'disconnection_actions' in f:
                actions = f['disconnection_actions/actions'][:]
                contingency_desc = actions[0].decode() if len(actions) > 0 else f'Scenario {scenario_id}'
            else:
                contingency_desc = f'Scenario {scenario_id}'
        
        # Run load flow
        results = create_powerfactory_based_results(h5_path)
        
        # Extract key metrics
        min_voltage = float(results.voltage_magnitudes.min())
        max_voltage = float(results.voltage_magnitudes.max())
        avg_voltage = float(results.voltage_magnitudes.mean())
        total_losses = float(results.total_losses_mw)
        
        # Count voltage violations
        violations_low = np.sum(results.voltage_magnitudes < 0.95)
        violations_high = np.sum(results.voltage_magnitudes > 1.05)
        
        return {
            'scenario': scenario_id,
            'contingency': contingency_desc,
            'min_voltage': min_voltage,
            'max_voltage': max_voltage,
            'avg_voltage': avg_voltage,
            'losses_mw': total_losses,
            'violations_low': violations_low,
            'violations_high': violations_high,
            'converged': results.converged
        }
        
    except Exception as e:
        print(f'❌ Scenario {scenario_id}: Failed to analyze - {str(e)}')
        return None

def create_voltage_comparison_plot(results_data):
    """Create voltage comparison plots across scenarios"""
    if not results_data:
        return
    
    scenarios = [r['scenario'] for r in results_data]
    min_voltages = [r['min_voltage'] for r in results_data]
    max_voltages = [r['max_voltage'] for r in results_data]
    avg_voltages = [r['avg_voltage'] for r in results_data]
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Voltage ranges
    plt.subplot(2, 2, 1)
    plt.plot(scenarios, min_voltages, 'ro-', label='Minimum Voltage', linewidth=2)
    plt.plot(scenarios, max_voltages, 'bo-', label='Maximum Voltage', linewidth=2)
    plt.plot(scenarios, avg_voltages, 'go-', label='Average Voltage', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Low Limit (0.95 pu)')
    plt.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='High Limit (1.05 pu)')
    plt.xlabel('Scenario ID')
    plt.ylabel('Voltage (pu)')
    plt.title('Voltage Profiles Across Contingency Scenarios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: System losses
    plt.subplot(2, 2, 2)
    losses = [r['losses_mw'] for r in results_data]
    plt.bar(scenarios, losses, color='orange', alpha=0.7)
    plt.xlabel('Scenario ID')
    plt.ylabel('System Losses (MW)')
    plt.title('System Losses by Scenario')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Voltage violations
    plt.subplot(2, 2, 3)
    violations_low = [r['violations_low'] for r in results_data]
    violations_high = [r['violations_high'] for r in results_data]
    
    x_pos = np.arange(len(scenarios))
    plt.bar(x_pos - 0.2, violations_low, 0.4, label='Low Voltage (<0.95 pu)', color='red', alpha=0.7)
    plt.bar(x_pos + 0.2, violations_high, 0.4, label='High Voltage (>1.05 pu)', color='blue', alpha=0.7)
    
    plt.xlabel('Scenario ID')
    plt.ylabel('Number of Violations')
    plt.title('Voltage Violations by Scenario')
    plt.xticks(x_pos, scenarios)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Severity assessment
    plt.subplot(2, 2, 4)
    severity_scores = []
    for r in results_data:
        # Calculate severity score based on minimum voltage and violations
        severity = (1.0 - r['min_voltage']) * 100 + r['violations_low'] * 2
        severity_scores.append(severity)
    
    colors = ['green' if s < 2 else 'orange' if s < 5 else 'red' for s in severity_scores]
    plt.bar(scenarios, severity_scores, color=colors, alpha=0.7)
    plt.xlabel('Scenario ID')
    plt.ylabel('Severity Score')
    plt.title('Contingency Severity Assessment')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path('Contingency Analysis/contingency_plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'contingency_summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return str(plots_dir / 'contingency_summary_analysis.png')

def main():
    print('📊 Contingency Analysis Results Summary')
    print('=' * 60)
    
    # Analyze several representative scenarios
    scenarios_to_analyze = [0, 2, 5, 10, 20, 50, 77, 100, 150, 196]
    results_data = []
    
    for scenario_id in scenarios_to_analyze:
        print(f'🔍 Analyzing Scenario {scenario_id}...', end=' ')
        result = analyze_contingency_scenario(scenario_id)
        if result:
            results_data.append(result)
            print(f'✅ Min V: {result["min_voltage"]:.3f} pu, Violations: {result["violations_low"]}')
        else:
            print('❌ Failed')
    
    if not results_data:
        print('❌ No successful scenarios analyzed!')
        return
    
    # Create summary DataFrame
    df = pd.DataFrame(results_data)
    
    print('\n📈 CONTINGENCY ANALYSIS SUMMARY TABLE')
    print('=' * 100)
    
    # Format the table for better readability
    summary_cols = ['scenario', 'min_voltage', 'max_voltage', 'avg_voltage', 'losses_mw', 'violations_low', 'violations_high']
    print(df[summary_cols].to_string(index=False, formatters={
        'min_voltage': '{:.3f}'.format,
        'max_voltage': '{:.3f}'.format,
        'avg_voltage': '{:.3f}'.format,
        'losses_mw': '{:.1f}'.format,
    }))
    
    # Statistics
    print('\n📊 SYSTEM STATISTICS ACROSS SCENARIOS')
    print('=' * 50)
    print(f'Total scenarios analyzed: {len(df)}')
    print(f'All scenarios converged: {df["converged"].all()}')
    print(f'Minimum voltage across all scenarios: {df["min_voltage"].min():.3f} pu')
    print(f'Maximum voltage across all scenarios: {df["max_voltage"].max():.3f} pu')
    print(f'Average system losses: {df["losses_mw"].mean():.1f} MW (range: {df["losses_mw"].min():.1f} - {df["losses_mw"].max():.1f} MW)')
    print(f'Scenarios with low voltage violations: {(df["violations_low"] > 0).sum()} of {len(df)}')
    print(f'Scenarios with high voltage violations: {(df["violations_high"] > 0).sum()} of {len(df)}')
    
    # Most critical scenario
    most_critical = df.loc[df['min_voltage'].idxmin()]
    print(f'\n⚠️  MOST CRITICAL SCENARIO:')
    print(f'   Scenario {most_critical["scenario"]}: {most_critical["contingency"][:70]}...')
    print(f'   Minimum voltage: {most_critical["min_voltage"]:.3f} pu')
    print(f'   Voltage violations: {most_critical["violations_low"]} buses below 0.95 pu')
    
    # Best scenario
    best_scenario = df.loc[df['min_voltage'].idxmax()]
    print(f'\n✅ BEST SCENARIO:')
    print(f'   Scenario {best_scenario["scenario"]}: {best_scenario["contingency"][:70]}...')
    print(f'   Minimum voltage: {best_scenario["min_voltage"]:.3f} pu')
    print(f'   Voltage violations: {best_scenario["violations_low"]} buses below 0.95 pu')
    
    # Create visualization
    print(f'\n📊 Creating contingency analysis visualization...')
    plot_path = create_voltage_comparison_plot(results_data)
    print(f'📈 Plot saved to: {plot_path}')
    
    # Show contingency types
    print(f'\n🔍 CONTINGENCY TYPES ANALYZED:')
    for _, row in df.iterrows():
        contingency_short = row['contingency'][:60] + ('...' if len(row['contingency']) > 60 else '')
        severity = '🟢' if row['min_voltage'] >= 0.95 else '🟡' if row['min_voltage'] >= 0.90 else '🔴'
        print(f'   {severity} Scenario {row["scenario"]:3d}: {contingency_short}')

if __name__ == '__main__':
    main()