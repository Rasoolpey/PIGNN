# contingency_analysis_results_viewer.py - 2025-08-03
"""
Contingency Analysis Results Viewer
Analyzes voltage sensitivity and Y-matrix results from converged scenarios

This script:
1. Scans all H5 files in the contingency_scenarios directory
2. Identifies converged scenarios with complete analysis
3. Selects a random converged case for detailed analysis
4. Displays voltage sensitivity results with analysis
5. Shows Y-matrix properties and structure
6. Provides insights on result quality and realism
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

# Configuration
H5_DIR = os.path.join(os.getcwd(), "contingency_scenarios")
SBASE_MVA = 100.0

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(f"🔍 CONTINGENCY ANALYSIS RESULTS VIEWER")
print("="*60)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Directory: {H5_DIR}")
print()

def scan_h5_files():
    """Scan H5 files and categorize them by analysis completion status"""
    
    print(f"📊 SCANNING H5 FILES...")
    
    if not os.path.exists(H5_DIR):
        print(f"❌ Directory not found: {H5_DIR}")
        return None
    
    h5_files = [f for f in os.listdir(H5_DIR) if f.endswith('.h5')]
    h5_files.sort()
    
    if not h5_files:
        print(f"❌ No H5 files found in {H5_DIR}")
        return None
    
    print(f"   📁 Found {len(h5_files)} H5 files")
    
    scenarios_info = {
        'all_scenarios': [],
        'converged_scenarios': [],
        'with_voltage_sensitivity': [],
        'with_y_matrix': [],
        'complete_scenarios': []  # Has all: convergence + VS + Y-matrix
    }
    
    for h5_file in h5_files:
        h5_path = os.path.join(H5_DIR, h5_file)
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Basic scenario info
                scenario_id = f['scenario_metadata']['scenario_id'][()]
                contingency_type = f['scenario_metadata']['contingency_type'][()].decode()
                description = f['scenario_metadata']['description'][()].decode()
                
                # Check load flow convergence
                lf_converged = f['load_flow_results']['convergence'][()]
                
                # Check voltage sensitivity
                has_vs = ('voltage_sensitivity' in f and 
                         'analysis_timestamp' in f['voltage_sensitivity'])
                
                # Check Y-matrix
                has_y_matrix = ('y_matrix' in f and 
                               'construction_timestamp' in f['y_matrix'])
                
                scenario_info = {
                    'file': h5_file,
                    'path': h5_path,
                    'scenario_id': scenario_id,
                    'contingency_type': contingency_type,
                    'description': description,
                    'converged': lf_converged,
                    'has_voltage_sensitivity': has_vs,
                    'has_y_matrix': has_y_matrix
                }
                
                scenarios_info['all_scenarios'].append(scenario_info)
                
                if lf_converged:
                    scenarios_info['converged_scenarios'].append(scenario_info)
                
                if has_vs:
                    scenarios_info['with_voltage_sensitivity'].append(scenario_info)
                
                if has_y_matrix:
                    scenarios_info['with_y_matrix'].append(scenario_info)
                
                if lf_converged and has_vs and has_y_matrix:
                    scenarios_info['complete_scenarios'].append(scenario_info)
                    
        except Exception as e:
            print(f"   ⚠️ Error reading {h5_file}: {e}")
    
    # Summary
    print(f"   📊 Analysis Summary:")
    print(f"      Total scenarios: {len(scenarios_info['all_scenarios'])}")
    print(f"      Converged: {len(scenarios_info['converged_scenarios'])}")
    print(f"      With voltage sensitivity: {len(scenarios_info['with_voltage_sensitivity'])}")
    print(f"      With Y-matrix: {len(scenarios_info['with_y_matrix'])}")
    print(f"      Complete (all modules): {len(scenarios_info['complete_scenarios'])}")
    
    return scenarios_info

def select_random_scenario(scenarios_info):
    """Select a random complete scenario for analysis"""
    
    complete_scenarios = scenarios_info['complete_scenarios']
    
    if not complete_scenarios:
        print(f"❌ No complete scenarios found!")
        # Fallback to converged scenarios
        converged_scenarios = scenarios_info['converged_scenarios']
        if converged_scenarios:
            print(f"   🔄 Falling back to converged scenarios...")
            selected = random.choice(converged_scenarios)
            print(f"   ⚠️ Note: This scenario may not have all analysis modules complete")
        else:
            print(f"   ❌ No converged scenarios available either!")
            return None
    else:
        selected = random.choice(complete_scenarios)
    
    print(f"\n🎯 SELECTED SCENARIO:")
    print(f"   📁 File: {selected['file']}")
    print(f"   🆔 Scenario ID: {selected['scenario_id']}")
    print(f"   🏷️ Type: {selected['contingency_type']}")
    print(f"   📋 Description: {selected['description']}")
    print(f"   ✅ Converged: {selected['converged']}")
    print(f"   🔬 Voltage Sensitivity: {selected['has_voltage_sensitivity']}")
    print(f"   🔧 Y-Matrix: {selected['has_y_matrix']}")
    
    return selected

def analyze_voltage_sensitivity(h5_path):
    """Analyze voltage sensitivity results"""
    
    print(f"\n🔬 VOLTAGE SENSITIVITY ANALYSIS:")
    print("-" * 40)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'voltage_sensitivity' not in f:
                print(f"   ❌ No voltage sensitivity data found")
                return None
            
            vs = f['voltage_sensitivity']
            
            # Analysis metadata
            analysis_time = vs['analysis_time_seconds'][()]
            p_step = vs['perturbation_step_P_MW'][()]
            q_step = vs['perturbation_step_Q_MVAR'][()]
            min_threshold = vs['min_sensitivity_threshold'][()]
            
            print(f"   ⚙️ Analysis Parameters:")
            print(f"      P perturbation: ±{p_step} MW")
            print(f"      Q perturbation: ±{q_step} MVAR")
            print(f"      Min threshold: {min_threshold}")
            print(f"      Analysis time: {analysis_time:.2f} seconds")
            
            vs_results = {'generators': {}, 'loads': {}}
            
            # Analyze generator sensitivities
            if 'generators' in vs:
                gen_group = vs['generators']
                num_gens = gen_group['num_generators_analyzed'][()]
                print(f"\n   🔋 GENERATOR SENSITIVITIES ({num_gens} generators):")
                
                total_gen_sensitivities = 0
                gen_names = []
                
                for gen_name in gen_group.keys():
                    if gen_name == 'num_generators_analyzed':
                        continue
                    
                    gen_data = gen_group[gen_name]
                    base_p = gen_data['base_P_MW'][()]
                    base_q = gen_data['base_Q_MVAR'][()]
                    
                    # P sensitivities
                    p_sens_count = 0
                    p_sens_values = []
                    if 'P_sensitivity' in gen_data:
                        p_sens_data = gen_data['P_sensitivity']
                        p_sens_count = p_sens_data['num_sensitive_buses'][()]
                        if p_sens_count > 0:
                            p_sens_values = p_sens_data['sensitivities'][:]
                    
                    # Q sensitivities
                    q_sens_count = 0
                    q_sens_values = []
                    if 'Q_sensitivity' in gen_data:
                        q_sens_data = gen_data['Q_sensitivity']
                        q_sens_count = q_sens_data['num_sensitive_buses'][()]
                        if q_sens_count > 0:
                            q_sens_values = q_sens_data['sensitivities'][:]
                    
                    total_sensitivities = p_sens_count + q_sens_count
                    total_gen_sensitivities += total_sensitivities
                    
                    gen_name_clean = gen_name.replace('_', ' ')
                    gen_names.append(gen_name_clean)
                    
                    print(f"      📋 {gen_name_clean}:")
                    print(f"         Base: P={base_p:.1f}MW, Q={base_q:.1f}MVAR")
                    print(f"         Sensitivities: {p_sens_count}P + {q_sens_count}Q = {total_sensitivities}")
                    
                    if len(p_sens_values) > 0:
                        print(f"         P sens range: [{np.min(p_sens_values):.6f}, {np.max(p_sens_values):.6f}] pu/MW")
                    if len(q_sens_values) > 0:
                        print(f"         Q sens range: [{np.min(q_sens_values):.6f}, {np.max(q_sens_values):.6f}] pu/MVAR")
                    
                    vs_results['generators'][gen_name_clean] = {
                        'base_P_MW': base_p,
                        'base_Q_MVAR': base_q,
                        'P_sensitivities': p_sens_values,
                        'Q_sensitivities': q_sens_values,
                        'total_sensitivities': total_sensitivities
                    }
                
                print(f"      📊 Total generator sensitivities: {total_gen_sensitivities}")
            
            # Analyze load sensitivities
            if 'loads' in vs:
                load_group = vs['loads']
                num_loads = load_group['num_loads_analyzed'][()]
                print(f"\n   🏠 LOAD SENSITIVITIES ({num_loads} loads):")
                
                total_load_sensitivities = 0
                load_names = []
                
                for load_name in load_group.keys():
                    if load_name == 'num_loads_analyzed':
                        continue
                    
                    load_data = load_group[load_name]
                    base_p = load_data['base_P_MW'][()]
                    base_q = load_data['base_Q_MVAR'][()]
                    
                    # P sensitivities
                    p_sens_count = 0
                    p_sens_values = []
                    if 'P_sensitivity' in load_data:
                        p_sens_data = load_data['P_sensitivity']
                        p_sens_count = p_sens_data['num_sensitive_buses'][()]
                        if p_sens_count > 0:
                            p_sens_values = p_sens_data['sensitivities'][:]
                    
                    # Q sensitivities
                    q_sens_count = 0
                    q_sens_values = []
                    if 'Q_sensitivity' in load_data:
                        q_sens_data = load_data['Q_sensitivity']
                        q_sens_count = q_sens_data['num_sensitive_buses'][()]
                        if q_sens_count > 0:
                            q_sens_values = q_sens_data['sensitivities'][:]
                    
                    total_sensitivities = p_sens_count + q_sens_count
                    total_load_sensitivities += total_sensitivities
                    
                    load_name_clean = load_name.replace('_', ' ')
                    load_names.append(load_name_clean)
                    
                    if total_sensitivities > 5:  # Only show loads with significant sensitivities
                        print(f"      📋 {load_name_clean}:")
                        print(f"         Base: P={base_p:.1f}MW, Q={base_q:.1f}MVAR")
                        print(f"         Sensitivities: {p_sens_count}P + {q_sens_count}Q = {total_sensitivities}")
                        
                        if len(p_sens_values) > 0:
                            print(f"         P sens range: [{np.min(p_sens_values):.6f}, {np.max(p_sens_values):.6f}] pu/MW")
                        if len(q_sens_values) > 0:
                            print(f"         Q sens range: [{np.min(q_sens_values):.6f}, {np.max(q_sens_values):.6f}] pu/MVAR")
                    
                    vs_results['loads'][load_name_clean] = {
                        'base_P_MW': base_p,
                        'base_Q_MVAR': base_q,
                        'P_sensitivities': p_sens_values,
                        'Q_sensitivities': q_sens_values,
                        'total_sensitivities': total_sensitivities
                    }
                
                print(f"      📊 Total load sensitivities: {total_load_sensitivities}")
                print(f"      📊 Total combined sensitivities: {total_gen_sensitivities + total_load_sensitivities}")
            
            return vs_results
            
    except Exception as e:
        print(f"   ❌ Error analyzing voltage sensitivity: {e}")
        return None

def analyze_y_matrix(h5_path):
    """Analyze Y-matrix results"""
    
    print(f"\n🔧 Y-MATRIX ANALYSIS:")
    print("-" * 40)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'y_matrix' not in f:
                print(f"   ❌ No Y-matrix data found")
                return None
            
            y_group = f['y_matrix']
            
            # Construction metadata
            construction_time = y_group['construction_time_seconds'][()]
            matrix_size = y_group['matrix_size'][()]
            
            print(f"   ⚙️ Construction Parameters:")
            print(f"      Matrix size: {matrix_size} x {matrix_size}")
            print(f"      Construction time: {construction_time:.2f} seconds")
            
            # Bus information
            bus_info = y_group['bus_information']
            bus_names = [name.decode() for name in bus_info['bus_names'][:]]
            num_buses = bus_info['num_buses'][()]
            
            print(f"      Number of buses: {num_buses}")
            
            # Load Y-matrix from sparse format
            matrix_group = y_group['admittance_matrix']
            data_real = matrix_group['data_real'][:]
            data_imag = matrix_group['data_imag'][:]
            data_complex = data_real + 1j * data_imag
            indices = matrix_group['indices'][:]
            indptr = matrix_group['indptr'][:]
            shape = tuple(matrix_group['shape'][:])
            
            # Reconstruct sparse matrix
            Y_matrix = sp.csr_matrix((data_complex, indices, indptr), shape=shape)
            
            print(f"\n   📊 MATRIX PROPERTIES:")
            
            # Matrix properties
            if 'matrix_properties' in y_group:
                props = y_group['matrix_properties']
                
                nnz = props['nnz'][()]
                density = props['density'][()]
                is_hermitian = props['is_hermitian'][()]
                
                print(f"      Non-zero elements: {nnz}")
                print(f"      Density: {density:.6f} ({density*100:.4f}%)")
                print(f"      Hermitian: {is_hermitian}")
                
                if 'condition_number' in props:
                    cond_num = props['condition_number'][()]
                    if not np.isnan(cond_num):
                        print(f"      Condition number: {cond_num:.2e}")
                
                if 'smallest_eigenvalue_magnitude' in props:
                    min_eig = props['smallest_eigenvalue_magnitude'][()]
                    max_eig = props['largest_eigenvalue_magnitude'][()]
                    if not (np.isnan(min_eig) or np.isnan(max_eig)):
                        print(f"      Eigenvalue range: [{min_eig:.2e}, {max_eig:.2e}]")
                
                if 'zero_diagonal_elements' in props:
                    zero_diag = props['zero_diagonal_elements'][()]
                    if zero_diag > 0:
                        print(f"      ⚠️ Zero diagonal elements: {zero_diag}")
                
                # Diagonal analysis
                Y_diag = Y_matrix.diagonal()
                print(f"\n   🔍 DIAGONAL ANALYSIS:")
                print(f"      Real part range: [{np.min(Y_diag.real):.4f}, {np.max(Y_diag.real):.4f}] S")
                print(f"      Imag part range: [{np.min(Y_diag.imag):.4f}, {np.max(Y_diag.imag):.4f}] S")
                print(f"      Magnitude range: [{np.min(np.abs(Y_diag)):.4f}, {np.max(np.abs(Y_diag)):.4f}] S")
            
            # Construction statistics
            if 'construction_statistics' in y_group:
                stats = y_group['construction_statistics']
                
                lines_added = stats['lines'][()]
                transformers_added = stats['transformers'][()]
                generators_added = stats['generators'][()]
                loads_added = stats['loads'][()]
                shunts_added = stats['shunts'][()]
                skipped = stats['skipped'][()]
                
                total_added = lines_added + transformers_added + generators_added + loads_added + shunts_added
                
                print(f"\n   🔧 CONSTRUCTION STATISTICS:")
                print(f"      Elements added: {total_added}")
                print(f"        Lines: {lines_added}")
                print(f"        Transformers: {transformers_added}")
                print(f"        Generators: {generators_added}")
                print(f"        Loads: {loads_added}")
                print(f"        Shunts: {shunts_added}")
                print(f"      Elements skipped: {skipped}")
                
                if total_added > 0:
                    success_rate = (total_added / (total_added + skipped)) * 100
                    print(f"      Success rate: {success_rate:.1f}%")
            
            # Sample some matrix values for realism check
            print(f"\n   🔬 SAMPLE MATRIX VALUES:")
            Y_dense = Y_matrix.toarray()
            
            # Show a few diagonal elements
            print(f"      Sample diagonal elements (first 5):")
            for i in range(min(5, matrix_size)):
                bus_name = bus_names[i] if i < len(bus_names) else f"Bus_{i}"
                diag_val = Y_dense[i, i]
                print(f"        {bus_name}: {diag_val.real:.4f} + j{diag_val.imag:.4f} S")
            
            # Show some off-diagonal elements
            print(f"      Sample off-diagonal elements:")
            off_diag_count = 0
            for i in range(min(5, matrix_size)):
                for j in range(min(5, matrix_size)):
                    if i != j and abs(Y_dense[i, j]) > 1e-6:
                        bus_i = bus_names[i] if i < len(bus_names) else f"Bus_{i}"
                        bus_j = bus_names[j] if j < len(bus_names) else f"Bus_{j}"
                        off_diag_val = Y_dense[i, j]
                        print(f"        Y[{bus_i}, {bus_j}]: {off_diag_val.real:.4f} + j{off_diag_val.imag:.4f} S")
                        off_diag_count += 1
                        if off_diag_count >= 3:
                            break
                if off_diag_count >= 3:
                    break
            
            return {
                'Y_matrix': Y_matrix,
                'bus_names': bus_names,
                'matrix_size': matrix_size,
                'properties': props if 'matrix_properties' in y_group else {},
                'construction_time': construction_time
            }
            
    except Exception as e:
        print(f"   ❌ Error analyzing Y-matrix: {e}")
        return None

def check_result_realism(vs_results, y_matrix_results):
    """Check the realism of the results"""
    
    print(f"\n🎯 RESULT REALISM ASSESSMENT:")
    print("-" * 40)
    
    realism_score = 0
    max_score = 100
    
    # Check voltage sensitivity realism
    if vs_results:
        print(f"   🔬 VOLTAGE SENSITIVITY REALISM:")
        
        # Check if sensitivity values are in reasonable range
        all_p_sens = []
        all_q_sens = []
        
        for element_type in ['generators', 'loads']:
            for element_name, data in vs_results[element_type].items():
                all_p_sens.extend(data['P_sensitivities'])
                all_q_sens.extend(data['Q_sensitivities'])
        
        if all_p_sens:
            p_range = [np.min(all_p_sens), np.max(all_p_sens)]
            print(f"      P sensitivity range: [{p_range[0]:.6f}, {p_range[1]:.6f}] pu/MW")
            
            # Typical P sensitivities should be in range 1e-6 to 1e-2 pu/MW
            if 1e-6 <= abs(p_range[0]) <= 1e-2 and 1e-6 <= abs(p_range[1]) <= 1e-2:
                print(f"      ✅ P sensitivity range is realistic")
                realism_score += 20
            else:
                print(f"      ⚠️ P sensitivity range may be unusual")
        
        if all_q_sens:
            q_range = [np.min(all_q_sens), np.max(all_q_sens)]
            print(f"      Q sensitivity range: [{q_range[0]:.6f}, {q_range[1]:.6f}] pu/MVAR")
            
            # Typical Q sensitivities should be in range 1e-6 to 1e-2 pu/MVAR
            if 1e-6 <= abs(q_range[0]) <= 1e-2 and 1e-6 <= abs(q_range[1]) <= 1e-2:
                print(f"      ✅ Q sensitivity range is realistic")
                realism_score += 20
            else:
                print(f"      ⚠️ Q sensitivity range may be unusual")
        
        # Check if generators have more sensitivity than loads (typically true)
        gen_sens_count = sum(data['total_sensitivities'] for data in vs_results['generators'].values())
        load_sens_count = sum(data['total_sensitivities'] for data in vs_results['loads'].values())
        
        if gen_sens_count > 0 and load_sens_count > 0:
            if gen_sens_count >= load_sens_count * 0.5:  # Generators should have reasonable sensitivity
                print(f"      ✅ Generator/load sensitivity ratio is reasonable")
                realism_score += 10
            else:
                print(f"      ⚠️ Generator sensitivity seems low compared to loads")
    
    # Check Y-matrix realism
    if y_matrix_results:
        print(f"\n   🔧 Y-MATRIX REALISM:")
        
        props = y_matrix_results['properties']
        Y_matrix = y_matrix_results['Y_matrix']
        matrix_size = y_matrix_results['matrix_size']
        
        # Check matrix density (should be sparse for power systems)
        density = props.get('density', 0)
        if density <= 0:  # If density not properly stored, calculate it
            nnz = Y_matrix.nnz
            density = nnz / (matrix_size * matrix_size) if matrix_size > 0 else 0
        
        if 0.01 <= density <= 0.3:  # Typical range for power system Y-matrices
            print(f"      ✅ Matrix density ({density:.4f}) is realistic for power systems")
            realism_score += 15
        elif density > 0.3:
            print(f"      ⚠️ Matrix density ({density:.4f}) is high but acceptable for contingency scenarios")
            realism_score += 10
        elif density > 0:
            print(f"      ⚠️ Matrix density ({density:.4f}) is low but acceptable for contingency scenarios")
            realism_score += 10
        else:
            print(f"      ⚠️ Matrix density cannot be determined")
        
        # Check if matrix is Hermitian (should be for passive networks)
        is_hermitian = props.get('is_hermitian', False)
        if is_hermitian:
            print(f"      ✅ Matrix is Hermitian (expected for power systems)")
            realism_score += 10
        else:
            print(f"      ℹ️ Matrix is not Hermitian (normal for systems with generators and complex loads)")
            realism_score += 8  # Still give most points since this is normal
        
        # Check condition number
        cond_num = props.get('condition_number', np.nan)
        if not np.isnan(cond_num):
            if cond_num < 1e12:
                print(f"      ✅ Condition number ({cond_num:.2e}) indicates good numerical stability")
                realism_score += 10
            else:
                print(f"      ⚠️ High condition number ({cond_num:.2e}) may indicate numerical issues")
        
        # Check diagonal elements (should be positive real parts for passive elements)
        Y_diag = Y_matrix.diagonal()
        positive_real_diag = np.sum(Y_diag.real > 0)
        total_diag = len(Y_diag)
        
        if positive_real_diag / total_diag > 0.8:  # Most diagonal elements should have positive real parts
            print(f"      ✅ {positive_real_diag}/{total_diag} diagonal elements have positive real parts")
            realism_score += 15
        else:
            print(f"      ⚠️ Only {positive_real_diag}/{total_diag} diagonal elements have positive real parts")
        
        # Check zero diagonal elements
        zero_diag = props.get('zero_diagonal_elements', 0)
        if zero_diag == 0:
            print(f"      ✅ No zero diagonal elements (good connectivity)")
        elif zero_diag < total_diag * 0.1:
            print(f"      ⚠️ {zero_diag} zero diagonal elements (may indicate islanded buses)")
        else:
            print(f"      ❌ Many zero diagonal elements ({zero_diag}) - significant connectivity issues")
    
    # Overall assessment
    print(f"\n   📊 OVERALL REALISM SCORE: {realism_score}/{max_score} ({realism_score/max_score*100:.1f}%)")
    
    if realism_score >= 80:
        print(f"   🎉 Excellent! Results appear highly realistic")
    elif realism_score >= 60:
        print(f"   ✅ Good! Results appear realistic with minor concerns")
    elif realism_score >= 40:
        print(f"   ⚠️ Fair! Results have some unusual characteristics")
    else:
        print(f"   ❌ Poor! Results may have significant issues")
    
    return realism_score

def create_visualization(vs_results, y_matrix_results, scenario_info):
    """Create visualizations of the results"""
    
    print(f"\n📊 CREATING VISUALIZATIONS...")
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Contingency Analysis Results - Scenario {scenario_info["scenario_id"]}\n{scenario_info["description"]}', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Voltage Sensitivity Distribution
        if vs_results:
            ax1 = axes[0, 0]
            
            all_sensitivities = []
            labels = []
            
            for element_type in ['generators', 'loads']:
                for element_name, data in vs_results[element_type].items():
                    if data['P_sensitivities']:
                        all_sensitivities.extend(data['P_sensitivities'])
                        labels.extend([f"{element_type[:-1].title()} P"] * len(data['P_sensitivities']))
                    if data['Q_sensitivities']:
                        all_sensitivities.extend(data['Q_sensitivities'])
                        labels.extend([f"{element_type[:-1].title()} Q"] * len(data['Q_sensitivities']))
            
            if len(all_sensitivities) > 0:  # Use len() instead of direct boolean evaluation
                # Convert to log scale for better visualization
                log_sens = np.log10(np.abs(all_sensitivities) + 1e-10)
                ax1.hist(log_sens, bins=20, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Log10(|Sensitivity|)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Voltage Sensitivity Distribution')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No Sensitivity Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Voltage Sensitivity Distribution')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Voltage Sensitivity Data', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Voltage Sensitivity Distribution')
        
        # Plot 2: Generator vs Load Sensitivities
        if vs_results:
            ax2 = axes[0, 1]
            
            gen_total = [data['total_sensitivities'] for data in vs_results['generators'].values()]
            load_total = [data['total_sensitivities'] for data in vs_results['loads'].values()]
            
            # Filter out zeros for better visualization
            gen_total = [x for x in gen_total if x > 0]
            load_total = [x for x in load_total if x > 0]
            
            if gen_total or load_total:
                data_to_plot = []
                labels_to_plot = []
                if len(gen_total) > 0:  # Use len() instead of direct boolean evaluation
                    data_to_plot.append(gen_total)
                    labels_to_plot.append('Generators')
                if len(load_total) > 0:  # Use len() instead of direct boolean evaluation
                    data_to_plot.append(load_total)
                    labels_to_plot.append('Loads')
                
                if len(data_to_plot) > 0:  # Only plot if we have data
                    ax2.boxplot(data_to_plot, labels=labels_to_plot)
                    ax2.set_ylabel('Number of Sensitive Buses')
                    ax2.set_title('Sensitivity Count Comparison')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No Sensitivity Counts', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Sensitivity Count Comparison')
            else:
                ax2.text(0.5, 0.5, 'No Sensitivity Counts', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Sensitivity Count Comparison')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Voltage Sensitivity Data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Sensitivity Count Comparison')
        
        # Plot 3: Y-Matrix Diagonal Elements
        if y_matrix_results:
            ax3 = axes[1, 0]
            
            Y_matrix = y_matrix_results['Y_matrix']
            Y_diag = Y_matrix.diagonal()
            
            # Plot real vs imaginary parts
            ax3.scatter(Y_diag.real, Y_diag.imag, alpha=0.6, s=30)
            ax3.set_xlabel('Real Part (S)')
            ax3.set_ylabel('Imaginary Part (S)')
            ax3.set_title('Y-Matrix Diagonal Elements')
            ax3.grid(True, alpha=0.3)
            
            # Add quadrant lines
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Y-Matrix Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Y-Matrix Diagonal Elements')
        
        # Plot 4: Y-Matrix Sparsity Pattern (sample)
        if y_matrix_results:
            ax4 = axes[1, 1]
            
            Y_matrix = y_matrix_results['Y_matrix']
            matrix_size = Y_matrix.shape[0]
            
            # Show sparsity pattern for smaller matrices or sample for large ones
            if matrix_size <= 50:
                # Show full pattern
                Y_dense = Y_matrix.toarray()
                sparsity_pattern = np.abs(Y_dense) > 1e-6
                im = ax4.imshow(sparsity_pattern, cmap='Blues', aspect='equal')
                ax4.set_title(f'Y-Matrix Sparsity Pattern ({matrix_size}x{matrix_size})')
            else:
                # Show sample (first 50x50)
                sample_size = min(50, matrix_size)
                Y_sample = Y_matrix[:sample_size, :sample_size].toarray()
                sparsity_pattern = np.abs(Y_sample) > 1e-6
                im = ax4.imshow(sparsity_pattern, cmap='Blues', aspect='equal')
                ax4.set_title(f'Y-Matrix Sparsity Pattern (Sample {sample_size}x{sample_size})')
            
            ax4.set_xlabel('Bus Index')
            ax4.set_ylabel('Bus Index')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Y-Matrix Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Y-Matrix Sparsity Pattern')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"contingency_analysis_scenario_{scenario_info['scenario_id']}.png"
        plot_path = os.path.join(H5_DIR, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   📊 Visualization saved: {plot_filename}")
        
    except Exception as e:
        print(f"   ❌ Error creating visualizations: {e}")

def export_detailed_data(vs_results, y_matrix_results, scenario_info):
    """Export detailed data to CSV files for further analysis"""
    
    print(f"\n📤 EXPORTING DETAILED DATA...")
    
    try:
        scenario_id = scenario_info['scenario_id']
        
        # Export voltage sensitivity data
        if vs_results:
            vs_data = []
            
            for element_type in ['generators', 'loads']:
                for element_name, data in vs_results[element_type].items():
                    # P sensitivities
                    for i, sens_val in enumerate(data['P_sensitivities']):
                        vs_data.append({
                            'element_type': element_type[:-1],  # Remove 's'
                            'element_name': element_name,
                            'perturbation_type': 'P',
                            'base_value_MW': data['base_P_MW'],
                            'base_value_MVAR': data['base_Q_MVAR'],
                            'sensitivity_value': sens_val,
                            'sensitivity_magnitude': abs(sens_val),
                            'bus_index': i
                        })
                    
                    # Q sensitivities
                    for i, sens_val in enumerate(data['Q_sensitivities']):
                        vs_data.append({
                            'element_type': element_type[:-1],
                            'element_name': element_name,
                            'perturbation_type': 'Q',
                            'base_value_MW': data['base_P_MW'],
                            'base_value_MVAR': data['base_Q_MVAR'],
                            'sensitivity_value': sens_val,
                            'sensitivity_magnitude': abs(sens_val),
                            'bus_index': i
                        })
            
            if vs_data:
                vs_df = pd.DataFrame(vs_data)
                vs_filename = f"voltage_sensitivity_scenario_{scenario_id}.csv"
                vs_path = os.path.join(H5_DIR, vs_filename)
                vs_df.to_csv(vs_path, index=False)
                print(f"   📄 Voltage sensitivity data exported: {vs_filename}")
                print(f"       Records: {len(vs_data)}")
        
        # Export Y-matrix summary data
        if y_matrix_results:
            Y_matrix = y_matrix_results['Y_matrix']
            bus_names = y_matrix_results['bus_names']
            
            # Create Y-matrix summary
            y_data = []
            Y_diag = Y_matrix.diagonal()
            
            for i, bus_name in enumerate(bus_names):
                if i < len(Y_diag):
                    diag_val = Y_diag[i]
                    y_data.append({
                        'bus_index': i,
                        'bus_name': bus_name,
                        'diagonal_real': diag_val.real,
                        'diagonal_imag': diag_val.imag,
                        'diagonal_magnitude': abs(diag_val),
                        'diagonal_angle_deg': np.angle(diag_val, deg=True)
                    })
            
            if y_data:
                y_df = pd.DataFrame(y_data)
                y_filename = f"y_matrix_summary_scenario_{scenario_id}.csv"
                y_path = os.path.join(H5_DIR, y_filename)
                y_df.to_csv(y_path, index=False)
                print(f"   📄 Y-matrix summary exported: {y_filename}")
                print(f"       Buses: {len(y_data)}")
        
        # Create combined summary
        summary_data = {
            'scenario_id': scenario_id,
            'contingency_type': scenario_info['contingency_type'],
            'description': scenario_info['description'],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        if vs_results:
            gen_count = len(vs_results['generators'])
            load_count = len(vs_results['loads'])
            total_sens = sum(data['total_sensitivities'] for data in vs_results['generators'].values()) + \
                        sum(data['total_sensitivities'] for data in vs_results['loads'].values())
            
            summary_data.update({
                'generators_analyzed': gen_count,
                'loads_analyzed': load_count,
                'total_sensitivities': total_sens
            })
        
        if y_matrix_results:
            summary_data.update({
                'y_matrix_size': y_matrix_results['matrix_size'],
                'y_matrix_nnz': y_matrix_results['properties'].get('nnz', 0),
                'y_matrix_density': y_matrix_results['properties'].get('density', 0),
                'y_matrix_condition_number': y_matrix_results['properties'].get('condition_number', np.nan),
                'construction_time_seconds': y_matrix_results['construction_time']
            })
        
        summary_filename = f"analysis_summary_scenario_{scenario_id}.json"
        summary_path = os.path.join(H5_DIR, summary_filename)
        
        import json
        with open(summary_path, 'w') as f:
            # Handle numpy types for JSON serialization
            summary_json = {}
            for key, value in summary_data.items():
                if isinstance(value, np.integer):
                    summary_json[key] = int(value)
                elif isinstance(value, np.floating):
                    summary_json[key] = float(value) if not np.isnan(value) else None
                else:
                    summary_json[key] = value
            
            json.dump(summary_json, f, indent=2)
        
        print(f"   📄 Analysis summary exported: {summary_filename}")
        
    except Exception as e:
        print(f"   ❌ Error exporting data: {e}")

def main():
    """Main execution function"""
    
    # Scan H5 files
    scenarios_info = scan_h5_files()
    if not scenarios_info:
        return
    
    # Select random scenario
    selected_scenario = select_random_scenario(scenarios_info)
    if not selected_scenario:
        return
    
    h5_path = selected_scenario['path']
    
    # Analyze voltage sensitivity
    vs_results = analyze_voltage_sensitivity(h5_path)
    
    # Analyze Y-matrix
    y_matrix_results = analyze_y_matrix(h5_path)
    
    # Check result realism
    realism_score = 0
    if vs_results or y_matrix_results:
        realism_score = check_result_realism(vs_results, y_matrix_results)
    
    # Create visualizations
    if vs_results or y_matrix_results:
        create_visualization(vs_results, y_matrix_results, selected_scenario)
    
    # Export detailed data
    if vs_results or y_matrix_results:
        export_detailed_data(vs_results, y_matrix_results, selected_scenario)
    
    # Final summary
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 SUMMARY:")
    print(f"   🎯 Analyzed scenario: {selected_scenario['scenario_id']}")
    print(f"   🏷️ Type: {selected_scenario['contingency_type']}")
    print(f"   📋 Description: {selected_scenario['description']}")
    
    if vs_results:
        gen_count = len(vs_results['generators'])
        load_count = len(vs_results['loads'])
        total_sens = sum(data['total_sensitivities'] for data in vs_results['generators'].values()) + \
                    sum(data['total_sensitivities'] for data in vs_results['loads'].values())
        print(f"   🔬 Voltage sensitivity: {gen_count} generators + {load_count} loads = {total_sens} sensitivities")
    
    if y_matrix_results:
        matrix_size = y_matrix_results['matrix_size']
        # Fix: Get nnz and density from the correct source
        if 'properties' in y_matrix_results and y_matrix_results['properties']:
            nnz = y_matrix_results['properties'].get('nnz', 0)
            density = y_matrix_results['properties'].get('density', 0)
        else:
            # Fallback: calculate from matrix if properties not available
            Y_matrix = y_matrix_results['Y_matrix']
            nnz = Y_matrix.nnz
            density = Y_matrix.nnz / (matrix_size * matrix_size)
        print(f"   🔧 Y-matrix: {matrix_size}x{matrix_size}, {nnz} non-zeros, {density:.4f} density")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   🔍 The results show post-contingency system characteristics")
    print(f"   📊 Voltage sensitivities indicate how bus voltages respond to P/Q changes")
    print(f"   🔧 Y-matrix represents the complete network admittance structure")
    print(f"   📈 These can be used for:")
    print(f"      • Voltage stability analysis")
    print(f"      • Optimal power flow studies")
    print(f"      • Control system design")
    print(f"      • Security assessment")
    
    if realism_score >= 70:
        print(f"\n✅ CONFIDENCE: HIGH - Results appear realistic and suitable for analysis")
    elif realism_score >= 50:
        print(f"\n⚠️ CONFIDENCE: MEDIUM - Results are acceptable but review recommended")
    else:
        print(f"\n❌ CONFIDENCE: LOW - Results should be investigated before use")

if __name__ == "__main__":
    # Set random seed for reproducible selection
    random.seed(42)
    main()