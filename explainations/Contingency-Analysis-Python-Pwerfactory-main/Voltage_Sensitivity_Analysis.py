# Voltage_Sensitivity_Analysis.py - 2025-08-02
"""
Voltage Sensitivity Analysis Module for Post-Contingency Analysis
Compatible with existing H5 file structure from Contingency_Executor.py

This module:
1. Reads existing H5 files from contingency analysis
2. Identifies active generators and loads post-contingency
3. Performs systematic voltage sensitivity analysis by perturbing P and Q
4. Measures voltage changes on all buses (not just adjacent ones)
5. Stores results in the same H5 file structure
6. Updates analysis module status

Features:
- Finite difference method for sensitivity calculation
- Configurable perturbation steps
- Comprehensive bus coverage
- Robust error handling
- Integration with PowerFactory
"""

import sys, os, h5py, numpy as np
import time
from datetime import datetime

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT = "39 Bus New England System"
STUDY = "RMS_Simulation"
H5_DIR = os.path.join(os.getcwd(), "contingency_scenarios")

# Sensitivity analysis parameters
PERTURBATION_STEP_P_MW = 10.0      # Fixed MW step for P perturbation
PERTURBATION_STEP_Q_MVAR = 5.0     # Fixed MVAR step for Q perturbation
MIN_SENSITIVITY_THRESHOLD = 1e-6   # Minimum sensitivity to store
MAX_CONVERGENCE_ATTEMPTS = 3       # Max attempts if load flow fails

# ── Connect to PowerFactory ─────────────────────────────────────────────────
app = pf.GetApplication() or sys.exit("PF not running")
if hasattr(app, "ResetCalculation"):
    app.ResetCalculation()

print(f"🔬 VOLTAGE SENSITIVITY ANALYSIS MODULE")
print("="*60)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 H5 Directory: {H5_DIR}")
print(f"⚙️ P Perturbation: ±{PERTURBATION_STEP_P_MW} MW")
print(f"⚙️ Q Perturbation: ±{PERTURBATION_STEP_Q_MVAR} MVAR")
print()

# Helper functions
def has(o, a):
    try:
        return o.HasAttribute(a) if o else False
    except:
        return False

def get(o, a, d=np.nan):
    try:
        return o.GetAttribute(a) if has(o, a) else d
    except:
        return d

def safe_get_name(obj):
    try:
        return obj.loc_name if obj else "Unknown"
    except:
        return "Unknown"

def solve_load_flow_with_retry(max_attempts=MAX_CONVERGENCE_ATTEMPTS):
    """Solve load flow with retry mechanism"""
    
    comLdf = app.GetFromStudyCase("ComLdf")
    if not comLdf:
        return False, None, "Load flow object not found"
    
    # Configure for fast convergence
    comLdf.iopt_net = 0  # AC load flow
    comLdf.iopt_at = 0   # No automatic tap adjustment
    comLdf.errlf = 1e-3  # Relaxed tolerance for speed
    comLdf.maxiter = 30  # Reduced iterations for speed
    
    for attempt in range(max_attempts):
        ierr = comLdf.Execute()
        
        if ierr == 0:
            # Get bus voltages and angles
            buses = app.GetCalcRelevantObjects("*.ElmTerm")
            voltages = {}
            angles = {}
            
            for bus in buses:
                bus_name = safe_get_name(bus)
                voltages[bus_name] = get(bus, "m:u", np.nan)
                angles[bus_name] = get(bus, "m:phiu", np.nan)
            
            return True, {'voltages': voltages, 'angles': angles}, f"Converged in attempt {attempt+1}"
        
        # If failed, try slightly different tolerance
        comLdf.errlf = comLdf.errlf * 2  # Relax tolerance
    
    return False, None, f"Failed to converge after {max_attempts} attempts"

def read_scenario_data(h5_path):
    """Read active generators and loads from H5 file"""
    
    scenario_data = {
        'generators': {},
        'loads': {},
        'disconnected_elements': [],
        'scenario_info': {}
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Read scenario metadata
            if 'scenario_metadata' in f:
                meta = f['scenario_metadata']
                scenario_data['scenario_info'] = {
                    'scenario_id': meta['scenario_id'][()],
                    'contingency_type': meta['contingency_type'][()].decode(),
                    'description': meta['description'][()].decode()
                }
            
            # Read disconnection actions
            if 'disconnection_actions' in f:
                actions = f['disconnection_actions']
                if 'actions' in actions:
                    action_list = actions['actions'][:]
                    for action in action_list:
                        if isinstance(action, bytes):
                            action = action.decode()
                        scenario_data['disconnected_elements'].append(action)
            
            # Read power flow data to get active generators and loads
            if 'power_flow_data' in f:
                pf_data = f['power_flow_data']
                
                # Generator data
                if 'generation_data' in pf_data:
                    gen_data = pf_data['generation_data']
                    if 'generator_names' in gen_data:
                        gen_names = gen_data['generator_names'][:]
                        p_actual = gen_data['P_actual_MW'][:]
                        q_actual = gen_data['Q_actual_MVAR'][:]
                        
                        for i, name in enumerate(gen_names):
                            name = name.decode() if isinstance(name, bytes) else name
                            scenario_data['generators'][name] = {
                                'P_MW': p_actual[i],
                                'Q_MVAR': q_actual[i],
                                'active': True  # All generators in power_flow_data are active
                            }
                
                # Load data
                if 'load_data' in pf_data:
                    load_data = pf_data['load_data']
                    if 'load_names' in load_data:
                        load_names = load_data['load_names'][:]
                        p_actual = load_data['P_actual_MW'][:]
                        q_actual = load_data['Q_actual_MVAR'][:]
                        
                        for i, name in enumerate(load_names):
                            name = name.decode() if isinstance(name, bytes) else name
                            scenario_data['loads'][name] = {
                                'P_MW': p_actual[i],
                                'Q_MVAR': q_actual[i],
                                'active': True  # All loads in power_flow_data are active
                            }
    
    except Exception as e:
        print(f"      ❌ Error reading scenario data: {e}")
        return None
    
    return scenario_data

def apply_contingency_to_powerfactory(scenario_data):
    """Apply the contingency scenario to PowerFactory"""
    
    print(f"      🔄 Applying contingency to PowerFactory...")
    
    try:
        # Reset all elements to in-service first
        all_elements = (app.GetCalcRelevantObjects("*.ElmLne") + 
                       app.GetCalcRelevantObjects("*.ElmTr2,*.ElmXfr,*.ElmXfr3") +
                       app.GetCalcRelevantObjects("*.ElmSym") +
                       app.GetCalcRelevantObjects("*.ElmLod"))
        
        for element in all_elements:
            element.outserv = 0  # Put in service
        
        # Apply disconnections based on scenario description
        contingency_type = scenario_data['scenario_info']['contingency_type']
        description = scenario_data['scenario_info']['description']
        
        if contingency_type != 'BASE':
            # Parse description to find elements to disconnect
            # This is a simplified parser - you might need to enhance based on your naming convention
            
            for action in scenario_data['disconnected_elements']:
                if 'No elements disconnected' in action:
                    continue
                
                # Extract element type and name from action string
                # Expected format: "Element_Type Element_Name taken out of service"
                if 'taken out of service' in action:
                    parts = action.split(' taken out of service')[0]
                    
                    # Try to find and disconnect the element
                    if 'Line' in parts:
                        # Find line
                        line_name = parts.replace('Line ', '')
                        lines = app.GetCalcRelevantObjects("*.ElmLne")
                        for line in lines:
                            if safe_get_name(line) == line_name:
                                line.outserv = 1
                                print(f"         ✅ Disconnected line: {line_name}")
                                break
                    
                    elif 'Transformer' in parts:
                        # Find transformer
                        trafo_name = parts.replace('Transformer ', '')
                        trafos = app.GetCalcRelevantObjects("*.ElmTr2,*.ElmXfr,*.ElmXfr3")
                        for trafo in trafos:
                            if safe_get_name(trafo) == trafo_name:
                                trafo.outserv = 1
                                print(f"         ✅ Disconnected transformer: {trafo_name}")
                                break
                    
                    elif 'Generator' in parts:
                        # Find generator
                        gen_name = parts.replace('Generator ', '')
                        generators = app.GetCalcRelevantObjects("*.ElmSym")
                        for gen in generators:
                            if safe_get_name(gen) == gen_name:
                                gen.outserv = 1
                                print(f"         ✅ Disconnected generator: {gen_name}")
                                break
        
        # Solve base load flow to verify contingency is applied correctly
        success, _, message = solve_load_flow_with_retry()
        if success:
            print(f"         ✅ Contingency applied and load flow converged")
            return True
        else:
            print(f"         ❌ Load flow failed after applying contingency: {message}")
            return False
            
    except Exception as e:
        print(f"         ❌ Error applying contingency: {e}")
        return False

def perform_generator_sensitivity_analysis(generator_data):
    """Perform sensitivity analysis for all active generators"""
    
    print(f"      🔋 Analyzing generator sensitivities...")
    
    generator_sensitivities = {}
    
    # Get all PowerFactory generator objects
    pf_generators = app.GetCalcRelevantObjects("*.ElmSym")
    pf_gen_dict = {safe_get_name(gen): gen for gen in pf_generators if get(gen, "outserv", 0) == 0}
    
    # Get baseline voltages
    print(f"         📊 Obtaining baseline voltages...")
    success, baseline_results, _ = solve_load_flow_with_retry()
    if not success:
        print(f"         ❌ Failed to get baseline voltages")
        return {}
    
    baseline_voltages = baseline_results['voltages']
    
    for gen_name, gen_data in generator_data.items():
        if not gen_data['active']:
            continue
            
        if gen_name not in pf_gen_dict:
            print(f"         ⚠️ Generator {gen_name} not found in PowerFactory")
            continue
        
        pf_gen = pf_gen_dict[gen_name]
        base_p = gen_data['P_MW']
        base_q = gen_data['Q_MVAR']
        
        print(f"         🔋 Analyzing {gen_name} (P={base_p:.1f} MW, Q={base_q:.1f} MVAR)...")
        
        generator_sensitivities[gen_name] = {
            'P_sensitivity': {},
            'Q_sensitivity': {},
            'base_P_MW': base_p,
            'base_Q_MVAR': base_q
        }
        
        # P Sensitivity Analysis
        try:
            # Positive perturbation
            pf_gen.pgini = base_p + PERTURBATION_STEP_P_MW
            success_plus, results_plus, _ = solve_load_flow_with_retry()
            
            # Negative perturbation
            pf_gen.pgini = base_p - PERTURBATION_STEP_P_MW
            success_minus, results_minus, _ = solve_load_flow_with_retry()
            
            # Reset to baseline
            pf_gen.pgini = base_p
            
            if success_plus and success_minus:
                # Calculate sensitivities for all buses
                for bus_name in baseline_voltages.keys():
                    v_baseline = baseline_voltages[bus_name]
                    v_plus = results_plus['voltages'].get(bus_name, np.nan)
                    v_minus = results_minus['voltages'].get(bus_name, np.nan)
                    
                    if not (np.isnan(v_plus) or np.isnan(v_minus)):
                        # Central difference: dV/dP = (V+ - V-) / (2 * ΔP)
                        sensitivity = (v_plus - v_minus) / (2 * PERTURBATION_STEP_P_MW)
                        
                        if abs(sensitivity) > MIN_SENSITIVITY_THRESHOLD:
                            generator_sensitivities[gen_name]['P_sensitivity'][bus_name] = sensitivity
        
        except Exception as e:
            print(f"            ❌ Error in P sensitivity for {gen_name}: {e}")
        
        # Q Sensitivity Analysis
        try:
            # Positive perturbation
            pf_gen.qgini = base_q + PERTURBATION_STEP_Q_MVAR
            success_plus, results_plus, _ = solve_load_flow_with_retry()
            
            # Negative perturbation
            pf_gen.qgini = base_q - PERTURBATION_STEP_Q_MVAR
            success_minus, results_minus, _ = solve_load_flow_with_retry()
            
            # Reset to baseline
            pf_gen.qgini = base_q
            
            if success_plus and success_minus:
                # Calculate sensitivities for all buses
                for bus_name in baseline_voltages.keys():
                    v_baseline = baseline_voltages[bus_name]
                    v_plus = results_plus['voltages'].get(bus_name, np.nan)
                    v_minus = results_minus['voltages'].get(bus_name, np.nan)
                    
                    if not (np.isnan(v_plus) or np.isnan(v_minus)):
                        # Central difference: dV/dQ = (V+ - V-) / (2 * ΔQ)
                        sensitivity = (v_plus - v_minus) / (2 * PERTURBATION_STEP_Q_MVAR)
                        
                        if abs(sensitivity) > MIN_SENSITIVITY_THRESHOLD:
                            generator_sensitivities[gen_name]['Q_sensitivity'][bus_name] = sensitivity
        
        except Exception as e:
            print(f"            ❌ Error in Q sensitivity for {gen_name}: {e}")
    
    return generator_sensitivities

def perform_load_sensitivity_analysis(load_data):
    """Perform sensitivity analysis for all active loads"""
    
    print(f"      🏠 Analyzing load sensitivities...")
    
    load_sensitivities = {}
    
    # Get all PowerFactory load objects
    pf_loads = app.GetCalcRelevantObjects("*.ElmLod")
    pf_load_dict = {safe_get_name(load): load for load in pf_loads if get(load, "outserv", 0) == 0}
    
    # Get baseline voltages
    print(f"         📊 Obtaining baseline voltages...")
    success, baseline_results, _ = solve_load_flow_with_retry()
    if not success:
        print(f"         ❌ Failed to get baseline voltages")
        return {}
    
    baseline_voltages = baseline_results['voltages']
    
    for load_name, load_data_item in load_data.items():
        if not load_data_item['active']:
            continue
            
        if load_name not in pf_load_dict:
            print(f"         ⚠️ Load {load_name} not found in PowerFactory")
            continue
        
        pf_load = pf_load_dict[load_name]
        base_p = load_data_item['P_MW']
        base_q = load_data_item['Q_MVAR']
        
        print(f"         🏠 Analyzing {load_name} (P={base_p:.1f} MW, Q={base_q:.1f} MVAR)...")
        
        load_sensitivities[load_name] = {
            'P_sensitivity': {},
            'Q_sensitivity': {},
            'base_P_MW': base_p,
            'base_Q_MVAR': base_q
        }
        
        # P Sensitivity Analysis
        try:
            # Positive perturbation
            pf_load.plini = base_p + PERTURBATION_STEP_P_MW
            success_plus, results_plus, _ = solve_load_flow_with_retry()
            
            # Negative perturbation
            pf_load.plini = base_p - PERTURBATION_STEP_P_MW
            success_minus, results_minus, _ = solve_load_flow_with_retry()
            
            # Reset to baseline
            pf_load.plini = base_p
            
            if success_plus and success_minus:
                # Calculate sensitivities for all buses
                for bus_name in baseline_voltages.keys():
                    v_baseline = baseline_voltages[bus_name]
                    v_plus = results_plus['voltages'].get(bus_name, np.nan)
                    v_minus = results_minus['voltages'].get(bus_name, np.nan)
                    
                    if not (np.isnan(v_plus) or np.isnan(v_minus)):
                        # Central difference: dV/dP = (V+ - V-) / (2 * ΔP)
                        sensitivity = (v_plus - v_minus) / (2 * PERTURBATION_STEP_P_MW)
                        
                        if abs(sensitivity) > MIN_SENSITIVITY_THRESHOLD:
                            load_sensitivities[load_name]['P_sensitivity'][bus_name] = sensitivity
        
        except Exception as e:
            print(f"            ❌ Error in P sensitivity for {load_name}: {e}")
        
        # Q Sensitivity Analysis
        try:
            # Positive perturbation
            pf_load.qlini = base_q + PERTURBATION_STEP_Q_MVAR
            success_plus, results_plus, _ = solve_load_flow_with_retry()
            
            # Negative perturbation
            pf_load.qlini = base_q - PERTURBATION_STEP_Q_MVAR
            success_minus, results_minus, _ = solve_load_flow_with_retry()
            
            # Reset to baseline
            pf_load.qlini = base_q
            
            if success_plus and success_minus:
                # Calculate sensitivities for all buses
                for bus_name in baseline_voltages.keys():
                    v_baseline = baseline_voltages[bus_name]
                    v_plus = results_plus['voltages'].get(bus_name, np.nan)
                    v_minus = results_minus['voltages'].get(bus_name, np.nan)
                    
                    if not (np.isnan(v_plus) or np.isnan(v_minus)):
                        # Central difference: dV/dQ = (V+ - V-) / (2 * ΔQ)
                        sensitivity = (v_plus - v_minus) / (2 * PERTURBATION_STEP_Q_MVAR)
                        
                        if abs(sensitivity) > MIN_SENSITIVITY_THRESHOLD:
                            load_sensitivities[load_name]['Q_sensitivity'][bus_name] = sensitivity
        
        except Exception as e:
            print(f"            ❌ Error in Q sensitivity for {load_name}: {e}")
    
    return load_sensitivities

def save_sensitivity_results_to_h5(h5_path, generator_sensitivities, load_sensitivities, analysis_time):
    """Save sensitivity analysis results to H5 file"""
    
    print(f"      💾 Saving sensitivity results to H5 file...")
    
    try:
        with h5py.File(h5_path, 'a') as f:
            # Remove existing voltage_sensitivity group if it exists
            if 'voltage_sensitivity' in f:
                del f['voltage_sensitivity']
            
            # Create voltage sensitivity group
            vs_group = f.create_group('voltage_sensitivity')
            
            # Add metadata
            vs_group.create_dataset('analysis_timestamp', data=datetime.now().isoformat().encode())
            vs_group.create_dataset('analysis_time_seconds', data=analysis_time)
            vs_group.create_dataset('perturbation_step_P_MW', data=PERTURBATION_STEP_P_MW)
            vs_group.create_dataset('perturbation_step_Q_MVAR', data=PERTURBATION_STEP_Q_MVAR)
            vs_group.create_dataset('min_sensitivity_threshold', data=MIN_SENSITIVITY_THRESHOLD)
            
            # Save generator sensitivities
            if generator_sensitivities:
                gen_group = vs_group.create_group('generators')
                gen_group.create_dataset('num_generators_analyzed', data=len(generator_sensitivities))
                
                for gen_name, gen_sens in generator_sensitivities.items():
                    gen_subgroup = gen_group.create_group(gen_name.replace(' ', '_').replace('-', '_'))
                    
                    # Metadata
                    gen_subgroup.create_dataset('base_P_MW', data=gen_sens['base_P_MW'])
                    gen_subgroup.create_dataset('base_Q_MVAR', data=gen_sens['base_Q_MVAR'])
                    
                    # P sensitivities
                    if gen_sens['P_sensitivity']:
                        p_sens_group = gen_subgroup.create_group('P_sensitivity')
                        bus_names = list(gen_sens['P_sensitivity'].keys())
                        sensitivities = list(gen_sens['P_sensitivity'].values())
                        p_sens_group.create_dataset('bus_names', data=[name.encode() for name in bus_names])
                        p_sens_group.create_dataset('sensitivities', data=sensitivities)
                        p_sens_group.create_dataset('num_sensitive_buses', data=len(bus_names))
                    
                    # Q sensitivities
                    if gen_sens['Q_sensitivity']:
                        q_sens_group = gen_subgroup.create_group('Q_sensitivity')
                        bus_names = list(gen_sens['Q_sensitivity'].keys())
                        sensitivities = list(gen_sens['Q_sensitivity'].values())
                        q_sens_group.create_dataset('bus_names', data=[name.encode() for name in bus_names])
                        q_sens_group.create_dataset('sensitivities', data=sensitivities)
                        q_sens_group.create_dataset('num_sensitive_buses', data=len(bus_names))
            
            # Save load sensitivities
            if load_sensitivities:
                load_group = vs_group.create_group('loads')
                load_group.create_dataset('num_loads_analyzed', data=len(load_sensitivities))
                
                for load_name, load_sens in load_sensitivities.items():
                    load_subgroup = load_group.create_group(load_name.replace(' ', '_').replace('-', '_'))
                    
                    # Metadata
                    load_subgroup.create_dataset('base_P_MW', data=load_sens['base_P_MW'])
                    load_subgroup.create_dataset('base_Q_MVAR', data=load_sens['base_Q_MVAR'])
                    
                    # P sensitivities
                    if load_sens['P_sensitivity']:
                        p_sens_group = load_subgroup.create_group('P_sensitivity')
                        bus_names = list(load_sens['P_sensitivity'].keys())
                        sensitivities = list(load_sens['P_sensitivity'].values())
                        p_sens_group.create_dataset('bus_names', data=[name.encode() for name in bus_names])
                        p_sens_group.create_dataset('sensitivities', data=sensitivities)
                        p_sens_group.create_dataset('num_sensitive_buses', data=len(bus_names))
                    
                    # Q sensitivities
                    if load_sens['Q_sensitivity']:
                        q_sens_group = load_subgroup.create_group('Q_sensitivity')
                        bus_names = list(load_sens['Q_sensitivity'].keys())
                        sensitivities = list(load_sens['Q_sensitivity'].values())
                        q_sens_group.create_dataset('bus_names', data=[name.encode() for name in bus_names])
                        q_sens_group.create_dataset('sensitivities', data=sensitivities)
                        q_sens_group.create_dataset('num_sensitive_buses', data=len(bus_names))
            
            # Update analysis modules status
            if 'analysis_modules' in f:
                # Create new dataset for updated status
                modules_group = f['analysis_modules']
                if 'voltage_sensitivity_pending' in modules_group:
                    del modules_group['voltage_sensitivity_pending']
                modules_group.create_dataset('voltage_sensitivity', data=b'completed')
            
            print(f"         ✅ Sensitivity results saved successfully")
            
    except Exception as e:
        print(f"         ❌ Error saving sensitivity results: {e}")

def analyze_scenario_voltage_sensitivity(h5_path):
    """Perform complete voltage sensitivity analysis for a single scenario"""
    
    scenario_name = os.path.basename(h5_path)
    print(f"\n🎯 ANALYZING SCENARIO: {scenario_name}")
    
    start_time = time.time()
    
    # Step 1: Read scenario data
    print(f"   📖 Reading scenario data...")
    scenario_data = read_scenario_data(h5_path)
    if not scenario_data:
        print(f"   ❌ Failed to read scenario data")
        return False
    
    scenario_info = scenario_data['scenario_info']
    print(f"      📋 Scenario {scenario_info['scenario_id']}: {scenario_info['contingency_type']}")
    print(f"      📋 Description: {scenario_info['description']}")
    print(f"      🔋 Active generators: {len(scenario_data['generators'])}")
    print(f"      🏠 Active loads: {len(scenario_data['loads'])}")
    
    # Step 2: Connect to PowerFactory and activate project
    try:
        assert app.ActivateProject(PROJECT) == 0, "Project not found"
        study = next(c for c in app.GetProjectFolder("study").GetContents("*.IntCase")
                     if c.loc_name == STUDY)
        study.Activate()
        print(f"      ✅ PowerFactory project activated")
    except Exception as e:
        print(f"      ❌ Failed to activate PowerFactory project: {e}")
        return False
    
    # Step 3: Apply contingency to PowerFactory
    if not apply_contingency_to_powerfactory(scenario_data):
        print(f"      ❌ Failed to apply contingency")
        return False
    
    # Step 4: Perform generator sensitivity analysis
    generator_sensitivities = perform_generator_sensitivity_analysis(scenario_data['generators'])
    
    # Step 5: Perform load sensitivity analysis  
    load_sensitivities = perform_load_sensitivity_analysis(scenario_data['loads'])
    
    # Step 6: Save results to H5 file
    analysis_time = time.time() - start_time
    save_sensitivity_results_to_h5(h5_path, generator_sensitivities, load_sensitivities, analysis_time)
    
    # Step 7: Summary
    total_generator_sensitivities = sum(len(gen_sens['P_sensitivity']) + len(gen_sens['Q_sensitivity']) 
                                       for gen_sens in generator_sensitivities.values())
    total_load_sensitivities = sum(len(load_sens['P_sensitivity']) + len(load_sens['Q_sensitivity']) 
                                  for load_sens in load_sensitivities.values())
    
    print(f"   ✅ Voltage sensitivity analysis completed")
    print(f"      📊 Generator sensitivities calculated: {total_generator_sensitivities}")
    print(f"      📊 Load sensitivities calculated: {total_load_sensitivities}")
    print(f"      ⏱️ Analysis time: {analysis_time:.2f} seconds")
    
    return True

def main():
    """Main function to perform voltage sensitivity analysis on all scenarios"""
    
    # Find all H5 files
    h5_files = [os.path.join(H5_DIR, f) for f in os.listdir(H5_DIR) if f.endswith('.h5')]
    h5_files.sort()
    
    if not h5_files:
        print(f"❌ No H5 files found in {H5_DIR}")
        print(f"💡 Make sure you've run the contingency executor first")
        return
    
    print(f"📊 Found {len(h5_files)} H5 files: {', '.join(os.path.basename(f) for f in h5_files)}")
    
    successful_analyses = 0
    failed_analyses = 0
    
    # Process each scenario
    for h5_path in h5_files:
        try:
            success = analyze_scenario_voltage_sensitivity(h5_path)
            if success:
                successful_analyses += 1
            else:
                failed_analyses += 1
        except Exception as e:
            print(f"\n❌ Error analyzing {os.path.basename(h5_path)}: {e}")
            failed_analyses += 1
    
    # Final summary
    total_time = time.time() - start_time if 'start_time' in locals() else 0
    
    print(f"\n🎉 VOLTAGE SENSITIVITY ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 FINAL SUMMARY:")
    print(f"   🎯 Total scenarios processed: {len(h5_files)}")
    print(f"   ✅ Successful analyses: {successful_analyses}")
    print(f"   ❌ Failed analyses: {failed_analyses}")
    print(f"   📈 Success rate: {successful_analyses/len(h5_files)*100:.1f}%")
    print(f"   📁 H5 files updated: {successful_analyses}")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. ✅ Module 1: Contingency Generator - Complete")
    print(f"   2. ✅ Module 2: Contingency Executor - Complete") 
    print(f"   3. ✅ Module 3: Load Flow Data Collector - Complete")
    print(f"   4. ✅ Module 4: Voltage Sensitivity Analysis - Complete")
    print(f"   5. ⏳ Module 5: Y-Matrix Builder - Pending")
    
    print(f"\n💡 Each H5 file now contains:")
    print(f"   • Scenario metadata and load flow results")
    print(f"   • Comprehensive power flow data")
    print(f"   • Detailed system data with impedances")
    print(f"   • ✨ Voltage sensitivity analysis results")
    print(f"   • Y-matrix placeholder (ready for Module 5)")
    
    if successful_analyses > 0:
        print(f"\n📋 VOLTAGE SENSITIVITY DATA STRUCTURE:")
        print(f"   📁 voltage_sensitivity/")
        print(f"      📊 analysis_metadata (timestamp, parameters)")
        print(f"      📁 generators/")
        print(f"         📁 [generator_name]/")
        print(f"            📊 base_P_MW, base_Q_MVAR")
        print(f"            📁 P_sensitivity/ (bus_names, sensitivities)")
        print(f"            📁 Q_sensitivity/ (bus_names, sensitivities)")
        print(f"      📁 loads/")
        print(f"         📁 [load_name]/")
        print(f"            📊 base_P_MW, base_Q_MVAR")
        print(f"            📁 P_sensitivity/ (bus_names, sensitivities)")
        print(f"            📁 Q_sensitivity/ (bus_names, sensitivities)")

def create_sensitivity_summary_report():
    """Create a summary report of voltage sensitivity analysis results"""
    
    summary_file = os.path.join(H5_DIR, "voltage_sensitivity_summary_report.txt")
    
    with open(summary_file, 'w') as f:
        f.write("VOLTAGE SENSITIVITY ANALYSIS SUMMARY REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Project: {PROJECT}\n")
        f.write(f"Study Case: {STUDY}\n")
        f.write(f"Analysis Parameters:\n")
        f.write(f"  P Perturbation: ±{PERTURBATION_STEP_P_MW} MW\n")
        f.write(f"  Q Perturbation: ±{PERTURBATION_STEP_Q_MVAR} MVAR\n")
        f.write(f"  Min Sensitivity Threshold: {MIN_SENSITIVITY_THRESHOLD}\n\n")
        
        # Analyze each H5 file
        h5_files = [os.path.join(H5_DIR, f) for f in os.listdir(H5_DIR) if f.endswith('.h5')]
        h5_files.sort()
        
        f.write(f"Scenarios Analyzed: {len(h5_files)}\n")
        f.write("-" * 40 + "\n")
        
        for h5_path in h5_files:
            h5_file = os.path.basename(h5_path)
            
            try:
                with h5py.File(h5_path, 'r') as hf:
                    scenario_id = hf['scenario_metadata']['scenario_id'][()]
                    contingency_type = hf['scenario_metadata']['contingency_type'][()].decode()
                    description = hf['scenario_metadata']['description'][()].decode()
                    
                    f.write(f"\n{h5_file}:\n")
                    f.write(f"  Scenario ID: {scenario_id}\n")
                    f.write(f"  Type: {contingency_type}\n")
                    f.write(f"  Description: {description}\n")
                    
                    if 'voltage_sensitivity' in hf:
                        vs = hf['voltage_sensitivity']
                        analysis_time = vs['analysis_time_seconds'][()]
                        f.write(f"  Analysis Time: {analysis_time:.2f} seconds\n")
                        
                        if 'generators' in vs:
                            num_gens = vs['generators']['num_generators_analyzed'][()]
                            f.write(f"  Generators Analyzed: {num_gens}\n")
                        
                        if 'loads' in vs:
                            num_loads = vs['loads']['num_loads_analyzed'][()]
                            f.write(f"  Loads Analyzed: {num_loads}\n")
                        
                        # Count total sensitivities
                        total_sensitivities = 0
                        if 'generators' in vs:
                            for gen_name in vs['generators'].keys():
                                if gen_name == 'num_generators_analyzed':
                                    continue
                                gen_group = vs['generators'][gen_name]
                                if 'P_sensitivity' in gen_group:
                                    total_sensitivities += gen_group['P_sensitivity']['num_sensitive_buses'][()]
                                if 'Q_sensitivity' in gen_group:
                                    total_sensitivities += gen_group['Q_sensitivity']['num_sensitive_buses'][()]
                        
                        if 'loads' in vs:
                            for load_name in vs['loads'].keys():
                                if load_name == 'num_loads_analyzed':
                                    continue
                                load_group = vs['loads'][load_name]
                                if 'P_sensitivity' in load_group:
                                    total_sensitivities += load_group['P_sensitivity']['num_sensitive_buses'][()]
                                if 'Q_sensitivity' in load_group:
                                    total_sensitivities += load_group['Q_sensitivity']['num_sensitive_buses'][()]
                        
                        f.write(f"  Total Sensitivities: {total_sensitivities}\n")
                    else:
                        f.write(f"  Status: Voltage sensitivity analysis not completed\n")
                        
            except Exception as e:
                f.write(f"\n{h5_file}: Error reading file - {e}\n")
        
        f.write(f"\nAnalysis Method:\n")
        f.write(f"- Central finite difference method\n")
        f.write(f"- Systematic perturbation of P and Q for all active generators and loads\n")
        f.write(f"- Voltage sensitivity measured on all buses in the system\n")
        f.write(f"- Results stored in HDF5 format for efficient access\n")
    
    print(f"📄 Voltage sensitivity summary report saved: {os.path.basename(summary_file)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    create_sensitivity_summary_report()