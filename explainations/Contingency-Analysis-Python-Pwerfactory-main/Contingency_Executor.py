# contingency_executor_with_y_matrix.py - 2025-08-02
"""
Enhanced Contingency Executor with integrated Y-Matrix Builder
This version includes Y-Matrix construction directly in the contingency execution loop.

Execution sequence for each contingency:
1. Apply contingency (disconnect elements)
2. Solve load flow
3. Collect comprehensive power flow data
4. Collect detailed system data (impedances)
5. Perform voltage sensitivity analysis (if load flow converged)
6. Build Y-matrix (regardless of load flow convergence, but mark status)
7. Save all data to H5 file
8. Restore elements to original state

Key features:
- Y-Matrix Builder integrated into the main execution loop
- Handles both converged and non-converged load flow cases
- Maintains modular design with proper error handling
- Updates H5 file structure with Y-matrix data
"""

import sys, os, csv, h5py, numpy as np
import time
from datetime import datetime

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

# Import required modules
try:
    import Voltage_Sensitivity_Analysis as vs_analyzer
    VOLTAGE_SENSITIVITY_AVAILABLE = True
    print(f"✅ Voltage Sensitivity Analysis module imported")
except ImportError as e:
    VOLTAGE_SENSITIVITY_AVAILABLE = False
    print(f"⚠️ Voltage Sensitivity Analysis module not available: {e}")

try:
    import Y_Matrix_Builder as y_matrix_builder
    Y_MATRIX_BUILDER_AVAILABLE = True
    print(f"✅ Y-Matrix Builder module imported")
except ImportError as e:
    Y_MATRIX_BUILDER_AVAILABLE = False
    print(f"⚠️ Y-Matrix Builder module not available: {e}")

# ── Project settings ────────────────────────────────────────────────────────
PROJECT = "39 Bus New England System"
STUDY = "RMS_Simulation"
SBASE_MVA = 100.0
OUT_DIR = os.path.join(os.getcwd(), "contingency_scenarios")

# ── Input Configuration ─────────────────────────────────────────────────────
CONTINGENCY_CSV = os.path.join("contingency_out", "contingency_scenarios_20250803_114018.csv")

# Configuration flags
ENABLE_VOLTAGE_SENSITIVITY = True
ENABLE_Y_MATRIX_BUILDING = True
VS_PERTURBATION_STEP_P_MW = 10.0
VS_PERTURBATION_STEP_Q_MVAR = 5.0
MIN_SENSITIVITY_THRESHOLD = 1e-6

# ── Connect to PowerFactory ─────────────────────────────────────────────────
app = pf.GetApplication() or sys.exit("PF not running")
if hasattr(app, "ResetCalculation"):
    app.ResetCalculation()
assert app.ActivateProject(PROJECT) == 0, "Project not found"
study = next(c for c in app.GetProjectFolder("study").GetContents("*.IntCase")
             if c.loc_name == STUDY)
study.Activate()

print(f"⚡ INTEGRATED CONTINGENCY EXECUTOR WITH Y-MATRIX BUILDER")
print("="*70)
print(f"✅ {PROJECT} | {STUDY}")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📄 CSV file: {CONTINGENCY_CSV}")
print(f"📁 Output directory: {OUT_DIR}")
print()

# Helper functions (same as before)
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

def safe_get_class(obj):
    try:
        return obj.GetClassName() if obj else "Unknown"
    except:
        return "Unknown"

def as_bus(o):
    """Safely get bus object"""
    try:
        if not o:
            return None
        return o.cterm if has(o, "cterm") else o
    except:
        return o

def term(obj):
    """Get the terminal/bus from various PowerFactory objects"""
    if obj.GetClassName().startswith("ElmTr"):
        for f in ("bushv", "buslv", "bus1", "bus2", "cterm"):
            if has(obj, f):
                ptr = get(obj, f)
                if ptr:
                    return as_bus(ptr)
    else:
        for f in ("cterm", "sbus", "bus1", "bus", "bushv", "buslv"):
            if has(obj, f):
                ptr = get(obj, f) if f != "cterm" else obj
                if ptr:
                    return as_bus(ptr)
    return None

# ── Setup modules ──────────────────────────────────────────────────────────
def setup_data_collection_module():
    """Import and setup the data collection module"""
    try:
        import Load_Flow_Data_Collector as lf_collector
        import sys
        
        lf_collector_module = sys.modules['Load_Flow_Data_Collector']
        lf_collector_module.app = app
        lf_collector_module.SBASE_MVA = SBASE_MVA
        
        print(f"✅ Data collection module imported and configured")
        return lf_collector
    except ImportError as e:
        print(f"❌ Could not import Load_Flow_Data_Collector: {e}")
        return None
    except Exception as e:
        print(f"❌ Error setting up data collection module: {e}")
        return None

def setup_y_matrix_module():
    """Setup the Y-Matrix Builder module with proper app reference"""
    if not Y_MATRIX_BUILDER_AVAILABLE:
        return None
    
    try:
        # Configure Y-Matrix Builder module with our PowerFactory app
        import sys
        y_matrix_module = sys.modules['Y_Matrix_Builder']
        
        # The Y-Matrix Builder doesn't need the app object since it reads from H5 files
        # But we can pass some configuration if needed
        y_matrix_module.SBASE_MVA = SBASE_MVA
        
        print(f"✅ Y-Matrix Builder module configured")
        return y_matrix_builder
    except Exception as e:
        print(f"❌ Error setting up Y-Matrix Builder module: {e}")
        return None

# ── Enhanced Power Flow Data Collection (same as before) ──────────────────
def collect_comprehensive_power_flow_data():
    """Collect comprehensive actual MW/MVAR values from PowerFactory after load flow"""
    
    print(f"      📊 Collecting comprehensive power flow data...")
    
    power_data = {
        'system_totals': {
            'total_generation_MW': 0.0,
            'total_generation_MVAR': 0.0,
            'total_load_MW': 0.0,
            'total_load_MVAR': 0.0,
            'total_losses_MW': 0.0,
            'total_losses_MVAR': 0.0,
            'power_balance_MW': 0.0,
            'power_balance_MVAR': 0.0,
            'system_frequency_Hz': 50.0
        },
        'generation_data': {},
        'load_data': {},
        'line_data': {},
        'transformer_data': {},
        'collection_method': 'element_by_element'
    }
    
    try:
        # Generator data collection
        print(f"         🔋 Collecting generator data...")
        generators = [gen for gen in app.GetCalcRelevantObjects("*.ElmSym") 
                     if get(gen, "outserv", 0) == 0]
        
        for gen in generators:
            gen_name = safe_get_name(gen)
            P_actual = get(gen, "m:Psum:bus1", np.nan)
            Q_actual = get(gen, "m:Qsum:bus1", np.nan)
            P_setpoint = get(gen, "pgini", np.nan)
            Q_setpoint = get(gen, "qgini", np.nan)
            V_setpoint = get(gen, "uset", np.nan)
            P_max = get(gen, "Pmax", np.nan)
            P_min = get(gen, "Pmin", np.nan)
            Q_max = get(gen, "Qmax", np.nan)
            Q_min = get(gen, "Qmin", np.nan)
            
            gen_bus = term(gen)
            V_terminal = get(gen_bus, "m:u", np.nan) if gen_bus else np.nan
            
            power_data['generation_data'][gen_name] = {
                'P_actual_MW': P_actual if not np.isnan(P_actual) else 0.0,
                'Q_actual_MVAR': Q_actual if not np.isnan(Q_actual) else 0.0,
                'P_setpoint_MW': P_setpoint if not np.isnan(P_setpoint) else 0.0,
                'Q_setpoint_MVAR': Q_setpoint if not np.isnan(Q_setpoint) else 0.0,
                'V_setpoint_pu': V_setpoint if not np.isnan(V_setpoint) else 1.0,
                'V_terminal_pu': V_terminal if not np.isnan(V_terminal) else np.nan,
                'P_max_MW': P_max if not np.isnan(P_max) else np.nan,
                'P_min_MW': P_min if not np.isnan(P_min) else np.nan,
                'Q_max_MVAR': Q_max if not np.isnan(Q_max) else np.nan,
                'Q_min_MVAR': Q_min if not np.isnan(Q_min) else np.nan,
                'in_service': True
            }
            
            if not np.isnan(P_actual):
                power_data['system_totals']['total_generation_MW'] += P_actual
            if not np.isnan(Q_actual):
                power_data['system_totals']['total_generation_MVAR'] += Q_actual
        
        # Load data collection
        print(f"         🏠 Collecting load data...")
        loads = [load for load in app.GetCalcRelevantObjects("*.ElmLod") 
                if get(load, "outserv", 0) == 0]
        
        for load in loads:
            load_name = safe_get_name(load)
            P_load = get(load, "plini", np.nan)
            Q_load = get(load, "qlini", np.nan)
            P_actual = get(load, "m:Psum:bus1", np.nan)
            Q_actual = get(load, "m:Qsum:bus1", np.nan)
            
            load_bus = term(load)
            V_bus = get(load_bus, "m:u", np.nan) if load_bus else np.nan
            
            power_data['load_data'][load_name] = {
                'P_nominal_MW': P_load if not np.isnan(P_load) else 0.0,
                'Q_nominal_MVAR': Q_load if not np.isnan(Q_load) else 0.0,
                'P_actual_MW': P_actual if not np.isnan(P_actual) else P_load if not np.isnan(P_load) else 0.0,
                'Q_actual_MVAR': Q_actual if not np.isnan(Q_actual) else Q_load if not np.isnan(Q_load) else 0.0,
                'V_bus_pu': V_bus if not np.isnan(V_bus) else np.nan,
                'load_type': get(load, "lodtyp", 0),
                'in_service': True
            }
            
            P_total = P_actual if not np.isnan(P_actual) else P_load if not np.isnan(P_load) else 0.0
            Q_total = Q_actual if not np.isnan(Q_actual) else Q_load if not np.isnan(Q_load) else 0.0
            
            power_data['system_totals']['total_load_MW'] += P_total
            power_data['system_totals']['total_load_MVAR'] += Q_total
        
        # Line data collection (abbreviated for space)
        print(f"         ⚡ Collecting line data...")
        lines = [line for line in app.GetCalcRelevantObjects("*.ElmLne") 
                if get(line, "outserv", 0) == 0]
        
        for line in lines:
            line_name = safe_get_name(line)
            P_from = get(line, "m:Psum:bus1", np.nan)
            Q_from = get(line, "m:Qsum:bus1", np.nan)
            P_to = get(line, "m:Psum:bus2", np.nan)
            Q_to = get(line, "m:Qsum:bus2", np.nan)
            P_loss = get(line, "c:Ploss", np.nan)
            Q_loss = get(line, "c:Qloss", np.nan)
            loading = get(line, "c:loading", np.nan)
            current = get(line, "m:I:bus1", np.nan)
            
            power_data['line_data'][line_name] = {
                'P_from_MW': P_from if not np.isnan(P_from) else 0.0,
                'Q_from_MVAR': Q_from if not np.isnan(Q_from) else 0.0,
                'P_to_MW': P_to if not np.isnan(P_to) else 0.0,
                'Q_to_MVAR': Q_to if not np.isnan(Q_to) else 0.0,
                'P_loss_MW': P_loss if not np.isnan(P_loss) else 0.0,
                'Q_loss_MVAR': Q_loss if not np.isnan(Q_loss) else 0.0,
                'loading_percent': loading if not np.isnan(loading) else 0.0,
                'current_A': current if not np.isnan(current) else 0.0,
                'in_service': True
            }
            
            if not np.isnan(P_loss):
                power_data['system_totals']['total_losses_MW'] += P_loss
            if not np.isnan(Q_loss):
                power_data['system_totals']['total_losses_MVAR'] += Q_loss
        
        # Transformer data collection (abbreviated for space)
        print(f"         🔄 Collecting transformer data...")
        transformers = [trafo for trafo in app.GetCalcRelevantObjects("*.ElmTr2,*.ElmXfr,*.ElmXfr3") 
                       if get(trafo, "outserv", 0) == 0]
        
        for trafo in transformers:
            trafo_name = safe_get_name(trafo)
            if has(trafo, "bushv") and has(trafo, "buslv"):
                P_hv = get(trafo, "m:Psum:bushv", np.nan)
                Q_hv = get(trafo, "m:Qsum:bushv", np.nan)
                P_lv = get(trafo, "m:Psum:buslv", np.nan)
                Q_lv = get(trafo, "m:Qsum:buslv", np.nan)
            else:
                P_hv = get(trafo, "m:Psum:bus1", np.nan)
                Q_hv = get(trafo, "m:Qsum:bus1", np.nan)
                P_lv = get(trafo, "m:Psum:bus2", np.nan)
                Q_lv = get(trafo, "m:Qsum:bus2", np.nan)
            
            P_loss = get(trafo, "c:Ploss", np.nan)
            Q_loss = get(trafo, "c:Qloss", np.nan)
            loading = get(trafo, "c:loading", np.nan)
            tap_pos = get(trafo, "nntap", np.nan)
            
            power_data['transformer_data'][trafo_name] = {
                'P_hv_MW': P_hv if not np.isnan(P_hv) else 0.0,
                'Q_hv_MVAR': Q_hv if not np.isnan(Q_hv) else 0.0,
                'P_lv_MW': P_lv if not np.isnan(P_lv) else 0.0,
                'Q_lv_MVAR': Q_lv if not np.isnan(Q_lv) else 0.0,
                'P_loss_MW': P_loss if not np.isnan(P_loss) else 0.0,
                'Q_loss_MVAR': Q_loss if not np.isnan(Q_loss) else 0.0,
                'loading_percent': loading if not np.isnan(loading) else 0.0,
                'tap_position': tap_pos if not np.isnan(tap_pos) else 0,
                'in_service': True
            }
            
            if not np.isnan(P_loss):
                power_data['system_totals']['total_losses_MW'] += P_loss
            if not np.isnan(Q_loss):
                power_data['system_totals']['total_losses_MVAR'] += Q_loss
        
        # Calculate power balance
        gen_total = power_data['system_totals']['total_generation_MW']
        load_total = power_data['system_totals']['total_load_MW']
        loss_total = power_data['system_totals']['total_losses_MW']
        
        power_data['system_totals']['power_balance_MW'] = gen_total - load_total - loss_total
        
        gen_Q_total = power_data['system_totals']['total_generation_MVAR']
        load_Q_total = power_data['system_totals']['total_load_MVAR']
        loss_Q_total = power_data['system_totals']['total_losses_MVAR']
        
        power_data['system_totals']['power_balance_MVAR'] = gen_Q_total - load_Q_total - loss_Q_total
        
        study_case = app.GetActiveStudyCase()
        if study_case:
            power_data['system_totals']['system_frequency_Hz'] = get(study_case, "SetFrq", 50.0)
        
        print(f"         ✅ Power flow data collected successfully")
        print(f"         📊 Generation: {gen_total:.1f} MW, {gen_Q_total:.1f} MVAR")
        print(f"         📊 Load: {load_total:.1f} MW, {load_Q_total:.1f} MVAR")
        print(f"         📊 Losses: {loss_total:.1f} MW, {loss_Q_total:.1f} MVAR")
        print(f"         📊 Balance: {power_data['system_totals']['power_balance_MW']:.3f} MW, {power_data['system_totals']['power_balance_MVAR']:.3f} MVAR")
        
        power_data['collection_success'] = True
        
    except Exception as e:
        print(f"         ❌ Error collecting power flow data: {e}")
        power_data['collection_success'] = False
        power_data['error'] = str(e)
    
    return power_data

# ── Read contingency scenarios (same as before) ──────────────────────────
def read_contingency_scenarios(csv_path):
    """Read contingency scenarios from CSV file"""
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return None
    
    scenarios = []
    print(f"📖 READING CONTINGENCY SCENARIOS...")
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            scenario = {
                'scenario_id': int(row['scenario_id']),
                'contingency_type': row['contingency_type'],
                'outage_type': row['outage_type'],
                'description': row['description'],
                'severity': row['severity'],
                'num_elements_out': int(row['num_elements_out']),
                'elements_to_disconnect': []
            }
            
            if scenario['num_elements_out'] > 0:
                if row['element1_type'] != 'none':
                    element1 = {
                        'type': row['element1_type'],
                        'name': row['element1_name'],
                        'class': row['element1_class'],
                        'index': int(row['element1_index']),
                        'rating_capacity': float(row['element1_rating_capacity'])
                    }
                    scenario['elements_to_disconnect'].append(element1)
                
                if scenario['num_elements_out'] > 1 and row['element2_type'] != 'none':
                    element2 = {
                        'type': row['element2_type'],
                        'name': row['element2_name'],
                        'class': row['element2_class'],
                        'index': int(row['element2_index']),
                        'rating_capacity': float(row['element2_rating_capacity'])
                    }
                    scenario['elements_to_disconnect'].append(element2)
            
            scenarios.append(scenario)
    
    print(f"   ✅ Read {len(scenarios)} scenarios")
    base_cases = sum(1 for s in scenarios if s['contingency_type'] == 'BASE')
    n1_cases = sum(1 for s in scenarios if s['contingency_type'] == 'N-1')
    n2_cases = sum(1 for s in scenarios if s['contingency_type'] == 'N-2')
    print(f"   📊 Base cases: {base_cases}")
    print(f"   📊 N-1 contingencies: {n1_cases}")
    print(f"   📊 N-2 contingencies: {n2_cases}")
    
    return scenarios

# ── PowerFactory element management (same as before) ─────────────────────
def get_powerfactory_elements():
    """Get all PowerFactory elements that might need to be disconnected"""
    
    print(f"🔍 CATALOGING POWERFACTORY ELEMENTS...")
    
    elements_catalog = {
        'lines': {},
        'transformers': {},
        'generators': {},
        'loads': {}
    }
    
    lines = app.GetCalcRelevantObjects("*.ElmLne")
    for line in lines:
        name = safe_get_name(line)
        elements_catalog['lines'][name] = line
    
    transformers = app.GetCalcRelevantObjects("*.ElmTr2,*.ElmXfr,*.ElmXfr3")
    for trafo in transformers:
        name = safe_get_name(trafo)
        elements_catalog['transformers'][name] = trafo
    
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    for gen in generators:
        name = safe_get_name(gen)
        elements_catalog['generators'][name] = gen
    
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    for load in loads:
        name = safe_get_name(load)
        elements_catalog['loads'][name] = load
    
    print(f"   🔌 Lines: {len(elements_catalog['lines'])}")
    print(f"   🔄 Transformers: {len(elements_catalog['transformers'])}")
    print(f"   🔋 Generators: {len(elements_catalog['generators'])}")
    print(f"   🏠 Loads: {len(elements_catalog['loads'])}")
    
    return elements_catalog

def disconnect_element(element_obj, element_info):
    """Disconnect (take out of service) a PowerFactory element"""
    if not element_obj:
        return False, f"Element object not found"
    
    try:
        element_obj.outserv = 1
        action = f"{element_info['type'].title()} {element_info['name']} taken out of service"
        return True, action
    except Exception as e:
        return False, f"Error disconnecting {element_info['name']}: {e}"

def reconnect_element(element_obj, element_info):
    """Reconnect (put back in service) a PowerFactory element"""
    if not element_obj:
        return False, f"Element object not found"
    
    try:
        element_obj.outserv = 0
        return True, f"{element_info['name']} restored to service"
    except Exception as e:
        return False, f"Error reconnecting {element_info['name']}: {e}"

# ── Load flow execution (same as before) ─────────────────────────────────
def solve_load_flow():
    """Solve load flow and check convergence"""
    
    comLdf = app.GetFromStudyCase("ComLdf")
    if not comLdf:
        return False, None, "Load flow calculation object not found"
    
    comLdf.iopt_net = 0
    comLdf.iopt_at = 0
    comLdf.errlf = 1e-4
    comLdf.maxiter = 50
    
    start_time = time.time()
    ierr = comLdf.Execute()
    execution_time = time.time() - start_time
    
    if ierr == 0:
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        bus_voltages = []
        bus_names = []
        bus_angles = []
        
        for bus in buses:
            bus_names.append(safe_get_name(bus))
            bus_voltages.append(get(bus, "m:u", np.nan))
            bus_angles.append(get(bus, "m:phiu", np.nan))
        
        system_state = {
            'convergence': True,
            'execution_time': execution_time,
            'bus_names': bus_names,
            'bus_voltages': np.array(bus_voltages),
            'bus_angles': np.array(bus_angles),
            'min_voltage': np.nanmin(bus_voltages),
            'max_voltage': np.nanmax(bus_voltages),
            'num_buses': len(bus_voltages),
            'iterations': get(comLdf, "iiter", 0)
        }
        
        return True, system_state, "Load flow converged"
    else:
        system_state = {
            'convergence': False,
            'execution_time': execution_time,
            'error_code': ierr,
            'iterations': get(comLdf, "iiter", 0)
        }
        
        return False, system_state, f"Load flow failed to converge (error: {ierr})"

# ── Voltage sensitivity analysis integration (same as before) ────────────
def perform_integrated_voltage_sensitivity(scenario_data, h5_path):
    """Perform voltage sensitivity analysis as part of contingency execution"""
    
    if not ENABLE_VOLTAGE_SENSITIVITY or not VOLTAGE_SENSITIVITY_AVAILABLE:
        print(f"      ⏭️ Voltage sensitivity analysis skipped")
        return {'status': 'skipped', 'reason': 'disabled or module unavailable'}
    
    print(f"      🔬 Performing integrated voltage sensitivity analysis...")
    start_time = time.time()
    
    try:
        active_generators = {}
        active_loads = {}
        
        if 'power_flow_data' in scenario_data and scenario_data['power_flow_data'].get('collection_success', False):
            pf_data = scenario_data['power_flow_data']
            
            if 'generation_data' in pf_data:
                for gen_name, gen_info in pf_data['generation_data'].items():
                    if gen_info['in_service']:
                        active_generators[gen_name] = {
                            'P_MW': gen_info['P_actual_MW'],
                            'Q_MVAR': gen_info['Q_actual_MVAR'],
                            'active': True
                        }
            
            if 'load_data' in pf_data:
                for load_name, load_info in pf_data['load_data'].items():
                    if load_info['in_service']:
                        active_loads[load_name] = {
                            'P_MW': load_info['P_actual_MW'],
                            'Q_MVAR': load_info['Q_actual_MVAR'],
                            'active': True
                        }
        
        print(f"         🔋 Active generators: {len(active_generators)}")
        print(f"         🏠 Active loads: {len(active_loads)}")
        
        generator_sensitivities = vs_analyzer.perform_generator_sensitivity_analysis(active_generators)
        load_sensitivities = vs_analyzer.perform_load_sensitivity_analysis(active_loads)
        
        analysis_time = time.time() - start_time
        vs_analyzer.save_sensitivity_results_to_h5(h5_path, generator_sensitivities, load_sensitivities, analysis_time)
        
        total_generator_sensitivities = sum(len(gen_sens['P_sensitivity']) + len(gen_sens['Q_sensitivity']) 
                                           for gen_sens in generator_sensitivities.values())
        total_load_sensitivities = sum(len(load_sens['P_sensitivity']) + len(load_sens['Q_sensitivity']) 
                                      for load_sens in load_sensitivities.values())
        
        result = {
            'status': 'completed',
            'analysis_time_seconds': analysis_time,
            'generators_analyzed': len(active_generators),
            'loads_analyzed': len(active_loads),
            'total_generator_sensitivities': total_generator_sensitivities,
            'total_load_sensitivities': total_load_sensitivities,
            'total_sensitivities': total_generator_sensitivities + total_load_sensitivities
        }
        
        print(f"         ✅ Voltage sensitivity completed in {analysis_time:.2f} seconds")
        print(f"         📊 Total sensitivities: {result['total_sensitivities']}")
        
        return result
        
    except Exception as e:
        analysis_time = time.time() - start_time
        print(f"         ❌ Voltage sensitivity analysis failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'analysis_time_seconds': analysis_time
        }

# ── ★ NEW: Y-Matrix Builder Integration ★ ──────────────────────────────────
def perform_integrated_y_matrix_building(scenario_data, h5_path, load_flow_converged):
    """
    Perform Y-Matrix building as part of contingency execution
    
    Args:
        scenario_data: Current scenario data
        h5_path: Path to H5 file
        load_flow_converged: Boolean indicating if load flow converged
    
    Returns:
        dict: Y-Matrix building results
    """
    
    if not ENABLE_Y_MATRIX_BUILDING or not Y_MATRIX_BUILDER_AVAILABLE:
        print(f"      ⏭️ Y-Matrix building skipped")
        return {'status': 'skipped', 'reason': 'disabled or module unavailable'}
    
    print(f"      🔧 Performing integrated Y-Matrix building...")
    start_time = time.time()
    
    try:
        # Check if we have the necessary data in the H5 file
        # The detailed system data should be available regardless of load flow convergence
        
        # Read scenario data from H5 file using Y-Matrix Builder functions
        print(f"         📖 Reading scenario data from H5 file...")
        scenario_h5_data = y_matrix_builder.read_scenario_h5_data(h5_path)
        
        if not scenario_h5_data:
            print(f"         ❌ Failed to read scenario data from H5 file")
            return {
                'status': 'failed',
                'error': 'Failed to read scenario data from H5 file',
                'analysis_time_seconds': time.time() - start_time,
                'load_flow_status': 'converged' if load_flow_converged else 'failed'
            }
        
        # Build Y-matrix from impedance data
        print(f"         🔧 Constructing Y-matrix...")
        y_matrix_data = y_matrix_builder.build_y_matrix(scenario_h5_data)
        
        if not y_matrix_data:
            print(f"         ❌ Failed to build Y-matrix")
            return {
                'status': 'failed',
                'error': 'Y-matrix construction failed',
                'analysis_time_seconds': time.time() - start_time,
                'load_flow_status': 'converged' if load_flow_converged else 'failed'
            }
        
        # Save Y-matrix to H5 file
        construction_time = time.time() - start_time
        print(f"         💾 Saving Y-matrix to H5 file...")
        y_matrix_builder.save_y_matrix_to_h5(h5_path, y_matrix_data, construction_time)
        
        # Validate the constructed Y-matrix
        validation_results = y_matrix_builder.validate_y_matrix(y_matrix_data)
        
        # Prepare results
        properties = y_matrix_data['matrix_properties']
        elements_added = y_matrix_data['elements_added']
        
        result = {
            'status': 'completed',
            'construction_time_seconds': construction_time,
            'matrix_size': y_matrix_data['matrix_size'],
            'elements_added': elements_added,
            'nnz': properties.get('nnz', 0),
            'density': properties.get('density', 0),
            'condition_number': properties.get('condition_number', np.nan),
            'is_hermitian': properties.get('is_hermitian', False),
            'zero_diagonal_elements': properties.get('zero_diagonal_elements', 0),
            'validation_passed': validation_results['is_valid'],
            'warnings': validation_results['warnings'],
            'load_flow_status': 'converged' if load_flow_converged else 'failed'
        }
        
        total_elements = sum(elements_added[key] for key in elements_added if key != 'skipped')
        
        print(f"         ✅ Y-matrix construction completed in {construction_time:.2f} seconds")
        print(f"         📊 Matrix size: {y_matrix_data['matrix_size']} x {y_matrix_data['matrix_size']}")
        print(f"         📊 Elements added: {total_elements} ({elements_added['lines']}L + {elements_added['transformers']}T + "
              f"{elements_added['generators']}G + {elements_added['loads']}Ld + {elements_added['shunts']}S)")
        print(f"         📊 Non-zero elements: {properties.get('nnz', 0)}")
        print(f"         📊 Density: {properties.get('density', 0):.4f}")
        
        if not load_flow_converged:
            print(f"         ⚠️ Note: Y-matrix built despite load flow non-convergence")
        
        # Print warnings if any
        if validation_results['warnings']:
            for warning in validation_results['warnings'][:3]:  # Show first 3 warnings
                print(f"         ⚠️ {warning}")
        
        return result
        
    except Exception as e:
        construction_time = time.time() - start_time
        print(f"         ❌ Y-matrix building failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'construction_time_seconds': construction_time,
            'load_flow_status': 'converged' if load_flow_converged else 'failed'
        }

# ── Enhanced contingency execution with Y-Matrix integration ──────────────
def execute_contingency_scenario(scenario, elements_catalog, data_collector, y_matrix_module):
    """
    Execute a single contingency scenario with integrated Y-Matrix building
    
    Enhanced execution sequence:
    1. Apply contingency (disconnect elements)
    2. Solve load flow
    3. Collect comprehensive power flow data
    4. Collect detailed system data (impedances)
    5. Perform voltage sensitivity analysis (if load flow converged)
    6. ★ Build Y-matrix (regardless of load flow convergence) ★
    7. Save all data to H5 file
    8. Restore elements to original state
    """
    
    scenario_id = scenario['scenario_id']
    description = scenario['description']
    
    print(f"\n🎯 EXECUTING SCENARIO {scenario_id}:")
    print(f"   📋 Description: {description}")
    print(f"   🏷️ Type: {scenario['contingency_type']}")
    
    disconnected_elements = []
    disconnection_actions = []
    
    # Step 1: Disconnect elements if not base case
    if scenario['contingency_type'] != 'BASE':
        print(f"   🔌 Disconnecting elements...")
        
        for element_info in scenario['elements_to_disconnect']:
            element_type = element_info['type']
            element_name = element_info['name']
            
            element_obj = None
            if element_type == 'line':
                element_obj = elements_catalog['lines'].get(element_name)
            elif element_type == 'transformer':
                element_obj = elements_catalog['transformers'].get(element_name)
            elif element_type == 'generator':
                element_obj = elements_catalog['generators'].get(element_name)
            elif element_type == 'load':
                element_obj = elements_catalog['loads'].get(element_name)
            
            if element_obj:
                success, action = disconnect_element(element_obj, element_info)
                if success:
                    disconnected_elements.append((element_obj, element_info))
                    disconnection_actions.append(action)
                    print(f"      ✅ {action}")
                else:
                    print(f"      ❌ {action}")
            else:
                print(f"      ❌ Element not found: {element_name} ({element_type})")
    else:
        print(f"   📊 Base case - no elements to disconnect")
    
    # Step 2: Solve load flow
    print(f"   ⚡ Solving load flow...")
    lf_success, system_state, lf_message = solve_load_flow()
    
    if lf_success:
        print(f"      ✅ {lf_message}")
        print(f"      📊 Voltage range: {system_state['min_voltage']:.3f} - {system_state['max_voltage']:.3f} p.u.")
        print(f"      ⏱️ Time: {system_state['execution_time']:.3f} seconds")
        print(f"      🔄 Iterations: {system_state['iterations']}")
        
        # Step 3: Collect comprehensive power flow data
        print(f"   📊 Collecting comprehensive power flow data...")
        power_flow_data = collect_comprehensive_power_flow_data()
        
    else:
        print(f"      ❌ {lf_message}")
        print(f"      ⏱️ Time: {system_state['execution_time']:.3f} seconds")
        print(f"      🔄 Iterations: {system_state['iterations']}")
        power_flow_data = {'collection_success': False, 'error': 'Load flow failed'}
    
    # Step 4: Collect detailed load flow data (impedances, etc.)
    load_flow_data = None
    if data_collector:
        print(f"   📊 Collecting detailed system data...")
        try:
            load_flow_data = data_collector.collect_load_flow_data()
            if load_flow_data and 'collection_statistics' in load_flow_data:
                stats = load_flow_data['collection_statistics']
                total_impedances = stats['impedance_calculations_successful'] + stats['impedance_calculations_failed']
                success_rate = stats['impedance_calculations_successful'] / total_impedances * 100 if total_impedances > 0 else 0
                print(f"      ✅ Data collection completed")
                print(f"      📊 Impedance extraction: {stats['impedance_calculations_successful']}/{total_impedances} successful ({success_rate:.1f}%)")
        except Exception as e:
            print(f"      ❌ Data collection failed: {e}")
            load_flow_data = {'error': str(e)}
    
    # Step 5: Create scenario H5 filename and initial save
    h5_filename = f"scenario_{scenario_id}.h5"
    h5_path = os.path.join(OUT_DIR, h5_filename)
    
    # Step 6: Save initial data to H5 file
    print(f"   💾 Saving initial data to: {h5_filename}")
    
    scenario_results = {
        'scenario_info': scenario.copy(),
        'disconnection_actions': disconnection_actions,
        'load_flow_results': system_state,
        'power_flow_data': power_flow_data,
        'detailed_load_flow_data': load_flow_data,
        'execution_timestamp': datetime.now().isoformat(),
        'analysis_modules': {
            'data_collection': 'completed' if load_flow_data and 'error' not in load_flow_data else 'failed',
            'sensitivity_analysis': 'pending', 
            'y_matrix_building': 'pending'
        }
    }
    
    # Save initial data to H5 file
    save_scenario_to_h5_file(scenario_results, h5_path)
    
    # Step 7: Perform voltage sensitivity analysis (only if load flow converged)
    voltage_sensitivity_results = None
    if lf_success and ENABLE_VOLTAGE_SENSITIVITY:
        print(f"   🔬 Performing voltage sensitivity analysis...")
        voltage_sensitivity_results = perform_integrated_voltage_sensitivity(scenario_results, h5_path)
        
        if voltage_sensitivity_results['status'] == 'completed':
            print(f"      ✅ Voltage sensitivity completed")
            print(f"      📊 Generators analyzed: {voltage_sensitivity_results['generators_analyzed']}")
            print(f"      📊 Loads analyzed: {voltage_sensitivity_results['loads_analyzed']}")
            print(f"      📊 Total sensitivities: {voltage_sensitivity_results['total_sensitivities']}")
            print(f"      ⏱️ Analysis time: {voltage_sensitivity_results['analysis_time_seconds']:.2f} seconds")
            
            scenario_results['analysis_modules']['sensitivity_analysis'] = 'completed'
        else:
            print(f"      ❌ Voltage sensitivity failed: {voltage_sensitivity_results.get('error', 'Unknown error')}")
            scenario_results['analysis_modules']['sensitivity_analysis'] = 'failed'
    elif not lf_success:
        print(f"   ⏭️ Voltage sensitivity skipped (load flow failed)")
        voltage_sensitivity_results = {'status': 'skipped', 'reason': 'load flow failed'}
        scenario_results['analysis_modules']['sensitivity_analysis'] = 'skipped_lf_failed'
    
    # ★ NEW Step 8: Perform Y-Matrix building (regardless of load flow convergence) ★
    y_matrix_results = None
    if y_matrix_module and ENABLE_Y_MATRIX_BUILDING:
        print(f"   🔧 Performing Y-Matrix building...")
        y_matrix_results = perform_integrated_y_matrix_building(scenario_results, h5_path, lf_success)
        
        if y_matrix_results['status'] == 'completed':
            print(f"      ✅ Y-Matrix building completed")
            print(f"      📊 Matrix size: {y_matrix_results['matrix_size']} x {y_matrix_results['matrix_size']}")
            print(f"      📊 Non-zero elements: {y_matrix_results['nnz']}")
            print(f"      📊 Density: {y_matrix_results['density']:.4f}")
            print(f"      ⏱️ Construction time: {y_matrix_results['construction_time_seconds']:.2f} seconds")
            
            scenario_results['analysis_modules']['y_matrix_building'] = 'completed'
            
            # Note load flow status in Y-matrix results
            if not lf_success:
                print(f"      ⚠️ Y-Matrix built despite load flow non-convergence")
                
        elif y_matrix_results['status'] == 'failed':
            print(f"      ❌ Y-Matrix building failed: {y_matrix_results.get('error', 'Unknown error')}")
            scenario_results['analysis_modules']['y_matrix_building'] = 'failed'
        else:
            print(f"      ⏭️ Y-Matrix building skipped")
            scenario_results['analysis_modules']['y_matrix_building'] = 'skipped'
    else:
        print(f"   ⏭️ Y-Matrix building skipped (disabled or module unavailable)")
        y_matrix_results = {'status': 'skipped', 'reason': 'disabled or module unavailable'}
        scenario_results['analysis_modules']['y_matrix_building'] = 'skipped_disabled'
    
    # Step 9: Update H5 file with final analysis status
    with h5py.File(h5_path, 'a') as f:
        if 'analysis_modules' in f:
            modules_group = f['analysis_modules']
            
            # Update sensitivity analysis status
            if 'sensitivity_analysis' in modules_group:
                del modules_group['sensitivity_analysis']
            modules_group.create_dataset('sensitivity_analysis', 
                                       data=scenario_results['analysis_modules']['sensitivity_analysis'].encode())
            
            # Update Y-matrix building status
            if 'y_matrix_building' in modules_group:
                del modules_group['y_matrix_building']
            modules_group.create_dataset('y_matrix_building', 
                                       data=scenario_results['analysis_modules']['y_matrix_building'].encode())
            
            # Remove pending placeholders
            for placeholder in ['voltage_sensitivity_pending', 'y_matrix_pending']:
                if placeholder in modules_group:
                    del modules_group[placeholder]
    
    # Step 10: Restore elements to original state
    if disconnected_elements:
        print(f"   🔄 Restoring elements to service...")
        for element_obj, element_info in disconnected_elements:
            success, action = reconnect_element(element_obj, element_info)
            if success:
                print(f"      ✅ {action}")
            else:
                print(f"      ❌ {action}")
    
    # Add analysis results to scenario results for return
    scenario_results['voltage_sensitivity_results'] = voltage_sensitivity_results
    scenario_results['y_matrix_results'] = y_matrix_results
    
    return scenario_results, h5_path

# ── Save scenario to H5 file (enhanced for Y-matrix) ─────────────────────
def save_scenario_to_h5_file(scenario_results, h5_path):
    """Save scenario results to individual H5 file with comprehensive structure"""
    
    with h5py.File(h5_path, 'w') as f:
        
        # ═══ SCENARIO METADATA GROUP ═══
        meta_grp = f.create_group("scenario_metadata")
        meta_grp.create_dataset("scenario_id", data=scenario_results['scenario_info']['scenario_id'])
        meta_grp.create_dataset("contingency_type", data=scenario_results['scenario_info']['contingency_type'].encode())
        meta_grp.create_dataset("description", data=scenario_results['scenario_info']['description'].encode())
        meta_grp.create_dataset("severity", data=scenario_results['scenario_info']['severity'].encode())
        meta_grp.create_dataset("execution_timestamp", data=scenario_results['execution_timestamp'].encode())
        meta_grp.create_dataset("project_name", data=PROJECT.encode())
        meta_grp.create_dataset("study_case", data=STUDY.encode())
        meta_grp.create_dataset("num_elements_out", data=scenario_results['scenario_info']['num_elements_out'])
        
        # ═══ DISCONNECTION ACTIONS GROUP ═══
        actions_grp = f.create_group("disconnection_actions")
        if scenario_results['disconnection_actions']:
            actions_encoded = [action.encode() for action in scenario_results['disconnection_actions']]
            actions_grp.create_dataset("actions", data=actions_encoded)
            actions_grp.create_dataset("num_actions", data=len(scenario_results['disconnection_actions']))
        else:
            actions_grp.create_dataset("actions", data=[b"No elements disconnected"])
            actions_grp.create_dataset("num_actions", data=0)
        
        # ═══ LOAD FLOW RESULTS GROUP ═══
        lf_grp = f.create_group("load_flow_results")
        lf_results = scenario_results['load_flow_results']
        
        lf_grp.create_dataset("convergence", data=lf_results['convergence'])
        lf_grp.create_dataset("execution_time", data=lf_results['execution_time'])
        lf_grp.create_dataset("iterations", data=lf_results.get('iterations', 0))
        
        if lf_results['convergence']:
            # Bus data
            bus_grp = lf_grp.create_group("bus_data")
            bus_grp.create_dataset("bus_names", data=[name.encode() for name in lf_results['bus_names']])
            bus_grp.create_dataset("bus_voltages_pu", data=lf_results['bus_voltages'])
            bus_grp.create_dataset("bus_angles_deg", data=lf_results['bus_angles'])
            bus_grp.create_dataset("num_buses", data=lf_results['num_buses'])
            
            # Voltage statistics
            voltage_stats = lf_grp.create_group("voltage_statistics")
            voltage_stats.create_dataset("min_voltage_pu", data=lf_results['min_voltage'])
            voltage_stats.create_dataset("max_voltage_pu", data=lf_results['max_voltage'])
            
            # Count voltage violations
            voltages = lf_results['bus_voltages']
            valid_voltages = voltages[~np.isnan(voltages)]
            voltage_violations = np.sum((valid_voltages < 0.95) | (valid_voltages > 1.05))
            islanded_buses = np.sum(valid_voltages < 0.01)
            
            voltage_stats.create_dataset("voltage_violations_count", data=voltage_violations)
            voltage_stats.create_dataset("islanded_buses_count", data=islanded_buses)
            voltage_stats.create_dataset("avg_voltage_pu", data=np.nanmean(valid_voltages))
            voltage_stats.create_dataset("std_voltage_pu", data=np.nanstd(valid_voltages))
            
        else:
            lf_grp.create_dataset("error_code", data=lf_results['error_code'])
        
        # ═══ COMPREHENSIVE POWER FLOW DATA GROUP ═══
        if 'power_flow_data' in scenario_results and scenario_results['power_flow_data'].get('collection_success', False):
            power_grp = f.create_group("power_flow_data")
            pf_data = scenario_results['power_flow_data']
            
            # System totals
            totals_grp = power_grp.create_group("system_totals")
            for key, value in pf_data['system_totals'].items():
                totals_grp.create_dataset(key, data=value)
            
            # Generation data
            if pf_data['generation_data']:
                gen_grp = power_grp.create_group("generation_data")
                gen_names = list(pf_data['generation_data'].keys())
                gen_grp.create_dataset("generator_names", data=[name.encode() for name in gen_names])
                
                gen_params = ['P_actual_MW', 'Q_actual_MVAR', 'P_setpoint_MW', 'Q_setpoint_MVAR', 
                             'V_setpoint_pu', 'V_terminal_pu', 'P_max_MW', 'P_min_MW', 
                             'Q_max_MVAR', 'Q_min_MVAR']
                
                for param in gen_params:
                    values = [pf_data['generation_data'][name].get(param, np.nan) for name in gen_names]
                    gen_grp.create_dataset(param, data=values)
                
                gen_grp.create_dataset("num_generators", data=len(gen_names))
            
            # Load data
            if pf_data['load_data']:
                load_grp = power_grp.create_group("load_data")
                load_names = list(pf_data['load_data'].keys())
                load_grp.create_dataset("load_names", data=[name.encode() for name in load_names])
                
                load_params = ['P_nominal_MW', 'Q_nominal_MVAR', 'P_actual_MW', 'Q_actual_MVAR', 
                              'V_bus_pu', 'load_type']
                
                for param in load_params:
                    values = [pf_data['load_data'][name].get(param, np.nan) for name in load_names]
                    load_grp.create_dataset(param, data=values)
                
                load_grp.create_dataset("num_loads", data=len(load_names))
            
            # Line data
            if pf_data['line_data']:
                line_grp = power_grp.create_group("line_data")
                line_names = list(pf_data['line_data'].keys())
                line_grp.create_dataset("line_names", data=[name.encode() for name in line_names])
                
                line_params = ['P_from_MW', 'Q_from_MVAR', 'P_to_MW', 'Q_to_MVAR',
                              'P_loss_MW', 'Q_loss_MVAR', 'loading_percent', 'current_A']
                
                for param in line_params:
                    values = [pf_data['line_data'][name].get(param, np.nan) for name in line_names]
                    line_grp.create_dataset(param, data=values)
                
                line_grp.create_dataset("num_lines", data=len(line_names))
                
                # Line statistics
                loadings = np.array([pf_data['line_data'][name].get('loading_percent', 0) for name in line_names])
                overloaded_lines = np.sum(loadings > 100)
                line_grp.create_dataset("overloaded_lines_count", data=overloaded_lines)
                line_grp.create_dataset("max_loading_percent", data=np.nanmax(loadings))
                line_grp.create_dataset("avg_loading_percent", data=np.nanmean(loadings))
            
            # Transformer data
            if pf_data['transformer_data']:
                trafo_grp = power_grp.create_group("transformer_data")
                trafo_names = list(pf_data['transformer_data'].keys())
                trafo_grp.create_dataset("transformer_names", data=[name.encode() for name in trafo_names])
                
                trafo_params = ['P_hv_MW', 'Q_hv_MVAR', 'P_lv_MW', 'Q_lv_MVAR',
                               'P_loss_MW', 'Q_loss_MVAR', 'loading_percent', 'tap_position']
                
                for param in trafo_params:
                    values = [pf_data['transformer_data'][name].get(param, np.nan) for name in trafo_names]
                    trafo_grp.create_dataset(param, data=values)
                
                trafo_grp.create_dataset("num_transformers", data=len(trafo_names))
                
                # Transformer statistics
                loadings = np.array([pf_data['transformer_data'][name].get('loading_percent', 0) for name in trafo_names])
                overloaded_trafos = np.sum(loadings > 100)
                trafo_grp.create_dataset("overloaded_transformers_count", data=overloaded_trafos)
                trafo_grp.create_dataset("max_loading_percent", data=np.nanmax(loadings))
                trafo_grp.create_dataset("avg_loading_percent", data=np.nanmean(loadings))
        
        # ═══ DETAILED SYSTEM DATA GROUP ═══
        if scenario_results['detailed_load_flow_data'] and 'error' not in scenario_results['detailed_load_flow_data']:
            detailed_data = scenario_results['detailed_load_flow_data']
            detailed_grp = f.create_group("detailed_system_data")
            
            # Collection metadata
            detailed_grp.create_dataset("collection_timestamp", data=detailed_data['collection_timestamp'].encode())
            detailed_grp.create_dataset("collection_time_seconds", data=detailed_data['collection_time_seconds'])
            detailed_grp.create_dataset("data_quality", data=detailed_data['data_quality'].encode())
            
            # Collection statistics
            if 'collection_statistics' in detailed_data:
                stats_grp = detailed_grp.create_group("collection_statistics")
                for key, value in detailed_data['collection_statistics'].items():
                    stats_grp.create_dataset(key, data=value)
            
            # System summary
            if 'system_summary' in detailed_data:
                summary_grp = detailed_grp.create_group("system_summary")
                for key, value in detailed_data['system_summary'].items():
                    if isinstance(value, str):
                        summary_grp.create_dataset(key, data=value.encode())
                    else:
                        summary_grp.create_dataset(key, data=value)
            
            # ★ ENHANCED: Save detailed element data for Y-matrix building ★
            
            # Bus details
            if 'buses' in detailed_data:
                bus_detail_grp = detailed_grp.create_group("buses")
                buses = detailed_data['buses']
                
                bus_detail_grp.create_dataset("names", data=[name.encode() for name in buses['names']])
                bus_detail_grp.create_dataset("voltages_pu", data=buses['voltages_pu'])
                bus_detail_grp.create_dataset("voltage_angles_deg", data=buses['voltage_angles_deg'])
                bus_detail_grp.create_dataset("base_voltages_kV", data=buses['base_voltages_kV'])
                bus_detail_grp.create_dataset("in_service", data=buses['in_service'])
                bus_detail_grp.create_dataset("active_injection_MW", data=buses['active_injection_MW'])
                bus_detail_grp.create_dataset("reactive_injection_MVAR", data=buses['reactive_injection_MVAR'])
            
            # Line details with impedances
            if 'lines' in detailed_data:
                line_detail_grp = detailed_grp.create_group("lines")
                lines = detailed_data['lines']
                
                line_detail_grp.create_dataset("names", data=[name.encode() for name in lines['names']])
                line_detail_grp.create_dataset("from_buses", data=[bus.encode() for bus in lines['from_buses']])
                line_detail_grp.create_dataset("to_buses", data=[bus.encode() for bus in lines['to_buses']])
                
                # Impedance data
                impedances = lines['impedances']
                line_detail_grp.create_dataset("R_ohm", data=[imp['R_ohm'] for imp in impedances])
                line_detail_grp.create_dataset("X_ohm", data=[imp['X_ohm'] for imp in impedances])
                line_detail_grp.create_dataset("B_uS", data=[imp.get('B_uS', np.nan) for imp in impedances])
                line_detail_grp.create_dataset("Z_magnitude", data=[imp['Z_magnitude'] for imp in impedances])
                line_detail_grp.create_dataset("impedance_source", data=[imp['impedance_source'].encode() for imp in impedances])
            
            # Transformer details with impedances
            if 'transformers' in detailed_data:
                trafo_detail_grp = detailed_grp.create_group("transformers")
                transformers = detailed_data['transformers']
                
                trafo_detail_grp.create_dataset("names", data=[name.encode() for name in transformers['names']])
                trafo_detail_grp.create_dataset("classes", data=[cls.encode() for cls in transformers['classes']])
                trafo_detail_grp.create_dataset("from_buses", data=[bus.encode() for bus in transformers['from_buses']])
                trafo_detail_grp.create_dataset("to_buses", data=[bus.encode() for bus in transformers['to_buses']])
                
                # Impedance data
                impedances = transformers['impedances']
                trafo_detail_grp.create_dataset("R_ohm", data=[imp['R_ohm'] for imp in impedances])
                trafo_detail_grp.create_dataset("X_ohm", data=[imp['X_ohm'] for imp in impedances])
                trafo_detail_grp.create_dataset("tap_ratio", data=[imp.get('tap_ratio', 1.0) for imp in impedances])
                trafo_detail_grp.create_dataset("phase_shift_deg", data=[imp.get('phase_shift_deg', 0.0) for imp in impedances])
                trafo_detail_grp.create_dataset("impedance_source", data=[imp['impedance_source'].encode() for imp in impedances])
            
            # Generator details with impedances
            if 'generators' in detailed_data:
                gen_detail_grp = detailed_grp.create_group("generators")
                generators = detailed_data['generators']
                
                gen_detail_grp.create_dataset("names", data=[name.encode() for name in generators['names']])
                gen_detail_grp.create_dataset("buses", data=[bus.encode() for bus in generators['buses']])
                gen_detail_grp.create_dataset("active_power_MW", data=generators['active_power_MW'])
                gen_detail_grp.create_dataset("reactive_power_MVAR", data=generators['reactive_power_MVAR'])
                gen_detail_grp.create_dataset("voltage_setpoint_pu", data=generators['voltage_setpoint_pu'])
                gen_detail_grp.create_dataset("terminal_voltage_pu", data=generators['terminal_voltage_pu'])
                gen_detail_grp.create_dataset("active_power_limits_MW", data=generators['active_power_limits_MW'])
                gen_detail_grp.create_dataset("reactive_limits_min_MVAR", data=generators['reactive_limits_min_MVAR'])
                gen_detail_grp.create_dataset("reactive_limits_max_MVAR", data=generators['reactive_limits_max_MVAR'])
                
                # Impedance data
                impedances = generators['impedances']
                gen_detail_grp.create_dataset("xd_pu", data=[imp['xd_pu'] for imp in impedances])
                gen_detail_grp.create_dataset("xq_pu", data=[imp['xq_pu'] for imp in impedances])
                gen_detail_grp.create_dataset("xd_prime_pu", data=[imp['xd_prime_pu'] for imp in impedances])
                gen_detail_grp.create_dataset("S_rated_MVA", data=[imp['S_rated_MVA'] for imp in impedances])
                gen_detail_grp.create_dataset("V_rated_kV", data=[imp['V_rated_kV'] for imp in impedances])
            
            # Load details with impedances
            if 'loads' in detailed_data:
                load_detail_grp = detailed_grp.create_group("loads")
                loads = detailed_data['loads']
                
                load_detail_grp.create_dataset("names", data=[name.encode() for name in loads['names']])
                load_detail_grp.create_dataset("buses", data=[bus.encode() for bus in loads['buses']])
                load_detail_grp.create_dataset("active_power_MW", data=loads['active_power_MW'])
                load_detail_grp.create_dataset("reactive_power_MVAR", data=loads['reactive_power_MVAR'])
                
                # Impedance data
                impedances = loads['impedances']
                load_detail_grp.create_dataset("Z_equivalent_ohm", data=[imp.get('Z_equivalent_ohm', np.nan) for imp in impedances])
                load_detail_grp.create_dataset("R_equivalent_ohm", data=[imp.get('R_equivalent_ohm', np.nan) for imp in impedances])
                load_detail_grp.create_dataset("X_equivalent_ohm", data=[imp.get('X_equivalent_ohm', np.nan) for imp in impedances])
                load_detail_grp.create_dataset("bus_voltage_pu", data=[imp.get('bus_voltage_pu', np.nan) for imp in impedances])
            
            # Shunt details with impedances
            if 'shunts' in detailed_data:
                shunt_detail_grp = detailed_grp.create_group("shunts")
                shunts = detailed_data['shunts']
                
                shunt_detail_grp.create_dataset("names", data=[name.encode() for name in shunts['names']])
                shunt_detail_grp.create_dataset("buses", data=[bus.encode() for bus in shunts['buses']])
                shunt_detail_grp.create_dataset("reactive_power_MVAR", data=shunts['reactive_power_MVAR'])
                
                # Impedance data
                impedances = shunts['impedances']
                shunt_detail_grp.create_dataset("X_shunt_ohm", data=[imp.get('X_shunt_ohm', np.nan) for imp in impedances])
                shunt_detail_grp.create_dataset("capacitive", data=[imp.get('capacitive', False) for imp in impedances])
                shunt_detail_grp.create_dataset("Q_MVAR", data=[imp.get('Q_MVAR', np.nan) for imp in impedances])
        
        elif scenario_results['detailed_load_flow_data'] and 'error' in scenario_results['detailed_load_flow_data']:
            # Save error information
            error_grp = f.create_group("detailed_system_data")
            error_grp.create_dataset("error", data=scenario_results['detailed_load_flow_data']['error'].encode())
        
        # ═══ ANALYSIS MODULE STATUS GROUP ═══
        analysis_grp = f.create_group("analysis_modules")
        modules = scenario_results['analysis_modules']
        analysis_grp.create_dataset("data_collection", data=modules['data_collection'].encode())
        analysis_grp.create_dataset("sensitivity_analysis", data=modules['sensitivity_analysis'].encode())
        analysis_grp.create_dataset("y_matrix_building", data=modules['y_matrix_building'].encode())
        
        # Create placeholder groups for future modules (will be filled by respective modules)
        f.create_group("voltage_sensitivity")
        f.create_group("y_matrix")

# ── Main execution with Y-Matrix integration ──────────────────────────────
def main():
    """Main execution function with integrated Y-Matrix building"""
    
    start_time = time.time()
    
    # Setup modules
    data_collector = setup_data_collection_module()
    y_matrix_module = setup_y_matrix_module()
    
    # Read contingency scenarios
    scenarios = read_contingency_scenarios(CONTINGENCY_CSV)
    if not scenarios:
        return
    
    # Get PowerFactory elements
    elements_catalog = get_powerfactory_elements()
    
    # Prepare output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Execute each scenario
    print(f"\n🚀 EXECUTING INTEGRATED CONTINGENCY ANALYSIS...")
    print(f"   📊 Total scenarios: {len(scenarios)}")
    print(f"   📁 Output directory: {OUT_DIR}")
    
    # Module status
    if data_collector:
        print(f"   ✅ Data collection module ready")
    else:
        print(f"   ⚠️ Data collection module not available")
    
    if ENABLE_VOLTAGE_SENSITIVITY and VOLTAGE_SENSITIVITY_AVAILABLE:
        print(f"   ✅ Voltage sensitivity analysis enabled")
        print(f"   ⚙️ P perturbation: ±{VS_PERTURBATION_STEP_P_MW} MW")
        print(f"   ⚙️ Q perturbation: ±{VS_PERTURBATION_STEP_Q_MVAR} MVAR")
    else:
        print(f"   ⏭️ Voltage sensitivity analysis disabled")
    
    if ENABLE_Y_MATRIX_BUILDING and Y_MATRIX_BUILDER_AVAILABLE:
        print(f"   ✅ Y-Matrix building enabled")
        print(f"   🔧 Y-Matrix will be built for all scenarios (regardless of load flow convergence)")
    else:
        print(f"   ⏭️ Y-Matrix building disabled")
    
    # Initialize counters
    successful_scenarios = 0
    failed_scenarios = 0
    h5_files_created = []
    voltage_sensitivity_completed = 0
    y_matrix_completed = 0
    load_flow_converged = 0
    load_flow_failed = 0
    total_sensitivities = 0
    total_y_matrices = 0
    
    # Execute scenarios
    for i, scenario in enumerate(scenarios):
        try:
            print(f"\n{'='*70}")
            print(f"PROGRESS: {i+1}/{len(scenarios)}")
            
            # Execute contingency scenario with Y-Matrix integration
            scenario_results, h5_path = execute_contingency_scenario(
                scenario, elements_catalog, data_collector, y_matrix_module
            )
            
            # Track results
            lf_converged = scenario_results['load_flow_results']['convergence']
            if lf_converged:
                successful_scenarios += 1
                load_flow_converged += 1
                print(f"✅ Scenario {scenario['scenario_id']} completed successfully")
            else:
                failed_scenarios += 1
                load_flow_failed += 1
                print(f"⚠️ Scenario {scenario['scenario_id']} completed with load flow failure")
            
            # Track voltage sensitivity results
            vs_results = scenario_results.get('voltage_sensitivity_results')
            if vs_results and vs_results['status'] == 'completed':
                voltage_sensitivity_completed += 1
                total_sensitivities += vs_results['total_sensitivities']
            
            # Track Y-matrix results
            y_matrix_results = scenario_results.get('y_matrix_results')
            if y_matrix_results and y_matrix_results['status'] == 'completed':
                y_matrix_completed += 1
                total_y_matrices += 1
                print(f"   🔧 Y-Matrix: {y_matrix_results['matrix_size']}x{y_matrix_results['matrix_size']}, "
                      f"density={y_matrix_results['density']:.4f}")
            
            print(f"   💾 H5 file: {os.path.basename(h5_path)}")
            h5_files_created.append(h5_path)
                
        except Exception as e:
            failed_scenarios += 1
            print(f"❌ Error in scenario {scenario['scenario_id']}: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n🎉 INTEGRATED CONTINGENCY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"📊 FINAL SUMMARY:")
    print(f"   🎯 Total scenarios: {len(scenarios)}")
    print(f"   ✅ Load flow converged: {load_flow_converged}")
    print(f"   ❌ Load flow failed: {load_flow_failed}")
    print(f"   📈 Load flow success rate: {load_flow_converged/len(scenarios)*100:.1f}%")
    print(f"   ⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"   💾 H5 files created: {len(h5_files_created)}")
    
    # Module-specific summaries
    if ENABLE_VOLTAGE_SENSITIVITY and VOLTAGE_SENSITIVITY_AVAILABLE:
        print(f"   🔬 Voltage sensitivity completed: {voltage_sensitivity_completed}/{load_flow_converged}")
        print(f"   📊 Total sensitivities calculated: {total_sensitivities}")
        vs_success_rate = voltage_sensitivity_completed/load_flow_converged*100 if load_flow_converged > 0 else 0
        print(f"   📈 Voltage sensitivity success rate: {vs_success_rate:.1f}%")
    
    if ENABLE_Y_MATRIX_BUILDING and Y_MATRIX_BUILDER_AVAILABLE:
        print(f"   🔧 Y-Matrix completed: {y_matrix_completed}/{len(scenarios)}")
        y_matrix_success_rate = y_matrix_completed/len(scenarios)*100 if len(scenarios) > 0 else 0
        print(f"   📈 Y-Matrix success rate: {y_matrix_success_rate:.1f}%")
        print(f"   📊 Y-Matrices built for both converged and non-converged load flows")
    
    print(f"   📁 Output directory: {OUT_DIR}")
    
    print(f"\n🔄 ANALYSIS STATUS:")
    print(f"   1. ✅ Module 1: Contingency Generator - Complete")
    print(f"   2. ✅ Module 2: Contingency Executor - Complete")
    print(f"   3. ✅ Module 3: Load Flow Data Collector - Complete")
    if voltage_sensitivity_completed > 0:
        print(f"   4. ✅ Module 4: Voltage Sensitivity Analysis - Complete")
    else:
        print(f"   4. ⏳ Module 4: Voltage Sensitivity Analysis - Limited/Skipped")
    if y_matrix_completed > 0:
        print(f"   5. ✅ Module 5: Y-Matrix Builder - Complete")
    else:
        print(f"   5. ⏳ Module 5: Y-Matrix Builder - Limited/Skipped")
    
    print(f"\n💡 Each H5 file now contains:")
    print(f"   • Scenario metadata and disconnection actions")
    print(f"   • Load flow results (convergence status, voltages, angles)")
    print(f"   • Comprehensive power flow data (MW/MVAR for all elements)")
    print(f"   • Detailed system data (impedances, topology)")
    if voltage_sensitivity_completed > 0:
        print(f"   • Voltage sensitivity analysis results (for converged cases)")
    if y_matrix_completed > 0:
        print(f"   • ✨ Y-matrix (admittance matrix) with properties")
    
    return {
        'total_scenarios': len(scenarios),
        'load_flow_converged': load_flow_converged,
        'load_flow_failed': load_flow_failed,
        'voltage_sensitivity_completed': voltage_sensitivity_completed,
        'y_matrix_completed': y_matrix_completed,
        'h5_files_created': len(h5_files_created),
        'total_time_minutes': total_time/60
    }

# ── Enhanced summary report ────────────────────────────────────────────────
def create_integrated_summary_report():
    """Create a comprehensive summary report including Y-Matrix results"""
    
    summary_file = os.path.join(OUT_DIR, "integrated_contingency_summary_report.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("INTEGRATED CONTINGENCY ANALYSIS WITH Y-MATRIX SUMMARY REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Project: {PROJECT}\n")
        f.write(f"Study Case: {STUDY}\n\n")
        
        # List all H5 files
        h5_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.h5')]
        h5_files.sort()
        
        f.write(f"H5 Files Created: {len(h5_files)}\n")
        f.write("-" * 30 + "\n")
        
        # Counters for summary
        scenarios_with_lf_converged = 0
        scenarios_with_vs = 0
        scenarios_with_y_matrix = 0
        total_sensitivities = 0
        total_y_matrix_size = 0
        
        for h5_file in h5_files:
            h5_path = os.path.join(OUT_DIR, h5_file)
            file_size = os.path.getsize(h5_path) / 1024
            f.write(f"{h5_file:<25} {file_size:>8.1f} KB")
            
            try:
                with h5py.File(h5_path, 'r') as hf:
                    scenario_id = hf['scenario_metadata']['scenario_id'][()]
                    contingency_type = hf['scenario_metadata']['contingency_type'][()].decode()
                    
                    # Check load flow convergence
                    lf_converged = hf['load_flow_results']['convergence'][()]
                    if lf_converged:
                        scenarios_with_lf_converged += 1
                    
                    # Check voltage sensitivity
                    if 'voltage_sensitivity' in hf and 'analysis_timestamp' in hf['voltage_sensitivity']:
                        scenarios_with_vs += 1
                        
                        # Count sensitivities
                        vs = hf['voltage_sensitivity']
                        scenario_sensitivities = 0
                        if 'generators' in vs:
                            for gen_name in vs['generators'].keys():
                                if gen_name == 'num_generators_analyzed':
                                    continue
                                gen_group = vs['generators'][gen_name]
                                if 'P_sensitivity' in gen_group:
                                    scenario_sensitivities += gen_group['P_sensitivity']['num_sensitive_buses'][()]
                                if 'Q_sensitivity' in gen_group:
                                    scenario_sensitivities += gen_group['Q_sensitivity']['num_sensitive_buses'][()]
                        
                        if 'loads' in vs:
                            for load_name in vs['loads'].keys():
                                if load_name == 'num_loads_analyzed':
                                    continue
                                load_group = vs['loads'][load_name]
                                if 'P_sensitivity' in load_group:
                                    scenario_sensitivities += load_group['P_sensitivity']['num_sensitive_buses'][()]
                                if 'Q_sensitivity' in load_group:
                                    scenario_sensitivities += load_group['Q_sensitivity']['num_sensitive_buses'][()]
                        
                        total_sensitivities += scenario_sensitivities
                    
                    # Check Y-matrix
                    if 'y_matrix' in hf and 'construction_timestamp' in hf['y_matrix']:
                        scenarios_with_y_matrix += 1
                        matrix_size = hf['y_matrix']['matrix_size'][()]
                        total_y_matrix_size += matrix_size
                        
                        f.write(f" | LF:{'✓' if lf_converged else '✗'} VS:{'✓' if 'voltage_sensitivity' in hf else '✗'} Y:{matrix_size}x{matrix_size}")
                    else:
                        f.write(f" | LF:{'✓' if lf_converged else '✗'} VS:{'✓' if 'voltage_sensitivity' in hf else '✗'} Y:✗")
                        
            except Exception as e:
                f.write(f" | Error: {e}")
            
            f.write("\n")
        
        f.write(f"\nAnalysis Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total scenarios: {len(h5_files)}\n")
        f.write(f"Load flow converged: {scenarios_with_lf_converged}/{len(h5_files)} ({scenarios_with_lf_converged/len(h5_files)*100:.1f}%)\n")
        f.write(f"Voltage sensitivity completed: {scenarios_with_vs}/{len(h5_files)} ({scenarios_with_vs/len(h5_files)*100:.1f}%)\n")
        f.write(f"Y-matrix completed: {scenarios_with_y_matrix}/{len(h5_files)} ({scenarios_with_y_matrix/len(h5_files)*100:.1f}%)\n")
        f.write(f"Total sensitivities calculated: {total_sensitivities}\n")
        if scenarios_with_y_matrix > 0:
            avg_matrix_size = total_y_matrix_size / scenarios_with_y_matrix
            f.write(f"Average Y-matrix size: {avg_matrix_size:.0f} x {avg_matrix_size:.0f}\n")
        
        f.write(f"\nIntegrated Analysis Modules:\n")
        f.write(f"• Scenario metadata and disconnection actions\n")
        f.write(f"• Load flow results (convergence, voltages, angles)\n")
        f.write(f"• Comprehensive power flow data (MW/MVAR for all elements)\n")
        f.write(f"• Detailed system data with impedances\n")
        f.write(f"• Voltage sensitivity analysis results (for converged cases)\n")
        f.write(f"• ** Y-matrix (admittance matrix) with properties\n")
        
        f.write(f"\nKey Advantages of Integrated Approach:\n")
        f.write(f"• Y-matrix built for both converged and non-converged load flows\n")
        f.write(f"• All analysis in single execution loop (no separate runs needed)\n")
        f.write(f"• Consistent data across all modules\n")
        f.write(f"• Efficient execution with shared PowerFactory state\n")
        f.write(f"• Complete analysis results in single H5 file per scenario\n")
        
        if ENABLE_Y_MATRIX_BUILDING and Y_MATRIX_BUILDER_AVAILABLE:
            f.write(f"\nY-Matrix Building Parameters:\n")
            f.write(f"• Base power: {SBASE_MVA} MVA\n")
            f.write(f"• Built from impedance data collected during data collection phase\n")
            f.write(f"• Includes lines, transformers, generators, loads, and shunts\n")
            f.write(f"• Matrix properties calculated (eigenvalues, condition number)\n")
            f.write(f"• Stored in sparse format for efficiency\n")
        
        if ENABLE_VOLTAGE_SENSITIVITY and VOLTAGE_SENSITIVITY_AVAILABLE:
            f.write(f"\nVoltage Sensitivity Parameters:\n")
            f.write(f"• P perturbation: +/- {VS_PERTURBATION_STEP_P_MW} MW\n")
            f.write(f"• Q perturbation: +/- {VS_PERTURBATION_STEP_Q_MVAR} MVAR\n")
            f.write(f"• Minimum sensitivity threshold: {MIN_SENSITIVITY_THRESHOLD}\n")
            f.write(f"• Only performed for load flow converged cases\n")
    
    print(f"📄 Integrated analysis summary report saved: {os.path.basename(summary_file)}")

if __name__ == "__main__":
    results = main()
    create_integrated_summary_report()
    
    # Print final execution summary
    if results:
        print(f"\n📈 EXECUTION SUMMARY:")
        print(f"   🎯 Total scenarios processed: {results['total_scenarios']}")
        print(f"   ✅ Load flow converged: {results['load_flow_converged']}")
        print(f"   ❌ Load flow failed: {results['load_flow_failed']}")
        print(f"   🔬 Voltage sensitivity completed: {results['voltage_sensitivity_completed']}")
        print(f"   🔧 Y-matrix completed: {results['y_matrix_completed']}")
        print(f"   💾 H5 files created: {results['h5_files_created']}")
        print(f"   ⏱️ Total execution time: {results['total_time_minutes']:.1f} minutes")
        
        print(f"\n🎊 READY FOR YOUR POWER SYSTEM STUDY!")
        print(f"   📊 Use converged scenarios for your analysis")
        print(f"   🔍 Non-converged scenarios marked for exclusion")
        print(f"   📁 All data organized in individual H5 files per scenario")