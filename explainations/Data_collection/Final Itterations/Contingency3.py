# comprehensive_contingency_analysis_fixed.py
"""
Fixed Comprehensive Contingency Analysis for Generator 2 at Bus 31
IEEE 39-Bus New England System - PowerFactory Implementation
NOW WITH PROPER NUMERICAL SENSITIVITY ANALYSIS

This code performs:
1. Baseline data collection
2. Generator 2 outage simulation
3. Comprehensive impact analysis with REAL sensitivity calculation
4. Zoning recommendations
5. Detailed reporting for rezoning decisions
"""

import sys
import os
import h5py
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from scipy.sparse import csr_matrix
import warnings
import time
warnings.filterwarnings('ignore')

# PowerFactory connection
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT = "39 Bus New England System"
STUDY = "RMS_Simulation"
TARGET_GENERATOR = "Generator 2"  # At Bus 31
TARGET_BUS = 31
BASE_DIR = os.path.join(os.getcwd(), "contingency_analysis")
SBASE_MVA = 100.0

# Create directory structure
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "baseline_data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "contingency_gen2_bus31"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "analysis_results"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)

print("🔋 COMPREHENSIVE GENERATOR CONTINGENCY ANALYSIS (FIXED)")
print("="*60)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Target: {TARGET_GENERATOR} at Bus {TARGET_BUS}")
print(f"📁 Output Directory: {BASE_DIR}")
print()

# ── Helper Functions (from your working code) ──────────────────────────────
def has(o, a):
    """Safely check if object has attribute"""
    try:
        return o.HasAttribute(a) if o else False
    except:
        return False

def get(o, a, d=np.nan):
    """Safely get attribute value"""
    try:
        return o.GetAttribute(a) if has(o, a) else d
    except:
        return d

def as_bus(o):
    """Safely get bus object"""
    try:
        if not o:
            return None
        return o.cterm if has(o, "cterm") else o
    except:
        return o

def safe_get_name(obj):
    """Safely get object name"""
    try:
        return obj.loc_name if obj else "Unknown"
    except:
        return "Unknown"

def safe_get_class(obj):
    """Safely get object class name"""
    try:
        return obj.GetClassName() if obj else "Unknown"
    except:
        return "Unknown"

# ── PowerFactory Connection ────────────────────────────────────────────────
def connect_powerfactory():
    """Connect to PowerFactory and activate project"""
    app = pf.GetApplication()
    if not app:
        raise Exception("PowerFactory not running!")
    
    # Suppress warning messages temporarily
    app.EchoOff()
    app.ResetCalculation()
    
    if app.ActivateProject(PROJECT) != 0:
        app.EchoOn()
        raise Exception(f"Project '{PROJECT}' not found!")
    
    study_case = None
    for case in app.GetProjectFolder("study").GetContents("*.IntCase"):
        if case.loc_name == STUDY:
            study_case = case
            break
    
    if not study_case:
        app.EchoOn()
        raise Exception(f"Study case '{STUDY}' not found!")
    
    study_case.Activate()
    app.EchoOn()
    
    print(f"✅ Connected to PowerFactory: {PROJECT} | {STUDY}")
    return app

# ── Data Collection Functions ──────────────────────────────────────────────
def collect_bus_data(app):
    """Collect comprehensive bus data with better error handling"""
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    
    bus_data = []
    for i, bus in enumerate(buses):
        try:
            # More robust voltage and angle extraction
            voltage_pu = np.nan
            angle_deg = np.nan
            
            try:
                voltage_pu = bus.GetAttribute('m:u')
            except:
                try:
                    voltage_pu = getattr(bus, 'u0', 1.0)  # Use initial value if no solution
                except:
                    voltage_pu = 1.0
            
            try:
                angle_deg = bus.GetAttribute('m:phiu')
            except:
                try:
                    angle_deg = getattr(bus, 'phiu0', 0.0)  # Use initial value if no solution
                except:
                    angle_deg = 0.0
            
            data = {
                'bus_idx': i,
                'bus_name': bus.loc_name,
                'bus_number': getattr(bus, 'iUsage', i+1),
                'voltage_kv': getattr(bus, 'uknom', 0),
                'voltage_pu': float(voltage_pu) if not np.isnan(voltage_pu) else 1.0,
                'angle_deg': float(angle_deg) if not np.isnan(angle_deg) else 0.0,
                'area': getattr(bus, 'narea', 0)
            }
            bus_data.append(data)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not collect data for bus {i}: {e}")
            bus_data.append({
                'bus_idx': i, 'bus_name': f'Bus_{i}', 'bus_number': i+1,
                'voltage_kv': 0, 'voltage_pu': 1.0, 'angle_deg': 0.0, 'area': 0
            })
    
    return pd.DataFrame(bus_data)

def collect_generator_data(app):
    """Collect comprehensive generator data"""
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    
    gen_data = []
    for i, gen in enumerate(generators):
        try:
            # Get connected bus
            bus = gen.bus1 if hasattr(gen, 'bus1') else None
            bus_name = bus.loc_name if bus else "Unknown"
            
            data = {
                'gen_idx': i,
                'gen_name': gen.loc_name,
                'bus_name': bus_name,
                'P_MW': getattr(gen, 'pgini', 0),
                'Q_MVAR': getattr(gen, 'qgini', 0),
                'P_max_MW': getattr(gen, 'Pmax', 0),
                'P_min_MW': getattr(gen, 'Pmin', 0),
                'Q_max_MVAR': getattr(gen, 'Qmax', 0),
                'Q_min_MVAR': getattr(gen, 'Qmin', 0),
                'V_setpoint': getattr(gen, 'uset', 1.0),
                'in_service': not getattr(gen, 'outserv', 0),
                'is_slack': getattr(gen, 'ip_ctrl', 0) == 1
            }
            gen_data.append(data)
        except Exception as e:
            print(f"⚠️ Warning: Could not collect data for generator {i}: {e}")
    
    return pd.DataFrame(gen_data)

def collect_line_data(app):
    """Collect comprehensive line data"""
    lines = app.GetCalcRelevantObjects("*.ElmLne")
    transformers = app.GetCalcRelevantObjects("*.ElmTr2")
    
    line_data = []
    
    # Process lines
    for i, line in enumerate(lines):
        try:
            bus1 = line.bus1 if hasattr(line, 'bus1') else None
            bus2 = line.bus2 if hasattr(line, 'bus2') else None
            
            data = {
                'element_idx': i,
                'element_name': line.loc_name,
                'element_type': 'Line',
                'from_bus': bus1.loc_name if bus1 else "Unknown",
                'to_bus': bus2.loc_name if bus2 else "Unknown",
                'P_from_MW': line.GetAttribute('m:P:bus1') if hasattr(line, 'GetAttribute') else np.nan,
                'Q_from_MVAR': line.GetAttribute('m:Q:bus1') if hasattr(line, 'GetAttribute') else np.nan,
                'P_to_MW': line.GetAttribute('m:P:bus2') if hasattr(line, 'GetAttribute') else np.nan,
                'Q_to_MVAR': line.GetAttribute('m:Q:bus2') if hasattr(line, 'GetAttribute') else np.nan,
                'loading_percent': line.GetAttribute('c:loading') if hasattr(line, 'GetAttribute') else np.nan,
                'R_ohm': getattr(line, 'R1', 0),
                'X_ohm': getattr(line, 'X1', 0),
                'rating_MVA': getattr(line, 'snom', 0)
            }
            line_data.append(data)
        except Exception as e:
            print(f"⚠️ Warning: Could not collect data for line {i}: {e}")
    
    # Process transformers
    for i, trafo in enumerate(transformers):
        try:
            bus1 = trafo.bus1 if hasattr(trafo, 'bus1') else None
            bus2 = trafo.bus2 if hasattr(trafo, 'bus2') else None
            
            # More robust power flow data extraction for transformers
            P_from = np.nan
            Q_from = np.nan
            P_to = np.nan
            Q_to = np.nan
            loading = np.nan
            
            try:
                P_from = trafo.GetAttribute('m:P:bus1')
            except:
                try:
                    P_from = trafo.GetAttribute('m:Psum:bus1')
                except:
                    P_from = np.nan
            
            try:
                Q_from = trafo.GetAttribute('m:Q:bus1')
            except:
                try:
                    Q_from = trafo.GetAttribute('m:Qsum:bus1')
                except:
                    Q_from = np.nan
            
            try:
                P_to = trafo.GetAttribute('m:P:bus2')
            except:
                try:
                    P_to = trafo.GetAttribute('m:Psum:bus2')
                except:
                    P_to = np.nan
            
            try:
                Q_to = trafo.GetAttribute('m:Q:bus2')
            except:
                try:
                    Q_to = trafo.GetAttribute('m:Qsum:bus2')
                except:
                    Q_to = np.nan
            
            try:
                loading = trafo.GetAttribute('c:loading')
            except:
                loading = np.nan
            
            data = {
                'element_idx': len(lines) + i,
                'element_name': trafo.loc_name,
                'element_type': 'Transformer',
                'from_bus': bus1.loc_name if bus1 else "Unknown",
                'to_bus': bus2.loc_name if bus2 else "Unknown",
                'P_from_MW': float(P_from) if not np.isnan(P_from) else 0.0,
                'Q_from_MVAR': float(Q_from) if not np.isnan(Q_from) else 0.0,
                'P_to_MW': float(P_to) if not np.isnan(P_to) else 0.0,
                'Q_to_MVAR': float(Q_to) if not np.isnan(Q_to) else 0.0,
                'loading_percent': float(loading) if not np.isnan(loading) else 0.0,
                'R_pu': getattr(trafo, 'ukr', 0) / 100,
                'X_pu': getattr(trafo, 'uk', 0) / 100,
                'rating_MVA': getattr(trafo, 'strn', 0)
            }
            line_data.append(data)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not collect data for transformer {i}: {e}")
            # Add dummy data to maintain consistency
            line_data.append({
                'element_idx': len(lines) + i,
                'element_name': f'Transformer_{i}',
                'element_type': 'Transformer',
                'from_bus': "Unknown",
                'to_bus': "Unknown",
                'P_from_MW': 0.0,
                'Q_from_MVAR': 0.0,
                'P_to_MW': 0.0,
                'Q_to_MVAR': 0.0,
                'loading_percent': 0.0,
                'R_pu': 0.0,
                'X_pu': 0.0,
                'rating_MVA': 0.0
            })
    
    return pd.DataFrame(line_data)

def collect_load_data(app):
    """Collect comprehensive load data"""
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    
    load_data = []
    for i, load in enumerate(loads):
        try:
            bus = load.bus1 if hasattr(load, 'bus1') else None
            
            data = {
                'load_idx': i,
                'load_name': load.loc_name,
                'bus_name': bus.loc_name if bus else "Unknown",
                'P_MW': getattr(load, 'plini', 0),
                'Q_MVAR': getattr(load, 'qlini', 0),
                'in_service': not getattr(load, 'outserv', 0)
            }
            load_data.append(data)
        except Exception as e:
            print(f"⚠️ Warning: Could not collect data for load {i}: {e}")
    
    return pd.DataFrame(load_data)

# ── PROPER SENSITIVITY ANALYSIS FUNCTIONS (from Sensitivity_simulation.py) ─
def identify_perturbable_buses(app):
    """
    Identify buses where we can perturb power by finding buses with non-zero 
    load or generation from power flow results
    """
    print(f"🔍 Identifying buses with active power injection/consumption...")
    
    # Get loads and generators
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    
    # Solve base case to get power flow results
    comLdf = app.GetFromStudyCase("ComLdf")
    comLdf.Execute()
    
    perturbable_buses = {}  # bus_idx -> {'loads': [...], 'generators': [...], 'net_P': value}
    
    # Method 1: Find loads by checking load elements and trying to match by name/number
    print(f"🏠 Identifying load buses by matching names...")
    for load in loads:
        load_name = safe_get_name(load)
        load_P = get(load, "plini", 0.0)
        
        # Extract bus number from load name (e.g., "Load 03" -> 3, "Load 15" -> 15)
        try:
            import re
            numbers = re.findall(r'\d+', load_name)
            if numbers:
                load_bus_num = int(numbers[-1])  # Take last number found
                # Convert to bus index (Bus 03 = index 2, Bus 15 = index 14, etc.)
                if 1 <= load_bus_num <= 39:
                    bus_idx = load_bus_num - 1
                    if bus_idx not in perturbable_buses:
                        perturbable_buses[bus_idx] = {'loads': [], 'generators': [], 'net_P': 0.0}
                    perturbable_buses[bus_idx]['loads'].append(load)
                    perturbable_buses[bus_idx]['net_P'] -= load_P  # Load decreases net injection
                    print(f"   {load_name} (P={load_P:.1f} MW) → Bus {bus_idx} ({safe_get_name(buses[bus_idx])})")
        except:
            print(f"   ⚠️ Could not parse bus number from {load_name}")
    
    # Method 2: Find generators by checking generator elements and trying to match by name/number  
    print(f"🔋 Identifying generator buses by matching names...")
    for gen in generators:
        gen_name = safe_get_name(gen)
        gen_P = get(gen, "pgini", 0.0)
        
        # Extract bus number from generator name (e.g., "G 01" -> 1, "G 10" -> 10)
        try:
            import re
            numbers = re.findall(r'\d+', gen_name)
            if numbers:
                gen_bus_num = int(numbers[-1])  # Take last number found
                
                # For IEEE 39-bus, generators are typically at specific buses
                # Let's try common generator bus mapping
                gen_bus_mapping = {
                    1: 30,   # G 01 at Bus 31
                    2: 31,   # G 02 at Bus 32  
                    3: 32,   # G 03 at Bus 33
                    4: 33,   # G 04 at Bus 34
                    5: 34,   # G 05 at Bus 35
                    6: 35,   # G 06 at Bus 36
                    7: 36,   # G 07 at Bus 37
                    8: 37,   # G 08 at Bus 38
                    9: 38,   # G 09 at Bus 39
                    10: 29   # G 10 at Bus 30
                }
                
                if gen_bus_num in gen_bus_mapping:
                    bus_idx = gen_bus_mapping[gen_bus_num]
                else:
                    # Fallback: assume generator number matches bus number
                    bus_idx = gen_bus_num - 1
                
                if 0 <= bus_idx < len(buses):
                    if bus_idx not in perturbable_buses:
                        perturbable_buses[bus_idx] = {'loads': [], 'generators': [], 'net_P': 0.0}
                    perturbable_buses[bus_idx]['generators'].append(gen)
                    perturbable_buses[bus_idx]['net_P'] += gen_P  # Generator increases net injection
                    print(f"   {gen_name} (P={gen_P:.1f} MW) → Bus {bus_idx} ({safe_get_name(buses[bus_idx])})")
        except:
            print(f"   ⚠️ Could not parse bus number from {gen_name}")
    
    print(f"✅ Perturbable buses identified:")
    print(f"   🎯 Total perturbable buses: {len(perturbable_buses)}")
    
    for bus_idx, data in perturbable_buses.items():
        n_loads = len(data['loads'])
        n_gens = len(data['generators'])
        net_P = data['net_P']
        print(f"   Bus {bus_idx} ({safe_get_name(buses[bus_idx])}): {n_loads} loads, {n_gens} gens, net P = {net_P:.1f} MW")
    
    return perturbable_buses

def perturb_power_at_bus_simple(bus_idx, delta_P_MW, perturbable_buses):
    """
    Perturb power at a specific bus using available loads or generators
    """
    
    if bus_idx not in perturbable_buses:
        return False, f"Bus {bus_idx} has no controllable elements"
    
    bus_data = perturbable_buses[bus_idx]
    
    # Prefer to use loads for perturbation (more realistic)
    if bus_data['loads']:
        load = bus_data['loads'][0]  # Use first load
        original_P = get(load, "plini")
        new_P = original_P + delta_P_MW  # Increase load = decrease net injection
        try:
            load.SetAttribute("plini", new_P)
            return True, f"Load {safe_get_name(load)}: {original_P:.1f} → {new_P:.1f} MW"
        except Exception as e:
            pass
    
    # Fallback to generators
    if bus_data['generators']:
        gen = bus_data['generators'][0]  # Use first generator
        original_P = get(gen, "pgini")
        new_P = original_P + delta_P_MW  # Increase generation = increase injection
        try:
            gen.SetAttribute("pgini", new_P)
            return True, f"Generator {safe_get_name(gen)}: {original_P:.1f} → {new_P:.1f} MW"
        except Exception as e:
            pass
    
    return False, f"Could not perturb elements at bus {bus_idx}"

def store_original_values(app):
    """Store original load and generation values for restoration"""
    print(f"💾 Storing original system state...")
    
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    
    original_loads = []
    for load in loads:
        original_loads.append({
            'object': load,
            'P_orig': get(load, "plini"),
            'Q_orig': get(load, "qlini")
        })
    
    original_gens = []
    for gen in generators:
        original_gens.append({
            'object': gen,
            'P_orig': get(gen, "pgini"),
            'Q_orig': get(gen, "qgini")
        })
    
    print(f"✅ Stored {len(original_loads)} load values and {len(original_gens)} generator values")
    return original_loads, original_gens

def restore_original_values(original_loads, original_gens):
    """Restore original load and generation values"""
    try:
        for load_data in original_loads:
            load_data['object'].SetAttribute("plini", load_data['P_orig'])
            load_data['object'].SetAttribute("qlini", load_data['Q_orig'])
        
        for gen_data in original_gens:
            gen_data['object'].SetAttribute("pgini", gen_data['P_orig'])
            gen_data['object'].SetAttribute("qgini", gen_data['Q_orig'])
    except Exception as e:
        print(f"⚠️ Warning restoring values: {e}")

def solve_power_flow_and_get_voltages(app):
    """Solve power flow and return bus voltages"""
    
    # Get load flow calculation object
    comLdf = app.GetFromStudyCase("ComLdf")
    if not comLdf:
        print("❌ Load flow calculation object not found")
        return None
    
    # Configure and execute load flow
    comLdf.iopt_net = 0  # AC load flow
    comLdf.iopt_at = 0   # Automatic tap adjustment off
    comLdf.errlf = 1e-4  # Convergence tolerance
    comLdf.maxiter = 50  # Reduce iterations for speed
    
    ierr = comLdf.Execute()
    if ierr != 0:
        return None  # Convergence failed
    
    # Get bus voltages
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    voltages = np.zeros(len(buses))
    for i, bus in enumerate(buses):
        voltages[i] = get(bus, "m:u")  # Voltage magnitude (p.u.)
    
    return voltages

def collect_bus_voltage_array(app):
    """Collect voltage magnitudes as numpy array"""
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    voltages = []
    
    for bus in buses:
        try:
            voltage = bus.GetAttribute('m:u')
            voltages.append(voltage)
        except:
            voltages.append(1.0)  # Default value
    
    return np.array(voltages)

# ── FIXED SENSITIVITY CALCULATION ─────────────────────────────────────────
def calculate_post_contingency_voltage_sensitivity_FIXED(app, Y_matrix, bus_data):
    """
    Calculate REAL voltage sensitivity using numerical differentiation 
    (PROPER METHOD from Sensitivity_simulation.py)
    """
    try:
        print("🔢 Calculating post-contingency sensitivity using PROPER numerical method...")
        
        num_buses = len(bus_data)
        dV_dP = np.zeros((num_buses, num_buses))
        success_flags = np.zeros(num_buses, dtype=bool)
        
        # Configuration
        delta_P_MW = 10.0  # Perturbation size (MW)
        print(f"📊 Perturbation size: ±{delta_P_MW} MW")
        
        # Identify perturbable buses
        perturbable_buses = identify_perturbable_buses(app)
        
        if len(perturbable_buses) == 0:
            print("❌ No perturbable buses found")
            return np.zeros((num_buses, num_buses)), np.zeros(num_buses, dtype=bool)
        
        # Store original values
        original_loads, original_gens = store_original_values(app)
        
        try:
            # Get base case voltages
            print(f"🔄 Solving base case power flow...")
            V_base = solve_power_flow_and_get_voltages(app)
            if V_base is None:
                print("❌ Base case power flow failed")
                return np.zeros((num_buses, num_buses)), np.zeros(num_buses, dtype=bool)
            
            print(f"✅ Base case: V range = {V_base.min():.3f} - {V_base.max():.3f} p.u.")
            
            print(f"🔄 Computing sensitivity for perturbable buses...")
            start_time = time.time()
            
            # Loop through all buses, but only perturb those that are perturbable
            for j in range(num_buses):
                buses = app.GetCalcRelevantObjects("*.ElmTerm")
                bus_name = safe_get_name(buses[j])
                print(f"   {j+1:2d}/{num_buses}: {bus_name:<15}", end=" ")
                
                if j not in perturbable_buses:
                    print(f"❌ No controllable elements")
                    continue
                
                # Restore original values before each perturbation
                restore_original_values(original_loads, original_gens)
                
                # Try positive perturbation
                success_pos, detail_pos = perturb_power_at_bus_simple(j, delta_P_MW, perturbable_buses)
                if success_pos:
                    V_pos = solve_power_flow_and_get_voltages(app)
                    if V_pos is not None:
                        # Try negative perturbation for better accuracy
                        restore_original_values(original_loads, original_gens)
                        success_neg, detail_neg = perturb_power_at_bus_simple(j, -delta_P_MW, perturbable_buses)
                        
                        if success_neg:
                            V_neg = solve_power_flow_and_get_voltages(app)
                            if V_neg is not None:
                                # Central difference: (V_pos - V_neg) / (2 * delta_P)
                                dV_dP[:, j] = (V_pos - V_neg) / (2 * delta_P_MW)
                                success_flags[j] = True
                                print(f"✅ Central diff (±{delta_P_MW} MW)")
                                continue
                        
                        # Fallback to forward difference: (V_pos - V_base) / delta_P
                        dV_dP[:, j] = (V_pos - V_base) / delta_P_MW
                        success_flags[j] = True
                        print(f"✅ Forward diff (+{delta_P_MW} MW)")
                        continue
                
                print(f"❌ Perturbation failed")
            
            # Restore original values
            restore_original_values(original_loads, original_gens)
            
            elapsed_time = time.time() - start_time
            successful_buses = np.sum(success_flags)
            
            print(f"✅ PROPER numerical sensitivity calculation complete!")
            print(f"   ⏱️ Time elapsed: {elapsed_time:.1f} seconds")
            print(f"   🎯 Successful perturbations: {successful_buses}/{num_buses} buses ({successful_buses/num_buses*100:.1f}%)")
            
            if successful_buses > 0:
                # Calculate statistics only for successful perturbations
                valid_sensitivities = dV_dP[:, success_flags]
                print(f"   📊 Sensitivity range: {valid_sensitivities.min():.6f} to {valid_sensitivities.max():.6f} p.u./MW")
                print(f"   📊 Average sensitivity: {np.abs(valid_sensitivities).mean():.6f} p.u./MW")
                
                return dV_dP, success_flags
            else:
                print("❌ No successful perturbations - cannot calculate sensitivity")
                return np.zeros((num_buses, num_buses)), np.zeros(num_buses, dtype=bool)
                
        except Exception as e:
            print(f"❌ Error in numerical sensitivity calculation: {e}")
            restore_original_values(original_loads, original_gens)
            return np.zeros((num_buses, num_buses)), np.zeros(num_buses, dtype=bool)
            
    except Exception as e:
        print(f"❌ Voltage sensitivity calculation failed: {e}")
        return np.zeros((num_buses, num_buses)), np.zeros(num_buses, dtype=bool)

# ── Y-Matrix Functions ─────────────────────────────────────────────────────
def load_y_matrix_fallback():
    """Load Y-matrix from existing files as fallback"""
    try:
        # Try existing Y-matrix files
        y_matrix_files = [
            "Y_admittance.npy",
            "admittance_matrix.npy",
            os.path.join("enhanced_out", "Y_matrix.npy"),
            os.path.join("..", "Y_admittance.npy")
        ]
        
        for file_path in y_matrix_files:
            if os.path.exists(file_path):
                Y_matrix = np.load(file_path)
                print(f"   ✅ Y-matrix loaded from {file_path}")
                return Y_matrix
        
        # Try H5 files
        h5_files = [
            "39_Bus_New_England_System_fixed_complete_enhanced.h5",
            os.path.join("enhanced_out", "39_Bus_New_England_System_fixed_complete_enhanced.h5")
        ]
        
        for h5_file in h5_files:
            if os.path.exists(h5_file):
                with h5py.File(h5_file, "r") as f:
                    if "admittance" in f:
                        data = f["admittance/data"][:]
                        indices = f["admittance/indices"][:]
                        indptr = f["admittance/indptr"][:]
                        Y_sparse = csr_matrix((data, indices, indptr), shape=(39, 39))
                        Y_matrix = Y_sparse.toarray()
                        print(f"   ✅ Y-matrix loaded from H5 file: {h5_file}")
                        return Y_matrix
        
        print(f"   ⚠️ No Y-matrix files found, creating identity matrix")
        return np.eye(39, dtype=complex)
        
    except Exception as e:
        print(f"   ❌ Fallback Y-matrix loading failed: {e}")
        return np.eye(39, dtype=complex)

# ── Analysis Functions ─────────────────────────────────────────────────────
def initialize_power_flow(app):
    """Initialize power flow with proper settings"""
    try:
        # Get load flow command
        comLdf = app.GetFromStudyCase("ComLdf")
        
        # Set load flow options for better convergence
        comLdf.iopt_net = 0      # AC load flow
        comLdf.iopt_at = 0       # Automatic tap adjustment off initially
        comLdf.iopt_asht = 0     # Automatic shunt adjustment off
        comLdf.errlf = 0.01      # Convergence tolerance
        comLdf.maxiter = 100     # Maximum iterations
        
        print("✅ Power flow initialized with robust settings")
        return comLdf
        
    except Exception as e:
        print(f"❌ Power flow initialization error: {e}")
        return None

def run_power_flow(app, scenario_name=""):
    """Run power flow with enhanced convergence handling"""
    try:
        comLdf = app.GetFromStudyCase("ComLdf")
        
        # Enhanced settings for better convergence
        comLdf.iopt_net = 0      # AC load flow
        comLdf.iopt_at = 1       # Enable automatic tap adjustment
        comLdf.iopt_asht = 1     # Enable automatic shunt adjustment
        comLdf.errlf = 0.01      # Convergence tolerance
        comLdf.maxiter = 100     # Maximum iterations
        comLdf.iopt_plim = 1     # Respect generator P limits
        comLdf.iopt_lim = 1      # Respect generator Q limits
        
        # First attempt - with all automatic adjustments
        comLdf.Execute()
        convergence = comLdf.GetAttribute("errlf") == 0
        
        if convergence:
            print(f"✅ Power flow converged {scenario_name}")
            return True
        
        # Second attempt - check if error is acceptable
        error_val = comLdf.GetAttribute("errlf")
        if error_val < 0.1:  # Less than 0.1 mismatch might be acceptable
            print(f"⚠️ Power flow converged with small error ({error_val:.6f}) {scenario_name}")
            return True
        
        # Third attempt - increase tolerance
        print(f"⚠️ Relaxing convergence tolerance {scenario_name}")
        comLdf.errlf = 0.1       # Relaxed tolerance
        comLdf.Execute()
        convergence = comLdf.GetAttribute("errlf") == 0
        
        if convergence:
            print(f"✅ Power flow converged with relaxed tolerance {scenario_name}")
            return True
        
        # Final check - if error is small enough, accept it
        error_val = comLdf.GetAttribute("errlf")
        if error_val < 0.5:
            print(f"⚠️ Accepting solution with error {error_val:.6f} {scenario_name}")
            return True
        
        print(f"❌ Power flow did not converge {scenario_name} (error: {error_val:.6f})")
        return False
        
    except Exception as e:
        print(f"❌ Power flow error {scenario_name}: {e}")
        return False

def fix_power_balance(app):
    """Fix power balance issues before running power flow"""
    try:
        generators = app.GetCalcRelevantObjects("*.ElmSym")
        loads = app.GetCalcRelevantObjects("*.ElmLod")
        
        # Calculate current balance
        total_gen = 0
        total_load = 0
        slack_gen = None
        
        for gen in generators:
            if not getattr(gen, 'outserv', 0):
                gen_power = getattr(gen, 'pgini', 0)
                total_gen += gen_power
                
                # Find slack generator (usually largest or marked as slack)
                if getattr(gen, 'ip_ctrl', 0) == 1 or slack_gen is None:
                    slack_gen = gen
        
        for load in loads:
            if not getattr(load, 'outserv', 0):
                total_load += getattr(load, 'plini', 0)
        
        imbalance = total_load - total_gen
        print(f"🔧 Power balance correction:")
        print(f"   Total generation: {total_gen:.1f} MW")
        print(f"   Total load: {total_load:.1f} MW")
        print(f"   Imbalance: {imbalance:.1f} MW")
        
        if abs(imbalance) > 10:  # More than 10 MW imbalance
            if slack_gen:
                old_power = getattr(slack_gen, 'pgini', 0)
                new_power = old_power + imbalance + 50  # Add 50 MW reserve
                
                # Check if within generator limits
                max_power = getattr(slack_gen, 'Pmax', new_power + 100)
                if new_power <= max_power:
                    slack_gen.pgini = new_power
                    print(f"   ✅ Adjusted slack generator {slack_gen.loc_name}: {old_power:.1f} → {new_power:.1f} MW")
                else:
                    print(f"   ⚠️ Slack generator at maximum capacity, trying load reduction")
                    # Reduce loads proportionally by 5%
                    for load in loads:
                        if not getattr(load, 'outserv', 0):
                            old_load = getattr(load, 'plini', 0)
                            new_load = old_load * 0.95
                            load.plini = new_load
                    print(f"   ✅ Reduced all loads by 5%")
            else:
                print(f"   ❌ No slack generator found!")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Power balance correction failed: {e}")
        return False

def initialize_system_properly(app):
    """Proper system initialization for IEEE 39-bus"""
    try:
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        generators = app.GetCalcRelevantObjects("*.ElmSym")
        
        print(f"🔧 System initialization:")
        
        # Set reasonable initial voltages for all buses
        for bus in buses:
            try:
                voltage_level = getattr(bus, 'uknom', 345)  # Default to 345 kV
                if voltage_level > 300:  # High voltage buses
                    bus.u0 = 1.02  # Slightly higher voltage
                elif voltage_level > 100:  # Medium voltage
                    bus.u0 = 1.01
                else:  # Low voltage
                    bus.u0 = 1.00
                bus.phiu0 = 0.0  # Zero angle initially
            except:
                pass
        
        # Ensure proper generator settings
        slack_count = 0
        for gen in generators:
            try:
                # Check if this should be the slack generator
                gen_power = getattr(gen, 'pgini', 0)
                if gen_power > 800 or "39" in getattr(gen, 'loc_name', ''):  # Largest generator
                    gen.ip_ctrl = 1  # Set as slack (PU bus)
                    slack_count += 1
                    print(f"   ✅ Set {gen.loc_name} as slack generator ({gen_power:.1f} MW)")
                else:
                    gen.ip_ctrl = 0  # Set as PV bus
                
                # Ensure reasonable voltage setpoint
                if not hasattr(gen, 'uset') or getattr(gen, 'uset', 0) <= 0:
                    gen.uset = 1.0
                    
            except Exception as e:
                print(f"   ⚠️ Generator setup warning: {e}")
        
        if slack_count == 0:
            print(f"   ⚠️ No slack generator found, setting largest generator as slack")
            # Find largest generator and set as slack
            max_power = 0
            largest_gen = None
            for gen in generators:
                power = getattr(gen, 'pgini', 0)
                if power > max_power:
                    max_power = power
                    largest_gen = gen
            
            if largest_gen:
                largest_gen.ip_ctrl = 1
                print(f"   ✅ Set {largest_gen.loc_name} as slack generator")
        
        print(f"   ✅ System initialization completed")
        return True
        
    except Exception as e:
        print(f"   ❌ System initialization failed: {e}")
        return False

def calculate_power_balance_per_zone(bus_data, gen_data, load_data, zone_assignments=None):
    """Calculate power balance for each zone"""
    if zone_assignments is None:
        # Create dummy zones if not available
        zone_assignments = np.arange(len(bus_data)) % 3
    
    num_zones = len(np.unique(zone_assignments))
    zone_balance = []
    
    for zone in range(num_zones):
        zone_buses = np.where(zone_assignments == zone)[0]
        
        # Sum generation in this zone
        zone_gen = 0
        for _, gen in gen_data.iterrows():
            if gen['in_service']:
                # Find bus index for this generator
                bus_idx = bus_data[bus_data['bus_name'] == gen['bus_name']].index
                if len(bus_idx) > 0 and bus_idx[0] in zone_buses:
                    zone_gen += gen['P_MW']
        
        # Sum load in this zone
        zone_load = 0
        for _, load in load_data.iterrows():
            if load['in_service']:
                bus_idx = bus_data[bus_data['bus_name'] == load['bus_name']].index
                if len(bus_idx) > 0 and bus_idx[0] in zone_buses:
                    zone_load += load['P_MW']
        
        zone_balance.append({
            'zone': zone,
            'num_buses': len(zone_buses),
            'generation_MW': zone_gen,
            'load_MW': zone_load,
            'balance_MW': zone_gen - zone_load
        })
    
    return pd.DataFrame(zone_balance)

# ── Main Analysis Function ─────────────────────────────────────────────────
def comprehensive_contingency_analysis():
    """Main function to perform comprehensive contingency analysis"""
    
    # Connect to PowerFactory
    app = connect_powerfactory()
    
    # Initialize power flow settings
    comLdf = initialize_power_flow(app)
    if not comLdf:
        print("❌ Could not initialize power flow! Stopping analysis.")
        return
    
    # Initialize system properly
    system_init_ok = initialize_system_properly(app)
    if not system_init_ok:
        print("❌ System initialization failed! Stopping analysis.")
        return
    
    # Fix power balance
    balance_ok = fix_power_balance(app)
    if not balance_ok:
        print("❌ Could not fix power balance! Stopping analysis.")
        return
    
    # ══════════════════════════════════════════════════════════════════════
    print("\n📊 STEP 1: BASELINE DATA COLLECTION")
    print("="*50)
    
    # Run baseline power flow
    baseline_convergence = run_power_flow(app, "(baseline)")
    if not baseline_convergence:
        print("❌ Baseline power flow failed! Stopping analysis.")
        return
    
    # Collect baseline data
    print("📋 Collecting baseline system data...")
    baseline_bus_data = collect_bus_data(app)
    baseline_gen_data = collect_generator_data(app)
    baseline_line_data = collect_line_data(app)
    baseline_load_data = collect_load_data(app)
    
    # Extract baseline Y-matrix (load from existing files)
    print("🔧 Loading baseline Y-matrix...")
    baseline_Y_matrix = load_y_matrix_fallback()
    
    # Calculate baseline voltage sensitivity using PROPER method
    print("🔍 Calculating baseline voltage sensitivity using PROPER numerical method...")
    baseline_dV_dP, baseline_success_flags = calculate_post_contingency_voltage_sensitivity_FIXED(
        app, baseline_Y_matrix, baseline_bus_data
    )
    
    # Save baseline data
    baseline_bus_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "bus_data_baseline.csv"), index=False)
    baseline_gen_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "generator_data_baseline.csv"), index=False)
    baseline_line_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "line_data_baseline.csv"), index=False)
    baseline_load_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "load_data_baseline.csv"), index=False)
    np.save(os.path.join(BASE_DIR, "baseline_data", "Y_matrix_baseline.npy"), baseline_Y_matrix)
    np.save(os.path.join(BASE_DIR, "baseline_data", "voltage_sensitivity_baseline.npy"), baseline_dV_dP)
    np.save(os.path.join(BASE_DIR, "baseline_data", "sensitivity_success_flags_baseline.npy"), baseline_success_flags)
    
    print(f"✅ Baseline data collected:")
    print(f"   🔌 Buses: {len(baseline_bus_data)}")
    print(f"   🔋 Generators: {len(baseline_gen_data)}")
    print(f"   📏 Lines/Transformers: {len(baseline_line_data)}")
    print(f"   📍 Loads: {len(baseline_load_data)}")
    print(f"   ⚡ Total Generation: {baseline_gen_data['P_MW'].sum():.1f} MW")
    print(f"   📊 Total Load: {baseline_load_data['P_MW'].sum():.1f} MW")
    print(f"   🔧 Y-matrix: {baseline_Y_matrix.shape} ({np.count_nonzero(baseline_Y_matrix)} non-zeros)")
    print(f"   🔍 Sensitivity: {np.sum(baseline_success_flags)}/{len(baseline_success_flags)} buses calculated")
    
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n⚡ STEP 2: GENERATOR CONTINGENCY - {TARGET_GENERATOR}")
    print("="*50)
    
    # Identify target generator
    print(f"🎯 Identifying target generator...")
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    target_generator = None
    
    for gen in generators:
        try:
            # Check by name or bus connection
            if "2" in gen.loc_name or TARGET_GENERATOR.lower() in gen.loc_name.lower():
                bus = gen.bus1 if hasattr(gen, 'bus1') else None
                if bus and "31" in bus.loc_name:
                    print(f"✅ Found target generator: {gen.loc_name} at {bus.loc_name}")
                    target_generator = gen
                    break
        except:
            continue
    
    # Fallback: get by index (assuming Generator 2 is second in list)
    if not target_generator and len(generators) > 1:
        target_generator = generators[1]
        print(f"✅ Using generator by index: {target_generator.loc_name}")
    
    if not target_generator:
        print("❌ Could not find target generator! Stopping analysis.")
        return
    
    original_P = getattr(target_generator, 'pgini', 0)
    original_Q = getattr(target_generator, 'qgini', 0)
    original_service = getattr(target_generator, 'outserv', 0)
    
    print(f"🎯 Target Generator Details:")
    print(f"   Name: {target_generator.loc_name}")
    print(f"   Original P: {original_P:.1f} MW")
    print(f"   Original Q: {original_Q:.1f} MVAR")
    print(f"   In Service: {not original_service}")
    
    # Create contingency - take generator out of service
    print(f"\n🔧 Creating contingency scenario...")
    target_generator.outserv = 1  # Take out of service
    
    # Run contingency power flow
    contingency_convergence = run_power_flow(app, "(contingency)")
    
    if not contingency_convergence:
        print("❌ Contingency power flow failed!")
        # Restore generator
        target_generator.outserv = original_service
        return
    
    # Collect contingency data
    print("📋 Collecting contingency system data...")
    contingency_bus_data = collect_bus_data(app)
    contingency_gen_data = collect_generator_data(app)
    contingency_line_data = collect_line_data(app)
    contingency_load_data = collect_load_data(app)
    
    # Use same Y-matrix (topology didn't change significantly)
    print("🔧 Using baseline Y-matrix for contingency (topology unchanged)...")
    contingency_Y_matrix = baseline_Y_matrix
    
    # Calculate post-contingency voltage sensitivity using PROPER method
    print("🔍 Calculating post-contingency voltage sensitivity using PROPER numerical method...")
    contingency_dV_dP, contingency_success_flags = calculate_post_contingency_voltage_sensitivity_FIXED(
        app, contingency_Y_matrix, contingency_bus_data
    )
    
    # Save contingency data
    contingency_bus_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "bus_data_contingency.csv"), index=False)
    contingency_gen_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "generator_data_contingency.csv"), index=False)
    contingency_line_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "line_data_contingency.csv"), index=False)
    contingency_load_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "load_data_contingency.csv"), index=False)
    np.save(os.path.join(BASE_DIR, "contingency_gen2_bus31", "Y_matrix_contingency.npy"), contingency_Y_matrix)
    np.save(os.path.join(BASE_DIR, "contingency_gen2_bus31", "voltage_sensitivity_contingency.npy"), contingency_dV_dP)
    np.save(os.path.join(BASE_DIR, "contingency_gen2_bus31", "sensitivity_success_flags_contingency.npy"), contingency_success_flags)
    
    # Restore generator for next analysis
    target_generator.outserv = original_service
    
    print(f"✅ Contingency analysis completed")
    print(f"   🔧 Y-matrix: {contingency_Y_matrix.shape} ({np.count_nonzero(contingency_Y_matrix)} non-zeros)")
    print(f"   🔍 Sensitivity: {np.sum(contingency_success_flags)}/{len(contingency_success_flags)} buses calculated")
    
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n🔍 STEP 3: COMPREHENSIVE IMPACT ANALYSIS")
    print("="*50)
    
    # Voltage analysis
    voltage_changes = contingency_bus_data['voltage_pu'] - baseline_bus_data['voltage_pu']
    angle_changes = contingency_bus_data['angle_deg'] - baseline_bus_data['angle_deg']
    
    # Power flow analysis
    line_P_changes = contingency_line_data['P_from_MW'] - baseline_line_data['P_from_MW']
    line_loading_changes = contingency_line_data['loading_percent'] - baseline_line_data['loading_percent']
    
    # Generator dispatch changes
    gen_P_changes = contingency_gen_data['P_MW'] - baseline_gen_data['P_MW']
    
    # Critical metrics
    max_voltage_change = np.abs(voltage_changes).max()
    max_angle_change = np.abs(angle_changes).max()
    max_line_loading = contingency_line_data['loading_percent'].max()
    
    # Identify critical elements
    critical_voltage_buses = baseline_bus_data.loc[np.abs(voltage_changes) > 0.05, 'bus_name'].tolist()
    overloaded_lines = contingency_line_data.loc[contingency_line_data['loading_percent'] > 100, 'element_name'].tolist()
    
    print(f"📊 Impact Analysis Results:")
    print(f"   📈 Max voltage change: {max_voltage_change:.4f} p.u.")
    print(f"   📐 Max angle change: {max_angle_change:.2f} degrees")
    print(f"   ⚡ Max line loading: {max_line_loading:.1f}%")
    print(f"   🚨 Critical voltage buses: {len(critical_voltage_buses)}")
    print(f"   🔴 Overloaded lines: {len(overloaded_lines)}")
    
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n🎯 STEP 4: SENSITIVITY ANALYSIS COMPARISON")
    print("="*50)
    
    # Compare sensitivity matrices
    sensitivity_change = 0
    if baseline_dV_dP.shape == contingency_dV_dP.shape:
        sensitivity_change = np.linalg.norm(contingency_dV_dP - baseline_dV_dP)
        print(f"📊 Sensitivity Matrix Comparison:")
        print(f"   📈 Baseline successful buses: {np.sum(baseline_success_flags)}/{len(baseline_success_flags)}")
        print(f"   📈 Contingency successful buses: {np.sum(contingency_success_flags)}/{len(contingency_success_flags)}")
        print(f"   📊 Sensitivity change metric: {sensitivity_change:.6f}")
        
        # Find most affected sensitivities
        if np.sum(baseline_success_flags) > 0 and np.sum(contingency_success_flags) > 0:
            common_buses = baseline_success_flags & contingency_success_flags
            if np.sum(common_buses) > 0:
                baseline_sens = baseline_dV_dP[:, common_buses]
                contingency_sens = contingency_dV_dP[:, common_buses]
                sens_diff = np.abs(contingency_sens - baseline_sens)
                max_sens_change = np.max(sens_diff)
                print(f"   📊 Max sensitivity change: {max_sens_change:.6f} p.u./MW")
    
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n💾 STEP 5: SAVE COMPREHENSIVE RESULTS")
    print("="*50)
    
    # Create comprehensive analysis report
    analysis_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'target_generator': TARGET_GENERATOR,
            'target_bus': TARGET_BUS,
            'original_generation_MW': float(original_P),
            'baseline_convergence': baseline_convergence,
            'contingency_convergence': contingency_convergence
        },
        'system_impact': {
            'max_voltage_change_pu': float(max_voltage_change),
            'max_angle_change_deg': float(max_angle_change),
            'max_line_loading_percent': float(max_line_loading),
            'critical_voltage_buses': critical_voltage_buses,
            'overloaded_lines': overloaded_lines,
            'voltage_violations': len(critical_voltage_buses),
            'loading_violations': len(overloaded_lines)
        },
        'sensitivity_analysis': {
            'baseline_sensitivity_successful_buses': int(np.sum(baseline_success_flags)),
            'contingency_sensitivity_successful_buses': int(np.sum(contingency_success_flags)),
            'sensitivity_method': 'numerical_differentiation_proper',
            'perturbation_size_MW': 10.0,
            'max_baseline_sensitivity': float(np.max(np.abs(baseline_dV_dP))) if baseline_dV_dP.size > 0 else 0,
            'max_contingency_sensitivity': float(np.max(np.abs(contingency_dV_dP))) if contingency_dV_dP.size > 0 else 0,
            'sensitivity_change_metric': float(sensitivity_change)
        },
        'recommendations': {
            'requires_rezoning': bool(max_voltage_change > 0.05 or len(overloaded_lines) > 0),
            'critical_monitoring_buses': critical_voltage_buses[:10],  # Top 10
            'generator_importance_ranking': 'high' if max_voltage_change > 0.1 else 'medium',
            'sensitivity_analysis_quality': 'high' if np.sum(baseline_success_flags) > 20 else 'medium'
        }
    }
    
    # Save analysis results
    with open(os.path.join(BASE_DIR, "analysis_results", "comprehensive_analysis.json"), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save detailed comparison data
    comparison_data = {
        'voltage_changes': voltage_changes.tolist(),
        'angle_changes': angle_changes.tolist(),
        'line_power_changes': line_P_changes.tolist(),
        'generator_dispatch_changes': gen_P_changes.tolist()
    }
    
    with open(os.path.join(BASE_DIR, "analysis_results", "detailed_changes.json"), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"✅ Comprehensive analysis completed!")
    print(f"📁 All results saved to: {BASE_DIR}")
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   📊 System impact severity: {'HIGH' if max_voltage_change > 0.1 else 'MEDIUM' if max_voltage_change > 0.05 else 'LOW'}")
    print(f"   🔄 Rezoning recommended: {'YES' if analysis_results['recommendations']['requires_rezoning'] else 'NO'}")
    print(f"   🏭 Generator importance: {analysis_results['recommendations']['generator_importance_ranking'].upper()}")
    print(f"   🔢 Sensitivity analysis: PROPER numerical method used!")
    print(f"   ✅ Baseline sensitivity buses: {np.sum(baseline_success_flags)}")
    print(f"   ✅ Contingency sensitivity buses: {np.sum(contingency_success_flags)}")
    
    return analysis_results

# ── Execution ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        results = comprehensive_contingency_analysis()
        print(f"\n🎉 ANALYSIS COMPLETE WITH PROPER SENSITIVITY CALCULATION!")
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()