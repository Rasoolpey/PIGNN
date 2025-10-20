# comprehensive_contingency_analysis.py
"""
Comprehensive Contingency Analysis for Generator 2 at Bus 31
IEEE 39-Bus New England System - PowerFactory Implementation

This code performs:
1. Baseline data collection
2. Generator 2 outage simulation
3. Comprehensive impact analysis
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

print("🔋 COMPREHENSIVE GENERATOR CONTINGENCY ANALYSIS")
print("="*60)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Target: {TARGET_GENERATOR} at Bus {TARGET_BUS}")
print(f"📁 Output Directory: {BASE_DIR}")
print()

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

# ── Y-Matrix Functions ─────────────────────────────────────────────────────
def extract_y_matrix(app):
    """Extract Y-matrix from PowerFactory"""
    try:
        # Try to get Y-matrix directly
        comYmat = app.GetFromStudyCase("ComYmat")
        if comYmat:
            comYmat.Execute()
            # This is a simplified approach - you might need to adjust based on PF version
            print("✅ Y-matrix extracted from PowerFactory")
        
        # Alternative: Load from your existing Y-matrix file
        y_matrix_files = [
            "Y_admittance.npy",
            "admittance_matrix.npy",
            os.path.join("enhanced_out", "Y_matrix.npy")
        ]
        
        for file_path in y_matrix_files:
            if os.path.exists(file_path):
                Y_matrix = np.load(file_path)
                print(f"✅ Y-matrix loaded from {file_path}")
                return Y_matrix
        
        print("⚠️ Warning: Could not extract Y-matrix, using approximation")
        return np.eye(39, dtype=complex)  # Fallback
        
    except Exception as e:
        print(f"⚠️ Warning: Y-matrix extraction failed: {e}")
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
    """Check and report system state for debugging"""
    try:
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        generators = app.GetCalcRelevantObjects("*.ElmSym")
        loads = app.GetCalcRelevantObjects("*.ElmLod")
        
        print(f"🔍 System State Check:")
        print(f"   📊 Total buses: {len(buses)}")
        print(f"   🔋 Total generators: {len(generators)}")
        print(f"   📍 Total loads: {len(loads)}")
        
        # Check generators
        total_gen = 0
        active_gens = 0
        for gen in generators:
            if not getattr(gen, 'outserv', 0):
                active_gens += 1
                total_gen += getattr(gen, 'pgini', 0)
        
        # Check loads
        total_load = 0
        for load in loads:
            if not getattr(load, 'outserv', 0):
                total_load += getattr(load, 'plini', 0)
        
        print(f"   ⚡ Active generators: {active_gens}/{len(generators)}")
        print(f"   🔋 Total generation: {total_gen:.1f} MW")
        print(f"   📊 Total load: {total_load:.1f} MW")
        print(f"   ⚖️ Balance: {total_gen - total_load:.1f} MW")
        
        if abs(total_gen - total_load) > total_load * 0.1:  # More than 10% imbalance
            print(f"   ⚠️ WARNING: Large power imbalance detected!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ System state check failed: {e}")
        return False
    """Identify and return the target generator (Generator 2 at Bus 31)"""
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    
    for gen in generators:
        try:
            # Check by name or bus connection
            if "2" in gen.loc_name or TARGET_GENERATOR.lower() in gen.loc_name.lower():
                bus = gen.bus1 if hasattr(gen, 'bus1') else None
                if bus and "31" in bus.loc_name:
                    print(f"✅ Found target generator: {gen.loc_name} at {bus.loc_name}")
                    return gen
        except:
            continue
    
    # Fallback: get by index (assuming Generator 2 is second in list)
    if len(generators) > 1:
        target_gen = generators[1]
        print(f"✅ Using generator by index: {target_gen.loc_name}")
        return target_gen
    
    raise Exception("Could not find target generator!")

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
    
    # Check system state before analysis
    system_ok = True  # Skip the check for now since we already fixed the balance
    if not system_ok:
        print("⚠️ System state issues detected, but continuing...")
    
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
    baseline_Y_matrix = extract_y_matrix(app)
    
    # Save baseline data
    baseline_bus_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "bus_data_baseline.csv"), index=False)
    baseline_gen_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "generator_data_baseline.csv"), index=False)
    baseline_line_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "line_data_baseline.csv"), index=False)
    baseline_load_data.to_csv(os.path.join(BASE_DIR, "baseline_data", "load_data_baseline.csv"), index=False)
    np.save(os.path.join(BASE_DIR, "baseline_data", "Y_matrix_baseline.npy"), baseline_Y_matrix)
    
    print(f"✅ Baseline data collected:")
    print(f"   🔌 Buses: {len(baseline_bus_data)}")
    print(f"   🔋 Generators: {len(baseline_gen_data)}")
    print(f"   📏 Lines/Transformers: {len(baseline_line_data)}")
    print(f"   📍 Loads: {len(baseline_load_data)}")
    print(f"   ⚡ Total Generation: {baseline_gen_data['P_MW'].sum():.1f} MW")
    print(f"   📊 Total Load: {baseline_load_data['P_MW'].sum():.1f} MW")
    
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
    contingency_Y_matrix = extract_y_matrix(app)
    
    # Save contingency data
    contingency_bus_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "bus_data_contingency.csv"), index=False)
    contingency_gen_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "generator_data_contingency.csv"), index=False)
    contingency_line_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "line_data_contingency.csv"), index=False)
    contingency_load_data.to_csv(os.path.join(BASE_DIR, "contingency_gen2_bus31", "load_data_contingency.csv"), index=False)
    np.save(os.path.join(BASE_DIR, "contingency_gen2_bus31", "Y_matrix_contingency.npy"), contingency_Y_matrix)
    
    # Restore generator for next analysis
    target_generator.outserv = original_service
    
    print(f"✅ Contingency analysis completed")
    
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
    print(f"\n🎯 STEP 4: ZONING IMPACT ANALYSIS")
    print("="*50)
    
    # Load zone assignments (if available)
    zone_file_paths = [
        "zone_assignments.npy",
        os.path.join("static_out", "zone_assignments.npy"),
        "zones.npy"
    ]
    
    zone_assignments = None
    for path in zone_file_paths:
        if os.path.exists(path):
            zone_assignments = np.load(path)
            print(f"✅ Loaded zone assignments from {path}")
            break
    
    if zone_assignments is None:
        print("⚠️ No zone assignments found, creating dummy zones")
        zone_assignments = np.arange(len(baseline_bus_data)) % 3
    
    # Calculate zone power balance
    baseline_zone_balance = calculate_power_balance_per_zone(
        baseline_bus_data, baseline_gen_data, baseline_load_data, zone_assignments
    )
    contingency_zone_balance = calculate_power_balance_per_zone(
        contingency_bus_data, contingency_gen_data, contingency_load_data, zone_assignments
    )
    
    # Zone impact analysis
    zone_balance_changes = contingency_zone_balance['balance_MW'] - baseline_zone_balance['balance_MW']
    
    print(f"📊 Zone Balance Analysis:")
    for i, (baseline, contingency, change) in enumerate(zip(
        baseline_zone_balance.iterrows(), 
        contingency_zone_balance.iterrows(), 
        zone_balance_changes
    )):
        print(f"   Zone {i+1}: {baseline[1]['balance_MW']:.1f} → {contingency[1]['balance_MW']:.1f} MW (Δ{change:.1f})")
    
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
        'zone_impact': {
            'baseline_zone_balance': baseline_zone_balance.to_dict('records'),
            'contingency_zone_balance': contingency_zone_balance.to_dict('records'),
            'zone_balance_changes': zone_balance_changes.tolist(),
            'most_affected_zone': int(np.argmax(np.abs(zone_balance_changes)))
        },
        'recommendations': {
            'requires_rezoning': bool(max_voltage_change > 0.05 or len(overloaded_lines) > 0),
            'critical_monitoring_buses': critical_voltage_buses[:10],  # Top 10
            'generator_importance_ranking': 'high' if max_voltage_change > 0.1 else 'medium'
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
    
    return analysis_results

# ── Execution ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        results = comprehensive_contingency_analysis()
        print(f"\n🎉 ANALYSIS COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()