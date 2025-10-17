# load_flow_data_collector.py - 2025-08-02
"""
Load Flow Data Collection Module for Post-Contingency Analysis
Following the same structure as Feature_Extraction.py and First_model.py

This module is called during contingency execution to collect comprehensive
post-contingency system data including:

1. IMPEDANCE DATA:
   - Lines: Direct impedance values
   - Transformers: From type objects (uk%, ukr%)
   - Generators: From type objects (xd, xq, etc.)
   - Loads: Calculated from V²/S (equivalent impedance)
   - Shunts: Calculated from V²/Q

2. LOAD FLOW DATA:
   - Bus voltages, angles, injections
   - Line/transformer loadings, losses
   - Generator outputs, reactive reserves
   - Load demands, power factors
   - System frequency, total losses

Called as: collect_load_flow_data() -> returns comprehensive data dictionary
"""

import sys, os, numpy as np
from datetime import datetime
import time

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

# ── Global settings ────────────────────────────────────────────────────────
SBASE_MVA = 100.0  # Base power for per-unit calculations
µS_to_S = 1e-6     # Microsiemens to Siemens conversion

# Helper functions (consistent with existing modules)
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

# ── Impedance extraction functions ─────────────────────────────────────────
def get_line_impedance(line_obj):
    """Extract line impedance in Ohms"""
    
    # Direct impedance values
    R = get(line_obj, "r_ohm")
    X = get(line_obj, "x_ohm")
    B = get(line_obj, "bch")  # Total charging susceptance
    
    # If not available, try per-km values
    if np.isnan(R) or np.isnan(X):
        Rkm = get(line_obj, "r_km")
        Xkm = get(line_obj, "x_km")
        length = get(line_obj, "dline")
        
        if not (np.isnan(Rkm) or np.isnan(Xkm) or np.isnan(length)):
            R = Rkm * length
            X = Xkm * length
    
    # Try type object if still missing
    if (np.isnan(R) or np.isnan(X)) and has(line_obj, "typ_id") and line_obj.typ_id:
        typ = line_obj.typ_id
        Rkm = get(typ, "rline")
        Xkm = get(typ, "xline")
        length = get(line_obj, "dline")
        
        if not (np.isnan(Rkm) or np.isnan(Xkm) or np.isnan(length)):
            R = Rkm * length
            X = Xkm * length
    
    return {
        'R_ohm': R,
        'X_ohm': X, 
        'B_uS': B,
        'Z_magnitude': np.sqrt(R**2 + X**2) if not (np.isnan(R) or np.isnan(X)) else np.nan,
        'impedance_source': 'line_direct'
    }

def get_transformer_impedance(trafo_obj):
    """Extract transformer impedance from type object with enhanced fallback methods"""
    
    impedance_data = {
        'R_ohm': np.nan,
        'X_ohm': np.nan,
        'Z_magnitude': np.nan,
        'tap_ratio': get(trafo_obj, "tratio", 1.0),
        'phase_shift_deg': get(trafo_obj, "phitr", 0.0),
        'impedance_source': 'transformer_type'
    }
    
    if not (has(trafo_obj, 'typ_id') and trafo_obj.typ_id):
        impedance_data['impedance_source'] = 'transformer_missing_type'
        return impedance_data
    
    typ = trafo_obj.typ_id
    
    # Method 1: Try standard uk/ukr parameters
    uk_percent = get(typ, "uk")      # Total impedance %
    ukr_percent = get(typ, "ukr")    # Resistance %
    S_rated = get(typ, "strn")       # Rated power MVA
    V_rated = get(typ, "utrn_h")     # HV voltage kV
    
    if not np.isnan(uk_percent) and uk_percent > 0:
        if not (np.isnan(S_rated) or np.isnan(V_rated)) and S_rated > 0 and V_rated > 0:
            Z_base = (V_rated ** 2) / S_rated
            z_pu = uk_percent / 100.0
            r_pu = ukr_percent / 100.0 if not np.isnan(ukr_percent) else 0.0
            x_pu = np.sqrt(max(0, z_pu**2 - r_pu**2))
            
            impedance_data.update({
                'R_ohm': r_pu * Z_base,
                'X_ohm': x_pu * Z_base,
                'Z_magnitude': z_pu * Z_base,
                'uk_percent': uk_percent,
                'ukr_percent': ukr_percent,
                'S_rated_MVA': S_rated,
                'V_rated_kV': V_rated,
                'impedance_source': 'transformer_type_calculated'
            })
            return impedance_data
    
    # Method 2: Try alternative parameter names (uktr found to work!)
    alt_params = [
        ('uktr', 'ukrtr'),  # Alternative naming (this worked!)
        ('zk', 'rk'),       # German naming convention
        ('usc', 'rsc'),     # IEC naming
    ]
    
    for uk_alt, ukr_alt in alt_params:
        uk_val = get(typ, uk_alt)
        ukr_val = get(typ, ukr_alt)
        
        if not np.isnan(uk_val) and uk_val > 0:
            if not (np.isnan(S_rated) or np.isnan(V_rated)) and S_rated > 0 and V_rated > 0:
                Z_base = (V_rated ** 2) / S_rated
                z_pu = uk_val / 100.0
                r_pu = ukr_val / 100.0 if not np.isnan(ukr_val) else 0.0
                x_pu = np.sqrt(max(0, z_pu**2 - r_pu**2))
                
                impedance_data.update({
                    'R_ohm': r_pu * Z_base,
                    'X_ohm': x_pu * Z_base,
                    'Z_magnitude': z_pu * Z_base,
                    'uk_percent': uk_val,
                    'ukr_percent': ukr_val,
                    'S_rated_MVA': S_rated,
                    'V_rated_kV': V_rated,
                    'impedance_source': f'transformer_type_{uk_alt}'
                })
                return impedance_data
    
    # Method 3: Use typical values as last resort
    if not (np.isnan(S_rated) or np.isnan(V_rated)) and S_rated > 0 and V_rated > 0:
        # Use typical values based on transformer size
        if S_rated >= 1000:  # Large power transformers
            uk_typical = 12.0  # 12%
            ukr_typical = 0.5  # 0.5%
        elif S_rated >= 500:  # Medium transformers
            uk_typical = 10.0  # 10%
            ukr_typical = 0.8  # 0.8%
        else:  # Smaller transformers
            uk_typical = 8.0   # 8%
            ukr_typical = 1.0  # 1.0%
        
        Z_base = (V_rated ** 2) / S_rated
        z_pu = uk_typical / 100.0
        r_pu = ukr_typical / 100.0
        x_pu = np.sqrt(z_pu**2 - r_pu**2)
        
        impedance_data.update({
            'R_ohm': r_pu * Z_base,
            'X_ohm': x_pu * Z_base,
            'Z_magnitude': z_pu * Z_base,
            'uk_percent': uk_typical,
            'ukr_percent': ukr_typical,
            'S_rated_MVA': S_rated,
            'V_rated_kV': V_rated,
            'impedance_source': 'transformer_typical_values'
        })
    
    return impedance_data

def get_generator_impedance(gen_obj):
    """Extract generator impedance from type object - CORRECT PARAMETER NAMES"""
    
    impedance_data = {
        'xd_pu': np.nan,
        'xq_pu': np.nan,
        'xd_prime_pu': np.nan,
        'xd_double_prime_pu': np.nan,
        'xl_pu': np.nan,
        'ra_pu': np.nan,
        'H_s': get(gen_obj, "H"),
        'D_pu': get(gen_obj, "Damp"),
        'S_rated_MVA': np.nan,
        'V_rated_kV': np.nan,
        'impedance_source': 'generator_direct'
    }
    
    # Get impedance values from type object first
    if has(gen_obj, 'typ_id') and gen_obj.typ_id:
        typ = gen_obj.typ_id
        
        # Get rated values from TYPE object using CORRECT parameter names
        S_rated_MVA = get(typ, "sgn")      # Rated Apparent Power (MVA)
        V_rated_kV = get(typ, "ugn")       # Rated Voltage (kV)
        cosn = get(typ, "cosn")            # Rated Power Factor
        nphase = get(typ, "nphase")        # Number of Phases
        nslty = get(typ, "nslty")          # Connection
        
        print(f"            🔋 {safe_get_name(gen_obj)}:")
        print(f"               📋 Type: S_rated={S_rated_MVA:.1f}MVA, V_rated={V_rated_kV:.1f}kV" if not (np.isnan(S_rated_MVA) or np.isnan(V_rated_kV)) else f"               📋 Type: S_rated=nan, V_rated=nan")
        print(f"               📋 cosn={cosn:.3f}, nphase={nphase}, nslty={nslty}" if not (np.isnan(cosn) or np.isnan(nphase) or np.isnan(nslty)) else f"               📋 cosn/nphase/nslty have NaN values")
        
        # Update impedance data with type rated values
        impedance_data['S_rated_MVA'] = S_rated_MVA
        impedance_data['V_rated_kV'] = V_rated_kV
        
        # Get impedance values from type object
        impedance_data.update({
            'xd_pu': get(typ, "xd"),           # d-axis synchronous reactance
            'xq_pu': get(typ, "xq"),           # q-axis synchronous reactance  
            'xd_prime_pu': get(typ, "xds"),    # d-axis transient reactance
            'xd_double_prime_pu': get(typ, "xdss"), # d-axis subtransient reactance
            'xl_pu': get(typ, "xl"),           # Leakage reactance
            'ra_pu': get(typ, "ra", 0.01),     # Armature resistance (default if NaN)
            'cosn': cosn,                      # Rated power factor
            'nphase': nphase,                  # Number of phases
            'nslty': nslty,                    # Connection type
            'impedance_source': 'generator_type_correct_params'
        })
        
        # Calculate equivalent impedance for positive sequence
        xd = impedance_data['xd_pu']
        if not np.isnan(xd) and not (np.isnan(S_rated_MVA) or np.isnan(V_rated_kV)) and S_rated_MVA > 0 and V_rated_kV > 0:
            # Convert to Ohms using generator base
            Z_base = (V_rated_kV ** 2) / S_rated_MVA
            impedance_data['Xd_ohm'] = xd * Z_base
            impedance_data['Ra_ohm'] = impedance_data['ra_pu'] * Z_base
            
            print(f"               ✅ SUCCESS: Xd = {impedance_data['Xd_ohm']:.3f} Ω")
            print(f"                  (xd={xd:.3f} p.u., S_base={S_rated_MVA:.0f}MVA, V_base={V_rated_kV:.1f}kV)")
        else:
            print(f"               ❌ FAILED: Cannot calculate ohm values")
            print(f"                  xd={xd}, S_rated={S_rated_MVA}, V_rated={V_rated_kV}")
    else:
        print(f"            ❌ {safe_get_name(gen_obj)}: No type object found")
        impedance_data['impedance_source'] = 'generator_no_type'
    
    return impedance_data

def calculate_load_impedance(load_obj, bus_voltage_pu):
    """Calculate equivalent load impedance from P, Q and bus voltage"""
    
    P_MW = get(load_obj, "plini")
    Q_MVAR = get(load_obj, "qlini") 
    
    impedance_data = {
        'P_MW': P_MW,
        'Q_MVAR': Q_MVAR,
        'bus_voltage_pu': bus_voltage_pu,
        'Z_equivalent_ohm': np.nan,
        'R_equivalent_ohm': np.nan,
        'X_equivalent_ohm': np.nan,
        'impedance_source': 'load_calculated'
    }
    
    if not (np.isnan(P_MW) or np.isnan(Q_MVAR) or np.isnan(bus_voltage_pu)) and bus_voltage_pu > 0.1:
        # Get bus base voltage
        load_bus = term(load_obj)
        if load_bus:
            V_base_kV = get(load_bus, "uknom", 1.0)
            if not np.isnan(V_base_kV) and V_base_kV > 0:
                # Calculate actual voltage
                V_actual_kV = bus_voltage_pu * V_base_kV
                
                # Calculate apparent power
                S_MVA = np.sqrt(P_MW**2 + Q_MVAR**2)
                
                if S_MVA > 0:
                    # Z = V²/S (in Ohms)
                    Z_equivalent = (V_actual_kV ** 2) / S_MVA
                    
                    # Calculate R and X components
                    power_factor = P_MW / S_MVA
                    theta = np.arccos(power_factor)
                    
                    R_equivalent = Z_equivalent * np.cos(theta)
                    X_equivalent = Z_equivalent * np.sin(theta)
                    
                    impedance_data.update({
                        'Z_equivalent_ohm': Z_equivalent,
                        'R_equivalent_ohm': R_equivalent,
                        'X_equivalent_ohm': X_equivalent,
                        'S_MVA': S_MVA,
                        'power_factor': power_factor,
                        'V_base_kV': V_base_kV,
                        'V_actual_kV': V_actual_kV,
                        'impedance_source': 'load_calculated_complete'
                    })
    
    return impedance_data

def calculate_shunt_impedance(shunt_obj, bus_voltage_pu):
    """Calculate shunt impedance from reactive power and voltage - ENHANCED VERSION"""
    
    # Try multiple parameter names for reactive power
    Q_MVAR = get(shunt_obj, "qnom")
    if np.isnan(Q_MVAR):
        Q_MVAR = get(shunt_obj, "q")
        if np.isnan(Q_MVAR):
            Q_MVAR = get(shunt_obj, "Qnom")
            if np.isnan(Q_MVAR):
                Q_MVAR = get(shunt_obj, "qcap")  # For capacitors
                if np.isnan(Q_MVAR):
                    # Try actual reactive power from load flow results
                    Q_MVAR = get(shunt_obj, "m:Qsum:bus1")
    
    # Try to get susceptance and reactance directly
    G_uS = get(shunt_obj, "g") * µS_to_S * 1e6  # Convert to µS
    B_uS = get(shunt_obj, "b") * µS_to_S * 1e6  # Convert to µS
    
    # Try direct impedance parameters if available
    R_direct = get(shunt_obj, "R")  # Direct resistance
    X_direct = get(shunt_obj, "X")  # Direct reactance
    
    impedance_data = {
        'Q_MVAR': Q_MVAR,
        'G_uS': G_uS,
        'B_uS': B_uS,
        'bus_voltage_pu': bus_voltage_pu,
        'impedance_source': 'shunt_calculated'
    }
    
    print(f"            🔧 {safe_get_name(shunt_obj)}: Q={Q_MVAR:.3f}MVAR, B={B_uS:.1f}µS")
    
    # Method 1: Use direct reactance if available
    if not np.isnan(X_direct) and abs(X_direct) > SMALL_IMPEDANCE_THRESHOLD:
        impedance_data.update({
            'X_shunt_ohm': X_direct,
            'R_shunt_ohm': R_direct if not np.isnan(R_direct) else 0.0,
            'capacitive': X_direct < 0,  # Negative reactance = capacitive
            'impedance_source': 'shunt_direct_impedance'
        })
        print(f"               ✅ Direct X = {X_direct:.1f} Ω")
        return impedance_data
    
    # Method 2: Calculate from susceptance B
    if not np.isnan(B_uS) and abs(B_uS) > 1e-3:  # Minimum 0.001 µS
        shunt_bus = term(shunt_obj)
        if shunt_bus and not np.isnan(bus_voltage_pu) and bus_voltage_pu > 0.1:
            V_base_kV = get(shunt_bus, "uknom")
            if not np.isnan(V_base_kV) and V_base_kV > 0:
                V_actual_kV = bus_voltage_pu * V_base_kV
                
                # X = 1/B (susceptance to reactance)
                B_S = B_uS * 1e-6  # Convert µS to S
                X_shunt = 1.0 / abs(B_S) if abs(B_S) > 1e-12 else np.nan
                
                # Calculate Q from B and voltage for verification
                Q_calculated = B_S * (V_actual_kV ** 2)
                
                impedance_data.update({
                    'X_shunt_ohm': X_shunt,
                    'R_shunt_ohm': 0.0,  # Pure reactive
                    'capacitive': B_uS > 0,  # Positive B = capacitive
                    'Q_calculated_MVAR': Q_calculated,
                    'V_base_kV': V_base_kV,
                    'V_actual_kV': V_actual_kV,
                    'impedance_source': 'shunt_calculated_from_B'
                })
                print(f"               ✅ From B: X = {X_shunt:.1f} Ω, Q_calc = {Q_calculated:.3f} MVAR")
                return impedance_data
    
    # Method 3: Calculate from Q = V²/X (like loads but reactive only)
    if not (np.isnan(Q_MVAR) or np.isnan(bus_voltage_pu)) and abs(Q_MVAR) > 0.001 and bus_voltage_pu > 0.1:
        shunt_bus = term(shunt_obj)
        if shunt_bus:
            V_base_kV = get(shunt_bus, "uknom", 1.0)
            if not np.isnan(V_base_kV) and V_base_kV > 0:
                V_actual_kV = bus_voltage_pu * V_base_kV
                
                # X = V²/Q (reactive impedance) - similar to load calculation
                X_shunt = (V_actual_kV ** 2) / abs(Q_MVAR)
                
                impedance_data.update({
                    'X_shunt_ohm': X_shunt,
                    'R_shunt_ohm': 0.0,  # Pure reactive element
                    'capacitive': Q_MVAR < 0,  # Negative Q = capacitive (supplies reactive power)
                    'V_base_kV': V_base_kV,
                    'V_actual_kV': V_actual_kV,
                    'impedance_source': 'shunt_calculated_from_Q'
                })
                print(f"               ✅ From Q: X = {X_shunt:.1f} Ω ({'capacitive' if Q_MVAR < 0 else 'inductive'})")
                return impedance_data
    
    # Method 4: No valid data found
    print(f"               ⚠️ No valid shunt data found")
    impedance_data['impedance_source'] = 'shunt_no_data'
    return impedance_data

# ── Load flow data extraction ──────────────────────────────────────────────
def collect_bus_data():
    """Collect comprehensive bus data from load flow results"""
    
    buses = [as_bus(b) for b in app.GetCalcRelevantObjects("*.ElmTerm")]
    
    bus_data = {
        'names': [],
        'voltages_pu': [],
        'voltage_angles_deg': [],
        'active_injection_MW': [],
        'reactive_injection_MVAR': [],
        'base_voltages_kV': [],
        'bus_types': [],
        'in_service': []
    }
    
    for bus in buses:
        bus_data['names'].append(safe_get_name(bus))
        bus_data['voltages_pu'].append(get(bus, "m:u"))  # Voltage magnitude p.u.
        bus_data['voltage_angles_deg'].append(get(bus, "m:phiu"))  # Voltage angle deg
        bus_data['active_injection_MW'].append(get(bus, "m:Psum:bus1"))  # Net P injection
        bus_data['reactive_injection_MVAR'].append(get(bus, "m:Qsum:bus1"))  # Net Q injection
        bus_data['base_voltages_kV'].append(get(bus, "uknom"))  # Base voltage kV
        bus_data['bus_types'].append(bus.GetBusType() if hasattr(bus, "GetBusType") else -1)
        bus_data['in_service'].append(get(bus, "outserv", 0) == 0)  # 0 = in service
    
    # Convert to numpy arrays
    for key in bus_data:
        if key != 'names':
            bus_data[key] = np.array(bus_data[key])
    
    return bus_data

def collect_line_data():
    """Collect line loading and impedance data"""
    
    lines = [line for line in app.GetCalcRelevantObjects("*.ElmLne") if get(line, "outserv", 0) == 0]
    
    line_data = {
        'names': [],
        'from_buses': [],
        'to_buses': [],
        'active_power_from_MW': [],
        'reactive_power_from_MVAR': [],
        'active_power_to_MW': [],
        'reactive_power_to_MVAR': [],
        'current_from_A': [],
        'current_to_A': [],
        'loading_percent': [],
        'losses_MW': [],
        'losses_MVAR': [],
        'impedances': []
    }
    
    for line in lines:
        line_data['names'].append(safe_get_name(line))
        
        # Bus connections
        from_bus = term(line.bus1) if has(line, "bus1") else None
        to_bus = term(line.bus2) if has(line, "bus2") else None
        line_data['from_buses'].append(safe_get_name(from_bus) if from_bus else "Unknown")
        line_data['to_buses'].append(safe_get_name(to_bus) if to_bus else "Unknown")
        
        # Power flows
        line_data['active_power_from_MW'].append(get(line, "m:Psum:bus1"))
        line_data['reactive_power_from_MVAR'].append(get(line, "m:Qsum:bus1"))
        line_data['active_power_to_MW'].append(get(line, "m:Psum:bus2"))
        line_data['reactive_power_to_MVAR'].append(get(line, "m:Qsum:bus2"))
        
        # Currents
        line_data['current_from_A'].append(get(line, "m:I:bus1"))
        line_data['current_to_A'].append(get(line, "m:I:bus2"))
        
        # Loading
        line_data['loading_percent'].append(get(line, "c:loading"))
        
        # Losses
        line_data['losses_MW'].append(get(line, "c:Ploss"))
        line_data['losses_MVAR'].append(get(line, "c:Qloss"))
        
        # Impedance data
        line_data['impedances'].append(get_line_impedance(line))
    
    return line_data

def collect_transformer_data():
    """Collect transformer loading and impedance data"""
    
    transformers = [trafo for trafo in app.GetCalcRelevantObjects("*.ElmTr2,*.ElmXfr,*.ElmXfr3") 
                   if get(trafo, "outserv", 0) == 0]
    
    transformer_data = {
        'names': [],
        'classes': [],
        'from_buses': [],
        'to_buses': [],
        'active_power_from_MW': [],
        'reactive_power_from_MVAR': [],
        'active_power_to_MW': [],
        'reactive_power_to_MVAR': [],
        'current_from_A': [],
        'current_to_A': [],
        'loading_percent': [],
        'losses_MW': [],
        'losses_MVAR': [],
        'tap_positions': [],
        'impedances': []
    }
    
    for trafo in transformers:
        transformer_data['names'].append(safe_get_name(trafo))
        transformer_data['classes'].append(safe_get_class(trafo))
        
        # Bus connections (handle different transformer types)
        if has(trafo, "bushv"):
            from_bus = term(trafo.bushv)
            to_bus = term(trafo.buslv)
        elif has(trafo, "bus1"):
            from_bus = term(trafo.bus1)
            to_bus = term(trafo.bus2)
        else:
            from_bus = to_bus = None
            
        transformer_data['from_buses'].append(safe_get_name(from_bus) if from_bus else "Unknown")
        transformer_data['to_buses'].append(safe_get_name(to_bus) if to_bus else "Unknown")
        
        # Power flows
        transformer_data['active_power_from_MW'].append(get(trafo, "m:Psum:bushv"))
        transformer_data['reactive_power_from_MVAR'].append(get(trafo, "m:Qsum:bushv"))
        transformer_data['active_power_to_MW'].append(get(trafo, "m:Psum:buslv"))
        transformer_data['reactive_power_to_MVAR'].append(get(trafo, "m:Qsum:buslv"))
        
        # Currents
        transformer_data['current_from_A'].append(get(trafo, "m:I:bushv"))
        transformer_data['current_to_A'].append(get(trafo, "m:I:buslv"))
        
        # Loading and losses
        transformer_data['loading_percent'].append(get(trafo, "c:loading"))
        transformer_data['losses_MW'].append(get(trafo, "c:Ploss"))
        transformer_data['losses_MVAR'].append(get(trafo, "c:Qloss"))
        
        # Tap position
        transformer_data['tap_positions'].append(get(trafo, "nntap"))
        
        # Impedance data
        transformer_data['impedances'].append(get_transformer_impedance(trafo))
    
    return transformer_data

def collect_generator_data():
    """Collect generator output and impedance data - ENHANCED VERSION"""
    
    generators = [gen for gen in app.GetCalcRelevantObjects("*.ElmSym") if get(gen, "outserv", 0) == 0]
    
    generator_data = {
        'names': [],
        'buses': [],
        'active_power_MW': [],
        'reactive_power_MVAR': [],
        'voltage_setpoint_pu': [],
        'terminal_voltage_pu': [],
        'reactive_limits_min_MVAR': [],
        'reactive_limits_max_MVAR': [],
        'active_power_limits_MW': [],
        'power_factor': [],
        'impedances': []
    }
    
    print(f"         🔋 Processing {len(generators)} generators...")
    
    for gen in generators:
        generator_data['names'].append(safe_get_name(gen))
        
        gen_bus = term(gen)
        generator_data['buses'].append(safe_get_name(gen_bus) if gen_bus else "Unknown")
        
        # Actual outputs
        P_gen = get(gen, "m:Psum:bus1")
        Q_gen = get(gen, "m:Qsum:bus1")
        generator_data['active_power_MW'].append(P_gen)
        generator_data['reactive_power_MVAR'].append(Q_gen)
        
        # Voltage control
        generator_data['voltage_setpoint_pu'].append(get(gen, "uset"))
        generator_data['terminal_voltage_pu'].append(get(gen, "m:u1"))
        
        # Limits
        generator_data['reactive_limits_min_MVAR'].append(get(gen, "Qmin"))
        generator_data['reactive_limits_max_MVAR'].append(get(gen, "Qmax"))
        generator_data['active_power_limits_MW'].append(get(gen, "Pmax"))
        
        # Power factor
        if not (np.isnan(P_gen) or np.isnan(Q_gen)):
            S_gen = np.sqrt(P_gen**2 + Q_gen**2)
            pf = P_gen / S_gen if S_gen > 0 else np.nan
        else:
            pf = np.nan
        generator_data['power_factor'].append(pf)
        
        # FIXED: Impedance data with proper rated values
        generator_data['impedances'].append(get_generator_impedance(gen))
    
    print(f"         ✅ Generator data collection complete")
    return generator_data

def collect_load_data(bus_data):
    """Collect load data with calculated impedances"""
    
    loads = [load for load in app.GetCalcRelevantObjects("*.ElmLod") if get(load, "outserv", 0) == 0]
    
    # Create bus voltage lookup
    bus_voltage_dict = {}
    for i, bus_name in enumerate(bus_data['names']):
        bus_voltage_dict[bus_name] = bus_data['voltages_pu'][i]
    
    load_data = {
        'names': [],
        'buses': [],
        'active_power_MW': [],
        'reactive_power_MVAR': [],
        'impedances': []
    }
    
    for load in loads:
        load_data['names'].append(safe_get_name(load))
        
        load_bus = term(load)
        bus_name = safe_get_name(load_bus) if load_bus else "Unknown"
        load_data['buses'].append(bus_name)
        
        # Power consumption
        load_data['active_power_MW'].append(get(load, "plini"))
        load_data['reactive_power_MVAR'].append(get(load, "qlini"))
        
        # Calculate impedance
        bus_voltage = bus_voltage_dict.get(bus_name, np.nan)
        load_data['impedances'].append(calculate_load_impedance(load, bus_voltage))
    
    return load_data

def collect_shunt_data(bus_data):
    """Collect shunt data with calculated impedances - ENHANCED VERSION"""
    
    shunts = [shunt for shunt in app.GetCalcRelevantObjects("*.ElmShnt,*.ElmReac,*.ElmCap,*.ElmScal") 
             if get(shunt, "outserv", 0) == 0]
    
    # Create bus voltage lookup
    bus_voltage_dict = {}
    for i, bus_name in enumerate(bus_data['names']):
        bus_voltage_dict[bus_name] = bus_data['voltages_pu'][i]
    
    shunt_data = {
        'names': [],
        'buses': [],
        'reactive_power_MVAR': [],
        'impedances': []
    }
    
    print(f"         🔧 Processing {len(shunts)} shunts...")
    
    for shunt in shunts:
        shunt_name = safe_get_name(shunt)
        shunt_data['names'].append(shunt_name)
        
        shunt_bus = term(shunt)
        bus_name = safe_get_name(shunt_bus) if shunt_bus else "Unknown"
        shunt_data['buses'].append(bus_name)
        
        # Get reactive power (try multiple sources)
        Q_MVAR = get(shunt, "qnom")
        if np.isnan(Q_MVAR):
            Q_MVAR = get(shunt, "m:Qsum:bus1")  # Actual from load flow
        shunt_data['reactive_power_MVAR'].append(Q_MVAR)
        
        # Calculate impedance with enhanced method
        bus_voltage = bus_voltage_dict.get(bus_name, np.nan)
        impedance_result = calculate_shunt_impedance(shunt, bus_voltage)
        shunt_data['impedances'].append(impedance_result)
    
    print(f"         ✅ Shunt data collection complete")
    return shunt_data

def collect_system_summary():
    """Collect overall system summary data"""
    
    summary = {
        'total_generation_MW': 0,
        'total_load_MW': 0,
        'total_losses_MW': 0,
        'total_reactive_generation_MVAR': 0,
        'total_reactive_load_MVAR': 0,
        'total_reactive_losses_MVAR': 0,
        'system_frequency_Hz': get(app.GetActiveStudyCase(), "SetFrq", 50.0),
        'convergence_iterations': 0,
        'min_voltage_pu': np.nan,
        'max_voltage_pu': np.nan,
        'voltage_violations_count': 0,
        'overloaded_elements_count': 0
    }
    
    # Get system totals from PowerFactory
    summary['total_generation_MW'] = get(app.GetActiveStudyCase(), "c:Pgen", 0.0)
    summary['total_load_MW'] = get(app.GetActiveStudyCase(), "c:Pload", 0.0)
    summary['total_losses_MW'] = get(app.GetActiveStudyCase(), "c:Ploss", 0.0)
    
    # Try to get load flow convergence info
    comLdf = app.GetFromStudyCase("ComLdf")
    if comLdf:
        summary['convergence_iterations'] = get(comLdf, "iiter", 0)
    
    return summary

# ── Main data collection function ──────────────────────────────────────────
def collect_load_flow_data():
    """
    Main function to collect all post-contingency load flow data
    
    Returns:
        dict: Comprehensive data dictionary with all collected information
    """
    
    print(f"      📊 Collecting post-contingency load flow data...")
    start_time = time.time()
    
    # Initialize result dictionary
    load_flow_data = {
        'collection_timestamp': datetime.now().isoformat(),
        'collection_time_seconds': 0,
        'data_quality': 'complete'
    }
    
    try:
        # Collect bus data first (needed for impedance calculations)
        print(f"         🔌 Collecting bus data...")
        load_flow_data['buses'] = collect_bus_data()
        
        # Collect line data with impedances
        print(f"         ⚡ Collecting line data...")
        load_flow_data['lines'] = collect_line_data()
        
        # Collect transformer data with impedances
        print(f"         🔄 Collecting transformer data...")
        load_flow_data['transformers'] = collect_transformer_data()
        
        # Collect generator data with impedances
        print(f"         🔋 Collecting generator data...")
        load_flow_data['generators'] = collect_generator_data()
        
        # Collect load data with calculated impedances
        print(f"         🏠 Collecting load data...")
        load_flow_data['loads'] = collect_load_data(load_flow_data['buses'])
        
        # Collect shunt data with calculated impedances
        print(f"         🔧 Collecting shunt data...")
        load_flow_data['shunts'] = collect_shunt_data(load_flow_data['buses'])
        
        # Collect system summary
        print(f"         📈 Collecting system summary...")
        load_flow_data['system_summary'] = collect_system_summary()
        
        # Calculate data statistics
        bus_count = len(load_flow_data['buses']['names'])
        line_count = len(load_flow_data['lines']['names'])
        trafo_count = len(load_flow_data['transformers']['names'])
        gen_count = len(load_flow_data['generators']['names'])
        load_count = len(load_flow_data['loads']['names'])
        shunt_count = len(load_flow_data['shunts']['names'])
        
        # Calculate voltage statistics
        voltages = load_flow_data['buses']['voltages_pu']
        valid_voltages = voltages[~np.isnan(voltages)]
        
        if len(valid_voltages) > 0:
            load_flow_data['system_summary']['min_voltage_pu'] = np.min(valid_voltages)
            load_flow_data['system_summary']['max_voltage_pu'] = np.max(valid_voltages)
            
            # Count voltage violations (outside 0.95 - 1.05 p.u.)
            voltage_violations = np.sum((valid_voltages < 0.95) | (valid_voltages > 1.05))
            load_flow_data['system_summary']['voltage_violations_count'] = int(voltage_violations)
        
        # Count overloaded elements
        overloaded_count = 0
        
        # Check line loadings
        line_loadings = np.array(load_flow_data['lines']['loading_percent'])
        overloaded_count += np.sum(line_loadings[~np.isnan(line_loadings)] > 100)
        
        # Check transformer loadings
        trafo_loadings = np.array(load_flow_data['transformers']['loading_percent'])
        overloaded_count += np.sum(trafo_loadings[~np.isnan(trafo_loadings)] > 100)
        
        load_flow_data['system_summary']['overloaded_elements_count'] = int(overloaded_count)
        
        # Add collection statistics
        collection_time = time.time() - start_time
        load_flow_data['collection_time_seconds'] = collection_time
        
        load_flow_data['collection_statistics'] = {
            'buses_collected': bus_count,
            'lines_collected': line_count,
            'transformers_collected': trafo_count,
            'generators_collected': gen_count,
            'loads_collected': load_count,
            'shunts_collected': shunt_count,
            'total_elements': bus_count + line_count + trafo_count + gen_count + load_count + shunt_count,
            'impedance_calculations_successful': 0,  # Will be calculated below
            'impedance_calculations_failed': 0
        }
        
        # Count successful impedance calculations
        successful_impedances = 0
        failed_impedances = 0
        
        # Count line impedances
        for line_imp in load_flow_data['lines']['impedances']:
            if not np.isnan(line_imp['R_ohm']) and not np.isnan(line_imp['X_ohm']):
                successful_impedances += 1
            else:
                failed_impedances += 1
        
        # Count transformer impedances
        for trafo_imp in load_flow_data['transformers']['impedances']:
            if not np.isnan(trafo_imp['R_ohm']) and not np.isnan(trafo_imp['X_ohm']):
                successful_impedances += 1
            else:
                failed_impedances += 1
        
        # Count generator impedances
        for gen_imp in load_flow_data['generators']['impedances']:
            if not np.isnan(gen_imp['xd_pu']):
                successful_impedances += 1
            else:
                failed_impedances += 1
        
        # Count load impedances
        for load_imp in load_flow_data['loads']['impedances']:
            if not np.isnan(load_imp['Z_equivalent_ohm']):
                successful_impedances += 1
            else:
                failed_impedances += 1
        
        # Count shunt impedances
        for shunt_imp in load_flow_data['shunts']['impedances']:
            if 'X_shunt_ohm' in shunt_imp and not np.isnan(shunt_imp['X_shunt_ohm']):
                successful_impedances += 1
            else:
                failed_impedances += 1
        
        load_flow_data['collection_statistics']['impedance_calculations_successful'] = successful_impedances
        load_flow_data['collection_statistics']['impedance_calculations_failed'] = failed_impedances
        
        # Final summary message
        impedance_success_rate = successful_impedances / (successful_impedances + failed_impedances) * 100 if (successful_impedances + failed_impedances) > 0 else 0
        
        print(f"      ✅ Data collection complete!")
        print(f"         📊 Elements: {bus_count} buses, {line_count} lines, {trafo_count} transformers")
        print(f"         📊 Equipment: {gen_count} generators, {load_count} loads, {shunt_count} shunts")
        print(f"         📊 Impedances: {successful_impedances}/{successful_impedances + failed_impedances} successful ({impedance_success_rate:.1f}%)")
        print(f"         ⏱️ Collection time: {collection_time:.2f} seconds")
        
        if voltage_violations > 0:
            print(f"         ⚠️ Voltage violations: {voltage_violations} buses outside 0.95-1.05 p.u.")
        
        if overloaded_count > 0:
            print(f"         ⚠️ Overloaded elements: {overloaded_count} elements >100% loading")
        
    except Exception as e:
        print(f"      ❌ Error during data collection: {e}")
        load_flow_data['data_quality'] = 'error'
        load_flow_data['error_message'] = str(e)
        load_flow_data['collection_time_seconds'] = time.time() - start_time
    
    return load_flow_data

# ── Utility functions for data validation ─────────────────────────────────
def validate_collected_data(load_flow_data):
    """Validate the collected data for completeness and consistency"""
    
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'data_completeness': {}
    }
    
    try:
        # Check if main data groups exist
        required_groups = ['buses', 'lines', 'transformers', 'generators', 'loads', 'shunts', 'system_summary']
        
        for group in required_groups:
            if group not in load_flow_data:
                validation_results['errors'].append(f"Missing data group: {group}")
                validation_results['is_valid'] = False
            else:
                # Check if group has data
                if isinstance(load_flow_data[group], dict) and 'names' in load_flow_data[group]:
                    count = len(load_flow_data[group]['names'])
                    validation_results['data_completeness'][group] = count
                    
                    if count == 0 and group in ['buses', 'generators']:  # Critical elements
                        validation_results['errors'].append(f"No {group} found - this indicates a serious system issue")
                        validation_results['is_valid'] = False
                elif group == 'system_summary':
                    validation_results['data_completeness'][group] = 'summary_data'
        
        # Check voltage data consistency
        if 'buses' in load_flow_data:
            voltages = np.array(load_flow_data['buses']['voltages_pu'])
            zero_voltage_count = np.sum(voltages < 0.01)
            
            if zero_voltage_count > 0:
                validation_results['warnings'].append(f"{zero_voltage_count} buses have very low voltage (<0.01 p.u.) - possible islanding")
            
            # Check for NaN voltages
            nan_voltage_count = np.sum(np.isnan(voltages))
            if nan_voltage_count > 0:
                validation_results['warnings'].append(f"{nan_voltage_count} buses have NaN voltages")
        
        # Check impedance data quality
        if 'collection_statistics' in load_flow_data:
            stats = load_flow_data['collection_statistics']
            success_rate = stats.get('impedance_calculations_successful', 0)
            total_impedances = success_rate + stats.get('impedance_calculations_failed', 0)
            
            if total_impedances > 0:
                success_percentage = (success_rate / total_impedances) * 100
                if success_percentage < 50:
                    validation_results['warnings'].append(f"Low impedance extraction success rate: {success_percentage:.1f}%")
                elif success_percentage < 80:
                    validation_results['warnings'].append(f"Moderate impedance extraction success rate: {success_percentage:.1f}%")
    
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {e}")
        validation_results['is_valid'] = False
    
    return validation_results

def get_data_summary(load_flow_data):
    """Get a concise summary of collected data for logging"""
    
    if 'collection_statistics' not in load_flow_data:
        return "Data collection incomplete"
    
    stats = load_flow_data['collection_statistics']
    
    summary = (
        f"Collected: {stats['buses_collected']} buses, "
        f"{stats['lines_collected']} lines, "
        f"{stats['transformers_collected']} transformers, "
        f"{stats['generators_collected']} generators, "
        f"{stats['loads_collected']} loads, "
        f"{stats['shunts_collected']} shunts. "
        f"Impedances: {stats['impedance_calculations_successful']}/{stats['impedance_calculations_successful'] + stats['impedance_calculations_failed']} successful"
    )
    
    return summary

# ── Test function for standalone testing ──────────────────────────────────
def test_data_collection():
    """Test function for standalone module testing"""
    
    print(f"🧪 TESTING LOAD FLOW DATA COLLECTION MODULE")
    print("="*60)
    
    # Connect to PowerFactory (same as other modules)
    global app
    app = pf.GetApplication() or sys.exit("PF not running")
    
    PROJECT = "39 Bus New England System"
    STUDY = "RMS_Simulation"
    
    if hasattr(app, "ResetCalculation"):
        app.ResetCalculation()
    assert app.ActivateProject(PROJECT) == 0, "Project not found"
    study = next(c for c in app.GetProjectFolder("study").GetContents("*.IntCase")
                 if c.loc_name == STUDY)
    study.Activate()
    
    # Solve load flow first
    print(f"🔄 Solving load flow...")
    comLdf = app.GetFromStudyCase("ComLdf")
    if comLdf:
        comLdf.Execute()
        print(f"✅ Load flow solved")
    
    # Collect data
    data = collect_load_flow_data()
    
    # Validate data
    validation = validate_collected_data(data)
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Valid: {validation['is_valid']}")
    if validation['warnings']:
        print(f"   Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"     ⚠️ {warning}")
    if validation['errors']:
        print(f"   Errors: {len(validation['errors'])}")
        for error in validation['errors']:
            print(f"     ❌ {error}")
    
    print(f"\n📋 DATA SUMMARY:")
    print(f"   {get_data_summary(data)}")
    
    return data, validation

if __name__ == "__main__":
    # Run test if module is executed directly
    test_data_collection()