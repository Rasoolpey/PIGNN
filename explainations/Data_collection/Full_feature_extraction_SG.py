# Fixed_Complete_Feature_extraction_SG.py - 2025-07-21
"""
Fixed version that combines working impedance extraction from First_model.py
with comprehensive feature extraction from Full_feature_extraction_SG.py

Key fix: Properly extract R and X values before creating edge_tab dictionary
"""

import sys, os, h5py, numpy as np, yaml
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf
from datetime import datetime
from scipy.sparse import csr_matrix

# ── Project settings ────────────────────────────────────────────────────────
PROJECT = "39 Bus New England System"  # Change this to your project name
STUDY = "RMS_Simulation"
SBASE_MVA = 100.0
OUT_DIR = os.path.join(os.getcwd(), "enhanced_out")
µS_to_S = 1e-6

# ── Connect to PowerFactory ─────────────────────────────────────────────────
app = pf.GetApplication() or sys.exit("PF not running")
if hasattr(app, "ResetCalculation"):
    app.ResetCalculation()
assert app.ActivateProject(PROJECT) == 0, "Project not found"
study = next(c for c in app.GetProjectFolder("study").GetContents("*.IntCase")
             if c.loc_name == STUDY)
study.Activate()

print(f"🔧 FIXED COMPLETE POWER SYSTEM EXTRACTION")
print("="*60)
print(f"✅ {PROJECT} | {STUDY}")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Creating enhanced H5 file with ALL data (topology + control systems)")
print()

# ── Helper functions (from working code) ──────────────────────────────────
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

def term(obj):
    """Get the terminal/bus from various PowerFactory objects - ENHANCED VERSION"""
    if not obj:
        return None
        
    try:
        # For ElmTerm objects, return directly
        if obj.GetClassName() == "ElmTerm":
            return obj
            
        # For transformers (ElmTr2, ElmTr3) with different bus attributes
        if obj.GetClassName().startswith("ElmTr"):
            for f in ("bushv", "buslv", "bus1", "bus2"):
                if has(obj, f):
                    ptr = obj.GetAttribute(f)
                    if ptr:
                        # Handle nested cterm references
                        if has(ptr, "cterm"):
                            return ptr.cterm
                        elif ptr.GetClassName() == "ElmTerm":
                            return ptr
                        else:
                            return as_bus(ptr)
            return None
        
        # For other objects (lines, etc.)
        for f in ("bus1", "bus2", "cterm", "sbus", "bus", "bushv", "buslv"):
            if has(obj, f):
                ptr = obj.GetAttribute(f) if f != "cterm" else obj
                if ptr:
                    # Handle nested cterm references
                    if has(ptr, "cterm"):
                        return ptr.cterm
                    elif ptr.GetClassName() == "ElmTerm":
                        return ptr
                    else:
                        return as_bus(ptr)
        
        return None
        
    except Exception as e:
        print(f"❌ Error in term() for {safe_get_name(obj)}: {e}")
        return None

def safe_bus_idx(obj, idx_dict):
    """Safely get bus index for an object"""
    try:
        bus = term(obj)
        if bus and bus in idx_dict:
            return idx_dict[bus]
        else:
            return -1
    except Exception as e:
        return -1

# ── Impedance calculation functions (from working Feature_Extraction.py) ────
def tr2_imp(x):
    """Return (R,X) in Ohm for ElmTr2 transformers"""
    if has(x, 'typ_id') and x.typ_id:
        typ = x.typ_id
        r1pu = get(typ, 'r1pu')
        x1pu = get(typ, 'x1pu')
        
        if not (np.isnan(r1pu) or np.isnan(x1pu)):
            V_base = get(typ, 'utrn_h')
            S_base = get(typ, 'strn')
            
            if not (np.isnan(V_base) or np.isnan(S_base)) and V_base > 0 and S_base > 0:
                Z_base = (V_base ** 2) / S_base
                R_ohm = r1pu * Z_base
                X_ohm = x1pu * Z_base
                return R_ohm, X_ohm
    
    R = get(x, "r1")
    X = get(x, "x1")
    if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
        return R, X
    
    try:
        if has(x, 'bushv') and has(x, 'buslv') and x.bushv and x.buslv:
            bushv = x.bushv.cterm if hasattr(x.bushv, 'cterm') else x.bushv
            V_hv = get(bushv, 'uknom')
            
            if not np.isnan(V_hv) and V_hv > 0:
                S_base = SBASE_MVA
                Z_base = (V_hv ** 2) / S_base
                
                if has(x, 'typ_id') and x.typ_id:
                    typ = x.typ_id
                    r1pu = get(typ, 'r1pu')
                    x1pu = get(typ, 'x1pu')
                    
                    if not (np.isnan(r1pu) or np.isnan(x1pu)):
                        return r1pu * Z_base, x1pu * Z_base
    except:
        pass
    
    return np.nan, np.nan

def xfr_imp(x):
    """Return (R,X) in Ohm for any ElmXfr/ElmXfr3"""
    def to_ohm(rpu, xpu):
        Vbase = get(x, "un1") or get(x, "unom")
        Sbase = get(x, "snom") or SBASE_MVA
        if Vbase and Sbase:
            Zb = (Vbase * 1e3) ** 2 / (Sbase * 1e6)
            return rpu * Zb, xpu * Zb
        return np.nan, np.nan

    # Try explicit R/X
    R, X = get(x, "r1"), get(x, "x1")
    if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
        return R, X

    # Try explicit PU R/X
    Rpu, Xpu = get(x, "r1pu"), get(x, "x1pu")
    if not (np.isnan(Rpu) or np.isnan(Xpu) or (Rpu == 0 and Xpu == 0)):
        return to_ohm(Rpu, Xpu)

    # Copy missing values from type
    if has(x, "typ_id"):
        for a in ("r1", "x1", "r1pu", "x1pu", "uk", "ukr"):
            if has(x, a) and np.isnan(get(x, a)) and has(x.typ_id, a):
                val = get(x.typ_id, a)
                if not np.isnan(val):
                    x.SetAttribute(a, val)
        
        # Retry after type copy
        R, X = get(x, "r1"), get(x, "x1")
        if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
            return R, X
        Rpu, Xpu = get(x, "r1pu"), get(x, "x1pu")
        if not (np.isnan(Rpu) or np.isnan(Xpu) or (Rpu == 0 and Xpu == 0)):
            return to_ohm(Rpu, Xpu)

    # Final fallback – uk/ukr [%]
    uk, ukr = get(x, "uk"), get(x, "ukr")
    if not (np.isnan(uk) or uk == 0):
        Rpu = ukr / 100.0
        Xpu = np.sqrt(max((uk / 100.0) ** 2 - Rpu ** 2, 0))
        return to_ohm(Rpu, Xpu)

    return np.nan, np.nan

def branch_imp(e):
    """Get branch impedance for lines and transformers - WORKING VERSION"""
    cls = e.GetClassName()
    if cls.startswith("ElmLne"):
        R, X = get(e, "r1"), get(e, "x1")
        # Fallback to per-km
        if np.isnan(R) or R == 0:
            Rkm = get(e, "r_km")
            if np.isnan(Rkm):
                Rkm = get(e.typ_id, "rline")
            if not np.isnan(Rkm):
                R = Rkm * get(e, "dline")
        if np.isnan(X) or X == 0:
            Xkm = get(e, "x_km")
            if np.isnan(Xkm):
                Xkm = get(e.typ_id, "xline")
            if not np.isnan(Xkm):
                X = Xkm * get(e, "dline")
    elif cls.startswith("ElmXfr"):
        R, X = xfr_imp(e)
    elif cls.startswith("ElmTr"):
        R, X = tr2_imp(e)
    else:
        R, X = np.nan, np.nan
    return R, X

def tap_ratio(e):
    return get(e, "tratio", 1.0) if e.GetClassName().startswith("ElmXfr") else 1.0

def phase_shift(e):
    return get(e, "phitr", 0.0)

# ── Network structure detection (from Full_feature_extraction_SG.py) ────────
project = app.GetActiveProject()
network_model = project.GetContents("Network Model.IntPrjfolder")[0]
network_data = network_model.GetContents("Network Data.IntPrjfolder")[0]

# Robust grid detection
grid = None
grid_candidates = [
    "Grid.ElmNet",
    "*.ElmNet", 
    "Microgrid.ElmNet",
    "MV Grid.ElmNet",
    "Grid_1.ElmNet"
]

print(f"🔍 SEARCHING FOR GRID OBJECT...")
for candidate in grid_candidates:
    try:
        grids = network_data.GetContents(candidate)
        if grids:
            grid = grids[0]
            print(f"✅ Found grid: {grid.loc_name} ({candidate})")
            break
    except:
        continue

if not grid:
    try:
        all_nets = network_data.GetContents("*.ElmNet")
        if all_nets:
            grid = all_nets[0]
            print(f"✅ Using first available grid: {grid.loc_name}")
        else:
            print(f"❌ No ElmNet objects found!")
            all_elements = network_data.GetContents()
    except Exception as e:
        print(f"❌ Error accessing grids: {e}")
        all_elements = []

if grid:
    all_elements = grid.GetContents()
    print(f"📊 Found {len(all_elements)} elements in {grid.loc_name}")
else:
    print(f"📊 Found {len(all_elements)} elements in network data")

# ── Element classification ──────────────────────────────────────────────────
plants = [elem for elem in all_elements if elem.GetClassName() == 'ElmComp']
gens = [elem for elem in all_elements if elem.GetClassName() == 'ElmSym']
buses = [elem for elem in all_elements if elem.GetClassName() == 'ElmTerm']
lines = [elem for elem in all_elements if elem.GetClassName() == 'ElmLne']
transformers = [elem for elem in all_elements if elem.GetClassName().startswith('ElmTr') or elem.GetClassName().startswith('ElmXfr')]
loads = [elem for elem in all_elements if elem.GetClassName() == 'ElmLod']
shunts = [elem for elem in all_elements if elem.GetClassName() in ['ElmShnt', 'ElmReac', 'ElmCap', 'ElmScal']]

# Alternative PowerFactory element collection (for robustness)
if not buses:
    buses = [as_bus(b) for b in app.GetCalcRelevantObjects("*.ElmTerm")]
if not gens:
    gens = app.GetCalcRelevantObjects("*.ElmSym")
if not loads:
    loads = app.GetCalcRelevantObjects("*.ElmLod")
if not lines or not transformers:
    edges = app.GetCalcRelevantObjects("*.ElmLne,*.ElmTr2,*.ElmXfr,*.ElmXfr3")
    lines = [e for e in edges if e.GetClassName().startswith("ElmLne")]
    transformers = [e for e in edges if not e.GetClassName().startswith("ElmLne")]

# Combine lines and transformers for edge processing
edges = lines + transformers

# DSL elements for control systems
all_dsl = [elem for elem in all_elements if safe_get_class(elem) == 'ElmDsl']
avr_systems = [elem for elem in all_dsl if 'AVR' in safe_get_name(elem).upper()]
gov_systems = [elem for elem in all_dsl if 'GOV' in safe_get_name(elem).upper()]
pss_systems = [elem for elem in all_dsl if 'PSS' in safe_get_name(elem).upper()]

print(f"🏗️ COMPLETE SYSTEM INVENTORY:")
print(f"   🏭 Power Plants (ElmComp): {len(plants)}")
print(f"   🔋 Generators (ElmSym): {len(gens)}")
print(f"   🔌 Buses (ElmTerm): {len(buses)}")
print(f"   📏 Lines (ElmLne): {len(lines)}")
print(f"   🔄 Transformers: {len(transformers)}")
print(f"   ⚡ Total Edges: {len(edges)}")
print(f"   📍 Loads (ElmLod): {len(loads)}")
print(f"   🔧 Shunts: {len(shunts)}")
print(f"   🎮 DSL Models: {len(all_dsl)}")
print(f"   🎛️ AVR Systems: {len(avr_systems)}")
print(f"   ⚙️ Governor Systems: {len(gov_systems)}")
print(f"   🔧 PSS Systems: {len(pss_systems)}")

# Determine system type
system_type = "individual_generators" if len(plants) == 0 else "hierarchical_plants"
print(f"\n🔍 SYSTEM TYPE DETECTED: {system_type.replace('_', ' ').title()}")

# ── Bus mapping ────────────────────────────────────────────────────────────
nb = len(buses)
idx = {b: i for i, b in enumerate(buses)}

# ── Extract network data (FIXED VERSION) ──────────────────────────────────
print(f"\n🔍 EXTRACTING NETWORK TOPOLOGY...")

# Bus data
bus_tab = dict(
    name=np.array([b.loc_name for b in buses], dtype="S20"),
    Un_kV=np.array([get(b, "m:u0") for b in buses]),
    fn_Hz=np.array([get(b, "fbus") for b in buses]),
    area=np.array([get(b, "narea") for b in buses]),
    zone=np.array([get(b, "nzone") for b in buses]),
    bustype=np.array([b.GetBusType() if hasattr(b, "GetBusType") else -1 for b in buses]),
)

# Edge data with PROPER impedance extraction
print(f"🔧 EXTRACTING EDGE IMPEDANCES...")

# Extract impedance values for all edges FIRST
edge_impedances = []
valid_edges = []
from_indices = []
to_indices = []

for e in edges:
    # Get impedance using working function
    R, X = branch_imp(e)
    
    # Get bus connections
    bus_from = term(e.bus1 if has(e, "bus1") else (e.bushv if has(e, "bushv") else None))
    bus_to = term(e.bus2 if has(e, "bus2") else (e.buslv if has(e, "buslv") else None))
    
    if bus_from and bus_to and bus_from in idx and bus_to in idx:
        from_idx = idx[bus_from]
        to_idx = idx[bus_to]
        
        edge_impedances.append((R, X))
        valid_edges.append(e)
        from_indices.append(from_idx)
        to_indices.append(to_idx)
    else:
        print(f"⚠️ Skipping edge {safe_get_name(e)}: invalid bus connections")

print(f"✅ Valid edges: {len(valid_edges)}/{len(edges)}")

# Extract R and X arrays from impedances
R_values = [imp[0] for imp in edge_impedances]
X_values = [imp[1] for imp in edge_impedances]

# Extract B values with NaN handling
B_values = []
for e in valid_edges:
    B = get(e, "bch")
    if np.isnan(B):
        B = get(e, "b")
        if np.isnan(B):
            B = get(e, "bline")
            if np.isnan(B) and has(e, "typ_id") and e.typ_id:
                B = get(e.typ_id, "bline")
                if np.isnan(B):
                    B = get(e.typ_id, "bch")
            if np.isnan(B):
                B = 0.0
    B_values.append(B)

# Create edge table with VALID arrays
edge_cls = np.array([e.GetClassName() for e in valid_edges])
edge_type = np.where(np.char.startswith(edge_cls, "ElmLne"), 0,
                     np.where(np.char.startswith(edge_cls, "ElmXfr3"), 2, 1))

edge_tab = dict(
    from_idx=np.array(from_indices),
    to_idx=np.array(to_indices),
    name=np.array([e.loc_name for e in valid_edges], dtype="S24"),
    R_ohm=np.array(R_values),
    X_ohm=np.array(X_values),
    B_uS=np.array(B_values),
    tap_ratio=np.array([tap_ratio(e) for e in valid_edges]),
    phi_deg=np.array([phase_shift(e) for e in valid_edges]),
    rate_MVA=np.array([get(e, "snom") for e in valid_edges]),
    type_id=edge_type,
)

print(f"✅ Edge impedance extraction complete:")
print(f"   R_ohm - NaN count: {np.sum(np.isnan(edge_tab['R_ohm']))}")
print(f"   X_ohm - NaN count: {np.sum(np.isnan(edge_tab['X_ohm']))}")
print(f"   B_uS - NaN count: {np.sum(np.isnan(edge_tab['B_uS']))}")

# ── Control system parameter extraction ────────────────────────────────────
def extract_control_parameters(ctrl_obj):
    """Extract control system parameters for all types"""
    if not ctrl_obj:
        return {}
    
    ctrl_data = {
        'name': safe_get_name(ctrl_obj),
        'class': safe_get_class(ctrl_obj),
        'type': 'Unknown',
        'parameters': {}
    }
    
    name = ctrl_data['name'].upper()
    
    if 'AVR' in name:
        ctrl_data['type'] = 'AVR'
        avr_params = ['Ka', 'Ta', 'Ke', 'Te', 'Kf', 'Tf', 'Tr', 'Vrmax', 'Vrmin']
        for param in avr_params:
            val = get(ctrl_obj, param)
            if not np.isnan(val):
                ctrl_data['parameters'][param] = float(val)
                
    elif 'GOV' in name:
        ctrl_data['type'] = 'Governor'
        gov_params = ['K', 'T1', 'T2', 'T3', 'Pmax', 'Pmin', 'R', 'D']
        for param in gov_params:
            val = get(ctrl_obj, param)
            if not np.isnan(val):
                ctrl_data['parameters'][param] = float(val)
                
    elif 'PSS' in name:
        ctrl_data['type'] = 'PSS'
        pss_params = ['Kpss', 'Tw1', 'Tw2', 'T1', 'T2', 'T3', 'T4', 'Vsmax', 'Vsmin']
        for param in pss_params:
            val = get(ctrl_obj, param)
            if not np.isnan(val):
                ctrl_data['parameters'][param] = float(val)
    
    return ctrl_data

# ── Power system hierarchy extraction ──────────────────────────────────────
def extract_hierarchy():
    """Extract power system hierarchy"""
    plant_data = []
    generator_to_plant = {}
    control_systems_data = []
    
    if system_type == "hierarchical_plants":
        print(f"🔍 EXTRACTING HIERARCHICAL PLANT STRUCTURE...")
        
        for i, plant in enumerate(plants):
            plant_info = {
                'plant_idx': i,
                'plant_name': safe_get_name(plant),
                'generators': [],
                'control_systems': [],
                'total_capacity_MW': 0.0,
                'plant_type': 'thermal'
            }
            
            # Extract using pelm references
            try:
                pelm = get(plant, 'pelm')
                if pelm and isinstance(pelm, list):
                    for j, element_ref in enumerate(pelm):
                        if element_ref is not None:
                            element_class = safe_get_class(element_ref)
                            
                            if element_class == 'ElmSym':
                                for gen_idx, gen in enumerate(gens):
                                    if gen == element_ref:
                                        plant_info['generators'].append(gen_idx)
                                        generator_to_plant[gen_idx] = i
                                        
                                        p_rated = get(gen, "sgn")
                                        if np.isnan(p_rated):
                                            p_rated = get(gen, "snom")
                                        if np.isnan(p_rated):
                                            p_rated = get(gen, "pgini")
                                        
                                        if not np.isnan(p_rated):
                                            plant_info['total_capacity_MW'] += p_rated
                                        break
                            
                            elif element_class == 'ElmDsl':
                                ctrl_data = extract_control_parameters(element_ref)
                                control_systems_data.append(ctrl_data)
                                plant_info['control_systems'].append(ctrl_data)
                                        
            except Exception as e:
                print(f"Error processing plant {safe_get_name(plant)}: {e}")
            
            plant_data.append(plant_info)
    
    else:
        print(f"🔍 EXTRACTING INDIVIDUAL GENERATOR STRUCTURE...")
        
        for i, gen in enumerate(gens):
            p_rated = get(gen, "sgn")
            if np.isnan(p_rated):
                p_rated = get(gen, "snom") 
            if np.isnan(p_rated):
                p_rated = get(gen, "pgini")
            if np.isnan(p_rated):
                p_rated = 0.0
            
            plant_info = {
                'plant_idx': i,
                'plant_name': f"Virtual Plant {safe_get_name(gen)}",
                'generators': [i],
                'control_systems': [],
                'total_capacity_MW': p_rated,
                'plant_type': 'individual_generator'
            }
            
            generator_to_plant[i] = i
            plant_data.append(plant_info)
    
    return plant_data, generator_to_plant, control_systems_data

# Extract hierarchy
plant_data, gen_to_plant_map, control_systems_data = extract_hierarchy()

# Generator data
gen_tab = dict(
    bus_idx=np.array([safe_bus_idx(g, idx) for g in gens]),
    P_MW=np.array([get(g, "pgini") for g in gens]),
    Q_MVAR=np.array([get(g, "qgini") for g in gens]),
    Vset_pu=np.array([get(g, "uset") for g in gens]),
    H_s=np.array([get(g, "H") for g in gens]),
    D=np.array([get(g, "Damp") for g in gens]),
    X_d=np.array([get(g, "xd") for g in gens]),
    X_q=np.array([get(g, "xq") for g in gens]),
    model=np.array([g.typ_id.loc_name if has(g, "typ_id") and g.typ_id else "Unknown" 
                   for g in gens], dtype="S16"),
    name=np.array([g.loc_name for g in gens], dtype="S24"),
    plant_idx=np.array([gen_to_plant_map.get(i, -1) for i in range(len(gens))]),
)

# Load data
load_tab = dict(
    bus_idx=np.array([safe_bus_idx(l, idx) for l in loads]),
    P_MW=np.array([get(l, "plini") for l in loads]),
    Q_MVAR=np.array([get(l, "qlini") for l in loads]),
    model=np.array([l.typ_id.loc_name if has(l, "typ_id") and l.typ_id else "Unknown" 
                   for l in loads], dtype="S16"),
    name=np.array([l.loc_name for l in loads], dtype="S24"),
)

# Shunt data
def safe_shunt_value(shunt_obj, attr):
    val = get(shunt_obj, attr)
    return 0.0 if np.isnan(val) else val

shunt_tab = dict(
    bus_idx=np.array([safe_bus_idx(s, idx) for s in shunts]),
    G_uS=np.array([safe_shunt_value(s, "g") * µS_to_S * 1e6 for s in shunts]),
    B_uS=np.array([safe_shunt_value(s, "b") * µS_to_S * 1e6 for s in shunts]),
    Q_MVAR=np.array([get(s, "qnom") for s in shunts]),
    model=np.array([s.typ_id.loc_name if has(s, "typ_id") and s.typ_id else "Unknown" 
                   for s in shunts], dtype="S16"),
    name=np.array([s.loc_name for s in shunts], dtype="S24"),
)

print(f"📊 Network Components:")
print(f"   🔌 Buses: {nb}")
print(f"   ⚡ Edges: {len(valid_edges)}")
print(f"   📍 Loads: {len(loads)}")
print(f"   🔋 Generators: {len(gens)}")
print(f"   🔧 Shunts: {len(shunts)}")

# ── Build Y admittance matrix (PROPER METHOD from working code) ────────────
print(f"\n🎯 BUILDING Y ADMITTANCE MATRIX...")

# Initialize Y matrix as full complex matrix first for easier calculation
Y_full = np.zeros((nb, nb), dtype=complex)

# Process edges with valid impedance
valid_edges_for_Y = []
for i in range(len(edge_tab["from_idx"])):
    R = edge_tab["R_ohm"][i]
    X = edge_tab["X_ohm"][i]
    if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
        valid_edges_for_Y.append(i)

print(f"🔌 Valid edges for admittance matrix: {len(valid_edges_for_Y)}/{len(valid_edges)}")

# Build admittance matrix following the proper power system method
for i in valid_edges_for_Y:
    bus_from = edge_tab["from_idx"][i]
    bus_to = edge_tab["to_idx"][i]
    
    # Double-check bus indices are valid
    if bus_from < 0 or bus_to < 0 or bus_from >= nb or bus_to >= nb:
        print(f"⚠️ Skipping edge {i}: invalid bus indices ({bus_from}, {bus_to})")
        continue
        
    R, X = edge_tab["R_ohm"][i], edge_tab["X_ohm"][i]
    B_line = edge_tab["B_uS"][i] * µS_to_S
    tap = edge_tab["tap_ratio"][i]
    phi = np.radians(edge_tab["phi_deg"][i])
    
    # Series impedance and admittance
    Z_series = complex(R, X)
    Y_series = 1.0 / Z_series
    
    # Shunt admittance (half on each side for lines)
    Y_shunt_half = 1j * B_line / 2.0
    
    # Transformer tap handling
    tap_complex = tap * np.exp(1j * phi) if tap != 0 else 1.0
    
    if edge_tab["type_id"][i] == 0:  # Line - Pi model
        # Off-diagonal elements (mutual admittances)
        Y_full[bus_from, bus_to] += -Y_series
        Y_full[bus_to, bus_from] += -Y_series
        
        # Diagonal elements (self-admittances)
        Y_full[bus_from, bus_from] += Y_series + Y_shunt_half
        Y_full[bus_to, bus_to] += Y_series + Y_shunt_half
        
    else:  # Transformer
        # Handle tap ratio
        tap_sq = tap_complex * np.conj(tap_complex)
        
        # Off-diagonal elements
        Y_full[bus_from, bus_to] += -Y_series / np.conj(tap_complex)
        Y_full[bus_to, bus_from] += -Y_series / tap_complex
        
        # Diagonal elements
        Y_full[bus_from, bus_from] += Y_series / tap_sq + Y_shunt_half
        Y_full[bus_to, bus_to] += Y_series + Y_shunt_half

# Add explicit shunt elements to diagonal
for i in range(len(shunts)):
    bus_i = shunt_tab["bus_idx"][i]
    if bus_i >= 0 and bus_i < nb:  # Valid bus index
        G_sh = shunt_tab["G_uS"][i] * µS_to_S
        B_sh = shunt_tab["B_uS"][i] * µS_to_S
        Y_shunt = complex(G_sh, B_sh)
        
        # Add to diagonal (self-admittance)
        Y_full[bus_i, bus_i] += Y_shunt

# Convert to sparse matrix for storage
Y = csr_matrix(Y_full)

print(f"🎯 Admittance matrix Y: {Y.shape}, nnz = {Y.nnz}")

# Verify diagonal elements are not NaN
diagonal_elements = np.diag(Y_full)
nan_count = np.sum(np.isnan(diagonal_elements))
zero_count = np.sum(np.abs(diagonal_elements) < 1e-12)

print(f"📊 Y Matrix Diagnostics:")
print(f"   Diagonal elements: {len(diagonal_elements)}")
print(f"   NaN diagonal elements: {nan_count}")
print(f"   Zero diagonal elements: {zero_count}")
print(f"   Non-zero diagonal elements: {len(diagonal_elements) - nan_count - zero_count}")

if nan_count == 0:
    print(f"✅ Y matrix built successfully with {len(valid_edges_for_Y)} edges")
    
    # Show sample diagonal elements (Yii values)
    print(f"📋 Sample diagonal elements (Yii - self admittances):")
    for i in range(min(5, nb)):
        bus_name = buses[i].loc_name if i < len(buses) else f"Bus_{i}"
        yii = Y_full[i, i]
        print(f"   Y[{i},{i}] ({bus_name}): {yii:.6f}")
else:
    print(f"⚠️ Warning: Found {nan_count} NaN diagonal elements!")

# ── Create node and edge features for GNNs ─────────────────────────────────
print(f"\n🤖 CREATING GNN FEATURES...")

# Node features (per-bus aggregated data)
node_features = np.zeros((nb, 8))  # [V, P_load, Q_load, P_gen, Q_gen, G_sh, B_sh, area]
for i, bus in enumerate(buses):
    node_features[i, 0] = get(bus, "m:u0", 1.0)  # nominal voltage
    node_features[i, 7] = get(bus, "narea", 0)   # area

# Aggregate loads per bus
for j, load in enumerate(loads):
    bus_i = load_tab["bus_idx"][j]
    if bus_i >= 0:  # Valid bus index
        node_features[bus_i, 1] += load_tab["P_MW"][j]   # P_load
        node_features[bus_i, 2] += load_tab["Q_MVAR"][j] # Q_load

# Aggregate generators per bus
for j, gen in enumerate(gens):
    bus_i = gen_tab["bus_idx"][j]
    if bus_i >= 0:  # Valid bus index
        node_features[bus_i, 3] += gen_tab["P_MW"][j]    # P_gen
        node_features[bus_i, 4] += gen_tab["Q_MVAR"][j]  # Q_gen

# Aggregate shunts per bus
for j, shunt in enumerate(shunts):
    bus_i = shunt_tab["bus_idx"][j]
    if bus_i >= 0:  # Valid bus index
        node_features[bus_i, 5] += shunt_tab["G_uS"][j] * µS_to_S  # G_shunt
        node_features[bus_i, 6] += shunt_tab["B_uS"][j] * µS_to_S  # B_shunt

# Edge features
edge_features = np.column_stack([
    edge_tab["R_ohm"],
    edge_tab["X_ohm"],
    edge_tab["B_uS"] * µS_to_S,  # convert to S
    edge_tab["tap_ratio"],
    edge_tab["phi_deg"],
    edge_tab["rate_MVA"],
    edge_tab["type_id"]
])

print(f"✅ Created node features: {node_features.shape}")
print(f"✅ Created edge features: {edge_features.shape}")

# ── Create comprehensive overview ──────────────────────────────────────────
grid_overview = {
    'metadata': {
        'extraction_date': datetime.now().isoformat(),
        'project_name': PROJECT,
        'study_case': STUDY,
        'base_mva': SBASE_MVA,
        'system_type': system_type,
        'description': f'{PROJECT} with Complete Network Topology and Control Systems'
    },
    
    'network_statistics': {
        'buses': len(buses),
        'generators': len(gens),
        'power_plants': len(plants) if system_type == "hierarchical_plants" else 0,
        'virtual_plants': len(plant_data) if system_type == "individual_generators" else 0,
        'transmission_lines': len(lines),
        'transformers': len(transformers),
        'total_edges': len(valid_edges),
        'valid_edges': len(valid_edges_for_Y),
        'loads': len(loads),
        'shunts': len(shunts),
        'y_matrix_nnz': Y.nnz,
        'total_generation_mw': sum(plant['total_capacity_MW'] for plant in plant_data),
        'total_load_mw': sum(get(load, "plini", 0.0) for load in loads)
    },
    
    'control_systems_inventory': {
        'avr_systems': len(avr_systems),
        'governor_systems': len(gov_systems),
        'pss_systems': len(pss_systems),
        'total_control_devices': len(avr_systems) + len(gov_systems) + len(pss_systems),
        'total_dsl_models': len(all_dsl)
    },
    
    'power_plants_summary': [],
    'control_systems_details': control_systems_data
}

# Populate power plants summary
for plant in plant_data:
    plant_summary = {
        'name': plant['plant_name'],
        'type': plant['plant_type'],
        'generators': len(plant['generators']),
        'capacity_mw': plant['total_capacity_MW'],
        'control_systems': len(plant['control_systems'])
    }
    grid_overview['power_plants_summary'].append(plant_summary)

# ── Save complete enhanced HDF5 data ───────────────────────────────────────
print(f"\n💾 SAVING COMPLETE ENHANCED H5 FILE...")
os.makedirs(OUT_DIR, exist_ok=True)
h5_path = os.path.join(OUT_DIR, f"{PROJECT.replace(' ', '_')}_fixed_complete_enhanced.h5")

with h5py.File(h5_path, "w") as f:
    # ═══════════════════════════════════════════════════════════════════════
    # NETWORK TOPOLOGY DATA
    # ═══════════════════════════════════════════════════════════════════════
    
    # Bus data
    bus_grp = f.create_group("bus")
    for k, v in bus_tab.items():
        bus_grp.create_dataset(k, data=v)
    
    # Edge data  
    edge_grp = f.create_group("edge")
    for k, v in edge_tab.items():
        edge_grp.create_dataset(k, data=v)
    
    # Load data
    load_grp = f.create_group("load")
    for k, v in load_tab.items():
        load_grp.create_dataset(k, data=v)
    
    # Generator data
    gen_grp = f.create_group("gen")
    for k, v in gen_tab.items():
        gen_grp.create_dataset(k, data=v)
    
    # Shunt data
    shunt_grp = f.create_group("shunt")
    for k, v in shunt_tab.items():
        shunt_grp.create_dataset(k, data=v)
    
    # GNN features
    feat_grp = f.create_group("features")
    feat_grp.create_dataset("node_features", data=node_features)
    feat_grp.create_dataset("edge_features", data=edge_features)
    feat_grp.create_dataset("feature_names_node", 
                           data=[s.encode() for s in ["V_kV", "P_load", "Q_load", "P_gen", "Q_gen", "G_shunt", "B_shunt", "area"]])
    feat_grp.create_dataset("feature_names_edge",
                           data=[s.encode() for s in ["R_ohm", "X_ohm", "B_S", "tap_ratio", "phi_deg", "rate_MVA", "type_id"]])
    
    # Y Admittance matrix (sparse format)
    Y_grp = f.create_group("admittance")
    Y_grp.create_dataset("data", data=Y.data)
    Y_grp.create_dataset("indices", data=Y.indices)
    Y_grp.create_dataset("indptr", data=Y.indptr)
    Y_grp.create_dataset("shape", data=Y.shape)
    Y_grp.create_dataset("nnz", data=Y.nnz)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ENHANCED POWER PLANT DATA
    # ═══════════════════════════════════════════════════════════════════════
    
    # Power plant hierarchy
    plant_grp = f.create_group("power_plants")
    plant_grp.create_dataset("num_plants", data=len(plant_data))
    plant_grp.create_dataset("system_type", data=system_type.encode())
    
    for i, plant in enumerate(plant_data):
        plant_subgrp = plant_grp.create_group(f"plant_{i}")
        plant_subgrp.create_dataset("name", data=plant['plant_name'].encode())
        plant_subgrp.create_dataset("type", data=plant['plant_type'].encode())
        plant_subgrp.create_dataset("capacity_MW", data=plant['total_capacity_MW'])
        plant_subgrp.create_dataset("generators", data=plant['generators'])
        
        # Control systems for this plant
        if plant['control_systems']:
            ctrl_grp = plant_subgrp.create_group("control_systems")
            for j, ctrl in enumerate(plant['control_systems']):
                ctrl_subgrp = ctrl_grp.create_group(f"control_{j}")
                ctrl_subgrp.create_dataset("name", data=ctrl['name'].encode())
                ctrl_subgrp.create_dataset("type", data=ctrl['type'].encode())
                
                # Store parameters
                if ctrl['parameters']:
                    param_grp = ctrl_subgrp.create_group("parameters")
                    for param_name, param_value in ctrl['parameters'].items():
                        param_grp.create_dataset(param_name, data=param_value)
    
    # Control systems summary
    ctrl_summary_grp = f.create_group("control_systems_summary")
    ctrl_summary_grp.create_dataset("total_avr", data=len(avr_systems))
    ctrl_summary_grp.create_dataset("total_gov", data=len(gov_systems))
    ctrl_summary_grp.create_dataset("total_pss", data=len(pss_systems))
    ctrl_summary_grp.create_dataset("total_dsl", data=len(all_dsl))
    
    # Comprehensive metadata
    meta_grp = f.create_group("metadata")
    meta_grp.create_dataset("extraction_date", data=datetime.now().isoformat().encode())
    meta_grp.create_dataset("system_type", data=system_type.encode())
    meta_grp.create_dataset("total_buses", data=len(buses))
    meta_grp.create_dataset("total_generators", data=len(gens))
    meta_grp.create_dataset("total_plants", data=len(plant_data))
    meta_grp.create_dataset("total_capacity_MW", data=sum(plant['total_capacity_MW'] for plant in plant_data))
    meta_grp.create_dataset("total_edges", data=len(valid_edges))
    meta_grp.create_dataset("valid_edges", data=len(valid_edges_for_Y))
    meta_grp.create_dataset("y_matrix_nnz", data=Y.nnz)

# Save YAML overview
yaml_path = os.path.join(OUT_DIR, f"{PROJECT.replace(' ', '_')}_fixed_overview.yml")
with open(yaml_path, 'w') as f:
    yaml.dump(grid_overview, f, default_flow_style=False, sort_keys=False, indent=2)

print(f"✅ Complete enhanced data saved to: {h5_path}")
print(f"   📁 Size: {os.path.getsize(h5_path) / 1024:.1f} KB")
print(f"✅ Grid overview saved to: {yaml_path}")

# ── Final comprehensive summary ────────────────────────────────────────────
print(f"\n📊 COMPLETE EXTRACTION SUMMARY:")
print("="*60)
print(f"🏗️ System Type: {system_type.replace('_', ' ').title()}")
print(f"🔌 Buses: {nb}")
print(f"⚡ Edges: {len(valid_edges)} (valid for Y: {len(valid_edges_for_Y)})")
print(f"🎯 Y Matrix: {Y.shape} with {Y.nnz} non-zeros")
print(f"📍 Loads: {len(loads)} (total: {load_tab['P_MW'].sum():.1f} MW)")
print(f"🔋 Generators: {len(gens)} (total: {gen_tab['P_MW'].sum():.1f} MW)")
print(f"🔧 Shunts: {len(shunts)}")

if system_type == "hierarchical_plants":
    print(f"🏭 Power Plants: {len(plant_data)}")
else:
    print(f"🏭 Virtual Plants (1 per generator): {len(plant_data)}")

print(f"🎛️ Control Systems: {len(control_systems_data)} found")
print(f"   • AVR: {len(avr_systems)}")
print(f"   • GOV: {len(gov_systems)}")  
print(f"   • PSS: {len(pss_systems)}")
print(f"   • Total DSL: {len(all_dsl)}")

if len(plant_data) > 0:
    plants_with_control = sum(1 for plant in plant_data if len(plant['control_systems']) > 0)
    print(f"🎯 Plants with Control Systems: {plants_with_control}/{len(plant_data)} ({plants_with_control/len(plant_data)*100:.0f}%)")

print(f"\n🎉 FIXED EXTRACTION COMPLETE!")
print(f"📁 Output directory: {OUT_DIR}")
print(f"   • Fixed enhanced HDF5: {os.path.basename(h5_path)}")
print(f"   • Grid overview YAML: {os.path.basename(yaml_path)}")
print(f"\n🚀 Key fixes applied:")
print(f"   ✅ Proper impedance extraction using working branch_imp() function")
print(f"   ✅ R and X arrays properly defined before edge_tab creation")
print(f"   ✅ Robust bus connection handling")
print(f"   ✅ Y matrix construction follows working code pattern")
print(f"   ✅ Complete error handling and validation")
print(f"\n💡 This file should work without NameError issues!")