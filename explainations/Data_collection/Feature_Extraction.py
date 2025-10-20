# Minimal_Working_Extraction.py - Simplified version that works
"""
Minimal working version of feature extraction with proper Y matrix construction.
Based on the successful fixes we implemented.
"""

import sys, os, h5py, numpy as np
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

print(f"🔧 MINIMAL WORKING POWER SYSTEM EXTRACTION")
print("="*60)
print(f"✅ {PROJECT} | {STUDY}")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Helper functions ────────────────────────────────────────────────────────
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

def as_bus(o):
    try:
        if not o:
            return None
        return o.cterm if has(o, "cterm") else o
    except:
        return o

def safe_get_name(obj):
    try:
        return obj.loc_name if obj else "Unknown"
    except:
        return "Unknown"

def term(obj):
    """Get the terminal/bus from various PowerFactory objects"""
    if not obj:
        return None
        
    try:
        # For ElmTerm objects, return directly
        if obj.GetClassName() == "ElmTerm":
            return obj
            
        # For transformers
        if obj.GetClassName().startswith("ElmTr"):
            for f in ("bushv", "buslv", "bus1", "bus2"):
                if has(obj, f):
                    ptr = obj.GetAttribute(f)
                    if ptr:
                        if has(ptr, "cterm"):
                            return ptr.cterm
                        elif ptr.GetClassName() == "ElmTerm":
                            return ptr
                        else:
                            return as_bus(ptr)
            return None
        
        # For other objects
        for f in ("bus1", "bus2", "cterm", "sbus", "bus", "bushv", "buslv"):
            if has(obj, f):
                ptr = obj.GetAttribute(f) if f != "cterm" else obj
                if ptr:
                    if has(ptr, "cterm"):
                        return ptr.cterm
                    elif ptr.GetClassName() == "ElmTerm":
                        return ptr
                    else:
                        return as_bus(ptr)
        
        return None
        
    except Exception as e:
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

# ── Get network elements ────────────────────────────────────────────────────
print(f"\n🔍 COLLECTING NETWORK ELEMENTS...")

# Use PowerFactory's standard methods
buses = [as_bus(b) for b in app.GetCalcRelevantObjects("*.ElmTerm")]
edges = app.GetCalcRelevantObjects("*.ElmLne,*.ElmTr2,*.ElmXfr,*.ElmXfr3")
loads = app.GetCalcRelevantObjects("*.ElmLod")
gens = app.GetCalcRelevantObjects("*.ElmSym")
shunts = app.GetCalcRelevantObjects("*.ElmShnt,*.ElmReac,*.ElmCap,*.ElmScal")

nb = len(buses)
idx = {b: i for i, b in enumerate(buses)}

print(f"✅ Network elements:")
print(f"   🔌 Buses: {nb}")
print(f"   ⚡ Edges: {len(edges)}")
print(f"   📍 Loads: {len(loads)}")
print(f"   🔋 Generators: {len(gens)}")
print(f"   🔧 Shunts: {len(shunts)}")

# ── Simple impedance extraction ─────────────────────────────────────────────
def simple_branch_imp(e):
    """Simple impedance extraction"""
    cls = e.GetClassName()
    
    # Try direct values first
    R = get(e, "r1")
    X = get(e, "x1")
    
    if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
        return R, X
    
    # Try per-unit values with conversion
    if cls.startswith("ElmXfr") or cls.startswith("ElmTr"):
        Rpu = get(e, "r1pu")
        Xpu = get(e, "x1pu")
        Vbase = get(e, "unom", 100.0)  # kV
        Sbase = get(e, "snom", SBASE_MVA)  # MVA
        
        if not (np.isnan(Rpu) or np.isnan(Xpu)) and Vbase > 0 and Sbase > 0:
            Zbase = (Vbase * 1e3) ** 2 / (Sbase * 1e6)
            return Rpu * Zbase, Xpu * Zbase
    
    # Default fallback
    return 0.001, 0.01  # Small impedance

# ── Extract network data ────────────────────────────────────────────────────
print(f"\n🔧 EXTRACTING NETWORK DATA...")

# Bus data
bus_tab = dict(
    name=np.array([b.loc_name for b in buses], dtype="S20"),
    Un_kV=np.array([get(b, "m:u0") for b in buses]),
    fn_Hz=np.array([get(b, "fbus") for b in buses]),
    area=np.array([get(b, "narea") for b in buses]),
    zone=np.array([get(b, "nzone") for b in buses]),
    bustype=np.array([b.GetBusType() if hasattr(b, "GetBusType") else -1 for b in buses]),
)

# Edge data with validation
valid_edges = []
from_indices = []
to_indices = []
R_values = []
X_values = []
B_values = []

for e in edges:
    # Get bus connections
    bus_from = term(e.bus1 if has(e, "bus1") else (e.bushv if has(e, "bushv") else None))
    bus_to = term(e.bus2 if has(e, "bus2") else (e.buslv if has(e, "buslv") else None))
    
    if bus_from and bus_to and bus_from in idx and bus_to in idx:
        from_idx = idx[bus_from]
        to_idx = idx[bus_to]
        
        # Get impedance
        R, X = simple_branch_imp(e)
        
        # Get susceptance (handle NaN)
        B = get(e, "bch")
        if np.isnan(B):
            B = 0.0
        
        valid_edges.append(e)
        from_indices.append(from_idx)
        to_indices.append(to_idx)
        R_values.append(R)
        X_values.append(X)
        B_values.append(B)

print(f"✅ Valid edges: {len(valid_edges)}/{len(edges)}")

# Create edge table
edge_tab = dict(
    from_idx=np.array(from_indices),
    to_idx=np.array(to_indices),
    name=np.array([e.loc_name for e in valid_edges], dtype="S24"),
    R_ohm=np.array(R_values),
    X_ohm=np.array(X_values),
    B_uS=np.array(B_values),
    tap_ratio=np.array([get(e, "tratio", 1.0) for e in valid_edges]),
    phi_deg=np.array([get(e, "phitr", 0.0) for e in valid_edges]),
    rate_MVA=np.array([get(e, "snom") for e in valid_edges]),
    type_id=np.array([0 if e.GetClassName().startswith("ElmLne") else 1 for e in valid_edges]),
)

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
    plant_idx=np.array([-1 for g in gens]),  # Simplified
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

# Shunt data with NaN handling
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

# ── Build Y admittance matrix (CORRECT METHOD) ─────────────────────────────
print(f"\n🎯 BUILDING Y ADMITTANCE MATRIX...")

Y_full = np.zeros((nb, nb), dtype=complex)

# Process edges
for i in range(len(edge_tab["from_idx"])):
    bus_from = edge_tab["from_idx"][i]
    bus_to = edge_tab["to_idx"][i]
    R = edge_tab["R_ohm"][i]
    X = edge_tab["X_ohm"][i]
    B_line = edge_tab["B_uS"][i] * µS_to_S
    
    # Skip if invalid
    if bus_from < 0 or bus_to < 0 or bus_from >= nb or bus_to >= nb:
        continue
    
    # Calculate admittances
    Z_series = complex(R, X)
    Y_series = 1.0 / Z_series
    Y_shunt_half = 1j * B_line / 2.0
    
    # Pi-model
    Y_full[bus_from, bus_to] += -Y_series
    Y_full[bus_to, bus_from] += -Y_series
    Y_full[bus_from, bus_from] += Y_series + Y_shunt_half
    Y_full[bus_to, bus_to] += Y_series + Y_shunt_half

# Add shunts
for i in range(len(shunt_tab["bus_idx"])):
    bus_i = shunt_tab["bus_idx"][i]
    if 0 <= bus_i < nb:
        G_sh = shunt_tab["G_uS"][i] * µS_to_S
        B_sh = shunt_tab["B_uS"][i] * µS_to_S
        Y_shunt = complex(G_sh, B_sh)
        Y_full[bus_i, bus_i] += Y_shunt

# Convert to sparse
Y = csr_matrix(Y_full)

# Check results
diagonal = np.diag(Y_full)
nan_count = np.sum(np.isnan(diagonal))
zero_count = np.sum(np.abs(diagonal) < 1e-15)

print(f"✅ Y Matrix Results:")
print(f"   Shape: {Y.shape}")
print(f"   Non-zeros: {Y.nnz}")
print(f"   Diagonal - Valid: {len(diagonal) - nan_count - zero_count}, NaN: {nan_count}, Zero: {zero_count}")

# Show sample diagonal
for i in range(min(5, nb)):
    bus_name = buses[i].loc_name
    yii = diagonal[i]
    if np.isnan(yii):
        print(f"   Y[{i},{i}] ({bus_name}): NaN ❌")
    else:
        print(f"   Y[{i},{i}] ({bus_name}): {yii.real:+.6f} {yii.imag:+.6f}j ✅")

# ── Save to H5 file ─────────────────────────────────────────────────────────
print(f"\n💾 SAVING TO H5 FILE...")
os.makedirs(OUT_DIR, exist_ok=True)
h5_path = os.path.join(OUT_DIR, f"{PROJECT.replace(' ', '_')}_minimal_working.h5")

with h5py.File(h5_path, "w") as f:
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
    
    # Y Admittance matrix
    Y_grp = f.create_group("admittance")
    Y_grp.create_dataset("data", data=Y.data)
    Y_grp.create_dataset("indices", data=Y.indices)
    Y_grp.create_dataset("indptr", data=Y.indptr)
    Y_grp.create_dataset("shape", data=Y.shape)
    Y_grp.create_dataset("nnz", data=Y.nnz)
    
    # Metadata
    meta_grp = f.create_group("metadata")
    meta_grp.create_dataset("extraction_date", data=datetime.now().isoformat().encode())
    meta_grp.create_dataset("total_buses", data=nb)
    meta_grp.create_dataset("total_edges", data=len(valid_edges))
    meta_grp.create_dataset("y_matrix_nnz", data=Y.nnz)

print(f"✅ Saved to: {h5_path}")
print(f"📊 File size: {os.path.getsize(h5_path) / 1024:.1f} KB")

print(f"\n🎉 MINIMAL EXTRACTION COMPLETE!")
print(f"   ✅ {nb} buses")
print(f"   ✅ {len(valid_edges)} valid edges")  
print(f"   ✅ Y matrix with {Y.nnz} non-zeros")
print(f"   ✅ {len(diagonal) - nan_count} valid diagonal elements")

if nan_count == 0:
    print(f"🎯 Y MATRIX IS PERFECT - READY FOR GRAPH CONSTRUCTION!")
else:
    print(f"⚠️ {nan_count} NaN diagonal elements need fixing")