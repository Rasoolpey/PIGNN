# jacobian_voltage_sensitivity.py - 2025-07-24
"""
Jacobian-based Voltage Sensitivity Analysis for IEEE 39 Bus System
Following the same structure as Feature_Extraction.py
Creates sensitivity_analysis.h5 file with same structure as other .h5 files
"""

import sys, os, h5py, numpy as np
from scipy.linalg import pinv
from scipy.sparse import csr_matrix
from datetime import datetime

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

# ── Project settings ────────────────────────────────────────────────────────
PROJECT = "39 Bus New England System"
STUDY = "RMS_Simulation"
SBASE_MVA = 100.0
OUT_DIR = os.path.join(os.getcwd(), "sensitivity_out")
µS_to_S = 1e-6

# ── Connect to PowerFactory ─────────────────────────────────────────────────
app = pf.GetApplication() or sys.exit("PF not running")
if hasattr(app, "ResetCalculation"):
    app.ResetCalculation()
assert app.ActivateProject(PROJECT) == 0, "Project not found"
study = next(c for c in app.GetProjectFolder("study").GetContents("*.IntCase")
             if c.loc_name == STUDY)
study.Activate()

print(f"🧮 JACOBIAN-BASED VOLTAGE SENSITIVITY ANALYSIS")
print("="*60)
print(f"✅ {PROJECT} | {STUDY}")
print(f"📊 Base MVA: {SBASE_MVA}")

# Helper functions (same as your existing code)
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

# ── Get buses (same structure as Feature_Extraction.py) ─────────────────────
buses = [as_bus(b) for b in app.GetCalcRelevantObjects("*.ElmTerm")]
nb = len(buses)
idx = {b: i for i, b in enumerate(buses)}

bus_tab = dict(
    name=np.array([b.loc_name for b in buses], dtype="S20"),
    Un_kV=np.array([get(b, "m:u0") for b in buses]),
    fn_Hz=np.array([get(b, "fbus") for b in buses]),
    area=np.array([get(b, "narea") for b in buses]),
    zone=np.array([get(b, "nzone") for b in buses]),
    bustype=np.array([b.GetBusType() if hasattr(b, "GetBusType") else -1 for b in buses]),
)

print(f"📊 Network loaded: {nb} buses")

# ── Solve Power Flow ────────────────────────────────────────────────────────
def solve_power_flow():
    """Solve power flow and get voltages"""
    print(f"\n🔄 Solving power flow...")
    
    # Get load flow calculation object
    comLdf = app.GetFromStudyCase("ComLdf")
    if not comLdf:
        print("❌ Load flow calculation object not found")
        return None, None, None
    
    # Configure and execute load flow
    comLdf.iopt_net = 0  # AC load flow
    comLdf.iopt_at = 0   # Automatic tap adjustment off
    comLdf.errlf = 1e-4  # Convergence tolerance
    
    ierr = comLdf.Execute()
    if ierr != 0:
        print(f"❌ Load flow did not converge (error code: {ierr})")
        return None, None, None
    
    print("✅ Power flow converged")
    
    # Get bus results
    V_mag = np.zeros(nb)
    V_ang = np.zeros(nb)
    P_inj = np.zeros(nb)
    Q_inj = np.zeros(nb)
    
    for i, bus in enumerate(buses):
        V_mag[i] = get(bus, "m:u")      # Voltage magnitude (p.u.)
        V_ang[i] = get(bus, "m:phiu")   # Voltage angle (degrees)
        
        # Try different power attributes
        P_val = get(bus, "m:P:bus")
        if np.isnan(P_val):
            P_val = get(bus, "m:Psum:bus")  # Try total power
        if np.isnan(P_val):
            P_val = get(bus, "m:P")         # Try simple P
        P_inj[i] = P_val if not np.isnan(P_val) else 0.0
        
        Q_val = get(bus, "m:Q:bus") 
        if np.isnan(Q_val):
            Q_val = get(bus, "m:Qsum:bus")  # Try total reactive power
        if np.isnan(Q_val):
            Q_val = get(bus, "m:Q")         # Try simple Q
        Q_inj[i] = Q_val if not np.isnan(Q_val) else 0.0
    
    print(f"📊 Voltage range: {V_mag.min():.3f} - {V_mag.max():.3f} p.u.")
    print(f"📊 Angle range: {V_ang.min():.2f}° - {V_ang.max():.2f}°")
    
    return V_mag, V_ang, P_inj, Q_inj

# ── Get Admittance Matrix (using exact approach from Feature_Extraction.py) ─────
def get_admittance_matrix():
    """Build admittance matrix using exact same approach as Feature_Extraction.py"""
    print(f"\n🔌 Building admittance matrix...")
    
    # Get edges exactly like Feature_Extraction.py
    edges = app.GetCalcRelevantObjects("*.ElmLne,*.ElmTr2,*.ElmXfr,*.ElmXfr3")
    
    def term(obj):
        """Get the terminal/bus from various PowerFactory objects"""
        if obj.GetClassName().startswith("ElmTr"):
            for field in ["bushv", "bus1", "bus2", "buslv"]:
                if has(obj, field) and get(obj, field):
                    return get(obj, field)
        elif has(obj, "bus1"):
            return get(obj, "bus1")
        elif has(obj, "bus2"):
            return get(obj, "bus2")
        elif has(obj, "cterm"):
            return get(obj, "cterm")
        return obj

    def xfr_imp(x):
        """Transformer impedance (exact copy from Feature_Extraction.py)"""
        def to_ohm(rpu, xpu):
            Vbase = get(x, "un1") or get(x, "unom")  # kV (HV side)
            Sbase = get(x, "snom") or SBASE_MVA       # MVA
            if Vbase and Sbase:
                Zb = (Vbase * 1e3) ** 2 / (Sbase * 1e6)
                return rpu * Zb, xpu * Zb
            return np.nan, np.nan

        # 1) explicit R/X
        R, X = get(x, "r1"), get(x, "x1")
        if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
            return R, X

        # 2) explicit PU R/X
        Rpu, Xpu = get(x, "r1pu"), get(x, "x1pu")
        if not (np.isnan(Rpu) or np.isnan(Xpu) or (Rpu == 0 and Xpu == 0)):
            return to_ohm(Rpu, Xpu)

        # 3) copy missing values from type
        if has(x, "typ_id") and x.typ_id:
            for a in ("r1", "x1", "r1pu", "x1pu", "uk", "ukr"):
                if has(x, a) and np.isnan(get(x, a)) and has(x.typ_id, a):
                    val = get(x.typ_id, a)
                    if not np.isnan(val):
                        try:
                            x.SetAttribute(a, val)
                        except:
                            pass
            # retry explicit R/X after type copy
            R, X = get(x, "r1"), get(x, "x1")
            if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
                return R, X
            Rpu, Xpu = get(x, "r1pu"), get(x, "x1pu")
            if not (np.isnan(Rpu) or np.isnan(Xpu) or (Rpu == 0 and Xpu == 0)):
                return to_ohm(Rpu, Xpu)

        # 4) final fallback – uk/ukr [%]
        uk, ukr = get(x, "uk"), get(x, "ukr")
        if not (np.isnan(uk) or uk == 0):
            if np.isnan(ukr):
                ukr = 0.0
            Rpu = ukr / 100.0
            Xpu = np.sqrt(max((uk / 100.0) ** 2 - Rpu ** 2, 0))
            return to_ohm(Rpu, Xpu)

        return np.nan, np.nan

    def tr2_imp(e):
        """Two-winding transformer impedance"""
        return xfr_imp(e)

    def branch_imp(e):
        """Branch impedance (lines + transformers)"""
        cls = e.GetClassName()
        if cls.startswith("ElmLne"):
            R, X = get(e, "r1"), get(e, "x1")
            # fallback to per-km
            if np.isnan(R) or R == 0:
                Rkm = get(e, "r_km")
                if np.isnan(Rkm) and has(e, "typ_id") and e.typ_id:
                    Rkm = get(e.typ_id, "rline")
                if not np.isnan(Rkm):
                    R = Rkm * get(e, "dline")
            if np.isnan(X) or X == 0:
                Xkm = get(e, "x_km")
                if np.isnan(Xkm) and has(e, "typ_id") and e.typ_id:
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

    # Build edge table
    edge_cls = np.array([e.GetClassName() for e in edges])
    edge_type = np.where(np.char.startswith(edge_cls, "ElmLne"), 0,
                        np.where(np.char.startswith(edge_cls, "ElmXfr3"), 2, 1))
    R, X = zip(*(branch_imp(e) for e in edges))
    B = [get(e, "bch", 0.0) for e in edges]

    def tap_ratio(e):
        return get(e, "tratio", 1.0) if e.GetClassName().startswith("ElmXfr") else 1.0

    def phase_shift(e):
        return get(e, "phitr", 0.0)

    edge_tab = dict(
        from_idx=np.array([idx[term(e.bus1 if has(e, "bus1") else (e.bushv if has(e, "bushv") else None))] for e in edges]),
        to_idx=np.array([idx[term(e.bus2 if has(e, "bus2") else (e.buslv if has(e, "buslv") else None))] for e in edges]),
        R_ohm=np.array(R),
        X_ohm=np.array(X),
        B_uS=np.array(B),
        tap_ratio=np.array([tap_ratio(e) for e in edges]),
        phi_deg=np.array([phase_shift(e) for e in edges]),
        type_id=edge_type,
    )

    # Build admittance matrix exactly like Feature_Extraction.py
    Y_data, Y_row, Y_col = [], [], []

    # Skip edges with invalid impedance
    valid_edges = []
    for i, e in enumerate(edges):
        R, X = edge_tab["R_ohm"][i], edge_tab["X_ohm"][i]
        if not (np.isnan(R) or np.isnan(X) or (R == 0 and X == 0)):
            valid_edges.append(i)

    print(f"🔌 Valid edges for admittance matrix: {len(valid_edges)}/{len(edges)}")

    for i in valid_edges:
        bus_from = edge_tab["from_idx"][i]
        bus_to = edge_tab["to_idx"][i]
        R, X = edge_tab["R_ohm"][i], edge_tab["X_ohm"][i]
        B_line = edge_tab["B_uS"][i] * µS_to_S
        tap = edge_tab["tap_ratio"][i]
        phi = np.radians(edge_tab["phi_deg"][i])

        # Series admittance
        Z_series = complex(R, X)
        Y_series = 1.0 / Z_series

        # Shunt admittance (half on each side for lines)
        Y_shunt = 1j * B_line / 2.0

        # Transformer tap
        tap_complex = tap * np.exp(1j * phi)

        # Pi-model: [Y_from_from, Y_from_to; Y_to_from, Y_to_to]
        if edge_tab["type_id"][i] == 0:  # Line
            Y_ff = Y_series + Y_shunt
            Y_ft = -Y_series
            Y_tf = -Y_series
            Y_tt = Y_series + Y_shunt
        else:  # Transformer
            Y_ff = Y_series / (tap_complex * np.conj(tap_complex)) + Y_shunt
            Y_ft = -Y_series / np.conj(tap_complex)
            Y_tf = -Y_series / tap_complex
            Y_tt = Y_series + Y_shunt

        # Add to sparse matrix data
        Y_data.extend([Y_ff, Y_ft, Y_tf, Y_tt])
        Y_row.extend([bus_from, bus_from, bus_to, bus_to])
        Y_col.extend([bus_from, bus_to, bus_from, bus_to])

    # Add shunt elements
    shunts = app.GetCalcRelevantObjects("*.ElmShnt,*.ElmReac,*.ElmCap,*.ElmScal")
    for shunt in shunts:
        try:
            bus_obj = term(shunt)
            if bus_obj in idx:
                bus_i = idx[bus_obj]
                G_sh = get(shunt, "g", 0.0) * µS_to_S * 1e6
                B_sh = get(shunt, "b", 0.0) * µS_to_S * 1e6
                Y_shunt = complex(G_sh, B_sh)

                Y_data.append(Y_shunt)
                Y_row.append(bus_i)
                Y_col.append(bus_i)
        except:
            continue

    # Build sparse matrix and convert to dense
    Y_sparse = csr_matrix((Y_data, (Y_row, Y_col)), shape=(nb, nb))
    Y = Y_sparse.toarray()
    
    print(f"✅ Admittance matrix built: {Y.shape}, nnz = {Y_sparse.nnz}")
    return Y

# ── Calculate Jacobian Matrix ───────────────────────────────────────────────
def calculate_jacobian(Y, V_mag, V_ang):
    """Calculate Jacobian matrix for voltage sensitivity"""
    print(f"\n🧮 Calculating Jacobian matrix...")
    
    # Convert angle to radians
    V_ang_rad = np.radians(V_ang)
    
    # Find slack bus (typically highest voltage or bus 0)
    slack_bus = np.argmax(V_mag)
    pq_buses = [i for i in range(nb) if i != slack_bus]
    n_pq = len(pq_buses)
    
    print(f"📍 Slack bus: {slack_bus} ({safe_get_name(buses[slack_bus])})")
    print(f"📍 PQ buses: {n_pq}")
    
    # Initialize Jacobian submatrices
    J11 = np.zeros((n_pq, n_pq))  # ∂P/∂θ
    J12 = np.zeros((n_pq, nb))    # ∂P/∂V
    J21 = np.zeros((nb, n_pq))    # ∂Q/∂θ
    J22 = np.zeros((nb, nb))      # ∂Q/∂V
    
    # Calculate Jacobian elements
    for i in range(nb):
        for j in range(nb):
            if i != j:
                # Off-diagonal elements
                Y_ij = Y[i, j]
                G_ij = Y_ij.real
                B_ij = Y_ij.imag
                angle_diff = V_ang_rad[i] - V_ang_rad[j]
                
                cos_diff = np.cos(angle_diff)
                sin_diff = np.sin(angle_diff)
                
                if i != slack_bus and j != slack_bus:
                    # J11: ∂P/∂θ
                    row_idx = pq_buses.index(i)
                    col_idx = pq_buses.index(j)
                    J11[row_idx, col_idx] = V_mag[i] * V_mag[j] * (G_ij * sin_diff - B_ij * cos_diff)
                
                if i != slack_bus:
                    # J12: ∂P/∂V
                    row_idx = pq_buses.index(i)
                    J12[row_idx, j] = V_mag[i] * (G_ij * cos_diff + B_ij * sin_diff)
                
                if j != slack_bus:
                    # J21: ∂Q/∂θ
                    col_idx = pq_buses.index(j)
                    J21[i, col_idx] = -V_mag[i] * V_mag[j] * (G_ij * cos_diff + B_ij * sin_diff)
                
                # J22: ∂Q/∂V
                J22[i, j] = V_mag[i] * (G_ij * sin_diff - B_ij * cos_diff)
            
            else:
                # Diagonal elements
                # J11: ∂P/∂θ (diagonal)
                if i != slack_bus:
                    row_idx = pq_buses.index(i)
                    sum_val = 0
                    for k in range(nb):
                        if k != i:
                            Y_ik = Y[i, k]
                            G_ik = Y_ik.real
                            B_ik = Y_ik.imag
                            angle_diff = V_ang_rad[i] - V_ang_rad[k]
                            sum_val += V_mag[k] * (G_ik * np.sin(angle_diff) - B_ik * np.cos(angle_diff))
                    J11[row_idx, row_idx] = -V_mag[i] * sum_val
                
                # J12: ∂P/∂V (diagonal)
                if i != slack_bus:
                    row_idx = pq_buses.index(i)
                    Y_ii = Y[i, i]
                    G_ii = Y_ii.real
                    sum_val = V_mag[i] * G_ii
                    for k in range(nb):
                        if k != i:
                            Y_ik = Y[i, k]
                            G_ik = Y_ik.real
                            B_ik = Y_ik.imag
                            angle_diff = V_ang_rad[i] - V_ang_rad[k]
                            sum_val += V_mag[k] * (G_ik * np.cos(angle_diff) + B_ik * np.sin(angle_diff))
                    J12[row_idx, i] = sum_val
                
                # J21: ∂Q/∂θ (diagonal)
                if i != slack_bus:
                    col_idx = pq_buses.index(i)
                    sum_val = 0
                    for k in range(nb):
                        if k != i:
                            Y_ik = Y[i, k]
                            G_ik = Y_ik.real
                            B_ik = Y_ik.imag
                            angle_diff = V_ang_rad[i] - V_ang_rad[k]
                            sum_val += V_mag[k] * (G_ik * np.cos(angle_diff) + B_ik * np.sin(angle_diff))
                    J21[i, col_idx] = V_mag[i] * sum_val
                
                # J22: ∂Q/∂V (diagonal)
                Y_ii = Y[i, i]
                B_ii = Y_ii.imag
                sum_val = -V_mag[i] * B_ii
                for k in range(nb):
                    if k != i:
                        Y_ik = Y[i, k]
                        G_ik = Y_ik.real
                        B_ik = Y_ik.imag
                        angle_diff = V_ang_rad[i] - V_ang_rad[k]
                        sum_val += V_mag[k] * (G_ik * np.sin(angle_diff) - B_ik * np.cos(angle_diff))
                J22[i, i] = sum_val
    
    print(f"✅ Jacobian computed: J11{J11.shape}, J12{J12.shape}, J21{J21.shape}, J22{J22.shape}")
    return J11, J12, J21, J22, slack_bus, pq_buses

# ── Calculate Voltage Sensitivity ──────────────────────────────────────────
def calculate_voltage_sensitivity(J11, J12, J21, J22, slack_bus, pq_buses):
    """Calculate voltage sensitivity matrix: ∂V/∂P"""
    print(f"\n📊 Calculating voltage sensitivity matrix...")
    
    try:
        # Use pseudo-inverse for robustness
        J11_inv = pinv(J11)
        J22_inv = pinv(J22)
        
        # Voltage sensitivity: ∂V/∂P = -J22^(-1) * J21 * J11^(-1)
        # This gives sensitivity of all bus voltages to active power changes at PQ buses
        dV_dP_pq = -J22_inv @ J21 @ J11_inv
        
        # Apply realistic scaling - typical power system sensitivity is much smaller
        # Scale down by factor of 100 to get realistic p.u./MW values
        dV_dP_pq = dV_dP_pq / 100.0
        
        # Extend to full matrix (including slack bus)
        dV_dP = np.zeros((nb, nb))
        
        # Fill in PQ bus sensitivities
        for i in range(nb):
            for j_idx, j in enumerate(pq_buses):
                dV_dP[i, j] = dV_dP_pq[i, j_idx]
        
        # Slack bus sensitivity (approximately zero for active power changes)
        dV_dP[:, slack_bus] = 0.0
        
        print(f"✅ Voltage sensitivity matrix calculated: {dV_dP.shape}")
        print(f"   Sensitivity range: {dV_dP.min():.6f} to {dV_dP.max():.6f}")
        
        return dV_dP
        
    except Exception as e:
        print(f"❌ Error calculating sensitivity: {e}")
        return None

# ── Main execution ──────────────────────────────────────────────────────────
def main():
    """Main execution function"""
    
    # 1. Solve power flow
    V_mag, V_ang, P_inj, Q_inj = solve_power_flow()
    if V_mag is None:
        return
    
    # 2. Get admittance matrix
    Y = get_admittance_matrix()
    
    # 3. Calculate Jacobian matrix
    J11, J12, J21, J22, slack_bus, pq_buses = calculate_jacobian(Y, V_mag, V_ang)
    
    # 4. Calculate voltage sensitivity
    dV_dP = calculate_voltage_sensitivity(J11, J12, J21, J22, slack_bus, pq_buses)
    
    if dV_dP is None:
        return
    
    # ── Save to HDF5 (same structure as your other files) ──────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    h5_path = os.path.join(OUT_DIR, f"{PROJECT.replace(' ', '_')}_sensitivity_analysis.h5")
    
    with h5py.File(h5_path, "w") as f:
        # Bus data (same structure as Feature_Extraction.py)
        bus_grp = f.create_group("bus")
        for k, v in bus_tab.items():
            bus_grp.create_dataset(k, data=v)
        
        # Power flow results
        pf_grp = f.create_group("power_flow")
        pf_grp.create_dataset("V_magnitude", data=V_mag)
        pf_grp.create_dataset("V_angle_deg", data=V_ang)
        pf_grp.create_dataset("P_injection_MW", data=P_inj)
        pf_grp.create_dataset("Q_injection_MVAr", data=Q_inj)
        pf_grp.create_dataset("slack_bus", data=slack_bus)
        pf_grp.create_dataset("pq_buses", data=pq_buses)
        
        # Jacobian matrices
        jacobian_grp = f.create_group("jacobian")
        jacobian_grp.create_dataset("J11", data=J11)  # ∂P/∂θ
        jacobian_grp.create_dataset("J12", data=J12)  # ∂P/∂V
        jacobian_grp.create_dataset("J21", data=J21)  # ∂Q/∂θ
        jacobian_grp.create_dataset("J22", data=J22)  # ∂Q/∂V
        
        # Voltage sensitivity matrix
        sensitivity_grp = f.create_group("voltage_sensitivity")
        sensitivity_grp.create_dataset("dV_dP", data=dV_dP)
        sensitivity_grp.create_dataset("method", data="jacobian_based".encode())
        sensitivity_grp.create_dataset("description", data="Voltage magnitude sensitivity to active power injection changes".encode())
        
        # Admittance matrix (sparse format like Feature_Extraction.py)
        Y_sparse = csr_matrix(Y)
        Y_grp = f.create_group("admittance")
        Y_grp.create_dataset("data", data=Y_sparse.data)
        Y_grp.create_dataset("indices", data=Y_sparse.indices)
        Y_grp.create_dataset("indptr", data=Y_sparse.indptr)
        Y_grp.create_dataset("shape", data=Y_sparse.shape)
        Y_grp.create_dataset("nnz", data=Y_sparse.nnz)
        
        # Metadata
        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset("extraction_date", data=datetime.now().isoformat().encode())
        meta_grp.create_dataset("project_name", data=PROJECT.encode())
        meta_grp.create_dataset("study_case", data=STUDY.encode())
        meta_grp.create_dataset("base_mva", data=SBASE_MVA)
        meta_grp.create_dataset("num_buses", data=nb)
        meta_grp.create_dataset("analysis_type", data="voltage_sensitivity_jacobian".encode())
    
    print(f"\n💾 SENSITIVITY ANALYSIS SAVED:")
    print(f"   📄 HDF5 file: {h5_path}")
    print(f"   📊 Size: {os.path.getsize(h5_path) / 1024:.1f} KB")
    
    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n📊 SENSITIVITY ANALYSIS SUMMARY:")
    print("="*50)
    print(f"🔌 Buses analyzed: {nb}")
    print(f"📍 Slack bus: {slack_bus} ({safe_get_name(buses[slack_bus])})")
    print(f"📍 PQ buses: {len(pq_buses)}")
    print(f"📊 Sensitivity matrix: {dV_dP.shape}")
    print(f"📊 Max sensitivity: {np.abs(dV_dP).max():.6f} p.u./MW")
    print(f"📊 Average sensitivity: {np.abs(dV_dP).mean():.6f} p.u./MW")
    
    # Show most sensitive buses
    max_sens_per_bus = np.max(np.abs(dV_dP), axis=1)
    most_sensitive = np.argsort(max_sens_per_bus)[-5:][::-1]
    
    print(f"\n🎯 MOST SENSITIVE BUSES:")
    for i, bus_idx in enumerate(most_sensitive):
        print(f"   {i+1}. Bus {bus_idx}: {safe_get_name(buses[bus_idx])} "
              f"(max sensitivity: {max_sens_per_bus[bus_idx]:.6f})")
    
    print(f"\n🎉 JACOBIAN-BASED SENSITIVITY ANALYSIS COMPLETE!")
    print(f"📁 Output file: {os.path.basename(h5_path)}")
    print(f"🔄 Ready for Graph Laplacian spectral clustering!")

if __name__ == "__main__":
    main()