# --- Fast CSV export via ComRes --------------------------------------------
import os, sys, time
from datetime import datetime
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

app = pf.GetApplication()
if app is None:
    raise SystemExit("❌ Could not connect to PowerFactory.")
if hasattr(app, "ResetCalculation"):
    app.ResetCalculation()

# ---------------------------------------------------------------------------
PROJECT, STUDY = "39 Bus New England System", "RMS_Simulation"
if app.ActivateProject(PROJECT):
    raise SystemExit(f"❌ project '{PROJECT}' not found")
study = next((c for c in app.GetProjectFolder("study")
              .GetContents("*.IntCase") if c.loc_name == STUDY), None)
if not study:
    raise SystemExit(f"❌ study case '{STUDY}' not found")
study.Activate()

comInc = app.GetFromStudyCase("ComInc")
comSim = app.GetFromStudyCase("ComSim")
if not (comInc and comSim):
    raise SystemExit("❌ ComInc/ComSim missing")

# --- identical fixed-step settings for BOTH commands -----------------------
STEP = 0.1       # this value is milisecond
for cmd in (comInc, comSim):
    cmd.iopt_adapt = 0          # fixed step
    cmd.dtgrd      = STEP       # solver step
    cmd.dtout      = STEP       # output sampling step
comInc.iopt_sim, comInc.iopt_show, comInc.start = "rms", 0, -0.1
comSim.tstop = 5

# --- prepare result file ---------------------------------------------------
comInc.Execute()
elmRes = comInc.p_resvar
elmRes.Clear()
for bus in app.GetCalcRelevantObjects("*.ElmTerm"):
    elmRes.AddVariable(bus, "m:u")
elmRes.InitialiseWriting()
comSim.p_resvar = elmRes


# --- optional: simple load-change event ------------------------------------
evt_folder = app.GetFromStudyCase("IntEvt")
if evt_folder:
    for e in evt_folder.GetContents(): e.Delete()
    load = app.GetCalcRelevantObjects("*.ElmLod")[0]
    evt = evt_folder.CreateObject("EvtLod", "Load_Change_Test")
    evt.p_target, evt.time, evt.dP, evt.dQ = load, 1.0, 0.1, 0.05

# --- run simulation --------------------------------------------------------
print("🔄 Running RMS simulation …")
t0 = time.time(); app.EchoOff()
comInc.Execute(); app.EchoOn()
comSim.Execute()
print(f"✅ Simulation finished in {time.time()-t0:.2f} s")

# --- one-shot CSV export via ComRes ----------------------------------------
out_dir = os.path.join(os.getcwd(), "data"); os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(
    out_dir, f"rms_bus_voltages_{datetime.now():%Y%m%d_%H%M%S}.csv")

comRes = app.GetFromStudyCase("ComRes") or study.CreateObject("ComRes", "CSVExport")
comRes.pResult   = elmRes          # result file to export
comRes.iopt_exp  = 6               # 6 = CSV, 4 = TXT, etc.
comRes.f_name    = csv_path        # target path
comRes.iopt_sep  = 1               # 1 = system separator, 0 = semicolon
comRes.iopt_honly= 0               # 0 = data + header
comRes.iopt_csel = 0               # 0 = all variables, 1 = selected only
comRes.Execute()                   # <<< fast disk dump

elmRes.Load()  
rows, cols = elmRes.GetNumberOfRows(), elmRes.GetNumberOfColumns()
print(f"🎉 {rows} rows × {cols} columns written to {csv_path}")
