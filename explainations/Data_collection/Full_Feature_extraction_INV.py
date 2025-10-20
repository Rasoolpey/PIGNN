# microgrid_inverter_extraction.py - 2025-07-20
"""
Advanced microgrid extraction focusing on inverter-based resources and embedded controllers.
Extracts control parameters directly from power electronic devices.
"""

import sys, os, h5py, numpy as np, yaml
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf
from datetime import datetime

# ── Project settings ────────────────────────────────────────────────────────
PROJECT = "39 Bus New England System"   # Change this to your microgrid project name "14 Bus System" Or "MV Microgrid T" or "39 Bus New England System"
STUDY = "RMS_Simulation"
SBASE_MVA = 100.0
OUT_DIR = os.path.join(os.getcwd(), "microgrid_out")
µS_to_S = 1e-6

# ── Connect to PowerFactory ─────────────────────────────────────────────────
app = pf.GetApplication() or sys.exit("PF not running")
if hasattr(app, "ResetCalculation"):
    app.ResetCalculation()
assert app.ActivateProject(PROJECT) == 0, "Project not found"
study = next(c for c in app.GetProjectFolder("study").GetContents("*.IntCase")
             if c.loc_name == STUDY)
study.Activate()

print(f"🔧 MICROGRID INVERTER CONTROL EXTRACTION")
print("="*60)
print(f"✅ {PROJECT} | {STUDY}")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Robust helper functions
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

def get_all_attributes(obj):
    """Get all available attributes of an object"""
    attributes = {}
    try:
        if hasattr(obj, 'GetAttributeNames'):
            attr_names = obj.GetAttributeNames()
            for attr in attr_names:
                try:
                    val = get(obj, attr)
                    if isinstance(val, (int, float, str)) and not (isinstance(val, float) and np.isnan(val)):
                        attributes[attr] = val
                except:
                    continue
    except:
        pass
    return attributes

# ── Enhanced project structure detection ────────────────────────────────────
project = app.GetActiveProject()
network_model = project.GetContents("Network Model.IntPrjfolder")[0]
network_data = network_model.GetContents("Network Data.IntPrjfolder")[0]

# Robust grid detection for microgrids
grid = None
grid_candidates = [
    "Grid.ElmNet", "*.ElmNet", "Microgrid.ElmNet", "MV Grid.ElmNet", 
    "LV Grid.ElmNet", "Grid_1.ElmNet", "Main Grid.ElmNet"
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
    except:
        print(f"❌ No ElmNet objects found, using network_data directly")
        all_elements = network_data.GetContents()

if grid:
    all_elements = grid.GetContents()
    print(f"📊 Found {len(all_elements)} elements in {grid.loc_name}")

# ── Comprehensive element classification for microgrids ─────────────────────
print(f"\n🔍 ANALYZING ALL ELEMENT TYPES...")

# Group elements by class
element_classes = {}
for elem in all_elements:
    class_name = safe_get_class(elem)
    if class_name not in element_classes:
        element_classes[class_name] = []
    element_classes[class_name].append(elem)

print(f"📊 ELEMENT TYPES FOUND:")
for class_name, elements in element_classes.items():
    print(f"   {class_name}: {len(elements)} objects")

# Standard power system elements
buses = element_classes.get('ElmTerm', [])
gens = element_classes.get('ElmSym', [])
loads = element_classes.get('ElmLod', [])
lines = element_classes.get('ElmLne', [])
transformers = []
for class_name, elements in element_classes.items():
    if class_name.startswith('ElmTr'):
        transformers.extend(elements)

# Power electronic and renewable elements
static_generators = element_classes.get('ElmGenstat', [])
pv_systems = element_classes.get('ElmPvsys', [])
pv_inverters = element_classes.get('ElmPvimod', [])
wind_generators = element_classes.get('ElmWind', [])
batteries = element_classes.get('ElmBattery', []) + element_classes.get('ElmEss', [])
vsc_converters = element_classes.get('ElmVsc', [])
vsi_converters = element_classes.get('ElmVsi', [])

# Find all inverter-like and power electronic devices
inverter_classes = ['ElmGenstat', 'ElmPvimod', 'ElmVsc', 'ElmVsi', 'ElmPvsys', 'ElmWind']
all_inverters = []
for class_name in inverter_classes:
    if class_name in element_classes:
        all_inverters.extend(element_classes[class_name])

# Control and composite models
plants = element_classes.get('ElmComp', [])
dsl_models = element_classes.get('ElmDsl', [])
composite_models = element_classes.get('ElmComp', [])

print(f"\n🏗️ MICROGRID COMPONENT INVENTORY:")
print(f"   🔌 Buses: {len(buses)}")
print(f"   🔋 Synchronous Generators: {len(gens)}")
print(f"   ⚡ Static Generators: {len(static_generators)}")
print(f"   🔆 PV Systems: {len(pv_systems)}")
print(f"   🔆 PV Inverters: {len(pv_inverters)}")
print(f"   💨 Wind Generators: {len(wind_generators)}")
print(f"   🔋 Batteries/ESS: {len(batteries)}")
print(f"   🔄 VSC Converters: {len(vsc_converters)}")
print(f"   🔄 VSI Converters: {len(vsi_converters)}")
print(f"   📍 Loads: {len(loads)}")
print(f"   📏 Lines: {len(lines)}")
print(f"   🔄 Transformers: {len(transformers)}")
print(f"   🏭 Composite Models: {len(plants)}")
print(f"   🎮 DSL Models: {len(dsl_models)}")
print(f"   🔄 All Inverter Devices: {len(all_inverters)}")

# ── Enhanced control parameter search for composite models ──────────────────
def find_associated_controls(device_obj):
    """Find control models associated with a device through various relationships"""
    
    associated_controls = []
    device_name = safe_get_name(device_obj)
    
    # Method 1: Check for composite frame (ElmComp) relationships
    try:
        # Look for composite models that might contain this device
        for comp in plants:  # ElmComp objects
            comp_name = safe_get_name(comp)
            
            # Check if device name is related to composite name
            if any(part in comp_name.lower() for part in device_name.lower().split()) or \
               any(part in device_name.lower() for part in comp_name.lower().split()):
                
                print(f"   🔗 Found related composite: {comp_name}")
                
                # Extract all elements from the composite
                try:
                    if has(comp, 'pelm') and comp.pelm:
                        pelm_elements = comp.pelm if isinstance(comp.pelm, list) else [comp.pelm]
                        
                        for elem in pelm_elements:
                            if elem and safe_get_class(elem) == 'ElmDsl':
                                ctrl_data = {
                                    'source': 'composite_pelm',
                                    'composite_name': comp_name,
                                    'name': safe_get_name(elem),
                                    'class': safe_get_class(elem),
                                    'parameters': get_all_attributes(elem)
                                }
                                associated_controls.append(ctrl_data)
                                print(f"     • Found DSL: {ctrl_data['name']} ({len(ctrl_data['parameters'])} params)")
                except Exception as e:
                    print(f"     ❌ Error accessing pelm: {e}")
    
    except Exception as e:
        print(f"   ❌ Error in composite search: {e}")
    
    # Method 2: Search for DSL models with similar names
    for dsl in dsl_models:
        dsl_name = safe_get_name(dsl)
        
        # Check for name similarity or number matching
        if any(part in dsl_name.lower() for part in device_name.lower().split()) or \
           any(part in device_name.lower() for part in dsl_name.lower().split()):
            
            ctrl_data = {
                'source': 'name_matching',
                'name': dsl_name,
                'class': safe_get_class(dsl),
                'parameters': get_all_attributes(dsl)
            }
            associated_controls.append(ctrl_data)
            print(f"   🔗 Found related DSL: {dsl_name} ({len(ctrl_data['parameters'])} params)")
    
    # Method 3: Check for frame references or parent objects
    try:
        if has(device_obj, 'pFrame') and device_obj.pFrame:
            frame = device_obj.pFrame
            frame_name = safe_get_name(frame)
            print(f"   🔗 Device has frame: {frame_name}")
            
            # Get all frame contents
            if hasattr(frame, 'GetContents'):
                frame_contents = frame.GetContents()
                for content in frame_contents:
                    if safe_get_class(content) == 'ElmDsl':
                        ctrl_data = {
                            'source': 'frame_contents',
                            'frame_name': frame_name,
                            'name': safe_get_name(content),
                            'class': safe_get_class(content),
                            'parameters': get_all_attributes(content)
                        }
                        associated_controls.append(ctrl_data)
                        print(f"     • Frame DSL: {ctrl_data['name']} ({len(ctrl_data['parameters'])} params)")
    
    except Exception as e:
        print(f"   ❌ Error in frame search: {e}")
    
    return associated_controls

# ── Enhanced inverter control parameter extraction ─────────────────────────
def extract_enhanced_inverter_controls(inverter_obj):
    """Extract detailed control parameters including associated DSL models"""
    
    if not inverter_obj:
        return {}
    
    inv_data = {
        'name': safe_get_name(inverter_obj),
        'class': safe_get_class(inverter_obj),
        'type': 'Unknown',
        'control_parameters': {},
        'operational_parameters': {},
        'protection_parameters': {},
        'associated_controls': [],
        'all_attributes': {}
    }
    
    # Get all attributes first
    all_attrs = get_all_attributes(inverter_obj)
    inv_data['all_attributes'] = all_attrs
    
    class_name = inv_data['class']
    
    # Extract basic device parameters (same as before)
    if class_name == 'ElmGenstat':
        inv_data['type'] = 'Static_Generator'
        
        # Control parameters (look for control-related attributes)
        control_params = [attr for attr in all_attrs.keys() 
                         if any(pattern in attr.lower() for pattern in ['kp', 'ki', 'kd', 'gain', 'ctrl', 'droop'])]
        
        # Operational parameters
        operational_params = [
            'sgn', 'cosn', 'Pmax_uc', 'Pmin_uc', 'Qmax_uc', 'Qmin_uc',
            'pgini', 'qgini', 'uset', 'nphase'
        ]
        
        # Protection parameters
        protection_params = [
            'fmax', 'fmin', 'Imax', 'Inom', 'Umax', 'Umin'
        ]
    
    # Extract parameter values
    for param_list, param_dict in [
        (control_params if 'control_params' in locals() else [], inv_data['control_parameters']),
        (operational_params if 'operational_params' in locals() else [], inv_data['operational_parameters']),
        (protection_params if 'protection_params' in locals() else [], inv_data['protection_parameters'])
    ]:
        for param in param_list:
            val = get(inverter_obj, param)
            if not (isinstance(val, float) and np.isnan(val)):
                param_dict[param] = val
    
    # Find associated control models
    print(f"   🔍 Searching for associated controls...")
    inv_data['associated_controls'] = find_associated_controls(inverter_obj)
    
    return inv_data

# ── Extract all inverter control data ──────────────────────────────────────
print(f"\n🔍 EXTRACTING INVERTER CONTROL PARAMETERS...")

all_inverter_data = []
for i, inverter in enumerate(all_inverters):
    print(f"\n🔄 Analyzing {safe_get_name(inverter)} ({safe_get_class(inverter)}):")
    
    inv_data = extract_enhanced_inverter_controls(inverter)
    all_inverter_data.append(inv_data)
    
    print(f"   Type: {inv_data['type']}")
    print(f"   Control parameters: {len(inv_data['control_parameters'])}")
    print(f"   Operational parameters: {len(inv_data['operational_parameters'])}")
    print(f"   Protection parameters: {len(inv_data['protection_parameters'])}")
    print(f"   Associated controls: {len(inv_data['associated_controls'])}")
    print(f"   Total attributes: {len(inv_data['all_attributes'])}")
    
    # Show first few control parameters
    if inv_data['control_parameters']:
        print(f"   Sample control params:")
        for j, (param, value) in enumerate(inv_data['control_parameters'].items()):
            if j < 3:
                print(f"     {param}: {value}")
            elif j == 3:
                print(f"     ... and {len(inv_data['control_parameters'])-3} more")
                break
    
    # Show associated control information
    for j, ctrl in enumerate(inv_data['associated_controls']):
        print(f"   Associated control {j+1}: {ctrl['name']} ({ctrl['source']})")
        if ctrl['parameters']:
            print(f"     Parameters: {len(ctrl['parameters'])}")
            # Show first few parameters
            for k, (param, value) in enumerate(ctrl['parameters'].items()):
                if k < 2:
                    print(f"       {param}: {value}")
                elif k == 2:
                    print(f"       ... and {len(ctrl['parameters'])-2} more")
                    break

# ── Traditional DSL control extraction (if any) ────────────────────────────
traditional_controls = []
for dsl in dsl_models:
    ctrl_data = {
        'name': safe_get_name(dsl),
        'class': safe_get_class(dsl),
        'type': 'DSL_Model',
        'parameters': get_all_attributes(dsl)
    }
    traditional_controls.append(ctrl_data)

print(f"\n🎛️ TRADITIONAL DSL CONTROLS: {len(traditional_controls)}")

# ── Create comprehensive microgrid overview ────────────────────────────────
microgrid_overview = {
    'metadata': {
        'extraction_date': datetime.now().isoformat(),
        'project_name': PROJECT,
        'study_case': STUDY,
        'base_mva': SBASE_MVA,
        'system_type': 'microgrid_with_inverters',
        'description': f'{PROJECT} - Advanced Microgrid with Inverter Control Extraction'
    },
    
    'network_statistics': {
        'buses': len(buses),
        'synchronous_generators': len(gens),
        'static_generators': len(static_generators),
        'pv_systems': len(pv_systems),
        'pv_inverters': len(pv_inverters),
        'wind_generators': len(wind_generators),
        'batteries': len(batteries),
        'vsc_converters': len(vsc_converters),
        'vsi_converters': len(vsi_converters),
        'total_inverters': len(all_inverters),
        'loads': len(loads),
        'lines': len(lines),
        'transformers': len(transformers),
        'dsl_models': len(dsl_models)
    },
    
    'inverter_control_summary': {
        'total_inverter_devices': len(all_inverters),
        'devices_with_control_params': len([inv for inv in all_inverter_data if inv['control_parameters']]),
        'devices_with_associated_controls': len([inv for inv in all_inverter_data if inv['associated_controls']]),
        'total_control_parameters': sum(len(inv['control_parameters']) for inv in all_inverter_data),
        'total_operational_parameters': sum(len(inv['operational_parameters']) for inv in all_inverter_data),
        'total_protection_parameters': sum(len(inv['protection_parameters']) for inv in all_inverter_data),
        'total_associated_controls': sum(len(inv['associated_controls']) for inv in all_inverter_data),
    },
    
    'inverter_types': {},
    'detailed_inverter_data': all_inverter_data,
    'traditional_controls': traditional_controls
}

# Analyze inverter types
for inv_data in all_inverter_data:
    inv_type = inv_data['type']
    if inv_type not in microgrid_overview['inverter_types']:
        microgrid_overview['inverter_types'][inv_type] = 0
    microgrid_overview['inverter_types'][inv_type] += 1

# ── Save comprehensive data ────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

# Save YAML overview
yaml_path = os.path.join(OUT_DIR, f"{PROJECT.replace(' ', '_')}_microgrid_analysis.yml")
with open(yaml_path, 'w') as f:
    yaml.dump(microgrid_overview, f, default_flow_style=False, sort_keys=False, indent=2)

# Save detailed HDF5 data
h5_path = os.path.join(OUT_DIR, f"{PROJECT.replace(' ', '_')}_microgrid_detailed.h5")

with h5py.File(h5_path, "w") as f:
    # Metadata
    meta_grp = f.create_group("metadata")
    meta_grp.create_dataset("extraction_date", data=datetime.now().isoformat().encode())
    meta_grp.create_dataset("system_type", data="microgrid_with_inverters".encode())
    meta_grp.create_dataset("total_inverters", data=len(all_inverters))
    
    # Inverter data
    inv_grp = f.create_group("inverters")
    inv_grp.create_dataset("num_inverters", data=len(all_inverter_data))
    
    for i, inv_data in enumerate(all_inverter_data):
        inv_subgrp = inv_grp.create_group(f"inverter_{i}")
        inv_subgrp.create_dataset("name", data=inv_data['name'].encode())
        inv_subgrp.create_dataset("class", data=inv_data['class'].encode())
        inv_subgrp.create_dataset("type", data=inv_data['type'].encode())
        
        # Control parameters
        if inv_data['control_parameters']:
            ctrl_grp = inv_subgrp.create_group("control_parameters")
            for param_name, param_value in inv_data['control_parameters'].items():
                try:
                    ctrl_grp.create_dataset(param_name, data=param_value)
                except:
                    ctrl_grp.create_dataset(param_name, data=str(param_value).encode())
        
        # Operational parameters
        if inv_data['operational_parameters']:
            op_grp = inv_subgrp.create_group("operational_parameters")
            for param_name, param_value in inv_data['operational_parameters'].items():
                try:
                    op_grp.create_dataset(param_name, data=param_value)
                except:
                    op_grp.create_dataset(param_name, data=str(param_value).encode())
        
        # Associated control data
        if inv_data['associated_controls']:
            assoc_grp = inv_subgrp.create_group("associated_controls")
            assoc_grp.create_dataset("num_controls", data=len(inv_data['associated_controls']))
            
            for j, ctrl in enumerate(inv_data['associated_controls']):
                ctrl_subgrp = assoc_grp.create_group(f"control_{j}")
                ctrl_subgrp.create_dataset("name", data=ctrl['name'].encode())
                ctrl_subgrp.create_dataset("source", data=ctrl['source'].encode())
                
                if ctrl['parameters']:
                    param_grp = ctrl_subgrp.create_group("parameters")
                    for param_name, param_value in ctrl['parameters'].items():
                        try:
                            param_grp.create_dataset(param_name, data=param_value)
                        except:
                            param_grp.create_dataset(param_name, data=str(param_value).encode())

print(f"\n💾 MICROGRID DATA SAVED:")
print(f"   📄 YAML analysis: {yaml_path}")
print(f"   📊 Size: {os.path.getsize(yaml_path) / 1024:.1f} KB")
print(f"   📄 HDF5 detailed data: {h5_path}")
print(f"   📊 Size: {os.path.getsize(h5_path) / 1024:.1f} KB")

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n📊 MICROGRID EXTRACTION SUMMARY:")
print("="*60)
print(f"🔄 Total Inverter Devices: {len(all_inverters)}")
print(f"🎛️ Devices with Control Parameters: {len([inv for inv in all_inverter_data if inv['control_parameters']])}")
print(f"🔗 Devices with Associated Controls: {len([inv for inv in all_inverter_data if inv['associated_controls']])}")
print(f"⚙️ Total Control Parameters Extracted: {sum(len(inv['control_parameters']) for inv in all_inverter_data)}")
print(f"🔧 Total Operational Parameters: {sum(len(inv['operational_parameters']) for inv in all_inverter_data)}")
print(f"🛡️ Total Protection Parameters: {sum(len(inv['protection_parameters']) for inv in all_inverter_data)}")
print(f"🔗 Total Associated Controls: {sum(len(inv['associated_controls']) for inv in all_inverter_data)}")

print(f"\n🔄 INVERTER TYPES FOUND:")
for inv_type, count in microgrid_overview['inverter_types'].items():
    print(f"   {inv_type}: {count} devices")

print(f"\n🎛️ TRADITIONAL DSL CONTROLS: {len(traditional_controls)}")

print(f"\n🎉 MICROGRID INVERTER EXTRACTION COMPLETE!")
print(f"📁 Output directory: {OUT_DIR}")
print(f"🚀 Ready for advanced microgrid control analysis!")