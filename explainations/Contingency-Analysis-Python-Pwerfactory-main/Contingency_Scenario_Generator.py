# contingency_scenario_generator.py - 2025-08-02
"""
Contingency Scenario Generator for IEEE 39-Bus System
Following the same structure as Feature_Extraction.py and First_model.py

Generates N-1 and N-2 contingency scenarios with intelligent filtering:
- N-1: All single line/transformer/generator outages
- N-2: All combinations EXCEPT two generators simultaneously
- Configurable: All scenarios OR specific number with random selection
- Always includes Scenario 0 as base case (no outages)

Output: CSV file with all contingency definitions
"""

import sys, os, csv, numpy as np
from datetime import datetime
import itertools
import random

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

# ── Project settings ────────────────────────────────────────────────────────
PROJECT = "39 Bus New England System"
STUDY = "RMS_Simulation"
OUT_DIR = os.path.join(os.getcwd(), "contingency_out")

# ── Scenario Configuration ──────────────────────────────────────────────────

# Option 1: Generate specific number of scenarios (comment out Option 2 if using this)
GENERATE_LIMITED_SCENARIOS = True
MAX_SCENARIOS = 200 # Total scenarios including base case (so 149 contingencies)
N1_PERCENTAGE = 0.30  # 30% N-1, 70% N-2

# Option 2: Generate ALL possible scenarios (comment out Option 1 if using this)
# GENERATE_LIMITED_SCENARIOS = False
# MAX_SCENARIOS = None  # Will be ignored
# N1_PERCENTAGE = None  # Will be ignored

# ── Connect to PowerFactory ─────────────────────────────────────────────────
app = pf.GetApplication() or sys.exit("PF not running")
if hasattr(app, "ResetCalculation"):
    app.ResetCalculation()
assert app.ActivateProject(PROJECT) == 0, "Project not found"
study = next(c for c in app.GetProjectFolder("study").GetContents("*.IntCase")
             if c.loc_name == STUDY)
study.Activate()

print(f"🎯 CONTINGENCY SCENARIO GENERATOR")
print("="*60)
print(f"✅ {PROJECT} | {STUDY}")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if GENERATE_LIMITED_SCENARIOS:
    print(f"🎲 Mode: LIMITED SCENARIOS")
    print(f"   📊 Total scenarios: {MAX_SCENARIOS} (including base case)")
    print(f"   📊 N-1 percentage: {N1_PERCENTAGE*100:.0f}%")
    print(f"   📊 N-2 percentage: {(1-N1_PERCENTAGE)*100:.0f}%")
else:
    print(f"🌐 Mode: ALL POSSIBLE SCENARIOS")

print()

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

# ── Extract system elements ─────────────────────────────────────────────────
def extract_system_elements():
    """Extract all system elements for contingency analysis"""
    
    print(f"🔍 EXTRACTING SYSTEM ELEMENTS...")
    
    # Get all network elements (same approach as Feature_Extraction.py)
    lines = app.GetCalcRelevantObjects("*.ElmLne")
    transformers = app.GetCalcRelevantObjects("*.ElmTr2,*.ElmXfr,*.ElmXfr3") 
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    
    print(f"   🔌 Lines: {len(lines)}")
    print(f"   🔄 Transformers: {len(transformers)}")
    print(f"   🔋 Generators: {len(generators)}")
    print(f"   🏠 Loads: {len(loads)}")
    
    # Create element dictionaries for easy access
    elements = {
        'lines': [],
        'transformers': [],
        'generators': [],
        'loads': []
    }
    
    # Process lines
    for i, line in enumerate(lines):
        elements['lines'].append({
            'index': i,
            'object': line,
            'name': safe_get_name(line),
            'class': safe_get_class(line),
            'type': 'line',
            'rating_mva': get(line, "snom", 0.0),
            'length_km': get(line, "dline", 0.0)
        })
    
    # Process transformers
    for i, trafo in enumerate(transformers):
        elements['transformers'].append({
            'index': i,
            'object': trafo,
            'name': safe_get_name(trafo),
            'class': safe_get_class(trafo),
            'type': 'transformer',
            'rating_mva': get(trafo, "snom", 0.0),
            'tap_ratio': get(trafo, "tratio", 1.0)
        })
    
    # Process generators
    for i, gen in enumerate(generators):
        elements['generators'].append({
            'index': i,
            'object': gen,
            'name': safe_get_name(gen),
            'class': safe_get_class(gen),
            'type': 'generator',
            'capacity_mw': get(gen, "sgn", 0.0),
            'active_power_mw': get(gen, "pgini", 0.0)
        })
    
    # Process loads (for reference, not typically used in contingencies)
    for i, load in enumerate(loads):
        elements['loads'].append({
            'index': i,
            'object': load,
            'name': safe_get_name(load),
            'class': safe_get_class(load),
            'type': 'load',
            'active_power_mw': get(load, "plini", 0.0),
            'reactive_power_mvar': get(load, "qlini", 0.0)
        })
    
    return elements

# ── Generate N-1 contingencies ─────────────────────────────────────────────
def generate_n1_contingencies(elements):
    """Generate all N-1 (single element outage) contingencies"""
    
    n1_scenarios = []
    
    # N-1 Line outages
    for line_data in elements['lines']:
        scenario = {
            'scenario_id': len(n1_scenarios) + 1,  # Will be adjusted later
            'contingency_type': 'N-1',
            'outage_type': 'line',
            'description': f"Line outage: {line_data['name']}",
            'elements_out': [
                {
                    'element_type': 'line',
                    'element_name': line_data['name'],
                    'element_class': line_data['class'],
                    'element_index': line_data['index'],
                    'rating_mva': line_data['rating_mva']
                }
            ],
            'severity': 'medium',  # Can be adjusted based on line importance
        }
        n1_scenarios.append(scenario)
    
    # N-1 Transformer outages
    for trafo_data in elements['transformers']:
        scenario = {
            'scenario_id': len(n1_scenarios) + 1,
            'contingency_type': 'N-1', 
            'outage_type': 'transformer',
            'description': f"Transformer outage: {trafo_data['name']}",
            'elements_out': [
                {
                    'element_type': 'transformer',
                    'element_name': trafo_data['name'],
                    'element_class': trafo_data['class'],
                    'element_index': trafo_data['index'],
                    'rating_mva': trafo_data['rating_mva']
                }
            ],
            'severity': 'high',  # Transformers are usually more critical
        }
        n1_scenarios.append(scenario)
    
    # N-1 Generator outages
    for gen_data in elements['generators']:
        scenario = {
            'scenario_id': len(n1_scenarios) + 1,
            'contingency_type': 'N-1',
            'outage_type': 'generator',
            'description': f"Generator outage: {gen_data['name']}",
            'elements_out': [
                {
                    'element_type': 'generator',
                    'element_name': gen_data['name'],
                    'element_class': gen_data['class'],
                    'element_index': gen_data['index'],
                    'capacity_mw': gen_data['capacity_mw']
                }
            ],
            'severity': 'high' if gen_data['capacity_mw'] > 500 else 'medium',
        }
        n1_scenarios.append(scenario)
    
    print(f"✅ Generated {len(n1_scenarios)} N-1 contingencies:")
    print(f"   🔌 Line outages: {len(elements['lines'])}")
    print(f"   🔄 Transformer outages: {len(elements['transformers'])}")
    print(f"   🔋 Generator outages: {len(elements['generators'])}")
    
    return n1_scenarios

# ── Generate N-2 contingencies ─────────────────────────────────────────────
def generate_n2_contingencies(elements):
    """Generate N-2 contingencies (excluding two generators simultaneously)"""
    
    print(f"\n🔄 GENERATING N-2 CONTINGENCIES...")
    print(f"   ⚠️ Excluding: Two generators simultaneously")
    
    n2_scenarios = []
    
    # All transmission elements (lines + transformers)
    transmission_elements = elements['lines'] + elements['transformers']
    generators = elements['generators']
    
    # N-2: Two transmission elements
    print(f"   🔌 Processing transmission combinations...")
    for i, elem1 in enumerate(transmission_elements):
        for j, elem2 in enumerate(transmission_elements[i+1:], i+1):
            scenario = {
                'scenario_id': len(n2_scenarios) + 1,
                'contingency_type': 'N-2',
                'outage_type': f"{elem1['type']}+{elem2['type']}",
                'description': f"Double outage: {elem1['name']} + {elem2['name']}",
                'elements_out': [
                    {
                        'element_type': elem1['type'],
                        'element_name': elem1['name'],
                        'element_class': elem1['class'],
                        'element_index': elem1['index'],
                        'rating_mva': elem1.get('rating_mva', 0.0)
                    },
                    {
                        'element_type': elem2['type'],
                        'element_name': elem2['name'],
                        'element_class': elem2['class'],
                        'element_index': elem2['index'],
                        'rating_mva': elem2.get('rating_mva', 0.0)
                    }
                ],
                'severity': 'critical',
            }
            n2_scenarios.append(scenario)
    
    # N-2: One transmission element + One generator
    print(f"   ⚡ Processing transmission+generator combinations...")
    for trans_elem in transmission_elements:
        for gen_elem in generators:
            scenario = {
                'scenario_id': len(n2_scenarios) + 1,
                'contingency_type': 'N-2',
                'outage_type': f"{trans_elem['type']}+generator",
                'description': f"Mixed outage: {trans_elem['name']} + {gen_elem['name']}",
                'elements_out': [
                    {
                        'element_type': trans_elem['type'],
                        'element_name': trans_elem['name'],
                        'element_class': trans_elem['class'],
                        'element_index': trans_elem['index'],
                        'rating_mva': trans_elem.get('rating_mva', 0.0)
                    },
                    {
                        'element_type': 'generator',
                        'element_name': gen_elem['name'],
                        'element_class': gen_elem['class'],
                        'element_index': gen_elem['index'],
                        'capacity_mw': gen_elem['capacity_mw']
                    }
                ],
                'severity': 'critical',
            }
            n2_scenarios.append(scenario)
    
    # Note: We deliberately EXCLUDE N-2 generator combinations
    n_transmission = len(transmission_elements)
    n_generators = len(generators)
    
    total_n2_combinations = (n_transmission * (n_transmission - 1)) // 2  # Transmission pairs
    total_n2_combinations += n_transmission * n_generators  # Transmission + Generator
    excluded_gen_combinations = (n_generators * (n_generators - 1)) // 2  # Generator pairs (excluded)
    
    print(f"✅ Generated {len(n2_scenarios)} N-2 contingencies:")
    print(f"   🔌 Transmission pairs: {(n_transmission * (n_transmission - 1)) // 2}")
    print(f"   ⚡ Transmission+Generator: {n_transmission * n_generators}")
    print(f"   🚫 Excluded generator pairs: {excluded_gen_combinations}")
    
    return n2_scenarios

# ── Select scenarios based on configuration ────────────────────────────────
def select_scenarios(n1_scenarios, n2_scenarios):
    """Select scenarios based on configuration (all or limited)"""
    
    if not GENERATE_LIMITED_SCENARIOS:
        # Return all scenarios
        all_scenarios = n1_scenarios + n2_scenarios
        print(f"\n📊 USING ALL SCENARIOS:")
        print(f"   🎯 Total contingencies: {len(all_scenarios)}")
        return all_scenarios
    
    else:
        # Select limited scenarios
        available_contingencies = MAX_SCENARIOS - 1  # Reserve 1 for base case
        n1_count = int(available_contingencies * N1_PERCENTAGE)
        n2_count = available_contingencies - n1_count
        
        print(f"\n🎲 SELECTING LIMITED SCENARIOS:")
        print(f"   📊 Available contingencies: {available_contingencies}")
        print(f"   📊 N-1 scenarios needed: {n1_count}")
        print(f"   📊 N-2 scenarios needed: {n2_count}")
        
        # Randomly select N-1 scenarios
        if n1_count > len(n1_scenarios):
            print(f"   ⚠️ Warning: Requested {n1_count} N-1 scenarios, only {len(n1_scenarios)} available")
            selected_n1 = n1_scenarios.copy()
        else:
            selected_n1 = random.sample(n1_scenarios, n1_count)
        
        # Randomly select N-2 scenarios
        if n2_count > len(n2_scenarios):
            print(f"   ⚠️ Warning: Requested {n2_count} N-2 scenarios, only {len(n2_scenarios)} available")
            selected_n2 = n2_scenarios.copy()
        else:
            selected_n2 = random.sample(n2_scenarios, n2_count)
        
        selected_scenarios = selected_n1 + selected_n2
        
        print(f"   ✅ Selected {len(selected_n1)} N-1 scenarios")
        print(f"   ✅ Selected {len(selected_n2)} N-2 scenarios")
        print(f"   🎯 Total selected: {len(selected_scenarios)}")
        
        return selected_scenarios

# ── Save scenarios to CSV ──────────────────────────────────────────────────
def save_scenarios_to_csv(scenarios, elements):
    """Save all scenarios to CSV file"""
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"contingency_scenarios_{timestamp}.csv"
    csv_path = os.path.join(OUT_DIR, csv_filename)
    
    print(f"\n💾 SAVING SCENARIOS TO CSV:")
    print(f"   📄 File: {csv_filename}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'scenario_id',
            'contingency_type', 
            'outage_type',
            'description',
            'severity',
            'num_elements_out',
            'element1_type',
            'element1_name', 
            'element1_class',
            'element1_index',
            'element1_rating_capacity',
            'element2_type',
            'element2_name',
            'element2_class', 
            'element2_index',
            'element2_rating_capacity',
            'generation_date',
            'notes'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write base case (Scenario 0)
        writer.writerow({
            'scenario_id': 0,
            'contingency_type': 'BASE',
            'outage_type': 'none',
            'description': 'Base case - no outages',
            'severity': 'none',
            'num_elements_out': 0,
            'element1_type': 'none',
            'element1_name': 'none',
            'element1_class': 'none', 
            'element1_index': -1,
            'element1_rating_capacity': 0.0,
            'element2_type': 'none',
            'element2_name': 'none',
            'element2_class': 'none',
            'element2_index': -1,
            'element2_rating_capacity': 0.0,
            'generation_date': datetime.now().isoformat(),
            'notes': 'Reference case with all elements in service'
        })
        
        # Write contingency scenarios
        for i, scenario in enumerate(scenarios, 1):
            # Adjust scenario ID to account for base case
            scenario['scenario_id'] = i
            
            elements_out = scenario['elements_out']
            num_elements = len(elements_out)
            
            # Get first element details
            elem1 = elements_out[0] if num_elements > 0 else {}
            elem2 = elements_out[1] if num_elements > 1 else {}
            
            row_data = {
                'scenario_id': scenario['scenario_id'],
                'contingency_type': scenario['contingency_type'],
                'outage_type': scenario['outage_type'],
                'description': scenario['description'],
                'severity': scenario['severity'],
                'num_elements_out': num_elements,
                'element1_type': elem1.get('element_type', 'none'),
                'element1_name': elem1.get('element_name', 'none'),
                'element1_class': elem1.get('element_class', 'none'),
                'element1_index': elem1.get('element_index', -1),
                'element1_rating_capacity': elem1.get('rating_mva', elem1.get('capacity_mw', 0.0)),
                'element2_type': elem2.get('element_type', 'none'),
                'element2_name': elem2.get('element_name', 'none'),
                'element2_class': elem2.get('element_class', 'none'),
                'element2_index': elem2.get('element_index', -1),
                'element2_rating_capacity': elem2.get('rating_mva', elem2.get('capacity_mw', 0.0)),
                'generation_date': datetime.now().isoformat(),
                'notes': f"Auto-generated {scenario['contingency_type']} contingency"
            }
            
            writer.writerow(row_data)
    
    print(f"   ✅ Saved {len(scenarios) + 1} scenarios (including base case)")
    print(f"   📊 File size: {os.path.getsize(csv_path) / 1024:.1f} KB")
    
    return csv_path

# ── Main execution ─────────────────────────────────────────────────────────
def main():
    """Main execution function"""
    
    # Set random seed for reproducible results (optional)
    random.seed(42)
    
    # Extract system elements
    elements = extract_system_elements()
    
    # Generate N-1 contingencies
    print(f"\n🔄 GENERATING N-1 CONTINGENCIES...")
    n1_scenarios = generate_n1_contingencies(elements)
    
    # Generate N-2 contingencies
    n2_scenarios = generate_n2_contingencies(elements)
    
    # Select scenarios based on configuration
    selected_scenarios = select_scenarios(n1_scenarios, n2_scenarios)
    
    # Save to CSV
    csv_path = save_scenarios_to_csv(selected_scenarios, elements)
    
    # Summary
    print(f"\n🎉 CONTINGENCY SCENARIO GENERATION COMPLETE!")
    print("="*60)
    print(f"📊 SUMMARY:")
    print(f"   🎯 Total scenarios: {len(selected_scenarios) + 1} (including base case)")
    print(f"   🔌 System elements analyzed:")
    print(f"      • Lines: {len(elements['lines'])}")
    print(f"      • Transformers: {len(elements['transformers'])}")
    print(f"      • Generators: {len(elements['generators'])}")
    print(f"      • Loads: {len(elements['loads'])} (for reference)")
    
    n1_count = sum(1 for s in selected_scenarios if s['contingency_type'] == 'N-1')
    n2_count = sum(1 for s in selected_scenarios if s['contingency_type'] == 'N-2')
    
    print(f"   📈 Scenario breakdown:")
    print(f"      • Base case: 1")
    print(f"      • N-1 contingencies: {n1_count}")
    print(f"      • N-2 contingencies: {n2_count}")
    
    if GENERATE_LIMITED_SCENARIOS:
        total_possible_n1 = len(n1_scenarios)
        total_possible_n2 = len(n2_scenarios)
        print(f"   🎲 Selection from:")
        print(f"      • Available N-1: {total_possible_n1}")
        print(f"      • Available N-2: {total_possible_n2}")
        print(f"      • Total possible: {total_possible_n1 + total_possible_n2}")
    
    print(f"\n📁 Output file: {os.path.basename(csv_path)}")
    print(f"📁 Output directory: {OUT_DIR}")
    print(f"\n🚀 Ready for Module 2: Contingency Executor!")

if __name__ == "__main__":
    main()