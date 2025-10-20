# scenario_generator_powerbalance_h5_filtered.py
"""
GÉNÉRATEUR DE SCÉNARIOS AVEC ÉQUILIBRAGE AUTOMATIQUE DE PUISSANCE - VERSION FILTRÉE
=====================================

Basé sur le style de First_model.py et les scripts d'extraction.
Corrige automatiquement l'équilibre Pgen = Pload avant simulation.
Sortie unifiée dans un fichier H5 avec toutes les données critiques.

NOUVELLES FONCTIONNALITÉS:
1. ✅ Correction automatique de l'équilibrage de puissance (UNE SEULE FOIS)
2. ✅ Génération de TOUS les scénarios simples
3. ✅ Génération de TOUS les scénarios doubles AUTORISÉS (filtrage intelligent)
4. 🚫 Exclusion des combinaisons gen/gen, gen/load, load/load
5. ✅ Calcul de sensibilité de tension (méthode numérique proper)
6. ✅ Extraction de la matrice Y d'admittance
7. ✅ Sauvegarde unifiée H5 avec structure standardisée
"""

import sys
import os
import h5py
import numpy as np
import pandas as pd
import time
from datetime import datetime
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# PowerFactory connection (using your exact path)
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

# ── Configuration (adaptable) ──────────────────────────────────────────────
PROJECT = "39 Bus New England System"  # Modifiable
STUDY = "RMS_Simulation"
SBASE_MVA = 100.0
BASE_DIR = os.path.join(os.getcwd(), "scenario_analysis")
H5_OUTPUT_FILE = os.path.join(BASE_DIR, f"{PROJECT.replace(' ', '_')}_scenarios_complete.h5")

# Create directories
os.makedirs(BASE_DIR, exist_ok=True)

print("🚀 GÉNÉRATEUR DE SCÉNARIOS COMPLET AVEC FILTRAGE INTELLIGENT")
print("="*70)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Projet: {PROJECT}")
print(f"💾 Output H5: {H5_OUTPUT_FILE}")
print(f"🔒 Stratégie: Équilibrage initial unique - Pas de rééquilibrage par scénario")
print(f"🚫 Filtrage: Exclusion gen/gen, gen/load, load/load")
print(f"✅ Inclusion: TOUS les autres scénarios possibles")
print()

# ── Helper Functions (from your working scripts) ───────────────────────────
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

# ── PowerFactory Connection (from First_model.py style) ────────────────────
def connect_and_setup():
    """Connect to PowerFactory and setup project"""
    app = pf.GetApplication()
    if not app:
        raise Exception("❌ PowerFactory not running!")
    
    # Reset and connect (like First_model.py)
    if hasattr(app, "ResetCalculation"):
        app.ResetCalculation()
    
    if app.ActivateProject(PROJECT) != 0:
        raise Exception(f"❌ Project '{PROJECT}' not found!")
    
    # Find and activate study case
    study_case = None
    for case in app.GetProjectFolder("study").GetContents("*.IntCase"):
        if case.loc_name == STUDY:
            study_case = case
            break
    
    if not study_case:
        raise Exception(f"❌ Study case '{STUDY}' not found!")
    
    study_case.Activate()
    print(f"✅ Connected: {PROJECT} | {STUDY}")
    return app

# ── SYSTÈME D'ÉQUILIBRAGE DE PUISSANCE INITIAL (CRUCIAL!) ──────────────────
def initialize_and_balance_system(app):
    """
    INITIALISATION UNIQUE: Corrige l'équilibre de puissance UNE SEULE FOIS
    et sauvegarde l'état équilibré pour tous les scénarios suivants.
    """
    print("🔧 INITIALISATION ET ÉQUILIBRAGE DU SYSTÈME")
    print("-" * 50)
    
    # Get all elements
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    
    # Calculate current totals
    total_gen_original = 0
    total_load = 0
    zero_power_gens = []
    active_gens = []
    
    print("📊 État initial du système:")
    for gen in generators:
        if not getattr(gen, 'outserv', 0):  # If in service
            gen_power = get(gen, 'pgini', 0)
            total_gen_original += gen_power
            
            if abs(gen_power) < 1.0:  # Générateur essentiellement à zéro
                zero_power_gens.append(gen)
                print(f"   🔋 {safe_get_name(gen)}: {gen_power:.1f} MW (⚠️  ZÉRO - À CORRIGER)")
            else:
                active_gens.append(gen)
                print(f"   🔋 {safe_get_name(gen)}: {gen_power:.1f} MW")
    
    for load in loads:
        if not getattr(load, 'outserv', 0):  # If in service
            load_power = get(load, 'plini', 0)
            total_load += load_power
    
    print(f"\n📈 Bilan énergétique initial:")
    print(f"   💡 Génération totale: {total_gen_original:.1f} MW")
    print(f"   🏠 Charge totale: {total_load:.1f} MW")
    print(f"   ⚖️  Déséquilibre: {total_gen_original - total_load:.1f} MW")
    print(f"   🔋 Générateurs à zéro: {len(zero_power_gens)}")
    print(f"   ⚡ Générateurs actifs: {len(active_gens)}")
    
    # Si déséquilibre significatif, corriger UNE SEULE FOIS
    imbalance = total_load - total_gen_original
    balanced_state = {}  # Pour sauvegarder l'état équilibré
    
    if abs(imbalance) > 10:  # Plus de 10 MW de déséquilibre
        print(f"\n🔧 CORRECTION REQUISE: {imbalance:.1f} MW")
        
        if len(zero_power_gens) > 0:
            # Stratégie 1: Utiliser les générateurs à zéro en priorité
            adjustment_per_gen = imbalance / len(zero_power_gens)
            print(f"   📈 Attribution aux générateurs à zéro: {adjustment_per_gen:.1f} MW chacun")
            
            for gen in zero_power_gens:
                old_power = get(gen, 'pgini', 0)
                new_power = max(0, adjustment_per_gen)  # Pas de génération négative
                
                try:
                    gen.SetAttribute('pgini', new_power)
                    balanced_state[safe_get_name(gen)] = {
                        'object': gen,
                        'original_power': old_power,
                        'balanced_power': new_power,
                        'was_adjusted': True
                    }
                    print(f"     ✅ {safe_get_name(gen)}: {old_power:.1f} → {new_power:.1f} MW")
                except Exception as e:
                    print(f"     ❌ {safe_get_name(gen)}: erreur - {e}")
            
        elif len(active_gens) > 0:
            # Stratégie 2: Utiliser le plus gros générateur (slack)
            largest_gen = max(active_gens, key=lambda g: get(g, 'pgini', 0))
            old_power = get(largest_gen, 'pgini', 0)
            new_power = old_power + imbalance
            
            if new_power > 0:
                try:
                    largest_gen.SetAttribute('pgini', new_power)
                    balanced_state[safe_get_name(largest_gen)] = {
                        'object': largest_gen,
                        'original_power': old_power,
                        'balanced_power': new_power,
                        'was_adjusted': True
                    }
                    print(f"   ✅ Slack {safe_get_name(largest_gen)}: {old_power:.1f} → {new_power:.1f} MW")
                except Exception as e:
                    print(f"   ❌ Slack adjustment failed: {e}")
            else:
                print(f"   ⚠️ Cannot adjust - would result in negative generation")
        
        # Sauvegarder TOUS les générateurs (même non-modifiés) pour référence
        for gen in generators:
            gen_name = safe_get_name(gen)
            if gen_name not in balanced_state:
                balanced_state[gen_name] = {
                    'object': gen,
                    'original_power': get(gen, 'pgini', 0),
                    'balanced_power': get(gen, 'pgini', 0),
                    'was_adjusted': False
                }
        
        # Vérification finale
        total_gen_final = 0
        for gen in generators:
            if not getattr(gen, 'outserv', 0):
                total_gen_final += get(gen, 'pgini', 0)
        
        final_imbalance = total_gen_final - total_load
        print(f"\n✅ RÉSULTAT DE L'ÉQUILIBRAGE:")
        print(f"   💡 Génération finale: {total_gen_final:.1f} MW")
        print(f"   🏠 Charge totale: {total_load:.1f} MW")
        print(f"   ⚖️  Déséquilibre final: {final_imbalance:.1f} MW")
        
        if abs(final_imbalance) < 50:  # Acceptable
            print(f"   ✅ Système équilibré avec succès!")
            return True, balanced_state
        else:
            print(f"   ⚠️ Déséquilibre encore important!")
            return False, balanced_state
    else:
        # Pas de correction nécessaire, mais sauvegarder l'état actuel
        print(f"✅ Équilibre déjà acceptable (±{imbalance:.1f} MW)")
        for gen in generators:
            gen_name = safe_get_name(gen)
            balanced_state[gen_name] = {
                'object': gen,
                'original_power': get(gen, 'pgini', 0),
                'balanced_power': get(gen, 'pgini', 0),
                'was_adjusted': False
            }
        return True, balanced_state

# ── RESTAURATION RAPIDE DE L'ÉTAT ÉQUILIBRÉ ───────────────────────────────
def restore_balanced_state(balanced_state):
    """
    Restaure rapidement l'état équilibré sauvegardé lors de l'initialisation.
    Utilisé avant chaque scénario pour repartir de l'état équilibré.
    """
    try:
        for gen_name, gen_data in balanced_state.items():
            gen_obj = gen_data['object']
            balanced_power = gen_data['balanced_power']
            
            # Restaurer la puissance équilibrée
            gen_obj.SetAttribute('pgini', balanced_power)
            
            # S'assurer que le générateur est en service (sauf si on va le débrancher)
            if getattr(gen_obj, 'outserv', 0) != 0:
                gen_obj.SetAttribute('outserv', 0)
        
        return True
    except Exception as e:
        print(f"   ⚠️ Erreur lors de la restauration: {e}")
        return False

# ── Data Collection Functions (adapted from your scripts) ──────────────────
def collect_system_data(app):
    """Collect all essential system data for H5 storage"""
    print("📊 Collecte des données système...")
    
    # Get all elements
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    lines = app.GetCalcRelevantObjects("*.ElmLne")
    transformers = app.GetCalcRelevantObjects("*.ElmTr2")
    
    # Bus data
    bus_data = {
        'names': np.array([safe_get_name(bus) for bus in buses], dtype='S24'),
        'voltage_pu': np.array([get(bus, 'm:u', 1.0) for bus in buses]),
        'angle_deg': np.array([get(bus, 'm:phiu', 0.0) for bus in buses]),
        'voltage_kv': np.array([get(bus, 'uknom', 0.0) for bus in buses])
    }
    
    # Generator data (Pgen array)
    Pgen = np.zeros(len(buses))
    Qgen = np.zeros(len(buses))
    gen_bus_mapping = {}
    
    for i, gen in enumerate(generators):
        if not getattr(gen, 'outserv', 0):  # If in service
            # Find connected bus
            try:
                bus = getattr(gen, 'bus1', None)
                if bus:
                    bus_idx = buses.index(bus)
                    gen_power = get(gen, 'pgini', 0)
                    gen_q = get(gen, 'qgini', 0)
                    Pgen[bus_idx] += gen_power
                    Qgen[bus_idx] += gen_q
                    gen_bus_mapping[i] = bus_idx
            except:
                pass
    
    # Load data (Pload array)
    Pload = np.zeros(len(buses))
    Qload = np.zeros(len(buses))
    load_bus_mapping = {}
    
    for i, load in enumerate(loads):
        if not getattr(load, 'outserv', 0):  # If in service
            try:
                bus = getattr(load, 'bus1', None)
                if bus:
                    bus_idx = buses.index(bus)
                    load_power = get(load, 'plini', 0)
                    load_q = get(load, 'qlini', 0)
                    Pload[bus_idx] += load_power
                    Qload[bus_idx] += load_q
                    load_bus_mapping[i] = bus_idx
            except:
                pass
    
    # Edge data for admittance
    edge_data = []
    
    # Process lines
    for line in lines:
        try:
            bus1 = getattr(line, 'bus1', None)
            bus2 = getattr(line, 'bus2', None)
            if bus1 and bus2:
                bus1_idx = buses.index(bus1)
                bus2_idx = buses.index(bus2)
                
                R = get(line, 'R1', 0.001)  # Resistance
                X = get(line, 'X1', 0.001)  # Reactance
                
                edge_data.append({
                    'from_bus': bus1_idx,
                    'to_bus': bus2_idx,
                    'R': R,
                    'X': X,
                    'element_type': 'line',
                    'name': safe_get_name(line)
                })
        except:
            pass
    
    # Process transformers
    for trafo in transformers:
        try:
            bus1 = getattr(trafo, 'bus1', None)
            bus2 = getattr(trafo, 'bus2', None)
            if bus1 and bus2:
                bus1_idx = buses.index(bus1)
                bus2_idx = buses.index(bus2)
                
                # Transformer impedance in pu
                uk = get(trafo, 'uk', 10.0)   # Short circuit voltage %
                ukr = get(trafo, 'ukr', 1.0)  # Resistance %
                
                X_pu = uk / 100.0
                R_pu = ukr / 100.0
                
                edge_data.append({
                    'from_bus': bus1_idx,
                    'to_bus': bus2_idx,
                    'R': R_pu,
                    'X': X_pu,
                    'element_type': 'transformer',
                    'name': safe_get_name(trafo)
                })
        except:
            pass
    
    print(f"   ✅ {len(buses)} buses, {len(edge_data)} branches")
    print(f"   ✅ Pgen total: {Pgen.sum():.1f} MW")
    print(f"   ✅ Pload total: {Pload.sum():.1f} MW")
    
    return {
        'buses': bus_data,
        'Pgen': Pgen,
        'Qgen': Qgen,
        'Pload': Pload,
        'Qload': Qload,
        'edges': edge_data,
        'gen_bus_mapping': gen_bus_mapping,
        'load_bus_mapping': load_bus_mapping
    }

# ── Admittance Matrix Construction ─────────────────────────────────────────
def construct_admittance_matrix(system_data):
    """Construct Y matrix from edge data"""
    print("🔧 Construction de la matrice d'admittance...")
    
    num_buses = len(system_data['buses']['names'])
    Y_matrix = np.zeros((num_buses, num_buses), dtype=complex)
    
    for edge in system_data['edges']:
        i = edge['from_bus']
        j = edge['to_bus']
        R = edge['R']
        X = edge['X']
        
        if R == 0 and X == 0:  # Avoid division by zero
            continue
            
        # Admittance = 1 / (R + jX)
        Y_ij = 1.0 / (R + 1j * X)
        
        # Fill Y matrix
        Y_matrix[i, j] -= Y_ij  # Off-diagonal negative
        Y_matrix[j, i] -= Y_ij  # Symmetric
        Y_matrix[i, i] += Y_ij  # Diagonal positive
        Y_matrix[j, j] += Y_ij  # Diagonal positive
    
    # Convert to sparse for storage
    Y_sparse = csr_matrix(Y_matrix)
    
    print(f"   ✅ Matrice {num_buses}x{num_buses}, {Y_sparse.nnz} éléments non-zéros")
    
    return Y_matrix, Y_sparse

# ── Voltage Sensitivity Calculation (proper numerical method) ──────────────
def calculate_voltage_sensitivity_numerical(app, system_data):
    """
    Calculate dV/dP using numerical differentiation 
    (following the proper method from your sensitivity script)
    """
    print("🔢 Calcul de la sensibilité de tension (méthode numérique)...")
    
    num_buses = len(system_data['buses']['names'])
    dV_dP = np.zeros((num_buses, num_buses))
    success_flags = np.zeros(num_buses, dtype=bool)
    
    delta_P_MW = 10.0  # Perturbation size
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    
    # Store original values
    original_loads = [(load, get(load, 'plini')) for load in loads]
    original_gens = [(gen, get(gen, 'pgini')) for gen in generators]
    
    # Get base case voltages
    if not solve_power_flow(app):
        print("   ❌ Base case power flow failed")
        return dV_dP, success_flags
    
    V_base = np.array([get(bus, 'm:u', 1.0) for bus in buses])
    
    try:
        # Calculate sensitivity for buses with controllable elements
        for bus_idx in range(num_buses):
            # Find controllable elements at this bus
            perturbable_loads = [load for load in loads 
                               if getattr(load, 'bus1', None) == buses[bus_idx] 
                               and not getattr(load, 'outserv', 0)]
            
            perturbable_gens = [gen for gen in generators 
                              if getattr(gen, 'bus1', None) == buses[bus_idx] 
                              and not getattr(gen, 'outserv', 0)]
            
            if not (perturbable_loads or perturbable_gens):
                continue
            
            # Try positive perturbation
            perturbed = False
            
            if perturbable_loads:
                # Increase load
                load = perturbable_loads[0]
                old_P = get(load, 'plini')
                new_P = old_P + delta_P_MW
                try:
                    load.SetAttribute('plini', new_P)
                    perturbed = True
                except:
                    pass
            elif perturbable_gens:
                # Increase generation
                gen = perturbable_gens[0]
                old_P = get(gen, 'pgini')
                new_P = old_P + delta_P_MW
                try:
                    gen.SetAttribute('pgini', new_P)
                    perturbed = True
                except:
                    pass
            
            if perturbed and solve_power_flow(app):
                V_pert = np.array([get(bus, 'm:u', 1.0) for bus in buses])
                dV_dP[:, bus_idx] = (V_pert - V_base) / delta_P_MW
                success_flags[bus_idx] = True
            
            # Restore original values
            for load, orig_P in original_loads:
                try:
                    load.SetAttribute('plini', orig_P)
                except:
                    pass
            for gen, orig_P in original_gens:
                try:
                    gen.SetAttribute('pgini', orig_P)
                except:
                    pass
    
    except Exception as e:
        print(f"   ❌ Sensitivity calculation error: {e}")
    
    successful_buses = np.sum(success_flags)
    print(f"   ✅ Sensibilité calculée pour {successful_buses}/{num_buses} buses")
    
    return dV_dP, success_flags

def solve_power_flow(app):
    """Solve power flow with robust settings"""
    try:
        comLdf = app.GetFromStudyCase("ComLdf")
        comLdf.iopt_net = 0      # AC load flow
        comLdf.iopt_at = 1       # Automatic tap adjustment
        comLdf.errlf = 0.1       # Relaxed tolerance
        comLdf.maxiter = 50      # Maximum iterations
        
        ierr = comLdf.Execute()
        return ierr == 0
    except:
        return False

# ── Scenario Generation (Modified - NO POWER BALANCE INSIDE) ──────────────
def generate_scenario_single_outage(app, element_name, element_type, balanced_state):
    """
    Generate single element outage scenario.
    NOTE: Power balance is NOT recalculated here - we start from balanced_state.
    """
    
    if element_type == "generator":
        elements = app.GetCalcRelevantObjects("*.ElmSym")
    elif element_type == "line":
        elements = app.GetCalcRelevantObjects("*.ElmLne")
    elif element_type == "load":
        elements = app.GetCalcRelevantObjects("*.ElmLod")
    elif element_type == "transformer":
        elements = app.GetCalcRelevantObjects("*.ElmTr2")
        elements.extend(app.GetCalcRelevantObjects("*.ElmTr3"))
    else:
        return None
    
    # Find the element
    target_element = None
    for elem in elements:
        if element_name.lower() in safe_get_name(elem).lower():
            target_element = elem
            break
    
    if not target_element:
        print(f"   ❌ Element '{element_name}' not found")
        return None
    
    # Store original state
    original_outserv = getattr(target_element, 'outserv', 0)
    
    try:
        # STEP 1: Restore balanced state (NO power balance calculation)
        print(f"   🔄 Restoration de l'état équilibré...")
        if not restore_balanced_state(balanced_state):
            print(f"   ❌ Failed to restore balanced state")
            return None
        
        # STEP 2: Create outage
        print(f"   ⚡ Création de la panne {element_type}: {element_name}")
        target_element.SetAttribute('outserv', 1)
        
        # STEP 3: Solve power flow (without power balance recalculation)
        print(f"   🔄 Résolution du load flow...")
        if solve_power_flow(app):
            print(f"   ✅ Load flow convergé")
            
            # Collect data
            scenario_data = collect_system_data(app)
            
            # Calculate Y matrix
            Y_matrix, Y_sparse = construct_admittance_matrix(scenario_data)
            
            # Calculate sensitivity
            dV_dP, success_flags = calculate_voltage_sensitivity_numerical(app, scenario_data)
            
            scenario_result = {
                'scenario_name': f"{element_type}_outage_{element_name}",
                'scenario_type': f"{element_type}_outage",
                'outage_element': element_name,
                'convergence': True,
                'system_data': scenario_data,
                'Y_matrix': Y_matrix,
                'Y_sparse': Y_sparse,
                'sensitivity_dV_dP': dV_dP,
                'sensitivity_success_flags': success_flags,
                'power_balance_info': 'Used pre-balanced state (no recalculation)',
                'balanced_state_applied': True
            }
            
            print(f"   ✅ Scenario '{element_name}' completed successfully")
        else:
            print(f"   ❌ Power flow failed for '{element_name}'")
            scenario_result = None
            
    except Exception as e:
        print(f"   ❌ Error in scenario '{element_name}': {e}")
        scenario_result = None
    finally:
        # Restore original state
        try:
            target_element.SetAttribute('outserv', original_outserv)
        except:
            pass
    
    return scenario_result

# ── COMPLETE Scenario List Generation (TOUS LES ÉLÉMENTS) ─────────────────
def generate_comprehensive_scenario_list(app):
    """Generate COMPLETE list of scenarios based on ALL system elements"""
    print("🔍 Génération COMPLÈTE de la liste de scénarios...")
    
    # Get ALL elements from PowerFactory
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    lines = app.GetCalcRelevantObjects("*.ElmLne")
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    transformers = app.GetCalcRelevantObjects("*.ElmTr2")
    transformers3w = app.GetCalcRelevantObjects("*.ElmTr3")  # 3-winding transformers
    
    scenarios = []
    
    # Generator outages (ALL GENERATORS - NO LIMIT)
    print(f"   🔋 Générateurs trouvés: {len(generators)}")
    for gen in generators:
        if not getattr(gen, 'outserv', 0):  # Only in-service generators
            gen_name = safe_get_name(gen)
            scenarios.append((gen_name, "generator"))
    
    # Line outages (ALL LINES - NO LIMIT)
    print(f"   📏 Lignes trouvées: {len(lines)}")
    for line in lines:
        if not getattr(line, 'outserv', 0):  # Only in-service lines
            line_name = safe_get_name(line)
            scenarios.append((line_name, "line"))
    
    # Load outages (ALL LOADS - NO LIMIT)
    print(f"   📍 Charges trouvées: {len(loads)}")
    for load in loads:
        if not getattr(load, 'outserv', 0):  # Only in-service loads
            load_name = safe_get_name(load)
            scenarios.append((load_name, "load"))
    
    # 2-winding Transformer outages (ALL TRANSFORMERS - NO LIMIT)
    print(f"   🔄 Transformateurs 2W trouvés: {len(transformers)}")
    for trafo in transformers:
        if not getattr(trafo, 'outserv', 0):  # Only in-service transformers
            trafo_name = safe_get_name(trafo)
            scenarios.append((trafo_name, "transformer"))
    
    # 3-winding Transformer outages (ALL 3W TRANSFORMERS - NO LIMIT)
    print(f"   🔄 Transformateurs 3W trouvés: {len(transformers3w)}")
    for trafo3w in transformers3w:
        if not getattr(trafo3w, 'outserv', 0):  # Only in-service 3W transformers
            trafo3w_name = safe_get_name(trafo3w)
            scenarios.append((trafo3w_name, "transformer"))
    
    print(f"   ✅ Total de scénarios SIMPLES générés: {len(scenarios)}")
    print(f"      • Générateurs: {len([s for s in scenarios if s[1] == 'generator'])}")
    print(f"      • Lignes: {len([s for s in scenarios if s[1] == 'line'])}")
    print(f"      • Charges: {len([s for s in scenarios if s[1] == 'load'])}")
    print(f"      • Transformateurs: {len([s for s in scenarios if s[1] == 'transformer'])}")
    
    return scenarios

# ── COMPLETE DOUBLE CONTINGENCY GENERATION (AVEC FILTRAGE INTELLIGENT) ───
def generate_double_contingency_scenarios_complete(app, max_combinations=None):
    """
    Generate TOUS LES double contingency scenarios AUTORISÉS
    EXCLUSIONS: gen/gen, gen/load, load/load combinations
    INCLUSIONS: TOUS les autres (line/line, line/gen, line/load, line/transformer, transformer/*)
    """
    print(f"🔍 Génération COMPLÈTE des scénarios de contingence double (FILTRÉS)...")
    print(f"🚫 EXCLUSIONS: gen/gen, gen/load, load/load")
    print(f"✅ INCLUSIONS: TOUS les autres scénarios possibles") 
    
    # Get ALL elements
    generators = [gen for gen in app.GetCalcRelevantObjects("*.ElmSym") 
                 if not getattr(gen, 'outserv', 0)]
    lines = [line for line in app.GetCalcRelevantObjects("*.ElmLne") 
             if not getattr(line, 'outserv', 0)]
    loads = [load for load in app.GetCalcRelevantObjects("*.ElmLod") 
             if not getattr(load, 'outserv', 0)]
    transformers = [trafo for trafo in app.GetCalcRelevantObjects("*.ElmTr2") 
                   if not getattr(trafo, 'outserv', 0)]
    transformers3w = [trafo for trafo in app.GetCalcRelevantObjects("*.ElmTr3") 
                     if not getattr(trafo, 'outserv', 0)]
    
    # Create categorized lists
    all_elements = []
    
    # Add generators
    for gen in generators:
        all_elements.append((safe_get_name(gen), "generator", gen))
    
    # Add lines  
    for line in lines:
        all_elements.append((safe_get_name(line), "line", line))
    
    # Add loads
    for load in loads:
        all_elements.append((safe_get_name(load), "load", load))
        
    # Add transformers (2W and 3W)
    for trafo in transformers:
        all_elements.append((safe_get_name(trafo), "transformer", trafo))
    for trafo3w in transformers3w:
        all_elements.append((safe_get_name(trafo3w), "transformer", trafo3w))
    
    print(f"   📊 Éléments totaux disponibles: {len(all_elements)}")
    print(f"      • Générateurs: {len(generators)}")
    print(f"      • Lignes: {len(lines)}")
    print(f"      • Charges: {len(loads)}")
    print(f"      • Transformateurs 2W: {len(transformers)}")
    print(f"      • Transformateurs 3W: {len(transformers3w)}")
    
    # FONCTION DE FILTRAGE INTELLIGENT
    def is_combination_allowed(type1, type2):
        """
        Détermine si une combinaison de types d'éléments est autorisée
        RETOURNE FALSE pour les combinaisons à exclure
        """
        # Combinaisons interdites (selon votre demande)
        forbidden_combinations = {
            ("generator", "generator"),  # gen/gen
            ("generator", "load"),       # gen/load  
            ("load", "generator"),       # load/gen (symétrique)
            ("load", "load")             # load/load
        }
        
        return (type1, type2) not in forbidden_combinations
    
    # Generate filtered combinations
    double_scenarios = []
    from itertools import combinations
    
    total_combinations = 0
    filtered_out = 0
    
    print(f"   ⏳ Génération des combinaisons (peut prendre quelques minutes)...")
    
    for (name1, type1, obj1), (name2, type2, obj2) in combinations(all_elements, 2):
        total_combinations += 1
        
        # APPLIQUER LE FILTRE ICI
        if not is_combination_allowed(type1, type2):
            filtered_out += 1
            continue  # Skip cette combinaison
        
        # Si on arrive ici, la combinaison est autorisée
        scenario_name = f"double_{type1}_{name1}_AND_{type2}_{name2}"
        double_scenarios.append({
            'scenario_name': scenario_name,
            'scenario_type': 'double_contingency',
            'outage_elements': [(name1, type1, obj1), (name2, type2, obj2)],
            'description': f'{type1} {name1} + {type2} {name2}'
        })
        
        # Limiter seulement si max_combinations est spécifié
        if max_combinations and len(double_scenarios) >= max_combinations:
            print(f"   ⚠️  Limitation appliquée: {max_combinations} scénarios doubles")
            break
    
    print(f"   📊 Combinaisons totales possibles: {total_combinations}")
    print(f"   🚫 Combinaisons filtrées (exclues): {filtered_out}")
    print(f"   ✅ Scénarios doubles retenus: {len(double_scenarios)}")
    
    # Afficher le détail des combinaisons retenues
    combination_stats = {}
    for scenario in double_scenarios:
        type1 = scenario['outage_elements'][0][1]
        type2 = scenario['outage_elements'][1][1]
        combo_key = f"{type1}/{type2}" if type1 <= type2 else f"{type2}/{type1}"
        combination_stats[combo_key] = combination_stats.get(combo_key, 0) + 1
    
    print(f"   📋 Répartition des combinaisons retenues:")
    for combo, count in sorted(combination_stats.items()):
        print(f"      • {combo}: {count} scénarios")
    
    return double_scenarios

# ── Double Contingency Scenario Generation ────────────────────────────────
def generate_double_contingency_scenario(app, double_scenario_def, balanced_state):
    """
    Generate double contingency scenario (2 simultaneous outages)
    """
    outage_elements = double_scenario_def['outage_elements']
    
    # Store original states
    original_states = []
    for element_name, element_type, element_obj in outage_elements:
        original_states.append((element_obj, getattr(element_obj, 'outserv', 0)))
    
    try:
        # STEP 1: Restore balanced state
        print(f"   🔄 Restoration de l'état équilibré...")
        if not restore_balanced_state(balanced_state):
            print(f"   ❌ Failed to restore balanced state")
            return None
        
        # STEP 2: Create double outage
        print(f"   ⚡⚡ Création de la double panne...")
        for element_name, element_type, element_obj in outage_elements:
            element_obj.SetAttribute('outserv', 1)
            print(f"      • {element_type}: {element_name}")
        
        # STEP 3: Solve power flow
        print(f"   🔄 Résolution du load flow...")
        if solve_power_flow(app):
            print(f"   ✅ Load flow convergé")
            
            # Collect data
            scenario_data = collect_system_data(app)
            
            # Calculate Y matrix
            Y_matrix, Y_sparse = construct_admittance_matrix(scenario_data)
            
            # Calculate sensitivity
            dV_dP, success_flags = calculate_voltage_sensitivity_numerical(app, scenario_data)
            
            scenario_result = {
                'scenario_name': double_scenario_def['scenario_name'],
                'scenario_type': 'double_contingency',
                'outage_elements': [elem[0] for elem in outage_elements],
                'convergence': True,
                'system_data': scenario_data,
                'Y_matrix': Y_matrix,
                'Y_sparse': Y_sparse,
                'sensitivity_dV_dP': dV_dP,
                'sensitivity_success_flags': success_flags,
                'power_balance_info': 'Used pre-balanced state (no recalculation)',
                'description': double_scenario_def['description'],
                'balanced_state_applied': True
            }
            
            print(f"   ✅ Double scenario completed successfully")
            return scenario_result
        else:
            print(f"   ❌ Power flow failed for double scenario")
            return None
            
    except Exception as e:
        print(f"   ❌ Error in double scenario: {e}")
        return None
    finally:
        # Restore original states
        for element_obj, original_outserv in original_states:
            try:
                element_obj.SetAttribute('outserv', original_outserv)
            except:
                pass

# ── H5 Storage Functions ───────────────────────────────────────────────────
def save_scenario_to_h5(h5_file, scenario_name, scenario_data):
    """Save single scenario data to H5 file"""
    
    with h5py.File(h5_file, 'a') as f:
        # Create scenario group
        if scenario_name in f:
            del f[scenario_name]  # Remove if exists
        
        scenario_grp = f.create_group(scenario_name)
        
        # Metadata
        scenario_grp.attrs['scenario_type'] = scenario_data['scenario_type']
        scenario_grp.attrs['convergence'] = scenario_data['convergence']
        scenario_grp.attrs['timestamp'] = datetime.now().isoformat()
        scenario_grp.attrs['balanced_state_applied'] = scenario_data.get('balanced_state_applied', True)
        
        # Handle outage element(s)
        if isinstance(scenario_data['outage_element'], list):
            scenario_grp.attrs['outage_elements'] = str(scenario_data['outage_element'])
        else:
            scenario_grp.attrs['outage_element'] = scenario_data['outage_element']
        
        # System data
        sys_data = scenario_data['system_data']
        
        # Bus data
        bus_grp = scenario_grp.create_group('buses')
        for key, value in sys_data['buses'].items():
            bus_grp.create_dataset(key, data=value)
        
        # Power arrays (CRITICAL FOR YOUR MODEL)
        scenario_grp.create_dataset('Pgen', data=sys_data['Pgen'])
        scenario_grp.create_dataset('Qgen', data=sys_data['Qgen'])
        scenario_grp.create_dataset('Pload', data=sys_data['Pload'])
        scenario_grp.create_dataset('Qload', data=sys_data['Qload'])
        
        # Admittance matrix (CRITICAL FOR YOUR MODEL)
        Y_sparse = scenario_data['Y_sparse']
        admittance_grp = scenario_grp.create_group('admittance')
        admittance_grp.create_dataset('data', data=Y_sparse.data)
        admittance_grp.create_dataset('indices', data=Y_sparse.indices)
        admittance_grp.create_dataset('indptr', data=Y_sparse.indptr)
        admittance_grp.create_dataset('shape', data=Y_sparse.shape)
        admittance_grp.create_dataset('nnz', data=Y_sparse.nnz)
        
        # Voltage sensitivity (CRITICAL FOR YOUR MODEL)
        sensitivity_grp = scenario_grp.create_group('sensitivity')
        sensitivity_grp.create_dataset('dV_dP', data=scenario_data['sensitivity_dV_dP'])
        sensitivity_grp.create_dataset('success_flags', data=scenario_data['sensitivity_success_flags'])
        
        # Edge data for graph reconstruction
        edge_grp = scenario_grp.create_group('edges')
        if sys_data['edges']:
            edge_grp.create_dataset('from_bus', data=[e['from_bus'] for e in sys_data['edges']])
            edge_grp.create_dataset('to_bus', data=[e['to_bus'] for e in sys_data['edges']])
            edge_grp.create_dataset('R', data=[e['R'] for e in sys_data['edges']])
            edge_grp.create_dataset('X', data=[e['X'] for e in sys_data['edges']])

def initialize_h5_file(h5_file):
    """Initialize H5 file with baseline data"""
    print(f"💾 Initialisation du fichier H5: {h5_file}")
    
    with h5py.File(h5_file, 'w') as f:
        # Global metadata
        f.attrs['project'] = PROJECT
        f.attrs['study_case'] = STUDY
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['base_mva'] = SBASE_MVA
        f.attrs['purpose'] = 'Complete scenario analysis with intelligent filtering'
        f.attrs['power_balance_strategy'] = 'single_initial_correction'
        f.attrs['scenario_filtering'] = 'exclude_gen_gen_gen_load_load_load'
        
        # Create main groups
        f.create_group('scenarios')
        f.create_group('baseline')

# ── CHOIX DU MODE DE GÉNÉRATION ──────────────────────────────────────────
def choose_scenario_generation_mode():
    """
    Let user choose between different generation modes
    """
    print(f"\n" + "="*60)
    print(f"🎯 CHOIX DU MODE DE GÉNÉRATION DE SCÉNARIOS")
    print(f"="*60)
    print(f"")
    print(f"📋 OPTION 1: Mode SIMPLE (~30 scénarios)")
    print(f"   • Échantillon de générateurs")
    print(f"   • 10 lignes principales") 
    print(f"   • 5 charges principales")
    print(f"   • 5 transformateurs")
    print(f"   ⏱️  Temps: ~15-30 minutes")
    print(f"")
    print(f"📋 OPTION 2: Mode COMPLET SIMPLE (~100 scénarios)")
    print(f"   • TOUS les générateurs")
    print(f"   • TOUTES les lignes")
    print(f"   • TOUTES les charges") 
    print(f"   • TOUS les transformateurs")
    print(f"   ⏱️  Temps: ~2-5 heures")
    print(f"")
    print(f"📋 OPTION 3: Mode COMPLET AVEC DOUBLES (~2000+ scénarios)")
    print(f"   • Tous les scénarios simples")
    print(f"   • PLUS TOUS les scénarios doubles AUTORISÉS:")
    print(f"     ✅ line/line, line/gen, line/load, line/transformer")
    print(f"     ✅ transformer/transformer, transformer/gen, transformer/load") 
    print(f"     🚫 EXCLUS: gen/gen, gen/load, load/load")
    print(f"   ⏱️  Temps: ~20-40 heures")
    print(f"")
    
    # Configuration automatique (vous pouvez changer cette valeur)
    AUTO_MODE = 1  # 1=Simple, 2=Complet Simple, 3=Complet avec doubles
    
    if AUTO_MODE == 1:
        print(f"🔧 Mode automatique sélectionné: SIMPLE (~30 scénarios)")
        return "simple"
    elif AUTO_MODE == 2:
        print(f"🔧 Mode automatique sélectionné: COMPLET SIMPLE (~100 scénarios)")  
        return "complete"
    elif AUTO_MODE == 3:
        print(f"🔧 Mode automatique sélectionné: COMPLET AVEC DOUBLES (~2000+ scénarios)")
        print(f"🚫 Combinaisons gen/gen, gen/load, load/load seront filtrées")
        return "massive"
    else:
        print(f"🔧 Mode par défaut: COMPLET SIMPLE")
        return "complete"

# ── SIMPLE Scenario Generation (Pour tests rapides) ──────────────────────
def generate_simple_scenario_list(app):
    """Generate LIMITED list of scenarios for quick testing (~30 scenarios)"""
    print("🔍 Génération de la liste SIMPLE de scénarios...")
    
    # Get all elements from PowerFactory
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    lines = app.GetCalcRelevantObjects("*.ElmLne")
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    transformers = app.GetCalcRelevantObjects("*.ElmTr2")
    
    scenarios = []
    
    # Generator outages (ALL generators for small systems)
    print(f"   🔋 Générateurs trouvés: {len(generators)}")
    for gen in generators:
        if not getattr(gen, 'outserv', 0):  # Only in-service generators
            gen_name = safe_get_name(gen)
            scenarios.append((gen_name, "generator"))
    
    # Line outages (limit to avoid too many scenarios)
    print(f"   📏 Lignes trouvées: {len(lines)}")
    important_lines = lines[:10]  # Take first 10 lines
    for line in important_lines:
        if not getattr(line, 'outserv', 0):  # Only in-service lines
            line_name = safe_get_name(line)
            scenarios.append((line_name, "line"))
    
    # Load outages (limit to major loads)
    print(f"   📍 Charges trouvées: {len(loads)}")
    major_loads = [load for load in loads if get(load, 'plini', 0) > 50]  # Only loads > 50 MW
    for load in major_loads[:5]:  # Take first 5 major loads
        if not getattr(load, 'outserv', 0):  # Only in-service loads
            load_name = safe_get_name(load)
            scenarios.append((load_name, "load"))
    
    # Transformer outages (limit to critical transformers)
    print(f"   🔄 Transformateurs trouvés: {len(transformers)}")
    critical_trafos = transformers[:5]  # Take first 5 transformers
    for trafo in critical_trafos:
        if not getattr(trafo, 'outserv', 0):  # Only in-service transformers
            trafo_name = safe_get_name(trafo)
            scenarios.append((trafo_name, "transformer"))
    
    print(f"   ✅ Total de scénarios SIMPLES générés: {len(scenarios)}")
    print(f"      • Générateurs: {len([s for s in scenarios if s[1] == 'generator'])}")
    print(f"      • Lignes: {len([s for s in scenarios if s[1] == 'line'])}")
    print(f"      • Charges: {len([s for s in scenarios if s[1] == 'load'])}")
    print(f"      • Transformateurs: {len([s for s in scenarios if s[1] == 'transformer'])}")
    
    return scenarios

# ── Enhanced Scenario Generation with Initial Balancing ───────────────────
def main_scenario_generation_comprehensive():
    """Enhanced main function with SINGLE initial power balance correction"""
    
    # Connect to PowerFactory
    app = connect_and_setup()
    
    print("\n" + "="*60)
    print("🔧 ÉTAPE 1: INITIALISATION ET ÉQUILIBRAGE UNIQUE DU SYSTÈME")
    print("="*60)
    
    # UNIQUE POWER BALANCE CORRECTION AT START
    balance_success, balanced_state = initialize_and_balance_system(app)
    if not balance_success:
        print("❌ Échec de l'équilibrage initial. Arrêt.")
        return
    
    print(f"✅ Système équilibré - État sauvegardé pour tous les scénarios")
    print(f"🔒 Cet équilibrage ne sera PLUS refait pendant les scénarios")
    
    # Generate comprehensive scenario list automatically
    generation_mode = choose_scenario_generation_mode()
    
    if generation_mode == "simple":
        comprehensive_scenarios = generate_simple_scenario_list(app)
        double_scenarios = []
    elif generation_mode == "complete":
        comprehensive_scenarios = generate_comprehensive_scenario_list(app)
        double_scenarios = []
    elif generation_mode == "massive":
        # First get all single scenarios
        comprehensive_scenarios = generate_comprehensive_scenario_list(app)
        # Then add FILTERED double contingencies
        double_scenarios = generate_double_contingency_scenarios_complete(app)
        print(f"   📊 Total avec doubles FILTRÉS: {len(comprehensive_scenarios)} + {len(double_scenarios)} = {len(comprehensive_scenarios) + len(double_scenarios)}")
    else:
        comprehensive_scenarios = generate_comprehensive_scenario_list(app)
        double_scenarios = []
    
    # Initialize consolidated H5 file
    initialize_h5_file(H5_OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("📊 ÉTAPE 2: GÉNÉRATION DU SCÉNARIO BASELINE")
    print("="*60)
    
    # Generate baseline scenario (with balanced state)
    print(f"🔄 Génération du scénario baseline avec état équilibré...")
    if solve_power_flow(app):
        baseline_data = collect_system_data(app)
        Y_matrix, Y_sparse = construct_admittance_matrix(baseline_data)
        dV_dP, success_flags = calculate_voltage_sensitivity_numerical(app, baseline_data)
        
        baseline_scenario = {
            'scenario_name': 'baseline',
            'scenario_type': 'baseline',
            'outage_element': 'none',
            'convergence': True,
            'system_data': baseline_data,
            'Y_matrix': Y_matrix,
            'Y_sparse': Y_sparse,
            'sensitivity_dV_dP': dV_dP,
            'sensitivity_success_flags': success_flags,
            'power_balance_info': 'Initial system balancing completed',
            'balanced_state_applied': True
        }
        
        # Save baseline to individual file
        baseline_h5_file = os.path.join(BASE_DIR, "scenario_000_baseline.h5")
        with h5py.File(baseline_h5_file, 'w') as f:
            f.attrs['project'] = PROJECT
            f.attrs['study_case'] = STUDY
            f.attrs['scenario_number'] = 0
            f.attrs['scenario_type'] = 'baseline'
            f.attrs['outage_element'] = 'none'
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['base_mva'] = SBASE_MVA
            f.attrs['power_balanced'] = True
            f.attrs['balance_method'] = 'initial_zero_generator_adjustment'
        
        save_scenario_to_h5(baseline_h5_file, 'scenario_data', baseline_scenario)
        save_scenario_to_h5(H5_OUTPUT_FILE, 'baseline', baseline_scenario)
        print(f"✅ Baseline saved to: {os.path.basename(baseline_h5_file)}")
        
        # Display baseline power balance for verification
        total_pgen = baseline_data['Pgen'].sum()
        total_pload = baseline_data['Pload'].sum()
        print(f"   📊 Baseline vérifiée - Pgen: {total_pgen:.1f} MW, Pload: {total_pload:.1f} MW")
        print(f"   ⚖️  Déséquilibre baseline: {total_pgen - total_pload:.1f} MW")
    else:
        print("❌ Baseline power flow failed!")
        return
    
    print("\n" + "="*60)
    print("⚡ ÉTAPE 3: GÉNÉRATION DES SCÉNARIOS SIMPLES (SANS RÉÉQUILIBRAGE)")
    print("="*60)
    print(f"🔒 NOTE: Tous les scénarios utilisent l'état équilibré initial")
    print(f"🚫 AUCUN rééquilibrage pendant les scénarios de contingence")
    
    scenario_count = 0
    successful_scenarios = 0
    failed_scenarios = []
    
    # Process single contingency scenarios
    for element_name, element_type in comprehensive_scenarios:
        scenario_count += 1
        print(f"\n🔄 Scenario {scenario_count}/{len(comprehensive_scenarios)}: {element_type} outage '{element_name}'")
        
        # Pass balanced_state to scenario generation (NO power balance inside)
        scenario_result = generate_scenario_single_outage(app, element_name, element_type, balanced_state)
        
        if scenario_result:
            # Create individual H5 file for this scenario
            safe_element_name = element_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
            scenario_h5_file = os.path.join(BASE_DIR, f"scenario_{scenario_count:03d}_{element_type}_{safe_element_name}.h5")
            
            # Initialize individual scenario file
            with h5py.File(scenario_h5_file, 'w') as f:
                f.attrs['project'] = PROJECT
                f.attrs['study_case'] = STUDY
                f.attrs['scenario_number'] = scenario_count
                f.attrs['scenario_type'] = element_type
                f.attrs['outage_element'] = element_name
                f.attrs['creation_date'] = datetime.now().isoformat()
                f.attrs['base_mva'] = SBASE_MVA
                f.attrs['power_balanced'] = True
                f.attrs['balance_method'] = 'restored_from_initial_balance'
                f.attrs['balance_recalculated'] = False  # Important flag!
            
            # Save scenario to individual file
            save_scenario_to_h5(scenario_h5_file, 'scenario_data', scenario_result)
            
            # Also save to main consolidated file
            scenario_name = f"scenario_{scenario_count:03d}_{element_type}_{safe_element_name}"
            save_scenario_to_h5(H5_OUTPUT_FILE, scenario_name, scenario_result)
            
            successful_scenarios += 1
            print(f"   ✅ Saved to: {os.path.basename(scenario_h5_file)}")
            
            # Display power balance verification for critical scenarios
            if scenario_count <= 5:  # Show for first 5 scenarios
                total_pgen = scenario_result['system_data']['Pgen'].sum()
                total_pload = scenario_result['system_data']['Pload'].sum()
                print(f"   📊 Vérification - Pgen: {total_pgen:.1f} MW, Pload: {total_pload:.1f} MW")
                
        else:
            failed_scenarios.append((element_name, element_type))
            print(f"   ❌ Failed: {element_type} '{element_name}'")
    
    # Process double contingency scenarios if in massive mode
    if generation_mode == "massive" and double_scenarios:
        print(f"\n" + "="*60)
        print(f"⚡⚡ ÉTAPE 4: GÉNÉRATION DES SCÉNARIOS DOUBLES")
        print("="*60)
        print(f"🚫 Rappel: Combinaisons gen/gen, gen/load, load/load sont FILTRÉES")
        
        for i, double_scenario in enumerate(double_scenarios):
            scenario_count += 1
            print(f"\n🔄 Double Scenario {i+1}/{len(double_scenarios)}: {double_scenario['description']}")
            
            # Generate double contingency
            double_result = generate_double_contingency_scenario(app, double_scenario, balanced_state)
            
            if double_result:
                # Create individual H5 file for double scenario
                safe_scenario_name = double_scenario['scenario_name'].replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
                scenario_h5_file = os.path.join(BASE_DIR, f"scenario_{scenario_count:03d}_double_{safe_scenario_name}.h5")
                
                # Initialize individual scenario file
                with h5py.File(scenario_h5_file, 'w') as f:
                    f.attrs['project'] = PROJECT
                    f.attrs['study_case'] = STUDY
                    f.attrs['scenario_number'] = scenario_count
                    f.attrs['scenario_type'] = 'double_contingency'
                    f.attrs['outage_elements'] = str([elem[0] for elem in double_scenario['outage_elements']])
                    f.attrs['creation_date'] = datetime.now().isoformat()
                    f.attrs['base_mva'] = SBASE_MVA
                    f.attrs['power_balanced'] = True
                    f.attrs['balance_method'] = 'restored_from_initial_balance'
                    f.attrs['balance_recalculated'] = False
                
                # Save scenario to individual file
                save_scenario_to_h5(scenario_h5_file, 'scenario_data', double_result)
                
                # Also save to main consolidated file
                scenario_name = f"scenario_{scenario_count:03d}_double_{safe_scenario_name}"
                save_scenario_to_h5(H5_OUTPUT_FILE, scenario_name, double_result)
                
                successful_scenarios += 1
                print(f"   ✅ Saved to: {os.path.basename(scenario_h5_file)}")
            else:
                failed_scenarios.append((double_scenario['scenario_name'], 'double_contingency'))
                print(f"   ❌ Failed: double {double_scenario['scenario_name']}")

    print("\n" + "="*60)
    print("📈 BILAN FINAL DES SCÉNARIOS")
    print("="*60)
    print(f"✅ Scénarios réussis: {successful_scenarios + 1}/{scenario_count + 1} (incluant baseline)")
    print(f"❌ Scénarios échoués: {len(failed_scenarios)}")
    print(f"🔒 Équilibrage initial utilisé: OUI (1 seule fois)")
    print(f"🚫 Rééquilibrage pendant scénarios: NON (comme demandé)")
    
    if failed_scenarios:
        print(f"\n📋 Scénarios échoués:")
        for element_name, element_type in failed_scenarios:
            print(f"   • {element_type}: {element_name}")
    
    print(f"\n💾 FICHIERS GÉNÉRÉS:")
    print(f"   📁 Répertoire: {BASE_DIR}")
    print(f"   📄 Fichier consolidé: {os.path.basename(H5_OUTPUT_FILE)}")
    print(f"   📄 Fichiers individuels: {successful_scenarios + 1} fichiers H5")
    
    # Create detailed summary file
    summary_data = {
        'generation_info': {
            'total_scenarios_attempted': scenario_count,
            'successful_scenarios': successful_scenarios,
            'failed_scenarios': len(failed_scenarios),
            'success_rate_percent': round((successful_scenarios / scenario_count) * 100, 1),
            'baseline_included': True,
            'power_balance_strategy': 'single_initial_correction',
            'balance_recalculated_per_scenario': False,
            'scenario_filtering_applied': True,
            'filtering_rules': 'exclude_gen_gen_gen_load_load_load',
            'generation_mode': generation_mode
        },
        'power_balance_info': {
            'initial_balance_success': balance_success,
            'balanced_generators_count': len([g for g in balanced_state.values() if g['was_adjusted']]),
            'baseline_total_pgen_MW': float(baseline_data['Pgen'].sum()),
            'baseline_total_pload_MW': float(baseline_data['Pload'].sum()),
            'baseline_imbalance_MW': float(baseline_data['Pgen'].sum() - baseline_data['Pload'].sum())
        },
        'system_info': {
            'project': PROJECT,
            'study_case': STUDY,
            'num_buses': len(baseline_data['buses']['names']),
            'num_edges': len(baseline_data['edges']),
            'successful_sensitivity_buses': int(np.sum(success_flags))
        },
        'scenario_breakdown': {
            'single_scenarios': len(comprehensive_scenarios),
            'double_scenarios': len(double_scenarios) if 'double_scenarios' in locals() else 0,
            'total_scenarios_generated': len(comprehensive_scenarios) + (len(double_scenarios) if 'double_scenarios' in locals() else 0)
        },
        'failed_scenarios': failed_scenarios,
        'generated_files': {
            'baseline': 'scenario_000_baseline.h5',
            'consolidated': os.path.basename(H5_OUTPUT_FILE),
            'individual_scenarios': []
        }
    }
    
    # List all successfully generated individual files
    individual_files = []
    success_count = 0
    
    # Single scenarios
    for i, (element_name, element_type) in enumerate(comprehensive_scenarios, 1):
        if (element_name, element_type) not in failed_scenarios:
            safe_element_name = element_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
            filename = f"scenario_{i:03d}_{element_type}_{safe_element_name}.h5"
            individual_files.append(filename)
            success_count += 1
    
    # Double scenarios
    if generation_mode == "massive" and 'double_scenarios' in locals():
        double_start_count = len(comprehensive_scenarios)
        for i, double_scenario in enumerate(double_scenarios, 1):
            if (double_scenario['scenario_name'], 'double_contingency') not in failed_scenarios:
                safe_scenario_name = double_scenario['scenario_name'].replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
                filename = f"scenario_{double_start_count + i:03d}_double_{safe_scenario_name}.h5"
                individual_files.append(filename)
                success_count += 1
    
    summary_data['generated_files']['individual_scenarios'] = individual_files
    
    # Save summary as JSON
    import json
    summary_file = os.path.join(BASE_DIR, "scenario_generation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"   📊 Résumé détaillé: {os.path.basename(summary_file)}")
    
    # Create file index for easy reference
    file_index = []
    file_index.append({
        'scenario_number': 0,
        'filename': 'scenario_000_baseline.h5',
        'scenario_type': 'baseline',
        'outage_element': 'none',
        'description': 'System baseline with power balance correction',
        'power_balanced': True,
        'balance_method': 'initial_zero_generator_adjustment'
    })
    
    # Add single scenarios to index
    success_count = 0
    for i, (element_name, element_type) in enumerate(comprehensive_scenarios, 1):
        if (element_name, element_type) not in failed_scenarios:
            safe_element_name = element_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
            filename = f"scenario_{i:03d}_{element_type}_{safe_element_name}.h5"
            file_index.append({
                'scenario_number': i,
                'filename': filename,
                'scenario_type': element_type,
                'outage_element': element_name,
                'description': f'{element_type.title()} outage: {element_name}',
                'power_balanced': True,
                'balance_method': 'restored_from_initial_balance'
            })
            success_count += 1
    
    # Add double scenarios to index
    if generation_mode == "massive" and 'double_scenarios' in locals():
        double_start_count = len(comprehensive_scenarios)
        for i, double_scenario in enumerate(double_scenarios, 1):
            if (double_scenario['scenario_name'], 'double_contingency') not in failed_scenarios:
                safe_scenario_name = double_scenario['scenario_name'].replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
                filename = f"scenario_{double_start_count + i:03d}_double_{safe_scenario_name}.h5"
                file_index.append({
                    'scenario_number': double_start_count + i,
                    'filename': filename,
                    'scenario_type': 'double_contingency',
                    'outage_element': str([elem[0] for elem in double_scenario['outage_elements']]),
                    'description': double_scenario['description'],
                    'power_balanced': True,
                    'balance_method': 'restored_from_initial_balance'
                })
    
    # Save file index as CSV for easy viewing
    file_index_df = pd.DataFrame(file_index)
    index_file = os.path.join(BASE_DIR, "scenario_file_index.csv")
    file_index_df.to_csv(index_file, index=False)
    
    print(f"   📋 Index des fichiers: {os.path.basename(index_file)}")
    
    print(f"\n🎉 GÉNÉRATION COMPLÈTE TERMINÉE!")
    print(f"📊 Taux de succès global: {(successful_scenarios + 1)/(scenario_count + 1)*100:.1f}%")
    print(f"🗂️  Total de fichiers H5 créés: {successful_scenarios + 1}")
    print(f"💾 Espace disque utilisé: ~{(successful_scenarios + 1) * 2:.1f} MB (estimation)")
    print(f"⚖️  Méthode d'équilibrage: UNIQUE à l'initialisation (comme demandé)")
    print(f"🚫 Filtrage appliqué: Exclusion gen/gen, gen/load, load/load")
    
    if generation_mode == "massive":
        print(f"🔢 Scénarios simples: {len(comprehensive_scenarios)}")
        if 'double_scenarios' in locals():
            print(f"🔢 Scénarios doubles: {len(double_scenarios)}")
            print(f"🔢 Total scénarios: {len(comprehensive_scenarios) + len(double_scenarios)}")
    
    return summary_data

# ── Execution Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Script entry point - executes comprehensive scenario generation
    WITH SINGLE INITIAL POWER BALANCE CORRECTION AND INTELLIGENT FILTERING
    """
    print("🚀 GÉNÉRATEUR DE SCÉNARIOS COMPLET AVEC FILTRAGE INTELLIGENT")
    print("=" * 70)
    print(f"📅 Début d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔒 STRATÉGIE: Équilibrage initial unique - Pas de rééquilibrage par scénario")
    print(f"🚫 FILTRAGE: Exclusion gen/gen, gen/load, load/load des doubles")
    print(f"✅ INCLUSION: TOUS les autres scénarios possibles")
    
    try:
        # Execute comprehensive scenario generation
        start_time = time.time()
        
        results = main_scenario_generation_comprehensive()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n⏱️  TEMPS D'EXÉCUTION TOTAL: {execution_time/60:.1f} minutes")
        print(f"🎯 GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
        
        if results:
            print(f"\n📊 STATISTIQUES FINALES:")
            print(f"   • Mode de génération: {results['generation_info']['generation_mode']}")
            print(f"   • Scénarios générés: {results['generation_info']['successful_scenarios'] + 1}")
            print(f"   • Scénarios simples: {results['scenario_breakdown']['single_scenarios']}")
            print(f"   • Scénarios doubles: {results['scenario_breakdown']['double_scenarios']}")
            print(f"   • Taux de succès: {results['generation_info']['success_rate_percent']}%")
            print(f"   • Stratégie équilibrage: {results['generation_info']['power_balance_strategy']}")
            print(f"   • Rééquilibrage par scénario: {results['generation_info']['balance_recalculated_per_scenario']}")
            print(f"   • Filtrage appliqué: {results['generation_info']['scenario_filtering_applied']}")
            print(f"   • Règles de filtrage: {results['generation_info']['filtering_rules']}")
            print(f"   • Générateurs ajustés initialement: {results['power_balance_info']['balanced_generators_count']}")
            print(f"   • Déséquilibre final baseline: {results['power_balance_info']['baseline_imbalance_MW']:.1f} MW")
            print(f"   • Buses avec sensibilité: {results['system_info']['successful_sensitivity_buses']}")
        
        print(f"\n✅ TOUS LES FICHIERS H5 SONT PRÊTS POUR VOS MODÈLES GNN!")
        print(f"🔑 AVANTAGES DE CETTE APPROCHE:")
        print(f"   • ⚡ Plus rapide (pas de rééquilibrage répétitif)")
        print(f"   • 🎯 Cohérent (même état de base pour tous les scénarios)")
        print(f"   • 🧪 Isolé (seul l'effet de la panne est étudié)")
        print(f"   • 📊 Comparable (tous les scénarios sur la même base)")
        print(f"   • 🚫 Filtré (exclusion des combinaisons problématiques)")
        print(f"   • 🔍 Complet (tous les scénarios autorisés sont inclus)")
        
    except Exception as e:
        print(f"\n❌ ERREUR DURANT L'EXÉCUTION:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 SUGGESTIONS DE DÉBOGAGE:")
        print(f"   1. Vérifiez que PowerFactory est ouvert")
        print(f"   2. Vérifiez le nom du projet dans la configuration")
        print(f"   3. Vérifiez que le study case existe")
        print(f"   4. Vérifiez les permissions d'écriture dans le répertoire")
        print(f"   5. Vérifiez que le système initial peut être équilibré")
        
    finally:
        print(f"\n📅 Fin d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)