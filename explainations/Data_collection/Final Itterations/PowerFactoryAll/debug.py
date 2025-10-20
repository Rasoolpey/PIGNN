# diagnostic_powerfactory.py
"""
Script de diagnostic pour vérifier si PowerFactory simule vraiment
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# PowerFactory connection
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

PROJECT = "39 Bus New England System"
STUDY = "RMS_Simulation"

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

def safe_get_name(obj):
    try:
        return obj.loc_name if obj else "Unknown"
    except:
        return "Unknown"

def diagnostic_powerfactory():
    """Diagnostic complet PowerFactory"""
    
    print("🔍 DIAGNOSTIC POWERFACTORY COMPLET")
    print("="*50)
    print(f"📅 {datetime.now().strftime('%H:%M:%S')}")
    
    # 1. Test de connexion
    print("\n1️⃣ TEST DE CONNEXION")
    print("-"*30)
    
    try:
        app = pf.GetApplication()
        if not app:
            print("❌ PowerFactory pas trouvé!")
            return False
        print("✅ PowerFactory connecté")
        
        # Reset
        if hasattr(app, "ResetCalculation"):
            app.ResetCalculation()
            print("✅ Calculs resetés")
        
        # Activer projet
        if app.ActivateProject(PROJECT) != 0:
            print(f"❌ Projet '{PROJECT}' pas trouvé!")
            return False
        print(f"✅ Projet '{PROJECT}' activé")
        
        # Study case
        study_case = None
        for case in app.GetProjectFolder("study").GetContents("*.IntCase"):
            if case.loc_name == STUDY:
                study_case = case
                break
        
        if not study_case:
            print(f"❌ Study case '{STUDY}' pas trouvé!")
            return False
        
        study_case.Activate()
        print(f"✅ Study case '{STUDY}' activé")
        
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False
    
    # 2. Test de load flow baseline
    print("\n2️⃣ TEST LOAD FLOW BASELINE")
    print("-"*30)
    
    def solve_power_flow_verbose(app):
        """Solve power flow avec diagnostic"""
        try:
            print("   🔄 Récupération ComLdf...")
            comLdf = app.GetFromStudyCase("ComLdf")
            if not comLdf:
                print("   ❌ ComLdf non trouvé!")
                return False
            
            print("   🔧 Configuration load flow...")
            comLdf.iopt_net = 0      # AC load flow
            comLdf.iopt_at = 1       # Automatic tap adjustment
            comLdf.errlf = 0.1       # Relaxed tolerance
            comLdf.maxiter = 50      # Maximum iterations
            
            print("   ⚡ Exécution load flow...")
            start_time = time.time()
            ierr = comLdf.Execute()
            end_time = time.time()
            
            print(f"   ⏱️  Temps d'exécution: {(end_time - start_time)*1000:.1f} ms")
            print(f"   📊 Code retour: {ierr}")
            
            if ierr == 0:
                print("   ✅ Load flow convergé!")
                return True
            else:
                print(f"   ❌ Load flow échoué (code {ierr})")
                return False
            
        except Exception as e:
            print(f"   ❌ Erreur load flow: {e}")
            return False
    
    # Test baseline
    baseline_success = solve_power_flow_verbose(app)
    
    if not baseline_success:
        print("❌ Load flow baseline échoué - ARRÊT")
        return False
    
    # 3. Collecter données baseline
    print("\n3️⃣ COLLECTE DONNÉES BASELINE")
    print("-"*30)
    
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    generators = app.GetCalcRelevantObjects("*.ElmSym")
    lines = app.GetCalcRelevantObjects("*.ElmLne")
    
    print(f"   🏪 Buses: {len(buses)}")
    print(f"   🔋 Générateurs: {len(generators)}")
    print(f"   📏 Lignes: {len(lines)}")
    
    # Voltages baseline
    baseline_voltages = []
    for bus in buses:
        voltage = get(bus, 'm:u', 1.0)
        baseline_voltages.append(voltage)
    
    baseline_voltages = np.array(baseline_voltages)
    print(f"   📊 Tensions baseline: {baseline_voltages.min():.3f} - {baseline_voltages.max():.3f} pu")
    
    # 4. Test d'une panne réelle
    print("\n4️⃣ TEST PANNE RÉELLE")
    print("-"*30)
    
    # Trouver un générateur
    test_gen = None
    for gen in generators:
        if not getattr(gen, 'outserv', 0):
            test_gen = gen
            break
    
    if not test_gen:
        print("❌ Aucun générateur en service trouvé!")
        return False
    
    gen_name = safe_get_name(test_gen)
    print(f"   🎯 Test avec générateur: {gen_name}")
    
    # Puissance avant
    power_before = get(test_gen, 'pgini', 0)
    print(f"   ⚡ Puissance avant: {power_before:.1f} MW")
    
    # Créer panne
    original_outserv = getattr(test_gen, 'outserv', 0)
    print(f"   🔧 État original outserv: {original_outserv}")
    
    try:
        print("   🚫 Mise hors service...")
        test_gen.SetAttribute('outserv', 1)
        new_outserv = getattr(test_gen, 'outserv', 0)
        print(f"   ✅ Nouvel état outserv: {new_outserv}")
        
        # Résoudre avec panne
        print("   🔄 Load flow avec panne...")
        panne_success = solve_power_flow_verbose(app)
        
        if panne_success:
            # Collecter nouvelles tensions
            panne_voltages = []
            for bus in buses:
                voltage = get(bus, 'm:u', 1.0)
                panne_voltages.append(voltage)
            
            panne_voltages = np.array(panne_voltages)
            print(f"   📊 Tensions avec panne: {panne_voltages.min():.3f} - {panne_voltages.max():.3f} pu")
            
            # Calculer différences
            voltage_diff = np.abs(panne_voltages - baseline_voltages)
            max_diff = voltage_diff.max()
            mean_diff = voltage_diff.mean()
            
            print(f"   🔍 Différence max tension: {max_diff:.6f} pu")
            print(f"   🔍 Différence moyenne: {mean_diff:.6f} pu")
            
            if max_diff > 0.001:
                print("   ✅ PANNE DÉTECTÉE - Tensions changent significativement!")
            elif max_diff > 0.0001:
                print("   ⚠️  Petit changement détecté")
            else:
                print("   ❌ AUCUN CHANGEMENT - Panne pas effective!")
        
    except Exception as e:
        print(f"   ❌ Erreur panne: {e}")
    
    finally:
        # Restaurer
        try:
            test_gen.SetAttribute('outserv', original_outserv)
            print(f"   🔄 Générateur restauré")
        except:
            print(f"   ⚠️  Erreur restauration")
    
    # 5. Test vitesse réelle
    print("\n5️⃣ TEST VITESSE MULTIPLE SCÉNARIOS")
    print("-"*30)
    
    test_elements = []
    
    # Prendre quelques générateurs
    for gen in generators[:3]:
        if not getattr(gen, 'outserv', 0):
            test_elements.append((safe_get_name(gen), "generator", gen))
    
    # Prendre quelques lignes
    for line in lines[:3]:
        if not getattr(line, 'outserv', 0):
            test_elements.append((safe_get_name(line), "line", line))
    
    print(f"   🎯 Test sur {len(test_elements)} éléments")
    
    start_total = time.time()
    
    for i, (name, type_elem, obj) in enumerate(test_elements):
        print(f"   📊 Scénario {i+1}: {type_elem} {name}")
        
        original_state = getattr(obj, 'outserv', 0)
        
        try:
            start_scenario = time.time()
            
            # Créer panne
            obj.SetAttribute('outserv', 1)
            
            # Résoudre
            success = solve_power_flow_verbose(app)
            
            end_scenario = time.time()
            scenario_time = end_scenario - start_scenario
            
            print(f"      ⏱️  Temps scénario: {scenario_time*1000:.1f} ms")
            print(f"      📊 Succès: {success}")
            
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
        
        finally:
            # Restaurer
            try:
                obj.SetAttribute('outserv', original_state)
            except:
                pass
    
    end_total = time.time()
    total_time = end_total - start_total
    
    print(f"\n   📊 RÉSULTATS VITESSE:")
    print(f"      ⏱️  Temps total: {total_time:.2f} secondes")
    print(f"      ⏱️  Temps par scénario: {total_time/len(test_elements):.2f} secondes")
    print(f"      📈 Scénarios/seconde: {len(test_elements)/total_time:.1f}")
    
    if total_time < 1:
        print("      ⚠️  TRÈS RAPIDE - Vérifier si simulations réelles!")
    elif total_time < 5:
        print("      ✅ Vitesse normale pour système simple")
    else:
        print("      ⏳ Vitesse normale pour système complexe")
    
    return True

if __name__ == "__main__":
    print("🚀 DIAGNOSTIC POWERFACTORY")
    print("="*50)
    
    success = diagnostic_powerfactory()
    
    if success:
        print(f"\n✅ DIAGNOSTIC TERMINÉ")
    else:
        print(f"\n❌ DIAGNOSTIC ÉCHOUÉ")