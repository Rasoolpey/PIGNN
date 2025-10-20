# main.py
"""
SCRIPT PRINCIPAL - GÉNÉRATEUR DE SCÉNARIOS COMPLET AVEC FILTRAGE INTELLIGENT
=============================================================================

Point d'entrée principal pour l'exécution du générateur de scénarios.
Utilise tous les modules pour orchestrer la génération complète.

Usage:
    python main.py

Configuration:
    Modifiez les variables PROJECT, STUDY, et MODE selon vos besoins.
"""

import sys
import os
import time
from datetime import datetime

# Import du module orchestrateur principal
from Storage import H5StorageOrchestrator

# ── Configuration (adaptable) ──────────────────────────────────────────────
PROJECT = "39 Bus New England System"  # Modifiable
STUDY = "RMS_Simulation"
BASE_DIR = os.path.join(os.getcwd(), "scenario_analysis")

def main():
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
    print(f"🎯 Projet: {PROJECT}")
    print(f"📁 Répertoire de sortie: {BASE_DIR}")
    print()
    
    try:
        # Execute comprehensive scenario generation
        start_time = time.time()
        
        # Initialize orchestrator
        orchestrator = H5StorageOrchestrator(BASE_DIR, PROJECT, STUDY)
        
        # Run complete generation
        results = orchestrator.main_scenario_generation_comprehensive()
        
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
        
        return 0  # Success
        
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
        
        return 1  # Error
        
    finally:
        print(f"\n📅 Fin d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

def run_custom_configuration():
    """Exemple d'utilisation avec configuration personnalisée"""
    
    config = {
        'project': '39 Bus New England System',
        'study': 'RMS_Simulation',
        'output_dir': './custom_scenario_analysis'
    }
    
    print("🔧 Exécution avec configuration personnalisée...")
    
    orchestrator = H5StorageOrchestrator(
        config['output_dir'], 
        config['project'], 
        config['study']
    )
    
    results = orchestrator.run_generation_pipeline(config)
    return results

def run_baseline_only():
    """Exemple d'utilisation pour générer seulement le baseline"""
    
    print("📊 Génération du baseline seulement...")
    
    # Cette fonctionnalité peut être ajoutée dans l'orchestrateur
    # pour des tests rapides ou des analyses spécifiques
    
    orchestrator = H5StorageOrchestrator(BASE_DIR, PROJECT, STUDY)
    
    # Connect and setup
    app = orchestrator.pf_engine.connect_and_setup()
    orchestrator.balance_manager = PowerBalanceManager(orchestrator.pf_engine)
    orchestrator.data_engine = DataAnalysisEngine(orchestrator.pf_engine)
    
    # Balance system
    balance_success, balanced_state = orchestrator.balance_manager.initialize_and_balance_system()
    
    if balance_success:
        # Generate only baseline
        if orchestrator.pf_engine.solve_power_flow():
            baseline_data = orchestrator.data_engine.collect_system_data()
            Y_matrix, Y_sparse = orchestrator.data_engine.construct_admittance_matrix(baseline_data)
            dV_dP, success_flags = orchestrator.data_engine.calculate_voltage_sensitivity_numerical(baseline_data)
            
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
            
            # Save baseline only
            baseline_h5_file = os.path.join(BASE_DIR, "baseline_only.h5")
            orchestrator.save_scenario_to_h5(baseline_h5_file, 'baseline', baseline_scenario)
            
            print(f"✅ Baseline saved to: {baseline_h5_file}")
            return baseline_scenario
    
    return None

if __name__ == "__main__":
    """
    Point d'entrée principal du script
    """
    
    # Vérifier les arguments de ligne de commande pour des modes spéciaux
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "baseline":
            print("🎯 Mode baseline seulement")
            result = run_baseline_only()
            sys.exit(0 if result else 1)
            
        elif mode == "custom":
            print("🎯 Mode configuration personnalisée")
            result = run_custom_configuration()
            sys.exit(0 if result else 1)
            
        elif mode == "help":
            print("📋 UTILISATION:")
            print("   python main.py              # Exécution normale complète")
            print("   python main.py baseline     # Génération baseline seulement")
            print("   python main.py custom       # Configuration personnalisée")
            print("   python main.py help         # Afficher cette aide")
            print()
            print("📝 CONFIGURATION:")
            print(f"   Projet PowerFactory: {PROJECT}")
            print(f"   Study case: {STUDY}")
            print(f"   Répertoire de sortie: {BASE_DIR}")
            print()
            print("🔧 MODIFICATION:")
            print("   Éditez les variables PROJECT, STUDY, BASE_DIR dans main.py")
            sys.exit(0)
            
        else:
            print(f"❌ Mode '{mode}' non reconnu. Utilisez 'python main.py help' pour l'aide.")
            sys.exit(1)
    
    # Exécution normale
    exit_code = main()
    sys.exit(exit_code)