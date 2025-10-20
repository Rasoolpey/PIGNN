# h5_storage_orchestrator.py
"""
MODULE 5: H5 Storage & Orchestration
====================================
Toute la logique de sauvegarde H5, orchestration complète du processus,
génération de résumés et statistiques, fonction main() et point d'entrée.
"""

import os
import h5py
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Import des autres modules
from PowerFactoryConnexion import PowerFactoryEngine
from BalanceSystemInitial import PowerBalanceManager
from DataCollect import DataAnalysisEngine
from ScenarioManage import ScenarioManager

class H5StorageOrchestrator:
    def __init__(self, base_dir, project_name, study_case):
        self.base_dir = base_dir
        self.project = project_name
        self.study = study_case
        self.h5_output_file = os.path.join(base_dir, f"{project_name.replace(' ', '_')}_scenarios_complete.h5")
        
        # Create directories
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize engines
        self.pf_engine = PowerFactoryEngine(project_name, study_case)
        self.balance_manager = None
        self.data_engine = None
        self.scenario_manager = None
        
    def initialize_h5_file(self):
        """Initialize H5 file with baseline data"""
        print(f"💾 Initialisation du fichier H5: {self.h5_output_file}")
        
        with h5py.File(self.h5_output_file, 'w') as f:
            # Global metadata
            f.attrs['project'] = self.project
            f.attrs['study_case'] = self.study
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['base_mva'] = 100.0
            f.attrs['purpose'] = 'Complete scenario analysis with intelligent filtering'
            f.attrs['power_balance_strategy'] = 'single_initial_correction'
            f.attrs['scenario_filtering'] = 'exclude_gen_gen_gen_load_load_load'
            
            # Create main groups
            f.create_group('scenarios')
            f.create_group('baseline')

    def save_scenario_to_h5(self, h5_file, scenario_name, scenario_data):
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
            if isinstance(scenario_data.get('outage_elements', scenario_data.get('outage_element')), list):
                scenario_grp.attrs['outage_elements'] = str(scenario_data.get('outage_elements', scenario_data.get('outage_element')))
            else:
                scenario_grp.attrs['outage_element'] = scenario_data.get('outage_element', 'none')
            
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

    def create_detailed_summary(self, results):
        """Create comprehensive summary and statistics"""
        summary_data = {
            'generation_info': {
                'total_scenarios_attempted': results.get('total_scenarios_attempted', 0),
                'successful_scenarios': results.get('successful_scenarios', 0),
                'failed_scenarios': len(results.get('failed_scenarios', [])),
                'success_rate_percent': round((results.get('successful_scenarios', 0) / max(results.get('total_scenarios_attempted', 1), 1)) * 100, 1),
                'baseline_included': True,
                'power_balance_strategy': 'single_initial_correction',
                'balance_recalculated_per_scenario': False,
                'scenario_filtering_applied': True,
                'filtering_rules': 'exclude_gen_gen_gen_load_load_load',
                'generation_mode': results.get('generation_mode', 'complete')
            },
            'power_balance_info': results.get('power_balance_info', {}),
            'system_info': results.get('system_info', {}),
            'scenario_breakdown': results.get('scenario_breakdown', {}),
            'failed_scenarios': results.get('failed_scenarios', []),
            'generated_files': results.get('generated_files', {})
        }
        
        # Save summary as JSON
        summary_file = os.path.join(self.base_dir, "scenario_generation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"   📊 Résumé détaillé: {os.path.basename(summary_file)}")
        return summary_data

    def main_scenario_generation_comprehensive(self):
        """
        Enhanced main function with SINGLE initial power balance correction
        Orchestration complète du processus
        """
        
        # Connect to PowerFactory
        app = self.pf_engine.connect_and_setup()
        
        # Initialize other engines
        self.balance_manager = PowerBalanceManager(self.pf_engine)
        self.data_engine = DataAnalysisEngine(self.pf_engine)
        self.scenario_manager = ScenarioManager(self.pf_engine, self.balance_manager, self.data_engine)
        
        print("\n" + "="*60)
        print("🔧 ÉTAPE 1: INITIALISATION ET ÉQUILIBRAGE UNIQUE DU SYSTÈME")
        print("="*60)
        
        # UNIQUE POWER BALANCE CORRECTION AT START
        balance_success, balanced_state = self.balance_manager.initialize_and_balance_system()
        if not balance_success:
            print("❌ Échec de l'équilibrage initial. Arrêt.")
            return None
        
        print(f"✅ Système équilibré - État sauvegardé pour tous les scénarios")
        print(f"🔒 Cet équilibrage ne sera PLUS refait pendant les scénarios")
        
        # Generate comprehensive scenario list automatically
        generation_mode = self.scenario_manager.choose_scenario_generation_mode()
        
        if generation_mode == "simple":
            comprehensive_scenarios = self.scenario_manager.generate_simple_scenario_list()
            double_scenarios = []
        elif generation_mode == "complete":
            comprehensive_scenarios = self.scenario_manager.generate_comprehensive_scenario_list()
            double_scenarios = []
        elif generation_mode == "massive":
            # First get all single scenarios
            comprehensive_scenarios = self.scenario_manager.generate_comprehensive_scenario_list()
            # Then add FILTERED double contingencies
            double_scenarios = self.scenario_manager.generate_double_contingency_scenarios_complete()
            print(f"   📊 Total avec doubles FILTRÉS: {len(comprehensive_scenarios)} + {len(double_scenarios)} = {len(comprehensive_scenarios) + len(double_scenarios)}")
        else:
            comprehensive_scenarios = self.scenario_manager.generate_comprehensive_scenario_list()
            double_scenarios = []
        
        # Initialize consolidated H5 file
        self.initialize_h5_file()
        
        print("\n" + "="*60)
        print("📊 ÉTAPE 2: GÉNÉRATION DU SCÉNARIO BASELINE")
        print("="*60)
        
        # Generate baseline scenario (with balanced state)
        print(f"🔄 Génération du scénario baseline avec état équilibré...")
        if self.pf_engine.solve_power_flow():
            baseline_data = self.data_engine.collect_system_data()
            Y_matrix, Y_sparse = self.data_engine.construct_admittance_matrix(baseline_data)
            dV_dP, success_flags = self.data_engine.calculate_voltage_sensitivity_numerical(baseline_data)
            
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
            baseline_h5_file = os.path.join(self.base_dir, "scenario_000_baseline.h5")
            with h5py.File(baseline_h5_file, 'w') as f:
                f.attrs['project'] = self.project
                f.attrs['study_case'] = self.study
                f.attrs['scenario_number'] = 0
                f.attrs['scenario_type'] = 'baseline'
                f.attrs['outage_element'] = 'none'
                f.attrs['creation_date'] = datetime.now().isoformat()
                f.attrs['base_mva'] = 100.0
                f.attrs['power_balanced'] = True
                f.attrs['balance_method'] = 'initial_zero_generator_adjustment'
            
            self.save_scenario_to_h5(baseline_h5_file, 'scenario_data', baseline_scenario)
            self.save_scenario_to_h5(self.h5_output_file, 'baseline', baseline_scenario)
            print(f"✅ Baseline saved to: {os.path.basename(baseline_h5_file)}")
            
            # Display baseline power balance for verification
            total_pgen = baseline_data['Pgen'].sum()
            total_pload = baseline_data['Pload'].sum()
            print(f"   📊 Baseline vérifiée - Pgen: {total_pgen:.1f} MW, Pload: {total_pload:.1f} MW")
            print(f"   ⚖️  Déséquilibre baseline: {total_pgen - total_pload:.1f} MW")
        else:
            print("❌ Baseline power flow failed!")
            return None
        
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
            
            # Generate scenario
            scenario_result = self.scenario_manager.generate_scenario_single_outage(element_name, element_type)
            
            if scenario_result:
                # Create individual H5 file for this scenario
                safe_element_name = element_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
                scenario_h5_file = os.path.join(self.base_dir, f"scenario_{scenario_count:03d}_{element_type}_{safe_element_name}.h5")
                
                # Initialize individual scenario file
                with h5py.File(scenario_h5_file, 'w') as f:
                    f.attrs['project'] = self.project
                    f.attrs['study_case'] = self.study
                    f.attrs['scenario_number'] = scenario_count
                    f.attrs['scenario_type'] = element_type
                    f.attrs['outage_element'] = element_name
                    f.attrs['creation_date'] = datetime.now().isoformat()
                    f.attrs['base_mva'] = 100.0
                    f.attrs['power_balanced'] = True
                    f.attrs['balance_method'] = 'restored_from_initial_balance'
                    f.attrs['balance_recalculated'] = False  # Important flag!
                
                # Save scenario to individual file
                self.save_scenario_to_h5(scenario_h5_file, 'scenario_data', scenario_result)
                
                # Also save to main consolidated file
                scenario_name = f"scenario_{scenario_count:03d}_{element_type}_{safe_element_name}"
                self.save_scenario_to_h5(self.h5_output_file, scenario_name, scenario_result)
                
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
                double_result = self.scenario_manager.generate_double_contingency_scenario(double_scenario)
                
                if double_result:
                    # Create individual H5 file for double scenario
                    safe_scenario_name = double_scenario['scenario_name'].replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
                    scenario_h5_file = os.path.join(self.base_dir, f"scenario_{scenario_count:03d}_double_{safe_scenario_name}.h5")
                    
                    # Initialize individual scenario file
                    with h5py.File(scenario_h5_file, 'w') as f:
                        f.attrs['project'] = self.project
                        f.attrs['study_case'] = self.study
                        f.attrs['scenario_number'] = scenario_count
                        f.attrs['scenario_type'] = 'double_contingency'
                        f.attrs['outage_elements'] = str([elem[0] for elem in double_scenario['outage_elements']])
                        f.attrs['creation_date'] = datetime.now().isoformat()
                        f.attrs['base_mva'] = 100.0
                        f.attrs['power_balanced'] = True
                        f.attrs['balance_method'] = 'restored_from_initial_balance'
                        f.attrs['balance_recalculated'] = False
                    
                    # Save scenario to individual file
                    self.save_scenario_to_h5(scenario_h5_file, 'scenario_data', double_result)
                    
                    # Also save to main consolidated file
                    scenario_name = f"scenario_{scenario_count:03d}_double_{safe_scenario_name}"
                    self.save_scenario_to_h5(self.h5_output_file, scenario_name, double_result)
                    
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
        print(f"   📁 Répertoire: {self.base_dir}")
        print(f"   📄 Fichier consolidé: {os.path.basename(self.h5_output_file)}")
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
                'project': self.project,
                'study_case': self.study,
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
                'consolidated': os.path.basename(self.h5_output_file),
                'individual_scenarios': []
            }
        }
        
        # List all successfully generated individual files
        individual_files = []
        
        # Single scenarios
        for i, (element_name, element_type) in enumerate(comprehensive_scenarios, 1):
            if (element_name, element_type) not in failed_scenarios:
                safe_element_name = element_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
                filename = f"scenario_{i:03d}_{element_type}_{safe_element_name}.h5"
                individual_files.append(filename)
        
        # Double scenarios
        if generation_mode == "massive" and 'double_scenarios' in locals():
            double_start_count = len(comprehensive_scenarios)
            for i, double_scenario in enumerate(double_scenarios, 1):
                if (double_scenario['scenario_name'], 'double_contingency') not in failed_scenarios:
                    safe_scenario_name = double_scenario['scenario_name'].replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
                    filename = f"scenario_{double_start_count + i:03d}_double_{safe_scenario_name}.h5"
                    individual_files.append(filename)
        
        summary_data['generated_files']['individual_scenarios'] = individual_files
        
        # Save summary as JSON
        summary_file = os.path.join(self.base_dir, "scenario_generation_summary.json")
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
        index_file = os.path.join(self.base_dir, "scenario_file_index.csv")
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

    def run_generation_pipeline(self, config):
        """Run complete generation pipeline with configuration"""
        # Update configuration
        if 'project' in config:
            self.project = config['project']
        if 'study' in config:
            self.study = config['study']
        if 'output_dir' in config:
            self.base_dir = config['output_dir']
            os.makedirs(self.base_dir, exist_ok=True)
        
        # Run main generation
        return self.main_scenario_generation_comprehensive()