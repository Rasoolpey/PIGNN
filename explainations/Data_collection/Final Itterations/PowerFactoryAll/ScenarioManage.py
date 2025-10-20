# scenario_management.py
"""
MODULE 4: Scenario Generation & Management
==========================================
Génération de listes de scénarios (simple/complet/massif), logique de filtrage 
intelligent des combinaisons, génération des scénarios simples et doubles.
"""

from itertools import combinations
from PowerFactoryConnexion import safe_get_name, get

class ScenarioManager:
    def __init__(self, pf_engine, balance_manager, data_engine):
        self.pf_engine = pf_engine
        self.balance_manager = balance_manager
        self.data_engine = data_engine
        
    def choose_scenario_generation_mode(self):
        """Let user choose between different generation modes"""
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
        AUTO_MODE = 2  # 1=Simple, 2=Complet Simple, 3=Complet avec doubles
        
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

    def generate_simple_scenario_list(self):
        """Generate LIMITED list of scenarios for quick testing (~30 scenarios)"""
        print("🔍 Génération de la liste SIMPLE de scénarios...")
        
        app = self.pf_engine.app
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

    def generate_comprehensive_scenario_list(self):
        """Generate COMPLETE list of scenarios based on ALL system elements"""
        print("🔍 Génération COMPLÈTE de la liste de scénarios...")
        
        app = self.pf_engine.app
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

    def generate_double_contingency_scenarios_complete(self, max_combinations=None):
        """
        Generate TOUS LES double contingency scenarios AUTORISÉS
        EXCLUSIONS: gen/gen, gen/load, load/load combinations
        INCLUSIONS: TOUS les autres (line/line, line/gen, line/load, line/transformer, transformer/*)
        """
        print(f"🔍 Génération COMPLÈTE des scénarios de contingence double (FILTRÉS)...")
        print(f"🚫 EXCLUSIONS: gen/gen, gen/load, load/load")
        print(f"✅ INCLUSIONS: TOUS les autres scénarios possibles") 
        
        app = self.pf_engine.app
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
        
        # Generate filtered combinations
        double_scenarios = []
        
        total_combinations = 0
        filtered_out = 0
        
        print(f"   ⏳ Génération des combinaisons (peut prendre quelques minutes)...")
        
        for (name1, type1, obj1), (name2, type2, obj2) in combinations(all_elements, 2):
            total_combinations += 1
            
            # APPLIQUER LE FILTRE ICI
            if not self.is_combination_allowed(type1, type2):
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

    def is_combination_allowed(self, type1, type2):
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

    def generate_scenario_single_outage(self, element_name, element_type):
        """
        Generate single element outage scenario.
        NOTE: Power balance is NOT recalculated here - we start from balanced_state.
        """
        
        # STEP 1: Restore balanced state (NO power balance calculation)
        print(f"   🔄 Restoration de l'état équilibré...")
        if not self.balance_manager.restore_balanced_state():
            print(f"   ❌ Failed to restore balanced state")
            return None
        
        # STEP 2: Create outage
        print(f"   ⚡ Création de la panne {element_type}: {element_name}")
        target_element, original_outserv = self.pf_engine.create_outage(element_name, element_type)
        
        if not target_element:
            print(f"   ❌ Element '{element_name}' not found")
            return None
        
        try:
            # STEP 3: Solve power flow (without power balance recalculation)
            print(f"   🔄 Résolution du load flow...")
            if self.pf_engine.solve_power_flow():
                print(f"   ✅ Load flow convergé")
                
                # Collect data
                scenario_data = self.data_engine.collect_system_data()
                
                # Calculate Y matrix
                Y_matrix, Y_sparse = self.data_engine.construct_admittance_matrix(scenario_data)
                
                # Calculate sensitivity
                dV_dP, success_flags = self.data_engine.calculate_voltage_sensitivity_numerical(scenario_data)
                
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
                return scenario_result
            else:
                print(f"   ❌ Power flow failed for '{element_name}'")
                return None
                
        except Exception as e:
            print(f"   ❌ Error in scenario '{element_name}': {e}")
            return None
        finally:
            # Restore original state
            self.pf_engine.restore_element(target_element, original_outserv)

    def generate_double_contingency_scenario(self, double_scenario_def):
        """
        Generate double contingency scenario (2 simultaneous outages)
        """
        outage_elements = double_scenario_def['outage_elements']
        
        # Store original states
        original_states = []
        
        try:
            # STEP 1: Restore balanced state
            print(f"   🔄 Restoration de l'état équilibré...")
            if not self.balance_manager.restore_balanced_state():
                print(f"   ❌ Failed to restore balanced state")
                return None
            
            # STEP 2: Create double outage
            print(f"   ⚡⚡ Création de la double panne...")
            for element_name, element_type, element_obj in outage_elements:
                original_outserv = getattr(element_obj, 'outserv', 0)
                original_states.append((element_obj, original_outserv))
                element_obj.SetAttribute('outserv', 1)
                print(f"      • {element_type}: {element_name}")
            
            # STEP 3: Solve power flow
            print(f"   🔄 Résolution du load flow...")
            if self.pf_engine.solve_power_flow():
                print(f"   ✅ Load flow convergé")
                
                # Collect data
                scenario_data = self.data_engine.collect_system_data()
                
                # Calculate Y matrix
                Y_matrix, Y_sparse = self.data_engine.construct_admittance_matrix(scenario_data)
                
                # Calculate sensitivity
                dV_dP, success_flags = self.data_engine.calculate_voltage_sensitivity_numerical(scenario_data)
                
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