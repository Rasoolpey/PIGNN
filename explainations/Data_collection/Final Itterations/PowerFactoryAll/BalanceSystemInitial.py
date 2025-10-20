# power_balance_system.py
"""
MODULE 2: Power Balance & System Initialization
==============================================
Analyse complète de l'équilibre du système, correction automatique 
de l'équilibrage (UNE fois), sauvegarde et restauration des états équilibrés.
"""

import numpy as np
from datetime import datetime
from PowerFactoryConnexion import safe_get_name, get

class PowerBalanceManager:
    def __init__(self, pf_engine):
        self.pf_engine = pf_engine
        self.balanced_state = None
        self.balance_history = []
        
    def initialize_and_balance_system(self):
        """
        INITIALISATION UNIQUE: Corrige l'équilibre de puissance UNE SEULE FOIS
        et sauvegarde l'état équilibré pour tous les scénarios suivants.
        """
        print("🔧 INITIALISATION ET ÉQUILIBRAGE DU SYSTÈME")
        print("-" * 50)
        
        app = self.pf_engine.app
        if not app:
            raise Exception("❌ PowerFactory not connected!")
        
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
                self.balanced_state = balanced_state
                return True, balanced_state
            else:
                print(f"   ⚠️ Déséquilibre encore important!")
                self.balanced_state = balanced_state
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
            self.balanced_state = balanced_state
            return True, balanced_state

    def restore_balanced_state(self):
        """
        Restaure rapidement l'état équilibré sauvegardé lors de l'initialisation.
        Utilisé avant chaque scénario pour repartir de l'état équilibré.
        """
        if not self.balanced_state:
            print("   ⚠️ Aucun état équilibré sauvegardé!")
            return False
            
        try:
            for gen_name, gen_data in self.balanced_state.items():
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
    
    def analyze_current_balance(self):
        """Analyze current power balance state"""
        app = self.pf_engine.app
        if not app:
            return None
            
        generators = app.GetCalcRelevantObjects("*.ElmSym")
        loads = app.GetCalcRelevantObjects("*.ElmLod")
        
        total_gen = sum(get(gen, 'pgini', 0) for gen in generators if not getattr(gen, 'outserv', 0))
        total_load = sum(get(load, 'plini', 0) for load in loads if not getattr(load, 'outserv', 0))
        
        return {
            'total_generation_mw': total_gen,
            'total_load_mw': total_load,
            'imbalance_mw': total_gen - total_load,
            'num_generators': len([g for g in generators if not getattr(g, 'outserv', 0)]),
            'num_loads': len([l for l in loads if not getattr(l, 'outserv', 0)])
        }
    
    def get_balance_correction_strategy(self, imbalance_mw):
        """Determine best strategy for power balance correction"""
        if abs(imbalance_mw) < 10:
            return "no_correction_needed"
        elif imbalance_mw > 0:  # Too much generation
            return "reduce_generation"
        else:  # Too much load
            return "increase_generation"