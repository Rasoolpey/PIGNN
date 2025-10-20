# data_analysis_engine.py
"""
MODULE 3: Data Collection & Analysis Engine
==========================================
Collection complète des données système, construction de la matrice d'admittance,
calcul de sensibilité de tension (méthode numérique), analyses électriques avancées.
"""

import numpy as np
from scipy.sparse import csr_matrix
from PowerFactoryConnexion import safe_get_name, get

class DataAnalysisEngine:
    def __init__(self, pf_engine):
        self.pf_engine = pf_engine
        
    def collect_system_data(self):
        """Collect all essential system data for H5 storage"""
        print("📊 Collecte des données système...")
        
        app = self.pf_engine.app
        if not app:
            raise Exception("❌ PowerFactory not connected!")
        
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

    def construct_admittance_matrix(self, system_data):
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

    def calculate_voltage_sensitivity_numerical(self, system_data):
        """
        Calculate dV/dP using numerical differentiation 
        (following the proper method from your sensitivity script)
        """
        print("🔢 Calcul de la sensibilité de tension (méthode numérique)...")
        
        app = self.pf_engine.app
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
        if not self.pf_engine.solve_power_flow():
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
                
                if perturbed and self.pf_engine.solve_power_flow():
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

    def extract_network_topology(self):
        """Extract complete network topology"""
        app = self.pf_engine.app
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        
        topology = {
            'num_buses': len(buses),
            'bus_names': [safe_get_name(bus) for bus in buses],
            'connections': []
        }
        
        # Extract connections from lines and transformers
        lines = app.GetCalcRelevantObjects("*.ElmLne")
        transformers = app.GetCalcRelevantObjects("*.ElmTr2")
        transformers3w = app.GetCalcRelevantObjects("*.ElmTr3")
        
        # Process lines
        for line in lines:
            try:
                bus1 = getattr(line, 'bus1', None)
                bus2 = getattr(line, 'bus2', None)
                if bus1 and bus2:
                    topology['connections'].append({
                        'from_bus': safe_get_name(bus1),
                        'to_bus': safe_get_name(bus2),
                        'element_type': 'line',
                        'element_name': safe_get_name(line),
                        'in_service': not getattr(line, 'outserv', 0)
                    })
            except:
                pass
        
        # Process 2W transformers
        for trafo in transformers:
            try:
                bus1 = getattr(trafo, 'bus1', None)
                bus2 = getattr(trafo, 'bus2', None)
                if bus1 and bus2:
                    topology['connections'].append({
                        'from_bus': safe_get_name(bus1),
                        'to_bus': safe_get_name(bus2),
                        'element_type': 'transformer_2w',
                        'element_name': safe_get_name(trafo),
                        'in_service': not getattr(trafo, 'outserv', 0)
                    })
            except:
                pass
        
        # Process 3W transformers
        for trafo3w in transformers3w:
            try:
                bus1 = getattr(trafo3w, 'bus1', None)
                bus2 = getattr(trafo3w, 'bus2', None)
                bus3 = getattr(trafo3w, 'bus3', None)
                
                # Create connections for each winding pair
                if bus1 and bus2:
                    topology['connections'].append({
                        'from_bus': safe_get_name(bus1),
                        'to_bus': safe_get_name(bus2),
                        'element_type': 'transformer_3w',
                        'element_name': safe_get_name(trafo3w) + '_12',
                        'in_service': not getattr(trafo3w, 'outserv', 0)
                    })
                if bus1 and bus3:
                    topology['connections'].append({
                        'from_bus': safe_get_name(bus1),
                        'to_bus': safe_get_name(bus3),
                        'element_type': 'transformer_3w',
                        'element_name': safe_get_name(trafo3w) + '_13',
                        'in_service': not getattr(trafo3w, 'outserv', 0)
                    })
                if bus2 and bus3:
                    topology['connections'].append({
                        'from_bus': safe_get_name(bus2),
                        'to_bus': safe_get_name(bus3),
                        'element_type': 'transformer_3w',
                        'element_name': safe_get_name(trafo3w) + '_23',
                        'in_service': not getattr(trafo3w, 'outserv', 0)
                    })
            except:
                pass
        
        print(f"   ✅ Topologie extraite: {len(topology['connections'])} connexions")
        return topology

    def calculate_power_arrays(self):
        """Calculate Pgen, Qgen, Pload, Qload arrays"""
        app = self.pf_engine.app
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        generators = app.GetCalcRelevantObjects("*.ElmSym")
        loads = app.GetCalcRelevantObjects("*.ElmLod")
        
        num_buses = len(buses)
        Pgen = np.zeros(num_buses)
        Qgen = np.zeros(num_buses)
        Pload = np.zeros(num_buses)
        Qload = np.zeros(num_buses)
        
        # Generator powers
        for gen in generators:
            if not getattr(gen, 'outserv', 0):
                try:
                    bus = getattr(gen, 'bus1', None)
                    if bus:
                        bus_idx = buses.index(bus)
                        Pgen[bus_idx] += get(gen, 'pgini', 0)
                        Qgen[bus_idx] += get(gen, 'qgini', 0)
                except:
                    pass
        
        # Load powers
        for load in loads:
            if not getattr(load, 'outserv', 0):
                try:
                    bus = getattr(load, 'bus1', None)
                    if bus:
                        bus_idx = buses.index(bus)
                        Pload[bus_idx] += get(load, 'plini', 0)
                        Qload[bus_idx] += get(load, 'qlini', 0)
                except:
                    pass
        
        return {
            'Pgen': Pgen,
            'Qgen': Qgen,
            'Pload': Pload,
            'Qload': Qload,
            'total_pgen': Pgen.sum(),
            'total_pload': Pload.sum(),
            'power_balance': Pgen.sum() - Pload.sum()
        }

    def analyze_voltage_profile(self):
        """Analyze voltage profile across all buses"""
        app = self.pf_engine.app
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        
        voltages_pu = np.array([get(bus, 'm:u', 1.0) for bus in buses])
        angles_deg = np.array([get(bus, 'm:phiu', 0.0) for bus in buses])
        nominal_kv = np.array([get(bus, 'uknom', 0.0) for bus in buses])
        
        voltage_analysis = {
            'voltages_pu': voltages_pu,
            'angles_deg': angles_deg,
            'nominal_kv': nominal_kv,
            'min_voltage': np.min(voltages_pu),
            'max_voltage': np.max(voltages_pu),
            'mean_voltage': np.mean(voltages_pu),
            'std_voltage': np.std(voltages_pu),
            'violations_low': np.sum(voltages_pu < 0.95),
            'violations_high': np.sum(voltages_pu > 1.05),
            'total_buses': len(buses)
        }
        
        print(f"   📊 Analyse de tension: {voltage_analysis['violations_low']} violations basses, {voltage_analysis['violations_high']} violations hautes")
        
        return voltage_analysis

    def extract_element_details(self):
        """Extract detailed information about all system elements"""
        app = self.pf_engine.app
        
        element_details = {
            'generators': [],
            'loads': [],
            'lines': [],
            'transformers_2w': [],
            'transformers_3w': []
        }
        
        # Generator details
        generators = app.GetCalcRelevantObjects("*.ElmSym")
        for gen in generators:
            try:
                gen_info = {
                    'name': safe_get_name(gen),
                    'bus': safe_get_name(getattr(gen, 'bus1', None)),
                    'pgini': get(gen, 'pgini', 0),
                    'qgini': get(gen, 'qgini', 0),
                    'pmax': get(gen, 'Pmax', 0),
                    'pmin': get(gen, 'Pmin', 0),
                    'qmax': get(gen, 'Qmax', 0),
                    'qmin': get(gen, 'Qmin', 0),
                    'in_service': not getattr(gen, 'outserv', 0)
                }
                element_details['generators'].append(gen_info)
            except:
                pass
        
        # Load details
        loads = app.GetCalcRelevantObjects("*.ElmLod")
        for load in loads:
            try:
                load_info = {
                    'name': safe_get_name(load),
                    'bus': safe_get_name(getattr(load, 'bus1', None)),
                    'plini': get(load, 'plini', 0),
                    'qlini': get(load, 'qlini', 0),
                    'in_service': not getattr(load, 'outserv', 0)
                }
                element_details['loads'].append(load_info)
            except:
                pass
        
        # Line details
        lines = app.GetCalcRelevantObjects("*.ElmLne")
        for line in lines:
            try:
                line_info = {
                    'name': safe_get_name(line),
                    'from_bus': safe_get_name(getattr(line, 'bus1', None)),
                    'to_bus': safe_get_name(getattr(line, 'bus2', None)),
                    'R1': get(line, 'R1', 0),
                    'X1': get(line, 'X1', 0),
                    'B1': get(line, 'B1', 0),
                    'length': get(line, 'dline', 0),
                    'in_service': not getattr(line, 'outserv', 0)
                }
                element_details['lines'].append(line_info)
            except:
                pass
        
        # Transformer 2W details
        transformers = app.GetCalcRelevantObjects("*.ElmTr2")
        for trafo in transformers:
            try:
                trafo_info = {
                    'name': safe_get_name(trafo),
                    'from_bus': safe_get_name(getattr(trafo, 'bus1', None)),
                    'to_bus': safe_get_name(getattr(trafo, 'bus2', None)),
                    'uk': get(trafo, 'uk', 0),
                    'ukr': get(trafo, 'ukr', 0),
                    'snom': get(trafo, 'Snom', 0),
                    'in_service': not getattr(trafo, 'outserv', 0)
                }
                element_details['transformers_2w'].append(trafo_info)
            except:
                pass
        
        # Transformer 3W details
        transformers3w = app.GetCalcRelevantObjects("*.ElmTr3")
        for trafo3w in transformers3w:
            try:
                trafo3w_info = {
                    'name': safe_get_name(trafo3w),
                    'bus1': safe_get_name(getattr(trafo3w, 'bus1', None)),
                    'bus2': safe_get_name(getattr(trafo3w, 'bus2', None)),
                    'bus3': safe_get_name(getattr(trafo3w, 'bus3', None)),
                    'snom': get(trafo3w, 'Snom', 0),
                    'in_service': not getattr(trafo3w, 'outserv', 0)
                }
                element_details['transformers_3w'].append(trafo3w_info)
            except:
                pass
        
        print(f"   ✅ Détails extraits: {len(element_details['generators'])} gens, {len(element_details['loads'])} charges, {len(element_details['lines'])} lignes")
        
        return element_details

    def validate_system_data(self, system_data):
        """Validate collected system data for consistency"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check power balance
            total_pgen = system_data['Pgen'].sum()
            total_pload = system_data['Pload'].sum()
            imbalance = abs(total_pgen - total_pload)
            
            if imbalance > 100:  # More than 100 MW imbalance
                validation_results['errors'].append(f"Large power imbalance: {imbalance:.1f} MW")
                validation_results['valid'] = False
            elif imbalance > 10:  # More than 10 MW imbalance
                validation_results['warnings'].append(f"Power imbalance: {imbalance:.1f} MW")
            
            # Check for NaN values
            for key, array in system_data.items():
                if isinstance(array, np.ndarray):
                    if np.any(np.isnan(array)):
                        validation_results['errors'].append(f"NaN values found in {key}")
                        validation_results['valid'] = False
            
            # Check voltage levels
            voltages = system_data['buses']['voltage_pu']
            if np.any(voltages < 0.8) or np.any(voltages > 1.2):
                validation_results['warnings'].append("Extreme voltage levels detected")
            
            # Check network connectivity
            if len(system_data['edges']) == 0:
                validation_results['errors'].append("No network edges found")
                validation_results['valid'] = False
            
            print(f"   🔍 Validation: {'✅ Valide' if validation_results['valid'] else '❌ Erreurs'}")
            if validation_results['warnings']:
                print(f"   ⚠️  {len(validation_results['warnings'])} avertissements")
            if validation_results['errors']:
                print(f"   ❌ {len(validation_results['errors'])} erreurs")
                
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results