"""
Validators for graph structure and physics constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_base import PowerGridGraph, PhaseType, CouplingMatrix


class GraphValidator:
    """Validate power grid graph structure and physics"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_graph_structure(self, graph: PowerGridGraph) -> Tuple[bool, List[str]]:
        """
        Validate graph structure integrity.
        
        Returns:
            (is_valid, list_of_errors)
        """
        self.errors = []
        
        # Check all nodes have three phases
        for node_id, phases in graph.nodes.items():
            if len(phases) != 3:
                self.errors.append(f"Node {node_id} doesn't have 3 phases")
            
            for phase in PhaseType:
                if phase not in phases:
                    self.errors.append(f"Node {node_id} missing phase {phase.value}")
        
        # Check all edges have three phases
        for edge_id, phases in graph.edges.items():
            if len(phases) != 3:
                self.errors.append(f"Edge {edge_id} doesn't have 3 phases")
            
            # Check edge endpoints exist
            for phase in PhaseType:
                if phase in phases:
                    edge = phases[phase]
                    if edge.from_node_id not in graph.nodes:
                        self.errors.append(f"Edge {edge_id} references non-existent from_node {edge.from_node_id}")
                    if edge.to_node_id not in graph.nodes:
                        self.errors.append(f"Edge {edge_id} references non-existent to_node {edge.to_node_id}")
        
        # Check coupling matrices
        for node_id in graph.nodes:
            if node_id in graph.node_couplings:
                coupling = graph.node_couplings[node_id]
                if coupling.matrix.shape != (3, 3):
                    self.errors.append(f"Node {node_id} coupling matrix is not 3x3")
        
        for edge_id in graph.edges:
            if edge_id in graph.edge_couplings:
                coupling = graph.edge_couplings[edge_id]
                if coupling.matrix.shape != (3, 3):
                    self.errors.append(f"Edge {edge_id} coupling matrix is not 3x3")
        
        return len(self.errors) == 0, self.errors
    
    def validate_physics_constraints(self, graph: PowerGridGraph) -> Tuple[bool, List[str]]:
        """
        Validate physical constraints.
        
        Returns:
            (is_valid, list_of_warnings)
        """
        self.warnings = []
        
        # Check voltage magnitudes
        for node_id, phases in graph.nodes.items():
            for phase, node in phases.items():
                v_mag = abs(node.voltage_pu)
                if v_mag < 0.5:
                    self.warnings.append(f"Node {node_id} phase {phase.value} voltage too low: {v_mag:.3f} pu")
                elif v_mag > 1.5:
                    self.warnings.append(f"Node {node_id} phase {phase.value} voltage too high: {v_mag:.3f} pu")
        
        # Check impedances
        for edge_id, phases in graph.edges.items():
            for phase, edge in phases.items():
                if hasattr(edge, 'R_ohm') and hasattr(edge, 'X_ohm'):
                    if edge.R_ohm < 0:
                        self.warnings.append(f"Edge {edge_id} has negative resistance")
                    
                    Z_mag = abs(complex(edge.R_ohm, edge.X_ohm))
                    if Z_mag < 1e-6:
                        self.warnings.append(f"Edge {edge_id} impedance too small: {Z_mag:.6f} ohm")
        
        # Check coupling matrix properties
        for edge_id, coupling in graph.edge_couplings.items():
            if not coupling.is_symmetric():
                self.warnings.append(f"Edge {edge_id} coupling matrix is not symmetric")
            
            # Check positive definiteness (for passive components)
            eigenvalues = np.linalg.eigvals(coupling.matrix)
            if np.any(np.real(eigenvalues) < 0):
                self.warnings.append(f"Edge {edge_id} coupling matrix has negative real eigenvalues")
        
        # Check generator limits
        for node_id, phases in graph.nodes.items():
            node = phases[PhaseType.A]  # Check one phase
            if node.node_type == 'generator' and hasattr(node, 'Q_min_MVAR'):
                if hasattr(node, 'Q_actual_MVAR'):
                    if node.Q_actual_MVAR < node.Q_min_MVAR:
                        self.warnings.append(f"Generator {node_id} below Q_min")
                    if node.Q_actual_MVAR > node.Q_max_MVAR:
                        self.warnings.append(f"Generator {node_id} above Q_max")
        
        return len(self.warnings) == 0, self.warnings
    
    def check_power_balance(self, graph: PowerGridGraph) -> Dict[str, float]:
        """
        Check system power balance.
        
        Returns:
            Dictionary with power balance information
        """
        total_gen_p = 0.0
        total_gen_q = 0.0
        total_load_p = 0.0
        total_load_q = 0.0
        
        # Sum generation and load
        for node_id, phases in graph.nodes.items():
            for phase, node in phases.items():
                if node.node_type == 'generator':
                    if hasattr(node, 'P_actual_MW'):
                        total_gen_p += node.P_actual_MW
                    if hasattr(node, 'Q_actual_MVAR'):
                        total_gen_q += node.Q_actual_MVAR
                
                elif node.node_type == 'load':
                    if hasattr(node, 'P_MW'):
                        total_load_p += node.P_MW
                    if hasattr(node, 'Q_MVAR'):
                        total_load_q += node.Q_MVAR
                
                # Check for load at generator bus
                if 'load_P_MW' in node.properties:
                    total_load_p += node.properties['load_P_MW']
                if 'load_Q_MVAR' in node.properties:
                    total_load_q += node.properties['load_Q_MVAR']
        
        # Calculate losses (generation - load)
        losses_p = total_gen_p - total_load_p
        losses_q = total_gen_q - total_load_q
        
        # Calculate percentages
        loss_percent_p = (losses_p / total_gen_p * 100) if total_gen_p > 0 else 0
        loss_percent_q = (losses_q / total_gen_q * 100) if total_gen_q > 0 else 0
        
        return {
            'total_generation_MW': total_gen_p,
            'total_generation_MVAR': total_gen_q,
            'total_load_MW': total_load_p,
            'total_load_MVAR': total_load_q,
            'losses_MW': losses_p,
            'losses_MVAR': losses_q,
            'loss_percent_P': loss_percent_p,
            'loss_percent_Q': loss_percent_q,
            'power_factor_gen': total_gen_p / np.sqrt(total_gen_p**2 + total_gen_q**2) if total_gen_p > 0 else 0,
            'power_factor_load': total_load_p / np.sqrt(total_load_p**2 + total_load_q**2) if total_load_p > 0 else 0
        }
    
    def validate_coupling_matrices(self, graph: PowerGridGraph) -> Tuple[bool, List[str]]:
        """
        Detailed validation of coupling matrices.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Node coupling validation
        for node_id, coupling in graph.node_couplings.items():
            matrix = coupling.matrix
            
            # Check size
            if matrix.shape != (3, 3):
                issues.append(f"Node {node_id}: Coupling matrix shape {matrix.shape} != (3,3)")
                continue
            
            # Check symmetry
            if not np.allclose(matrix, matrix.T):
                issues.append(f"Node {node_id}: Coupling matrix not symmetric")
            
            # Check diagonal dominance (typical for physical systems)
            for i in range(3):
                diag = abs(matrix[i, i])
                off_diag_sum = sum(abs(matrix[i, j]) for j in range(3) if j != i)
                if diag < off_diag_sum:
                    issues.append(f"Node {node_id}: Not diagonally dominant at row {i}")
        
        # Edge coupling validation
        for edge_id, coupling in graph.edge_couplings.items():
            matrix = coupling.matrix
            
            # Check size
            if matrix.shape != (3, 3):
                issues.append(f"Edge {edge_id}: Coupling matrix shape {matrix.shape} != (3,3)")
                continue
            
            # Check symmetry
            if not np.allclose(matrix, matrix.T):
                issues.append(f"Edge {edge_id}: Coupling matrix not symmetric")
            
            # Check physical reasonableness
            # Mutual impedance should be less than self impedance
            for i in range(3):
                self_impedance = abs(matrix[i, i])
                for j in range(3):
                    if i != j:
                        mutual_impedance = abs(matrix[i, j])
                        if mutual_impedance > self_impedance:
                            issues.append(f"Edge {edge_id}: Mutual impedance ({i},{j}) > self impedance")
            
            # Check positive real part (passive component)
            eigenvalues = np.linalg.eigvals(matrix)
            if np.any(np.real(eigenvalues) < -1e-10):
                issues.append(f"Edge {edge_id}: Matrix has negative real eigenvalues (non-passive)")
        
        return len(issues) == 0, issues
    
    def validate_connectivity(self, graph: PowerGridGraph) -> Tuple[bool, List[str]]:
        """
        Check if graph is connected.
        
        Returns:
            (is_connected, list_of_islands)
        """
        # Build adjacency for phase A (assuming phases are connected similarly)
        adjacency = {}
        for node_id in graph.nodes:
            adjacency[node_id] = set()
        
        for edge_id, phases in graph.edges.items():
            edge = phases[PhaseType.A]
            adjacency[edge.from_node_id].add(edge.to_node_id)
            adjacency[edge.to_node_id].add(edge.from_node_id)
        
        # Find connected components using DFS
        visited = set()
        islands = []
        
        for node_id in graph.nodes:
            if node_id not in visited:
                island = []
                stack = [node_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        island.append(current)
                        stack.extend(adjacency[current] - visited)
                
                islands.append(island)
        
        if len(islands) == 1:
            return True, []
        else:
            island_descriptions = []
            for i, island in enumerate(islands):
                island_descriptions.append(f"Island {i+1}: {', '.join(island)}")
            return False, island_descriptions
    
    def generate_validation_report(self, graph: PowerGridGraph) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 60)
        report.append("POWER GRID GRAPH VALIDATION REPORT")
        report.append("=" * 60)
        
        # System info
        info = graph.get_system_info()
        report.append(f"\nSystem Information:")
        report.append(f"  Components: {info['n_components']}")
        report.append(f"  Branches: {info['n_branches']}")
        report.append(f"  Base MVA: {info['base_mva']}")
        report.append(f"  Frequency: {info['frequency_hz']} Hz")
        
        # Structure validation
        report.append(f"\n1. Structure Validation:")
        is_valid, errors = self.validate_graph_structure(graph)
        if is_valid:
            report.append("   ✓ Graph structure is valid")
        else:
            report.append("   ✗ Structure errors found:")
            for error in errors[:5]:  # Show first 5 errors
                report.append(f"     - {error}")
            if len(errors) > 5:
                report.append(f"     ... and {len(errors)-5} more errors")
        
        # Physics validation
        report.append(f"\n2. Physics Constraints:")
        is_valid, warnings = self.validate_physics_constraints(graph)
        if is_valid:
            report.append("   ✓ All physics constraints satisfied")
        else:
            report.append(f"   ⚠ {len(warnings)} warnings found:")
            for warning in warnings[:5]:
                report.append(f"     - {warning}")
            if len(warnings) > 5:
                report.append(f"     ... and {len(warnings)-5} more warnings")
        
        # Power balance
        report.append(f"\n3. Power Balance:")
        balance = self.check_power_balance(graph)
        report.append(f"   Generation: {balance['total_generation_MW']:.1f} MW")
        report.append(f"   Load: {balance['total_load_MW']:.1f} MW")
        report.append(f"   Losses: {balance['losses_MW']:.1f} MW ({balance['loss_percent_P']:.1f}%)")
        
        if balance['loss_percent_P'] > 5:
            report.append("   ⚠ WARNING: Losses exceed 5%!")
        else:
            report.append("   ✓ Losses within acceptable range")
        
        # Connectivity
        report.append(f"\n4. Connectivity:")
        is_connected, islands = self.validate_connectivity(graph)
        if is_connected:
            report.append("   ✓ System is fully connected")
        else:
            report.append(f"   ✗ System has {len(islands)} islands:")
            for island in islands[:3]:
                report.append(f"     - {island}")
        
        # Coupling matrices
        report.append(f"\n5. Coupling Matrices:")
        is_valid, issues = self.validate_coupling_matrices(graph)
        if is_valid:
            report.append("   ✓ All coupling matrices are valid")
        else:
            report.append(f"   ⚠ {len(issues)} issues found in coupling matrices")
            for issue in issues[:3]:
                report.append(f"     - {issue}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)