"""
Build admittance matrix from three-phase graph structure.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_base import PowerGridGraph, PhaseType


class AdmittanceMatrixBuilder:
    """Build Y-matrix from power grid graph"""
    
    def __init__(self, graph: PowerGridGraph):
        """
        Initialize with a power grid graph.
        
        Args:
            graph: PowerGridGraph object
        """
        self.graph = graph
        self.node_to_index = {}
        self.index_to_node = {}
        self._build_index_mapping()
    
    def _build_index_mapping(self):
        """Create mapping between nodes and matrix indices"""
        index = 0
        for node_id, phase_nodes in self.graph.nodes.items():
            for phase in PhaseType:
                node_phase_id = f"{node_id}_{phase.value}"
                self.node_to_index[node_phase_id] = index
                self.index_to_node[index] = node_phase_id
                index += 1
    
    def build_y_matrix(self, use_sparse: bool = True) -> np.ndarray:
        """
        Build complete 3N x 3N admittance matrix.
        
        Args:
            use_sparse: Use sparse matrix representation
        
        Returns:
            Y-matrix (sparse or dense)
        """
        n_nodes = len(self.node_to_index)
        
        if use_sparse:
            Y = sp.lil_matrix((n_nodes, n_nodes), dtype=complex)
        else:
            Y = np.zeros((n_nodes, n_nodes), dtype=complex)
        
        # Add edge contributions
        for edge_id, phase_edges in self.graph.edges.items():
            self._add_edge_to_matrix(Y, edge_id, phase_edges)
        
        # Add shunt contributions
        for node_id, phase_nodes in self.graph.nodes.items():
            self._add_shunt_to_matrix(Y, node_id, phase_nodes)
        
        if use_sparse:
            return Y.tocsr()
        return Y
    
    def _add_edge_to_matrix(self, Y, edge_id: str, phase_edges: Dict):
        """Add edge contribution to Y-matrix"""
        # Get coupling matrix for this edge
        coupling = self.graph.edge_couplings.get(edge_id)
        if coupling is None:
            return
        
        # Get impedance matrix
        Z_matrix = coupling.matrix
        
        # Invert to get admittance
        try:
            Y_edge = np.linalg.inv(Z_matrix)
        except np.linalg.LinAlgError:
            # Singular matrix - skip
            return
        
        # Get node indices for all phases
        from_indices = []
        to_indices = []
        
        for phase in PhaseType:
            edge = phase_edges[phase]
            from_node_id = f"{edge.from_node_id}_{phase.value}"
            to_node_id = f"{edge.to_node_id}_{phase.value}"
            
            from_indices.append(self.node_to_index.get(from_node_id))
            to_indices.append(self.node_to_index.get(to_node_id))
        
        # Add to Y-matrix (accounting for all phase couplings)
        for i, phase_i in enumerate(PhaseType):
            for j, phase_j in enumerate(PhaseType):
                if from_indices[i] is not None and from_indices[j] is not None:
                    # From-from diagonal
                    Y[from_indices[i], from_indices[j]] += Y_edge[i, j]
                
                if to_indices[i] is not None and to_indices[j] is not None:
                    # To-to diagonal
                    Y[to_indices[i], to_indices[j]] += Y_edge[i, j]
                
                if from_indices[i] is not None and to_indices[j] is not None:
                    # From-to off-diagonal
                    Y[from_indices[i], to_indices[j]] -= Y_edge[i, j]
                
                if to_indices[i] is not None and from_indices[j] is not None:
                    # To-from off-diagonal
                    Y[to_indices[i], from_indices[j]] -= Y_edge[i, j]
    
    def _add_shunt_to_matrix(self, Y, node_id: str, phase_nodes: Dict):
        """Add shunt admittance to Y-matrix"""
        for phase in PhaseType:
            node = phase_nodes[phase]
            node_phase_id = f"{node_id}_{phase.value}"
            idx = self.node_to_index.get(node_phase_id)
            
            if idx is None:
                continue
            
            # Add shunt admittance
            Y_shunt = 0 + 0j
            
            if hasattr(node, 'shunt_G_pu') and hasattr(node, 'shunt_B_pu'):
                Y_shunt = complex(node.shunt_G_pu, node.shunt_B_pu)
            elif 'shunt_G_pu' in node.properties:
                Y_shunt = complex(node.properties.get('shunt_G_pu', 0),
                                node.properties.get('shunt_B_pu', 0))
            
            if abs(Y_shunt) > 1e-10:
                Y[idx, idx] += Y_shunt
    
    def get_node_admittance_matrix(self, node_id: str) -> np.ndarray:
        """Get 3x3 admittance matrix for a specific node"""
        Y_node = np.zeros((3, 3), dtype=complex)
        
        indices = []
        for phase in PhaseType:
            node_phase_id = f"{node_id}_{phase.value}"
            indices.append(self.node_to_index.get(node_phase_id))
        
        Y_full = self.build_y_matrix(use_sparse=False)
        
        for i in range(3):
            for j in range(3):
                if indices[i] is not None and indices[j] is not None:
                    Y_node[i, j] = Y_full[indices[i], indices[j]]
        
        return Y_node
    
    def validate_matrix(self, Y) -> Dict:
        """Validate Y-matrix properties"""
        if sp.issparse(Y):
            Y_dense = Y.toarray()
        else:
            Y_dense = Y
        
        n = Y_dense.shape[0]
        
        # Check symmetry
        is_symmetric = np.allclose(Y_dense, Y_dense.T)
        
        # Check for zero diagonal elements
        zero_diag = np.sum(np.abs(np.diag(Y_dense)) < 1e-10)
        
        # Calculate condition number
        try:
            cond_number = np.linalg.cond(Y_dense)
        except:
            cond_number = float('inf')
        
        # Check power balance (sum of each row should be close to zero for lossless)
        row_sums = np.abs(np.sum(Y_dense, axis=1))
        max_imbalance = np.max(row_sums)
        
        return {
            'matrix_size': n,
            'is_symmetric': is_symmetric,
            'zero_diagonal_elements': zero_diag,
            'condition_number': cond_number,
            'max_row_imbalance': max_imbalance,
            'sparsity': 1.0 - np.count_nonzero(Y_dense) / (n * n)
        }
    
    def export_to_numpy(self, filename: str, Y=None):
        """Export Y-matrix to numpy file"""
        if Y is None:
            Y = self.build_y_matrix(use_sparse=False)
        
        np.save(filename, Y)
        
        # Also save the index mapping
        mapping = {
            'node_to_index': self.node_to_index,
            'index_to_node': self.index_to_node
        }
        np.save(filename.replace('.npy', '_mapping.npy'), mapping)