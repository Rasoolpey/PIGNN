"""
Core data structures for three-phase power grid graph.
Implements three separate phase graphs with coupling matrices.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field


class PhaseType(Enum):
    """Three-phase system phases"""
    A = 'a'
    B = 'b'
    C = 'c'


@dataclass
class Node:
    """Base node representing a single phase of a bus"""
    id: str
    parent_id: str  # Parent component ID (e.g., "Bus_1")
    phase: PhaseType
    node_type: str  # 'bus', 'generator', 'load'
    
    # Electrical properties
    voltage_pu: complex = 1.0 + 0j
    voltage_base_kv: float = 1.0
    
    # Graph properties
    neighbors: List[str] = field(default_factory=list)
    incoming_edges: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    
    # Additional properties
    properties: Dict = field(default_factory=dict)
    
    @property
    def voltage_angle_rad(self) -> float:
        """Get voltage angle in radians"""
        return np.angle(self.voltage_pu)
    
    def add_neighbor(self, neighbor_id: str):
        """Add a neighboring node"""
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
    
    def add_incoming_edge(self, edge_id: str):
        """Add incoming edge"""
        if edge_id not in self.incoming_edges:
            self.incoming_edges.append(edge_id)
    
    def add_outgoing_edge(self, edge_id: str):
        """Add outgoing edge"""
        if edge_id not in self.outgoing_edges:
            self.outgoing_edges.append(edge_id)


@dataclass
class Edge:
    """Base edge representing a single phase of a line/transformer"""
    id: str
    parent_id: str  # Parent component ID (e.g., "Line_1_2")
    phase: PhaseType
    from_node_id: str
    to_node_id: str
    edge_type: str  # 'line', 'transformer'
    
    # Electrical properties (single phase impedance)
    impedance: complex = 0.01 + 0.1j
    
    # Additional properties
    properties: Dict = field(default_factory=dict)
    
    @property
    def admittance(self) -> complex:
        """Get admittance (1/Z)"""
        if abs(self.impedance) > 1e-10:
            return 1.0 / self.impedance
        return 0.0 + 0.0j


@dataclass
class CouplingMatrix:
    """3x3 coupling matrix for inter-phase interactions"""
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=complex))
    coupling_type: str = 'impedance'  # 'impedance' or 'admittance'
    
    def __post_init__(self):
        """Validate matrix dimensions"""
        if self.matrix.shape != (3, 3):
            raise ValueError("Coupling matrix must be 3x3")
    
    def is_symmetric(self) -> bool:
        """Check if matrix is symmetric (passive component)"""
        return np.allclose(self.matrix, self.matrix.T)
    
    def get_mutual_coupling(self, phase1: PhaseType, phase2: PhaseType) -> complex:
        """Get coupling between two phases"""
        i = ord(phase1.value) - ord('a')
        j = ord(phase2.value) - ord('a')
        return self.matrix[i, j]


class PowerGridGraph:
    """Three-phase power grid graph with coupling"""
    
    def __init__(self):
        # Nodes: Dict[component_id] -> Dict[phase] -> Node
        self.nodes: Dict[str, Dict[PhaseType, Node]] = {}
        
        # Edges: Dict[component_id] -> Dict[phase] -> Edge
        self.edges: Dict[str, Dict[PhaseType, Edge]] = {}
        
        # Coupling matrices
        self.node_couplings: Dict[str, CouplingMatrix] = {}
        self.edge_couplings: Dict[str, CouplingMatrix] = {}
        
        # Metadata
        self.base_mva: float = 100.0
        self.frequency_hz: float = 60.0
        self.properties: Dict = {}
    
    def add_node(self, node_id: str, node_type: str, **kwargs) -> str:
        """
        Add a three-phase node component.
        
        Args:
            node_id: Unique identifier for the component
            node_type: Type of node ('bus', 'generator', 'load')
            **kwargs: Additional properties for each phase
        
        Returns:
            node_id: The ID of the added node
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        
        self.nodes[node_id] = {}
        
        # Create node for each phase
        for phase in PhaseType:
            phase_node_id = f"{node_id}_{phase.value}"
            self.nodes[node_id][phase] = Node(
                id=phase_node_id,
                parent_id=node_id,
                phase=phase,
                node_type=node_type,
                **kwargs
            )
        
        # Initialize coupling matrix (default: no coupling)
        self.node_couplings[node_id] = CouplingMatrix()
        
        return node_id
    
    def add_edge(self, edge_id: str, from_node_id: str, to_node_id: str, 
                 edge_type: str, **kwargs) -> str:
        """
        Add a three-phase edge component.
        
        Args:
            edge_id: Unique identifier
            from_node_id: Source node
            to_node_id: Target node
            edge_type: 'line', 'transformer'
            **kwargs: Additional parameters
        """
        if edge_id in self.edges:
            raise ValueError(f"Edge {edge_id} already exists")
        
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Both nodes must exist before adding edge")
        
        self.edges[edge_id] = {}
        
        # Create edge for each phase
        for phase in PhaseType:
            phase_edge_id = f"{edge_id}_{phase.value}"
            self.edges[edge_id][phase] = Edge(
                id=phase_edge_id,
                parent_id=edge_id,
                phase=phase,
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                edge_type=edge_type,
                **kwargs
            )
            
            # Register edge with nodes
            self.nodes[from_node_id][phase].add_outgoing_edge(phase_edge_id)
            self.nodes[to_node_id][phase].add_incoming_edge(phase_edge_id)
        
        # Initialize coupling matrix
        self.edge_couplings[edge_id] = CouplingMatrix()
        
        return edge_id
    
    def set_node_coupling(self, node_id: str, coupling_matrix: np.ndarray):
        """Set coupling matrix for a node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        self.node_couplings[node_id] = CouplingMatrix(coupling_matrix.copy())
    
    def set_edge_coupling(self, edge_id: str, coupling_matrix: np.ndarray):
        """Set coupling matrix for an edge"""
        if edge_id not in self.edges:
            raise ValueError(f"Edge {edge_id} not found")
        
        self.edge_couplings[edge_id] = CouplingMatrix(coupling_matrix.copy())
    
    def add_edge_coupling(self, edge_id: str, coupling_matrix: np.ndarray, coupling_type: str = 'impedance'):
        """Add coupling matrix for an edge"""
        if edge_id not in self.edges:
            raise ValueError(f"Edge {edge_id} not found")
        
        self.edge_couplings[edge_id] = CouplingMatrix(coupling_matrix.copy(), coupling_type)
    
    def get_node(self, node_id: str, phase: Optional[PhaseType] = None):
        """Get node by ID and optionally phase"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        if phase is None:
            return self.nodes[node_id]
        else:
            return self.nodes[node_id][phase]
    
    def get_edge(self, edge_id: str, phase: Optional[PhaseType] = None):
        """Get edge by ID and optionally phase"""
        if edge_id not in self.edges:
            raise ValueError(f"Edge {edge_id} not found")
        
        if phase is None:
            return self.edges[edge_id]
        else:
            return self.edges[edge_id][phase]
    
    def get_neighbors(self, node_id: str, phase: PhaseType) -> List[str]:
        """Get neighboring nodes for a specific phase"""
        node = self.get_node(node_id, phase)
        neighbors = []
        
        # From outgoing edges
        for edge_id in node.outgoing_edges:
            edge = self._find_edge_by_phase_id(edge_id)
            if edge:
                neighbors.append(edge.to_node_id)
        
        # From incoming edges
        for edge_id in node.incoming_edges:
            edge = self._find_edge_by_phase_id(edge_id)
            if edge:
                neighbors.append(edge.from_node_id)
        
        return list(set(neighbors))  # Remove duplicates
    
    def _find_edge_by_phase_id(self, phase_edge_id: str) -> Optional[Edge]:
        """Find edge by its phase-specific ID"""
        for edge_id, phase_edges in self.edges.items():
            for phase, edge in phase_edges.items():
                if edge.id == phase_edge_id:
                    return edge
        return None
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        
        # Count by type
        node_types = {}
        edge_types = {}
        
        for node_id, phases in self.nodes.items():
            node_type = phases[PhaseType.A].node_type
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for edge_id, phases in self.edges.items():
            edge_type = phases[PhaseType.A].edge_type
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'n_components': n_nodes,
            'n_branches': n_edges,
            'n_phases': 3,
            'node_types': node_types,
            'edge_types': edge_types,
            'base_mva': self.base_mva,
            'frequency_hz': self.frequency_hz
        }