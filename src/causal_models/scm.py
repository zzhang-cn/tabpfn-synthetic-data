"""Structural Causal Model implementation."""

import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class StructuralCausalModel:
    """Structural Causal Model for synthetic data generation.
    
    An SCM consists of:
    - A causal graph (DAG)
    - Structural assignments (functions) for each node
    - Noise distributions for each node
    """
    
    def __init__(self, graph: nx.DiGraph, edge_functions: Dict[Tuple, Any],
                 noise_params: Optional[Dict] = None):
        """Initialize the SCM.
        
        Args:
            graph: Causal DAG
            edge_functions: Dictionary mapping edges to functions
            noise_params: Parameters for noise generation
        """
        self.graph = graph
        self.edge_functions = edge_functions
        self.noise_params = noise_params or {}
        
        # Topological order for propagation
        self.topological_order = list(nx.topological_sort(graph))
        
        # Node types (feature, target, latent)
        self.node_types = {}
        
        logger.debug(f"Initialized SCM with {graph.number_of_nodes()} nodes")
    
    def sample(self, n_samples: int, initialization_data: Optional[np.ndarray] = None, 
               vector_dim: int = 8) -> Dict[int, np.ndarray]:
        """Sample data from the SCM.
        
        Args:
            n_samples: Number of samples to generate
            initialization_data: Initial values for root nodes (n_samples, n_root_nodes, vector_dim)
            vector_dim: Dimension of node vectors
            
        Returns:
            Dictionary mapping node indices to data arrays (n_samples, vector_dim)
        """
        # Initialize data storage
        node_data = {}
        
        # Get root nodes (nodes with no parents)
        root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        # Initialize root nodes
        if initialization_data is not None:
            # Use provided initialization
            for i, node in enumerate(root_nodes):
                if i < initialization_data.shape[1]:
                    node_data[node] = initialization_data[:, i, :]  # (n_samples, vector_dim)
                else:
                    node_data[node] = self._sample_noise(node, n_samples, vector_dim)
        else:
            # Sample from noise distributions
            for node in root_nodes:
                node_data[node] = self._sample_noise(node, n_samples, vector_dim)
        
        # Propagate through graph in topological order
        for node in self.topological_order:
            if node in root_nodes:
                continue  # Already initialized
            
            # Get parent nodes
            parents = list(self.graph.predecessors(node))
            
            if not parents:
                # No parents, sample from noise
                node_data[node] = self._sample_noise(node, n_samples, vector_dim)
            else:
                # Compute from parents
                parent_data = [node_data[p] for p in parents]
                
                # Initialize with zeros
                node_value = np.zeros((n_samples, vector_dim))
                
                # Apply edge functions
                for parent in parents:
                    edge = (parent, node)
                    if edge in self.edge_functions:
                        func = self.edge_functions[edge]
                        parent_value = node_data[parent]  # (n_samples, vector_dim)
                        
                        # Apply function
                        contribution = func(parent_value)  # (n_samples, vector_dim)
                        node_value = node_value + contribution
                
                # Add noise
                noise = self._sample_noise(node, n_samples, vector_dim)
                node_data[node] = node_value + noise
        
        return node_data
    
    def _sample_noise(self, node: int, n_samples: int, vector_dim: int = 8) -> np.ndarray:
        """Sample noise for a node.
        
        Args:
            node: Node index
            n_samples: Number of samples
            vector_dim: Vector dimension
            
        Returns:
            Noise samples (n_samples, vector_dim)
        """
        # Get noise parameters for this node
        params = self.noise_params.get(node, {})
        noise_type = params.get('type', 'normal')
        
        if noise_type == 'normal':
            mean = params.get('mean', 0)
            std = params.get('std', 0.1)
            return np.random.normal(mean, std, (n_samples, vector_dim))
        elif noise_type == 'uniform':
            low = params.get('low', -0.1)
            high = params.get('high', 0.1)
            return np.random.uniform(low, high, (n_samples, vector_dim))
        elif noise_type == 'zero':
            return np.zeros((n_samples, vector_dim))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def set_node_types(self, feature_nodes: List[int], target_nodes: List[int]):
        """Set node types for features and targets.
        
        Args:
            feature_nodes: Nodes to use as features
            target_nodes: Nodes to use as targets
        """
        for node in self.graph.nodes():
            if node in feature_nodes:
                self.node_types[node] = 'feature'
            elif node in target_nodes:
                self.node_types[node] = 'target'
            else:
                self.node_types[node] = 'latent'
        
        logger.debug(f"Set {len(feature_nodes)} feature nodes and {len(target_nodes)} target nodes")
    
    def extract_dataset(self, node_data: Dict[int, np.ndarray],
                       feature_nodes: Optional[List[int]] = None,
                       target_nodes: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature and target arrays from node data.
        
        Args:
            node_data: Dictionary of node data
            feature_nodes: Nodes to use as features
            target_nodes: Nodes to use as targets
            
        Returns:
            Tuple of (features, targets)
        """
        if feature_nodes is None:
            feature_nodes = [n for n, t in self.node_types.items() if t == 'feature']
        if target_nodes is None:
            target_nodes = [n for n, t in self.node_types.items() if t == 'target']
        
        # Stack features
        if feature_nodes:
            features = np.column_stack([node_data[n] for n in feature_nodes])
        else:
            features = None
        
        # Stack targets
        if target_nodes:
            if len(target_nodes) == 1:
                targets = node_data[target_nodes[0]]
            else:
                targets = np.column_stack([node_data[n] for n in target_nodes])
        else:
            targets = None
        
        return features, targets
    
    def get_causal_parents(self, node: int) -> List[int]:
        """Get causal parents of a node.
        
        Args:
            node: Node index
            
        Returns:
            List of parent node indices
        """
        return list(self.graph.predecessors(node))
    
    def get_causal_children(self, node: int) -> List[int]:
        """Get causal children of a node.
        
        Args:
            node: Node index
            
        Returns:
            List of child node indices
        """
        return list(self.graph.successors(node))
    
    def get_markov_blanket(self, node: int) -> List[int]:
        """Get Markov blanket of a node.
        
        The Markov blanket consists of:
        - Parents
        - Children  
        - Parents of children (co-parents)
        
        Args:
            node: Node index
            
        Returns:
            List of nodes in Markov blanket
        """
        blanket = set()
        
        # Add parents
        blanket.update(self.graph.predecessors(node))
        
        # Add children
        children = list(self.graph.successors(node))
        blanket.update(children)
        
        # Add co-parents (parents of children)
        for child in children:
            blanket.update(self.graph.predecessors(child))
        
        # Remove the node itself if present
        blanket.discard(node)
        
        return list(blanket)
