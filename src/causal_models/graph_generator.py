"""Causal graph generation for synthetic datasets."""

import networkx as nx
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

from ..utils.distributions import DistributionSampler

logger = logging.getLogger(__name__)


class CausalGraphGenerator:
    """Generate causal graphs for synthetic datasets.
    
    This class implements various methods for generating directed acyclic
    graphs (DAGs) that represent causal relationships in synthetic data.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the graph generator.
        
        Args:
            config: Graph configuration parameters
            seed: Random seed for reproducibility
        """
        self.config = config
        self.sampler = DistributionSampler(seed)
        self.rng = np.random.RandomState(seed)
        logger.info("Initialized CausalGraphGenerator")
    
    def generate_graph(self, n_nodes: Optional[int] = None) -> nx.DiGraph:
        """Generate a random directed acyclic graph.
        
        Args:
            n_nodes: Number of nodes (sampled if None)
            
        Returns:
            Generated DAG
        """
        if n_nodes is None:
            n_nodes = self._sample_n_nodes()
        
        graph_type = self.config.get('graph_type', 'scale_free')
        logger.debug(f"Generating {graph_type} graph with {n_nodes} nodes")
        
        if graph_type == 'scale_free':
            graph = self._generate_scale_free_dag(n_nodes)
        elif graph_type == 'erdos_renyi':
            graph = self._generate_erdos_renyi_dag(n_nodes)
        elif graph_type == 'barabasi_albert':
            graph = self._generate_barabasi_albert_dag(n_nodes)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        # Ensure graph is connected if required
        if self.config.get('ensure_connected', True):
            graph = self._ensure_connected(graph)
        
        # Add metadata to graph
        graph.graph['n_nodes'] = n_nodes
        graph.graph['type'] = graph_type
        graph.graph['n_edges'] = graph.number_of_edges()
        
        logger.info(f"Generated {graph_type} DAG with {n_nodes} nodes and {graph.number_of_edges()} edges")
        
        return graph
    
    def _sample_n_nodes(self) -> int:
        """Sample number of nodes from configuration.
        
        Returns:
            Number of nodes
        """
        node_config = self.config.get('n_nodes', {})
        distribution = node_config.get('distribution', 'uniform')
        
        if distribution == 'log_uniform':
            n = self.sampler.sample_log_uniform(
                node_config.get('min', 5),
                node_config.get('max', 100)
            )
            return int(n[0])
        elif distribution == 'uniform':
            return self.rng.randint(
                node_config.get('min', 5),
                node_config.get('max', 100) + 1
            )
        else:
            raise ValueError(f"Unknown distribution for n_nodes: {distribution}")
    
    def _generate_scale_free_dag(self, n_nodes: int) -> nx.DiGraph:
        """Generate scale-free DAG using growing network with redirection.
        
        This implements the growing network with redirection model from
        Krapivsky & Redner (2001), which generates scale-free networks.
        
        Args:
            n_nodes: Number of nodes
            
        Returns:
            Scale-free DAG
        """
        G = nx.DiGraph()
        G.add_node(0)
        
        # Sample redirection probability
        redir_config = self.config.get('redirection_probability', {})
        if redir_config.get('distribution') == 'gamma':
            p_redirect = self.rng.gamma(
                redir_config.get('alpha', 2.0),
                1.0 / redir_config.get('beta', 1.0)
            )
            p_redirect = np.clip(p_redirect, 0, 1)
        else:
            p_redirect = redir_config.get('value', 0.5)
        
        logger.debug(f"Using redirection probability: {p_redirect:.3f}")
        
        # Growing network process
        for new_node in range(1, n_nodes):
            G.add_node(new_node)
            
            # Determine number of edges to add (typically 1 for DAG)
            n_edges = 1
            
            for _ in range(n_edges):
                if G.number_of_edges() > 0 and self.rng.random() > p_redirect:
                    # Preferential attachment
                    in_degrees = dict(G.in_degree())
                    total_degree = sum(in_degrees.values())
                    
                    if total_degree > 0:
                        # Calculate probabilities proportional to in-degree
                        nodes = list(range(new_node))
                        probs = np.array([in_degrees.get(i, 0) + 1 for i in nodes])
                        probs = probs / probs.sum()
                        target = self.rng.choice(nodes, p=probs)
                    else:
                        target = self.rng.randint(0, new_node)
                else:
                    # Random attachment (redirection)
                    target = self.rng.randint(0, new_node)
                
                # Add edge maintaining DAG property
                # Always point from higher to lower node index to avoid cycles
                if new_node > target:
                    if not G.has_edge(new_node, target):
                        G.add_edge(new_node, target)
                else:
                    if not G.has_edge(target, new_node):
                        G.add_edge(target, new_node)
        
        return self._ensure_dag(G)
    
    def _generate_erdos_renyi_dag(self, n_nodes: int) -> nx.DiGraph:
        """Generate Erdős-Rényi random DAG.
        
        Args:
            n_nodes: Number of nodes
            
        Returns:
            Random DAG
        """
        # Edge probability to maintain reasonable density
        p_edge = min(2.0 / n_nodes, 0.3)
        
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        
        # Add edges only from higher to lower indices (ensures DAG)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if self.rng.random() < p_edge:
                    G.add_edge(j, i)  # Edge from j to i where j > i
        
        return G
    
    def _generate_barabasi_albert_dag(self, n_nodes: int) -> nx.DiGraph:
        """Generate Barabási-Albert preferential attachment DAG.
        
        Args:
            n_nodes: Number of nodes
            
        Returns:
            BA model DAG
        """
        m = min(2, n_nodes - 1)  # Number of edges to attach from new node
        
        # Start with initial connected nodes
        G = nx.DiGraph()
        G.add_nodes_from(range(m))
        
        # Add initial edges
        for i in range(m - 1):
            G.add_edge(i + 1, i)
        
        # Preferential attachment process
        for new_node in range(m, n_nodes):
            G.add_node(new_node)
            
            # Select m targets based on degree
            degrees = dict(G.degree())
            if sum(degrees.values()) > 0:
                nodes = list(range(new_node))
                probs = np.array([degrees.get(i, 0) + 1 for i in nodes])
                probs = probs / probs.sum()
                
                targets = self.rng.choice(
                    nodes, size=min(m, new_node), 
                    replace=False, p=probs
                )
                
                for target in targets:
                    # Add edge maintaining DAG property
                    if not G.has_edge(new_node, target):
                        G.add_edge(new_node, target)
        
        return self._ensure_dag(G)
    
    def _ensure_dag(self, G: nx.DiGraph) -> nx.DiGraph:
        """Ensure graph is a DAG by removing cycles.
        
        Args:
            G: Input graph
            
        Returns:
            DAG version of the graph
        """
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                # Remove the edge that creates the cycle
                G.remove_edge(*cycle[0][:2])
                logger.debug(f"Removed cycle edge: {cycle[0][:2]}")
            except nx.NetworkXNoCycle:
                break
        
        return G
    
    def _ensure_connected(self, G: nx.DiGraph) -> nx.DiGraph:
        """Ensure graph is weakly connected.
        
        Args:
            G: Input DAG
            
        Returns:
            Connected DAG
        """
        if nx.is_weakly_connected(G):
            return G
        
        # Get weakly connected components
        components = list(nx.weakly_connected_components(G))
        if len(components) <= 1:
            return G
        
        logger.debug(f"Connecting {len(components)} components")
        
        # Connect components maintaining DAG property
        main_component = max(components, key=len)
        for component in components:
            if component == main_component:
                continue
            
            # Connect component to main component
            source = min(component)
            target = min(main_component)
            
            if source < target:
                G.add_edge(target, source)
            else:
                G.add_edge(source, target)
        
        return G
    
    def visualize_graph(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Get graph statistics for visualization.
        
        Args:
            G: Graph to analyze
            
        Returns:
            Dictionary of graph statistics
        """
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'is_connected': nx.is_weakly_connected(G),
            'avg_degree': np.mean([d for _, d in G.degree()]),
            'max_in_degree': max(dict(G.in_degree()).values()) if G.nodes else 0,
            'max_out_degree': max(dict(G.out_degree()).values()) if G.nodes else 0,
        }
        
        return stats
