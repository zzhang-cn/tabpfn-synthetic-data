"""Motif-based causal graph generation using Pearl's fundamental patterns."""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CausalMotif(Enum):
    """Pearl's three fundamental causal motifs."""
    CHAIN = "chain"  # A → B → C (mediation)
    FORK = "fork"    # A ← B → C (common cause/confounder)
    COLLIDER = "collider"  # A → B ← C (common effect)


class MotifBasedGraphGenerator:
    """Generate causal graphs using hierarchical composition of Pearl's motifs.
    
    This approach creates more realistic causal structures by:
    1. Building from fundamental causal patterns
    2. Composing motifs hierarchically
    3. Maintaining desired graph properties (scale-free, small-world, etc.)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.motif_counts = {motif: 0 for motif in CausalMotif}
        
    def generate_graph(self, 
                      n_nodes: int,
                      target_density: float = 0.1,
                      motif_probs: Optional[Dict[CausalMotif, float]] = None) -> nx.DiGraph:
        """Generate a causal graph using motif composition.
        
        Args:
            n_nodes: Number of nodes in the graph
            target_density: Target edge density (0-1)
            motif_probs: Probability of each motif type
            
        Returns:
            Generated causal DAG
        """
        if motif_probs is None:
            # Default: balanced mix with slight preference for chains
            motif_probs = {
                CausalMotif.CHAIN: 0.4,
                CausalMotif.FORK: 0.3,
                CausalMotif.COLLIDER: 0.3
            }
        
        # Initialize graph with nodes
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        
        # Reset motif counts
        self.motif_counts = {motif: 0 for motif in CausalMotif}
        
        # Build graph by adding motifs
        target_edges = int(n_nodes * (n_nodes - 1) * target_density / 2)
        
        while G.number_of_edges() < target_edges and self._can_add_motif(G, n_nodes):
            motif_type = self._sample_motif(motif_probs)
            self._add_motif(G, motif_type, n_nodes)
        
        # Ensure DAG property
        G = self._ensure_dag(G)
        
        # Add graph metadata
        G.graph['motif_counts'] = self.motif_counts.copy()
        G.graph['generation_method'] = 'motif_based'
        
        logger.info(f"Generated graph with {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges using motifs: {self.motif_counts}")
        
        return G
    
    def _sample_motif(self, motif_probs: Dict[CausalMotif, float]) -> CausalMotif:
        """Sample a motif type based on probabilities.
        
        Args:
            motif_probs: Probability distribution over motif types
            
        Returns:
            Selected motif type
        """
        motifs = list(motif_probs.keys())
        probs = list(motif_probs.values())
        probs = np.array(probs) / np.sum(probs)  # Normalize
        
        return self.rng.choice(motifs, p=probs)
    
    def _add_motif(self, G: nx.DiGraph, motif_type: CausalMotif, n_nodes: int):
        """Add a motif to the graph.
        
        Args:
            G: Current graph
            motif_type: Type of motif to add
            n_nodes: Total number of nodes
        """
        if motif_type == CausalMotif.CHAIN:
            self._add_chain(G, n_nodes)
        elif motif_type == CausalMotif.FORK:
            self._add_fork(G, n_nodes)
        elif motif_type == CausalMotif.COLLIDER:
            self._add_collider(G, n_nodes)
        
        self.motif_counts[motif_type] += 1
    
    def _add_chain(self, G: nx.DiGraph, n_nodes: int):
        """Add a chain motif: A → B → C.
        
        This represents mediation or sequential causation.
        """
        # Sample three nodes
        nodes = self._sample_nodes_for_motif(G, n_nodes, 3)
        if nodes is None:
            return
        
        a, b, c = nodes
        
        # Add edges maintaining temporal order (lower index → higher index for DAG)
        if a < b < c:
            G.add_edge(a, b)
            G.add_edge(b, c)
        elif a < c < b:
            G.add_edge(a, c)
            G.add_edge(c, b)
        elif b < a < c:
            G.add_edge(b, a)
            G.add_edge(a, c)
        elif b < c < a:
            G.add_edge(b, c)
            G.add_edge(c, a)
        elif c < a < b:
            G.add_edge(c, a)
            G.add_edge(a, b)
        else:  # c < b < a
            G.add_edge(c, b)
            G.add_edge(b, a)
    
    def _add_fork(self, G: nx.DiGraph, n_nodes: int):
        """Add a fork motif: A ← B → C.
        
        This represents a common cause or confounder.
        """
        nodes = self._sample_nodes_for_motif(G, n_nodes, 3)
        if nodes is None:
            return
        
        # B is the common cause
        nodes_sorted = sorted(nodes)
        
        # Prefer middle node as common cause for better balance
        if self.rng.random() < 0.6:
            b = nodes_sorted[1]
            a, c = nodes_sorted[0], nodes_sorted[2]
        else:
            # Random assignment
            b = self.rng.choice(nodes)
            remaining = [n for n in nodes if n != b]
            a, c = remaining[0], remaining[1]
        
        # Add edges from common cause
        if b < a:
            G.add_edge(b, a)
        else:
            G.add_edge(a, b)  # Reverse to maintain DAG
            
        if b < c:
            G.add_edge(b, c)
        else:
            G.add_edge(c, b)  # Reverse to maintain DAG
    
    def _add_collider(self, G: nx.DiGraph, n_nodes: int):
        """Add a collider motif: A → B ← C.
        
        This represents a common effect or collision node.
        """
        nodes = self._sample_nodes_for_motif(G, n_nodes, 3)
        if nodes is None:
            return
        
        nodes_sorted = sorted(nodes)
        
        # Prefer middle node as collider
        if self.rng.random() < 0.6:
            b = nodes_sorted[1]
            a, c = nodes_sorted[0], nodes_sorted[2]
        else:
            # Random assignment
            b = self.rng.choice(nodes)
            remaining = [n for n in nodes if n != b]
            a, c = remaining[0], remaining[1]
        
        # Add edges to common effect
        if a < b:
            G.add_edge(a, b)
        else:
            G.add_edge(b, a)  # Reverse to maintain DAG
            
        if c < b:
            G.add_edge(c, b)
        else:
            G.add_edge(b, c)  # Reverse to maintain DAG
    
    def _sample_nodes_for_motif(self, G: nx.DiGraph, n_nodes: int, 
                                motif_size: int) -> Optional[List[int]]:
        """Sample nodes for a motif, preferring to connect to existing structure.
        
        Args:
            G: Current graph
            n_nodes: Total nodes
            motif_size: Number of nodes needed
            
        Returns:
            List of node indices or None if cannot sample
        """
        if n_nodes < motif_size:
            return None
        
        # Strategy: Mix of connecting to existing nodes and adding isolated nodes
        existing_nodes = list(G.nodes())
        
        if G.number_of_edges() == 0:
            # First motif: random selection
            return self.rng.choice(existing_nodes, motif_size, replace=False).tolist()
        
        # Prefer to connect to existing structure (creates more realistic graphs)
        if self.rng.random() < 0.7 and G.number_of_edges() > 0:
            # Connect to existing structure
            connected_nodes = [n for n in existing_nodes if G.degree(n) > 0]
            isolated_nodes = [n for n in existing_nodes if G.degree(n) == 0]
            
            selected = []
            
            # Include at least one connected node
            if connected_nodes:
                n_connected = min(len(connected_nodes), self.rng.randint(1, motif_size))
                selected.extend(self.rng.choice(connected_nodes, n_connected, replace=False))
            
            # Fill remaining with isolated or random nodes
            remaining_needed = motif_size - len(selected)
            if remaining_needed > 0:
                candidates = isolated_nodes if isolated_nodes else existing_nodes
                candidates = [n for n in candidates if n not in selected]
                
                if len(candidates) >= remaining_needed:
                    selected.extend(self.rng.choice(candidates, remaining_needed, replace=False))
                else:
                    return None
            
            return selected
        else:
            # Random selection
            return self.rng.choice(existing_nodes, motif_size, replace=False).tolist()
    
    def _can_add_motif(self, G: nx.DiGraph, n_nodes: int) -> bool:
        """Check if we can add another motif.
        
        Args:
            G: Current graph
            n_nodes: Total nodes
            
        Returns:
            Whether another motif can be added
        """
        # Need at least 3 nodes for any motif
        if n_nodes < 3:
            return False
        
        # Check if we've reached maximum density
        max_edges = n_nodes * (n_nodes - 1) / 2
        if G.number_of_edges() >= max_edges * 0.5:  # Cap at 50% density
            return False
        
        return True
    
    def _ensure_dag(self, G: nx.DiGraph) -> nx.DiGraph:
        """Ensure the graph is a DAG by removing cycles.
        
        Args:
            G: Input graph
            
        Returns:
            DAG version of the graph
        """
        # For motif-based construction with careful edge direction,
        # cycles should be rare, but we check anyway
        cycles_removed = 0
        
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                # Remove the edge that completes the cycle
                G.remove_edge(*cycle[0][:2])
                cycles_removed += 1
            except nx.NetworkXNoCycle:
                break
        
        if cycles_removed > 0:
            logger.warning(f"Removed {cycles_removed} edges to ensure DAG property")
        
        return G
    
    def generate_hierarchical_graph(self, n_nodes: int, n_levels: int = 3) -> nx.DiGraph:
        """Generate a hierarchical graph with nested motif structures.
        
        This creates more complex, realistic causal structures by:
        1. Creating clusters of motifs
        2. Connecting clusters with higher-level motifs
        3. Maintaining scale-free or small-world properties
        
        Args:
            n_nodes: Total number of nodes
            n_levels: Number of hierarchical levels
            
        Returns:
            Hierarchical causal graph
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        
        # Divide nodes into clusters
        nodes_per_cluster = max(3, n_nodes // (2 ** n_levels))
        clusters = []
        
        for i in range(0, n_nodes, nodes_per_cluster):
            cluster = list(range(i, min(i + nodes_per_cluster, n_nodes)))
            if len(cluster) >= 3:
                clusters.append(cluster)
        
        # Build motifs within clusters (level 1)
        for cluster in clusters:
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(cluster)
            
            # Add motifs within cluster
            n_motifs = len(cluster) // 3
            for _ in range(n_motifs):
                motif_nodes = self.rng.choice(cluster, 3, replace=False)
                motif_type = self.rng.choice(list(CausalMotif))
                self._add_motif_to_nodes(G, motif_nodes, motif_type)
        
        # Connect clusters with higher-level motifs (level 2+)
        if len(clusters) >= 2:
            for level in range(2, min(n_levels + 1, len(clusters))):
                # Select representative nodes from each cluster
                representatives = []
                for cluster in clusters[:3 * (len(clusters) // 3)]:  # Groups of 3
                    # Prefer hub nodes (high degree) as representatives
                    degrees = [(n, G.degree(n)) for n in cluster]
                    degrees.sort(key=lambda x: x[1], reverse=True)
                    representatives.append(degrees[0][0] if degrees else cluster[0])
                
                # Connect representatives with motifs
                for i in range(0, len(representatives) - 2, 3):
                    motif_nodes = representatives[i:i+3]
                    motif_type = self.rng.choice([CausalMotif.CHAIN, CausalMotif.FORK])
                    self._add_motif_to_nodes(G, motif_nodes, motif_type)
        
        return self._ensure_dag(G)
    
    def _add_motif_to_nodes(self, G: nx.DiGraph, nodes: List[int], motif_type: CausalMotif):
        """Add a specific motif to specific nodes.
        
        Args:
            G: Graph to modify
            nodes: Exactly 3 nodes for the motif
            motif_type: Type of motif to create
        """
        if len(nodes) != 3:
            return
        
        a, b, c = sorted(nodes)
        
        if motif_type == CausalMotif.CHAIN:
            G.add_edge(a, b)
            G.add_edge(b, c)
        elif motif_type == CausalMotif.FORK:
            G.add_edge(b, a)  # b is common cause
            G.add_edge(b, c)
        elif motif_type == CausalMotif.COLLIDER:
            G.add_edge(a, b)  # b is common effect
            G.add_edge(c, b)
    
    def analyze_motifs(self, G: nx.DiGraph) -> Dict[CausalMotif, int]:
        """Analyze the motif composition of an existing graph.
        
        Args:
            G: Graph to analyze
            
        Returns:
            Count of each motif type found
        """
        motif_counts = {motif: 0 for motif in CausalMotif}
        
        # Check all triples of nodes
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    triple = [nodes[i], nodes[j], nodes[k]]
                    motif = self._identify_motif(G, triple)
                    if motif:
                        motif_counts[motif] += 1
        
        return motif_counts
    
    def _identify_motif(self, G: nx.DiGraph, nodes: List[int]) -> Optional[CausalMotif]:
        """Identify the motif type for three nodes.
        
        Args:
            G: Graph containing the nodes
            nodes: Three nodes to check
            
        Returns:
            Motif type or None if no motif
        """
        a, b, c = nodes
        edges = []
        
        # Check all possible edges
        for n1 in nodes:
            for n2 in nodes:
                if n1 != n2 and G.has_edge(n1, n2):
                    edges.append((n1, n2))
        
        if len(edges) != 2:
            return None  # Not a basic motif
        
        # Analyze edge pattern
        edge_set = set(edges)
        
        # Check for chain
        for mid in nodes:
            others = [n for n in nodes if n != mid]
            if (others[0], mid) in edge_set and (mid, others[1]) in edge_set:
                return CausalMotif.CHAIN
            if (others[1], mid) in edge_set and (mid, others[0]) in edge_set:
                return CausalMotif.CHAIN
        
        # Check for fork (common cause)
        for cause in nodes:
            others = [n for n in nodes if n != cause]
            if (cause, others[0]) in edge_set and (cause, others[1]) in edge_set:
                return CausalMotif.FORK
        
        # Check for collider (common effect)
        for effect in nodes:
            others = [n for n in nodes if n != effect]
            if (others[0], effect) in edge_set and (others[1], effect) in edge_set:
                return CausalMotif.COLLIDER
        
        return None
