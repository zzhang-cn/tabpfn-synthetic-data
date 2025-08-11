"""Initialization data sampling for SCM root nodes."""

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
import logging

from ..utils.distributions import DistributionSampler

logger = logging.getLogger(__name__)


class InitializationSampler:
    """Sample initialization data for SCM root nodes.
    
    This class handles the generation of initial values for root nodes
    in the causal graph, with support for various distributions and
    non-independence between samples.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the sampler.
        
        Args:
            config: Initialization configuration
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.dist_sampler = DistributionSampler(seed)
        
        logger.debug("Initialized InitializationSampler")
    
    def sample(self, n_samples: int, graph: nx.DiGraph, vector_dim: int = 8) -> np.ndarray:
        """Sample initialization data for root nodes.
        
        Args:
            n_samples: Number of samples
            graph: Causal graph
            vector_dim: Dimension of node vectors
            
        Returns:
            Initialization data array (n_samples, n_root_nodes, vector_dim)
        """
        # Get root nodes (no parents)
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        n_roots = len(root_nodes)
        
        if n_roots == 0:
            logger.warning("No root nodes found in graph")
            return np.empty((n_samples, 0, vector_dim))
        
        logger.debug(f"Sampling initialization for {n_roots} root nodes with vector_dim={vector_dim}")
        
        # Sample distribution type
        dist_types = self.config.get('distribution_types', ['normal', 'uniform', 'mixed'])
        dist_type = self.rng.choice(dist_types)
        
        # Sample base data
        if dist_type == 'normal':
            data = self._sample_normal(n_samples, n_roots, vector_dim)
        elif dist_type == 'uniform':
            data = self._sample_uniform(n_samples, n_roots, vector_dim)
        elif dist_type == 'mixed':
            data = self._sample_mixed(n_samples, n_roots, vector_dim)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        # Apply non-independence if configured
        #if self.config.get('non_independence.enabled', True):
            #data = self._apply_non_independence(data)
        
        return data
    
    def _sample_normal(self, n_samples: int, n_features: int, vector_dim: int) -> np.ndarray:
        """Sample from normal distribution.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            vector_dim: Vector dimension
            
        Returns:
            Normal samples (n_samples, n_features, vector_dim)
        """
        params = self.config.get('normal', {})
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        
        return self.rng.normal(mean, std, (n_samples, n_features, vector_dim))
    
    def _sample_uniform(self, n_samples: int, n_features: int, vector_dim: int) -> np.ndarray:
        """Sample from uniform distribution.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            vector_dim: Vector dimension
            
        Returns:
            Uniform samples (n_samples, n_features, vector_dim)
        """
        params = self.config.get('uniform', {})
        low = params.get('low', -1)
        high = params.get('high', 1)
        
        return self.rng.uniform(low, high, (n_samples, n_features, vector_dim))
    
    def _sample_mixed(self, n_samples: int, n_features: int, vector_dim: int) -> np.ndarray:
        """Sample from mixed distributions.
        
        Each feature gets a random distribution type.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            vector_dim: Vector dimension
            
        Returns:
            Mixed samples (n_samples, n_features, vector_dim)
        """
        data = np.zeros((n_samples, n_features, vector_dim))
        
        for j in range(n_features):
            if self.rng.random() < 0.5:
                # Normal
                data[:, j, :] = self._sample_normal(n_samples, 1, vector_dim)[:, 0, :]
            else:
                # Uniform
                data[:, j, :] = self._sample_uniform(n_samples, 1, vector_dim)[:, 0, :]
        
        return data
    
    def _apply_non_independence(self, data: np.ndarray) -> np.ndarray:
        """Apply non-independence between samples.
        
        This creates correlations between samples by mixing them with
        prototype samples, following the TabPFN paper approach.
        
        Args:
            data: Independent samples (n_samples, n_features, vector_dim)
            
        Returns:
            Non-independent samples (n_samples, n_features, vector_dim)
        """
        n_samples, n_features, vector_dim = data.shape
        
        # Get non-independence parameters
        ni_config = self.config.get('non_independence', {})
        prototype_fraction = ni_config.get('prototype_fraction', 0.3)
        temperature = ni_config.get('temperature', 1.0)
        
        # Number of prototypes
        n_prototypes = max(1, int(n_samples * prototype_fraction))
        
        # Select prototype samples
        prototype_idx = self.rng.choice(n_samples, n_prototypes, replace=False)
        prototypes = data[prototype_idx].copy()  # (n_prototypes, n_features, vector_dim)
        
        # Mix samples with prototypes
        mixed_data = np.zeros_like(data)
        
        for i in range(n_samples):
            if i in prototype_idx:
                # Prototypes remain unchanged
                mixed_data[i] = data[i]
            else:
                # Sample mixing weights from Dirichlet distribution
                # Temperature controls concentration
                alpha = np.ones(n_prototypes) / temperature
                weights = self.rng.dirichlet(alpha)  # (n_prototypes,)
                
                # Mix with prototypes
                # weights @ prototypes gives (n_features, vector_dim)
                mixed_data[i] = np.tensordot(weights, prototypes, axes=([0], [0]))
                
                # Add some noise to maintain variation
                noise_scale = 0.1
                mixed_data[i] += self.rng.normal(0, noise_scale, (n_features, vector_dim))
        
        return mixed_data
    
    def sample_with_correlation(self, n_samples: int, n_features: int,
                               correlation: float = 0.5) -> np.ndarray:
        """Sample data with specified correlation between features.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            correlation: Correlation strength (0-1)
            
        Returns:
            Correlated samples
        """
        # Create correlation matrix
        corr_matrix = np.eye(n_features)
        
        # Add off-diagonal correlations
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Random correlation up to specified strength
                corr = self.rng.uniform(-correlation, correlation)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # Ensure positive definite
        min_eig = np.min(np.linalg.eigvals(corr_matrix))
        if min_eig < 0:
            corr_matrix -= 1.1 * min_eig * np.eye(n_features)
        
        # Generate correlated normal samples
        mean = np.zeros(n_features)
        samples = self.rng.multivariate_normal(mean, corr_matrix, n_samples)
        
        return samples
    
    def sample_clustered(self, n_samples: int, n_features: int,
                        n_clusters: int = 3) -> np.ndarray:
        """Sample data with cluster structure.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_clusters: Number of clusters
            
        Returns:
            Clustered samples
        """
        data = np.zeros((n_samples, n_features))
        
        # Assign samples to clusters
        cluster_assignments = self.rng.randint(0, n_clusters, n_samples)
        
        # Generate cluster centers
        centers = self.rng.normal(0, 2, (n_clusters, n_features))
        
        # Generate samples around centers
        for i in range(n_samples):
            cluster = cluster_assignments[i]
            center = centers[cluster]
            
            # Add noise around center
            noise = self.rng.normal(0, 0.5, n_features)
            data[i] = center + noise
        
        return data
