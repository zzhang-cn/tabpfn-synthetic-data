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
    
    def sample(self, n_samples: int, graph: nx.DiGraph) -> np.ndarray:
        """Sample initialization data for root nodes.
        
        Args:
            n_samples: Number of samples
            graph: Causal graph
            
        Returns:
            Initialization data array (n_samples, n_root_nodes)
        """
        # Get root nodes (no parents)
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        n_roots = len(root_nodes)
        
        if n_roots == 0:
            logger.warning("No root nodes found in graph")
            return np.empty((n_samples, 0))
        
        logger.debug(f"Sampling initialization for {n_roots} root nodes")
        
        # Sample distribution type
        dist_types = self.config.get('distribution_types', ['normal', 'uniform', 'mixed'])
        dist_type = self.rng.choice(dist_types)
        
        # Sample base data
        if dist_type == 'normal':
            data = self._sample_normal(n_samples, n_roots)
        elif dist_type == 'uniform':
            data = self._sample_uniform(n_samples, n_roots)
        elif dist_type == 'mixed':
            data = self._sample_mixed(n_samples, n_roots)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        # Apply non-independence if configured
        if self.config.get('non_independence.enabled', True):
            data = self._apply_non_independence(data)
        
        return data
    
    def _sample_normal(self, n_samples: int, n_features: int) -> np.ndarray:
        """Sample from normal distribution.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Normal samples
        """
        params = self.config.get('normal', {})
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        
        return self.rng.normal(mean, std, (n_samples, n_features))
    
    def _sample_uniform(self, n_samples: int, n_features: int) -> np.ndarray:
        """Sample from uniform distribution.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Uniform samples
        """
        params = self.config.get('uniform', {})
        low = params.get('low', -1)
        high = params.get('high', 1)
        
        return self.rng.uniform(low, high, (n_samples, n_features))
    
    def _sample_mixed(self, n_samples: int, n_features: int) -> np.ndarray:
        """Sample from mixed distributions.
        
        Each feature gets a random distribution type.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Mixed samples
        """
        data = np.zeros((n_samples, n_f
