"""Discretization edge for creating categorical features."""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .edge_functions import EdgeFunction

logger = logging.getLogger(__name__)


class DiscretizationEdge(EdgeFunction):
    """Edge that discretizes continuous values into categories."""
    
    def __init__(self, thresholds: np.ndarray, embeddings: Optional[np.ndarray] = None):
        """Initialize discretization edge.
        
        Args:
            thresholds: Threshold values for binning
            embeddings: Optional embedding vectors for each category
        """
        self.thresholds = np.sort(thresholds)
        self.n_categories = len(thresholds) + 1
        
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            # Default: one-hot style embeddings
            self.embeddings = np.eye(self.n_categories)
        
        logger.debug(f"Created discretization edge with {self.n_categories} categories")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply discretization.
        
        Args:
            x: Input continuous values
            
        Returns:
            Discretized values or embeddings
        """
        # Digitize input into bins
        categories = np.digitize(x, self.thresholds)
        
        # If embeddings are 1D (scalar per category), return scalar
        if self.embeddings.shape[1] == 1:
            return self.embeddings[categories].flatten()
        
        # Otherwise return embedding vectors
        return self.embeddings[categories]
    
    def get_params(self) -> Dict[str, Any]:
        """Get discretization parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'type': 'discretization',
            'n_categories': self.n_categories,
            'thresholds': self.thresholds.tolist(),
            'embedding_dim': self.embeddings.shape[1]
        }
    
    @classmethod
    def create_random(cls, config: Dict[str, Any],
                     rng: Optional[np.random.RandomState] = None) -> 'DiscretizationEdge':
        """Create a random discretization edge.
        
        Args:
            config: Configuration dictionary
            rng: Random number generator
            
        Returns:
            Random discretization edge
        """
        if rng is None:
            rng = np.random.RandomState()
        
        disc_config = config.get('discretization', {})
        
        # Sample number of categories
        n_cat_range = disc_config.get('n_categories', {'min': 2, 'max': 10})
        n_categories = rng.randint(n_cat_range['min'], n_cat_range['max'] + 1)
        
        # Generate thresholds
        # Use quantiles of a standard normal for reasonable spacing
        quantiles = np.linspace(0.1, 0.9, n_categories - 1)
        thresholds = np.percentile(rng.normal(0, 1, 10000), quantiles * 100)
        
        # Generate embeddings
        embedding_dim = disc_config.get('embedding_dim', 1)
        
        if embedding_dim == 1:
            # Scalar embeddings - can be arbitrary values
            embeddings = rng.normal(0, 1, (n_categories, 1))
        else:
            # Vector embeddings - orthogonal or random
            if n_categories <= embedding_dim and rng.random() < 0.5:
                # Use orthogonal embeddings (one-hot style)
                embeddings = np.eye(n_categories, embedding_dim)
            else:
                # Random embeddings
                embeddings = rng.normal(0, 1 / np.sqrt(embedding_dim), 
                                       (n_categories, embedding_dim))
        
        return cls(thresholds, embeddings)
    
    def get_category(self, x: np.ndarray) -> np.ndarray:
        """Get category indices for input values.
        
        Args:
            x: Input values
            
        Returns:
            Category indices
        """
        return np.digitize(x, self.thresholds)
    
    def get_one_hot(self, x: np.ndarray) -> np.ndarray:
        """Get one-hot encoding for input values.
        
        Args:
            x: Input values
            
        Returns:
            One-hot encoded array
        """
        categories = self.get_category(x)
        n_samples = len(categories)
        one_hot = np.zeros((n_samples, self.n_categories))
        one_hot[np.arange(n_samples), categories] = 1
        return one_hot
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'type': 'discretization',
            'thresholds': self.thresholds.tolist(),
            'embeddings': self.embeddings.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscretizationEdge':
        """Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Discretization edge
        """
        thresholds = np.array(data['thresholds'])
        embeddings = np.array(data['embeddings'])
        
        return cls(thresholds, embeddings)
