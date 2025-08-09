"""Discretization edge for creating categorical features."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy.spatial.distance import cdist

from .edge_functions import EdgeFunction, embedding_init

logger = logging.getLogger(__name__)


class DiscretizationEdge(EdgeFunction):
    """Edge that discretizes vector values into categories using nearest neighbor mapping."""
    
    def __init__(self, prototype_vectors: np.ndarray, embedding_vectors: np.ndarray):
        """Initialize discretization edge.
        
        Args:
            prototype_vectors: Prototype vectors for nearest neighbor mapping (K, vector_dim)
            embedding_vectors: Embedding vectors for each category (K, vector_dim)
        """
        self.prototype_vectors = prototype_vectors
        self.embedding_vectors = embedding_vectors
        self.n_categories = len(prototype_vectors)
        
        logger.debug(f"Created discretization edge with {self.n_categories} categories")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply discretization using nearest neighbor mapping.
        
        Args:
            x: Input vectors (n_samples, vector_dim)
            
        Returns:
            Embedding vectors (n_samples, vector_dim)
        """
        # Compute distances to all prototypes
        distances = cdist(x, self.prototype_vectors, metric='euclidean')  # (n_samples, K)
        
        # Find nearest prototype for each sample
        nearest_indices = np.argmin(distances, axis=1)  # (n_samples,)
        
        # Map to embedding vectors
        embedded = self.embedding_vectors[nearest_indices]  # (n_samples, vector_dim)
        
        return embedded
    
    def get_categorical_indices(self, x: np.ndarray) -> np.ndarray:
        """Get categorical indices without embedding (for feature selection).
        
        Args:
            x: Input vectors (n_samples, vector_dim)
            
        Returns:
            Categorical indices (n_samples,)
        """
        # Compute distances to all prototypes
        distances = cdist(x, self.prototype_vectors, metric='euclidean')
        
        # Find nearest prototype for each sample
        nearest_indices = np.argmin(distances, axis=1)
        
        return nearest_indices
    
    def get_params(self) -> Dict[str, Any]:
        """Get discretization parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'type': 'discretization',
            'n_categories': self.n_categories,
            'vector_dim': self.prototype_vectors.shape[1]
        }
    
    @classmethod
    def create_random(cls, config: Dict[str, Any],
                     rng: Optional[np.random.RandomState] = None,
                     vector_dim: int = 8) -> 'DiscretizationEdge':
        """Create a random discretization edge.
        
        Args:
            config: Configuration dictionary
            rng: Random number generator
            vector_dim: Vector dimension
            
        Returns:
            Random discretization edge
        """
        if rng is None:
            rng = np.random.RandomState()
        
        disc_config = config.get('discretization', {})
        
        # Sample number of categories using gamma distribution as per paper
        # Gamma distribution with offset of 2 for minimum 2 categories
        alpha = disc_config.get('gamma_alpha', 2.0)
        scale = disc_config.get('gamma_scale', 1.5)
        
        n_categories = max(2, int(np.round(rng.gamma(alpha, scale))) + 2)
        # Cap at reasonable maximum
        n_categories = min(n_categories, disc_config.get('max_categories', 15))
        
        # Generate prototype vectors for nearest neighbor mapping
        prototype_vectors = embedding_init((n_categories, vector_dim), rng=rng)
        
        # Generate embedding vectors for categorical representation
        embedding_vectors = embedding_init((n_categories, vector_dim), rng=rng)
        
        return cls(prototype_vectors, embedding_vectors)
    
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
