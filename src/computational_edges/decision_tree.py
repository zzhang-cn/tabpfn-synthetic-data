"""Decision tree edge implementation."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from .edge_functions import EdgeFunction

logger = logging.getLogger(__name__)


class TreeNode:
    """Node in a decision tree."""
    
    def __init__(self, is_leaf: bool = False, value: Optional[float] = None,
                 threshold: Optional[float] = None, 
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        """Initialize tree node.
        
        Args:
            is_leaf: Whether this is a leaf node
            value: Value for leaf nodes
            threshold: Split threshold for internal nodes
            left: Left child (values <= threshold)
            right: Right child (values > threshold)
        """
        self.is_leaf = is_leaf
        self.value = value
        self.threshold = threshold
        self.left = left
        self.right = right
    
    def predict(self, x: float) -> float:
        """Predict value for input.
        
        Args:
            x: Input value
            
        Returns:
            Predicted value
        """
        if self.is_leaf:
            return self.value
        
        if x <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = {
            'is_leaf': self.is_leaf,
            'value': self.value,
            'threshold': self.threshold
        }
        
        if not self.is_leaf:
            data['left'] = self.left.to_dict()
            data['right'] = self.right.to_dict()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeNode':
        """Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Tree node
        """
        if data['is_leaf']:
            return cls(is_leaf=True, value=data['value'])
        else:
            left = cls.from_dict(data['left'])
            right = cls.from_dict(data['right'])
            return cls(is_leaf=False, threshold=data['threshold'],
                      left=left, right=right)


class DecisionTreeEdge(EdgeFunction):
    """Decision tree transformation for edges."""
    
    def __init__(self, root: TreeNode):
        """Initialize decision tree edge.
        
        Args:
            root: Root node of the tree
        """
        self.root = root
        self.n_nodes = self._count_nodes(root)
        self.depth = self._get_depth(root)
        
        logger.debug(f"Created decision tree edge with depth {self.depth}")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply decision tree transformation to vectors.
        
        Args:
            x: Input vectors (n_samples, vector_dim)
            
        Returns:
            Transformed vectors (n_samples, vector_dim)
        """
        # Use L2 norm of vectors as scalar input to decision tree
        norms = np.linalg.norm(x, axis=1)  # (n_samples,)
        
        # Apply tree to norms
        transformed_norms = np.zeros_like(norms)
        for i, norm in enumerate(norms):
            transformed_norms[i] = self.root.predict(norm)
        
        # Scale original vectors by the ratio of transformed to original norms
        # Avoid division by zero
        original_norms = norms + 1e-8
        scale_factors = transformed_norms / original_norms
        
        # Scale each vector
        result = x * scale_factors.reshape(-1, 1)
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        """Get decision tree parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'type': 'decision_tree',
            'n_nodes': self.n_nodes,
            'depth': self.depth
        }
    
    def _count_nodes(self, node: Optional[TreeNode]) -> int:
        """Count nodes in tree.
        
        Args:
            node: Current node
            
        Returns:
            Number of nodes
        """
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def _get_depth(self, node: Optional[TreeNode]) -> int:
        """Get tree depth.
        
        Args:
            node: Current node
            
        Returns:
            Tree depth
        """
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))
    
    @classmethod
    def create_random(cls, config: Dict[str, Any],
                     rng: Optional[np.random.RandomState] = None,
                     vector_dim: int = 8) -> 'DecisionTreeEdge':
        """Create a random decision tree edge.
        
        Args:
            config: Configuration dictionary
            rng: Random number generator
            vector_dim: Vector dimension (not used directly but kept for interface consistency)
            
        Returns:
            Random decision tree edge
        """
        if rng is None:
            rng = np.random.RandomState()
        
        tree_config = config.get('decision_tree', {})
        max_depth = tree_config.get('max_depth', 3)
        
        # Build random tree
        root = cls._build_random_tree(rng, max_depth, current_depth=0)
        
        return cls(root)
    
    @classmethod
    def _build_random_tree(cls, rng: np.random.RandomState, 
                          max_depth: int, current_depth: int) -> TreeNode:
        """Build a random decision tree.
        
        Args:
            rng: Random number generator
            max_depth: Maximum tree depth
            current_depth: Current depth in tree
            
        Returns:
            Root node of random tree
        """
        # Probability of creating a leaf increases with depth
        p_leaf = 0.3 + 0.7 * (current_depth / max(max_depth, 1))
        
        if current_depth >= max_depth or rng.random() < p_leaf:
            # Create leaf node
            value = rng.normal(0, 1)
            return TreeNode(is_leaf=True, value=value)
        
        # Create internal node
        threshold = rng.normal(0, 1)
        left = cls._build_random_tree(rng, max_depth, current_depth + 1)
        right = cls._build_random_tree(rng, max_depth, current_depth + 1)
        
        return TreeNode(is_leaf=False, threshold=threshold, left=left, right=right)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'type': 'decision_tree',
            'root': self.root.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionTreeEdge':
        """Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Decision tree edge
        """
        root = TreeNode.from_dict(data['root'])
        return cls(root)
    
    def get_splits(self) -> List[float]:
        """Get all split thresholds in the tree.
        
        Returns:
            List of split thresholds
        """
        splits = []
        
        def collect_splits(node):
            if node is None or node.is_leaf:
                return
            splits.append(node.threshold)
            collect_splits(node.left)
            collect_splits(node.right)
        
        collect_splits(self.root)
        return sorted(splits)
