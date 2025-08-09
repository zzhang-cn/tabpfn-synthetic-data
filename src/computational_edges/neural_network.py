"""Neural network edge implementation."""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

from .edge_functions import EdgeFunction

logger = logging.getLogger(__name__)


class NeuralNetworkEdge(EdgeFunction):
    """Neural network transformation for edges."""
    
    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray],
                 activations: List[str]):
        """Initialize neural network edge.
        
        Args:
            weights: List of weight matrices for each layer
            biases: List of bias vectors for each layer
            activations: List of activation functions for each layer
        """
        self.weights = weights
        self.biases = biases
        self.activations = activations
        
        # Activation function mappings
        self.activation_funcs = {
            'identity': lambda x: x,
            'tanh': np.tanh,
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'relu': lambda x: np.maximum(0, x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
            'sin': np.sin,
            'cos': np.cos,
            'abs': np.abs,
            'square': lambda x: x ** 2,
            'sqrt': lambda x: np.sqrt(np.abs(x)),
            'log': lambda x: np.log(np.abs(x) + 1e-8),
            'exp': lambda x: np.exp(np.clip(x, -10, 10)),
            'softplus': lambda x: np.log(1 + np.exp(np.clip(x, -500, 500)))
        }
        
        logger.debug(f"Created NN edge with {len(weights)} layers")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply neural network transformation.
        
        Args:
            x: Input data (n_samples, vector_dim)
            
        Returns:
            Transformed data (n_samples, vector_dim)
        """
        # Forward pass through network
        h = x
        for i, (W, b, act_name) in enumerate(zip(self.weights, self.biases, self.activations)):
            # Linear transformation
            h = h @ W + b
            
            # Apply activation
            if act_name in self.activation_funcs:
                h = self.activation_funcs[act_name](h)
            else:
                logger.warning(f"Unknown activation {act_name}, using identity")
                pass
        
        return h
    
    def get_params(self) -> Dict[str, Any]:
        """Get neural network parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'type': 'neural_network',
            'n_layers': len(self.weights),
            'layer_sizes': [W.shape for W in self.weights],
            'activations': self.activations
        }
    
    @classmethod
    def create_random(cls, config: Dict[str, Any], 
                     rng: Optional[np.random.RandomState] = None,
                     vector_dim: int = 8) -> 'NeuralNetworkEdge':
        """Create a random neural network edge.
        
        Args:
            config: Configuration dictionary
            rng: Random number generator
            vector_dim: Input/output vector dimension
            
        Returns:
            Random neural network edge
        """
        if rng is None:
            rng = np.random.RandomState()
        
        nn_config = config.get('neural_network', {})
        
        # Sample network architecture
        n_layers = rng.randint(1, 4)  # 1-3 hidden layers
        hidden_dims = nn_config.get('hidden_dims', [8, 16, 32, 64])
        activations_pool = nn_config.get('activations', 
            ['tanh', 'sigmoid', 'relu', 'sin', 'identity'])
        
        # Build network
        weights = []
        biases = []
        activations = []
        
        # Input/output dimension is vector_dim
        input_dim = vector_dim
        
        for i in range(n_layers):
            # Sample hidden dimension
            if i < n_layers - 1:
                output_dim = rng.choice(hidden_dims)
            else:
                # Final layer outputs vector_dim to maintain dimensionality
                output_dim = vector_dim
                # Final layer outputs vector_dim to maintain dimensionality
                output_dim = vector_dim
            
            # Initialize weights (Xavier/He initialization)
            activation = rng.choice(activations_pool)
            
            if activation in ['relu', 'leaky_relu', 'elu']:
                # He initialization for ReLU variants
                std = np.sqrt(2.0 / input_dim)
            else:
                # Xavier initialization for others
                std = np.sqrt(1.0 / input_dim)
            
            W = rng.normal(0, std, (input_dim, output_dim))
            b = np.zeros(output_dim)
            
            weights.append(W)
            biases.append(b)
            activations.append(activation)
            
            input_dim = output_dim
        
        return cls(weights, biases, activations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'type': 'neural_network',
            'weights': [W.tolist() for W in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'activations': self.activations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralNetworkEdge':
        """Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Neural network edge
        """
        weights = [np.array(W) for W in data['weights']]
        biases = [np.array(b) for b in data['biases']]
        activations = data['activations']
        
        return cls(weights, biases, activations)
