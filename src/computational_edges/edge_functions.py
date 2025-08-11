"""Base classes and factory for computational edge functions."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def xavier_uniform_init(shape: tuple, gain: float = 1.0, 
                       rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Xavier/Glorot uniform initialization.
    
    Args:
        shape: Shape of the weight tensor
        gain: Scaling factor
        rng: Random number generator
        
    Returns:
        Initialized weights
    """
    if rng is None:
        rng = np.random.RandomState()
    
    fan_in = shape[0] if len(shape) >= 2 else 1
    fan_out = shape[1] if len(shape) >= 2 else shape[0]
    
    # Xavier uniform bound
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, shape)


def xavier_normal_init(shape: tuple, gain: float = 1.0, 
                      rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Xavier/Glorot normal initialization.
    
    Args:
        shape: Shape of the weight tensor
        gain: Scaling factor
        rng: Random number generator
        
    Returns:
        Initialized weights
    """
    if rng is None:
        rng = np.random.RandomState()
    
    fan_in = shape[0] if len(shape) >= 2 else 1
    fan_out = shape[1] if len(shape) >= 2 else shape[0]
    
    # Xavier normal std
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, shape)


def he_uniform_init(shape: tuple, 
                   rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """He uniform initialization (for ReLU activations).
    
    Args:
        shape: Shape of the weight tensor
        rng: Random number generator
        
    Returns:
        Initialized weights
    """
    if rng is None:
        rng = np.random.RandomState()
    
    fan_in = shape[0] if len(shape) >= 2 else 1
    
    # He uniform bound
    bound = np.sqrt(6.0 / fan_in)
    return rng.uniform(-bound, bound, shape)


def he_normal_init(shape: tuple, 
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """He normal initialization (for ReLU activations).
    
    Args:
        shape: Shape of the weight tensor
        rng: Random number generator
        
    Returns:
        Initialized weights
    """
    if rng is None:
        rng = np.random.RandomState()
    
    fan_in = shape[0] if len(shape) >= 2 else 1
    
    # He normal std
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, shape)


def embedding_init(shape: tuple, scale: float = 1.0,
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Embedding initialization using scaled normal distribution.
    
    Args:
        shape: Shape of the embedding tensor
        scale: Scaling factor
        rng: Random number generator
        
    Returns:
        Initialized embeddings
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Use normal distribution scaled by 1/sqrt(embedding_dim)
    embedding_dim = shape[-1] if len(shape) >= 2 else shape[0]
    std = scale / np.sqrt(embedding_dim)
    return rng.normal(0, std, shape)


class EdgeFunction(ABC):
    """Abstract base class for edge functions in SCM."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the edge function.
        
        Args:
            x: Input data (n_samples, vector_dim)
            
        Returns:
            Transformed data (n_samples, vector_dim)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get function parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass


class IdentityEdge(EdgeFunction):
    """Identity function (no transformation)."""
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def get_params(self) -> Dict[str, Any]:
        return {'type': 'identity'}


class LinearEdge(EdgeFunction):
    """Linear transformation."""
    
    def __init__(self, weight_matrix: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None, vector_dim: int = 8):
        """Initialize linear edge.
        
        Args:
            weight_matrix: Transformation matrix (vector_dim, vector_dim)
            bias: Additive bias vector (vector_dim,)
            vector_dim: Dimension of node vectors
        """
        self.vector_dim = vector_dim
        if weight_matrix is not None:
            self.weight_matrix = weight_matrix
        else:
            # Default to scaled identity with some random perturbation
            self.weight_matrix = np.eye(vector_dim) + 0.1 * np.random.randn(vector_dim, vector_dim)
        
        if bias is not None:
            self.bias = bias
        else:
            self.bias = 0.1 * np.random.randn(vector_dim)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply linear transformation.
        
        Args:
            x: Input vectors (n_samples, vector_dim)
            
        Returns:
            Transformed vectors (n_samples, vector_dim)
        """
        return x @ self.weight_matrix.T + self.bias
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'linear',
            'vector_dim': self.vector_dim,
            'weight_shape': self.weight_matrix.shape,
            'bias_shape': self.bias.shape
        }


class NonlinearEdge(EdgeFunction):
    """Nonlinear transformation with various activation functions."""
    
    def __init__(self, activation: str = 'tanh', scale: float = 1.0):
        """Initialize nonlinear edge.
        
        Args:
            activation: Type of activation function
            scale: Scale factor
        """
        self.activation = activation
        self.scale = scale
        
        # Map activation names to functions
        self.activation_funcs = {
            'tanh': np.tanh,
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'relu': lambda x: np.maximum(0, x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'sin': np.sin,
            'cos': np.cos,
            'abs': np.abs,
            'square': lambda x: x ** 2,
            'sqrt': lambda x: np.sqrt(np.abs(x)),
            'log': lambda x: np.log(np.abs(x) + 1e-8),
            'exp': lambda x: np.exp(np.clip(x, -10, 10)),
            'identity': lambda x: x
        }
        
        if activation not in self.activation_funcs:
            raise ValueError(f"Unknown activation: {activation}")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply nonlinear transformation element-wise.
        
        Args:
            x: Input vectors (n_samples, vector_dim)
            
        Returns:
            Transformed vectors (n_samples, vector_dim)
        """
        func = self.activation_funcs[self.activation]
        return self.scale * func(x)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'nonlinear',
            'activation': self.activation,
            'scale': self.scale
        }
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        func = self.activation_funcs[self.activation]
        return self.scale * func(x)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'nonlinear',
            'activation': self.activation,
            'scale': self.scale
        }


class PolynomialEdge(EdgeFunction):
    """Polynomial transformation."""
    
    def __init__(self, coefficients: np.ndarray):
        """Initialize polynomial edge.
        
        Args:
            coefficients: Polynomial coefficients [c0, c1, c2, ...]
                         for c0 + c1*x + c2*x^2 + ...
        """
        self.coefficients = np.array(coefficients)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** i)
        return result
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'polynomial',
            'coefficients': self.coefficients.tolist()
        }


class SplineEdge(EdgeFunction):
    """Spline transformation using piece-wise cubic functions."""
    
    def __init__(self, knots: np.ndarray, coefficients: np.ndarray):
        """Initialize spline edge.
        
        Args:
            knots: Knot points for spline
            coefficients: Spline coefficients
        """
        self.knots = knots
        self.coefficients = coefficients
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy import interpolate
        
        # Create spline interpolator
        tck = (self.knots, self.coefficients, 3)  # 3 for cubic
        spline = interpolate.BSpline(*tck)
        
        return spline(x)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'spline',
            'knots': self.knots.tolist(),
            'coefficients': self.coefficients.tolist()
        }


class ThresholdEdge(EdgeFunction):
    """Threshold/step function."""
    
    def __init__(self, threshold: float = 0.0, low_value: float = 0.0, 
                 high_value: float = 1.0):
        """Initialize threshold edge.
        
        Args:
            threshold: Threshold value
            low_value: Output when input < threshold
            high_value: Output when input >= threshold
        """
        self.threshold = threshold
        self.low_value = low_value
        self.high_value = high_value
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < self.threshold, self.low_value, self.high_value)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'threshold',
            'threshold': self.threshold,
            'low_value': self.low_value,
            'high_value': self.high_value
        }


class NoiseEdge(EdgeFunction):
    """Edge that only adds noise."""
    
    def __init__(self, noise_std: float = 0.1, noise_type: str = 'normal'):
        """Initialize noise edge.
        
        Args:
            noise_std: Standard deviation of noise
            noise_type: Type of noise distribution
        """
        self.noise_std = noise_std
        self.noise_type = noise_type
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.noise_type == 'normal':
            noise = np.random.normal(0, self.noise_std, x.shape)
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_std, self.noise_std, x.shape)
        elif self.noise_type == 'laplace':
            noise = np.random.laplace(0, self.noise_std, x.shape)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return x + noise
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'noise',
            'noise_std': self.noise_std,
            'noise_type': self.noise_type
        }


class EdgeFunctionFactory:
    """Factory for creating edge functions."""
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the factory.
        
        Args:
            config: Configuration for edge functions
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Get vector dimension from config
        self.vector_dim = config.get('vector_dim', 8)
        
        # Get type probabilities
        self.type_probs = config.get('type_probabilities', {
            'neural_network': 0.4,
            'discretization': 0.2,
            'decision_tree': 0.2,
            'noise': 0.2
        })
        
        # Normalize probabilities
        total = sum(self.type_probs.values())
        self.type_probs = {k: v/total for k, v in self.type_probs.items()}
        
        logger.info("Initialized EdgeFunctionFactory")
    
    def create_random_edge(self, vector_dim: Optional[int] = None) -> EdgeFunction:
        """Create a random edge function.
        
        Args:
            vector_dim: Override vector dimension
        
        Returns:
            Random edge function
        """
        if vector_dim is None:
            vector_dim = self.vector_dim
            
        # Sample edge type
        edge_type = self.rng.choice(
            list(self.type_probs.keys()),
            p=list(self.type_probs.values())
        )
        
        return self.create_edge(edge_type, vector_dim)
    
    def create_edge(self, edge_type: str, vector_dim: Optional[int] = None) -> EdgeFunction:
        """Create an edge function of specified type.
        
        Args:
            edge_type: Type of edge function
            vector_dim: Vector dimension for the edge function
            
        Returns:
            Edge function
        """
        if vector_dim is None:
            vector_dim = self.vector_dim
            
        if edge_type == 'identity':
            return IdentityEdge()
        
        elif edge_type == 'linear':
            # Use Xavier initialization for better numerical stability
            weight_matrix = xavier_normal_init((vector_dim, vector_dim), gain=1.0, rng=self.rng)
            bias = self.rng.normal(0, 0.1, vector_dim)  # Initialize bias to small values
            return LinearEdge(weight_matrix, bias, vector_dim)
        
        elif edge_type == 'nonlinear':
            activations = self.config.get('neural_network', {}).get(
                'activations', ['tanh', 'sigmoid', 'relu']
            )
            activation = self.rng.choice(activations)
            scale = self.rng.uniform(0.5, 2.0)
            return NonlinearEdge(activation, scale)
        
        elif edge_type == 'noise':
            noise_config = self.config.get('noise', {})
            std_range = noise_config.get('std_range', [0.01, 0.5])
            noise_std = self.rng.uniform(*std_range)
            noise_type = noise_config.get('distribution', 'normal')
            return NoiseEdge(noise_std, noise_type)
        
        elif edge_type == 'neural_network':
            # Import here to avoid circular dependency
            from .neural_network import NeuralNetworkEdge
            return NeuralNetworkEdge.create_random(self.config, self.rng, vector_dim)
        
        elif edge_type == 'discretization':
            from .discretization import DiscretizationEdge
            return DiscretizationEdge.create_random(self.config, self.rng, vector_dim)
        
        elif edge_type == 'decision_tree':
            from .decision_tree import DecisionTreeEdge
            return DecisionTreeEdge.create_random(self.config, self.rng, vector_dim)
        
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")
    
    def create_edge_for_pair(self, source_node: int, target_node: int) -> EdgeFunction:
        """Create an edge function for a specific node pair.
        
        Args:
            source_node: Source node index
            target_node: Target node index
            
        Returns:
            Edge function
        """
        # Could use node properties to determine edge type
        # For now, just create random edge
        return self.create_random_edge()
