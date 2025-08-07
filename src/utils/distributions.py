"""Distribution sampling utilities for synthetic data generation."""

import numpy as np
from typing import Union, Tuple, Optional, List
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DistributionSampler:
    """Utility class for sampling from various distributions.
    
    Provides methods for sampling from distributions commonly used
    in the TabPFN synthetic data generation process.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the sampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        logger.debug(f"Initialized DistributionSampler with seed={seed}")
    
    def sample_log_uniform(self, low: float, high: float, 
                          size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample from log-uniform distribution.
        
        Args:
            low: Lower bound (must be positive)
            high: Upper bound
            size: Output shape
            
        Returns:
            Samples from log-uniform distribution
        """
        if low <= 0:
            raise ValueError("Lower bound must be positive for log-uniform distribution")
        
        log_low = np.log(low)
        log_high = np.log(high)
        log_samples = self.rng.uniform(log_low, log_high, size)
        return np.exp(log_samples)
    
    def sample_beta_scaled(self, alpha: float, beta: float,
                          range_min: float, range_max: float,
                          size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample from beta distribution and scale to range.
        
        Args:
            alpha: Beta distribution alpha parameter
            beta: Beta distribution beta parameter
            range_min: Minimum value of scaled range
            range_max: Maximum value of scaled range
            size: Output shape
            
        Returns:
            Scaled samples from beta distribution
        """
        samples = self.rng.beta(alpha, beta, size)
        return range_min + samples * (range_max - range_min)
    
    def sample_kumaraswamy(self, a: float, b: float,
                          size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample from Kumaraswamy distribution.
        
        The Kumaraswamy distribution is similar to the Beta distribution
        but has simpler closed-form CDF and inverse CDF.
        
        Args:
            a: First shape parameter (a > 0)
            b: Second shape parameter (b > 0)
            size: Output shape
            
        Returns:
            Samples from Kumaraswamy distribution
        """
        if a <= 0 or b <= 0:
            raise ValueError("Kumaraswamy parameters must be positive")
        
        u = self.rng.uniform(0, 1, size)
        return (1 - (1 - u) ** (1/b)) ** (1/a)
    
    def sample_truncated_normal(self, mean: float, std: float,
                               low: float, high: float,
                               size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample from truncated normal distribution.
        
        Args:
            mean: Mean of the underlying normal distribution
            std: Standard deviation of the underlying normal distribution
            low: Lower truncation bound
            high: Upper truncation bound
            size: Output shape
            
        Returns:
            Samples from truncated normal distribution
        """
        a = (low - mean) / std
        b = (high - mean) / std
        
        samples = stats.truncnorm.rvs(
            a, b, loc=mean, scale=std, 
            size=size, random_state=self.rng
        )
        return samples
    
    def sample_mixture(self, distributions: List[dict],
                      weights: Optional[np.ndarray] = None,
                      size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample from a mixture of distributions.
        
        Args:
            distributions: List of distribution specifications
                Each dict should have 'type' and parameters
            weights: Mixture weights (normalized internally)
            size: Output shape
            
        Returns:
            Samples from mixture distribution
        """
        n_dists = len(distributions)
        if weights is None:
            weights = np.ones(n_dists) / n_dists
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()
        
        # Determine output shape
        if isinstance(size, int):
            n_samples = size
            output_shape = (size,)
        else:
            n_samples = np.prod(size)
            output_shape = size
        
        # Sample component assignments
        components = self.rng.choice(n_dists, size=n_samples, p=weights)
        
        # Sample from each component
        samples = np.zeros(n_samples)
        for i, dist in enumerate(distributions):
            mask = components == i
            n_comp = mask.sum()
            
            if n_comp > 0:
                if dist['type'] == 'normal':
                    comp_samples = self.rng.normal(
                        dist.get('mean', 0),
                        dist.get('std', 1),
                        n_comp
                    )
                elif dist['type'] == 'uniform':
                    comp_samples = self.rng.uniform(
                        dist.get('low', -1),
                        dist.get('high', 1),
                        n_comp
                    )
                elif dist['type'] == 'gamma':
                    comp_samples = self.rng.gamma(
                        dist.get('shape', 2),
                        dist.get('scale', 1),
                        n_comp
                    )
                else:
                    raise ValueError(f"Unknown distribution type: {dist['type']}")
                
                samples[mask] = comp_samples
        
        return samples.reshape(output_shape)
    
    def sample_categorical(self, n_categories: int,
                          probabilities: Optional[np.ndarray] = None,
                          size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample from categorical distribution.
        
        Args:
            n_categories: Number of categories
            probabilities: Category probabilities (uniform if None)
            size: Output shape
            
        Returns:
            Integer samples from categorical distribution
        """
        if probabilities is None:
            probabilities = np.ones(n_categories) / n_categories
        else:
            probabilities = np.array(probabilities)
            probabilities = probabilities / probabilities.sum()
        
        return self.rng.choice(n_categories, size=size, p=probabilities)
    
    def sample_power_law(self, alpha: float, x_min: float = 1,
                        size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample from power law distribution.
        
        Args:
            alpha: Power law exponent (alpha > 1)
            x_min: Minimum value
            size: Output shape
            
        Returns:
            Samples from power law distribution
        """
        if alpha <= 1:
            raise ValueError("Alpha must be greater than 1 for power law distribution")
        
        u = self.rng.uniform(0, 1, size)
        return x_min * (1 - u) ** (-1 / (alpha - 1))
    
    def sample_yeo_johnson_params(self, size: Union[int, Tuple] = 1) -> np.ndarray:
        """Sample parameters for Yeo-Johnson transformation.
        
        Args:
            size: Number of parameters to sample
            
        Returns:
            Lambda parameters for Yeo-Johnson transformation
        """
        # Sample from range that produces diverse transformations
        return self.rng.uniform(-2, 2, size)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
        logger.debug(f"Reset seed to {seed}")
