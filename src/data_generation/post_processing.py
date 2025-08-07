"""Post-processing transformations for synthetic data."""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import logging

from ..utils.distributions import DistributionSampler

logger = logging.getLogger(__name__)


class PostProcessor:
    """Apply post-processing transformations to synthetic data.
    
    Includes:
    - Kumaraswamy warping
    - Quantization
    - Missing value introduction
    - Power transformations
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """Initialize post-processor.
        
        Args:
            config: Post-processing configuration
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.dist_sampler = DistributionSampler(seed)
        self.applied_transforms = []
        
        logger.debug("Initialized PostProcessor")
    
    def process(self, X: np.ndarray, y: np.ndarray, 
                task_type: str = 'classification') -> Tuple[np.ndarray, np.ndarray]:
        """Apply post-processing to features and target.
        
        Args:
            X: Feature matrix
            y: Target values
            task_type: Type of task
            
        Returns:
            Processed features and target
        """
        self.applied_transforms = []
        
        # Apply Kumaraswamy warping
        if self.config.get('kumaraswamy_warping.enabled', True):
            if self.rng.random() < self.config.get('kumaraswamy_warping.probability', 0.3):
                X = self._apply_kumaraswamy_warping(X)
                self.applied_transforms.append('kumaraswamy_warping')
        
        # Apply quantization
        if self.config.get('quantization.enabled', True):
            if self.rng.random() < self.config.get('quantization.probability', 0.3):
                X = self._apply_quantization(X)
                self.applied_transforms.append('quantization')
        
        # Apply missing values
        if self.config.get('missing_values.enabled', True):
            if self.rng.random() < self.config.get('missing_values.probability', 0.2):
                X = self._introduce_missing_values(X)
                self.applied_transforms.append('missing_values')
        
        # Process target for classification
        if task_type == 'classification':
            y = self._process_classification_target(y)
        
        logger.debug(f"Applied transforms: {self.applied_transforms}")
        
        return X, y
    
    def _apply_kumaraswamy_warping(self, X: np.ndarray) -> np.ndarray:
        """Apply Kumaraswamy warping to features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Warped features
        """
        n_samples, n_features = X.shape
        warped_X = X.copy()
        
        # Select features to warp
        n_warp = self.rng.randint(1, max(2, n_features // 2))
        warp_features = self.rng.choice(n_features, n_warp, replace=False)
        
        warp_config = self.config.get('kumaraswamy_warping', {})
        a_range = warp_config.get('a_range', [0.5, 2.0])
        b_range = warp_config.get('b_range', [0.5, 2.0])
        
        for feat_idx in warp_features:
            # Normalize to [0, 1]
            feat = X[:, feat_idx]
            feat_min, feat_max = feat.min(), feat.max()
            
            if feat_max - feat_min > 1e-8:
                feat_norm = (feat - feat_min) / (feat_max - feat_min)
                
                # Sample Kumaraswamy parameters
                a = self.rng.uniform(*a_range)
                b = self.rng.uniform(*b_range)
                
                # Apply Kumaraswamy CDF transformation
                feat_warped = (1 - (1 - feat_norm ** a) ** b)
                
                # Rescale to original range
                warped_X[:, feat_idx] = feat_warped * (feat_max - feat_min) + feat_min
        
        logger.debug(f"Applied Kumaraswamy warping to {n_warp} features")
        return warped_X
    
    def _apply_quantization(self, X: np.ndarray) -> np.ndarray:
        """Apply quantization to features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Quantized features
        """
        n_samples, n_features = X.shape
        quantized_X = X.copy()
        
        # Select features to quantize
        n_quantize = self.rng.randint(1, max(2, n_features // 3))
        quantize_features = self.rng.choice(n_features, n_quantize, replace=False)
        
        quant_config = self.config.get('quantization', {})
        n_bins_options = quant_config.get('n_bins', [5, 10, 20])
        
        for feat_idx in quantize_features:
            feat = X[:, feat_idx]
            n_bins = self.rng.choice(n_bins_options)
            
            # Create bins using quantiles for balanced bins
            quantiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(feat, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Discretize
            discrete_values = np.digitize(feat, bins[1:-1])
            
            # Map to bin centers
            bin_centers = []
            for i in range(n_bins):
                mask = discrete_values == i
                if mask.any():
                    bin_centers.append(feat[mask].mean())
                else:
                    bin_centers.append((bins[i] + bins[i+1]) / 2)
            
            quantized_X[:, feat_idx] = np.array(bin_centers)[discrete_values]
        
        logger.debug(f"Applied quantization to {n_quantize} features")
        return quantized_X
    
    def _introduce_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Introduce missing values to features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Features with missing values (NaN)
        """
        n_samples, n_features = X.shape
        X_missing = X.copy()
        
        missing_config = self.config.get('missing_values', {})
        missing_rate_range = missing_config.get('missing_rate', [0.0, 0.3])
        mechanism = missing_config.get('mechanism', 'mcar')
        
        # Sample missing rate
        missing_rate = self.rng.uniform(*missing_rate_range)
        
        if mechanism == 'mcar':
            # Missing Completely At Random
            mask = self.rng.random((n_samples, n_features)) < missing_rate
            X_missing[mask] = np.nan
            
        elif mechanism == 'mar':
            # Missing At Random (depends on other features)
            # Select conditioning feature
            cond_feature = self.rng.randint(n_features)
            
            for j in range(n_features):
                if j == cond_feature:
                    continue
                
                # Missing probability depends on conditioning feature
                cond_values = X[:, cond_feature]
                threshold = np.percentile(cond_values, 50)
                
                # Higher missing rate for values above threshold
                miss_prob = np.where(cond_values > threshold, 
                                    missing_rate * 1.5,
                                    missing_rate * 0.5)
                miss_prob = np.clip(miss_prob, 0, 1)
                
                mask = self.rng.random(n_samples) < miss_prob
                X_missing[mask, j] = np.nan
        
        else:
            # MNAR (Missing Not At Random) - depends on own value
            for j in range(n_features):
                values = X[:, j]
                threshold = np.percentile(values, 70)
                
                # Higher values more likely to be missing
                miss_prob = np.where(values > threshold,
                                    missing_rate * 2,
                                    missing_rate * 0.5)
                miss_prob = np.clip(miss_prob, 0, 1)
                
                mask = self.rng.random(n_samples) < miss_prob
                X_missing[mask, j] = np.nan
        
        n_missing = np.isnan(X_missing).sum()
        logger.debug(f"Introduced {n_missing} missing values ({mechanism})")
        
        return X_missing
    
    def _process_classification_target(self, y: np.ndarray) -> np.ndarray:
        """Process target for classification task.
        
        Args:
            y: Target values
            
        Returns:
            Integer class labels
        """
        # Ensure integer classes
        if not np.issubdtype(y.dtype, np.integer):
            # Discretize continuous target
            n_classes = len(np.unique(y))
            
            if n_classes > 10:
                # Too many unique values, bin them
                n_classes = self.rng.randint(2, 11)
                quantiles = np.linspace(0, 100, n_classes + 1)[1:-1]
                thresholds = np.percentile(y, quantiles)
                y = np.digitize(y, thresholds)
            else:
                # Map to integers
                unique_vals = np.unique(y)
                mapping = {val: i for i, val in enumerate(unique_vals)}
                y = np.array([mapping[val] for val in y])
        
        # Ensure classes start from 0
        y_min = y.min()
        if y_min != 0:
            y = y - y_min
        
        return y.astype(np.int32)
    
    def apply_power_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply power transformation to features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed features
        """
        # Use Yeo-Johnson transformation (handles negative values)
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        
        # Transform each feature independently
        X_transformed = X.copy()
        
        for j in range(X.shape[1]):
            if not np.any(np.isnan(X[:, j])):
                try:
                    X_transformed[:, j] = pt.fit_transform(X[:, j:j+1]).flatten()
                except:
                    # Skip if transformation fails
                    pass
        
        return X_transformed
    
    def get_applied_transforms(self) -> List[str]:
        """Get list of applied transformations.
        
        Returns:
            List of transformation names
        """
        return self.applied_transforms
