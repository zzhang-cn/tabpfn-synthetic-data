"""Main synthetic data generator following TabPFN methodology."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path

from ..utils.config import Config
from ..utils.distributions import DistributionSampler
from ..causal_models.graph_generator import CausalGraphGenerator
from ..causal_models.scm import StructuralCausalModel
from ..computational_edges.edge_functions import EdgeFunctionFactory
from .initialization import InitializationSampler
from .post_processing import PostProcessor

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic tabular datasets following TabPFN approach.
    
    This generator creates synthetic datasets by:
    1. Sampling a causal graph structure
    2. Assigning computational functions to edges
    3. Sampling initialization data
    4. Propagating through the graph
    5. Extracting features and targets
    6. Applying post-processing
    """
    
    def __init__(self, config_path: Optional[str] = None, seed: Optional[int] = None):
        """Initialize the generator.
        
        Args:
            config_path: Path to configuration file
            seed: Random seed for reproducibility
        """
        self.config = Config(config_path)
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
        # Initialize components
        self.graph_generator = CausalGraphGenerator(
            self.config.get('graph', {}), 
            seed=seed
        )
        self.edge_factory = EdgeFunctionFactory(
            self.config.get('edges', {}),
            seed=seed
        )
        self.init_sampler = InitializationSampler(
            self.config.get('initialization', {}),
            seed=seed
        )
        self.post_processor = PostProcessor(
            self.config.get('post_processing', {}),
            seed=seed
        )
        self.dist_sampler = DistributionSampler(seed=seed)
        
        logger.info("Initialized SyntheticDataGenerator")
    
    def generate_dataset(self, 
                        n_samples: Optional[int] = None,
                        n_features: Optional[int] = None,
                        task_type: str = 'classification',
                        return_metadata: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """Generate a synthetic dataset.
        
        Args:
            n_samples: Number of samples (sampled if None)
            n_features: Number of features (sampled if None)
            task_type: Type of task ('classification' or 'regression')
            return_metadata: Whether to return metadata about generation
            
        Returns:
            Dictionary with train/test data, optionally with metadata
        """
        # Sample dataset parameters
        if n_samples is None:
            n_samples = self._sample_n_samples()
        if n_features is None:
            n_features = self._sample_n_features()
        
        logger.info(f"Generating dataset with {n_samples} samples and {n_features} features")
        
        # Ensure we don't exceed max cells
        max_cells = self.config.get('dataset.max_cells', 75000)
        if n_samples * n_features > max_cells:
            n_samples = max_cells // n_features
            logger.warning(f"Reduced samples to {n_samples} to stay under max_cells limit")
        
        # Get vector dimension from config
        vector_dim = self.config.get('edges.vector_dim', 8)
        
        # Generate causal graph
        n_nodes = max(n_features + 1, int(n_features * 1.5))  # Extra nodes for latent variables
        graph = self.graph_generator.generate_graph(n_nodes)
        
        # Create edge functions
        edge_functions = {}
        for edge in graph.edges():
            edge_functions[edge] = self.edge_factory.create_random_edge(vector_dim)
        
        # Create SCM
        scm = StructuralCausalModel(graph, edge_functions)
        
        # Sample initialization data
        init_data = self.init_sampler.sample(n_samples, graph, vector_dim)
        
        # Generate data through SCM
        node_data = scm.sample(n_samples, init_data, vector_dim)
        
        # Select features and targets from all available vector components
        X, y, task_type_detected, feature_metadata = self._select_features_and_target(
            node_data, n_features, task_type, edge_functions
        )
        
        # Apply post-processing
        X, y = self.post_processor.process(X, y, task_type_detected)
        
        # Split into train/test
        train_ratio = self.config.get('dataset.train_val_split', 0.8)
        n_train = int(n_samples * train_ratio)
        
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        dataset = {
            'train': (X[train_idx], y[train_idx]),
            'test': (X[test_idx], y[test_idx]),
            'task_type': task_type_detected,
            'n_features': n_features,
            'n_samples': n_samples
        }
        
        if return_metadata:
            metadata = {
                'graph_stats': self.graph_generator.visualize_graph(graph),
                'feature_metadata': feature_metadata,
                'edge_types': {str(e): ef.get_params()['type'] 
                              for e, ef in edge_functions.items()},
                'post_processing_applied': self.post_processor.get_applied_transforms(),
                'vector_dim': vector_dim
            }
            return dataset, metadata
        
        return dataset
    
    def generate_batch(self, 
                      n_datasets: int,
                      task_type: str = 'classification',
                      parallel: bool = False) -> List[Dict]:
        """Generate multiple datasets.
        
        Args:
            n_datasets: Number of datasets to generate
            task_type: Type of task
            parallel: Whether to use parallel processing
            
        Returns:
            List of datasets
        """
        logger.info(f"Generating batch of {n_datasets} datasets")
        
        datasets = []
        for i in range(n_datasets):
            if i % 100 == 0 and i > 0:
                logger.info(f"Generated {i}/{n_datasets} datasets")
            
            dataset = self.generate_dataset(task_type=task_type)
            datasets.append(dataset)
        
        logger.info(f"Completed generating {n_datasets} datasets")
        return datasets
    
    def _sample_n_samples(self) -> int:
        """Sample number of samples.
        
        Returns:
            Number of samples
        """
        sample_config = self.config.get('dataset.n_samples', {})
        
        if sample_config.get('distribution') == 'uniform':
            return self.rng.randint(
                sample_config.get('min', 100),
                sample_config.get('max', 2048) + 1
            )
        else:
            # Default uniform sampling
            return self.rng.randint(100, 2049)
    
    def _sample_n_features(self) -> int:
        """Sample number of features.
        
        Returns:
            Number of features
        """
        feature_config = self.config.get('dataset.n_features', {})
        
        if feature_config.get('distribution') == 'beta':
            params = feature_config.get('params', {'alpha': 0.95, 'beta': 8.0})
            range_vals = feature_config.get('range', [1, 160])
            
            sample = self.dist_sampler.sample_beta_scaled(
                params['alpha'], params['beta'],
                range_vals[0], range_vals[1]
            )
            return int(sample[0])
        else:
            return self.rng.randint(1, 161)
    
    def _select_features_and_target(self, node_data: Dict[int, np.ndarray], 
                                   n_features: int, task_type: str,
                                   edge_functions: Dict) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
        """Select features and target from all vector components in nodes.
        
        Args:
            node_data: Data for all nodes (node_id -> (n_samples, vector_dim))
            n_features: Number of features to select
            task_type: Requested task type ('classification', 'regression', or 'auto')
            edge_functions: Dictionary of edge functions for discretization detection
            
        Returns:
            Tuple of (X, y, detected_task_type, metadata)
        """
        # Get all available features from all nodes
        all_features = []
        feature_metadata = []
        
        # Track which nodes have discretization edges (for categorical features)
        discretization_nodes = set()
        for (source, target), edge_func in edge_functions.items():
            if hasattr(edge_func, 'get_params') and edge_func.get_params().get('type') == 'discretization':
                discretization_nodes.add(target)
        
        # Collect all vector components from all nodes
        for node_id, data in node_data.items():
            # data shape: (n_samples, vector_dim)
            n_samples, vector_dim = data.shape
            
            if node_id in discretization_nodes:
                # For discretization nodes, get categorical values
                # Find the discretization edge function
                disc_edge = None
                for (source, target), edge_func in edge_functions.items():
                    if target == node_id and hasattr(edge_func, 'get_categorical_indices'):
                        disc_edge = edge_func
                        break
                
                if disc_edge is not None:
                    # Get categorical indices instead of continuous values
                    categorical_values = disc_edge.get_categorical_indices(data)  # (n_samples,)
                    all_features.append(categorical_values)
                    feature_metadata.append({
                        'node_id': node_id,
                        'component_id': 'categorical',
                        'type': 'categorical',
                        'n_categories': len(np.unique(categorical_values))
                    })
                else:
                    # Fallback: use continuous values from each vector component
                    for dim in range(vector_dim):
                        all_features.append(data[:, dim])
                        feature_metadata.append({
                            'node_id': node_id,
                            'component_id': dim,
                            'type': 'continuous'
                        })
            else:
                # For non-discretization nodes, use all vector components
                for dim in range(vector_dim):
                    all_features.append(data[:, dim])
                    feature_metadata.append({
                        'node_id': node_id,
                        'component_id': dim,
                        'type': 'continuous'
                    })
        
        # Shuffle features randomly
        feature_indices = list(range(len(all_features)))
        self.rng.shuffle(feature_indices)
        
        # Select n_features features
        selected_indices = feature_indices[:n_features]
        X_features = [all_features[i] for i in selected_indices]
        X_metadata = [feature_metadata[i] for i in selected_indices]
        
        # Select target from remaining features
        remaining_indices = feature_indices[n_features:]
        if not remaining_indices:
            # If no features left, use the last selected feature as target
            target_idx = selected_indices[-1]
            # Remove from features
            X_features = X_features[:-1]
            X_metadata = X_metadata[:-1]
        else:
            target_idx = remaining_indices[0]
        
        target_feature = all_features[target_idx]
        target_metadata = feature_metadata[target_idx]
        
        # Determine task type based on target
        if task_type == 'auto' or task_type == 'classification':
            if target_metadata['type'] == 'categorical':
                detected_task_type = 'classification'
                y = target_feature.astype(int)
            else:
                # Check if continuous feature looks categorical
                n_unique = len(np.unique(target_feature))
                if n_unique <= 10 and n_unique >= 2:
                    detected_task_type = 'classification'
                    y = target_feature.astype(int)
                else:
                    if task_type == 'classification':
                        # Force classification by discretizing
                        n_classes = min(n_unique, self.rng.randint(2, 11))
                        quantiles = np.linspace(0, 100, n_classes + 1)[1:-1]
                        thresholds = np.percentile(target_feature, quantiles)
                        y = np.digitize(target_feature, thresholds)
                        detected_task_type = 'classification'
                    else:
                        detected_task_type = 'regression'
                        y = target_feature
        else:
            # Regression
            detected_task_type = 'regression'
            y = target_feature
        
        # Stack features
        X = np.column_stack(X_features)  # (n_samples, n_features)
        
        metadata = {
            'feature_metadata': X_metadata,
            'target_metadata': target_metadata
        }
        
        return X, y, detected_task_type, metadata
    
    def _select_categorical_target(self, 
                                  node_data: Dict[int, np.ndarray],
                                  candidate_nodes: List[int]) -> int:
        """Select or create a categorical target node.
        
        Args:
            node_data: Data for all nodes
            candidate_nodes: Candidate nodes for target
            
        Returns:
            Target node index
        """
        # Look for naturally discrete nodes
        for node in candidate_nodes:
            data = node_data[node]
            n_unique = len(np.unique(data))
            
            # Check if naturally discrete with reasonable number of classes
            if n_unique <= 10 and n_unique >= 2:
                return node
        
        # If no good categorical found, discretize a continuous node
        if candidate_nodes:
            target_node = candidate_nodes[0]
            # Discretize into classes
            n_classes = self.rng.randint(2, 11)
            data = node_data[target_node]
            
            # Use quantiles for balanced classes
            quantiles = np.linspace(0, 100, n_classes + 1)[1:-1]
            thresholds = np.percentile(data, quantiles)
            node_data[target_node] = np.digitize(data, thresholds)
            
            return target_node
        
        # Fallback: use last node
        return list(node_data.keys())[-1]
    
    def save_dataset(self, dataset: Dict, path: Union[str, Path], format: str = 'npz'):
        """Save dataset to file.
        
        Args:
            dataset: Dataset dictionary
            path: Save path
            format: File format ('npz', 'csv', 'pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            X_train, y_train = dataset['train']
            X_test, y_test = dataset['test']
            
            np.savez(path,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    task_type=dataset['task_type'])
            
        elif format == 'csv':
            # Save as separate CSV files
            for split in ['train', 'test']:
                X, y = dataset[split]
                df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                df['target'] = y
                
                split_path = path.parent / f"{path.stem}_{split}.csv"
                df.to_csv(split_path, index=False)
        
        elif format == 'pkl':
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(dataset, f)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved dataset to {path}")
    
    @staticmethod
    def load_dataset(path: Union[str, Path], format: str = 'npz') -> Dict:
        """Load dataset from file.
        
        Args:
            path: File path
            format: File format
            
        Returns:
            Dataset dictionary
        """
        path = Path(path)
        
        if format == 'npz':
            data = np.load(path, allow_pickle=True)
            dataset = {
                'train': (data['X_train'], data['y_train']),
                'test': (data['X_test'], data['y_test']),
                'task_type': str(data['task_type'])
            }
            
        elif format == 'pkl':
            import pickle
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return dataset
