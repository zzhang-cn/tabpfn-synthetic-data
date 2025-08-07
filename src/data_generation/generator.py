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
        
        # Generate causal graph
        n_nodes = max(n_features + 1, int(n_features * 1.5))  # Extra nodes for latent variables
        graph = self.graph_generator.generate_graph(n_nodes)
        
        # Create edge functions
        edge_functions = {}
        for edge in graph.edges():
            edge_functions[edge] = self.edge_factory.create_random_edge()
        
        # Create SCM
        scm = StructuralCausalModel(graph, edge_functions)
        
        # Sample initialization data
        init_data = self.init_sampler.sample(n_samples, graph)
        
        # Generate data through SCM
        node_data = scm.sample(n_samples, init_data)
        
        # Select feature and target nodes
        all_nodes = list(graph.nodes())
        self.rng.shuffle(all_nodes)
        
        # Select nodes for features and target
        feature_nodes = all_nodes[:n_features]
        
        if task_type == 'classification':
            # Find or create categorical target
            target_node = self._select_categorical_target(node_data, all_nodes[n_features:])
        else:
            # Select continuous target
            target_node = all_nodes[n_features] if len(all_nodes) > n_features else all_nodes[-1]
        
        # Extract features and target
        X = np.column_stack([node_data[node] for node in feature_nodes])
        y = node_data[target_node]
        
        # Apply post-processing
        X, y = self.post_processor.process(X, y, task_type)
        
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
            'task_type': task_type,
            'n_features': n_features,
            'n_samples': n_samples
        }
        
        if return_metadata:
            metadata = {
                'graph_stats': self.graph_generator.visualize_graph(graph),
                'feature_nodes': feature_nodes,
                'target_node': target_node,
                'edge_types': {str(e): ef.get_params()['type'] 
                              for e, ef in edge_functions.items()},
                'post_processing_applied': self.post_processor.get_applied_transforms()
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
