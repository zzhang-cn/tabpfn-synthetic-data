"""Basic tests for synthetic data generation."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.distributions import DistributionSampler
from src.causal_models.graph_generator import CausalGraphGenerator
from src.data_generation.generator import SyntheticDataGenerator


def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.utils.config import Config
        from src.utils.distributions import DistributionSampler
        from src.causal_models.graph_generator import CausalGraphGenerator
        from src.causal_models.scm import StructuralCausalModel
        from src.computational_edges.edge_functions import EdgeFunctionFactory
        from src.data_generation.generator import SyntheticDataGenerator
        from src.data_generation.initialization import InitializationSampler
        from src.data_generation.post_processing import PostProcessor
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_loading():
    """Test configuration loading."""
    config = Config()
    
    # Test basic config access
    assert config.get('dataset.n_samples.min') is not None
    assert config.get('graph.graph_type') is not None
    
    # Test nested access
    assert isinstance(config.get('dataset'), dict)
    
    # Test default values
    assert config.get('nonexistent.key', 'default') == 'default'


def test_distribution_sampler():
    """Test distribution sampling utilities."""
    sampler = DistributionSampler(seed=42)
    
    # Test log-uniform sampling
    samples = sampler.sample_log_uniform(1, 100, size=100)
    assert samples.shape == (100,)
    assert np.all(samples >= 1)
    assert np.all(samples <= 100)
    
    # Test beta-scaled sampling
    samples = sampler.sample_beta_scaled(0.5, 0.5, 0, 10, size=50)
    assert samples.shape == (50,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)
    
    # Test Kumaraswamy sampling
    samples = sampler.sample_kumaraswamy(2, 2, size=30)
    assert samples.shape == (30,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_graph_generation():
    """Test causal graph generation."""
    config = Config()
    generator = CausalGraphGenerator(config.get('graph'), seed=42)
    
    # Test scale-free graph
    graph = generator.generate_graph(n_nodes=10)
    assert graph.number_of_nodes() == 10
    assert graph.number_of_edges() > 0
    
    # Check it's a DAG
    import networkx as nx
    assert nx.is_directed_acyclic_graph(graph)
    
    # Test graph statistics
    stats = generator.visualize_graph(graph)
    assert 'n_nodes' in stats
    assert 'n_edges' in stats
    assert stats['is_dag'] == True


def test_basic_data_generation():
    """Test basic synthetic data generation."""
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate classification dataset
    dataset = generator.generate_dataset(
        n_samples=100,
        n_features=5,
        task_type='classification'
    )
    
    # Check structure
    assert 'train' in dataset
    assert 'test' in dataset
    assert 'task_type' in dataset
    
    # Check data shapes
    X_train, y_train = dataset['train']
    X_test, y_test = dataset['test']
    
    assert X_train.shape[0] == 80  # 80% train
    assert X_test.shape[0] == 20   # 20% test
    assert X_train.shape[1] == 5   # 5 features
    assert X_test.shape[1] == 5
    
    # Check target is categorical
    assert np.issubdtype(y_train.dtype, np.integer)
    assert len(np.unique(y_train)) <= 10


def test_regression_generation():
    """Test regression dataset generation."""
    generator = SyntheticDataGenerator(seed=42)
    
    dataset = generator.generate_dataset(
        n_samples=150,
        n_features=8,
        task_type='regression'
    )
    
    X_train, y_train = dataset['train']
    X_test, y_test = dataset['test']
    
    # Check shapes
    assert X_train.shape == (120, 8)  # 80% of 150
    assert X_test.shape == (30, 8)    # 20% of 150
    
    # Check target is continuous
    assert np.issubdtype(y_train.dtype, np.floating)


def test_reproducibility():
    """Test that same seed produces same data."""
    gen1 = SyntheticDataGenerator(seed=123)
    gen2 = SyntheticDataGenerator(seed=123)
    
    data1 = gen1.generate_dataset(n_samples=50, n_features=3)
    data2 = gen2.generate_dataset(n_samples=50, n_features=3)
    
    X1_train, y1_train = data1['train']
    X2_train, y2_train = data2['train']
    
    np.testing.assert_array_equal(X1_train, X2_train)
    np.testing.assert_array_equal(y1_train, y2_train)


def test_metadata_generation():
    """Test generation with metadata."""
    generator = SyntheticDataGenerator(seed=42)
    
    dataset, metadata = generator.generate_dataset(
        n_samples=100,
        n_features=5,
        return_metadata=True
    )
    
    # Check metadata contents
    assert 'graph_stats' in metadata
    assert 'feature_nodes' in metadata
    assert 'target_node' in metadata
    assert 'edge_types' in metadata
    assert 'post_processing_applied' in metadata
    
    # Check consistency
    assert len(metadata['feature_nodes']) == 5


def test_edge_functions():
    """Test edge function creation."""
    from src.computational_edges.edge_functions import EdgeFunctionFactory
    
    config = Config()
    factory = EdgeFunctionFactory(config.get('edges'), seed=42)
    
    # Test creating different edge types
    edge_types = ['linear', 'nonlinear', 'noise']
    
    for edge_type in edge_types:
        edge = factory.create_edge(edge_type)
        
        # Test that edge can be applied
        x = np.random.randn(10)
        y = edge(x)
        assert y.shape == x.shape
        
        # Test params
        params = edge.get_params()
        assert 'type' in params


def test_post_processing():
    """Test post-processing transformations."""
    from src.data_generation.post_processing import PostProcessor
    
    config = Config()
    processor = PostProcessor(config.get('post_processing'), seed=42)
    
    # Create dummy data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    # Apply processing
    X_proc, y_proc = processor.process(X, y, task_type='classification')
    
    # Check shapes preserved
    assert X_proc.shape == X.shape
    assert y_proc.shape == y.shape
    
    # Check applied transforms
    transforms = processor.get_applied_transforms()
    assert isinstance(transforms, list)


if __name__ == "__main__":
    # Run tests
    test_imports()
    print("✓ Imports successful")
    
    test_config_loading()
    print("✓ Config loading successful")
    
    test_distribution_sampler()
    print("✓ Distribution sampler successful")
    
    test_graph_generation()
    print("✓ Graph generation successful")
    
    test_basic_data_generation()
    print("✓ Basic data generation successful")
    
    test_regression_generation()
    print("✓ Regression generation successful")
    
    test_reproducibility()
    print("✓ Reproducibility test successful")
    
    test_metadata_generation()
    print("✓ Metadata generation successful")
    
    test_edge_functions()
    print("✓ Edge functions successful")
    
    test_post_processing()
    print("✓ Post-processing successful")
    
    print("\n🎉 All tests passed!")
