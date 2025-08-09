"""Basic usage examples for synthetic data generation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_generation import SyntheticDataGenerator


def example_basic_generation():
    """Example of basic dataset generation."""
    print("=" * 50)
    print("BASIC DATASET GENERATION")
    print("=" * 50)
    
    # Create generator with seed for reproducibility
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate a classification dataset
    dataset, metadata = generator.generate_dataset(
        n_samples=1000,
        n_features=10,
        task_type='classification',
        return_metadata=True,
    )
    
    # Extract data
    X_train, y_train = dataset['train']
    X_test, y_test = dataset['test']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution: {np.bincount(y_train)}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

    return dataset


def example_with_metadata():
    """Example showing metadata extraction."""
    print("\n" + "=" * 50)
    print("GENERATION WITH METADATA")
    print("=" * 50)
    
    generator = SyntheticDataGenerator(seed=123)
    
    # Generate with metadata
    dataset, metadata = generator.generate_dataset(
        n_samples=500,
        n_features=15,
        task_type='regression',
        return_metadata=True
    )
    
    print("Graph statistics:")
    for key, value in metadata['graph_stats'].items():
        print(f"  {key}: {value}")
    
    print(f"\nFeature nodes: {metadata['feature_nodes'][:5]}...")
    print(f"Target node: {metadata['target_node']}")
    
    print("\nEdge types used:")
    edge_type_counts = {}
    for edge_type in metadata['edge_types'].values():
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    for edge_type, count in edge_type_counts.items():
        print(f"  {edge_type}: {count}")
    
    print(f"\nPost-processing applied: {metadata['post_processing_applied']}")
    
    return dataset, metadata


def example_batch_generation():
    """Example of generating multiple datasets."""
    print("\n" + "=" * 50)
    print("BATCH GENERATION")
    print("=" * 50)
    
    generator = SyntheticDataGenerator(seed=456)
    
    # Generate batch of datasets
    n_datasets = 10
    datasets = generator.generate_batch(
        n_datasets=n_datasets,
        task_type='classification'
    )
    
    print(f"Generated {len(datasets)} datasets")
    
    # Analyze dataset characteristics
    sample_counts = []
    feature_counts = []
    class_counts = []
    
    for dataset in datasets:
        X_train, y_train = dataset['train']
        X_test, y_test = dataset['test']
        
        total_samples = len(X_train) + len(X_test)
        sample_counts.append(total_samples)
        feature_counts.append(X_train.shape[1])
        class_counts.append(len(np.unique(y_train)))
    
    print(f"Sample count range: {min(sample_counts)} - {max(sample_counts)}")
    print(f"Feature count range: {min(feature_counts)} - {max(feature_counts)}")
    print(f"Class count range: {min(class_counts)} - {max(class_counts)}")
    
    return datasets


def example_custom_config():
    """Example using custom configuration."""
    print("\n" + "=" * 50)
    print("CUSTOM CONFIGURATION")
    print("=" * 50)
    
    # Create custom config file
    import yaml
    
    custom_config = {
        'dataset': {
            'n_samples': {'min': 500, 'max': 1000},
            'n_features': {
                'distribution': 'beta',
                'params': {'alpha': 2, 'beta': 5},
                'range': [5, 30]
            }
        },
        'graph': {
            'graph_type': 'erdos_renyi',
            'n_nodes': {'min': 10, 'max': 50}
        },
        'post_processing': {
            'missing_values': {
                'enabled': True,
                'probability': 0.5,
                'missing_rate': [0.1, 0.2]
            }
        }
    }
    
    # Save custom config
    config_path = Path('custom_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(custom_config, f)
    
    # Use custom config
    generator = SyntheticDataGenerator(config_path=config_path, seed=789)
    
    dataset = generator.generate_dataset(task_type='regression')
    X_train, y_train = dataset['train']
    
    print(f"Dataset shape: {X_train.shape}")
    print(f"Missing values: {np.isnan(X_train).sum()} ({np.isnan(X_train).mean()*100:.1f}%)")
    
    # Clean up
    config_path.unlink()
    
    return dataset


def example_visualization():
    """Example of visualizing generated data."""
    print("\n" + "=" * 50)
    print("DATA VISUALIZATION")
    print("=" * 50)
    
    generator = SyntheticDataGenerator(seed=999)
    
    dataset = generator.generate_dataset(
        n_samples=300,
        n_features=4,
        task_type='classification'
    )
    
    X_train, y_train = dataset['train']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Feature distributions
    for i, ax in enumerate(axes.flat):
        if i < X_train.shape[1]:
            ax.hist(X_train[:, i], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Feature {i+1} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
    
    plt.suptitle('Generated Feature Distributions')
    plt.tight_layout()
    
    # Save figure
    save_path = Path('generated_features.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()
    
    # Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = np.corrcoef(X_train.T)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, 
                square=True, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    save_path = Path('correlation_matrix.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved correlation matrix to {save_path}")
    plt.close()
    
    return dataset


def example_save_load():
    """Example of saving and loading datasets."""
    print("\n" + "=" * 50)
    print("SAVE AND LOAD DATASETS")
    print("=" * 50)
    
    generator = SyntheticDataGenerator(seed=111)
    
    # Generate dataset
    dataset = generator.generate_dataset(
        n_samples=200,
        n_features=6,
        task_type='regression'
    )
    
    # Save in different formats
    save_dir = Path('saved_datasets')
    save_dir.mkdir(exist_ok=True)
    
    # Save as NPZ
    npz_path = save_dir / 'dataset.npz'
    generator.save_dataset(dataset, npz_path, format='npz')
    print(f"Saved as NPZ: {npz_path}")
    
    # Save as CSV
    csv_path = save_dir / 'dataset.csv'
    generator.save_dataset(dataset, csv_path, format='csv')
    print(f"Saved as CSV: {csv_path.parent / f'{csv_path.stem}_train.csv'}")
    
    # Save as pickle
    pkl_path = save_dir / 'dataset.pkl'
    generator.save_dataset(dataset, pkl_path, format='pkl')
    print(f"Saved as pickle: {pkl_path}")
    
    # Load dataset
    loaded_dataset = SyntheticDataGenerator.load_dataset(npz_path, format='npz')
    X_train_loaded, y_train_loaded = loaded_dataset['train']
    X_train_orig, y_train_orig = dataset['train']
    
    # Verify loaded data matches original
    assert np.allclose(X_train_loaded, X_train_orig)
    assert np.allclose(y_train_loaded, y_train_orig)
    print("✓ Loaded dataset matches original")
    
    # Clean up
    import shutil
    shutil.rmtree(save_dir)
    
    return dataset


def main():
    """Run all examples."""
    print("\n🚀 TABPFN SYNTHETIC DATA GENERATION EXAMPLES\n")
    
    # Run examples
    #example_basic_generation()
    #example_with_metadata()
    #example_batch_generation()
    #example_custom_config()
    example_visualization()
    #example_save_load()
    
    print("\n✅ All examples completed successfully!")
    print("\nYou can now use the SyntheticDataGenerator in your own projects.")
    print("Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
