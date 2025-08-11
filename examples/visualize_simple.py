#!/usr/bin/env python3
"""
Simple example for visualizing TabPFN synthetic datasets.

This script demonstrates how to generate and visualize datasets with
multiple continuous features and categorical target classes using scatter plots.
The target is categorical while features are continuous.
Only unique feature pairs are plotted (avoiding duplicates like (i,j) and (j,i)).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation.generator import SyntheticDataGenerator


def simple_visualization_example():
    """Simple example of generating and visualizing a dataset with multiple continuous features and categorical target."""
    
    print("Creating TabPFN synthetic data generator with custom config...")
    
    # Create custom configuration to ensure:
    # 1. Categorical features have exactly 3 categories
    # 2. Higher probability of discretization edges for categorical targets
    # 3. But still allow continuous features for the feature set
    custom_config = {
        'dataset': {
            'n_samples': {'min': 100, 'max': 2048, 'distribution': 'uniform'},
            'n_features': {'distribution': 'beta', 'params': {'alpha': 0.95, 'beta': 8.0}, 'range': [1, 160]},
            'max_cells': 75000,
            'train_val_split': 0.8
        },
        'graph': {
            'n_nodes': {'distribution': 'uniform', 'min': 5, 'max': 10},
            'redirection_probability': {'distribution': 'gamma', 'alpha': 2.0, 'beta': 1.0},
            'graph_type': 'scale_free',
            'ensure_connected': True
        },
        'edges': {
            'vector_dim': 4,
            'type_probabilities': {
                'linear': 0.2,
                'nonlinear': 0.4,
                'neural_network': 0.2,
                'discretization': 0.2,  # Higher probability for categorical features
                #'decision_tree': 0.3,
            },
            'discretization': {
                'gamma_alpha': 1.0,  # Lower alpha to favor fewer categories
                'gamma_scale': 1.0,  # Lower scale to favor fewer categories
                'max_categories': 3  # Limit to exactly 3 categories
            }
        },
    }
    
    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(custom_config, f, default_flow_style=False, indent=2)
        config_path = f.name
    
    try:
        generator = SyntheticDataGenerator(config_path=config_path)
        
        print("Generating dataset with features...")
        # Generate dataset with features, forced classification
        dataset, metadata = generator.generate_dataset(
            n_samples=200,  # Good size for visualization
            n_features=4,
            task_type='classification',  # Force classification task
            return_metadata=True
        )
        
        # Combine train and test data
        X_train, y_train = dataset['train']
        X_test, y_test = dataset['test']
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        
        print(f"Generated dataset:")
        print(f"  - Samples: {len(X)}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Target classes: {len(np.unique(y))}")
        print(f"  - Task type: {dataset['task_type']}")
        
        # Verify that target has exactly 3 classes
        n_classes = len(np.unique(y))
        print(f"Target has {n_classes} classes")
        
        # Check feature types
        print(f"\nFeature type verification:")
        categorical_features = 0
        continuous_features = 0

        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        for i, feat_meta in enumerate(metadata['feature_metadata']):
            feat_type = feat_meta['type']
            if feat_type == 'categorical':
                categorical_features += 1
                print(f"  Feature {i}: {feat_type} (Node {feat_meta['node_id']}, "
                      f"{feat_meta.get('n_categories', 'N/A')} categories)")
            else:
                continuous_features += 1
                print(f"  Feature {i}: {feat_type} (Node {feat_meta['node_id']}, "
                      f"Dim {feat_meta['component_id']})")
        
        target_meta = metadata['target_metadata'] 
        print(f"  Target: {target_meta['type']} (Node {target_meta['node_id']})")
        
        # Verify all features are continuous
        if continuous_features == len(metadata['feature_metadata']):
            print("✓ All features are continuous as requested")
        else:
            print(f"Warning: {categorical_features} categorical features found, expected all continuous")
        
        # Verify target is categorical
        if target_meta['type'] == 'categorical':
            print("✓ Target is categorical as requested")
        else:
            print(f"Warning: Target is {target_meta['type']}, expected categorical")
        
    finally:
        # Clean up temporary config file
        import os
        if os.path.exists(config_path):
            os.unlink(config_path)
    
    # Create scatter plots for unique feature pairs
    n_features = X.shape[1]
    
    # Generate unique feature pairs (i,j) where i < j to avoid duplicates
    feature_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            feature_pairs.append((i, j, f"Feature {i} vs Feature {j}"))
    
    n_pairs = len(feature_pairs)
    
    # Calculate subplot grid dimensions
    if n_pairs <= 3:
        rows, cols = 1, n_pairs
    elif n_pairs <= 6:
        rows, cols = 2, 3
    elif n_pairs <= 9:
        rows, cols = 3, 3
    elif n_pairs <= 12:
        rows, cols = 3, 4
    elif n_pairs <= 15:
        rows, cols = 3, 5
    else:
        # For larger numbers, use square-ish grid
        rows = int(np.ceil(np.sqrt(n_pairs)))
        cols = int(np.ceil(n_pairs / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle(f'TabPFN Synthetic Dataset - {n_features} Features, {len(np.unique(y))} Target Classes', fontsize=16)
    
    # Handle case where we have only one subplot
    if n_pairs == 1:
        axes = [axes]
    # Handle case where we have one row
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Get unique classes and colors
    unique_classes = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
    
    for plot_idx, (i, j, title) in enumerate(feature_pairs):
        ax = axes[plot_idx] if n_pairs > 1 else axes[0]
        
        # Plot each class with different color/marker
        for class_idx, class_label in enumerate(unique_classes):
            mask = y == class_label
            ax.scatter(
                X[mask, i], X[mask, j],
                c=colors[class_idx % len(colors)],
                marker=markers[class_idx % len(markers)],
                label=f'Class {class_label}',
                alpha=0.7,
                s=40
            )
        
        # Add feature type info to labels
        feat_i_meta = metadata['feature_metadata'][i] if i < len(metadata['feature_metadata']) else None
        feat_j_meta = metadata['feature_metadata'][j] if j < len(metadata['feature_metadata']) else None
        
        xlabel = f'Feature {i}'
        ylabel = f'Feature {j}'
        
        if feat_i_meta:
            if feat_i_meta['type'] == 'categorical':
                xlabel += f" (Cat)"
            else:
                xlabel += " (Cont)"
        
        if feat_j_meta:
            if feat_j_meta['type'] == 'categorical':
                ylabel += f" (Cat)"
            else:
                ylabel += " (Cont)"
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    if n_pairs < rows * cols:
        for plot_idx in range(n_pairs, rows * cols):
            if plot_idx < len(axes):
                axes[plot_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f'dataset_visualization_{n_features}features.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_path}")
    print(f"Generated {n_pairs} unique feature pair plots")
    
    return X, y, fig, metadata


if __name__ == "__main__":
    X, y, fig, metadata = simple_visualization_example()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(f"✓ Generated dataset with {X.shape[1]} features and {len(np.unique(y))} target classes")
    
    # Summary of feature types
    feature_types = [meta['type'] for meta in metadata['feature_metadata']]
    categorical_count = feature_types.count('categorical')
    continuous_count = feature_types.count('continuous')
    
    print(f"✓ Features: {continuous_count} continuous, {categorical_count} categorical")
    print(f"✓ Target: {metadata['target_metadata']['type']}")
    print(f"✓ Target classes: {sorted(np.unique(y).tolist())}")
    
    # Check if requirements are met
    all_features_continuous = continuous_count == len(feature_types)
    target_is_categorical = metadata['target_metadata']['type'] == 'categorical'
    has_3_classes = len(np.unique(y)) == 3
    
    if all_features_continuous and target_is_categorical:
        print("\n🎉 Requirements met:")
        print("   • All features are continuous ✓")
        print("   • Target is categorical ✓") 
        print(f"   • Target has {len(np.unique(y))} classes")
    else:
        print("\n⚠️  Requirements check:")
        print(f"   • All features continuous: {'✓' if all_features_continuous else '✗'}")
        print(f"   • Target categorical: {'✓' if target_is_categorical else '✗'}")
        print(f"   • Target classes: {len(np.unique(y))}")
        if not all_features_continuous or not target_is_categorical:
            print("\n💡 Try running again with different seed or config adjustments")
    
    # Display the plot
    print(f"\nDisplaying visualization...")
    print(f"Visualization saved as: dataset_visualization_{X.shape[1]}features.png")
    plt.show()