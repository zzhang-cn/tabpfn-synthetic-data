# TabPFN Synthetic Data Generation

A reimplementation of the synthetic data generation process from **"Accurate predictions on small data with a tabular foundation model"** (Hollmann et al., Nature 2025). This repository provides a modular, extensible framework for generating synthetic tabular datasets using Structural Causal Models (SCMs), following the methodology used to train TabPFN.

## 📚 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Understanding the Code](#-understanding-the-code)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Testing](#-testing)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

TabPFN achieves state-of-the-art performance on small tabular datasets by being trained on millions of synthetic datasets. This repository implements the synthetic data generation pipeline that creates these diverse, realistic datasets through:

1. **Causal Graph Generation**: Creating directed acyclic graphs (DAGs) that represent causal relationships
2. **Computational Edge Functions**: Assigning various transformations (neural networks, discretization, decision trees) to edges
3. **Data Propagation**: Flowing data through the causal graph
4. **Post-processing**: Applying realistic transformations (warping, quantization, missing values)

## ✨ Key Features

- **🔄 Structural Causal Models**: Generate data with complex causal relationships
- **🧠 Multiple Edge Types**: Neural networks, discretization, decision trees, and noise functions
- **📊 Realistic Post-processing**: Kumaraswamy warping, quantization, missing value introduction
- **🔧 Highly Configurable**: YAML-based configuration system
- **🎲 Reproducible**: Seed-based random generation for reproducibility
- **💾 Multiple Export Formats**: Save as NPZ, CSV, or pickle
- **🧪 Well-tested**: Comprehensive test suite included
- **📖 Documented**: Extensive documentation and examples

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tabpfn-synthetic-data.git
cd tabpfn-synthetic-data

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
# Run tests to verify everything is working
python tests/test_basic.py
```

## 🎮 Quick Start

```python
from src.data_generation import SyntheticDataGenerator

# Create a generator
generator = SyntheticDataGenerator(seed=42)

# Generate a dataset (task type auto-detected based on selected target)
dataset = generator.generate_dataset(
    n_samples=1000,
    n_features=20,
    task_type='auto'  # Can be 'classification', 'regression', or 'auto'
)

# Extract the data
X_train, y_train = dataset['train']
X_test, y_test = dataset['test']

print(f"Training shape: {X_train.shape}")
print(f"Task type: {dataset['task_type']}")
if dataset['task_type'] == 'classification':
    print(f"Number of classes: {len(np.unique(y_train))}")
```

## 🧠 Understanding the Code

### Core Concepts

The synthetic data generation follows the **Structural Causal Model (SCM)** framework. Here's how to understand the codebase:

#### 1. **The Generation Pipeline**

```
Config → Graph Generation → Edge Assignment → Data Initialization → Propagation → Post-processing → Dataset
```

Each step is modular and can be understood independently:

#### 2. **Key Components Explained**

**📊 Causal Graph (`src/causal_models/`)**
- **What it does**: Creates the "skeleton" of relationships between variables
- **Key insight**: Not all variables directly influence each other; the graph defines which variables can affect others
- **Code to explore**: `graph_generator.py` - see how different graph types (scale-free, Erdős-Rényi) create different dependency structures

**🔄 Edge Functions (`src/computational_edges/`)**
- **What it does**: Defines HOW one variable influences another
- **Key insight**: Real-world relationships aren't just linear; they can be neural networks, step functions, or discretizations
- **Code to explore**: 
  - `neural_network.py` - Complex non-linear transformations
  - `discretization.py` - Continuous → categorical transformations
  - `decision_tree.py` - Rule-based relationships

**🎲 Initialization (`src/data_generation/initialization.py`)**
- **What it does**: Creates the "root" values that flow through the graph
- **Key insight**: Samples can be correlated (non-independence) to create more realistic datasets
- **Code to explore**: `_apply_non_independence()` method - see how prototypes create sample clusters

**🔧 Post-processing (`src/data_generation/post_processing.py`)**
- **What it does**: Makes synthetic data more realistic
- **Key insight**: Real data has imperfections - missing values, quantization, non-linear distortions
- **Code to explore**: 
  - `_apply_kumaraswamy_warping()` - Non-linear feature transformation
  - `_introduce_missing_values()` - Realistic missing data patterns (MCAR, MAR, MNAR)

### Understanding the Flow

Here's a step-by-step walkthrough of what happens when you call `generate_dataset()`:

```python
# In src/data_generation/generator.py

def generate_dataset():
    # 1. Sample parameters (how many samples, features?)
    n_samples = self._sample_n_samples()  # e.g., 1000
    n_features = self._sample_n_features()  # e.g., 20
    
    # 2. Create a causal graph (who influences whom?)
    graph = self.graph_generator.generate_graph(n_nodes=30)
    # Result: A DAG with 30 nodes and various connections
    
    # 3. Assign functions to edges (how do they influence?)
    for edge in graph.edges():
        edge_functions[edge] = self.edge_factory.create_random_edge()
    # Result: Edge (5→12) might be a neural network, (3→7) might be discretization
    
    # 4. Create root data (starting values)
    init_data = self.init_sampler.sample(n_samples, graph)
    # Result: Random values for nodes with no parents
    
    # 5. Propagate through graph (compute all values)
    node_data = scm.sample(n_samples, init_data)
    # Result: Each node now has data based on its parents
    
    # 6. Select features and target (what do we observe?)
    feature_nodes = [1, 5, 8, 12, ...]  # n_features nodes
    target_node = 15  # One node as target
    
    # 7. Apply post-processing (make it realistic)
    X, y = self.post_processor.process(X, y)
    # Result: Added missing values, warping, etc.
    
    return dataset
```

### Key Design Patterns

1. **Factory Pattern**: `EdgeFunctionFactory` creates different edge types based on configuration
2. **Builder Pattern**: The generator builds datasets step-by-step
3. **Strategy Pattern**: Different post-processing strategies can be applied
4. **Dependency Injection**: Components receive configuration and random generators

### Debugging Tips

1. **Enable logging**: Set logging level to DEBUG in config
2. **Use metadata**: Call `generate_dataset(return_metadata=True)` to inspect the generation process
3. **Visualize graphs**: Use NetworkX to visualize the generated causal graphs
4. **Start small**: Generate tiny datasets (5 samples, 3 features) to trace through the logic

## 📁 Project Structure

```
tabpfn-synthetic-data/
├── configs/
│   └── default_config.yaml      # Default configuration
├── src/
│   ├── causal_models/           # Graph generation & SCM
│   │   ├── graph_generator.py   # Creates causal DAGs
│   │   └── scm.py              # Structural Causal Model
│   ├── computational_edges/     # Edge transformations
│   │   ├── edge_functions.py   # Base classes & factory
│   │   ├── neural_network.py   # NN transformations
│   │   ├── discretization.py   # Continuous → discrete
│   │   └── decision_tree.py    # Tree-based functions
│   ├── data_generation/         # Main generation pipeline
│   │   ├── generator.py        # Main generator class
│   │   ├── initialization.py   # Root node sampling
│   │   └── post_processing.py  # Realistic transformations
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration management
│       └── distributions.py    # Distribution sampling
├── tests/                       # Test suite
├── examples/                    # Usage examples
└── notebooks/                   # Jupyter notebooks

```

## ⚙️ Configuration

The generation process is controlled by `configs/default_config.yaml`:

```yaml
# Dataset parameters
dataset:
  n_samples:
    min: 100
    max: 2048
  n_features:
    distribution: "beta"
    params: {alpha: 0.95, beta: 8.0}
    range: [1, 160]

# Graph structure
graph:
  graph_type: "scale_free"  # or "erdos_renyi", "barabasi_albert"
  n_nodes:
    distribution: "log_uniform"
    min: 5
    max: 100

# Edge functions
edges:
  type_probabilities:
    neural_network: 0.4
    discretization: 0.2
    decision_tree: 0.2
    noise: 0.2

# Post-processing
post_processing:
  missing_values:
    enabled: true
    probability: 0.2
    missing_rate: [0.0, 0.3]
```

## 📊 Examples

### Generate Multiple Datasets

```python
# Generate a batch of datasets
datasets = generator.generate_batch(
    n_datasets=100,
    task_type='classification'
)
```

### Custom Configuration

```python
# Use custom configuration
generator = SyntheticDataGenerator(
    config_path='my_config.yaml',
    seed=42
)
```

### Extract Metadata

```python
# Get generation metadata
dataset, metadata = generator.generate_dataset(
    return_metadata=True
)

print(f"Graph type: {metadata['graph_stats']['n_edges']} edges")
print(f"Edge types used: {metadata['edge_types']}")
```

See `examples/basic_usage.py` for more comprehensive examples.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
python tests/test_basic.py
```

## 📝 Citation

If you use this code in your research, please cite the original TabPFN paper:

```bibtex
@article{hollmann2025tabpfn,
  title={Accurate predictions on small data with a tabular foundation model},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and 
          Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and 
          Schirrmeister, Robin Tibor and Hutter, Frank},
  journal={Nature},
  volume={637},
  pages={319--326},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The TabPFN team for their groundbreaking work
- The paper authors: Noah Hollmann, Samuel Müller, and colleagues
- The open-source community for the amazing tools that make this possible

## 📬 Contact

For questions or feedback, please open an issue on GitHub or contact zzhang@gmail.com

---

**Note**: This is an independent reimplementation for research and educational purposes, generated by Claude Opus 4.1 based on the methodology described in the TabPFN paper. For the official TabPFN implementation, please refer to the [original repository](https://github.com/automl/TabPFN).
