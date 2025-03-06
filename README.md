# SONSC: Self-Organizing Neural-Selective Clustering Algorithm

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Introduction
SONSC is a self-organizing clustering algorithm that combines neural networks and selective mechanisms to automatically determine the optimal number of clusters. The algorithm implements an innovative approach to clustering by utilizing both cohesion and separation metrics.

## Key Features
- Automatic determination of optimal cluster numbers
- Self-organizing mechanism for improved clustering quality
- RCI (Relative Clustering Index) calculation
- Scikit-learn API compatibility
- Support for high-dimensional data
- Tested on standard datasets (MNIST, IRIS, CIFAR-10)

## Installation

```bash
pip install git+https://github.com/YourUsername/SONSC-Algorithm.git
```

## Quick Start

### Basic Example
```python
from sonsc import SONSC
import numpy as np

# Create synthetic data
X = np.random.randn(1000, 2)

# Initialize SONSC
model = SONSC(k_initial=2)

# Fit model
model.fit(X)

# Get cluster labels
labels = model.labels_

# Get cluster centers
centers = model.cluster_centers_

# Get optimal number of clusters
optimal_k = model.k_

# Get RCI score
rci_score = model.best_rci_
```

### MNIST Example
```python
from sklearn.datasets import fetch_openml
from sonsc import SONSC
import matplotlib.pyplot as plt

# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X[:5000]  # Use subset for faster processing
X = X / 255.0  # Normalize data

# Fit SONSC
model = SONSC(k_initial=2)
model.fit(X)

# Print results
print(f"Optimal number of clusters: {model.k_}")
print(f"RCI Score: {model.best_rci_:.4f}")
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| k_initial | Initial number of clusters | 2 |
| max_iterations | Maximum number of iterations | 100 |
| tolerance | Convergence threshold | 1e-4 |
| random_state | Random seed for reproducibility | None |

## Attributes

| Attribute | Description |
|-----------|-------------|
| labels_ | Cluster labels for each sample |
| cluster_centers_ | Coordinates of cluster centers |
| k_ | Optimal number of clusters found |
| best_rci_ | Best RCI score achieved |

## Evaluation Metrics

### RCI (Relative Clustering Index)
```math
RCI = \frac{S - C}{S + C}
```
where:
- C: Cluster cohesion (intra-cluster similarity)
- S: Cluster separation (inter-cluster distance)

## Experimental Results

### MNIST Dataset
- Optimal clusters: [number]
- RCI Score: [value]
- Execution time: [time]

### IRIS Dataset
- Optimal clusters: [number]
- RCI Score: [value]
- Execution time: [time]

### CIFAR-10 Dataset
- Optimal clusters: [number]
- RCI Score: [value]
- Execution time: [time]

## Project Structure
```