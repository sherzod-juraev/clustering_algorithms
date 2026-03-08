# ML Clustering

**ML Clustering** is a lightweight educational implementation of classical **unsupervised clustering algorithms** written in pure Python using **NumPy**.

The goal of this project is to provide **clean, readable, and modular implementations** of clustering algorithms while maintaining a structure similar to real machine learning libraries.

Currently implemented algorithms:

* **K-Means** [More info →](./kmeans/README.md#k-means)
* **DBSCAN** [More info →](./dbscan/README.md#dbscan)

# Implemented Algorithms

## K-Means

A centroid-based clustering algorithm that partitions data into **k clusters**.

Features:

* KMeans clustering
* **KMeans++ centroid initialization**
* random centroid initialization
* multiple distance metrics
* convergence tolerance control
* parameter validation via decorators
* `fit` and `predict` interface

Supported metrics:

```
euclidean
chebyshev
```

## DBSCAN

A **density-based clustering algorithm** capable of discovering clusters of arbitrary shapes and detecting noise points.

Features:

* automatic cluster discovery
* noise point detection
* BFS-style cluster expansion
* custom queue implementation using linked list
* distance metric selection
* parameter validation

Supported metrics:

```
euclidean
chebyshev
```

## Project Structure

```
ml_clustering
│
├── kmeans
│   ├── __init__.py
│   ├── kmeans.py
│   ├── _kmeans_pp.py
│   └── _validator.py
│
├── dbscan
│   ├── __init__.py
│   ├── dbscan.py
│   ├── _validator.py
│   └── queue
│       ├── __init__.py
│       ├── exception.py
│       ├── node.py
│       └── queue.py
│
└── README.md
```

## Design Principles

This project follows several design principles:

### Modular Architecture

Each algorithm is implemented as an independent module.

### Input Validation

Decorators are used to validate parameters and data before execution.

### Educational Clarity

The code prioritizes readability and transparency over heavy optimization.

### Reproducibility

Randomized operations support `random_state`.

## License

MIT License
