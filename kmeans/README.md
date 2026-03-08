# K-Means

A clean and educational implementation of the **K-Means clustering algorithm** written in pure Python using **NumPy** and **SciPy**.

This module provides a clustering model with support for **KMeans++ initialization**, **multiple distance metrics**, and **input validation via decorators**.

## Overview

K-Means is an **unsupervised learning algorithm** used to partition data into `k` clusters.

The algorithm works by iteratively:

1. Initializing cluster centroids
2. Assigning each point to the nearest centroid
3. Updating centroids based on cluster means
4. Repeating until convergence

This implementation includes:

* K-Means++ initialization
* Random centroid initialization
* Distance metric selection
* Parameter validation
* Predicting clusters for new data

## Features

* K-Means clustering implementation
* K-Means++ centroid initialization
* Random centroid initialization
* Distance metrics support
* Decorator-based parameter validation
* Predict API for new data

## Supported Distance Metrics

The following distance metrics are supported:

```
euclidean
chebyshev
```

Distance computation is handled using **scipy.spatial.distance.cdist**.

## Parameters

| Parameter    | Type        | Description                                    |
| ------------ | ----------- | ---------------------------------------------- |
| n_clusters   | int         | Number of clusters                             |
| max_iter     | int         | Maximum number of iterations                   |
| metric       | str         | Distance metric (`euclidean`, `chebyshev`)     |
| tol          | float       | Convergence tolerance                          |
| init         | str         | Centroid initialization (`kmeans++`, `random`) |
| random_state | int or None | Random seed for reproducibility                |

## Example Usage

```python
import numpy as np
from kmeans import Kmeans

X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

model = Kmeans(
    n_clusters=2,
    init="kmeans++",
    random_state=42
)

model.fit(X)

print(model.labels_)
print(model.centroids)
```

## Algorithm Workflow

The algorithm follows these steps:

1. Initialize centroids using:

   * **K-Means++**
   * or **random selection**

2. Compute distance between each point and centroid.

3. Assign each point to the nearest centroid.

4. Update centroids by computing the **mean of assigned points**.

5. Repeat until:

   * centroids stop changing within `tol`
   * or `max_iter` is reached.

## K-Means++ Initialization

K-Means++ improves centroid initialization by spreading centroids across the dataset.

Steps:

1. Choose first centroid randomly.
2. Compute distance from each point to nearest centroid.
3. Select next centroid with probability proportional to squared distance.
4. Repeat until all centroids are chosen.

Benefits:

* Faster convergence
* Better clustering stability
* Reduced chance of poor initialization

## Input Validation

Validation is implemented using **decorators**.

### Parameter validation

Ensures:

* Correct parameter types
* Valid parameter ranges
* Valid initialization methods

### Fit validation

Ensures:

* Input must be `numpy.ndarray`
* Input must be **2-dimensional**

## Attributes

After fitting the model:

| Attribute | Description                   |
| --------- | ----------------------------- |
| centroids | Cluster centroid coordinates  |
| labels_   | Cluster label for each sample |

## Notes

* If a cluster receives **no points**, its centroid remains unchanged.
* Distance computations are vectorized using **SciPy** for efficiency.
* The implementation is designed for **educational clarity** and experimentation.