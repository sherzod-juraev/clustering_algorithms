# DBSCAN

A pure Python implementation of the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm using **NumPy** and a custom **FIFO Queue** structure.

This implementation focuses on **algorithmic clarity and educational value**, showing how density-based clustering works internally without relying on external machine learning libraries.

## Overview

**DBSCAN** is an unsupervised clustering algorithm that groups together points that are **densely packed** while marking points in low-density regions as **noise**.

Unlike algorithms such as K-Means, DBSCAN:

* **does not require specifying the number of clusters**
* **can detect arbitrarily shaped clusters**
* **automatically identifies noise/outliers**

Clusters are formed by expanding dense regions using a **breadth-first search–like process**.

## Key Concepts

DBSCAN relies on two parameters:

### `eps`

Maximum distance between two points for them to be considered neighbors.

### `minpnts`

Minimum number of points required within the `eps` radius to form a **dense region**.

# Point Types

During clustering, points are classified into three categories:

### Core Point

A point with at least `minpnts` neighbors within distance `eps`.

### Border Point

A point that lies within the neighborhood of a core point but does not have enough neighbors to be a core point itself.

### Noise Point

A point that is not reachable from any core point.

## Algorithm Workflow

The algorithm proceeds as follows:

1. Mark all points as **unvisited**.
2. Iterate through each point in the dataset.
3. If the point has already been visited, skip it.
4. Compute distances from the point to all other points.
5. Identify neighbors within distance `eps`.

If the number of neighbors is **less than `minpnts`**:

* Mark the point as **noise**.

If the number of neighbors is **greater than or equal to `minpnts`**:

* Start a **new cluster**.
* Assign the point and its neighbors to that cluster.
* Use a **queue-based expansion** process:

  * Add neighbors to a queue.
  * Iteratively process points from the queue.
  * Expand the cluster whenever a new **core point** is discovered.

This process continues until the cluster can no longer be expanded.

## Distance Metrics

This implementation supports two distance metrics:

```
euclidean
chebyshev
```

Distances are computed using **NumPy vectorized operations**.

## Labels

After fitting the model, each point receives a label:

| Label     | Meaning                                          |
| --------- | ------------------------------------------------ |
| -2        | Unvisited point (internal state during training) |
| -1        | Noise point                                      |
| 0,1,2,... | Cluster identifiers                              |

## Parameters

| Parameter | Type         | Description                                           |
| --------- | ------------ | ----------------------------------------------------- |
| eps       | float or int | Neighborhood radius                                   |
| minpnts   | int          | Minimum number of neighbors required for a core point |
| metric    | str          | Distance metric (`euclidean` or `chebyshev`)          |

## Example Usage

```python
import numpy as np
from dbscan import DBSCAN

X = np.array([
    [1,2],
    [2,2],
    [2,3],
    [8,7],
    [8,8],
    [25,80]
])

model = DBSCAN(
    eps=2.0,
    minpnts=2,
    metric="euclidean"
)

model.fit(X)

print(model.labels)
```

## Queue-Based Cluster Expansion

To expand clusters efficiently, this implementation uses a **custom FIFO queue** implemented with a **linked list**.

The queue is responsible for managing the **breadth-first expansion** of density-connected points.

The expansion process works like this:

1. A core point is discovered.
2. Its neighbors are added to a queue.
3. Each neighbor is processed sequentially.
4. If a neighbor is also a core point, its neighbors are added to the queue.
5. The cluster grows until no new density-connected points remain.

This approach closely follows the original **DBSCAN cluster expansion logic**.

## Input Validation

Validation is implemented using **decorators**.

The following checks are performed:

### Parameter validation

* `eps` must be `float` or `int`
* `minpnts` must be `int`
* `metric` must be `"euclidean"` or `"chebyshev"`

### Fit validation

* Input must be `numpy.ndarray`
* Input must be **two-dimensional**

These checks help prevent runtime errors and ensure the algorithm receives valid input data.

---

## Notes

* This implementation prioritizes **clarity and algorithm transparency** rather than maximum performance.
* Cluster expansion follows a **queue-driven breadth-first search pattern**.
* The algorithm automatically detects **outliers and noise points**.

---

## Educational Purpose

This implementation is designed to help understand:

* density-based clustering
* cluster expansion mechanics
* neighbor search using distance metrics
* queue-based traversal in clustering algorithms