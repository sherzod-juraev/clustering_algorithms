from typing import Literal
from .queue import Queue
from ._validator import validation_params, validation_fit
import numpy as np


class DBSCAN:
    """
    DBSCAN clustering algorithm implementation using a custom Queue.

    Parameter
    ---------

    eps (float | int):
        Maximum distance to consider points as neighbors.
    minpnts (int):
        Minimum number of points required to form a dense region.
    metric (Literal['euclidean', 'chebyshev']):
        Distance metric to use.
    labels (np.ndarray):
        Cluster labels assigned to each data point after fitting.
        -1 indicates noise points, -2 indicates unvisited points.
    """

    @validation_params
    def __init__(
            self,
            eps: float | int,
            minpnts: int = 4,
            metric: Literal['euclidean', 'chebyshev'] = 'chebyshev'
    ):

        self.eps = eps
        self.minpnts = minpnts
        self.metric = metric

    def calculate_dis(self, core: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Calculate the distance from a core point to all points in X using the chosen metric.

        Args:
            core (np.ndarray): A single point of shape (features,) to compute distances from.
            X (np.ndarray): Array of points of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of distances from core to each point in X.
        """
        if self.metric == 'euclidean':
            dis = np.linalg.norm(X - core, axis=1)
            return dis
        elif self.metric == 'chebyshev':
            dis = np.max(np.abs(X - core), axis=1)
            return dis

    @validation_fit
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """
        Fit the DBSCAN algorithm to the data X and assign cluster labels.

        Args:
            X (np.ndarray): 2D array of shape (n_samples, n_features).

        Returns:
            DBSCAN: The fitted DBSCAN instance with cluster labels in self.labels.

        Notes:
            - Labels: -2 indicates unvisited points, -1 indicates noise, 0,1,... indicate cluster IDs.
            - Uses a custom Queue to manage the BFS-like expansion of core points.
        """

        n = X.shape[0]
        visited = np.full(n, False, dtype=bool)
        self.labels = np.full(n, -2, dtype=int)
        cur_label_id = 0
        for i in range(n):
            if visited[i] == True:
                continue
            visited[i] = True
            dis = self.calculate_dis(X[i, :], X)
            ind: np.ndarray = np.where(dis <= self.eps)[0]
            if dis[ind].shape[0] >= self.minpnts:
                self.labels[ind] = cur_label_id
                queue = Queue()
                queue.add_list(ind.tolist())
                while len(queue) != 0:
                    core_ind = queue.dequeue()
                    if visited[core_ind] == True:
                        continue
                    visited[core_ind] = True
                    dis = self.calculate_dis(X[core_ind, :], X)
                    ind = np.where(dis <= self.eps)[0]
                    if dis[ind].shape[0] >= self.minpnts:  # core point
                        self.labels[ind] = cur_label_id
                        queue.add_list(ind.tolist())
                    else:
                        self.labels[core_ind] = -1  # noise
                cur_label_id += 1  # new class
            else:
                self.labels[i] = -1
        return self