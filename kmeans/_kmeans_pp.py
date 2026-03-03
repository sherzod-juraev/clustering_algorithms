from scipy.spatial.distance import cdist
from typing import Literal
import numpy as np


class KmeansPP:
    """
    Implementation of the KMeans++ initialization algorithm for clustering.

    KMeans++ is an improved method for selecting the initial centroids for the KMeans
    algorithm. It increases the likelihood of faster convergence and reduces the chances
    of poor clustering due to unlucky random initialization.

    Attributes
    ----------
    n_clusters : int
        Number of centroids (clusters) to initialize.
    metric : {'euclidean', 'chebyshev'}
        Distance metric used to compute distances between points and centroids.
    rng : np.random.Generator
        NumPy random number generator initialized with a given seed for reproducibility.
    """

    def __init__(
            self,
            n_clusters: int = 2,
            metric: Literal['euclidean', 'chebyshev'] = 'chebyshev',
            random_state: int | None = None
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.rng = np.random.default_rng(random_state)

    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the KMeans++ algorithm.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of initialized centroids of shape (n_clusters, n_features).

        Notes
        -----
        The KMeans++ initialization algorithm works as follows:
            1. Choose the first centroid randomly from the data points.
            2. For each remaining centroid:
                a. Compute squared distances from each point to the nearest existing centroid.
                b. Compute probabilities proportional to these distances.
                c. Select a new centroid according to these probabilities.
        """
        n = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        first_ind = self.rng.integers(0, n)
        centroids[0] = X[first_ind].copy()
        for i in range(1, self.n_clusters):
            prob = self.__calculate_probability(X, centroids[:i, :])
            new_cen_ind = self.rng.choice(n, p=prob)
            centroids[i] = X[new_cen_ind].copy()
        return centroids

    def __calculate_probability(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute the probability of each sample being chosen as the next centroid.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape (n_samples, n_features).
        centroids : np.ndarray
            Current centroids of shape (current_clusters, n_features).

        Returns
        -------
        np.ndarray
            Array of probabilities for each point, summing to 1.

        Notes
        -----
        - Each probability is proportional to the squared distance to the nearest existing centroid.
        - A small epsilon (1e-12) is added to prevent division by zero.
        """
        dist_sq = cdist(X, centroids, metric=self.metric)
        min_dist_sq = dist_sq.min(axis=1)
        prob = min_dist_sq / (min_dist_sq.sum() + 1e-12)
        return prob