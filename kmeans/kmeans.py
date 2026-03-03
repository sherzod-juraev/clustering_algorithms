from ._kmeans_pp import KmeansPP
from ._validator import validation_params, validation_fit
from typing import Literal
from scipy.spatial.distance import cdist
import numpy as np


class Kmeans:
    """
    Implementation of the KMeans clustering algorithm with support for KMeans++
    and random initialization. Includes input validation via decorators.

    This class allows you to cluster data into k clusters, iteratively updating
    cluster centroids until convergence. It also ensures parameters and usage
    are validated to prevent runtime errors or misconfigurations.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form.
    max_iter : int
        Maximum number of iterations for the KMeans algorithm.
    metric : {'euclidean', 'chebyshev'}
        Distance metric used to compute distances between points and centroids.
    tol : float
        Convergence tolerance. If all centroid changes are below this value, the algorithm stops.
    init : {'kmeans++', 'random'}
        Method for initializing cluster centroids.
    random_state : int or None
        Seed for random number generator to ensure reproducibility.
    centroids : np.ndarray or None
        Array of cluster centroid coordinates after fitting. Shape (n_clusters, n_features).
    labels_ : np.ndarray or None
        Array of cluster labels for each point after fitting. Shape (n_samples,).
    __fitted : bool
        Internal flag indicating whether the model has been fitted.
    """
    @validation_params
    def __init__(self,
                 n_clusters: int = 2,
                 max_iter: int = 100,
                 metric: Literal['euclidean', 'chebyshev'] = 'chebyshev',
                 tol: float = 1e-4,
                 init: Literal['kmeans++', 'random'] = 'kmeans++',
                 random_state: int | None = None
                 ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric=metric
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.__fitted = False


    def __initialize_centroids(self, X: np.ndarray):
        """
        Initialize cluster centroids based on the specified initialization method.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape (n_samples, n_features).

        Notes
        -----
        - If `init='kmeans++'`, centroids are initialized using KMeans++ algorithm.
        - If `init='random'`, centroids are selected randomly from the data points.
        """

        if self.init == 'kmeans++':
            self.centroids = KmeansPP(self.n_clusters, random_state=self.random_state).initialize_centroids(X)
        elif self.init == 'random':
            n_samples, m_features = X.shape[0], X.shape[1]
            rng = np.random.default_rng(self.random_state)
            ind = rng.choice(n_samples, size=self.n_clusters, replace=False)
            self.centroids = X[ind].copy()
        else:
            raise ValueError('Invalid init method')

    def __calculate_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the nearest centroid for each sample in the dataset.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of cluster labels (indices of nearest centroid) for each sample.
        """

        dist_sq = cdist(X, self.centroids, metric=self.metric)
        cluster_labels = dist_sq.argmin(axis=1)
        return cluster_labels

    def __update_centroids(self, X: np.ndarray):
        """
        Update cluster centroids based on current cluster assignments.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape (n_samples, n_features).

        Notes
        -----
        - Each centroid is updated as the mean of points assigned to that cluster.
        - If a cluster has no assigned points, its previous centroid remains unchanged.
        """
        for i in range(self.n_clusters):
            centroid = X[self.labels_ == i]
            if len(centroid) > 0:
                self.centroids[i] = centroid.mean(axis=0)


    @validation_fit
    def fit(self, X: np.ndarray) -> 'Kmeans':
        """
        Fit the KMeans clustering model to the dataset.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape (n_samples, n_features).

        Returns
        -------
        self : Kmeans
            Returns the fitted KMeans instance with updated centroids and labels.

        Notes
        -----
        - Iteratively updates centroids until convergence or until `max_iter` is reached.
        - Convergence is determined by whether all centroid movements are below `tol`.
        - After fitting, `self.labels_` contains the cluster label for each point.
        """
        self.__initialize_centroids(X)
        for i in range(self.max_iter):
            self.labels_ = self.__calculate_distance(X)
            old_centroids = self.centroids.copy()
            self.__update_centroids(X)
            if np.allclose(old_centroids, self.centroids, atol=self.tol):
                break
        self.__fitted = True
        return self


    @validation_fit
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels to new data points based on fitted centroids.

        Parameters
        ----------
        X : np.ndarray
            New dataset of shape (n_samples, n_features) to assign to clusters.

        Returns
        -------
        np.ndarray
            Cluster labels for each data point.

        Raises
        ------
        Exception
            If the model has not been fitted yet. `fit` must be called first.

        Notes
        -----
        - Uses the distance metric specified at initialization to assign clusters.
        """

        if not self.__fitted:
            raise Exception("Model is not fitted yet. Call `fit` first")
        cluster_labels = self.__calculate_distance(X)
        return cluster_labels