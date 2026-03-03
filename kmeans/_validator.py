from functools import wraps
import numpy as np

# kmeans __init__
def validation_params(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        if not isinstance(self.n_clusters, int):
            raise TypeError(f"n_clusters must be int, got {type(self.n_clusters).__name__}")
        if not isinstance(self.max_iter, int):
            raise TypeError(f"max_iter must be int, got {type(self.max_iter).__name__}")
        if self.metric not in ['euclidean', 'chebyshev']:
            raise ValueError(f"metric must be 'euclidean' or 'chebyshev', got {self.metric}")
        if not isinstance(self.tol, float):
            raise TypeError(f"tol must be float, got {type(self.tol).__name__}")
        if self.tol < 0 or 1 < self.tol:
            raise ValueError(f"tol must be between 0 and 1, got {self.tol}")
        if self.init not in ['kmeans++', 'random']:
            raise ValueError(f"init must be 'kmeans++' or 'random', got {self.init}")
        if not isinstance(self.random_state, int | None):
            raise TypeError(f"random_state must be int or None, got {self.random_state}")
        return res
    return wrapper

def validation_fit(func):
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy.ndarray, got {type(X).__name__}")
        if X.ndim != 2:
            raise ValueError(f"X.ndim must be 2, got {X.ndim}")
        return func(self, X, *args, **kwargs)
    return wrapper