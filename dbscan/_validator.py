from functools import wraps
import numpy as np

# dbscan __init__
def validation_params(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        try:
            if not isinstance(self.eps, float | int):
                raise TypeError(f"eps must be float or int, got {type(self.eps).__name__}")
            if not isinstance(self.minpnts, int):
                raise TypeError(f"minpnts must be int, got {type(self.minpnts).__name__}")
            if self.metric not in ['euclidean', 'chebyshev']:
                raise ValueError(f"metric must be 'euclidean' or 'chebyshev', got '{self.metric}'")
        except Exception as ex:
            del res
            raise ex
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