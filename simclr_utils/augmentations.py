import numpy as np
def time_flip_transform_vectorized(X):
    """
    Reversing the direction of time
    """
    return X[:, ::-1, :]