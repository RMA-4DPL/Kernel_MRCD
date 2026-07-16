import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

def custom_rbf_kernel(x1, x2, gamma=1.0):
    """
    Compute the RBF kernel between two vectors x1 and x2.
    
    Parameters:
    - x1, x2: Input vectors.
    - gamma: Kernel parameter.
    
    Returns:
    - Kernel value (scalar).
    """
    squared_distance = np.sum((x1 - x2) ** 2)
    return np.exp(-gamma * squared_distance)

def custom_linear_kernel(x1, x2):
    return x1 @ x2

import numpy as np
from scipy.spatial.distance import cdist, pdist


class LinKernel:
    """Linear kernel: K(x1, x2) = x1 @ x2.T"""

    def __call__(self, x1, x2):
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return x1 @ x2.T


class RbfKernel:
    """Gaussian RBF kernel: K(x1, x2) = exp(-||x1 - x2||^2 / sigma2)"""

    def __init__(self, sigma2=1):
        self.sigma2 = sigma2

    def __call__(self, x1, x2):
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        sqdist = cdist(x1, x2, metric="sqeuclidean")
        return np.exp(-sqdist / self.sigma2)


class AutoRbfKernel(RbfKernel):
    """RbfKernel whose bandwidth is set automatically from the data, as the
    squared median pairwise Euclidean distance."""

    def __init__(self, x):
        x = np.asarray(x)
        sigma2 = np.median(pdist(x)) ** 2
        super().__init__(sigma2)


def get_kernel(kernel_name):
    kernels = {'linear': LinKernel(),
               'rbf': RbfKernel()}
    
    if kernel_name in kernels:
        return kernels[kernel_name]
    else:
        print(f"{kernel_name} is not currently implemented.")