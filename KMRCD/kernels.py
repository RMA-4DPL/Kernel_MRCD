"""
Kernel models used by :mod:`KMRCD.kmrcd`.

``KMRCD.m`` operates on an arbitrary kernel object exposing a ``compute``
method (``kModel.compute(X1, X2)`` -> Gram matrix). The original MATLAB
project references ``LinKernel``, ``RbfKernel`` and ``AutoRbfKernel`` classes
that are not part of the ``KMRCD`` folder (they live elsewhere in the
upstream ivranckx/kMRCD repository). They are reimplemented here, following
the standard definitions, so that the converted algorithm is runnable.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist


class LinKernel:
    """Linear kernel: K(x1, x2) = x1 @ x2.T"""

    def compute(self, x1, x2):
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return x1 @ x2.T


class RbfKernel:
    """Gaussian RBF kernel: K(x1, x2) = exp(-||x1 - x2||^2 / sigma2)"""

    def __init__(self, sigma2):
        self.sigma2 = sigma2

    def compute(self, x1, x2):
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
