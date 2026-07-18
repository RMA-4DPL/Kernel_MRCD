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
        if x2 is None:
            x2 = x1
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return x1 @ x2.T


class RbfKernel:
    """RBF Kernel: K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))"""

    def __init__(self, sigma=1):
        self.sigma = sigma

    def compute(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        if x1.shape[0] * x2.shape[0] >= 500_000:
            x1_sq = np.einsum("ij,ij->i", x1, x1)
            x2_sq = x1_sq if x2 is x1 else np.einsum("ij,ij->i", x2, x2)
            sqdist = x1_sq[:, None] + x2_sq[None, :] - 2 * x1 @ x2.T
            np.maximum(sqdist, 0, out=sqdist)
        else:
            sqdist = cdist(x1, x2, metric="sqeuclidean")
        return np.exp(-sqdist / (2 * self.sigma**2))


class AutoRbfKernel(RbfKernel):
    """RbfKernel whose bandwidth is set automatically from the data, as the
    squared median pairwise Euclidean distance."""

    def __init__(self, x):
        x = np.asarray(x)
        distances = pdist(x)**2
        sigma = np.sqrt(np.median(distances))
        super().__init__(sigma)
