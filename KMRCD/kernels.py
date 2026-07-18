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
from scipy.spatial.distance import cdist


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
    
    """ --- Previous implementation, kept for comparison ---------------------
    
    Built the full O(n^2) distance matrix (via sklearn's parallel
    pairwise_distances), subsampled points down to _MAX_POINTS first to
    bound its memory, and used a sentinel-diagonal trick so the exact
    np.median could be taken over the whole matrix without a separate
    O(n^2) extraction step. Superseded because np.median on the full
    matrix is single-threaded and became the dominant cost at large n
    (~32s / ~10GB at n=50,000), and because the point-subsampling was
    only needed to bound that O(n^2) matrix, which the current
    direct-pair-sampling approach never builds.
    
    _MAX_POINTS = 50_000
    
    def __init__(self, x):
        x = np.asarray(x).astype(np.float32)
        n = x.shape[0]
        if n > self._MAX_POINTS:
            stride = int(np.ceil(n / self._MAX_POINTS))
            x = x[::stride]
            n = x.shape[0]
        full_sqdist = pairwise_distances(x, metric="sqeuclidean", n_jobs=-1)
        # The diagonal (zeros) would bias the median low. Overwriting it
        # with an equal split of sentinel values below/above the true
        # min/max pushes those entries to the extreme ends of the sorted
        # order, so they land outside the median position without needing
        # to extract the off-diagonal entries into a separate O(n^2) array.
        lo = n // 2
        diag_vals = np.empty(n, dtype=full_sqdist.dtype)
        diag_vals[:lo] = -np.inf
        diag_vals[lo:] = np.inf
        np.fill_diagonal(full_sqdist, diag_vals)
        sigma = np.sqrt(np.median(full_sqdist)).astype(np.float64)
        super().__init__(sigma)
    # ------------------------------------------------------------------------
    """

    _MEDIAN_SAMPLE_SIZE = 1_000_000

    def __init__(self, x):
        x = np.asarray(x).astype(np.float32)
        n = x.shape[0]

        # The median is a bandwidth heuristic, not an exact quantity, so
        # instead of building the full O(n^2) distance matrix, estimate it
        # from a large random sample of point pairs (i != j).
        rng = np.random.default_rng()
        m = min(self._MEDIAN_SAMPLE_SIZE, n * (n - 1) // 2)
        i = rng.integers(0, n, size=m, dtype=np.int32)
        j = rng.integers(0, n, size=m, dtype=np.int32)
        clash = i == j
        while np.any(clash):
            j[clash] = rng.integers(0, n, size=clash.sum())
            clash = i == j

        sqdist = np.sum((x[i] - x[j]) ** 2, axis=1)
        sigma = np.sqrt(np.median(sqdist)).astype(np.float64)
        super().__init__(sigma)
