"""
Python port of ``KMRCD/utils.m`` (the ``Utils`` static class).

Conventions
-----------
MATLAB is 1-indexed and every function below returned/consumed plain
vectors/matrices; here everything is translated to 0-indexed numpy arrays
with the same formulas, so ordering/indices coming out of these functions
(e.g. ``sdo``, ``spatial_rank``, ...) are directly usable to index numpy
arrays.

``unimcd`` is not part of the ``KMRCD`` folder (it is an external univariate
MCD routine, e.g. from the LIBRA toolbox). ``reweighted_mean`` below is a
faithful translation of ``Utils.reweightedMean`` and already implements the
consistency-correction + reweighting step of the univariate MCD once a raw
h-subset is known. ``unimcd`` completes that algorithm by adding the missing
first step: finding the raw h-subset as the contiguous window (in sorted
order) with minimal variance -- the standard approach (Rousseeuw & Leroy).
"""

import numpy as np
from scipy.stats import chi2, gamma, norm

_EPS = np.finfo(float).eps


def _mad(x, flag=0, axis=0):
    """MATLAB-compatible ``mad``: flag=0 -> mean absolute deviation about the
    mean (MATLAB's default), flag=1 -> median absolute deviation about the
    median. No consistency constant is applied, matching MATLAB's ``mad``."""
    x = np.asarray(x, dtype=float)
    if flag == 0:
        center = np.mean(x, axis=axis, keepdims=True)
    else:
        center = np.median(x, axis=axis, keepdims=True)
    dev = np.abs(x - center)
    return np.mean(dev, axis=axis) if flag == 0 else np.median(dev, axis=axis)


def mcd_cons(p, alpha):
    qalpha = chi2.ppf(alpha, df=p)
    caI = gamma.cdf(qalpha / 2, a=p / 2 + 1, scale=1) / alpha
    return 1.0 / caI


# ---------------------------------------------------------------------
# Initial estimators; return the h-subset indices in ascending order.
# ---------------------------------------------------------------------

def spatial_median(K):
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    assert n > 0
    gamma_ = np.ones(n) / n
    for _ in range(10):
        w = 1.0 / np.sqrt(np.diag(K) - 2 * (K @ gamma_) + gamma_ @ K @ gamma_)
        gamma_ = w / w.sum()
    return gamma_


def w_scale(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    med = np.median(x, axis=0)
    sigma0 = _mad(x, flag=1, axis=0)
    w = ((1 - ((x - med) / 4.5) ** 2) ** 2) * (np.abs(x - med) < 4.5) / sigma0
    loc = np.sum(x * w, axis=0) / np.sum(w, axis=0)

    sigma0 = _mad(x, flag=1, axis=0)
    b = 3 * norm.ppf(3 / 4)
    nes = n * (2 * ((1 - b ** 2) * norm.cdf(b) - b * norm.pdf(b) + b ** 2) - 1)
    rc = np.minimum(((x - loc) / sigma0) ** 2, 3 ** 2)
    scale = sigma0 ** 2 / nes * np.sum(rc, axis=0)
    return np.sqrt(scale)


def kernel_ogk(h_indices, K):
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    n_h = len(h_indices)
    K_h = K[np.ix_(h_indices, h_indices)]
    Kt = K[:, h_indices]

    # Covariance matrix
    K_tilde = center(K_h)
    U, S_F, _ = np.linalg.svd(K_tilde)
    mask = S_F > 1000 * _EPS
    U = U[:, mask]
    S_F = S_F[mask]
    U_scaled = U / np.sqrt(S_F)[None, :]

    # Step 1: Compute E and B
    o = np.ones(n_h)
    g = o / n_h
    K_Phi_PhiTilde = Kt - np.outer(Kt @ g, o)
    B_F = K_Phi_PhiTilde @ U_scaled
    lambda_F = w_scale(B_F)

    # Step 2: Estimate the center
    K_Adapted = K_Phi_PhiTilde @ U_scaled @ np.diag(1.0 / lambda_F) @ U_scaled.T @ K_Phi_PhiTilde.T
    gamma_c = spatial_median(K_Adapted)

    # Step 3: Calculate Mahalanobis
    Kt_cCov = (Kt
               - np.outer(np.ones(n), gamma_c @ Kt)
               - np.outer(Kt @ g, o)
               + (gamma_c @ Kt @ g))
    mahal_F = np.sum((Kt_cCov @ U_scaled @ np.diag(lambda_F ** -2)) * (Kt_cCov @ U_scaled), axis=1)
    return np.argsort(mahal_F, kind="stable")


def spatial_median_estimator(K, alpha):
    K = np.asarray(K, dtype=float)
    assert K.shape[0] == K.shape[1]
    n = K.shape[0]

    g = spatial_median(K)
    dist = np.diag(K) - 2 * (K @ g) + g @ K @ g
    h_indices = np.argsort(dist, kind="stable")

    return kernel_ogk(h_indices[: int(np.ceil(n * alpha))], K)


def sscm(K):
    K = np.asarray(K, dtype=float)
    assert K.shape[0] == K.shape[1]
    n = K.shape[0]
    g = spatial_median(K)

    o = np.ones(n)
    Kg = K @ g
    Kc = K - np.outer(o, g @ K) - np.outer(Kg, o) + (g @ K @ g)
    d = np.diag(K) - 2 * Kg + g @ K @ g
    sqrtD = np.sqrt(1.0 / d)
    K_tilde = sqrtD[:, None] * Kc * sqrtD[None, :]

    U, S_F, _ = np.linalg.svd(K_tilde)
    mask = S_F > 1000 * _EPS
    U = U[:, mask]
    S_F = S_F[mask]
    U_scaled = U / np.sqrt(S_F)[None, :]

    # Kernel OGK
    K_Phi_PhiTilde = (K - np.outer(Kg, o)) * sqrtD[None, :]
    B_F = K_Phi_PhiTilde @ U_scaled
    lambda_F = w_scale(B_F)

    K_Adapted = K_Phi_PhiTilde @ U_scaled @ np.diag(1.0 / lambda_F) @ U_scaled.T @ K_Phi_PhiTilde.T
    gamma_c = spatial_median(K_Adapted)

    K_cCov = K - np.outer(o, gamma_c @ K) - np.outer(Kg, o) + (g @ K @ gamma_c)
    K_cCov = K_cCov * sqrtD[None, :]
    mahal_F = np.sum((K_cCov @ U_scaled @ np.diag(lambda_F ** -2)) * (K_cCov @ U_scaled), axis=1)
    return np.argsort(mahal_F, kind="stable")


def sdo(K, alpha, rng=None):
    K = np.asarray(K, dtype=float)
    assert K.shape[0] == K.shape[1]
    n = K.shape[0]
    rng = rng if rng is not None else np.random.default_rng()

    g = np.zeros(n)
    for _ in range(500):
        i, j = rng.choice(n, size=2, replace=False)
        lam = np.zeros(n)
        lam[i] = 1.0
        lam[j] = -1.0
        a = (K @ lam) / np.sqrt(lam @ K @ lam)
        sdo_val = np.abs(a - np.median(a)) / _mad(a, flag=0)
        mask = sdo_val > g
        g[mask] = sdo_val[mask]

    h_indices = np.argsort(g, kind="stable")
    return kernel_ogk(h_indices[: int(np.ceil(n * alpha))], K)


def spatial_rank(K, alpha):
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    diagK = np.diag(K)
    ook = np.zeros(n)
    for k in range(n):
        tmpA = K[k, k] - K[:, [k]] - K[[k], :] + K
        tmpB = np.sqrt(K[k, k] + diagK - 2 * K[k, :])
        tmpC = np.outer(tmpB, tmpB)
        mask = np.ones((n, n), dtype=bool)
        mask[k, :] = False
        mask[:, k] = False
        ook[k] = np.sum(tmpA[mask] / tmpC[mask])
    ook = (1.0 / n) * np.sqrt(ook)

    h_indices = np.argsort(ook, kind="stable")
    return kernel_ogk(h_indices[: int(np.ceil(n * alpha))], K)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def reweighted_mean(y, mask):
    y = np.asarray(y, dtype=float).ravel()
    mask = np.asarray(mask)
    xorig = y
    h = len(mask)

    initmean = np.mean(xorig[mask])
    initcov = np.var(xorig[mask], ddof=1)

    res = (xorig - initmean) ** 2 / initcov
    sortres = np.sort(res)
    factor = sortres[h - 1] / chi2.ppf(h / len(y), df=1)
    initcov = factor * initcov

    res = (xorig - initmean) ** 2 / initcov
    quantile = chi2.ppf(0.975, df=1)
    weights = (res <= quantile).astype(float)

    tmcd = np.sum(xorig * weights) / np.sum(weights)
    smcd = np.sqrt(np.sum((xorig - tmcd) ** 2 * weights) / (np.sum(weights) - 1))
    return tmcd, smcd


def unimcd(y, h):
    """Univariate MCD location/scale of ``y`` using an h-subset of size
    ``h``. See the module docstring: the raw h-subset is the contiguous
    window (in sorted order) of minimal variance; the correction and
    reweighting steps then follow ``reweighted_mean``."""
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    if h >= n:
        return reweighted_mean(y, np.arange(n))

    order = np.argsort(y, kind="stable")
    sorted_y = y[order]

    nwindows = n - h + 1
    cumsum = np.concatenate(([0.0], np.cumsum(sorted_y)))
    cumsum2 = np.concatenate(([0.0], np.cumsum(sorted_y ** 2)))
    window_sum = cumsum[h:] - cumsum[:nwindows]
    window_sumsq = cumsum2[h:] - cumsum2[:nwindows]
    window_var = (window_sumsq - window_sum ** 2 / h) / (h - 1)

    best_start = int(np.argmin(window_var))
    mask = order[best_start:best_start + h]
    return reweighted_mean(y, mask)


def center(omega, kt=None):
    """Centering of the kernel matrix."""
    omega = np.asarray(omega, dtype=float)
    meanvec = omega.mean(axis=1)
    MM = meanvec.mean()
    if kt is None:
        return omega - meanvec[:, None] - meanvec[None, :] + MM
    kt = np.asarray(kt, dtype=float)
    meanvecT = kt.mean(axis=1)
    return kt - meanvec[None, :] - meanvecT[:, None] + MM
