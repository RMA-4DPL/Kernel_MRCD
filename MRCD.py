"""
Minimum Regularized Covariance Determinant (MRCD) estimator.

Python port of the reference R implementation in
R_MRCD/MRCD_R_code_20180322/{MRCD.R, helperfunctionsMRCD.R, r6pack.R}
(Boudt, Rousseeuw, Vanduffel & Verdonck (2020), "The Minimum Regularized
Covariance Determinant estimator", Statistics and Computing).

Conventions
-----------
The R code stores data as a p x n matrix (features x observations). This
module instead follows the usual Python/numpy convention: data is an
(n_samples, n_features) array, one observation per row. All formulas below
are the row/column-transposed equivalent of the R code.

Notable, deliberate deviations from the reference R code
----------------------------------------------------------------
* If ``rho`` is supplied by the caller, the R code never assigns ``initV``/
  ``setsV`` (they are only computed inside ``if(is.null(rho))``), so a fixed
  ``rho`` crashes the original implementation. Here, all six initial subsets
  are simply used as candidate starting points.
* At the very end, the R code recomputes the Mahalanobis distances using a
  matrix ``mX`` that (when ``rescale=TRUE`` and ``rescaleSVD=TRUE``, the
  default) was never rotated back out of the whitened target-SVD space
  before the final destandardization step is applied to it -- while ``mu``
  and ``icov`` *are* correctly rotated back to the original data space. That
  looks like a bug in the reference code. Here, the final distances are
  computed directly from the original input ``X`` together with the
  (correctly backtransformed) ``mu``/``icov``, which is what the R code
  appears to intend.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm, rankdata
from scipy.linalg import cho_factor, cho_solve


# ----------------------------------------------------------------------
# Basic helper functions (helperfunctionsMRCD.R)
# ----------------------------------------------------------------------

def scfactor(alpha, p):
    """Consistency scaling factor for the subset covariance (scfactor in R)."""
    return alpha / chi2.cdf(chi2.ppf(alpha, df=p), df=p + 2)


def condnumber(X):
    """Ratio of largest to smallest eigenvalue of a symmetric matrix."""
    eigvals = np.linalg.eigvalsh(X)
    return eigvals[-1] / eigvals[0]


def _inv_sympd(X):
    """Inverse of a symmetric positive definite matrix via its Cholesky factor."""
    c, low = cho_factor(X, lower=True)
    return cho_solve((c, low), np.eye(X.shape[0]))


def _qn_col(X):
    from statsmodels.robust.scale import qn_scale
    if X.ndim == 1:
        return qn_scale(X)
    return qn_scale(X, axis=0)


def _scale_tau2(x, c1=4.5, c2=3.0):
    """Robust scale estimator scaleTau2 (robustbase), used to build the
    diagonal target when data is not first standardized (rescale=False)."""
    med = np.median(x)
    sigma0 = np.median(np.abs(x - med))
    if sigma0 == 0:
        return 0.0
    xs = (x - med) / sigma0
    w = 1.0 - (xs / c1) ** 2
    w = np.clip(w, 0.0, None) ** 2
    mu = np.sum(x * w) / np.sum(w)
    z = (x - mu) / sigma0
    rho = np.minimum(z ** 2, c2 ** 2)
    # asymptotic normalizing constant E[min(Z^2, c2^2)] for Z ~ N(0,1)
    nEs2 = (2 * norm.cdf(c2) - 2 * c2 * norm.pdf(c2) - 1
            + c2 ** 2 * 2 * (1 - norm.cdf(c2)))
    tau2 = sigma0 ** 2 * np.sum(rho) / (len(x) * nEs2)
    return np.sqrt(tau2)


def _qn_cormat(X):
    """Pairwise Qn-based correlation matrix (Qncormat in R).

    Batches all pairwise sum/difference columns into two large qn_scale
    calls instead of O(p^2) individual ones -- qn_scale's per-call Python
    overhead otherwise dominates once p is more than a few dozen.
    """
    n, p = X.shape
    R = np.eye(p)
    i_idx, j_idx = np.triu_indices(p, k=1)
    if i_idx.size == 0:
        return R
    s_sum = _qn_col(X[:, i_idx] + X[:, j_idx])
    s_diff = _qn_col(X[:, i_idx] - X[:, j_idx])
    vals = (s_sum - s_diff) / 4.0
    R[i_idx, j_idx] = vals
    R[j_idx, i_idx] = vals
    return R


def _kendall_cormat(X):
    """Pairwise Kendall's tau correlation matrix (cor.fk in R's pcaPP)."""
    n, p = X.shape
    R = np.eye(p)
    for i in range(p - 1):
        for j in range(i + 1, p):
            tau = _kendall_tau_fast(X[:, i], X[:, j])
            R[i, j] = R[j, i] = tau
    return R


def _kendall_tau_fast(x, y):
    from scipy.stats import kendalltau
    tau, _ = kendalltau(x, y)
    return tau


def _spearman_cormat(X):
    """Spearman correlation matrix = Pearson correlation of the ranks.
    (Avoids scipy.stats.spearmanr's inconsistent return shape for p==2.)"""
    ranks = np.apply_along_axis(rankdata, 0, X)
    return np.atleast_2d(np.corrcoef(ranks, rowvar=False))


def _equicorrelation_matrix(X, target, mindet=0, maxcond=1000):
    """Correlation-structure part shared by TargetCov and TargetCorr."""
    n, p = X.shape
    if target == 0:
        return np.eye(p)
    elif target == 1:
        cortmp = _kendall_cormat(X)
        cortmp = np.sin(0.5 * np.pi * cortmp)
        iu = np.triu_indices(p, k=1)
        constcor = cortmp[iu].mean()
        lower_bound = min(0.0, -1.0 / (p - 1) + 0.01)
        if constcor <= lower_bound:
            constcor = lower_bound
        return constcor * np.ones((p, p)) + (1 - constcor) * np.eye(p)
    elif target == 2:
        Rq = _qn_cormat(X)
        iu = np.triu_indices(p, k=1)
        constcor = np.median(Rq[iu])
        lower_bound = min(0.0, -1.0 / (p - 1) + 0.01)
        if constcor <= lower_bound:
            constcor = lower_bound
        return constcor * np.ones((p, p)) + (1 - constcor) * np.eye(p)
    elif target == 3:
        R = _spearman_cormat(X)
        eigvals, eigvecs = np.linalg.eigh(R)
        Lambda = np.maximum(eigvals, eigvals[-1] / maxcond)
        return eigvecs @ np.diag(Lambda) @ eigvecs.T
    else:
        raise ValueError(f"Unknown target structure: {target}")


def target_cov(X, target=0):
    """TargetCov: robust covariance-scale target matrix for raw (non
    rescaled) data."""
    n, p = X.shape
    vD = np.array([_scale_tau2(X[:, j]) for j in range(p)])
    nz = vD[vD != 0]
    if nz.size:
        vD[vD == 0] = nz.min()
    mD = np.diag(vD)
    R = _equicorrelation_matrix(X, target)
    return mD @ R @ mD


def target_corr(X, target=0, mindet=0):
    """TargetCorr: target correlation matrix for standardized data."""
    return _equicorrelation_matrix(X, target, mindet=mindet)


def rcov(XX, mu, rho, mT, scfac, invert=False):
    """RCOV: convex-combination regularized covariance of a subset."""
    h, p = XX.shape
    E = XX - mu
    S = E.T @ E / h
    rcov_mat = rho * mT + (1 - rho) * scfac * S
    out = {"rho": rho, "mT": mT, "cov": S, "rcov": rcov_mat}
    if invert:
        if p > h:
            nu = (1 - rho) * scfac
            U = E.T / np.sqrt(h)  # p x h
            out["inv_rcov"] = inv_smw(rho, mT, nu, U)
        else:
            out["inv_rcov"] = _inv_sympd(rcov_mat)
    return out


def inv_smw(rho, mT, nu, U):
    """InvSMW: Sherman-Morrison-Woodbury inverse for p > h, assuming mT has
    an equicorrelation structure (identity or constant off-diagonal)."""
    p = mT.shape[0]
    vD = np.sqrt(np.diag(mT))
    imD = np.diag(1.0 / vD)
    R = imD @ mT @ imD
    constcor = R[1, 0]
    I = np.eye(p)
    J = np.ones((p, p))
    imR = 1.0 / (1 - constcor) * (I - constcor / (1 + (p - 1) * constcor) * J)
    imB = (1.0 / rho) * imD @ imR @ imD
    h = U.shape[1]
    Temp = _inv_sympd(np.eye(h) + nu * (U.T @ (imB @ U)))
    return imB - (imB @ U) @ (nu * Temp) @ (U.T @ imB)


def cstep_mrcd(X, rho, mT, alpha=0.75, h=None, index=None, maxit=200,
               rng=None):
    """cstep_mrcd: generalized C-steps to a locally optimal h-subset."""
    n, p = X.shape
    if h is None:
        h = int(np.floor(alpha * n))
    scfac = scfactor(alpha, p)

    if index is None:
        rng = rng or np.random.default_rng()
        index = rng.choice(n, size=h, replace=False)

    XX = X[index, :]
    mu = XX.mean(axis=0)
    ret = rcov(XX, mu, rho, mT, scfac, invert=True)
    inv_S = ret["inv_rcov"]

    diffs = X - mu
    vdst = np.einsum("ij,jk,ik->i", diffs, inv_S, diffs)
    index = np.sort(np.argsort(vdst, kind="stable")[:h])

    iterno = 1
    while iterno < maxit:
        XX = X[index, :]
        mu = XX.mean(axis=0)
        ret = rcov(XX, mu, rho, mT, scfac, invert=True)
        inv_S = ret["inv_rcov"]

        diffs = X - mu
        vdst = np.einsum("ij,jk,ik->i", diffs, inv_S, diffs)
        nndex = np.sort(np.argsort(vdst, kind="stable")[:h])

        if np.array_equal(nndex, index):
            break
        index = nndex
        iterno += 1

    return {
        "index": index, "numit": iterno, "mu": mu, "cov": ret["rcov"],
        "icov": ret["inv_rcov"], "rho": ret["rho"], "mT": ret["mT"],
        "dist": vdst, "scfac": scfac,
    }


# ----------------------------------------------------------------------
# Deterministic starting subsets (r6pack.R): DetMCD-style six initial sets
# ----------------------------------------------------------------------

def _initset(data, scalefn, P):
    """Order all observations by Mahalanobis distance in the basis P,
    using a robust location/scale computed within that basis."""
    proj = data @ P
    lam = scalefn(proj)
    sqrtcov = P @ np.diag(lam) @ P.T
    sqrtinvcov = P @ np.diag(1.0 / lam) @ P.T
    estloc = np.median(data @ sqrtinvcov, axis=0) @ sqrtcov
    centeredx = (data - estloc) @ P
    dist = np.sum((centeredx / lam) ** 2, axis=1)
    return np.argsort(dist, kind="stable")


def _ogkscatter(Y, scalefn):
    """OGK pairwise scatter matrix. Batches all pairwise sum/difference
    columns into two large scalefn calls instead of O(p^2) individual ones."""
    n, p = Y.shape
    U = np.eye(p)
    i_idx, j_idx = np.triu_indices(p, k=1)
    if i_idx.size:
        s_sum = scalefn(Y[:, i_idx] + Y[:, j_idx])
        s_diff = scalefn(Y[:, i_idx] - Y[:, j_idx])
        vals = (s_sum ** 2 - s_diff ** 2) / 4.0
        U[i_idx, j_idx] = vals
        U[j_idx, i_idx] = vals
    _, P = np.linalg.eigh(U)
    return P


def _class_pc(X):
    """classPC: classical (non-robust) PCA via SVD."""
    n = X.shape[0]
    center = X.mean(axis=0)
    Xc = X - center
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigenvalues = S ** 2 / max(n - 1, 1)
    loadings = Vt.T
    return center, loadings, eigenvalues


def r6pack(X, h, scalefn=_qn_col):
    """r6pack: six deterministic initial h-subsets (DetMCD, Hubert et al.
    2012), each expanded to a full ordering of all n observations by
    increasing Mahalanobis distance (full.h=TRUE in the R code)."""
    n, p = X.shape
    Xs = (X - np.median(X, axis=0)) / scalefn(X)
    nsets = 6
    hsets = np.zeros((h, nsets), dtype=int)

    # 1. hyperbolic tangent correlation
    y1 = np.tanh(Xs)
    R1 = np.corrcoef(y1, rowvar=False)
    _, P = np.linalg.eigh(R1)
    hsets[:, 0] = _initset(Xs, scalefn, P)[:h]

    # 2. Spearman correlation
    R2 = _spearman_cormat(Xs)
    _, P = np.linalg.eigh(R2)
    hsets[:, 1] = _initset(Xs, scalefn, P)[:h]

    # 3. normal-scores correlation
    ranks = np.apply_along_axis(rankdata, 0, Xs)
    y3 = norm.ppf((ranks - 1.0 / 3) / (n + 1.0 / 3))
    R3 = np.corrcoef(y3, rowvar=False)
    _, P = np.linalg.eigh(R3)
    hsets[:, 2] = _initset(Xs, scalefn, P)[:h]

    # 4. spatial sign covariance matrix
    znorm = np.sqrt(np.sum(Xs ** 2, axis=1))
    ii = znorm > np.finfo(float).eps
    Xnrmd = Xs.copy()
    Xnrmd[ii, :] = Xs[ii, :] / znorm[ii, None]
    SCM = Xnrmd.T @ Xnrmd
    _, P = np.linalg.eigh(SCM)
    hsets[:, 3] = _initset(Xs, scalefn, P)[:h]

    # 5. matches the reference R code: the covariance-of-closest-half-based
    # set is computed but then unconditionally overwritten with set 4.
    hsets[:, 4] = hsets[:, 3]

    # 6. OGK scatter
    P = _ogkscatter(Xs, scalefn)
    hsets[:, 5] = _initset(Xs, scalefn, P)[:h]

    hsetsN = np.zeros((n, nsets), dtype=int)
    for k in range(nsets):
        Xk = Xs[hsets[:, k], :]
        center, loadings, eigenvalues = _class_pc(Xk)
        score = (Xs - center) @ loadings
        lam = np.sqrt(np.abs(eigenvalues))
        lam[lam == 0] = 1.0
        dist = np.sum((score / lam) ** 2, axis=1)
        hsetsN[:, k] = np.argsort(dist, kind="stable")

    return hsetsN


# ----------------------------------------------------------------------
# Main MRCD estimator (MRCD.R)
# ----------------------------------------------------------------------

def mrcd(X, target=0, h=None, alpha=0.75, rho=None, rescale=True,
          rescale_svd=True, bc=False, maxcond=1000, minscale=0.001,
          mindet=0, objective="geom", maxit=200, random_state=None):
    """Compute the Minimum Regularized Covariance Determinant estimator.

    Parameters
    ----------
    X : (n_samples, n_features) array
    target : {0, 1, 2, 3}
        Structure of the regularization target matrix.
        0: diagonal (scale-only), 1: Kendall-tau based equicorrelation,
        2: Qn-based equicorrelation, 3: eigenvalue-clipped Spearman matrix.
    h : int, optional
        Subset size. Either h or alpha must effectively determine it.
    alpha : float
        Proportion of the data used in the h-subset (0.5 to 1).
    rho : float, optional
        Regularization parameter; estimated automatically if None.
    rescale : bool
        Robustly standardize the columns of X first (median / Qn).
    rescale_svd : bool
        Whiten by the eigendecomposition of the target matrix.
    bc : bool
        Rescale the subset covariance to have the target's diagonal.
    maxcond : float
        Maximum condition number allowed when choosing rho.
    objective : {'det', 'geom'}
        Criterion used to pick the best of the six initial subsets.
    maxit : int
        Maximum number of generalized C-steps per initial subset.
    random_state : int or numpy.random.Generator, optional
        Used only as a fallback if an initial subset ever needs to be
        drawn at random (not expected in normal operation).

    Returns
    -------
    dict with keys mu, cov, icov, rho, index, dist, mT, h, alpha
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    if objective == "det":
        def obj(cov):
            return np.linalg.slogdet(cov)[1]
    elif objective == "geom":
        def obj(cov):
            return np.linalg.slogdet(cov)[1] / p
    else:
        raise ValueError("objective must be 'det' or 'geom'")

    # 1. Robustly standardize the columns
    if rescale:
        loc = np.median(X, axis=0)
        scale = _qn_col(X)
        scale = np.maximum(scale, minscale)
        U = (X - loc) / scale
        mT = target_corr(U, target=target, mindet=mindet)
    else:
        mT = target_cov(X, target=target)
        U = X.copy()
        loc = np.zeros(p)
        scale = np.ones(p)

    # 2. Whiten by the SVD of the target matrix
    if rescale_svd:
        Teigval, Q = np.linalg.eigh(mT)
        sqL = np.sqrt(Teigval)
        isqL = 1.0 / sqL
        Xw = U @ Q @ np.diag(isqL)
        Twork = np.eye(p)
    else:
        Xw = U
        Twork = mT

    # 3. Six deterministic initial h-subsets
    if h is None:
        h = int(np.floor(alpha * n))
    else:
        alpha = h / n

    mind = r6pack(Xw, h=h)
    mind = mind[:h, :]
    scfac = scfactor(alpha=alpha, p=p)
    nsets = mind.shape[1]

    # 3.4 - 3.5: pick rho as the largest well-conditioned rho_k
    if rho is None:
        rho6pack = np.zeros(nsets)
        is_identity = np.allclose(Twork, np.eye(p))
        for k in range(nsets):
            Xsub = Xw[mind[:, k], :]
            mu_sub = Xsub.mean(axis=0)
            E = Xsub - mu_sub
            S = E.T @ E / h

            if is_identity:
                eigvals = np.linalg.eigvalsh(scfac * S)
                e1, ep = eigvals[0], eigvals[-1]

                def fncond(r):
                    return (r + (1 - r) * ep) / (r + (1 - r) * e1) - maxcond
            else:
                def fncond(r, Twork=Twork, S=S):
                    rc = r * Twork + (1 - r) * scfac * S
                    return condnumber(rc) - maxcond

            try:
                root = brentq(fncond, 1e-5, 0.99)
            except ValueError:
                grid = np.concatenate(([1e-6], np.arange(0.001, 0.991, 0.001), [0.999999]))
                vals = np.abs([fncond(g) for g in grid])
                root = grid[np.argmin(vals)]
            rho6pack[k] = root

        cutoffrho = max(0.1, np.median(rho6pack))
        valid = np.where(rho6pack <= cutoffrho)[0]
        if valid.size == 0:
            raise RuntimeError("None of the initial subsets is well-conditioned")
        rho = rho6pack[valid].max()
        initV = valid.min()
        setsV = valid[valid != initV]
    else:
        initV = 0
        setsV = np.arange(1, nsets)

    # 3.6 - 3.7: generalized C-steps from each initial subset, keep the best
    ret = cstep_mrcd(Xw, rho=rho, mT=Twork, h=h, alpha=alpha,
                      index=mind[:, initV], maxit=maxit, rng=rng)
    objret = obj(ret["cov"])
    hindex = ret["index"]
    for k in setsV:
        tmp = cstep_mrcd(Xw, rho=rho, mT=Twork, h=h, alpha=alpha,
                          index=mind[:, k], maxit=maxit, rng=rng)
        objtmp = obj(tmp["cov"])
        if objtmp <= objret:
            ret, objret, hindex = tmp, objtmp, tmp["index"]

    c_alpha = ret["scfac"]
    XX = Xw[hindex, :]
    E = XX - ret["mu"]
    weightedScov = E.T @ E / h

    # 4. Optional rescale so the subset covariance matches the target's diagonal
    if bc:
        vd = np.diag(Twork) / np.diag(weightedScov)
        sqrtD = np.diag(np.sqrt(vd))
        weightedScov = sqrtD @ weightedScov @ sqrtD
        E = E @ sqrtD
    else:
        sqrtD = c_alpha * np.eye(p)

    MRCDmu = Xw[hindex, :].mean(axis=0)
    MRCDcov = rho * Twork + (1 - rho) * weightedScov

    if p > n and target <= 1:
        # Note: MRCDcov above is the convex combination of the target and the
        # *unscaled* subset covariance weightedScov (no scfac consistency
        # factor), matching the R reference; nu must match that, i.e. use a
        # factor of 1 here rather than the scfac used during the C-steps.
        nu = (1 - rho) * 1.0
        Umat = E.T / np.sqrt(h)
        iMRCDcov = inv_smw(rho, Twork, nu, Umat)
    else:
        iMRCDcov = _inv_sympd(MRCDcov)

    # Backtransform out of the target-SVD whitening
    if rescale_svd:
        MRCDcov = Q @ np.diag(sqL) @ MRCDcov @ np.diag(sqL) @ Q.T
        iMRCDcov = Q @ np.diag(isqL) @ iMRCDcov @ np.diag(isqL) @ Q.T
        Twork = Q @ np.diag(sqL) @ Twork @ np.diag(sqL) @ Q.T

    # Backtransform out of the robust standardization
    if rescale:
        Dx = np.diag(scale)
        iDx = np.diag(1.0 / scale)
        MRCDmu = Dx @ MRCDmu + loc
        MRCDcov = Dx @ MRCDcov @ Dx
        Twork = Dx @ Twork @ Dx
        iMRCDcov = iDx @ iMRCDcov @ iDx

    # Mahalanobis distances of all n observations in the original data space
    diffs = X - MRCDmu
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diffs, iMRCDcov, diffs))

    return {
        "mu": MRCDmu,
        "cov": MRCDcov,
        "icov": iMRCDcov,
        "rho": rho,
        "index": hindex,
        "dist": dist,
        "mT": Twork,
        "h": h,
        "alpha": alpha,
    }
