from itertools import combinations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.covariance import LedoitWolf
from statsmodels.robust.scale import qn_scale
from helper_functions import calc_cov, qn_scale_overwrite
from sklearn.utils.extmath import fast_logdet
import scipy
from scipy.stats import rankdata, chi2
from scipy.special import ndtri
from scipy.optimize import brentq
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from KMRCD import Kernel_MRCD

# Set by _ogk_eigenvectors just before forking workers, so children inherit it
# via copy-on-write instead of having the (potentially large) array pickled
# through the executor's IPC channel on every task.
_ogk_worker_data = None


def _ogk_init_worker(data):
    global _ogk_worker_data
    _ogk_worker_data = data


def _ogk_pairwise_covariance(row, column):
    data = _ogk_worker_data
    return (
        qn_scale(data[:, column] + data[:, row]) ** 2
        - qn_scale(data[:, column] - data[:, row]) ** 2
    ) / 4.0

def sample_covariance(N):
    N_t = N.reshape(-1, N.shape[-1])
    mean_N = np.mean(N_t, axis=0)
    cov = calc_cov(N)

    return mean_N, cov

def ledoit_wolf(N):
    cov = LedoitWolf().fit(N.reshape((-1, N.shape[-1])))

    return cov.location_, cov.covariance_

class Shrinkage():
    def __init__(self, shrinkage=0.1):
        self.shrinkage = shrinkage

    def __call__(self, N, shrinkage=None):
        N_t = N.reshape(-1, N.shape[-1])
        mean_N = np.mean(N_t, axis=0)
        cov = calc_cov(N)
        shrinkage = shrinkage if shrinkage is not None else self.shrinkage
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0])

        return mean_N, cov

    def set_shrinkage(self, shrinkage=0.1):
        self.shrinkage = shrinkage

    def load_config(self, config_dict):
        if 'shrinkage' in config_dict:
            self.set_shrinkage(config_dict['shrinkage'])

class Diagonal_Loading():
    def __init__(self, reg=0.1):
        self.reg = reg

    def __call__(self, N, reg=None):
        N_t = N.reshape(-1, N.shape[-1])
        mean_N = np.mean(N_t, axis=0)
        cov = calc_cov(N)
        reg = reg if reg is not None else self.reg
        cov = cov + reg * np.eye(cov.shape[0])

        return mean_N, cov

    def set_reg(self, reg=0.1):
        self.reg = reg

    def load_config(self, config_dict):
        if 'reg' in config_dict:
            self.set_reg(config_dict['reg'])

class MCD():
    def __init__(self, support_fraction=0.75):
        self.support_fraction=support_fraction

    def custom_code(self, X):
        mean, cov = self.mcd(X)
        return mean, cov
    
    def __call__(self, X):
        from sklearn.covariance import MinCovDet
        if X.ndim > 2:
           X = X.reshape((-1, X.shape[-1]))
        model = MinCovDet(support_fraction = self.support_fraction, random_state = np.random.RandomState(4)).fit(X)

        return model.location_, model.covariance_

    def set_support_fraction(self, support_fraction=None):
        self.support_fraction = support_fraction

    def load_config(self, config_dict):
        if 'support_fraction' in config_dict:
            self.set_support_fraction(config_dict['support_fraction'])

    def mcd(self, X):
        random_state = np.random.RandomState(4)
        H, W, B = X.shape
        X_t = np.reshape(X, (-1,B))
        n_samples, n_features = X_t.shape
        if self.support_fraction is not None:
            n_support = int(self.support_fraction * n_samples)
        else:
            n_support = min(int(np.ceil(0.5 * (n_samples + n_features + 1))), n_samples)
        
        # 1. Find candidate supports on subsets
        # a. split the set in subsets of size ~ 300
        n_subsets = n_samples // 300
        n_samples_subsets = n_samples // n_subsets
        samples_shuffle = random_state.permutation(n_samples)
        h_subset = int(np.ceil(n_samples_subsets * (n_support / float(n_samples))))
        
        # b. perform a total of 500 trials
        n_trials_tot = 500
        # c. select 10 best (mean, covariance) for each subset
        n_best_sub = 10
        n_trials = max(10, n_trials_tot // n_subsets)
        n_best_tot = n_subsets * n_best_sub
        all_best_means = np.zeros((n_best_tot, n_features))
        try:
            all_best_cov = np.zeros((n_best_tot, n_features, n_features))
        except MemoryError:
            # The above is too big. Let's try with something much small
            # (and less optimal)
            n_best_tot = 10
            all_best_cov = np.zeros((n_best_tot, n_features, n_features))
            n_best_sub = 2
        for i in range(n_subsets):
            low_bound = i * n_samples_subsets
            high_bound = low_bound + n_samples_subsets
            current_subset = X_t[samples_shuffle[low_bound:high_bound]]
            best_means_sub, best_cov_sub, _, _ = self.select_candidates(
                current_subset,
                h_subset,
                n_trials=n_trials,
                n_best_sub=n_best_sub,
                n_iter=2,
                cov_computation=calc_cov,
                random_state=random_state,
            )
            subset_slice = np.arange(i * n_best_sub, (i + 1) * n_best_sub)
            all_best_means[subset_slice] = best_means_sub
            all_best_cov[subset_slice] = best_cov_sub

        # 2. Pool the candidate supports into a merged set
        # (possibly the full dataset)
        n_samples_merged = min(1500, n_samples)
        h_merged = int(np.ceil(n_samples_merged * (n_support / float(n_samples))))

        n_best_merged = 10
        # find the best couples (location, covariance) on the merged set
        selection = random_state.permutation(n_samples)[:n_samples_merged]
        means_merged, cov_merged, supports_merged, dist_merged = self.select_candidates(
            X_t[selection],
            h_merged,
            n_trials=(all_best_means, all_best_cov),
            n_best_sub=n_best_merged,
            cov_computation=calc_cov,
            random_state=random_state,
        )
        # select the best couple on the full dataset
        means_full, cov_full, supports_full, d = self.select_candidates(
            X_t,
            n_support,
            n_trials=(means_merged, cov_merged),
            n_best_sub=1,
            cov_computation=calc_cov,
            random_state=random_state,
        )
        mean = means_full[0]
        cov = cov_full[0]
        self.support = supports_full[0]
        self.dist = d[0]
        return mean, cov

    def select_candidates(self, current_subset, h_subset, n_trials, n_best_sub=1, n_iter=10, cov_computation=calc_cov, random_state=None):
        
        if isinstance(n_trials, int):
            run_from_estimates = False
        elif isinstance(n_trials, tuple):
            run_from_estimates = True
            estimates_list = n_trials
            n_trials = estimates_list[0].shape[0]
        
        all_estimates = []
        if not run_from_estimates:
            for j in range(n_trials):
                all_estimates.append(
                    self._c_step(
                        X=current_subset,
                        n_support=h_subset,
                        remaining_iterations=n_iter,
                        cov_computation=cov_computation,
                        random_state=random_state,
                    )
                )
        else:
            # perform computations from every given initial estimates
            for j in range(n_trials):
                initial_estimates = (estimates_list[0][j], estimates_list[1][j])
                all_estimates.append(
                    self._c_step(
                        X=current_subset,
                        n_support=h_subset,
                        remaining_iterations=n_iter,
                        initial_estimates=initial_estimates,
                        cov_computation=cov_computation,
                        random_state=random_state,
                    )
                )
        all_locs_sub, all_covs_sub, all_dets_sub, all_supports_sub, all_dist_sub = zip(
        *all_estimates
        )
        # find the `n_best` best results among the `n_trials` ones
        index_best = np.argsort(all_dets_sub)[:n_best_sub]
        best_mean = np.asarray(all_locs_sub)[index_best]
        best_cov = np.asarray(all_covs_sub)[index_best]
        best_supports = np.asarray(all_supports_sub)[index_best]
        best_dist = np.asarray(all_dist_sub)[index_best]

        return best_mean, best_cov, best_supports, best_dist

    def _c_step(self, X, n_support, random_state=None, initial_estimates=None, remaining_iterations=10, cov_computation=calc_cov):
        n_samples, n_features = X.shape
        dist = np.inf

        # Initialisation
        if initial_estimates is None:
            # compute initial robust estimates from a random subset
            support_indices = random_state.permutation(n_samples)[:n_support]
        else:
            # get initial robust estimates from the function parameters
            mean = initial_estimates[0]
            cov = initial_estimates[1]
            # run a special iteration for that case (to get an initial support_indices)
            precision = scipy.linalg.pinvh(cov)
            X_centered = X - mean
            dist = (np.dot(X_centered, precision) * X_centered).sum(1)
            # compute new estimates
            support_indices = np.argpartition(dist, n_support - 1)[:n_support]

        X_support = X[support_indices]
        mean = X_support.mean(0)
        cov = cov_computation(X_support)

        # Iterative procedure for Minimum Covariance Determinant computation
        det = fast_logdet(cov)
        # If the data already has singular covariance, calculate the precision,
        # as the loop below will not be entered.
        if np.isinf(det):
            precision = scipy.linalg.pinvh(cov)
        
        previous_det = np.inf

        while det < previous_det and remaining_iterations > 0 and not np.isinf(det):
            previous_mean = mean
            previous_cov = cov
            previous_det = det
            previous_support_indices = support_indices

            precision = scipy.linalg.pinvh(cov)
            X_centered = X - mean
            dist = (np.dot(X_centered, precision) * X_centered).sum(axis=1)

            # compute new estimates
            support_indices = np.argpartition(dist, n_support - 1)[:n_support]

            X_support = X[support_indices]
            mean = X_support.mean(axis=0)
            cov = cov_computation(X_support)
            det = fast_logdet(cov)
            # update remaining iterations for early stopping
            remaining_iterations -= 1

        # Calc dist for the last iteration
        previous_dist = dist
        dist = (np.dot(X - mean, precision) * (X - mean)).sum(axis=1)

        # Check if best fit already found (det => 0, logdet => -inf)
        if np.isinf(det):
            results = mean, cov, det, support_indices, dist
        
        # Check convergence
        if np.allclose(det, previous_det):
            # c_step procedure converged
            print(
                "Optimal couple (mean, covariance) found before"
                " ending iterations (%d left)" % (remaining_iterations)
            )
            results = mean, cov, det, support_indices, dist
        elif det > previous_det:
            # determinant has increased (should not happen)
            print(
                "Determinant has increased; this should not happen: "
                "log(det) > log(previous_det) (%.15f > %.15f). "
                "You may want to try with a higher value of "
                "support_fraction (current value: %.3f)."
                % (det, previous_det, n_support / n_samples)
            )
            results = (
                previous_mean,
                previous_cov,
                previous_det,
                previous_support_indices,
                previous_dist,
            )
        # Check early stopping
        if remaining_iterations == 0:
            print("Maximum number of iterations reached")
            results = mean, cov, det, support_indices, dist

        mean, cov, det, support_indices, dist = results
        # Convert from list of indices to boolean mask.
        support = np.bincount(support_indices, minlength=n_samples).astype(bool)
        return mean, cov, det, support, dist

class MRCD():
    def __init__(self, rho=0.1, alpha=0.75, h=None, target='identity', maxcond=50.0, max_steps=10):
        
        if not 0.5 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.5 and 1.0")
        if h is not None and (not isinstance(h, int) or h < 1):
            raise ValueError("h must be a positive integer or None")
        if max_steps < 1:
            raise ValueError("maxcsteps must be at least 1")
        if rho is not None and not 0.0 < rho < 1.0:
            raise ValueError("rho must be strictly between 0 and 1")
        if target not in {"identity", "equicorrelation"}:
            raise ValueError("target must be 'identity' or 'equicorrelation'")
        if maxcond <= 1.0:
            raise ValueError("maxcond must be greater than 1")
        self.rho = rho
        self.alpha = alpha
        self.target = target
        self.h = h
        self.maxcond = maxcond
        self.max_steps = max_steps
        
    def __call__(self, N):
        mean, cov = self.rmcd(N)

        return mean, cov

    def standardize(self,X):
        loc = np.median(X, axis=0)
        if X.shape[0] > 40000: # Workaround to crashes in qn_scale if X is too large (I'm assuming a memory access issue) ~50x slower.
            scale = qn_scale_overwrite(X, axis=0)
        else:
            scale = qn_scale(X, axis=0)
        scale = np.maximum(scale, 0.001)
        U = (X - loc) / scale
        return U, loc, scale

    def create_T(self, X, target='identity'):
        features = X.shape[1]
        if target == "identity":
            return np.eye(features)
        if target != "equicorrelation":
            raise ValueError("target must be 'identity' or 'equicorrelation'")
        if features == 1:
            return np.eye(1)
        ranks = np.column_stack([rankdata(X[:, column], method="average") for column in range(features)])
        spearman = np.corrcoef(ranks, rowvar=False)
        transformed = np.sin(0.5 * np.pi * spearman)
        correlation = float(np.mean(transformed[np.triu_indices(features, k=1)]))
        lower_bound = min(0.0, -1.0 / (features - 1) + 0.01)
        correlation = max(correlation, lower_bound)
        return correlation * np.ones((features, features)) + (1.0 - correlation) * np.eye(features)
    
    def equicorrelation_eigensystem(self, target):
        """Return the analytic Helmert eigensystem used by R's ``eigenEQ``."""
        dimension = target.shape[0]
        if target.shape != (dimension, dimension) or dimension < 2:
            raise ValueError("target must be a square matrix with dimension at least 2")

        correlation = float(target[0, 1])
        helmert = np.zeros((dimension, dimension))
        helmert[:, 0] = 1.0 / np.sqrt(dimension)
        for column in range(1, dimension):
            helmert[:column, column] = 1.0 / np.sqrt((column + 1) * column)
            helmert[column, column] = -column / np.sqrt((column + 1) * column)

        eigenvalues = np.concatenate(
            ([1.0 + (dimension - 1) * correlation], np.full(dimension - 1, 1.0 - correlation))
        )
        return eigenvalues, helmert
    
    def _eigenvectors(self, data):
        return np.linalg.eigh(data)[1]
    
    def _ogk_eigenvectors(self, data, n_jobs=20):
        dimensions = data.shape[1]
        matrix = np.eye(dimensions)
        pairs = list(combinations(range(dimensions), 2))
        # qn_scale's Cython implementation holds the GIL, so threads add contention
        # instead of speedup here; a fork-based process pool is used instead.
        # (joblib's loky backend is skipped: its bootstrapping breaks under this
        # Python's default "forkserver" multiprocessing start method.)
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=ctx, initializer=_ogk_init_worker, initargs=(data,)
        ) as executor:
            covariances = list(executor.map(_ogk_pairwise_covariance, *zip(*pairs)))
        for (row, column), covariance in zip(pairs, covariances):
            matrix[column, row] = covariance
            matrix[row, column] = covariance
        return self._eigenvectors(matrix)
    
    def _initset(self,data, eigenvectors, h):
        projected = data @ eigenvectors
        scales = np.maximum(np.array([qn_scale(projected[:, column]) for column in range(projected.shape[1])]), 1e-12)
        square_root_covariance = (eigenvectors * scales) @ eigenvectors.T
        inverse_square_root_covariance = (eigenvectors / scales) @ eigenvectors.T
        location = np.median(data @ inverse_square_root_covariance, axis=0) @ square_root_covariance
        centered_projection = (data - location) @ eigenvectors
        distances = np.sum((centered_projection / scales) ** 2, axis=1)
        return np.argsort(distances, kind="stable")[:h]
    
    def initial_subsets(self, data, h):
        """Return the six deterministic h-subsets from ``r6pack``."""
        scaled, _, _ = self.standardize(data)
        samples, dimensions = scaled.shape
        spatial_norm = np.linalg.norm(scaled, axis=1)
        spatial_sign = scaled.copy()
        nonzero = spatial_norm > np.finfo(float).eps
        spatial_sign[nonzero] /= spatial_norm[nonzero, None]

        tanh_vectors = self._eigenvectors(np.corrcoef(np.tanh(scaled), rowvar=False))
        spearman_vectors = self._eigenvectors(np.corrcoef(np.apply_along_axis(rankdata, 0, scaled), rowvar=False))
        normal_scores = ndtri(np.apply_along_axis(rankdata, 0, scaled) - 1.0 / 3.0) / (samples + 1.0 / 3.0)
        score_vectors = self._eigenvectors(np.corrcoef(normal_scores, rowvar=False))
        spatial_vectors = self._eigenvectors(spatial_sign.T @ spatial_sign)
        half = int(np.ceil(samples / 2.0))
        bacon_indices = np.argsort(spatial_norm, kind="stable")[:half]
        bacon_vectors = self._eigenvectors(np.cov(scaled[bacon_indices], rowvar=False, bias=False))
        ogk_vectors = self._ogk_eigenvectors(scaled)

        starts = [tanh_vectors, spearman_vectors, score_vectors, spatial_vectors, bacon_vectors, ogk_vectors]
        return np.column_stack([self._initset(scaled, eigenvectors, h) for eigenvectors in starts])

    def _rho_for_subset(self,centered_subset, consistency_factor):
        eigenvalues = np.linalg.eigvalsh(consistency_factor * (centered_subset.T @ centered_subset) / (centered_subset.shape[0] - 1))
        smallest = float(np.min(eigenvalues))
        largest = float(np.max(eigenvalues))

        def condition_gap(rho: float) -> float:
            return (rho + (1.0 - rho) * largest) / (rho + (1.0 - rho) * smallest) - self.maxcond

        try:
            return float(brentq(condition_gap, 0.00001, 0.99))
        except ValueError:
            grid = np.concatenate(([0.000001], np.arange(0.001, 0.991, 0.001), [0.999999]))
            return float(grid[np.argmin(np.abs([condition_gap(value) for value in grid]))])

    def _select_rho(self, data, subsets, consistency_factor):
        per_subset = np.array(
            [self._rho_for_subset(data[subsets[:, column]] - np.mean(data[subsets[:, column]], axis=0), consistency_factor) for column in range(subsets.shape[1])]
        )
        cutoff = max(0.1, float(np.median(per_subset)))
        selected = np.flatnonzero(per_subset <= cutoff)
        if selected.size == 0:
            raise ValueError("none of the initial subsets is well-conditioned")
        return float(np.max(per_subset[selected])), selected

    def inverse_regularized_covariance(self, centered_data, target, consistency_factor):
        """Return R's regularized covariance and its inverse.

        ``centered_data`` is arranged as samples by features. The SMW expression
        mirrors ``.InvSMW`` when features exceed the subset size.
        """
        samples, features = centered_data.shape
        sample_covariance = centered_data.T @ centered_data / samples
        covariance = self.rho * target + (1.0 - self.rho) * consistency_factor * sample_covariance

        if features <= samples:
            precision = cho_solve(cho_factor(covariance, lower=False, check_finite=False), np.eye(features))
            return covariance, precision

        diagonal_scale = np.sqrt(np.diag(target))
        inverse_diagonal_scale = np.diag(1.0 / diagonal_scale)
        correlation_target = inverse_diagonal_scale @ target @ inverse_diagonal_scale
        equicorrelation = float(correlation_target[1, 0]) if features > 1 else 0.0
        identity = np.eye(features)
        inverse_correlation = identity if features == 1 else (
            identity - equicorrelation / (1.0 + (features - 1) * equicorrelation) * np.ones((features, features))
        ) / (1.0 - equicorrelation)
        inverse_base = inverse_diagonal_scale @ inverse_correlation @ inverse_diagonal_scale / self.rho
        scale = (1.0 - self.rho) * consistency_factor
        gram = np.eye(samples) + scale * centered_data @ inverse_base @ centered_data.T / samples
        correction = cho_solve(cho_factor(gram, lower=False, check_finite=False), np.eye(samples))
        precision = inverse_base - (inverse_base @ centered_data.T / np.sqrt(samples)) @ (
            scale * correction
        ) @ (centered_data @ inverse_base / np.sqrt(samples))
        return covariance, precision

    def rmcd(self, X):
        H, W, B = X.shape
        X_t = np.reshape(X, (-1,B))
        n_samples, n_features = X_t.shape

        if self.h is None:
            h = self.h
            if self.alpha is not None:
                h = int(self.alpha * n_samples)
            else:
                h = min(int(np.ceil(0.5 * (n_samples + n_features + 1))), n_samples)
        alpha = h / n_samples

        working, median, scales = self.standardize(X_t)
        T = self.create_T(X_t, target=self.target)

        target_eigenvalues, target_eigenvectors = None, None
        if self.target == 'equicorrelation':
            target_eigenvalues, target_eigenvectors = self.equicorrelation_eigensystem(T)
            working = working @ target_eigenvectors @ np.diag(1.0 / np.sqrt(target_eigenvalues))

        subsets = self.initial_subsets(working, h)
        consistency_factor =  alpha / chi2.cdf(chi2.ppf(alpha, df=n_features), df=n_features + 2)

        if self.rho is None:
            self.rho, selected_starts = self._select_rho(working, subsets, consistency_factor)
        else:
            selected_starts = np.arange(subsets.shape[1])

        best = None
        cstep_counts = np.zeros(subsets.shape[1], dtype=int)
        for start in selected_starts:
            candidate = self._c_step(working, subsets[:, start], consistency_factor)
            cstep_counts[start] = candidate[4]
            # objective = fast_logdet(candidate[2])
            objective = np.linalg.slogdet(candidate[2])[1]
            if best is None or objective < np.linalg.slogdet(best[2])[1]:
                best = (*candidate, int(start))
        assert best is not None

        best_indices, _, _, _, _, best_start = best
        subset = working[best_indices]
        working_center = np.mean(subset, axis=0)
        centered_subset = subset - working_center
        sample_covariance = centered_subset.T @ centered_subset / (h - 1)
        working_covariance = self.rho * np.eye(n_features) + (1.0 - self.rho) * consistency_factor * sample_covariance
        _, working_precision = self.inverse_regularized_covariance(
        centered_subset * np.sqrt(h / (h - 1)), np.eye(n_features), consistency_factor
        )
        working_target = np.eye(n_features)
        if target_eigenvalues is not None and target_eigenvectors is not None:
            square_root = np.diag(np.sqrt(target_eigenvalues))
            inverse_square_root = np.diag(1.0 / np.sqrt(target_eigenvalues))
            working_center = target_eigenvectors @ square_root @ working_center
            working_covariance = target_eigenvectors @ square_root @ working_covariance @ square_root @ target_eigenvectors.T
            working_precision = target_eigenvectors @ inverse_square_root @ working_precision @ inverse_square_root @ target_eigenvectors.T
            working_target = target_eigenvectors @ square_root @ working_target @ square_root @ target_eigenvectors.T

        self.covariance = (scales[:, None] * working_covariance) * scales[None, :]
        self.precision = (working_precision / scales[:, None]) / scales[None, :]
        self.center = scales * working_center + median
        self.fitted_target = (scales[:, None] * working_target) * scales[None, :]
        self.residuals = X_t - self.center
        self.distances = np.einsum("ij,jk,ik->i", self.residuals, self.precision, self.residuals)

        return self.center, self.covariance

    def _c_step(self, data, indices, consistency_factor):
        n_samples, n_features = data.shape
        indices = np.sort(indices)
        h = indices.size
        for iteration in range(1, self.max_steps + 1):
            subset = data[indices]
            center = np.mean(subset, axis=0)
            covariance, precision = self.inverse_regularized_covariance(subset - center, np.eye(data.shape[1]), consistency_factor)
            residuals = data - center
            distances = np.einsum("ij,jk,ik->i", residuals, precision, residuals)
            updated = np.sort(np.argsort(distances)[:h])
            if np.array_equal(updated, indices) or iteration == self.max_steps:
                return indices, center, covariance, precision, iteration
            indices = updated
        
    def set_rho(self, rho=0.5):
        self.rho = rho

    def load_config(self, config_dict):
        if 'rho' in config_dict:
            self.set_rho(config_dict['rho'])
        if 'alpha' in config_dict:
            self.alpha = config_dict['alpha']
        if 'h' in config_dict:
            self.h = config_dict['h']
        if 'target' in config_dict:
            self.target = config_dict['target']
        if 'maxcond' in config_dict:
            self.maxcond = config_dict['maxcond']
        if 'max_steps' in config_dict:
            self.max_steps = config_dict['max_steps']

class KMRCD():
    def __init__(self, alpha=0.75, maxcond=50.0, max_steps=10, kernel_method=None):
        
        if not 0.5 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.5 and 1.0")
        if max_steps < 1:
            raise ValueError("maxcsteps must be at least 1")
        if maxcond <= 1.0:
            raise ValueError("maxcond must be greater than 1")
        self.alpha = alpha
        self.maxcond = maxcond
        self.max_steps = max_steps
        self.kernel_method = kernel_method
        
    def __call__(self, N):
        KRMCD_model = Kernel_MRCD(alpha = self.alpha, kernel=self.kernel_method, c_step_iterations_allowed=self.max_steps, maxcond=self.maxcond)
        N_t = N.reshape(-1, N.shape[-1])
        self.solution = KRMCD_model.run_algorithm(N_t)

        return self.solution.hsubset_indices
        

    def load_config(self, config_dict):
        if 'alpha' in config_dict:
            self.alpha = config_dict['alpha']
        if 'maxcond' in config_dict:
            self.maxcond = config_dict['maxcond']
        if 'max_steps' in config_dict:
            self.max_steps = config_dict['max_steps']
        if 'kernel' in config_dict:
            self.kernel_method = config_dict['kernel']


def select_background_model(background_model="sample"):
    background_dict = {"Sample": sample_covariance,
                       "LedoitWolf": ledoit_wolf,
                       "Shrinkage": Shrinkage(),
                      "Diagonal": Diagonal_Loading(),
                      "MCD": MCD(),
                      "MRCD": MRCD(),
                      "KMRCD": KMRCD()}
    
    if background_model in background_dict:
        return background_dict[background_model]
    else:
        print(f'{background_model} not known.')

