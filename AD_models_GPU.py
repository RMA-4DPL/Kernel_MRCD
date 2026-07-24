import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import cho_factor, cho_solve
import torch
import torch.nn.functional as F
from kernels import RbfKernel, AutoRbfKernel

def _safe_cho_factor(cov, lower=True):
    """cho_factor with a diagonal-loading fallback for (near-)singular matrices.

    Sample covariances built from few background points relative to the
    dimensionality are rank-deficient (rank <= n_bg - 1), which makes LAPACK's
    potrf fail outright. Retry with a small ridge added to the diagonal,
    growing geometrically, until the factorization succeeds.
    """
    try:
        return cho_factor(cov, lower=lower, check_finite=False)
    except np.linalg.LinAlgError:
        pass
    n = cov.shape[0]
    reg = np.finfo(cov.dtype).eps * max(np.trace(cov) / n, 1.0)
    for _ in range(20):
        try:
            return cho_factor(cov + reg * np.eye(n, dtype=cov.dtype), lower=lower, check_finite=False)
        except np.linalg.LinAlgError:
            reg *= 10
    raise np.linalg.LinAlgError("cho_factor: matrix remains non positive-definite after diagonal loading")

def _safe_cholesky_torch(cov):
    """torch.linalg.cholesky with the same diagonal-loading fallback as
    _safe_cho_factor, for (near-)singular background covariances."""
    try:
        return torch.linalg.cholesky(cov)
    except torch._C._LinAlgError:
        pass
    n = cov.shape[0]
    eye = torch.eye(n, dtype=cov.dtype, device=cov.device)
    reg = torch.finfo(cov.dtype).eps * max((torch.trace(cov) / n).item(), 1.0)
    for _ in range(20):
        try:
            return torch.linalg.cholesky(cov + reg * eye)
        except torch._C._LinAlgError:
            reg *= 10
    raise torch._C._LinAlgError("cholesky: matrix remains non positive-definite after diagonal loading")

def _rbf_kernel_torch(x1, x2, sigma):
    """RBF Gram matrix computed on-device, matching kernels.RbfKernel.compute."""
    sqdist = torch.cdist(x1, x2, p=2) ** 2
    return torch.exp(-sqdist / (2 * sigma**2))

class RX():
    def __init__(self, cov=None, mean_N=None, device=None, kernel=False, gamma=None, reg=0.1):
        self.cov = cov
        self.mean_N = mean_N
        self.device = device
        self.kernel = kernel
        self.gamma = gamma
        self.reg = reg
        self.gpu = False

    def __call__(self, X, N):
        if self.kernel and self.gpu:
            return self.run_kernel_gpu(X, N)
        elif self.kernel:
            return self.run_kernel(X, N)
        elif self.gpu:
            return self.run_gpu(X, N)
        else:
            return self.run_cpu(X, N)

    def run_kernel(self, X, N):
        """
        Kernel RX (KRX) anomaly detector.

        Background samples and test pixels are implicitly mapped into an
        RKHS via an RBF kernel, and the linear RX (Mahalanobis distance)
        statistic is evaluated there using only kernel evaluations, following
        Kwon & Nasrabadi, "Kernel RX-Algorithm: A Nonlinear Anomaly Detector
        for Hyperspectral Imagery" (2005):

            delta(x) = k_x^T (K_c + reg*I)^-1 k_x

        where K_c is the (n_bg, n_bg) centered Gram matrix of the background
        samples and k_x is the centered vector of kernel evaluations between
        x and every background sample.
        """
        H, W, B = X.shape
        x_t = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        N_t = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
        if self.mean_N is None:
            self.mean_N = np.mean(N_t, axis=0)
        x_t = x_t - self.mean_N
        N_t = N_t - self.mean_N

        if type(self.kernel) is RbfKernel:
            self.kernel = AutoRbfKernel(N_t)

        K_bg = self.kernel.compute(N_t, N_t) # (n_bg, n_bg) background Gram matrix
        if self.cov is None:
            self.cov = (1 - self.reg) * K_bg + (N_t.shape[0] - 1) * self.reg * np.eye(N_t.shape[0]) # (8) in the paper
        c, low = _safe_cho_factor(self.cov, lower=True)
        k_x = self.kernel.compute(x_t, N_t) # (n_test, n_bg) cross-kernel between test pixels and background
        solved = cho_solve((c, low), k_x.T, check_finite=False) # K_reg_inv @ k_x.T, (n_bg, n_test)
        kxx = self.kernel.diag(x_t) # k(x, x) per test pixel, O(n_test) instead of O(n_test^2)
        scores = kxx - (1 - self.reg) * np.sum(k_x.T * solved, axis=0) # (9) in the paper, diag(k_x @ K_reg_inv @ k_x.T)
        scores = scores/self.reg

        return scores.reshape(H, W)

        # --- Previous implementation, kept for comparison ---------------------
        #
        # Built K_tilde as the Gram matrix of the *test* pixels X against
        # themselves (an (n_test, n_test) matrix) instead of the background N,
        # contradicting this method's own docstring/algorithm: every pixel was
        # implicitly scored against a "background" made of the entire test
        # image rather than the actual MCD-selected background set N (which
        # was computed above only to get mean_N, then discarded). For large
        # images this (n_test, n_test) Gram matrix is also why a subsampling
        # fallback had to be bolted on in main.py for kernel mode. Fixed above
        # to build the (n_bg, n_bg) Gram matrix from N, matching the paper.
        #
        # if type(self.kernel) is RbfKernel:
        #     self.kernel = AutoRbfKernel(x_t)
        #
        # K_tilde = self.kernel.compute(x_t, x_t)
        # if self.cov is None:
        #     self.cov = (1 - self.reg) * K_tilde + (x_t.shape[0] - 1) * self.reg * np.eye(x_t.shape[0]) # (8) in the paper
        # c, low = cho_factor(self.cov, lower=True, check_finite=False)
        # solved = cho_solve((c, low), K_tilde, check_finite=False) # K_reg_inv @ K_tilde
        # kt_diag = np.diag(K_tilde)
        # scores = kt_diag - (1 - self.reg) * np.sum(K_tilde * solved, axis=0) # (9) in the paper, diag(K_tilde @ K_reg_inv @ K_tilde) since both symmetric
        # scores = scores/self.reg
        #
        # return scores.reshape(H, W)
        # ------------------------------------------------------------------------

    def run_kernel_gpu(self, X, N):
        """
        Torch/GPU version of run_kernel. Same algorithm (see run_kernel's
        docstring), but the Gram matrices and the K_reg solve are computed
        with torch tensors on self.device instead of scipy/LAPACK on the CPU.

        The AutoRbfKernel bandwidth is still fit on the CPU via numpy (a
        cheap O(sample size) operation, done once and cached on self.kernel),
        so this produces the exact same sigma as run_kernel. Kernel Gram
        matrices are computed in float64 to match scipy.spatial.distance.cdist's
        precision (it always computes in double regardless of input dtype),
        which keeps results numerically identical to run_kernel to solver
        tolerance rather than just "close" at float32 precision.
        """
        H, W, B = X.shape
        x_np = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        N_np = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
        if self.mean_N is None:
            self.mean_N = np.mean(N_np, axis=0)
        x_np = x_np - self.mean_N
        N_np = N_np - self.mean_N

        if type(self.kernel) is RbfKernel:
            self.kernel = AutoRbfKernel(N_np)

        with torch.no_grad():
            x_t = torch.from_numpy(x_np).double().to(self.device)
            N_t = torch.from_numpy(N_np).double().to(self.device)
            sigma = float(self.kernel.sigma)

            K_bg = _rbf_kernel_torch(N_t, N_t, sigma) # (n_bg, n_bg) background Gram matrix
            if self.cov is None:
                n_bg = N_t.shape[0]
                eye = torch.eye(n_bg, dtype=torch.float64, device=self.device)
                cov_t = (1 - self.reg) * K_bg + (n_bg - 1) * self.reg * eye # (8) in the paper
                self.cov = cov_t.cpu().numpy()
            else:
                cov_t = torch.from_numpy(self.cov).to(self.device)
            L = _safe_cholesky_torch(cov_t)
            k_x = _rbf_kernel_torch(x_t, N_t, sigma) # (n_test, n_bg) cross-kernel between test pixels and background
            solved = torch.cholesky_solve(k_x.T, L) # K_reg_inv @ k_x.T, (n_bg, n_test)
            kxx = torch.from_numpy(self.kernel.diag(x_np)).to(self.device) # k(x, x) per test pixel
            scores = kxx - (1 - self.reg) * torch.sum(k_x.T * solved, dim=0) # (9) in the paper
            scores = scores / self.reg

        return scores.reshape(H, W).detach().cpu().numpy()

    def _median_gamma(self, N_bg):
        # Median heuristic (matches kernels.py's AutoRbfKernel): bandwidth is
        # the reciprocal of the median squared pairwise distance among
        # background samples.
        sqdist = pdist(N_bg, metric='sqeuclidean')
        med = max(np.median(sqdist), 1e-12)
        self.gamma = med

    def set_gamma(self, gamma=None):
        self.gamma = gamma
        self._kernel_bg = None

    def get_gamma(self):
        return self.gamma

    def set_reg(self, reg=0.1):
        self.reg = reg
        self._kernel_bg = None

    def get_reg(self):
        return self.reg

    def load_config(self, config_dict):
        if config_dict.get('kernel') is not None:
            self.set_kernel(config_dict['kernel'])
        if config_dict.get('gamma') is not None:
            self.set_gamma(config_dict['gamma'])
        if config_dict.get('reg') is not None:
            self.set_reg(config_dict['reg'])
        if config_dict.get('batch_size') is not None:
            self.batch_size = config_dict['batch_size']


    def run_gpu(self, X, N):
        H, W, B = X.shape
        x_t = torch.from_numpy(X).float().to(self.device)
        x_t = torch.reshape(x_t, (-1, B)).contiguous()

        with torch.no_grad():
            if self.cov is not None:
                cov_t = torch.from_numpy(self.cov).float().to(self.device).contiguous()
                mean_N = torch.from_numpy(self.mean_N).float().to(self.device).contiguous()
            else:
                N_t = torch.from_numpy(N).float().to(self.device)
                N_t = torch.reshape(N_t, (-1, B)).contiguous()
                mean_N = torch.mean(N_t, dim=0, dtype=torch.float32).to(self.device)
                N_t = N_t - mean_N[None, :] # Overwrite N_t to save memory
                cov_t = N_t.transpose(0, 1) @ N_t
                cov_t /= float(N_t.shape[0] - 1)
            x_t = x_t - mean_N
            solved = torch.linalg.solve(cov_t, x_t.unsqueeze(-1)).squeeze(-1)
            return torch.reshape(torch.sum(x_t * solved, dim=1), (H, W)).detach().cpu().numpy()

    def run_cpu(self, X, N):
        H, W, B = X.shape
        x_t = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)

        if self.cov is not None:
            cov = self.cov
            mean_N = self.mean_N
        else:
            N_t = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
            mean_N = np.mean(N_t, axis=0)
            N_t = N_t - mean_N
            cov = (N_t.T @ N_t) / (N_t.shape[0] - 1)

        x_t = x_t - mean_N
        c, low = _safe_cho_factor(cov, lower=True)
        solved = cho_solve((c, low), x_t.T, check_finite=False).T
        return np.reshape(np.sum(x_t * solved, axis=1), (H, W))

    def set_mean_N(self, mean_N=None):
        self.mean_N = mean_N

    def set_cov(self, cov=None):
        self.cov = cov

    def set_device(self, device=None):
        self.device = device

    def set_kernel(self, kernel=None):
        self.kernel = kernel

class AMF():
    def __init__(self, cov=None, mean_N=None, device=None, kernel=False, gamma=None, reg=0.1, batch_size=2000):
        self.cov = cov
        self.mean_N = mean_N
        self.device = device
        self.kernel = kernel
        self.gamma = gamma
        self.reg = reg
        self.batch_size = batch_size
        self.gpu = False
        self._cache_key = None
        self._k_tilde_cache = None
        self._cho_cache = None
        self._kinv_kx_cache = None
        self._cache_key_gpu = None
        self._k_tilde_cache_gpu = None
        self._cho_cache_gpu = None
        self._kinv_kx_cache_gpu = None

    def __call__(self, X, N, T):
        if self.kernel and self.gpu:
            return self.run_kernel_gpu(X, N, T)
        elif self.kernel:
            return self.run_kernel(X, N, T)
        elif self.gpu:
            return self.run_gpu(X, N, T)
        else:
            return self.run_cpu(X, N, T)

    def run_kernel(self, X, N, T):
        """
        Kernel AMF (KAMF) anomaly detector.

        Generalizes RX.run_kernel's shrinkage-regularized kernel trick (see
        its docstring, eqs. 8-9) from the quadratic RX form to the bilinear
        adaptive matched filter form via

            g(a, b) = [K(a, b) - (1 - reg) * k_a^T S^-1 k_b] / reg

        where S = (1-reg)*K_bg + reg*(n_bg-1)*I is the regularized
        background Gram matrix (eq. 8, built from the background samples N
        this time, since the target T is not itself a background point) and
        k_a is the vector of kernel evaluations between a and every
        background sample. AMF's score is then g(t, x)^2 / g(t, t) --
        exactly the kernelized numerator/denominator of the linear AMF
        statistic (t^T Sigma^-1 x)^2 / (t^T Sigma^-1 t).
        """
        H, W, B = X.shape
        x_t = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        t_t = np.ascontiguousarray(T.reshape(-1, B), dtype=np.float32)
        N_t = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
        if self.mean_N is None:
            self.mean_N = np.mean(N_t, axis=0)
        x_t = x_t - self.mean_N
        t_t = t_t - self.mean_N
        N_t = N_t - self.mean_N

        if type(self.kernel) is RbfKernel:
            self.kernel = AutoRbfKernel(N_t)

        cache_key = (id(N), id(self.mean_N)) # K_bg only depends on the background, not X or T
        if self._cache_key == cache_key:
            k_tilde = self._k_tilde_cache
        else:
            k_tilde = self.kernel.compute(N_t, N_t) # (n_bg, n_bg) background Gram matrix
            self._cache_key = cache_key
            self._k_tilde_cache = k_tilde
        if self.cov is None:
            self.cov = (1 - self.reg) * k_tilde + (N_t.shape[0] - 1) * self.reg * np.eye(N_t.shape[0]) # (8) in the paper
            self._cho_cache = None
            self._kinv_kx_cache = None
        if self._cho_cache is None:
            self._cho_cache = _safe_cho_factor(self.cov, lower=True)
        c, low = self._cho_cache

        k_t = self.kernel.compute(t_t, N_t) # (1, n_bg) target-vs-background cross-kernel
        Kinv_kt = cho_solve((c, low), k_t.T, check_finite=False) # K_reg_inv @ k_t.T
        g_tt = (self.kernel.compute(t_t, t_t)[0, 0] - (1 - self.reg) * (k_t @ Kinv_kt)[0, 0]) / self.reg

        t_x = self.kernel.compute(t_t, x_t) # (1, n_test) raw direct kernel between target and each test pixel
        if self._kinv_kx_cache is None:
            # K_reg_inv @ k_x.T, (n_bg, n_test); depends only on the background
            # and X, not T, so it is reused across the per-target call loop
            # (T varies, X/N stay fixed) instead of being recomputed every call.
            k_x = self.kernel.compute(x_t, N_t) # (n_test, n_bg) test-pixel-vs-background cross-kernel
            self._kinv_kx_cache = cho_solve((c, low), k_x.T, check_finite=False)
        Kinv_kx = self._kinv_kx_cache
        g_tx = (t_x - (1 - self.reg) * (k_t @ Kinv_kx)) / self.reg

        scores = g_tx ** 2 / g_tt
        return scores.reshape(H, W)

        # --- Previous implementation, kept for comparison ---------------------
        #
        # Same bug as RX.run_kernel: k_tilde was the Gram matrix of the test
        # pixels X (via self.kernel.compute(x_t), i.e. x_t against itself)
        # rather than the background N, so every pixel was implicitly matched
        # against the whole test image instead of the MCD-selected background.
        # t_x here served double duty as both the raw K(t, x) term and (since
        # "background" was X) the target-vs-background cross-kernel -- that
        # conflation is why it had to be split into t_x and k_t above.
        #
        # if type(self.kernel) is RbfKernel:
        #     self.kernel = AutoRbfKernel(x_t)
        #
        # cache_key = (id(X), id(N), id(self.mean_N))
        # if self._cache_key == cache_key:
        #     k_tilde = self._k_tilde_cache
        # else:
        #     k_tilde = self.kernel.compute(x_t)
        #     self._cache_key = cache_key
        #     self._k_tilde_cache = k_tilde
        # if self.cov is None:
        #     self.cov = (1 - self.reg) * k_tilde + (x_t.shape[0] - 1) * self.reg * np.eye(x_t.shape[0]) # (8) in the paper
        #     self._cho_cache = None
        # if self._cho_cache is None:
        #     self._cho_cache = cho_factor(self.cov, lower=True, check_finite=False)
        # c, low = self._cho_cache
        #
        # t_x = self.kernel.compute(t_t, x_t)
        # Kinv_tx = cho_solve((c, low), t_x.T, check_finite=False) # K_reg_inv @ t_x.T
        # g_tt = (self.kernel.compute(t_t, t_t)[0, 0] - (1 - self.reg) * (t_x @ Kinv_tx)[0, 0]) / self.reg
        #
        # g_tx = (t_x.T - (1 - self.reg) * (k_tilde @ Kinv_tx)) / self.reg
        #
        # scores = g_tx ** 2 / g_tt
        # return scores.reshape(H, W)
        # ------------------------------------------------------------------------

    def run_kernel_gpu(self, X, N, T):
        """Torch/GPU version of run_kernel; see its docstring for the algorithm."""
        H, W, B = X.shape
        x_np = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        t_np = np.ascontiguousarray(T.reshape(-1, B), dtype=np.float32)
        N_np = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
        if self.mean_N is None:
            self.mean_N = np.mean(N_np, axis=0)
        x_np = x_np - self.mean_N
        t_np = t_np - self.mean_N
        N_np = N_np - self.mean_N

        if type(self.kernel) is RbfKernel:
            self.kernel = AutoRbfKernel(N_np)
        sigma = float(self.kernel.sigma)

        with torch.no_grad():
            x_t = torch.from_numpy(x_np).double().to(self.device)
            t_t = torch.from_numpy(t_np).double().to(self.device)
            N_t = torch.from_numpy(N_np).double().to(self.device)

            cache_key = (id(N), id(self.mean_N)) # K_bg only depends on the background, not X or T
            if self._cache_key_gpu == cache_key:
                k_tilde = self._k_tilde_cache_gpu
            else:
                k_tilde = _rbf_kernel_torch(N_t, N_t, sigma) # (n_bg, n_bg) background Gram matrix
                self._cache_key_gpu = cache_key
                self._k_tilde_cache_gpu = k_tilde
            if self.cov is None:
                n_bg = N_t.shape[0]
                eye = torch.eye(n_bg, dtype=torch.float64, device=self.device)
                cov_t = (1 - self.reg) * k_tilde + (n_bg - 1) * self.reg * eye # (8) in the paper
                self.cov = cov_t.cpu().numpy()
                self._cho_cache_gpu = None
                self._kinv_kx_cache_gpu = None
            else:
                cov_t = torch.from_numpy(self.cov).to(self.device)
            if self._cho_cache_gpu is None:
                self._cho_cache_gpu = _safe_cholesky_torch(cov_t)
            L = self._cho_cache_gpu

            k_t = _rbf_kernel_torch(t_t, N_t, sigma) # (1, n_bg) target-vs-background cross-kernel
            Kinv_kt = torch.cholesky_solve(k_t.T, L) # K_reg_inv @ k_t.T
            g_tt = (_rbf_kernel_torch(t_t, t_t, sigma)[0, 0] - (1 - self.reg) * (k_t @ Kinv_kt)[0, 0]) / self.reg

            t_x = _rbf_kernel_torch(t_t, x_t, sigma) # (1, n_test) raw direct kernel between target and each test pixel
            if self._kinv_kx_cache_gpu is None:
                # K_reg_inv @ k_x.T, (n_bg, n_test); depends only on the background
                # and X, not T, so it is reused across the per-target call loop
                # (T varies, X/N stay fixed) instead of being recomputed every call.
                k_x = _rbf_kernel_torch(x_t, N_t, sigma) # (n_test, n_bg) test-pixel-vs-background cross-kernel
                self._kinv_kx_cache_gpu = torch.cholesky_solve(k_x.T, L)
            Kinv_kx = self._kinv_kx_cache_gpu
            g_tx = (t_x - (1 - self.reg) * (k_t @ Kinv_kx)) / self.reg

            scores = g_tx ** 2 / g_tt
        return scores.reshape(H, W).detach().cpu().numpy()

    def _median_gamma(self, N_bg):
        sqdist = pdist(N_bg, metric='sqeuclidean')
        med = max(np.median(sqdist), 1e-12)
        self.gamma = med

    def set_gamma(self, gamma=None):
        self.gamma = gamma

    def get_gamma(self):
        return self.gamma

    def set_reg(self, reg=0.1):
        self.reg = reg

    def get_reg(self):
        return self.reg

    def set_kernel(self, kernel=None):
        self.kernel = kernel
        self._cache_key = None
        self._k_tilde_cache = None
        self._cho_cache = None
        self._kinv_kx_cache = None
        self._cache_key_gpu = None
        self._k_tilde_cache_gpu = None
        self._cho_cache_gpu = None
        self._kinv_kx_cache_gpu = None

    def load_config(self, config_dict):
        if config_dict.get('kernel') is not None:
            self.set_kernel(config_dict['kernel'])
        if config_dict.get('gamma') is not None:
            self.set_gamma(config_dict['gamma'])
        if config_dict.get('reg') is not None:
            self.set_reg(config_dict['reg'])
        if config_dict.get('batch_size') is not None:
            self.batch_size = config_dict['batch_size']

    def run_gpu(self, X, N, T):
        H, W, B = X.shape
        x_t = torch.from_numpy(X).float().to(self.device)
        t_t = torch.from_numpy(T).float().to(self.device)
        x_t = torch.reshape(x_t, (-1, B)).contiguous()
        t_t = torch.reshape(t_t, (-1, B)).contiguous()

        with torch.no_grad():
            if self.cov is not None:
                cov_t = torch.from_numpy(self.cov).float().to(self.device).contiguous()
                mean_N = torch.from_numpy(self.mean_N).float().to(self.device).contiguous()
            else:
                N_t = torch.from_numpy(N).float().to(self.device)
                N_t = torch.reshape(N_t, (-1, B)).contiguous()
                mean_N = torch.mean(N_t, dim=0, dtype=torch.float32).to(self.device)
                N_t = N_t - mean_N[None, :] # Overwrite N_t to save memory
                cov_t = N_t.transpose(0, 1) @ N_t
                cov_t /= float(N_t.shape[0] - 1)
            t_t = t_t - mean_N
            x_t = x_t - mean_N
            solved_x = torch.linalg.solve(cov_t, x_t.unsqueeze(-1)).squeeze(-1)
            solved_t = torch.linalg.solve(cov_t, t_t.unsqueeze(-1)).squeeze(-1)
            numerator = torch.sum(t_t * solved_x, dim=1) ** 2
            denominator = torch.sum(t_t * solved_t, dim=1)
            score = numerator / denominator
        return torch.reshape(score, (H, W)).detach().cpu().numpy()

    def run_cpu(self, X, N, T):
        H, W, B = X.shape
        x_t = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        t_t = np.ascontiguousarray(T.reshape(-1, B), dtype=np.float32)

        if self.cov is not None:
            cov = self.cov
            mean_N = self.mean_N
        else:
            N_t = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
            mean_N = np.mean(N_t, axis=0)
            N_t = N_t - mean_N
            cov = (N_t.T @ N_t) / (N_t.shape[0] - 1)

        x_t = x_t - mean_N
        t_t = t_t - mean_N
        c, low = _safe_cho_factor(cov, lower=True)
        solved_x = cho_solve((c, low), x_t.T, check_finite=False).T
        solved_t = cho_solve((c, low), t_t.T, check_finite=False).T
        numerator = np.sum(t_t * solved_x, axis=1) ** 2
        denominator = np.sum(t_t * solved_t, axis=1)
        return np.reshape(numerator / denominator, (H, W))

    def set_mean_N(self, mean_N=None):
        self.mean_N = mean_N

    def set_cov(self, cov=None):
        self.cov = cov
        self._cho_cache = None
        self._kinv_kx_cache = None
        self._cho_cache_gpu = None
        self._kinv_kx_cache_gpu = None

    def set_device(self, device=None):
        self.device = device

class ACE():
    def __init__(self, cov=None, mean_N=None, device=None, kernel=False, gamma=None, reg=0.1, batch_size=2000):
        self.cov = cov
        self.mean_N = mean_N
        self.device = device
        self.kernel = kernel
        self.gamma = gamma
        self.reg = reg
        self.batch_size = batch_size
        self.gpu = False
        self._cache_key = None
        self._k_tilde_cache = None
        self._cho_cache = None
        self._gxx_cache = None
        self._cache_key_gpu = None
        self._k_tilde_cache_gpu = None
        self._cho_cache_gpu = None
        self._gxx_cache_gpu = None

    def __call__(self, X, N, T):
        if self.kernel and self.gpu:
            return self.run_kernel_gpu(X, N, T)
        elif self.kernel:
            return self.run_kernel(X, N, T)
        elif self.gpu:
            return self.run_gpu(X, N, T)
        else:
            return self.run_cpu(X, N, T)

    def run_kernel(self, X, N, T):
        """
        Kernel ACE (KACE) anomaly detector.

        Same shrinkage-regularized kernel trick and bilinear form g(a, b) as
        AMF.run_kernel, but normalized by both the target's and the test
        pixel's own g(., .) term, mirroring the linear ACE statistic
        (t^T Sigma^-1 x)^2 / [(t^T Sigma^-1 t)(x^T Sigma^-1 x)]:

            score(x) = g(t, x)^2 / (g(t, t) * g(x, x))

        Note g(x, x) is exactly RX.run_kernel's statistic for x against the
        same background.
        """
        H, W, B = X.shape
        x_t = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        t_t = np.ascontiguousarray(T.reshape(-1, B), dtype=np.float32)
        N_t = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
        if self.mean_N is None:
            self.mean_N = np.mean(N_t, axis=0)
        x_t = x_t - self.mean_N
        t_t = t_t - self.mean_N
        N_t = N_t - self.mean_N

        if type(self.kernel) is RbfKernel:
            self.kernel = AutoRbfKernel(N_t)

        cache_key = (id(N), id(self.mean_N)) # K_bg only depends on the background, not X or T
        if self._cache_key == cache_key:
            k_tilde = self._k_tilde_cache
        else:
            k_tilde = self.kernel.compute(N_t, N_t) # (n_bg, n_bg) background Gram matrix
            self._cache_key = cache_key
            self._k_tilde_cache = k_tilde
        if self.cov is None:
            self.cov = (1 - self.reg) * k_tilde + (N_t.shape[0] - 1) * self.reg * np.eye(N_t.shape[0]) # (8) in the paper
            self._cho_cache = None
            self._gxx_cache = None
        if self._cho_cache is None:
            self._cho_cache = _safe_cho_factor(self.cov, lower=True)
        c, low = self._cho_cache

        k_t = self.kernel.compute(t_t, N_t) # (1, n_bg) target-vs-background cross-kernel
        Kinv_kt = cho_solve((c, low), k_t.T, check_finite=False) # K_reg_inv @ k_t.T
        g_tt = (self.kernel.compute(t_t, t_t)[0, 0] - (1 - self.reg) * (k_t @ Kinv_kt)[0, 0]) / self.reg

        t_x = self.kernel.compute(t_t, x_t) # (1, n_test) raw direct kernel between target and each test pixel
        k_x = self.kernel.compute(x_t, N_t) # (n_test, n_bg) test-pixel-vs-background cross-kernel
        g_tx = (t_x.T - (1 - self.reg) * (k_x @ Kinv_kt)) / self.reg
        g_tx = g_tx[:, 0]

        if self._gxx_cache is None:
            kxx = self.kernel.diag(x_t) # k(x, x) per test pixel, O(n_test) instead of O(n_test^2)
            Kinv_kx = cho_solve((c, low), k_x.T, check_finite=False) # K_reg_inv @ k_x.T
            g_xx = kxx - (1 - self.reg) * np.sum(k_x.T * Kinv_kx, axis=0) # (9) in the paper
            self._gxx_cache = g_xx / self.reg
        g_xx = self._gxx_cache

        scores = g_tx ** 2 / (g_tt * g_xx)
        return scores.reshape(H, W)

        # --- Previous implementation, kept for comparison ---------------------
        #
        # Same bug as RX.run_kernel/AMF.run_kernel: k_tilde was the Gram
        # matrix of the test pixels X against themselves rather than the
        # background N, so g_xx (and g_tx via k_tilde) implicitly matched
        # every pixel against the whole test image instead of the
        # MCD-selected background. t_x served double duty as both the raw
        # K(t, x) term and the target-vs-background cross-kernel; that
        # conflation is why it's split into t_x and k_t above.
        #
        # if type(self.kernel) is RbfKernel:
        #     self.kernel = AutoRbfKernel(x_t)
        #
        # cache_key = (id(X), id(N), id(self.mean_N))
        # if self._cache_key == cache_key:
        #     k_tilde = self._k_tilde_cache
        # else:
        #     k_tilde = self.kernel.compute(x_t, x_t)
        #     self._cache_key = cache_key
        #     self._k_tilde_cache = k_tilde
        # if self.cov is None:
        #     self.cov = (1 - self.reg) * k_tilde + (x_t.shape[0] - 1) * self.reg * np.eye(x_t.shape[0]) # (8) in the paper
        #     self._cho_cache = None
        #     self._gxx_cache = None
        # if self._cho_cache is None:
        #     self._cho_cache = cho_factor(self.cov, lower=True, check_finite=False)
        # c, low = self._cho_cache
        #
        # t_x = self.kernel.compute(t_t, x_t)
        # Kinv_tx = cho_solve((c, low), t_x.T, check_finite=False) # K_reg_inv @ t_x.T
        # g_tt = (self.kernel.compute(t_t, t_t)[0, 0] - (1 - self.reg) * (t_x @ Kinv_tx)[0, 0]) / self.reg
        #
        # g_tx = (t_x.T - (1 - self.reg) * (k_tilde @ Kinv_tx)) / self.reg
        # g_tx = g_tx[:, 0]
        #
        # if self._gxx_cache is None:
        #     kt_diag = np.diag(k_tilde)
        #     Kinv_ktilde = cho_solve((c, low), k_tilde, check_finite=False) # K_reg_inv @ k_tilde
        #     g_xx = kt_diag - (1 - self.reg) * np.sum(k_tilde * Kinv_ktilde, axis=0) # (9) in the paper
        #     self._gxx_cache = g_xx / self.reg
        # g_xx = self._gxx_cache
        #
        # scores = g_tx ** 2 / (g_tt * g_xx)
        # return scores.reshape(H, W)
        # ------------------------------------------------------------------------

    def run_kernel_gpu(self, X, N, T):
        """Torch/GPU version of run_kernel; see its docstring for the algorithm."""
        H, W, B = X.shape
        x_np = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        t_np = np.ascontiguousarray(T.reshape(-1, B), dtype=np.float32)
        N_np = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
        if self.mean_N is None:
            self.mean_N = np.mean(N_np, axis=0)
        x_np = x_np - self.mean_N
        t_np = t_np - self.mean_N
        N_np = N_np - self.mean_N

        if type(self.kernel) is RbfKernel:
            self.kernel = AutoRbfKernel(N_np)
        sigma = float(self.kernel.sigma)

        with torch.no_grad():
            x_t = torch.from_numpy(x_np).double().to(self.device)
            t_t = torch.from_numpy(t_np).double().to(self.device)
            N_t = torch.from_numpy(N_np).double().to(self.device)

            cache_key = (id(N), id(self.mean_N)) # K_bg only depends on the background, not X or T
            if self._cache_key_gpu == cache_key:
                k_tilde = self._k_tilde_cache_gpu
            else:
                k_tilde = _rbf_kernel_torch(N_t, N_t, sigma) # (n_bg, n_bg) background Gram matrix
                self._cache_key_gpu = cache_key
                self._k_tilde_cache_gpu = k_tilde
            if self.cov is None:
                n_bg = N_t.shape[0]
                eye = torch.eye(n_bg, dtype=torch.float64, device=self.device)
                cov_t = (1 - self.reg) * k_tilde + (n_bg - 1) * self.reg * eye # (8) in the paper
                self.cov = cov_t.cpu().numpy()
                self._cho_cache_gpu = None
                self._gxx_cache_gpu = None
            else:
                cov_t = torch.from_numpy(self.cov).to(self.device)
            if self._cho_cache_gpu is None:
                self._cho_cache_gpu = _safe_cholesky_torch(cov_t)
            L = self._cho_cache_gpu

            k_t = _rbf_kernel_torch(t_t, N_t, sigma) # (1, n_bg) target-vs-background cross-kernel
            Kinv_kt = torch.cholesky_solve(k_t.T, L) # K_reg_inv @ k_t.T
            g_tt = (_rbf_kernel_torch(t_t, t_t, sigma)[0, 0] - (1 - self.reg) * (k_t @ Kinv_kt)[0, 0]) / self.reg

            t_x = _rbf_kernel_torch(t_t, x_t, sigma) # (1, n_test) raw direct kernel between target and each test pixel
            k_x = _rbf_kernel_torch(x_t, N_t, sigma) # (n_test, n_bg) test-pixel-vs-background cross-kernel
            g_tx = (t_x.T - (1 - self.reg) * (k_x @ Kinv_kt)) / self.reg
            g_tx = g_tx[:, 0]

            if self._gxx_cache_gpu is None:
                kxx = torch.from_numpy(self.kernel.diag(x_np)).to(self.device) # k(x, x) per test pixel
                Kinv_kx = torch.cholesky_solve(k_x.T, L) # K_reg_inv @ k_x.T
                g_xx = kxx - (1 - self.reg) * torch.sum(k_x.T * Kinv_kx, dim=0) # (9) in the paper
                self._gxx_cache_gpu = g_xx / self.reg
            g_xx = self._gxx_cache_gpu

            scores = g_tx ** 2 / (g_tt * g_xx)
        return scores.reshape(H, W).detach().cpu().numpy()

    def _median_gamma(self, N_bg):
        sqdist = pdist(N_bg, metric='sqeuclidean')
        med = max(np.median(sqdist), 1e-12)
        self.gamma = med

    def set_gamma(self, gamma=None):
        self.gamma = gamma

    def get_gamma(self):
        return self.gamma

    def set_reg(self, reg=0.1):
        self.reg = reg

    def get_reg(self):
        return self.reg

    def set_kernel(self, kernel=None):
        self.kernel = kernel
        self._cache_key = None
        self._k_tilde_cache = None
        self._cho_cache = None
        self._gxx_cache = None
        self._cache_key_gpu = None
        self._k_tilde_cache_gpu = None
        self._cho_cache_gpu = None
        self._gxx_cache_gpu = None

    def load_config(self, config_dict):
        if config_dict.get('kernel') is not None:
            self.set_kernel(config_dict['kernel'])
        if config_dict.get('gamma') is not None:
            self.set_gamma(config_dict['gamma'])
        if config_dict.get('reg') is not None:
            self.set_reg(config_dict['reg'])
        if config_dict.get('batch_size') is not None:
            self.batch_size = config_dict['batch_size']

    def run_gpu(self, X, N, T):
        H, W, B = X.shape
        x_t = torch.from_numpy(X).float().to(self.device)
        t_t = torch.from_numpy(T).float().to(self.device)
        x_t = torch.reshape(x_t, (-1, B)).contiguous()
        t_t = torch.reshape(t_t, (-1, B)).contiguous()

        with torch.no_grad():
            if self.cov is not None:
                cov_t = torch.from_numpy(self.cov).float().to(self.device).contiguous()
                mean_N = torch.from_numpy(self.mean_N).float().to(self.device).contiguous()
            else:
                N_t = torch.from_numpy(N).float().to(self.device)
                N_t = torch.reshape(N_t, (-1, B)).contiguous()
                mean_N = torch.mean(N_t, dim=0, dtype=torch.float32).to(self.device)
                N_t = N_t - mean_N[None, :] # Overwrite N_t to save memory
                cov_t = N_t.transpose(0, 1) @ N_t
                cov_t /= float(N_t.shape[0] - 1)
            t_t = t_t - mean_N
            x_t = x_t - mean_N
            solved_x = torch.linalg.solve(cov_t, x_t.unsqueeze(-1)).squeeze(-1)
            solved_t = torch.linalg.solve(cov_t, t_t.unsqueeze(-1)).squeeze(-1)
            numerator = torch.sum(t_t * solved_x, dim=1) ** 2
            denominator = torch.sum(t_t * solved_t, dim=1) * torch.sum(x_t * solved_x, dim=1)
            score = numerator / denominator
        return torch.reshape(score, (H, W)).detach().cpu().numpy()

    def run_cpu(self, X, N, T):
        H, W, B = X.shape
        x_t = np.ascontiguousarray(X.reshape(-1, B), dtype=np.float32)
        t_t = np.ascontiguousarray(T.reshape(-1, B), dtype=np.float32)

        if self.cov is not None:
            cov = self.cov
            mean_N = self.mean_N
        else:
            N_t = np.ascontiguousarray(N.reshape(-1, B), dtype=np.float32)
            mean_N = np.mean(N_t, axis=0)
            N_t = N_t - mean_N
            cov = (N_t.T @ N_t) / (N_t.shape[0] - 1)

        x_t = x_t - mean_N
        t_t = t_t - mean_N
        c, low = _safe_cho_factor(cov, lower=True)
        solved_x = cho_solve((c, low), x_t.T, check_finite=False).T
        solved_t = cho_solve((c, low), t_t.T, check_finite=False).T
        numerator = np.sum(t_t * solved_x, axis=1) ** 2
        denominator = np.sum(t_t * solved_t, axis=1) * np.sum(x_t * solved_x, axis=1)
        return np.reshape(numerator / denominator, (H, W))

    def set_mean_N(self, mean_N=None):
        self.mean_N = mean_N

    def set_cov(self, cov=None):
        self.cov = cov
        self._cho_cache = None
        self._gxx_cache = None
        self._cho_cache_gpu = None
        self._gxx_cache_gpu = None

    def set_device(self, device=None):
        self.device = device

class LRX():
    def __init__(self, window=(5, 15), cov=None, device=None):
        self.window = window
        self.cov = cov
        self.batch_size = 100
        self.device = device

    def __call__(self, X):
        """
        Local Reed-Xiaoli for Hyperspectral Anomaly Detection.

        Parameters:
        -----------
        X : ndarray of shape (H, W, B)
            The input hyperspectral image cube (Height, Width, Bands).

        Returns:
        --------
        anomaly_map : ndarray of shape (H, W)
            A 2D map containing the calculated anomaly score for each pixel.
        """
        H, W, B = X.shape
        inner, outer = self.window
        r_out = outer // 2
        divider = calc_subwindow_divider(X, self.window, size_limit=3e9)
        
        row_dividers = np.arange(0, H+1, H/divider).astype(np.int32)
        col_dividers = np.arange(0, W+1, W/divider).astype(np.int32)

        X = np.pad(X, ((r_out, r_out), (r_out, r_out), (0, 0)), mode='reflect').astype(np.float32, copy=False)
        anomaly_map = np.zeros((H, W), dtype=np.float32)
        for i in range(len(row_dividers)-1):
            for j in range(len(col_dividers)-1):
                row_start = row_dividers[i] 
                row_end = min(row_dividers[i+1], H)
                col_start = col_dividers[j]
                col_end = min(col_dividers[j+1], W) 
                number_of_rows, number_of_cols = row_end-row_start, col_end-col_start
                row_end_patch = row_end + r_out*2 # Add r_out*2 due to padding
                col_end_patch = col_end + r_out*2
                row_start_pad = row_start + r_out # Add r_out to "remove" padding and get the non padded indices
                row_end_pad = row_end + r_out
                col_start_pad = col_start + r_out
                col_end_pad = col_end + r_out

                patches = windowing(X[row_start:row_end_patch, 
                                      col_start:col_end_patch], 
                                      self.window)
                output_lrx = self.CRD(patches, 
                                               X[row_start_pad:row_end_pad,
                                                 col_start_pad:col_end_pad].reshape(-1, B))
                output_lrx = output_lrx.reshape((number_of_rows, number_of_cols))
                anomaly_map[row_start:row_end,
                            col_start:col_end] = output_lrx

        return anomaly_map

    def lrx_calc(self, N, x):
        """
        Local Reed-Xiaoli for Hyperspectral Anomaly Detection.

        Parameters:
        -----------
        N : ndarray of shape (batch, D, B)
            The background dictionary tensor for each target pixel.
        x : ndarray of shape (batch, B)
            The spectral signatures of the target pixels to be evaluated.

        Returns:
        --------
        anomaly_score : ndarray of shape (batch,)
            The calculated anomaly score for each target pixel.
        """
        D, B = N.shape[-2], N.shape[-1]
        N = np.ascontiguousarray(N.reshape(-1, D, B), dtype=np.float32)
        x = np.ascontiguousarray(x.reshape(-1, B), dtype=np.float32)
        n = x.shape[0]

        anomaly_score = np.empty(n, dtype=np.float32)
        mean_N = np.mean(N, axis=1)

        if self.cov is not None:
            cov = np.ascontiguousarray(self.cov, dtype=np.float32)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_N = N[start:end]
                batch_x = x[start:end]
                mean = mean_N[start:end]
                diff_x = batch_x - mean
                solved = np.linalg.solve(cov, diff_x[..., None])[..., 0]
                anomaly_score[start:end] = np.sqrt(np.sum(diff_x * solved, axis=1))
            return anomaly_score

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_N = N[start:end]
            batch_x = x[start:end]
            mean = mean_N[start:end]
            diff = batch_N - mean[:, None, :]
            diff_x = batch_x - mean
            cov = diff.transpose(0, 2, 1) @ diff
            cov /= (D - 1)
            solved = np.linalg.solve(cov, diff_x[..., None])[..., 0]
            anomaly_score[start:end] = np.sqrt(np.sum(diff_x * solved, axis=1)) # lrx from spectral does not do sqrt

        return anomaly_score

    def run_gpu(self, X):
        H, W, B = X.shape
        inner, outer = self.window
        r_out = outer // 2
        divider = calc_subwindow_divider(X, self.window, size_limit=3e9)
        
        row_dividers = np.arange(0, H+1, H/divider).astype(np.int32)
        col_dividers = np.arange(0, W+1, W/divider).astype(np.int32)

        X = torch.from_numpy(X).float().to(self.device).contiguous()
        X = X.permute(2, 0, 1)
        X = F.pad(X, pad=(r_out, r_out, r_out, r_out), mode='reflect')
        X = X.permute(1, 2, 0)
        anomaly_map = np.zeros((H, W), dtype=np.float32)
        anomaly_map = torch.from_numpy(anomaly_map).float().to(self.device).contiguous()
        for i in range(len(row_dividers)-1):
            for j in range(len(col_dividers)-1):
                row_start = row_dividers[i] 
                row_end = min(row_dividers[i+1], H)
                col_start = col_dividers[j]
                col_end = min(col_dividers[j+1], W) 
                number_of_rows, number_of_cols = row_end-row_start, col_end-col_start
                row_end_patch = row_end + r_out*2 # Add r_out*2 due to padding
                col_end_patch = col_end + r_out*2
                row_start_pad = row_start + r_out # Add r_out to "remove" padding and get the non padded indices
                row_end_pad = row_end + r_out
                col_start_pad = col_start + r_out
                col_end_pad = col_end + r_out

                patches = windowing_gpu(X[row_start:row_end_patch, 
                                      col_start:col_end_patch], 
                                      self.window)
                output_lrx = self.lrx_calc_gpu(patches, 
                                               X[row_start_pad:row_end_pad,
                                                 col_start_pad:col_end_pad].reshape(-1, B))
                output_lrx = output_lrx.reshape((number_of_rows, number_of_cols))
                anomaly_map[row_start:row_end,
                            col_start:col_end] = output_lrx

        anomaly_map = anomaly_map.detach().cpu().numpy()
        # del X, patches, output_lrx
        # torch.cuda.empty_cache()
        return anomaly_map
    
    def lrx_calc_gpu(self, N, x):
        """
        GPU-accelerated Local Reed-Xiaoli for Hyperspectral Anomaly Detection.

        Parameters:
        -----------
        N : ndarray of shape (batch, D, B)
            The background dictionary tensor for each target pixel.
        x : ndarray of shape (batch, B)
            The spectral signatures of the target pixels to be evaluated.

        Returns:
        --------
        anomaly_score : ndarray of shape (batch,)
            The calculated anomaly score for each target pixel.
        """
        
        D, B = N.shape[-2], N.shape[-1]
        # N_t = torch.from_numpy(N).float().to(device)
        # x_t = torch.from_numpy(x).float().to(device)
        N_t = torch.reshape(N, (-1, D, B)).contiguous()
        x_t = torch.reshape(x, (-1, B)).contiguous()
        n = x.shape[0]

        # anomaly_score = torch.from_numpy(np.empty(n, dtype=np.float32)).type(torch.float32).contiguous().to(device)
        with torch.no_grad():
            mean_N = torch.mean(N_t, dim=1, dtype=torch.float32).to(self.device)
            x_t = x_t - mean_N 
            if self.cov is not None:
                cov_t = torch.from_numpy(self.cov).float().to(self.device).contiguous()
            else:
                N_t = N_t - mean_N[:, None, :] # Overwrite N_t to save memory
                cov_t = N_t.transpose(1, 2) @ N_t
                cov_t /= float(D - 1)
            solved = torch.linalg.solve(cov_t, x_t.unsqueeze(-1)).squeeze(-1)
        return torch.sqrt(torch.sum(x_t * solved, dim=1))

    def load_config(self, config_dict):
        if config_dict.get('window') != None:
            self.set_window(config_dict['window'])
        if config_dict.get('cov') != None:
            self.set_cov(config_dict['cov'])
        if config_dict.get('batch_size') != None:
            self.batch_size = config_dict['batch_size']

    def set_window(self, window):
        self.window = window

    def get_window(self):
        return self.window

    def set_cov(self, cov):
        self.cov = cov

    def get_cov(self):
        return self.cov

    def get_device(self):
        return self.device
    
    def set_device(self, device):
        self.device = device

class CRD():
    def __init__(self, window=(5, 15), l=1e-3, device=None):
        self.window = window
        self.lamda = float(l)
        # Precompute the diagonal matrix for the regularization term based on the window size
        self._diag = self.lamda * np.eye(window[1]**2 - window[0]**2).astype(np.float32)
        # Lazily-built (lambda * I_B) for the band-space solve; depends on the
        # number of bands, which is only known once data arrives.
        self._diag_B = None
        self.batch_size = 100 # Batch size for processing pixels
        self.device = device

    def __call__(self, X):
        """
        Collaborative Representation-Based Detector (CRD) for Hyperspectral Anomaly Detection.

        Parameters:
        -----------
        X: ndarray of shape (H, W, B)
            The input hyperspectral image cube (Height, Width, Bands).

        Returns:
        --------
        anomaly_map: ndarray of shape (H, W)
            A 2D map containing the calculated anomaly score for each pixel.

        Notes
        -----
        Optimized version. The original implementation extracted the ring-shaped
        background window with a pure-Python double loop over every pixel (one
        ``windowing`` call each) and then reconstructed each pixel inside another
        Python loop. Here:

        * Patch extraction is fully vectorized with ``sliding_window_view`` plus a
          single boolean ring mask -- mathematically identical to calling
          ``windowing`` on the padded cube for every pixel, but without the
          per-pixel Python overhead.
        * The detector itself (:meth:`CRD`) is solved in batches as stacked linear
          systems, so the heavy lifting stays in vectorized LAPACK/BLAS calls.
        """
        H, W, B = X.shape
        if B <= self.window[1]**2 - self.window[0]**2:
            self._diag_B = self.lamda * np.eye(B).astype(np.float32)
        inner, outer = self.window
        r_out = outer // 2
        divider = calc_subwindow_divider(X, self.window, size_limit=3e9)
        
        row_dividers = np.arange(0, H+1, H/divider).astype(np.int32)
        col_dividers = np.arange(0, W+1, W/divider).astype(np.int32)

        X = np.pad(X, ((r_out, r_out), (r_out, r_out), (0, 0)), mode='reflect').astype(np.float32, copy=False)
        anomaly_map = np.zeros((H, W), dtype=np.float32)
        for i in range(len(row_dividers)-1):
            for j in range(len(col_dividers)-1):
                row_start = row_dividers[i] 
                row_end = min(row_dividers[i+1], H)
                col_start = col_dividers[j]
                col_end = min(col_dividers[j+1], W) 
                number_of_rows, number_of_cols = row_end-row_start, col_end-col_start
                row_end_patch = row_end + r_out*2 # Add r_out*2 due to padding
                col_end_patch = col_end + r_out*2
                row_start_pad = row_start + r_out # Add r_out to "remove" padding and get the non padded indices
                row_end_pad = row_end + r_out
                col_start_pad = col_start + r_out
                col_end_pad = col_end + r_out

                patches = windowing(X[row_start:row_end_patch, 
                                      col_start:col_end_patch], 
                                      self.window)
                output_lrx = self.CRD(patches, 
                                               X[row_start_pad:row_end_pad,
                                                 col_start_pad:col_end_pad].reshape(-1, B))
                output_lrx = output_lrx.reshape((number_of_rows, number_of_cols))
                anomaly_map[row_start:row_end,
                            col_start:col_end] = output_lrx

        return anomaly_map

    def CRD(self, N, x):
        """
        Collaborative Representation-Based Detector (CRD) for Hyperspectral Anomaly Detection.

        Parameters:
        -----------
        N : ndarray of shape (batch, D, B) -- or any shape whose last two axes are
            (D, B); it is flattened to (batch, D, B). Each row of a slice is a
            background pixel's spectral signature.
        x : ndarray of shape (batch, B)
            The spectral signatures of the target pixels to be evaluated.

        Returns:
        --------
        anomaly_score : ndarray of shape (batch,)
            The calculated anomaly score for each target pixel.

        Notes
        -----
        The CRD score is the L2 norm of the reconstruction residual

            r = x - N^T (N N^T + lambda I_D)^-1 N x .

        Two algebraically identical ways to evaluate it exist, and we pick the
        cheaper one per call based on whether there are more bands (B) or more
        background pixels (D):

        * **B <= D** (the usual hyperspectral case): the push-through identity
          ``N^T (N N^T + lambda I_D)^-1 N = (N^T N + lambda I_B)^-1 N^T`` collapses
          the residual to ``r = lambda (N^T N + lambda I_B)^-1 x``. This solves a
          small (B x B) system instead of a large (D x D) one, and -- crucially --
          it never forms ``x - x_reconstructed``, avoiding the catastrophic
          float32 cancellation that occurs for background pixels (where the
          reconstruction is almost exactly x). It is therefore both much faster
          and markedly more accurate.

        * **B > D**: the (D x D) system is the smaller one, so we use the direct
          reconstruction formulation.
        """
        # Flatten any leading axes into a single batch dimension.
        D, B = N.shape[-2], N.shape[-1]
        N = np.ascontiguousarray(N.reshape(-1, D, B), dtype=np.float32)
        x = np.ascontiguousarray(x.reshape(-1, B), dtype=np.float32)

        if B <= D:
            # Band-space solve: r = lambda * (N^T N + lambda I_B)^-1 x
            # if self._diag_B is None or self._diag_B.shape[0] != B:
            #     self._diag_B = self.lamda * np.eye(B, dtype=np.float32)
            M = N.transpose(0, 2, 1) @ N            # (batch, B, B)
            M += self._diag_B
            residual = self.lamda * np.linalg.solve(M, x[..., None])[..., 0]
        else:
            # Pixel-space solve (smaller when B > D):
            #   alpha_hat = (N N^T + lambda I_D)^-1 (N x);  r = x - N^T alpha_hat
            A = N @ N.transpose(0, 2, 1)            # (batch, D, D)
            A += self._diag
            b = N @ x[..., None]                    # (batch, D, 1)
            alpha = np.linalg.solve(A, b)
            x_reconstructed = (N.transpose(0, 2, 1) @ alpha)[..., 0]   # (batch, B)
            residual = x - x_reconstructed

        # Anomaly score is the Euclidean (L2) norm of the reconstruction residual.
        anomaly_score = np.sqrt(np.sum(residual ** 2, axis=1))

        return anomaly_score.squeeze()

    def run_gpu(self, X):
        H, W, B = X.shape
        if B <= self.window[1]**2 - self.window[0]**2:
            self._diag_B = self.lamda * np.eye(B).astype(np.float32)
            self._diag_B = torch.from_numpy(self._diag_B).to(self.device)
        else:
            self._diag = torch.from_numpy(self._diag).to(self.device)
        inner, outer = self.window
        r_out = outer // 2
        divider = calc_subwindow_divider(X, self.window, size_limit=3e9)
        
        row_dividers = np.arange(0, H+1, H/divider).astype(np.int32)
        col_dividers = np.arange(0, W+1, W/divider).astype(np.int32)

        X = torch.from_numpy(X).float().to(self.device).contiguous()
        X = X.permute(2, 0, 1)
        X = F.pad(X, pad=(r_out, r_out, r_out, r_out), mode='reflect')
        X = X.permute(1, 2, 0)
        anomaly_map = np.zeros((H, W), dtype=np.float32)
        anomaly_map = torch.from_numpy(anomaly_map).float().to(self.device).contiguous()
        for i in range(len(row_dividers)-1):
            for j in range(len(col_dividers)-1):
                row_start = row_dividers[i] 
                row_end = min(row_dividers[i+1], H)
                col_start = col_dividers[j]
                col_end = min(col_dividers[j+1], W) 
                number_of_rows, number_of_cols = row_end-row_start, col_end-col_start
                row_end_patch = row_end + r_out*2 # Add r_out*2 due to padding
                col_end_patch = col_end + r_out*2
                row_start_pad = row_start + r_out # Add r_out to "remove" padding and get the non padded indices
                row_end_pad = row_end + r_out
                col_start_pad = col_start + r_out
                col_end_pad = col_end + r_out

                patches = windowing_gpu(X[row_start:row_end_patch, 
                                      col_start:col_end_patch], 
                                      self.window)
                output_lrx = self.CRD_gpu(patches, 
                                               X[row_start_pad:row_end_pad,
                                                 col_start_pad:col_end_pad].reshape(-1, B))
                output_lrx = output_lrx.reshape((number_of_rows, number_of_cols))
                anomaly_map[row_start:row_end,
                            col_start:col_end] = output_lrx
        anomaly_map = anomaly_map.detach().cpu().numpy()
        # del X, patches, output_lrx
        # torch.cuda.empty_cache()
        return anomaly_map

    def CRD_gpu(self, N, x):
        D, B = N.shape[-2], N.shape[-1]
        N_t = N.to(self.device).view(-1, D, B)
        x_t = x.to(self.device).view(-1, B)

        with torch.no_grad():
            if B <= D:
                # band-space solve: r = lambda * (N^T N + lambda I_B)^-1 x
                N_t = N_t.transpose(1, 2) @ N_t # Overwrite N_t to save memory
                # add diag_B
                N_t += self._diag_B.unsqueeze(0)
                solved = torch.linalg.solve(N_t, x_t.unsqueeze(-1)).squeeze(-1)
                residual = float(self.lamda) * solved
            else:
                # pixel-space solve
                b = N_t @ x_t
                N_t = N_t @ N_t.transpose(1, 2) # Overwrite N_t to save memory
                N_t += self._diag.unsqueeze(0)
                alpha = torch.linalg.solve(N_t, b)
                x_reconstructed = N_t.transpose(1, 2) @ alpha
                residual = x_t - x_reconstructed.squeeze(-1)

        return torch.sqrt(torch.sum(residual ** 2, dim=1))

    def load_config(self, config_dict):
        if config_dict.get('window') != None:
            self.set_window(config_dict['window'])
        if config_dict.get('lamda') != None:
            self.set_lamda(config_dict['lamda'])
        if config_dict.get('batch_size') != None:
            self.batch_size = config_dict['batch_size']
        self._diag_B = None  # rebuilt lazily for the new lambda

    def set_window(self, window):
        self.window = window

    def get_window(self):
        return self.window

    def set_lamda(self, l):
        self.lamda = float(l)
        self._diag = self.lamda * np.eye(self.window[1]**2 - self.window[0]**2).astype(np.float32)

    def get_lamda(self):
        return self.lamda

    def get_device(self):
        return self.device
    
    def set_device(self, device):
        self.device = device

def create_AD_model(model_name="RX"):
    model_dict = {"RX": RX(),
                  "LRX": LRX(),
                  "CRD": CRD(),
                  'AMF': AMF(),
                  'ACE': ACE()}
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f'{model_name} not known.')

def windowing(X, window=(5, 15)):
    """Return values in a ring-shaped window around (row, col) with an inner exclusion region.

    X is a 3D matrix. window may be an int or a tuple (inner_radius, outer_radius).
    The returned array contains values between the inner and outer square neighborhoods.
    """
    H, W, B = X.shape
    inner, outer = window

    # Calculate half-windows for boundary padding
    r_out = outer // 2
    r_in = inner // 2
    win = 2 * r_out + 1  # full side length of the outer (square) window

    # Ring mask: full outer window with the central inner window excluded.
    # This reproduces exactly what `windowing` returns for an interior pixel,
    # including the C-order flattening of the kept entries.
    mask = np.ones((win, win), dtype=bool)
    lo = r_out - r_in
    hi = r_out + r_in + 1
    mask[lo:hi, lo:hi] = False
    D = int(mask.sum())  # number of background pixels in the ring

    # Vectorized extraction of every window at once.
    # sliding_window_view -> (H, W, B, win, win); reorder so the two spatial
    # window axes are adjacent to the band axis, then apply the ring mask.
    sw = np.lib.stride_tricks.sliding_window_view(X, (win, win), axis=(0, 1))
    del X
    sw = sw.transpose(0, 1, 3, 4, 2)              # (H, W, win, win, B)
    patches = sw[:, :, mask, :]                   # (H, W, D, B)

    # patches = np.ascontiguousarray(patches, dtype=np.float32).reshape(H * W, D, B)
    H0, W0 = H-r_out*2, W-r_out*2
    patches = patches.reshape(H0 * W0, D, B)
    return patches

def windowing_gpu(X, window=(5, 15)):
    """GPU implementation of windowing using PyTorch.

    X: torch.Tensor of shape (H, W, B) on GPU (or CPU).
    window: (inner, outer)
    Returns torch.Tensor of shape (H*W, D, B) with dtype float32 on same device.
    """
    H, W, B = X.shape
    inner, outer = window
    r_out = outer // 2
    r_in = inner // 2
    win = 2 * r_out + 1

    # build mask (on CPU) then move to X device
    mask = np.ones((win, win), dtype=bool)
    lo = r_out - r_in
    hi = r_out + r_in + 1
    mask[lo:hi, lo:hi] = False
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(X.device)
    D = int(mask.sum())

    # Prepare tensor: (H,W,B) -> (B,H,W)
    X = X.permute(2, 0, 1)

    # sliding windows using unfold -> (B, win, win, H, W)
    X = X.unfold(1, win, 1).unfold(2, win, 1)
    # reorder to (H, W, win*win, B)
    X = X.permute(1, 2, 3, 4, 0).contiguous()
    H0, W0 = X.shape[0], X.shape[1]
    X = X.view(H0, W0, win * win, B)

    # select ring positions
    sel = mask_flat.bool()
    del mask_flat
    patches = X[:, :, sel, :].reshape(H0 * W0, D, B).to(dtype=torch.float32)
    return patches

def calc_subwindow_divider(X, window, size_limit=5e8):
    H, W, B = X.shape
    inner, outer = window
    r_out = outer // 2
    window_size = outer**2 - inner**2
    X_size = X.size * 4 # Amount of elements in X * 4 bytes for np.float32
    size_windowing = X_size * window_size # Total size of resulting matrix with windowed data
    lim_exceeded = size_windowing / size_limit # Size limit of 3e9 results in roughly 30GB of GPU memory
    log4_lim_exceeded = np.emath.logn(4, lim_exceeded) # log4 due to halving of both H and W
    divider = int(np.ceil(np.pow(4, max(log4_lim_exceeded, 1)-1)))
    # if divider>1:
    #     pass
    while divider > 1 and H/divider < window[1]: # Make sure that the size of the subwindow is larger than the window size
        divider = divider - 1 
    return divider