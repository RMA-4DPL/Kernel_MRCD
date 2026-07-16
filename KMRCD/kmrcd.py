"""
kMRCD: outlier detection in non-elliptical data by kernel MRCD.
J. Schreurs, I. Vranckx et al.

Python port of ``KMRCD/KMRCD.m``.

The minimum regularized covariance determinant (MRCD) is a robust estimator
for multivariate location and scatter, which detects outliers by fitting a
robust covariance matrix to the data. The MRCD assumes that the observations
are elliptically distributed. However, this property does not always apply
to modern datasets. Together with the time criticality of industrial
processing, small n, large p problems pose a challenging problem for any
data analytics procedure. Both shortcomings are solved with the proposed
kernel Minimum Regularized Covariance Determinant estimator, where we
exploit the kernel trick to speed up computations. More specifically, the
MRCD location and scatter matrix estimate are computed in a kernel induced
feature space, where regularization ensures that the covariance matrix is
well-conditioned, invertible and defined for any dataset dimension.

Minimal working example
------------------------
    model = KMRCD(LinKernel())
    solution = model.run_algorithm(x, alpha=0.75)

Reference: https://github.com/Joachim-Sh/Outlier-detection-in-non-elliptical-data-by-kernel-MRCD 
Licenced under the Non-Profit Open Software License version 3.0 (NPOSL-3.0) 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .kernels import LinKernel, RbfKernel, AutoRbfKernel
from . import utils


class Kernel_MRCD:

    def __init__(self, alpha=0.5, kernel=None, c_step_iterations_allowed=100, maxcond=50):
        self.kernel = kernel if kernel is not None else LinKernel()
        self.c_step_iterations_allowed = c_step_iterations_allowed
        self.maxcond = maxcond
        self.alpha = alpha

    def run_algorithm(self, x):
        assert 0.5 <= self.alpha <= 1, (
            "The percentage of regular observations, alpha, should be in [0.5-1]"
        )

        # RoS-LSSVM hack, carried over from the MATLAB source: a bare
        # RbfKernel (not the automatically-tuned subclass) is not usable
        # here, so it is swapped for an AutoRbfKernel fit on the data.
        if type(self.kernel) is RbfKernel:
            print("Warning: kMRCD switches to AutoRbfKernel in case a RbfKernel was specified!!")
            self.kernel = AutoRbfKernel(x)

        x = np.asarray(x, dtype=float)
        n, p = x.shape
        K = self.kernel.compute(x, x)

        # Grab observation ranking from initial estimators
        solutions = [
            SimpleNamespace(name="SDO", outlyingness_indices=utils.sdo(K, self.alpha)),
            SimpleNamespace(name="SpatialRank", outlyingness_indices=utils.spatial_rank(K, self.alpha)),
            SimpleNamespace(name="SpatialMedian", outlyingness_indices=utils.spatial_median_estimator(K, self.alpha)),
            SimpleNamespace(name="SSCM", outlyingness_indices=utils.sscm(K)),
        ]

        scfac = utils.mcd_cons(p, self.alpha)
        h = int(np.ceil(n * self.alpha))

        # For all initial estimators, do:
        rho_list = np.full(len(solutions), np.nan)
        for i, sol in enumerate(solutions):
            sol.hsubset_indices = sol.outlyingness_indices[:h]

            # Determine rho for each estimator
            idx = sol.hsubset_indices
            s = np.linalg.svd(utils.center(self.kernel.compute(x[idx], x[idx])), compute_uv=False)
            nx = len(idx)
            e_min, e_max = s.min(), s.max()

            def fncond(rho, nx=nx, e_min=e_min, e_max=e_max):
                return (nx * rho + (1 - rho) * scfac * e_max) / (nx * rho + (1 - rho) * scfac * e_min) - self.maxcond

            try:
                rho_i = brentq(fncond, 1e-6, 0.99)
            except ValueError:
                # Find value closest to maxcond instead
                grid = np.linspace(1e-6, 1 - 1e-6, 1000)
                objgrid = np.abs([fncond(g) for g in grid])
                rho_i = grid[objgrid == objgrid.min()].min()
            rho_list[i] = rho_i

        # Set rho as max of the rho_i's obtained for each subset in previous step
        rho = rho_list[rho_list <= max(0.1, np.median(rho_list))].max()

        # Refine each initial estimation with C-steps
        Ktt_diag = np.diag(K)
        for sol in solutions:
            converged = False
            for iteration in range(1, self.c_step_iterations_allowed + 1):
                h_subset = sol.hsubset_indices
                Kx = self.kernel.compute(x[h_subset], x[h_subset])
                nx = Kx.shape[0]
                Kt = self.kernel.compute(x, x[h_subset])
                Kc = utils.center(Kx)
                Kt_c = utils.center(Kx, Kt)
                Kxx = Ktt_diag - (2 / nx) * Kt.sum(axis=1) + (1 / nx ** 2) * Kx.sum()
                M = (1 - rho) * scfac * Kc + nx * rho * np.eye(nx)
                smd = (1 / rho) * (Kxx - (1 - rho) * scfac * np.sum((Kt_c @ np.linalg.inv(M)) * Kt_c, axis=1))
                new_hsubset = np.argsort(smd, kind="stable")[:nx]
                sol.M = M
                sol.Kc = Kc
                # Redefine the h-subset
                sol.hsubset_indices = new_hsubset
                if set(h_subset).issubset(set(new_hsubset)):
                    print(f"Convergence at iteration {iteration}, {sol.name}")
                    sigma = np.linalg.svd(Kc, compute_uv=False)
                    sigma = (1 - rho) * scfac * sigma + len(new_hsubset) * rho
                    sol.obj = np.sum(np.log(sigma))
                    sol.smd = smd
                    converged = True
                    break
            assert converged, "no C-step convergence"

        # Select the solution with the lowest objective function ...
        solution = min(solutions, key=lambda s: s.obj)
        # ... the other solutions are simply discarded (not kept, unlike MATLAB's struct array).
        print(f"-> Best estimator is {solution.name}")

        solution.rho = rho
        solution.scfac = scfac

        # Outlier flagging procedure
        solution.rd = np.maximum(np.sqrt(solution.smd), 0)
        solution.ld = np.log(0.1 + solution.rd)
        tmcd, smcd = utils.unimcd(solution.ld, len(solution.hsubset_indices))
        solution.cutoff = np.exp(tmcd + norm.ppf(0.995) * smcd) - 0.1
        solution.flagged_outlier_indices = np.where(solution.rd > solution.cutoff)[0]

        return solution
