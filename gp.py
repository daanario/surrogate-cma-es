# gp.py
# by Daan

# Gaussian Process regression class with hyperparameter fitting
# Uses Cholesky solves for stability and speed instead of the naive full matrix inverse

import numpy as np
import copy
from kernel import GaussianKernel, MaternKernel
from scipy.optimize import minimize

class GPR:
    def __init__(self, x, y, kernel=GaussianKernel(), prior_mean=0.0,
                 noise=0.0, x_normalized=True):

        # store raw data
        self.raw_x = np.atleast_2d(x)
        self.raw_y = np.atleast_1d(y).reshape(-1, 1)

        # determines if GP should normalize points itself
        self.x_normalized = x_normalized

        # compute normalization stats (on raw data)
        self.x_mean = self.raw_x.mean(axis=0)
        self.x_std = self.raw_x.std(axis=0) + 1e-12
        self.y_mean = self.raw_y.mean()
        self.y_std = self.raw_y.std() + 1e-12

        # normalize x/y
        if x_normalized:
            self.x = self.normalize_x(self.raw_x)
        else:
            self.x = self.raw_x
        self.y = self.normalize_y(self.raw_y)

        self.kernel = kernel
        self.noise = noise
        self.prior_mean = prior_mean
        self.eps = 1e-7

        # Cholesky factor L and alpha
        self.L = None
        self.alpha = None

        # initialize factorizationß
        self.compute_cholesky()

    # normalization helpers
    def normalize_x(self, x):
        return (np.atleast_2d(x) - self.x_mean) / self.x_std

    def normalize_y(self, y):
        return (np.atleast_1d(y).reshape(-1, 1) - self.y_mean) / self.y_std

    def unnormalize_y(self, y):
        return y * self.y_std + self.y_mean

    def unnormalize_x(self, x):
        return x * self.x_std + self.x_mean

    # recompute Cholesky factorization from scratch
    def compute_cholesky(self):
        n = len(self.x)
        if n == 0:
            self.L = None
            self.alpha = None
            return

        # build K (full covariance matrix)
        K = self.kernel.matrix(self.x, self.x)

        # add noise + jitter to the diagonal
        K_y = K + (self.noise + self.eps) * np.eye(n)

        # Cholesky decompose into lower triangular matrix
        try:
            L = np.linalg.cholesky(K_y)
        except np.linalg.LinAlgError:
            raise RuntimeError("Cholesky failed: kernel matrix not PD")

        self.L = L

        # update alpha
        self._update_alpha()

    def _update_alpha(self):
        # Compute alpha = (K_y)^(-1)(y - prior_mean) via triangular solves
        if self.L is None or self.y is None:
            self.alpha = None
            return

        y_centered = self.y - self.prior_mean

        # forward solve
        v = np.linalg.solve(self.L, y_centered)

        # backward solve
        alpha = np.linalg.solve(self.L.T, v)

        self.alpha = alpha

    # Add points to archive and perform incremental Cholesky update
    def add_points(self, new_x, new_y):
        # Add a new data point and update Cholesky via rank-one update
        new_x = np.atleast_2d(new_x)
        new_y = np.atleast_1d(new_y).reshape(-1, 1)

        # update raw archive
        self.raw_x = np.vstack((self.raw_x, new_x))
        self.raw_y = np.vstack((self.raw_y, new_y))

        # normalize new point(s)
        if self.x_normalized:
            new_x_norm = self.normalize_x(new_x)
        else:
            new_x_norm = new_x
        new_y_norm = self.normalize_y(new_y)

        n_old = self.x.shape[0]

        # add normalized points to archive
        self.x = np.vstack((self.x, new_x_norm))
        self.y = np.vstack((self.y, new_y_norm))

        # if L not initialized or wrong size, rebuild from scratch and return early
        if self.L is None or self.L.shape[0] != n_old:
            self.compute_cholesky()
            return
        # else do fast Cholesky rank-one update
        # k_vec = k(x_old, x_new)
        k_vec = self.kernel.matrix(self.x[:n_old], self.x[n_old:]).reshape(n_old)

        # k_ss = k(x_new, x_new) + noise + eps
        k_ss = float(self.kernel.matrix(self.x[n_old:], self.x[n_old:])[0, 0])
        k_ss += (self.noise + self.eps)
        # Solve L_old v = k_vec
        v = np.linalg.solve(self.L, k_vec)

        # Compute new diagonal element
        diag_new = k_ss - np.dot(v, v)
        assert diag_new > 0, f"diag_new <= 0 (={diag_new}): kernel not PD"

        l_new = np.sqrt(diag_new)

        # Build enlarged L
        n_new = n_old + 1
        L_new = np.zeros((n_new, n_new))
        L_new[:n_old, :n_old] = self.L
        L_new[n_old, :n_old] = v
        L_new[n_old, n_old] = l_new

        self.L = L_new

        # update alpha
        self._update_alpha()

    # Hyperparameter fitting
    def fit_points(self, x, y):
        # MLE hyperparam fit using provided subset of points
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        _, len_scale = self.kernel.get_params()
        _, len_scale_bounds = self.kernel.param_bounds()

        def objective(len_arr):
            ls = float(len_arr[0])
            temp_k = copy.deepcopy(self.kernel)
            temp_k.set_params([self.kernel.alpha, ls])
            return self.neg_ll(x, y, self.prior_mean, temp_k)

        res = minimize(objective, [len_scale], bounds=[len_scale_bounds], options={"maxiter": 10})
        ls_hat = float(res.x[0])

        self.kernel.set_params([self.kernel.alpha, ls_hat])
        self.compute_cholesky()

    def fit(self):
        # MLE hyperparam fit on the full archive 
        _, len_scale = self.kernel.get_params()
        _, len_scale_bounds = self.kernel.param_bounds()

        def objective(len_arr):
            ls = float(len_arr[0])
            temp_k = copy.deepcopy(self.kernel)
            temp_k.set_params([self.kernel.alpha, ls])
            return self.neg_ll(self.x, self.y, self.prior_mean, temp_k)

        res = minimize(objective, [len_scale], bounds=[len_scale_bounds], options={"maxiter": 10})
        ls_hat = float(res.x[0])

        self.kernel.set_params([self.kernel.alpha, ls_hat])
        self.compute_cholesky()

    # Negative log-likelihood
    def neg_ll(self, x, y, mu, temp_kernel):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        n = len(y)

        cov = temp_kernel.matrix(x, x) + np.eye(n) * self.eps

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            raise RuntimeError("Cholesky failed in neg_ll")

        y_centered = y - mu
        v = np.linalg.solve(L, y_centered)
        quad = float(v.T @ v)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        ll = -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
        return -ll

    # set hyperparams directly
    def set_hyperparams(self, mu_hat, noise, kernel_params_hat):
        self.kernel.set_params(kernel_params_hat)
        self.noise = noise
        self.prior_mean = mu_hat
        self.compute_cholesky()

    # GP prediction
    def predict(self, x_new):
        x_new = np.atleast_2d(x_new)
        if self.x_normalized:
            x_new_norm = self.normalize_x(x_new)
        else:
            x_new_norm = x_new

        n_train = self.x.shape[0]
        n_test = x_new_norm.shape[0]

        k_star = self.kernel.matrix(self.x, x_new_norm)

        # mean
        if self.alpha is None:
            mu_norm = np.full(n_test, float(self.prior_mean))
        else:
            mu_norm = (self.prior_mean + k_star.T @ self.alpha).flatten()

        # covariance
        k_star_star = self.kernel.matrix(x_new_norm, x_new_norm)

        if self.L is None:
            cov_norm = k_star_star
        else:
            v = np.linalg.solve(self.L, k_star)
            cov_norm = k_star_star - v.T @ v

        # unnormalize covariance
        cov_unnorm = cov_norm * (self.y_std ** 2)

        mu = self.unnormalize_y(mu_norm)

        sigma = np.sqrt(np.maximum(np.diag(cov_unnorm), 0))

        if len(mu) == 1:
            return mu[0], sigma[0], cov_unnorm
        return mu, sigma, cov_unnorm