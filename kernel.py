# kernel.py
# by Daan

# Kernels useful for GP regression

import numpy as np

class BaseKernel:
    def __call__(self, x1, x2):
        raise NotImplementedError
    def get_params(self):
        raise NotImplementedError
    def set_params(self, params):
        raise NotImplementedError
    def param_bounds(self):
        raise NotImplementedError

class GaussianKernel(BaseKernel):
    # Gaussian/squared exponential kernel
    def __init__(self, alpha=1.0, sigma=1.0):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
    def __call__(self, arg1, arg2):
        r2 = np.sum((np.asarray(arg1) - np.asarray(arg2))**2)
        return float(self.alpha * np.exp(-0.5 * r2 / (self.sigma**2)))
    def matrix(self, x1s, x2s):
        dists = np.sum(x1s**2, 1)[:, None] + np.sum(x2s**2, 1)[None, :] - 2 * x1s @ x2s.T
        return self.alpha * np.exp(-0.5 / self.sigma**2 * dists)
    def get_params(self):
        return np.array([self.alpha, self.sigma])
    def set_params(self, params):
        self.alpha, self.sigma = float(params[0]), float(params[1])
    def param_bounds(self):
        return [(np.exp(-2), np.exp(6)), (np.exp(-2), np.exp(6))]

class MaternKernel(BaseKernel):
    # Matern 5/2 kernel
    def __init__(self, alpha=1.0, sigma=1.0):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
    def __call__(self, arg1, arg2):
        r = np.linalg.norm(np.asarray(arg1) - np.asarray(arg2))
        s = self.sigma
        sqrt5r = np.sqrt(5.0) * r
        return float(self.alpha * (1.0 + sqrt5r / s + 5.0 * r**2 / (3.0 * s**2)) * np.exp(-sqrt5r / s))
    def matrix(self, x1s, x2s):
        x1s, x2s = np.atleast_2d(x1s), np.atleast_2d(x2s)
        # pairwise Euclidean distances
        diff = x1s[:, None, :] - x2s[None, :, :]
        r = np.linalg.norm(diff, axis=2) # shape (n, m)
        s = self.sigma
        sqrt5r = np.sqrt(5.0) * r
        K = self.alpha * (1.0 + sqrt5r / s + 5.0 * r**2 / (3.0 * s**2)) * np.exp(-sqrt5r / s)
        return K
    def get_params(self):
        return np.array([self.alpha, self.sigma])
    def set_params(self, params):
        self.alpha, self.sigma = float(params[0]), float(params[1])
    def param_bounds(self):
        return [(np.exp(-2), np.exp(2)), (np.exp(-2), np.exp(2))]