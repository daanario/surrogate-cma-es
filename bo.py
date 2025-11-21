# bo.py
# by Daan

# Implementation of Bayesian Optimization with an EI acquisition function

import numpy as np
from kernel import GaussianKernel, MaternKernel
from gp import GPR
from scipy.stats import norm
from scipy.optimize import minimize

class Archive:
    # global archive of points and function values in untransformed (global) coordinates
    def __init__(self, x, y):
        self.x = np.atleast_2d(x) # x is an array of points with shape (n, d), where d is the dimensionality of the points
        self.y = np.atleast_1d(y).reshape(-1, 1) # y is a column vector of function values (n, 1)
    def add_points(self, new_x, new_y):
        # add new raw points to the archive
        self.x = np.vstack((self.x, new_x))
        self.y = np.vstack((self.y, new_y))

class BayesOpt:
    def __init__(self, kernel=None, bound_min=-5.0, bound_max=5.0):
        self.kernel = kernel or GaussianKernel()
        self.model = None
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.archive = None

    def polish(self, x_init):
        # Local refinement of the EI maximum using Nelder-Mead, optimizing log(EI + eps).
        eps = 1e-16

        def neg_log_ei(v):
            v = np.clip(v, self.bound_min, self.bound_max)
            ei_val = self.expected_improvement(v[None, :])

            # ensure ei_val is a scalar
            # expected_improvement sometimes returns scalar, sometimes array
            if np.ndim(ei_val) == 0:
                ei_val = float(ei_val)
            else:
                ei_val = float(ei_val[0])

            # numerical stabilization
            ei_val = max(ei_val, eps)

            return -np.log(ei_val)

        res = minimize(
            neg_log_ei,
            x_init,
            method="L-BFGS-B",
            options={"maxiter": 20},
        )

        return np.clip(res.x, self.bound_min, self.bound_max)
    
    def acquisition_function(self, in_dim, num_samples=500):
        # Global random samples
        X = np.random.uniform(
            low=self.bound_min,
            high=self.bound_max,
            size=(num_samples, in_dim),
        )

        # Expected Improvement (EI) for each point
        ei_vals = self.expected_improvement(X)
        ei_vals = np.atleast_1d(ei_vals).reshape(-1)

        # sanity check
        if not np.all(np.isfinite(ei_vals)):
            # fallback, return current best observed x
            idx_best = np.argmin(self.model.y)
            return self.model.unnormalize_x(self.model.x[idx_best])

        # choose the point with maximum EI
        idx = int(np.argmax(ei_vals))
        x_acq = X[idx]

        # polish using a few iterations of Nelder-Mead
        return self.polish(x_acq)
    
    def expected_improvement(self, x):
        x = np.atleast_2d(x)
        mu, std, _ = self.model.predict(x) # update posterior
        std = np.maximum(std, 1e-12) # ensure we don't divide by zero

        # predictions give unnormalized values, so f_star must also be unnormalized
        f_star = np.min(self.model.unnormalize_y(self.model.y))
        delta = f_star - mu
        z = delta / std
        ei = delta * norm.cdf(z) + std * norm.pdf(z)
        #print("debug EI: mu =", mu, " std =", std, " f_star =", f_star, " delta=", f_star-mu)
        return ei
    
    def optimize(self, objective_fn, x0, budget=20, start_samples=2):
        if budget < start_samples:
            print("Error! budget is smaller than the amount of starting samples")
            return 0
        
        in_dim = len(x0)

        bmin = self.bound_min
        bmax = self.bound_max

        # sample starting points        
        data_x = []
        data_y = []
        for i in range(0, start_samples):
            random_x = np.random.uniform(bmin, bmax, size=(in_dim,)) # sample x uniformly at random
            data_x.append(random_x)
            data_y.append(objective_fn(random_x))

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        # initialize GP model
        self.archive = Archive(data_x, data_y)
        self.model = GPR(data_x, data_y, kernel=self.kernel)

        # set starting value of prior mean (recommended by DTS-CMA-ES paper)
        #self.model.prior_mean = np.median(self.model.y)

        # set model's starting hyperparameters based on the random samples
        self.model.fit()

        n = start_samples
        for n in range(start_samples, budget):
            # choose which point to sample next
            # posterior distribution is computed when calling predict() inside acquisition function
            self.model = GPR(self.archive.x, self.archive.y, kernel=self.kernel)
            x_next = self.acquisition_function(in_dim)

            # observe objective function at acquired point
            y_next = objective_fn(x_next)

            # add acquired point and function value to GP model archive
            self.archive.add_points(x_next, y_next)
            
            # update hyperparameters every 10 iterations
            #if n % 10 == 0:
            #    self.model.fit()

            # update hyperparameters
            self.model.fit()

        opt_argmin = self.model.unnormalize_x(self.model.x[np.argmin(self.model.y, axis=0), :]).flatten()
        
        return opt_argmin

def bo_minimize(f, x0):
    # simple scipy-like wrapper function for minimizing an objective
    x0 = np.atleast_1d(x0)
    dim = x0.size
    budget = 50*dim # set budget according to Oswin's paper

    optimizer = BayesOpt(kernel=MaternKernel())
    return optimizer.optimize(f, x0, budget=budget)