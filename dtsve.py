# dtsve.py
# By Daan

# Implementation of the DTSVE-CMA-ES with weight variance selection criterion and imputed rank weights
# Based on "Recombination Weight Based Selection in the DTS-CMA-ES" by Oswin Krause (2022)

import numpy as np
from kernel import GaussianKernel, MaternKernel
from gp import GPR
from cmaes import CMA_ES
from scipy.stats import norm, chi

class Archive:
    # global archive of points and function values in untransformed (global) coordinates
    def __init__(self, x, y):
        self.x = np.atleast_2d(x) # x is an array of points with shape (n, d), where d is the dimensionality of the points
        self.y = np.atleast_1d(y).reshape(-1, 1) # y is a column vector of function values (n, 1)
    def add_points(self, new_x, new_y):
        # add new raw points to the archive
        self.x = np.vstack((self.x, new_x))
        self.y = np.vstack((self.y, new_y))

class DTSVE_CMA_ES:
    def __init__(self, x_init, kernel=None, population_size=None, recorder=None):
        self.kernel = kernel or GaussianKernel()
        self.cma = CMA_ES(x_init, population_size)
        self.cma.mean = np.asarray(x_init)
        self.cma.step_size = 8/3 # default for DTS-CMA-ES
        self.model = None # assume GP model
        self.prev_kernel_params = None # warm start: use previous length scale as starting point for each fit, since all data sets are similar with TSS1
        self._rng = np.random.default_rng()
        self.dim = len(x_init)
        self.archive = None
        self.recorder = recorder

        self.curr_best = np.inf
        self.archive_size = 0

    def train_model(self):
        # TSS1: choose up to N_max most recent points
        if self.dim == None:
            raise TypeError("dim must be an integer")
        N_max = 30 * self.dim
        
        # take up to N_max of raw points from the archive
        if self.archive_size < N_max:
            x = self.archive.x[-self.archive_size:, :]
            y = self.archive.y[-self.archive_size:, :]
        else:
            x = self.archive.x[-N_max:, :]
            y = self.archive.y[-N_max:, :]

        # transform selected points into CMA-ES basis
        x_local = self.cma.to_local_coords(x)
        
        r_max = chi.ppf(0.99, df=self.dim) + 2 # 4 * 0.99 quantile of sqrt(chi-square) with dim degrees of freedom
        
        # filter to points within maximum radius
        sel = np.linalg.norm(x_local, axis=1)<r_max
        x_local = x_local[sel]
        y = y[sel] # filter y to same points

        # create new model for the subset of CMA-ES local points
        self.model = GPR(x_local, y, kernel=self.kernel,
                         noise=0.0, prior_mean=0.0, x_normalized=False)
        
        if self.prev_kernel_params is not None:
            # use old length scale as starting point
            self.model.set_hyperparams(1.0, 0.0, self.prev_kernel_params)

        # current best should be the best out of the data we actually trained with,
        # not out of the entire archive
        if y.size > 0:
            self.curr_best = np.min(y)
            #self.curr_best = np.min(self.archive.y) # use unstandardized y
        else:
            print("x:", x_local)
            print("y:", y)
            raise ValueError("train_model(): x or y must not be empty!")
        
        # fit model hyperparameters according to selection of points
        #self.model.fit_points(x_local, y)
        self.model.fit()
        self.prev_kernel_params = self.model.kernel.get_params() # use this len scale as starting point for next training
    
    def rank_weight_stats(self, rng, pred_means, pred_cov, K=1000):
        lam = self.cma.params.lam
        mu  = self.cma.params.mu

        # Construct CMA-ES rank weights
        w_prime = np.array([
            np.log((lam+1)/2) - np.log(i+1) if i < mu else 0.0
            for i in range(lam)
        ])
        w_rank = w_prime / np.sum(w_prime)
        
        # Cholesky factorization (lower triangular matrix to easily compute samples vectorized)
        L = np.linalg.cholesky(pred_cov + 1e-12*np.eye(lam))

        # Draw all K standard-normal samples at once
        # rng is an instance of numpy.random.Generator
        Z = rng.standard_normal((lam, K))

        # Generate all K function samples
        F = pred_means[:, None] + L @ Z

        # Rank each column into weight matrix W 
        # argsort twice returns ranks
        order = np.argsort(F, axis=0) # indices sorted by value

        # allocate
        W = np.zeros_like(F)

        # assign CMA-ES weights
        W[order, np.arange(K)] = w_rank[:, None]

        # Compute statistics
        mean_w = np.mean(W, axis=1)
        var_w  = np.var(W, axis=1)

        return mean_w, var_w

    def optimize(self, objective_fn, budget):
        rng = np.random.default_rng()
        orig_ratio = 0.5 # hardcoded ratio
        for n in range(int(budget)):
            # if archive is non-existent or too small, run standard CMA-ES
            # until archive is large enough for fitting
            if self.archive_size < 3 * self.dim:
                x_pop = self.cma.ask()
                y_eval = np.array([objective_fn(x) for x in x_pop])
                x_pop_local = self.cma.to_local_coords(x_pop)
                if self.archive == None:
                    self.archive = Archive(x_pop, y_eval) 
                else:
                    self.archive.add_points(x_pop, np.atleast_1d(y_eval).reshape(-1, 1))
                self.cma.tell(x_pop, y_eval)
                if self.recorder != None:
                    self.recorder.record(self.cma, objective_fn, len(y_eval))
                self.archive_size = len(self.archive.y)
                continue
            
            x_pop = self.cma.ask() # in GLOBAL coordinates
            self.train_model()

            # only predict on CMA-ES local samples (since we always fit with points in local CMA-ES coordinates)
            x_pop_local = self.cma.to_local_coords(x_pop)

            pred_means, _, pred_cov = self.model.predict(x_pop_local)

            # select points with highest monte carlo variance to be original-evaluated
            _, var_w = self.rank_weight_stats(rng, pred_means, pred_cov)
            #print(f"[iter {n}]  Var(W) stats: min={var_w.min():.3e}, max={var_w.max():.3e}, mean={var_w.mean():.3e}")
            #n_orig = int(np.ceil(orig_ratio*self.cma.params.lam)) # amount of original evaluated points
            n_orig = 1
            if self.dim >= 10:
                n_orig = 2
            best_indices = np.argsort(-var_w)[:n_orig]

            # predictions and therefore EI were based on CMA-ES local coordinates
            # therefore transform back to global coordinates
            x_orig = self.cma.to_global_coords(x_pop_local[best_indices])
            y_orig = np.array([objective_fn(x) for x in x_orig]).reshape(-1,1) # observe function at best points

            # add original-evaluated points to model archive
            # GP will standardize y_orig
            self.archive.add_points(x_orig, y_orig)

            # re-train model with updated archive
            self.train_model()

            # predict after re-training
            pred_means2, _, pred_cov2 = self.model.predict(x_pop_local)
            
            # generate sample function values from posterior distribution
            mean_w, _ = self.rank_weight_stats(rng, pred_means2, pred_cov2)

            # put into writing:
            # path affected by this??
            
            # cut out points when variance is 0?

            # update archive size
            self.archive_size = len(self.archive.y)

            # CMA-ES update based on expected weights
            self.cma.tell(x_pop, -mean_w)
            if self.recorder != None:
                self.recorder.record(self.cma, objective_fn, len(y_orig))

        if self.archive != None:
            opt_argmin = self.archive.x[np.argmin(self.archive.y, axis=0), :].flatten()
        else:
            raise ValueError("Archive doesn't exist!")
        return opt_argmin

def dtsve_minimize(f, x0):
    # simple scipy-like wrapper function for minimizing an objective
    x0 = np.atleast_1d(x0)
    dim = x0.size
    budget = 50*dim # set max iteration budget

    optimizer = DTSVE_CMA_ES(x0, kernel=MaternKernel())
    return optimizer.optimize(f, budget=budget)