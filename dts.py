# dts.py
# By Daan

# Implementation of the DTS-CMA-ES with EI
# Based on "Gaussian Process Surrogate Models for the CMA Evolution Strategy" by Bajer et al. (2018)

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

def expected_improvement(pred_means, pred_stds, current_best=None):
    # EI acquisition function
    # predictions give unnormalized values, so make sure current_best is also unnormalized
    pred_means = np.atleast_1d(pred_means) # predicted means from GP
    pred_stds = np.atleast_1d(pred_stds) # predicted standard deviations from GP

    if current_best is None:
        raise ValueError("current_best must be provided to compute Expected Improvement")
    
    improvements = []
    for mu, sigma in zip(pred_means, pred_stds):
        sigma = np.maximum(sigma, 1e-12)
        delta = current_best - mu
        z = delta / sigma
        ei = delta * norm.cdf(z) + sigma * norm.pdf(z)
        improvements.append(ei)
    return np.array(improvements)

class DTS_CMA_ES:
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
        self.recorder = recorder # for data collection

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
        
        r_max = chi.ppf(0.99, df=self.dim) + 2 # 0.99 quantile of sqrt(chi-square) with dim degrees of freedom
        
        # heuristic to keep all points in low dimensions (filtering is too aggressive)
        if self.dim >= 2:
            # filter by radius, but ensure enough points remain (especially important for low dimensions)
            sel = np.linalg.norm(x_local, axis=1) < r_max
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
            #self.curr_best = np.min(self.archive.y)
        else:
            print("x:", x_local)
            print("y:", y)
            raise ValueError("train_model(): x or y must not be empty!")
        
        # fit model hyperparameters according to selection of points
        self.model.fit()
        self.prev_kernel_params = self.model.kernel.get_params() # use this len scale as starting point for next training

    def optimize(self, objective_fn, budget):
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

            pred_means, pred_stds, _ = self.model.predict(x_pop_local)

            # select multiple points with best expected improvement to be original-evaluated
            improvements = expected_improvement(pred_means, pred_stds, self.curr_best)
            
            # UNUSED: this n_orig uses more function evaluations
            #n_orig = int(np.ceil(0.5*self.cma.params.lam))
            
            n_orig = 1
            if self.dim >= 10:
                n_orig = 2
            
            best_indices = np.argsort(-improvements)[:n_orig]

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
            pred_means2, _, _ = self.model.predict(x_pop_local)
            
            y_orig = y_orig.flatten()

            pred_means2[best_indices] = y_orig # fitness-replace with the original-evaluated points

            # this clips the predictions; might cause issues since it does not preserve the ranking
            #y_min_raw = np.min(self.archive.y)
            #pred_means2 = np.maximum(pred_means2, y_min_raw)
            
            # rescale so prediction (unnormalized) is never smaller than current smallest y-value
            y_min_orig = np.min(self.archive.y)
            y_min_pred = np.min(pred_means2)
            delta = y_min_orig - y_min_pred

            if delta > 0:
                pred_means2 = pred_means2 + delta
            # else: no change

            # update archive size
            self.archive_size = len(self.archive.y)

            # CMA-ES update based on predicted means with fitness-replaced point(s)
            self.cma.tell(x_pop, pred_means2)
            
            if self.recorder != None:
                    self.recorder.record(self.cma, objective_fn, len(y_orig))

        if self.archive != None:
            opt_argmin = self.archive.x[np.argmin(self.archive.y, axis=0), :].flatten()
        else:
            raise ValueError("Archive doesn't exist!")
        return opt_argmin

def dts_ei_minimize(f, x0):
    # simple scipy-like wrapper function for minimizing an objective
    x0 = np.atleast_1d(x0)
    dim = x0.size
    budget = 150*dim # set max iteration budget

    optimizer = DTS_CMA_ES(x0, kernel=MaternKernel(), population_size=8+(np.floor(6*np.log(dim))))
    return optimizer.optimize(f, budget=budget)
