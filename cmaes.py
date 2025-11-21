# cmaes.py
# By Daan

# Credit: this code is largely based on Nikolaus Hansen's pycma code (cma.purecma.CMAES)
# https://github.com/CMA-ES/pycma

import numpy as np
from scipy.special import gamma

class CMA_ES_Params:
    # CMA-ES parameter container class
    def __init__(self, dim, population_size=None, lr=1.0):
        self.dim = int(dim)
        self.lam = int(population_size) if population_size is not None else int(4 + np.floor(3 * np.log(self.dim)))
        self.mu = self.lam // 2
        self.lr = float(lr)

        # raw log-weights (w_primes)
        w_primes = np.array([np.log((self.lam + 1) / 2.0) - np.log(i+1) if i < self.mu else 0.0
                             for i in range(self.lam)])
        self.ws = w_primes / np.sum(w_primes) # normalized weights (length lambda)
        self.mu_eff = 1.0 / np.sum(self.ws**2)

        # learning rates following common CMA-ES heuristics
        self.c_sigma = self.lr * (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 5.0)
        self.d_sigma = 2.0 * self.mu_eff / self.lam + 0.3 + self.c_sigma
        self.c_C = self.lr * (4.0 + self.mu_eff / self.dim) / (self.dim + 4.0 + 2.0 * self.mu_eff / self.dim)
        self.c_1 = self.lr * 2.0 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1,
                        self.lr * 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.dim + 2.0)**2 + self.mu_eff))
    def update_params(self):
        # raw log-weights (w_primes)
        w_primes = np.array([np.log((self.lam + 1) / 2.0) - np.log(i+1) if i < self.mu else 0.0
                             for i in range(self.lam)])
        self.ws = w_primes / np.sum(w_primes)  # normalized weights (length lambda)
        self.mu_eff = 1.0 / np.sum(self.ws**2)

        # learning rates following common CMA-ES heuristics
        self.c_sigma = self.lr * (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 5.0)
        self.d_sigma = 2.0 * self.mu_eff / self.lam + 0.3 + self.c_sigma
        self.c_C = self.lr * (4.0 + self.mu_eff / self.dim) / (self.dim + 4.0 + 2.0 * self.mu_eff / self.dim)
        self.c_1 = self.lr * 2.0 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1,
                        self.lr * 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.dim + 2.0)**2 + self.mu_eff))

    def chi_mean(self):
        # E||N(0,I_d)|| = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)
        d = self.dim
        return np.sqrt(2.0) * gamma((d + 1.0) / 2.0) / gamma(d / 2.0)

class CMA_ES:
    # CMA-ES core class responsible for querying and updating the CMA-ES
    def __init__(self, x0, population_size=None, lr=1.0):
        dim = len(x0)
        self.params = CMA_ES_Params(dim, population_size, lr)
        self.mean = np.asarray(x0).copy() # m
        self.cov = np.eye(dim) # C
        self.step_size = 1.0
        self.step_size_path = np.zeros(dim) #p_s
        self.cov_path = np.zeros(dim) # p_c
        self._rng = np.random.default_rng()

        # store eigendecomposition: C = B D^2 B^T
        self.B = np.eye(dim) # matrix of column eigenvectors (of C)
        self.eigvals = np.ones(dim) # row vector of eigenvalues of C
        
        # keep track of number of steps
        self.steps = 0
        
    def ask_and_tell(self, f):
        samples = self.ask()
        ys = [f(sample) for sample in samples]
        self.tell(samples,ys)
        return samples, ys
    
    def to_global_coords(self, zs):
        # transform points z from N(0, I) to points x from N(m,C)
        scaling = self.step_size * np.sqrt(self.eigvals)
        if zs.ndim == 1:
            return self.mean + (zs * scaling) @ self.B.T
        else:
            return self.mean.reshape(1,-1) + (zs * scaling.reshape(1,-1)) @ self.B.T

    def global_to_y_coords(self, xs):
        # transform points x from N(m, C) to points y from N(0, C)
        if xs.ndim == 1:
            return (xs - self.mean) / self.step_size
        else:
            return (xs - self.mean.reshape(1,-1)) / self.step_size
    
    def to_local_coords(self, xs):
        # transform points x from N(m, C) to points z from N(0, I)
        ys = self.global_to_y_coords(xs)
        inv_sqrt = 1.0 / np.sqrt(self.eigvals)
        if ys.ndim == 1:
            return (self.B.T @ ys) * inv_sqrt
        else:
            return (ys @ self.B) * inv_sqrt.reshape(1,-1)

    def get_rank_weights(self, fitness):
        rank_indices = np.argsort(np.argsort(fitness, axis=1), axis=1)
        return self.params.ws[rank_indices.flatten()].reshape(fitness.shape)

    def ask(self):
        lam = self.params.lam
        # zs sampled from N(0, I)
        zs = self._rng.standard_normal(size=(lam, self.params.dim))
        return self.to_global_coords(zs)

    def tell(self, xs, fitness):
        fitness = np.asarray(fitness)
        if len(fitness.shape)==1:
            fitness = fitness[None,:]
        samples_ws = self.get_rank_weights(fitness)
        mean_ws = np.mean(samples_ws, axis=0)
        self.update(xs, mean_ws)

    def update(self, xs, weights):
        p = self.params
        p.update_params() # update ES parameters according to heuristics
        dim = self.params.dim

        # we updated; count one step up
        xs = np.asarray(xs)
        self.steps += 1

        # update mean of the distribution
        new_mean = xs.T @ weights 
        
        # y_n sampled from N(0, C)
        y = self.global_to_y_coords(new_mean)
        
        # z_n sampled from N(0, I)
        z = self.to_local_coords(new_mean)

        # normalization factor for sigma evolution path
        psn = np.sqrt(p.c_sigma * (2 - p.c_sigma) * p.mu_eff)
        self.step_size_path = (1 - p.c_sigma) * self.step_size_path + psn * z # update evolution path for step size sigma

        # normalization factor for C evolution path
        pcn = np.sqrt(p.c_C * (2 - p.c_C) * p.mu_eff)

        # turn off rank-one accumulation when sigma increases quickly
        gamma = 1/(1-(1-p.c_sigma)**(2*self.steps)) # account for initial value of ps
        
        # gamma*||ps||^2 / N is 1 in expectation 
        h_sigma = 0
        if gamma*(np.sum(self.step_size_path**2) / dim) < 2 + 4./(dim+1):
            h_sigma = 1
        self.cov_path = (1 - p.c_C) * self.cov_path + pcn * h_sigma * y

        # adapt covariance matrix consisting of the 3 terms in the update rule
        c1a = p.c_1* (1 - (1-h_sigma**2) * p.c_C * (2-p.c_C)) # minor adjustment to c_1 for the variance loss from hsig
        self.cov *= 1 - c1a - p.c_mu
        for k, wk in enumerate(weights): # rank-mu update
            yk = self.global_to_y_coords(xs[k])
            self.cov += wk * p.c_mu*np.outer(yk,yk)
        self.cov += p.c_1 * np.outer(self.cov_path, self.cov_path) # rank-1 update

        # update the step size (copied from Oswin's code)
        self.step_size *= np.exp(min(1, (p.c_sigma/p.d_sigma) * (np.sum(self.step_size_path**2) / dim - 1) / 2))
        # Question: do we not need to compare the sigma norm to E[chi(dim)]?
        # does this have to with the fact that internally, the CMA-ES is in a N(0, I) space?

        self.mean = (1-p.lr) * self.mean + p.lr * new_mean

        # update eigendecomposition
        self.eigvals, self.B = np.linalg.eigh(self.cov)
        self.eigvals = np.maximum(self.eigvals, 1e-20) # clamp eigenvalues to small positive epsilon

def cma_minimize(f, x0):
    # simple scipy-like wrapper function for minimizing an objective
    x0 = np.atleast_1d(x0)
    dim = x0.size
    budget = 150*dim # set max iteration budget

    cma = CMA_ES(x0)
    for n in range(budget):
        cma.ask_and_tell(f)
    return cma.mean