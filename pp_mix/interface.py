import logging
import joblib
import os
import sys
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from itertools import combinations, product
from scipy.stats import multivariate_normal, norm
from .params_helper import make_params
from .src import python_exports as pp_mix_py


class ConditionalMCMC_gauss(object):
    def __init__(self, pp_params=None, prec_params=None,
                 jump_params=None, init_n_clus=2):
        
        self.params = make_params(pp_params,prec_params,jump_params)
        self.params.init_n_clus = init_n_clus

    def run(self, nburn, niter, thin, data, log_every=20, bernoulli=False):
        
        if data.ndim == 1:
            self.dim = 1
        else:
            self.dim = data.shape[1]
        
        
        
        out = pp_mix_py._run_pp_mix(
            nburn, niter, thin, data, self.params, bernoulli, log_every)
        
        self.chains = out
    
    def sample_predictive(self):
        if self.dim == 1:
            out = pp_mix_py._sample_predictive_univ(self.chains)
        else:
            out = pp_mix_py._sample_predictive_multi(self.chain,self.dim)
        return out 
    


class ConditionalMCMC_hks(object):
    def __init__(self, pp_params=None, prec_params=None,
                 jump_params=None, init_n_clus=2):
        
        self.params = make_params(pp_params,prec_params,jump_params)
        self.params.init_n_clus = init_n_clus

    def run(self, nburn, niter, thin, data, hakes_model, args,log_every=1,bernoulli=False):
        
        N = len(data)
        D = np.zeros(N)
        for i in range(N):
            D[i] = np.max(data[i]['Mark'])
        D = int(np.max(D))+1
        self.dim = D
        
        
        
        out = pp_mix_py._run_pp_mix(nburn, niter, thin, data, self.params, hakes_model, args,bernoulli, log_every)
        
        self.chains = out
    
    def sample_predictive(self):
        if self.dim == 1:
            out = pp_mix_py._sample_predictive_univ(self.chains)
        else:
            out = pp_mix_py._sample_predictive_multi(self.chain,self.dim)
        return out 



class ConditionalMCMC_nns(object):
    def __init__(self, pp_params=None, jump_params=None, init_n_clus=2):
        
        self.params = make_params(pp_params,jump_params)
        self.params.init_n_clus = init_n_clus

    def run(self, nburn, niter, thin, data, model, args,log_every=1,bernoulli=False):
        
        N = len(data.seq_times)
        D = model.mix_component[0].encode_emb.time_embedding.Wt.bias.shape[0]
        self.dim = D

        out = pp_mix_py._run_pp_mix(nburn, niter, thin, data, self.params, model, args,bernoulli,hawkes = False,neural = True, log_every = 1)
        
        self.chains = out
    
    def sample_predictive(self):
        if self.dim == 1:
            out = pp_mix_py._sample_predictive_univ(self.chains)
        else:
            out = pp_mix_py._sample_predictive_multi(self.chain,self.dim)
        return out 