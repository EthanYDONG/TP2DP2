import numpy as np
from scipy.stats import gamma
from scipy.linalg import block_diag
from scipy.stats import bernoulli
from scipy.special import expit
from scipy.stats import multinomial
from scipy.stats import dirichlet
from scipy.stats import multivariate_normal,norm
from .conditional_mcmc_imp import ConditionalMCMC
from .conditional_mcmc_imp import ConditionalMCMC_model,ConditionalMCMC_neural
from .proto import Params as params
from .proto import proto as proto
from .point_process.nrep_pp import NrepPP
import torch
import torch.nn as nn


class MultivariateConditionalMCMC(ConditionalMCMC):
    def __init__(self, pp_mix, h, g, params):
        
        super().__init__( pp_mix, h, g, params)
        self.pp_mix = pp_mix
        self.jump = h
        self.prec = g
        self.params = params
        self.min_proposal_sigma = 0.1
        self.max_proposal_sigma = 2.0

    def initialize_allocated_means(self):
        print('initialize_allocated_means multi')
        init_n_clus = self.params.init_n_clus
        if init_n_clus == 0:
            init_n_clus = 5
        if init_n_clus >= self.ndata:
            self.a_means = np.array([datum.transpose() for datum in self.data])
        else:
            a_means_index = np.arange(self.ndata)
            np.random.shuffle(a_means_index)
            
            self.a_means = np.array([self.data[index].transpose() for index in a_means_index[:init_n_clus]])

    def compute_grad_for_clus(self, clus, mean):
        grad = np.zeros(self.dim)
        for datum in self.data_by_clus[clus]:
            grad += datum - mean
        
        
        return grad
    
    def print_data_by_clus(self, clus):
        for d in self.data_by_clus[clus]:
            print(d.transpose())
    

    def get_state_as_proto(self):
        out = proto.MultivariateMixtureState()
        out.ma = self.a_means.shape[0]
        out.mna = self.na_means.shape[0]
        out.mtot = self.a_means.shape[0] + self.na_means.shape[0]

        for i in range(self.a_means.shape[0]):
            out.a_means.append(proto.EigenVector(self.a_means[i].transpose()))
            out.a_precs.append(proto.EigenMatrix(self.a_precs[i].get_prec()))

        for i in range(self.na_means.shape[0]):
            out.na_means.append(proto.EigenVector(self.na_means[i].transpose()))
            out.na_precs.append(proto.EigenMatrix(self.na_precs[i].get_prec()))

        out.a_jumps.append(proto.EigenVector(self.a_jumps))
        out.na_jumps.append(proto.EigenVector(self.na_jumps))

        out.clus_alloc.append(self.clus_alloc)

        out.u = self.u

        
        if isinstance(self.pp_mix,StraussPP):
            
            pass
            
        elif isinstance(self.pp_mix,NrepPP):
            
            pass
            

        return out
    
    def lpdf_given_clus(self, datum, mean, prec):
        mvn = multivariate_normal(mean=mean, cov=np.linalg.inv(prec))
        return mvn.logpdf(datum)
        

    def lpdf_given_clus_multi(self, datum, mean, prec):
        mean = mean.reshape(mean.shape[1])
        mvn = multivariate_normal(mean=mean, cov=np.linalg.inv(prec))
        return np.sum(mvn.logpdf(datum))
                

    def set_dim(datum):
        return len(datum)
 

    

class UnivariateConditionalMCMC(ConditionalMCMC):
    def __init__(self, pp_mix, h, g, params):
        
        super().__init__( pp_mix, h, g, params)
        self.pp_mix = pp_mix
        self.jump = h
        self.prec = g
        self.params = params
        self.min_proposal_sigma = 0.1
        self.max_proposal_sigma = 1.0

    def initialize_allocated_means(self):
        print('initialize_allocated_means start')
        init_n_clus = 4
        if init_n_clus >= self.ndata:
            self.a_means = np.array([datum for datum in self.data])
        else:
            a_means_index = np.arange(self.ndata)
            np.random.shuffle(a_means_index)
            self.a_means = np.array([self.data[index] for index in a_means_index[:init_n_clus]])
        
        print('initialize_allocated_means',type(self.pp_mix),'---',self.a_means.shape)
    def compute_grad_for_clus(self, clus, mean):
        grad = 0.0
        mean_ = mean[0]
        for datum in self.data_by_clus[clus]:
            grad += (mean_ * (-1) + datum) * self.a_precs[clus]

        out = np.array([grad])
        return out
    
    def print_data_by_clus(self, clus):
        for d in self.data_by_clus[clus]:
            print(d)

    def get_state_as_proto(self):
        out = proto.UnivariateMixtureState()
        out.ma = self.a_means.shape[0]
        out.mna = self.na_means.shape[0]
        out.mtot = self.a_means.shape[0] + self.na_means.shape[0]

        out.a_means = proto.EigenVector(np.reshape(self.a_means, (self.a_means.shape[0], 1)))
        out.na_means = proto.EigenVector(np.reshape(self.na_means, (self.na_means.shape[0], 1)))

        out.a_precs = proto.EigenVector(self.a_precs)
        out.na_precs = proto.EigenVector(self.na_precs)

        out.a_jumps = proto.EigenVector(self.a_jumps)
        out.na_jumps = proto.EigenVector(self.na_jumps)

        out.clus_alloc.append(self.clus_alloc)

        out.u = self.u

        if isinstance(self.pp_mix,StraussPP):
            
            pp_params = self.pp_mix.get_state_as_proto()
            out.pp_state = pp_params
        elif isinstance(self.pp_mix,NrepPP):
            
            pp_params = self.pp_mix.get_state_as_proto()
            out.pp_state = pp_params

        return out
    
    def lpdf_given_clus(self, datum, mean, prec):
        return norm.pdf(datum, mean, prec)
        

    def lpdf_given_clus_multi(self, datum, mean, prec):
        
        
        return np.sum(norm.pdf(datum, mean, prec))        

    def set_dim(datum):
        return len(datum)
    

class hawkesmcmc(ConditionalMCMC_model):
    def __init__(self, pp_mix, h, g, params):
        
        super().__init__( pp_mix, h, g, params)
        self.pp_mix = pp_mix
        self.jump = h
        self.prec = g  
        self.params = params
        self.min_proposal_sigma = 0.1
        self.max_proposal_sigma = 2.0

    

    def initialize_allocated_means(self, ranges):
        print('initialize_allocated_means multi')
        init_n_clus = self.params.init_n_clus
        if init_n_clus == 0:
            init_n_clus = 5
        
        
        
        
        
        
        
        
        samples = np.array([np.random.uniform(low, high/2, init_n_clus) for low, high in ranges])
        
        
        
        
        
        
        
        

        self.a_means = samples.T


    def compute_grad_for_clus(self, clus, mean):
        grad = np.zeros(self.dim)
        for datum in self.data_by_clus[clus]:
            grad += datum - mean
        
        
        return grad
    
    def print_data_by_clus(self, clus):
        for d in self.data_by_clus[clus]:
            print(d.transpose())
    

    def get_state_as_proto(self):
        out = proto.MultivariateMixtureState()
        out.ma = self.a_means.shape[0]
        out.mna = self.na_means.shape[0]
        out.mtot = self.a_means.shape[0] + self.na_means.shape[0]

        for i in range(self.a_means.shape[0]):
            out.a_means.append(proto.EigenVector(self.a_means[i].transpose()))
            out.a_precs.append(proto.EigenMatrix(self.a_precs[i]))

        for i in range(self.na_means.shape[0]):
            out.na_means.append(proto.EigenVector(self.na_means[i].transpose()))
            out.na_precs.append(proto.EigenMatrix(self.na_precs[i]))

        out.a_jumps.append(proto.EigenVector(self.a_jumps))
        out.na_jumps.append(proto.EigenVector(self.na_jumps))

        out.clus_alloc.append(self.clus_alloc)

        out.u = self.u

        
        if isinstance(self.pp_mix,StraussPP):
            
            out.pp_state = self.pp_mix.get_state_as_proto()
            
        elif isinstance(self.pp_mix,NrepPP):
            
            out.pp_state = self.pp_mix.get_state_as_proto()
            

        return out
    
    def lpdf_given_clus(self, datum, mean, prec):
        mvn = multivariate_normal(mean=mean, cov=np.linalg.inv(prec))
        return mvn.logpdf(datum)
        

    
    
    
    

    def lpdf_given_clus_multi(self,cluster_index):
        pass
                

    def set_dim(datum):
        return len(datum)






class neuralmcmc(ConditionalMCMC_neural):
    def __init__(self, pp_mix, h, params):
        
        super().__init__( pp_mix, h, params)
        self.pp_mix = pp_mix
        self.jump = h
        self.params = params
        self.min_proposal_sigma = 0.01
        self.max_proposal_sigma = 0.05



    def initialize_allocated_means(self, ranges):
        print('initialize_allocated_means multi')
        init_n_clus = self.params.init_n_clus
        if init_n_clus == 0:
            init_n_clus = 5
        
        
        samples = np.array([np.random.uniform(low, high, init_n_clus) for low, high in ranges])
        
        self.a_means = samples.T


    def compute_grad_for_clus(self, clus, mean):
        grad = np.zeros(self.dim)
        for datum in self.data_by_clus[clus]:
            grad += datum - mean
        
        
        return grad
    
    def print_data_by_clus(self, clus):
        for d in self.data_by_clus[clus]:
            print(d.transpose())
    

    def get_state_as_proto(self):
        out = proto.MultivariateMixtureState()
        out.ma = self.a_means.shape[0]
        out.mna = self.na_means.shape[0]
        out.mtot = self.a_means.shape[0] + self.na_means.shape[0]

        for i in range(self.a_means.shape[0]):
            out.a_means.append(proto.EigenVector(self.a_means[i].transpose()))

        for i in range(self.na_means.shape[0]):
            out.na_means.append(proto.EigenVector(self.na_means[i].transpose()))
        

        out.a_jumps.append(proto.EigenVector(self.a_jumps))
        out.na_jumps.append(proto.EigenVector(self.na_jumps))

        out.clus_alloc.append(self.clus_alloc)

        out.u = self.u

        
        if isinstance(self.pp_mix,NrepPP):
            
            out.pp_state = self.pp_mix.get_state_as_proto()
            

        return out
    
    def lpdf_given_clus(self, datum, mean, prec):
        mvn = multivariate_normal(mean=mean, cov=np.linalg.inv(prec))
        return mvn.logpdf(datum)
        

    
    
    
    

    def lpdf_given_clus_multi(self,cluster_index):
        pass
                

    def set_dim(datum):
        return len(datum)
















class BernoulliConditionalMCMC():
    def __init__(self, pp_mix, h, g, params):
        super().__init__( pp_mix, h, g, params)
        self.pp_mix = pp_mix
        self.jump = h
        
        
        self.params = params
        self.min_proposal_sigma = 0.025
        self.max_proposal_sigma = 0.3

    def initialize_allocated_means(self):
        init_n_clus = 5
        self.a_means = self.pp_mix.sample_n_points(init_n_clus)
        

    def lpdf_given_clus(self, x, mu, sigma):
        return np.sum(x * np.log(mu) + (1.0 - x) * np.log(1.0 - mu))

    def lpdf_given_clus_multi(self, x, mu, sigma):
        return np.sum([self.lpdf_given_clus(curr, mu, sigma) for curr in x])

    def compute_grad_for_clus(self, clus, mean):
        grad = np.zeros(self.dim)
        
        
        
        
        
        

        return grad
    
    def print_data_by_clus(self, clus):
        for d in self.data_by_clus[clus]:
            print(d.transpose())

    def get_state_as_proto(self, out):
        out = proto.BernoulliMixtureState()
        out.ma = self.a_means.shape[0]
        out.mna = self.na_means.shape[0]
        out.mtot = self.a_means.shape[0] + self.na_means.shape[0]

        for i in range(self.a_means.shape[0]):
            out.a_probs.append(proto.EigenVector(self.a_means[i].transpose()))
        for i in range(self.na_means.shape[0]):
            out.na_probs.append(proto.EigenVector(self.na_means[i].transpose()))

        out.a_jumps = proto.EigenMatrix(self.a_jumps)
        out.na_jumps = proto.EigenMatrix(self.na_jumps)

        out.clus_alloc.append(self.clus_alloc)

        out.u = self.u

        if isinstance(self.pp_mix,StraussPP):
            pp_params = proto.StraussState()
            self.pp_mix.get_state_as_proto(pp_params)
            out.pp_state = pp_params
        elif isinstance(self.pp_mix,NrepPP):
            pp_params = proto.NrepState()
            self.pp_mix.get_state_as_proto(pp_params)
            out.pp_state = pp_params

        return out

