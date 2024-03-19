from RMTPP import RMTPP
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import logging

class MCMC_RMTPP_Mixture(nn.Module):
    def __init__(self, h_dim, type_dim, cluster_num, n_samples, prior_concentration, device, loader):
        super(MCMC_RMTPP_Mixture, self).__init__()
        
        self.h_dim = h_dim
        self.type_dim = type_dim
        self.cluster_num = cluster_num
        self.device = device
        self.n_samples = n_samples
        self.prior_concentration = nn.Parameter(prior_concentration * torch.ones(1), requires_grad = False)
        self.post_concentration = nn.Parameter(prior_concentration +  (1. * n_samples / cluster_num)  * torch.ones(cluster_num), requires_grad = False)
        self.Seqs = loader
        self.true_label = deepcopy(self.Seqs.cluster_labels)
        self.mix_component = nn.ModuleList([RMTPP(h_dim, type_dim, device)])
        self.temp_k = 1
        
    def forward(self, seq_times, seq_types, pad_masks):
        
        LL = []
        NCL = []
        NVL = []
        for _, l in enumerate(self.mix_component):
            n_log_like, num_correct, num_valid = l(seq_times, seq_types, pad_masks)
            LL.append(-n_log_like)
            NCL.append(num_correct)
            NVL.append(num_valid)
        LL = torch.stack(LL, dim = -1)
        NCL = torch.stack(NCL, dim = -1)
        NVL = torch.stack(NVL, dim = -1)
        return LL, NCL, NVL
    def add_component(self, model):
    
        self.mix_component.append(deepcopy(model.mix_component[0]))
        
    def Loglike(self,cluster_idx,mu_prop = None,A_prop = None,index_prop = None):     
        if mu_prop is  None:
            modelest = self.mix_component[cluster_idx]
        else:
            modelest = self.mix_component[cluster_idx]
            
            modelest.encode_emb.time_embedding.Wt.bias = torch.nn.Parameter(torch.from_numpy(mu_prop).to(torch.float32))
        indexest = self.label
        label_k_indices = np.where(indexest == cluster_idx)[1]
        
        label_k_seqs_times = [self.Seqs.seq_times[idx] for idx in label_k_indices]
        label_k_seqs_types = [self.Seqs.seq_types[idx] for idx in label_k_indices]
        label_k_seqs_pad_masks = [self.Seqs.pad_masks[idx] for idx in label_k_indices]
                
        tensor_times = torch.stack(label_k_seqs_times)
        tensor_types = torch.stack(label_k_seqs_types)
        tensor_pad_masks = torch.stack(label_k_seqs_pad_masks)
    
        loglikelihood,_,_ = modelest(tensor_times,tensor_types,tensor_pad_masks)
        loglike_for_allseqinthiscluster = sum(-loglikelihood)
        return loglike_for_allseqinthiscluster

    def loglike_one_a(self,seq_one_times,seq_one_types,seq_one_pad_masks,cluster_idx):
        loglikelihood,_,_ = self.mix_component[cluster_idx](seq_one_times,seq_one_types,seq_one_pad_masks)
        return -loglikelihood
    def loglike_one_na(self,seq_one_times,seq_one_types,seq_one_pad_masks,cluster_idx):  
        loglikelihood,_,_ = self.modelest_na_list[cluster_idx](seq_one_times,seq_one_types,seq_one_pad_masks)
        return -loglikelihood

