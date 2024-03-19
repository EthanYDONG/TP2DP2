import argparse
import os
from cluster_loader import *
from RMTPP_dpp import MCMC_RMTPP_Mixture
from RMTPP_dpp_train import train
from metric import purity,calculate_purity
from sklearn.metrics.cluster import adjusted_rand_score
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm
from pp_mix.src.proto.proto import *
from pp_mix.src.proto.Params import *
from pp_mix.params_helper import *
import logging
import os
from datetime import datetime


log_filename = f"RMTPP-DPP-Kgt2.log"
logging.basicConfig(filename=log_filename, format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('-------------RMTPP START------------------')

def main():
        parser = argparse.ArgumentParser(description='RMTPP DPP for event sequence clustering')
        parser.add_argument('--K_GT', type=int, default=2, help='ground truth number of clusters')    
        parser.add_argument('--type_dim', type=int, default=1, help='Dimension of event')
        parser.add_argument('--assume_K', type=int, default=2, help='Initial cluster number')
        parser.add_argument('--h_dim', type=int, default=4, help='Dimension of hidden states')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
        parser.add_argument('--pre_epoch', type=int, default=5, help='Pretrain epoch')
        parser.add_argument('--train_epoch', type=int, default=2, help='Train epoch')
        parser.add_argument('--m_step', type=int, default=5, help='SGD stpes for maximization')
        parser.add_argument('--device', type=str, default='cpu', help='Device')
        parser.add_argument('--dpp_N', type=int, default=5)
        parser.add_argument('--dpp_nu', type=float, default=20)
        parser.add_argument('--dpp_rho', type=float, default=3)
        parser.add_argument('--nburn', type=int, default=0)  
        parser.add_argument('--niter', type=int, default=10000)
        parser.add_argument('--thin', type=int, default=5)
        parser.add_argument('--gamma_alpha', type=float, default=1.0)
        parser.add_argument('--gamma_beta', type=float, default=1.0)
        parser.add_argument('--init_sigma',type=float,default=0.3)
        args = parser.parse_args()

        path = 'data_K2.pkl'
        f = open(path, 'rb')
        data = pickle.load(f)
        
        tmax = data['t_max']
        dataset = data_preprocessor(path, args.type_dim, tmax, train_prop = 1, val_prop = 0, shuffle_flag = True)
        times, types, lengths, pad_masks, labels = dataset.get_data('train')
        loader = data_loader(times, types, lengths, pad_masks, labels)
        max_length = loader.seq_lengths.max() + 1
        train_size = loader.indexs.shape[0]
        
        model =  MCMC_RMTPP_Mixture(args.h_dim, args.type_dim, \
                        args.assume_K, train_size, 1. / args.assume_K, args.device,loader).to(args.device)
        def print_model_parameters(model):
                for name, param in model.named_parameters():
                        print(f"{name}: {param.shape}")
        print_model_parameters(model)
        
        
        logging.info(f'model true label:  {model.true_label}')
        train_optimizer = torch.optim.Adam(model.parameters(), args.lr)

        dpp_params = DPPParams()
        dpp_params.nu = args.dpp_nu
        dpp_params.rho = args.dpp_rho
        dpp_params.N = args.dpp_N
        gamma_jump_params = GammaParams()
        gamma_jump_params.alpha = args.gamma_alpha
        gamma_jump_params.beta = args.gamma_beta

        number_record = train(model, train_optimizer, loader, args.batch_size, args.pre_epoch, args.train_epoch,\
                args.m_step, args.device, dpp_params, gamma_jump_params,args)


if __name__ == '__main__':
    main()