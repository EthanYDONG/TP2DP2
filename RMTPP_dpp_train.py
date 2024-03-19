from tqdm import tqdm
import torch
from pp_mix.interface import ConditionalMCMC_nns
import time
import torch.nn as nn
import logging

def sample(loader, pp_params, jump_params, nn_model, args):

    dpp_sampler = ConditionalMCMC_nns(
    pp_params=pp_params, 
    jump_params=jump_params,
    init_n_clus=args.assume_K)
    start = time.time()

    nburn = args.nburn
    niter = args.niter
    logging.info('data groundtruth label: ',loader.cluster_labels)
    dpp_sampler.run(nburn, niter, 5,loader, nn_model, args)
    dpp_times = time.time() - start



def train(model, optimizer, loader, batch_size, pre_epochs, train_epochs, m_steps, device, dpp_params, gamma_jump_params,args):
    para_name = ['intensity_wb.bias',
                 'intensity_v.weight',
                 'intensity_wb.weight']
    
    print('Start pretraining...')

    gap = 10
    for i in tqdm(range(pre_epochs)):
        loader.shuffle()
        batch_iter = 0
        while True:
            seq_times, seq_types, pad_masks, _, end, cls_labels = loader.next_batch(batch_size = batch_size)
            optimizer.zero_grad()
            component_LL, _, _ = model(seq_times.to(device), seq_types.to(device), pad_masks.to(device))
            batch_loss = component_LL.sum()
            batch_loss.backward()
            optimizer.step()
            if end: break
        
    

    
        
   
    model.eval()
    model.to('cpu')
    logging.info(f'intensity_bias:{model.mix_component[0].intensity_wb.bias}')
    
    for _ in range(1,model.cluster_num):
        model.add_component(model)
        model_dict = model.mix_component[_].state_dict()
        for name in para_name:
            param_shape = model_dict[name].shape
            total_elements = torch.flatten(model_dict[name])
            low = total_elements - args.init_sigma
            high = total_elements + args.init_sigma
            mvn = torch.distributions.uniform.Uniform(low, high)
            sampled_value = nn.Parameter(mvn.sample().view(param_shape))
            model_dict[name].data = sampled_value
        model.mix_component[_].load_state_dict(model_dict)
        
    model_dict = model.mix_component[0].state_dict()
    for name in para_name:
        param_shape = model_dict[name].shape
        total_elements = torch.flatten(model_dict[name])
      
        low = total_elements - args.init_sigma
        high = total_elements + args.init_sigma
        mvn = torch.distributions.uniform.Uniform(low, high)
        sampled_value = nn.Parameter(mvn.sample().view(param_shape))
        model_dict[name].data = sampled_value
    model.mix_component[0].load_state_dict(model_dict)
    model.para_name = para_name

    print('Start MCMC training...')
    sample(loader, dpp_params, gamma_jump_params, model, args)
    



