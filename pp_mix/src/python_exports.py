#import pybind11
import numpy as np
import pandas as pd
import random
from .factory import make_pp,make_jump,make_prec
from .conditional_mcmc import *
from .proto.proto import EigenVector,EigenMatrix
from datetime import datetime
import logging
import time
import logging
import numpy as np
import sys
from metric import purity
from sklearn.metrics.cluster import adjusted_rand_score

def run_pp_mix_univ(burnin, niter, thin, data, params, log_every):
    ranges = np.vstack([np.min(data, axis=0), np.max(data, axis=0)]) * 2
    print("ranges: \n", ranges)

    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    g = make_prec(params)
    pp_mix.set_ranges(ranges)

    sampler = UnivariateConditionalMCMC(pp_mix, h, g, params)

    sampler.initialize(data)

    for i in range(burnin):
        sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)

    for i in range(niter):
        sampler.run_one()
        if i % thin == 0:
            curr = sampler.get_state_as_proto()
            out.append(curr)
        if (i + 1) % log_every == 0:
            print("Running, iter #", i + 1, " / ", niter)

    return out

def run_pp_mix_multi(burnin, niter, thin, data, params, log_every):
    random_seed = random.random()

    random.seed(42)

    ranges = np.vstack([np.min(data, axis=0), np.max(data, axis=0)]) * 2
    print("ranges: \n", ranges)

    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    g = make_prec(params)
    pp_mix.set_ranges(ranges)

    sampler = MultivariateConditionalMCMC(pp_mix, h, g, params)

    sampler.initialize(data)

    history = {'i': [], 'ma': [], 'mna': [], 'a_means': [], 'na_means': [], 'a_precs': [], 'na_precs': [], 'a_jumps': [], 'na_jumps': [], 'clus_alloc': [], 'u': [], 'beta': [], 'gamma': [], 'r': [],'ppstate':[]}

    for i in range(burnin):
        sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)
    
    for i in range(niter):
        sampler.run_one()
        if i % thin == 0:
            s = ""
            curr = None
            curr = sampler.get_state_as_proto()
            #print(curr.clus_alloc)
            out.append(curr)
        if (i + 1) % log_every == 0:
            print("Running, iter #", i + 1, " / ", niter)
            history['i'].append(i)
            history['ma'].append(curr.ma)
            history['mna'].append(curr.mna)
            history['a_means'].append(curr.a_means)
            history['na_means'].append(curr.na_means)
            history['a_precs'].append(curr.a_precs)
            history['na_precs'].append(curr.na_precs)
            history['a_jumps'].append(curr.a_jumps)
            history['na_jumps'].append(curr.na_jumps)
            history['clus_alloc'].append(curr.clus_alloc)
            history['u'].append(curr.u)
            history['ppstate'].append(curr.pp_state)


    np.save('history1.npy', history)
    return out

def log_history(log, curr,i):
    log.info(f'i: {i}')
    log.info(f'ma: {curr.ma}')
    log.info(f'mna: {curr.mna}')
    log.info(f'a_means: {[cura_means_i.data for cura_means_i in curr.a_means]}')
    log.info(f'na_means: {curr.na_means}')
    log.info(f'a_precs: {[cura_prec_i.data for cura_prec_i in curr.a_precs]}')
    log.info(f'na_precs: {curr.na_precs}')
    log.info(f'a_jumps: {[cura_jump_i.data for cura_jump_i in curr.a_jumps]}')
    log.info(f'na_jumps: {curr.na_jumps}')
    log.info(f'clus_alloc: {curr.clus_alloc}')
    log.info(f'u: {curr.u}')
    log.info(f'pp_state: {curr.pp_state}')

def run_pp_mix_hks(burnin, niter, thin, data, params, hakes_model,args,log_every):
    time_start = time.time()
    random_seed = random.random()

    mark_ranges = {}

    # Get unique marks across all data entries
    all_marks = np.unique(np.concatenate([entry['Mark'] for entry in data]))

    # Iterate through each unique mark
    for mark in all_marks:
        # List to store intensity for each mark across all data entries
        mark_intensity_list = []

        # Calculate intensity for each mark across all data entries
        for entry in data:
            mark_count = np.sum(entry['Mark'] == mark)
            total_time = entry['Time'][-1] - entry['Time'][0]
            intensity = mark_count / total_time
            mark_intensity_list.append(intensity)


        intensity_min = min(mark_intensity_list)
        intensity_max = max(mark_intensity_list)
        range_min = intensity_min / 2
        range_max = intensity_max * 2
        mark_ranges[mark] = {'min': range_min, 'max': range_max}


    ranges = np.vstack([[mark_ranges[mark]['min'], mark_ranges[mark]['max']] for mark in all_marks])


    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    g = make_prec(params)
    pp_mix.set_ranges(ranges.T)
    hakes_model.ranges = ranges
    sampler = hawkesmcmc(pp_mix, h, g, params)
    #datavec = [row_vector for row_vector in data]
    sampler.initialize(data, hakes_model)

    history = {'time':[], 'i': [], 'cluster_loglike':[], 'ma': [], 'mna': [], 'a_means': [], 'na_means': [], 'a_precs': [], 'na_precs': [], 'a_jumps': [], 'na_jumps': [], 'clus_alloc': [], 'u': [], 'beta': [], 'gamma': [], 'r': [],'ppstate':[]}

    for i in range(burnin):
        _ = sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)
    print('niter',niter)
    print('thin',thin)
    print('log_every',log_every)
    for i in range(niter):
        cluster_loglike = sampler.run_one()
        time_spend = time.time() - time_start
        print('iteration',i)

        if (i + 1) % log_every == 0:
            curr = None
            curr = sampler.get_state_as_proto()
            out.append(curr)
            print("Running, iter #", i + 1, " / ", niter)
            history['i'].append(i)
            history['time'].append(time_spend)
            history['cluster_loglike'].append(cluster_loglike)
            history['ma'].append(curr.ma)
            history['mna'].append(curr.mna)
            history['a_means'].append([cura_means_i.data for cura_means_i in curr.a_means])
            history['na_means'].append([curna_means_i.data for curna_means_i in curr.na_means])

            history['clus_alloc'].append(curr.clus_alloc)

        if (i+1)%1000 == 0:
            file_name = 'dpp_N:'+str(pp_mix.N)+' '+'dpp_nu:'+str(pp_mix.nu)+' '+'dpp_rho'+str(pp_mix.rho)+'iter'+str(i)+'delta_mu'+str(args.mudelta)
            file_name = str(datetime.now())+' '+file_name
            print("start making files")
            np.save(file_name+'.npy', history)
    print("finish")
    return out


def run_pp_mix_nns(burnin, niter, thin, data, params, model,args,test_loader,log_every):

    time_start = time.time()

    ranges = np.array([[-2, 2]] * model.mix_component[0].intensity_wb.bias.shape[0], dtype=np.float32)



    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    pp_mix.set_ranges(ranges.T)
    model.ranges = ranges
    sampler = neuralmcmc(pp_mix, h, params)

    sampler.initialize(data, model)


    history = {'time':[], 'i': [], 'cluster_loglike':[], 'ma': [], 'mna': [], 'a_means': [], 'na_means': [], 'a_precs': [], 'na_precs': [], 'a_jumps': [], 'na_jumps': [], 'clus_alloc': [], 'u': [], 'beta': [], 'gamma': [], 'r': [],'ppstate':[]}

    for i in range(burnin):
        _ = sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)
    print('niter',niter)
    print('thin',thin)
    print('log_every',log_every)
    for i in range(niter):
        cluster_loglike = sampler.run_one()

        time_spend = time.time() - time_start
        print('iteration',i)
        if (i + 1) % log_every == 0:
            curr = None
            curr = sampler.get_state_as_proto()
            out.append(curr)
            print("Running, iter #", i + 1, " / ", niter)
            history['i'].append(i)
            history['time'].append(time_spend)
            history['cluster_loglike'].append(cluster_loglike)
            history['ma'].append(curr.ma)
            history['mna'].append(curr.mna)
            history['a_means'].append([cura_means_i.data for cura_means_i in curr.a_means])
            history['na_means'].append([curna_means_i.data for curna_means_i in curr.na_means])

            history['clus_alloc'].append(curr.clus_alloc)

            logging.info(f'i: {i}')
            logging.info(f'ma: {curr.ma}')
            logging.info(f'mna: {curr.mna}')
            logging.info(f'a_means: {[cura_means_i.data for cura_means_i in curr.a_means]}')
            logging.info(f'na_means: {[curna_means_i.data for curna_means_i in curr.na_means]}')
            logging.info(f'a_jumps: {[cura_jump_i.data for cura_jump_i in curr.a_jumps]}')
            logging.info(f'na_jumps: {curr.na_jumps}')
            for j,component in enumerate(model.mix_component):
                logging.info(f'component:{j}')
                for name,param in model.mix_component[j].named_parameters():
                    if name in model.para_name:
                        logging.info(f'para_name:{name}')
                        logging.info(f'para_value:{param.data}')
                        
            logging.info(f'clus_alloc: {curr.clus_alloc}')
            logging.info(f'purity:{purity(np.array(curr.clus_alloc[0]),np.array(model.true_label))}')
            logging.info(f'ari:{adjusted_rand_score(np.array(curr.clus_alloc[0]),np.array(model.true_label))}')

        predict = torch.tensor([])
        if (i+1)%100 == 0:
            for idx,mdl in enumerate(model.mix_component):
                state_dict = mdl.state_dict()
                save_name = str(i)+' '+'model_'+str(idx)+'2.pt'
                torch.save(state_dict, save_name)

            component_LL_values = torch.tensor([])
            with torch.no_grad():
                # test_loader.shuffle()
                while True:
                    seq_times, seq_types, pad_masks, _, end, cls_labels = test_loader.next_batch(batch_size = args.batch_size)
                    component_LL, component_NCL, component_NVL = model(seq_times.to(args.device), seq_types.to(args.device), pad_masks.to(args.device))
                    
                    max_col_indices = torch.argmax(component_LL, dim=1)
                    predict = torch.cat([predict,max_col_indices], dim=0)
          
                    component_LL_values_batch = component_LL[torch.arange(component_LL.shape[0]), max_col_indices]
                    component_LL_values_batch = component_LL_values_batch.detach().cpu()
                    
                    component_LL_values = torch.cat([component_LL_values,component_LL_values_batch],dim=0)
                    # total_valid_num = total_valid_num + torch.sum(component_NVL_values)
                    if end: break
            new_row = {
            'alloc':predict,
            'component_LL_values':component_LL_values.mean()
            }
            test_loader.cursor = 0
            
            results = pd.DataFrame(columns=['component_LL_values'])
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
            results.to_excel('./r'+str(i)+'.xlsx')

    print("finish")
    return out



def _run_pp_mix(burnin, niter, thin, data, params, hakes_model, args, test_loader, bernoulli=False, hawkes = False,neural = True, log_every = 10):

    if (bernoulli):
        return run_pp_mix_bernoulli(burnin, niter, thin, data, params, log_every)
    elif (hawkes):
        return run_pp_mix_hks(burnin, niter, thin, data, params, hakes_model,args,log_every)
    elif (neural):
        return run_pp_mix_nns(burnin, niter, thin, data, params, hakes_model,args,test_loader,log_every)
    elif data.shape[0] == 1 or data.shape[1] == 1:
        return run_pp_mix_univ(burnin, niter, thin, data, params, log_every)
    else:
        return run_pp_mix_multi(burnin, niter, thin, data, params, log_every)


def _sample_predictive_univ(chain):

    niter = len(chain)

    out = np.zeros(niter)

    for i in range(niter):
        state = out[i]

        a_means = np.array(state.a_means.data)
        a_precs = np.array(state.a_precs)
        na_means = np.array(state.na_means)
        na_precs = np.array(state.na_precs)
        a_jumps = np.array(state.a_jumps)
        na_jumps = np.array(state.na_jumps)

        probas = np.concatenate([a_jumps, na_jumps])
        probas /= probas.sum()

        k = np.random.choice(np.arange(len(probas)), p=probas)

        if k < len(state.ma):
            mu = a_means[k]
            sig = 1.0 / np.sqrt(a_precs[k])
        else:
            mu = na_means[k - state.ma]
            sig = 1.0 / np.sqrt(na_precs[k - state.ma])

        out[i] = np.random.normal(mu, sig)
    return out

def _sample_predictive_multi(chain, dim):

    niter = len(chain)
    out = np.zeros((niter, dim))
    for i in range(niter):

        state = chain[i]

        a_jumps = np.array(state.a_jumps)
        na_jumps = np.array(state.na_jumps)
        probas = np.concatenate([a_jumps, na_jumps])
        probas /= probas.sum()

        k = np.random.choice(np.arange(len(probas)), p=probas)

        if k < len(state.a_means):
            mu = state.a_means[k]
            prec = state.a_precs[k]
        else:
            mu = state.na_means[k - len(state.a_means)]
            prec = state.na_precs()[k - len(state.a_means)]
        
        out[i] = np.random.multivariate_normal(mu, prec).T
    return out


def run_pp_mix_bernoulli(burnin, niter, thin, data, params, log_every):
    ranges = np.zeros((2, data.shape[1])) 

    ranges[0, :] = 0.0
    ranges[1, :] = 1.0

    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    g = make_prec(params)
    pp_mix.set_ranges(ranges)

    sampler = BernoulliConditionalMCMC(pp_mix, h, g, params)
    datavec = [row_vector for row_vector in data]
    sampler.initialize(datavec)

    for i in range(burnin):
        sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)
    
    for i in range(niter):
        print("start running***\n")
        sampler.run_one()
        if i % thin == 0:
            curr = sampler.get_state_as_proto()
            print(curr.clus_alloc)
            print(i)
            out.append(curr)
        if (i + 1) % log_every == 0:
            print("Running, iter #", i + 1, " / ", niter)
    return out
