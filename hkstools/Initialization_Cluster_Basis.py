import numpy as np
from scipy.sparse import csr_matrix


def Initialization_Cluster_Basis(Seqs, ClusterNum, baseType=None, bandwidth=None, landmark=None):
    
    N = len(Seqs)
    D = np.zeros(N)

    for i in range(N):
        D[i] = np.max(Seqs[i]['Mark'])
    
    D = int(np.max(D))+1
    
    model = {'K': ClusterNum, 'D': D}

    if baseType is None and bandwidth is None and landmark is None:
        sigma = np.zeros(N)
        Tmax = np.zeros(N)

        for i in range(N):
            sigma[i] = ((4 * np.std(Seqs[i]['Time'])**5) / (3 * len(Seqs[i]['Time'])))**0.2
            Tmax[i] = Seqs[i]['Time'][-1] + np.finfo(float).eps
        Tmax = np.mean(Tmax)

        model['kernel'] = 'gauss' 
        model['w'] = np.mean(sigma) 
        model['landmark'] = model['w'] * np.arange(0, np.ceil(Tmax / model['w'])) 

    elif baseType is not None and bandwidth is None and landmark is None:
        model['kernel'] = baseType
        model['w'] = 1
        model['landmark'] = 0

    elif baseType is not None and bandwidth is not None and landmark is None:
        model['kernel'] = baseType
        model['w'] = bandwidth
        model['landmark'] = 0

    else:
        model['kernel'] = baseType if baseType is not None else 'gauss'
        model['w'] = bandwidth if bandwidth is not None else 1
        model['landmark'] = landmark if landmark is not None else 0

    model['alpha'] = 1
    M = len(model['landmark'])
    model['beta'] = np.ones((D, M, model['K'], D)) / (M * D**2)

    model['b'] = np.ones((D, model['K'])) / D


    label = np.ceil(model['K'] * np.random.rand(1, N)).astype(int)
    model['label']=label
    model['R'] = csr_matrix((np.ones(N), (np.arange(N), label.flatten()-1)), shape=(N, model['K'])).toarray()

    return model