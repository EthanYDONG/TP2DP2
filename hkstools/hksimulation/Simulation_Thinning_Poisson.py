import numpy as np



def Simulation_Thinning_Poisson(mu, t_start, t_end):
    """
    Implement thinning method to simulate homogeneous Poisson processes

    """

    t = t_start
    history = []
    
    mt = np.sum(mu)

    while t < t_end:
        s = np.random.exponential(1/mt)
        t = t + s
        u = np.random.uniform(0, mt)
        sum_is = 0

        for d in range(len(mu)):
            sum_is = sum_is + mu[d]
            if sum_is >= u:
                break
        index = d
        history.append([t, index])

    history = np.array(history).T
    index = np.where(history[0, :] < t_end)
    history = history[:, index]
    

    return history