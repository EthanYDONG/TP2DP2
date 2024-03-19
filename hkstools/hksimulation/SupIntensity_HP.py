import numpy as np
from .Kernel import Kernel
def SupIntensity_HP(t, History, para, options):
    
    if not History:
        mt = np.sum(para['mu'])
    else:
        Time = History[0, :]
        index = Time <= t
        Time = Time[index]
        Event = History[1, index]

        MT = np.sum(para['mu']) * np.ones(options['M'])
        for m in range(1, options['M']+1):
            t_current = t + (m - 1) * options['tstep'] / options['M']

            basis = Kernel(t_current - Time, para)
            A = para['A'][Event, :, :]

            for c in range(0, para['A'].shape[2]):
                MT[m-1] = MT[m-1] + np.sum(basis * A[:,:,c])

        mt = np.max(MT)

    mt = mt * (mt > 0)

    return mt

if __name__ == '__main__':
    print('ok')
    SupIntensity_HP(0, 0, 0, 0)
    

