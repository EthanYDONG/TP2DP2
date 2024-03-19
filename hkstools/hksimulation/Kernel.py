import numpy as np

def Kernel(dt, para):
    """
    Compute the value of the kernel function at different time points.
    
    Parameters:
    - dt: Time differences between current time and historical events.
    - para: Dictionary containing kernel parameters.
        - para['kernel']: Type of the kernel function ('exp' or 'gauss').
        - para['landmark']: Landmarks for the kernel function.
        - para['w']: Bandwidth parameter for the kernel function.
    
    Returns:
    - g: Computed values of the kernel function.
    """
    dt = np.array(dt).flatten()
    
    landmarks = np.array(para['landmark'])[np.newaxis, :]
    
    dt_tiled = np.tile(dt[:, np.newaxis], (1, len(para['landmark'])))
    
    distance = dt_tiled - landmarks

    if para['kernel'] == 'exp':
        g = para['w'] * np.exp(-para['w'] * distance)
        g[g > 1] = 0

    elif para['kernel'] == 'gauss':
        g = np.exp(-(distance**2) / (2 * para['w']**2)) / (np.sqrt(2 * np.pi) * para['w'])

    else:
        print('Error: please assign a kernel function!')
    return g