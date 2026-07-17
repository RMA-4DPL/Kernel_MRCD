import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from KMRCD import LinKernel, RbfKernel, AutoRbfKernel

def custom_rbf_kernel(x1, x2, gamma=1.0):
    """
    Compute the RBF kernel between two vectors x1 and x2.
    
    Parameters:
    - x1, x2: Input vectors.
    - gamma: Kernel parameter.
    
    Returns:
    - Kernel value (scalar).
    """
    squared_distance = np.sum((x1 - x2) ** 2)
    return np.exp(-gamma * squared_distance)

def custom_linear_kernel(x1, x2):
    return x1 @ x2

def get_kernel(kernel_name):
    kernels = {'linear': LinKernel(),
               'rbf': RbfKernel()}
    
    if kernel_name in kernels:
        return kernels[kernel_name]
    else:
        print(f"{kernel_name} is not currently implemented.")