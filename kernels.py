from KMRCD import LinKernel, RbfKernel, AutoRbfKernel  # AutoRbfKernel re-exported for AD_models_GPU.py

def get_kernel(kernel_name):
    kernels = {'linear': LinKernel(),
               'rbf': RbfKernel()}

    if kernel_name in kernels:
        return kernels[kernel_name]
    else:
        print(f"{kernel_name} is not currently implemented.")