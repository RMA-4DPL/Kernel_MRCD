import numpy as np


def subsample_data_random(X, n_samples=1000, random_seed=4):
    np.random.seed(random_seed)
    H, W, B = X.shape
    X_t = X.reshape((-1, B))
    random_inds = np.random.choice(np.arange(len(X_t)), size=n_samples, replace=False)
    subsampled_data = X_t[random_inds]

    return subsampled_data

def get_subsampler(name='random'):
    samplers = {'random': subsample_data_random}

    if name in samplers:
        return samplers[name]
    else:
        print(f"{name} sampler is not implemented.")