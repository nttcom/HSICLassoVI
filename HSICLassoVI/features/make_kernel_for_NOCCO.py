import numpy as np
from joblib import Parallel, delayed

from .kernels import *


def make_kernel_for_NOCCO(X, Y, y_kernel, x_kernel='Gaussian', n_jobs=-1, discarded=0, B=0, M=1, eps=0.001):

    d, n = X.shape
    dy = Y.shape[0]

    L = compute_kernel_for_NOCCO(Y, y_kernel, B, M, eps, discarded)
    L = np.reshape(L,(n * B * M,1))

    result = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel_for_NOCCO)(
        np.reshape(X[k,:],(1,n)), x_kernel, k, B, M, eps, n, discarded) for k in range(d)])

    result = dict(result)

    K = np.array([result[k] for k in range(d)]).T
    KtL = np.dot(K.T, L)

    return K, KtL, L






def compute_kernel_for_NOCCO(x, kernel, B = 0, M = 1, eps = 0.001, discarded = 0):

    d,n = x.shape

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32)

    if kernel in ['Gaussian', 'RationalQuadratic', 'Matern32', 'Matern52', 'ExpSineSquared', 'DotProduct', 'Constant', 'Laplacian', 'Periodic']:
        x = (x / (x.std() + 10e-20)).astype(np.float32)

    st = 0
    ed = B ** 2
    index = np.arange(n)
    for m in range(M):
        np.random.seed(m)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            if kernel == 'Gaussian':
                k = kernel_gaussian(x[:,index[i:j]], x[:,index[i:j]], np.sqrt(d))
            elif kernel == 'Delta':
                k = kernel_delta_norm(x[:,index[i:j]], x[:, index[i:j]])
                
            elif kernel == 'White':
                k = kernel_white(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'RationalQuadratic':
                k = kernel_rational_quadratic(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'Matern32':
                k = kernel_matern(x[:,index[i:j]], x[:, index[i:j]], nu = 1.5)
            elif kernel == 'Matern52':
                k = kernel_matern(x[:,index[i:j]], x[:, index[i:j]], nu = 2.5)
            elif kernel == 'ExpSineSquared':
                k = kernel_exp_sine_squared(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'DotProduct':
                k = kernel_dot_product(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'Constant':
                k = kernel_constant(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'Laplacian':
                k = kernel_exponential(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'Periodic':
                k = kernel_periodic_exponential(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'Periodic':
                k = kernel_periodic_exponential(x[:,index[i:j]], x[:, index[i:j]])
            elif kernel == 'RandomFourier':
                k = kernel_Random_Fourier(x[:,index[i:j]], x[:, index[i:j]])
                
            else:
                raise ValueError('Kernel Error')

            k = np.dot(np.dot(H, k), H)
            k = k @ np.linalg.inv(k + eps*B*np.eye(B))

            k = k / (np.linalg.norm(k, 'fro') + 10e-10)
            K[st:ed] = k.flatten()
            st += B ** 2
            ed += B ** 2

    return K






def parallel_compute_kernel_for_NOCCO(x, kernel, feature_idx, B, M, eps, n, discarded):
    
    return (feature_idx, compute_kernel_for_NOCCO(x, kernel, B, M, eps, discarded))