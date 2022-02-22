import numpy as np
from sklearn.gaussian_process import kernels


def select_kernel(X_1D, kernel = 'Gaussian'):
    if kernel == 'Gaussian':
        K = kernel_gaussian(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'Delta':
        K = kernel_delta_norm(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
                
    elif kernel == 'White':
        K = kernel_white(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'RationalQuadratic':
        K = kernel_rational_quadratic(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'Matern32':
        K = kernel_matern(X_1D.reshape(1,-1), X_1D.reshape(1,-1), nu = 1.5)
    elif kernel == 'Matern52':
        K = kernel_matern(X_1D.reshape(1,-1), X_1D.reshape(1,-1), nu = 2.5)
    elif kernel == 'ExpSineSquared':
        K = kernel_exp_sine_squared(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'DotProduct':
        K = kernel_dot_product(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'Constant':
        K = kernel_constant(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'Laplacian':
        K = kernel_exponential(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'Periodic':
        K = kernel_periodic_exponential(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'Periodic':
        K = kernel_periodic_exponential(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
    elif kernel == 'RandomFourier':
        K = kernel_Random_Fourier(X_1D.reshape(1,-1), X_1D.reshape(1,-1))
                
    else:
        raise ValueError('Kernel Error')
        
    return K
        
        

# Delta kernel with norm
def kernel_delta_norm(X_in_1, X_in_2):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    K = np.zeros((n_1, n_2))
    u_list = np.unique(X_in_1)
    for ind in u_list:
        c_1 = np.sqrt(np.sum(X_in_1 == ind))
        c_2 = np.sqrt(np.sum(X_in_2 == ind))
        ind_1 = np.where(X_in_1 == ind)[1]
        ind_2 = np.where(X_in_2 == ind)[1]
        K[np.ix_(ind_1, ind_2)] = 1 / c_1 / c_2
    return K


# Delta kernel
def kernel_delta(X_in_1, X_in_2):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    K = np.zeros((n_1, n_2))
    u_list = np.unique(X_in_1)
    for ind in u_list:
        ind_1 = np.where(X_in_1 == ind)[1]
        ind_2 = np.where(X_in_2 == ind)[1]
        K[np.ix_(ind_1, ind_2)] = 1
    return K


# Gaussian kernel
def kernel_gaussian(X_in_1, X_in_2, sigma = 1.0):
    return kernels.RBF(length_scale = sigma).__call__(X_in_1.T, X_in_2.T)


# White kernel
def kernel_white(X_in_1, X_in_2, noise_level = 1.0):
    return kernels.WhiteKernel(noise_level = noise_level).__call__(X_in_1.T, X_in_2.T)


# Rational Quadratic kernel
def kernel_rational_quadratic(X_in_1, X_in_2, length_scale = 1.0, alpha = 1.0):
    return kernels.RationalQuadratic(length_scale = length_scale, alpha = alpha).__call__(X_in_1.T, X_in_2.T)


# Matern kernel
def kernel_matern(X_in_1, X_in_2, length_scale = 1.0, nu = 1.5):
    return kernels.Matern(length_scale = length_scale, nu = nu).__call__(X_in_1.T, X_in_2.T)


# Exp Sine Squared kernel
def kernel_exp_sine_squared(X_in_1, X_in_2, length_scale = 1.0, periodicity = 1.0):
    return kernels.ExpSineSquared(length_scale = length_scale, periodicity = periodicity).__call__(X_in_1.T, X_in_2.T)


# Dot Product kernel
def kernel_dot_product(X_in_1, X_in_2, sigma_0 = 1.0):
    return kernels.DotProduct(sigma_0 = sigma_0).__call__(X_in_1.T, X_in_2.T)


# Constant kernel
def kernel_constant(X_in_1, X_in_2, constant_value=1.0):
    return kernels.ConstantKernel(constant_value = constant_value).__call__(X_in_1.T, X_in_2.T)


# Laplacian kernel
def kernel_exponential(X_in_1, X_in_2, theta = 1.0):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    dist = np.abs(X_in_1.T - X_in_2)
    K = np.exp(-dist / theta)
    return K
    

# Periodic Exponential kernel
def kernel_periodic_exponential(X_in_1, X_in_2, theta_1 = 1.0, theta_2 = 1.0):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    dist = np.abs(X_in_1.T - X_in_2)
    K = np.exp(theta_1 * np.cos(dist / theta_2))
    return K


# Random Fourier kernel
def kernel_Random_Fourier(X_in_1, X_in_2, M = 1000, sigma = 1.0):
    omega = np.random.normal(scale = sigma, size = [M,1])
    omega_X_in_1 = omega * X_in_1
    omega_X_in_2 = omega * X_in_2
    K = (np.cos(omega_X_in_1).T @ np.cos(omega_X_in_2) + np.sin(omega_X_in_1).T @ np.sin(omega_X_in_2)) / M
    return K