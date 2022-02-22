import numpy as np
from six import string_types
import warnings

from ..features.make_kernel import make_kernel, compute_kernel
from ..features.make_kernel_for_NOCCO import make_kernel_for_NOCCO, compute_kernel_for_NOCCO
from .variational_inference import Updates


class Proposed_HSIC_Lasso(object):
    def __init__(self, lam, tol=1e-5, nu=1.5, numiter=100, objhowoften=1):
        self.input_file = None
        self.X_in = None
        self.Y_in = None
        self.KX = None
        self.KXtKy = None
        self.omega = None
        self.A = None
        self.lam = None
        self.featname = None
        self.lam = lam
        self.tol = tol
        self.nu = nu
        self.numiter = numiter
        self.objhowoften = objhowoften

    def input(self, *args, **_3to2kwargs):
        if 'output_list' in _3to2kwargs: output_list = _3to2kwargs['output_list']; del _3to2kwargs['output_list']
        else: output_list = ['y']

        self._check_args(args)
        if isinstance(args[0], string_types):
            self._input_data_file(args[0], output_list)
        elif isinstance(args[0], np.ndarray):
            if 'featname' in _3to2kwargs:
                featname = _3to2kwargs['featname']; del _3to2kwargs['featname']
            else: featname = ['%d' % x for x in range(1, args[0].shape[1] + 1)]

            if len(args) == 2:
                self._input_data_ndarray(args[0], args[1], featname)
            if len(args) == 3:
                self._input_data_ndarray(args[0], args[1], args[2])
        else:
            pass
        if self.X_in is None or self.Y_in is None:
            raise ValueError("Check your input data")
        self._check_shape()
        return True
    
    
    def classification_multi(self, B=20, M=3, n_jobs=-1, kernels=['Gaussian']):
        self._run_hsic_lasso_multi(B=B, 
                                   M=M,
                                   n_jobs=n_jobs,
                                   kernels=kernels,
                                   y_kernel = 'Delta')
    
    
    def regression_multi(self, B=20, M=3, n_jobs=-1, kernels=['Gaussian']):
        self._run_hsic_lasso_multi(B=B, 
                                   M=M,
                                   n_jobs=n_jobs,
                                   kernels=kernels,
                                   y_kernel = 'Gaussian')
    

    def _run_hsic_lasso_multi(self, B, M, n_jobs, kernels = ['Gaussian'], y_kernel = 'Gaussian'):

        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")
        n = self.X_in.shape[1]
        B = B if B else n
        numblocks = n / B
        discarded = n % B

        print(f'Block HSIC Lasso B = {B}.')

        if discarded:
            msg = f'B {B} must be an exact divisor of the number of samples {n}. Number of blocks {numblocks} will be approximated to {int(numblocks)}.'
            warnings.warn(msg, RuntimeWarning)
            numblocks = int(numblocks)

        M = 1 + bool(numblocks - 1) * (M - 1)
        print(f'M set to {M}.')
        print(f'Using {kernels[0]} kernel for the features and {y_kernel} kernel for the target.')

        
        K = len(kernels)
        X, Xty, Ky = [], [], []
        for kernel in kernels:
            _X, _Xty, _Ky = make_kernel(self.X_in, self.Y_in, y_kernel, kernel, n_jobs=n_jobs, discarded=discarded, B=B, M=M)
            X.append(_X)
            Xty.append(_Xty)
            Ky.append(_Ky)
            
        X, Xty, Ky = np.array(X).transpose(1,2,0), np.array(Xty).transpose(1,2,0), np.array(Ky).transpose(1,2,0)[:,0,:]

        self.KX = X * np.sqrt(1 / (numblocks * M))
        self.KXtKy = Xty * 1 / (numblocks * M)
        self.Ky = Ky * np.sqrt(1 / (numblocks * M))
        
        
        model = Updates(lam = self.lam, tol = self.tol, a = self.nu, numiter = self.numiter, objhowoften = self.objhowoften)
        self.eta, self.what, self.sigma, self.bound = model.fit(y = self.Ky, X = self.KX)
        self.eta_process, self.what_process, self.sigma_process, self.bound_process = model.fhat_process, model.what_process, model.sigmahat_process, model.bound_process
        self.omega = np.mean(self.what, axis=1)[:,None]
        self.A = list(np.argsort(np.abs(self.omega).flatten()))[::-1]

        return True
    
    def get_index(self):
        return self.A
    
    def get_index_score(self):
        return self.omega[self.A, -1]
    
    def get_features(self):
        index = self.get_index()

        return [self.featname[i] for i in index]
    

    def _check_args(self, args):
        if len(args) == 0 or len(args) >= 4:
            raise SyntaxError("Input as input_data(file_name) or \
                input_data(X_in, Y_in)")
        elif len(args) == 1:
            if isinstance(args[0], string_types):
                if len(args[0]) <= 4:
                    raise ValueError("Check your file name")
                else:
                    ext = args[0][-4:]
                    if ext == ".csv" or ext == ".tsv" or ext == ".mat":
                        pass
                    else:
                        raise TypeError("Input file is only .csv, .tsv .mat")
            else:
                raise TypeError("File name is only str")
        elif len(args) == 2:
            if isinstance(args[0], string_types):
                raise TypeError("Check arg type")
            elif isinstance(args[0], list):
                if isinstance(args[1], list):
                    pass
                else:
                    raise TypeError("Check arg type")
            elif isinstance(args[0], np.ndarray):
                if isinstance(args[1], np.ndarray):
                    pass
                else:
                    raise TypeError("Check arg type")
            else:
                raise TypeError("Check arg type")
        elif len(args) == 3:
            if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray) and isinstance(args[2], list):
                pass
            else:
                raise TypeError("Check arg type")

        return True

    def _input_data_file(self, file_name, output_list):
        ext = file_name[-4:]
        if ext == ".csv":
            self.X_in, self.Y_in, self.featname = input_csv_file(
                file_name, output_list=output_list)
        elif ext == ".tsv":
            self.X_in, self.Y_in, self.featname = input_tsv_file(
                file_name, output_list=output_list)
        elif ext == ".mat":
            self.X_in, self.Y_in, self.featname = input_matlab_file(file_name)
        return True

    def _input_data_list(self, X_in, Y_in):
        if isinstance(Y_in[0], list):
            raise ValueError("Check your input data")
        self.X_in = np.array(X_in).T
        self.Y_in = np.array(Y_in).reshape(1, len(Y_in))
        return True

    def _input_data_ndarray(self, X_in, Y_in, featname = None):
        if len(Y_in.shape) == 2:
            raise ValueError("Check your input data")
        self.X_in = X_in.T
        self.Y_in = Y_in.reshape(1, len(Y_in))
        self.featname = featname
        return True

    def _check_shape(self):
        _, x_col_len = self.X_in.shape
        y_row_len, y_col_len = self.Y_in.shape
        if x_col_len != y_col_len:
            raise ValueError(
                "The number of samples in input and output should be same")
        return True

    def _permute_data(self, seed=None):
        np.random.seed(seed)
        n = self.X_in.shape[1]

        perm = np.random.permutation(n)
        self.X_in = self.X_in[:, perm]
        self.Y_in = self.Y_in[:, perm]

        
        
        
        
        
        

        
class Proposed_NOCCO_Lasso(Proposed_HSIC_Lasso):
    def __init__(self, lam, tol=1e-5, nu=1.5, numiter=100, objhowoften=1, eps=0.001):
        super().__init__(lam, tol, nu, numiter, objhowoften)
        self.eps = eps
        
    def _run_hsic_lasso_multi(self, B, M, n_jobs, kernels = ['Gaussian'], y_kernel = 'Gaussian'):

        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")
        n = self.X_in.shape[1]
        B = B if B else n
        numblocks = n / B
        discarded = n % B

        print(f'Block HSIC Lasso B = {B}.')

        if discarded:
            msg = f'B {B} must be an exact divisor of the number of samples {n}. Number of blocks {numblocks} will be approximated to {int(numblocks)}.'
            warnings.warn(msg, RuntimeWarning)
            numblocks = int(numblocks)

        M = 1 + bool(numblocks - 1) * (M - 1)
        print(f'M set to {M}.')
        print(f'Using {kernels[0]} kernel for the features and {y_kernel} kernel for the target.')

        
        K = len(kernels)
        X, Xty, Ky = [], [], []
        for kernel in kernels:
            _X, _Xty, _Ky = make_kernel_for_NOCCO(self.X_in, self.Y_in, y_kernel, kernel, n_jobs=n_jobs, discarded=discarded, B=B, M=M, eps=self.eps)
            X.append(_X)
            Xty.append(_Xty)
            Ky.append(_Ky)
            
        X, Xty, Ky = np.array(X).transpose(1,2,0), np.array(Xty).transpose(1,2,0), np.array(Ky).transpose(1,2,0)[:,0,:]

        self.KX = X * np.sqrt(1 / (numblocks * M))
        self.KXtKy = Xty * 1 / (numblocks * M)
        self.Ky = Ky * np.sqrt(1 / (numblocks * M))
        
        
        model = Updates(lam = self.lam, tol = self.tol)
        self.eta, self.what, self.sigma, self.bound = model.fit(y = self.Ky, X = self.KX)
        self.bound *= -1
        self.eta_process, self.what_process, self.sigma_process, self.bound_process = model.fhat_process, model.what_process, model.sigmahat_process, model.bound_process
        self.omega = np.mean(self.what, axis=1)[:,None]
        self.A = list(np.argsort(np.abs(self.omega).flatten()))[::-1]

        return True
    
    
    
    
    
def input_csv_file(file_name, output_list=['y']):
    return input_txt_file(file_name, output_list, ',')


def input_tsv_file(file_name, output_list=['y']):
    return input_txt_file(file_name, output_list, '\t')


def input_txt_file(file_name, output_list, sep):
    df = pd.read_csv(file_name, sep=sep)

    featname = df.columns.tolist()
    input_index = list(range(0, len(featname)))
    output_index = []

    for output_name in output_list:
        if output_name in featname:
            tmp = featname.index(output_name)
            output_index.append(tmp)
            input_index.remove(tmp)
        else:
            raise ValueError("Output variable, %s, not found" % (output_name))

    for output_name in output_list:
        featname.remove(output_name)

    X_in = df.iloc[:, input_index].values.T

    if len(output_index) == 1:
        Y_in = df.iloc[:, output_index].values.reshape(1, len(df.index))
    else:
        Y_in = df.iloc[:, output_index].values.T

    return X_in, Y_in, featname


def input_matlab_file(file_name):
    data = spio.loadmat(file_name)

    if "X" in data.keys() and "Y" in data.keys():
        X_in = data["X"]
        Y_in = data["Y"]
    elif "X_in" in data.keys() and "Y_in" in data.keys():
        X_in = data["X_in"]
        Y_in = data["Y_in"]
    elif "x" in data.keys() and "y" in data.keys():
        X_in = data["x"]
        Y_in = data["y"]
    elif "x_in" in data.keys() and "y_in" in data.keys():
        X_in = data["x_in"]
        Y_in = data["y_in"]
    else:
        raise KeyError("not find input data")

    d = X_in.shape[0]
    featname = [('%d' % i) for i in range(1, d + 1)]

    return X_in, Y_in, featname