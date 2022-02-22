import numpy as np
import scipy.special


class Updates():
    def __init__(self, sigma=1.0, a=1.5, beta=0.0, lam = None, numiter=100, objhowoften=1, tol=1e-5, sigmaknown=False):
        self.sigma = sigma
        self.a = a
        self.beta = beta
        self.lam = lam
        self.numiter = numiter
        self.objhowoften = objhowoften
        self.tol = tol
        self.sigmaknown = sigmaknown
        self.active_set = None
        
    
    
    def compute_obj_fast(self, y, sigmasq, beta, f, zeta, Sigma, trS, wsq, tensor_cal6, tensor_cal7):
        sign_logdet_Sigma, logdet_Sigma = np.linalg.slogdet(Sigma.transpose(2,0,1))
        obj = 0.5 / sigmasq * np.sum((y - tensor_cal6) ** 2) + np.sum((0.5 * f.reshape(-1,1) * (wsq + trS) + 1) / zeta + (self.a + 0.5) * np.log(zeta)) + 0.5 * np.sum(-sign_logdet_Sigma * logdet_Sigma + tensor_cal7 / sigmasq) + self.K * (0.5 * self.N * np.log(2 * np.pi * sigmasq) - np.sum((beta + 0.5) * np.log(f)) + self.P * (-(self.a + 1.0) + (self.a + 0.5) * np.log(self.a + 0.5) - np.log(scipy.special.gamma(self.a + 0.5)) + np.log(scipy.special.gamma(self.a))))

        return obj        
    
    
    def __proximal_gradient_method(self, w, sigmasq, tensor_cal4_inv, XTy, XTX, absLSS):
        count, epsilon = 1, np.inf
        w_old = w
        
        if(self.active_set is None):
            self.active_set = np.arange(self.P)
        
        A = np.diagonal(tensor_cal4_inv[self.active_set][:,self.active_set], axis1 = 0, axis2 = 1).T
        
        while True:
            B = np.stack([XTy[self.active_set][:,k] + (np.eye(len(self.active_set))-1) * XTX[self.active_set][:,self.active_set, k] @ w_old[self.active_set][:,k] for k in range(self.K)], axis=-1)
            negative = B < (-sigmasq * self.lam[0] * self.N / absLSS[self.active_set])
            positive = B > (sigmasq * self.lam[1] * self.N / absLSS[self.active_set])
            to_zero = ~(negative + positive)
          
            w_new_active_set = 1 / A
            w_new_active_set[negative] *= B[negative] + sigmasq * self.lam[0] * self.N / absLSS[self.active_set][negative]
            w_new_active_set[positive] *= B[positive] - sigmasq * self.lam[1] * self.N / absLSS[self.active_set][positive]
            w_new_active_set[to_zero] = 0
            
            w_new = np.zeros(w_old.shape)
            w_new[self.active_set] = w_new_active_set

            count += 1
            
            epsilon_tmp = np.linalg.norm(w_new - w_old)
            if((epsilon - epsilon_tmp) < self.tol):
                if(np.linalg.norm(w_new) < self.tol):
                    w_new = np.copy(w_old)
                break
            elif(epsilon_tmp < self.tol):
                if(np.linalg.norm(w_new) < self.tol):
                    w_new = np.copy(w_old)
                break
            else:
                epsilon = epsilon_tmp
                
            w_old = np.copy(w_new)
        
        active_set = np.where(np.abs(w_new) > self.tol)[0]
        if(len(active_set) == 0):
            self.active_set = -1
        else:
            self.active_set = np.unique(np.where(np.abs(w_new) > self.tol)[0])
        return w_new

    

    def fit(self, *, y, X, f_init = None):
        sigma = self.sigma
        sigmasq = sigma ** 2
        
        self.N, self.P, self.K = X.shape

        f = np.ones(self.P) if f_init is None else np.asarray(f_init)
        Sigma = np.zeros([self.P, self.P, self.K])
        zeta = np.ones([self.P, self.K])
        trS = np.ones([self.P, self.K])
        wsq = np.ones([self.P, self.K])
        w = np.ones([self.P, self.K])

        Obj = [float('inf')]
        epsilon = 1
        c = 0
        
        y, X, beta = np.asarray(y), np.asarray(X), np.asarray(self.beta)
        XTX = np.matmul(X.transpose(2,1,0), X.transpose(2,0,1)).transpose(1,2,0)
        XTy = np.stack([X[:,:,k].T @ y[:,k] for k in range(self.K)], axis=-1)
        LSS = np.stack([np.linalg.inv(XTX[:,:,k]) @ XTy[:,k] for k in range(self.K)], axis=-1)
        absLSS = np.abs(LSS)
        
        f_save, w_save, sigma_save, bound_save = {}, {}, {}, {}

        while(c <= self.numiter and epsilon > self.tol):            
            zeta = (1 + 0.5 * f.reshape(-1,1) * (wsq + trS)) / (self.a + 0.5)
            f = self.K * (1 + 2 * beta) / np.sum((wsq + trS) / zeta, axis = 1)
            
            tensor_cal4_inv = np.stack([XTX[:,:,k] + sigmasq * np.diag(f / zeta[:,k]) for k in range(self.K)], axis=-1)
            tensor_cal4 = np.stack([np.linalg.inv(tensor_cal4_inv[:,:,k]) for k in range(self.K)], axis=-1)

            Sigma = sigmasq * tensor_cal4
            
            if(c == 0):
                w = np.stack([np.dot(np.matmul(tensor_cal4.transpose(2,0,1), X.transpose(2,1,0))[k,:,:], y.T[k,:]) for k in range(self.K)]).T
                c = 1
            else:
                w = self.__proximal_gradient_method(w, sigmasq, tensor_cal4_inv, XTy, XTX, absLSS)
            
            
            trS = np.diagonal(Sigma, axis1 = 0, axis2 = 1).T
            wsq = w**2

            if(self.sigmaknown == False):
                tensor_cal6 = np.stack([np.dot(X.transpose(2,0,1)[k,:,:], w.T[k,:]) for k in range(self.K)]).T
                tensor_cal7 = np.trace(np.matmul(XTX.transpose(2,1,0), Sigma.transpose(2,0,1)).transpose(1,2,0), axis1 = 0, axis2 = 1)
                
                sigmasq = (np.sum((y - tensor_cal6) ** 2) + np.sum(tensor_cal7)) / (self.N * self.K)

            if(c % self.objhowoften == 0):
                if(self.sigmaknown == False):
                    obj = self.compute_obj_fast(y, sigmasq, beta, f, zeta, Sigma, trS, wsq, tensor_cal6, tensor_cal7)
                else:
                    tensor_cal6 = np.stack([np.dot(X.transpose(2,0,1)[k,:,:], w.T[k,:]) for k in range(self.K)]).T
                    tensor_cal7 = np.trace(np.matmul(XTX.transpose(2,1,0), Sigma.transpose(2,0,1)).transpose(1,2,0), axis1 = 0, axis2 = 1)
                    obj = self.compute_obj_fast(y, sigmasq, beta, f, zeta, Sigma, trS, wsq, tensor_cal6, tensor_cal7)
                    
            epsilon = np.abs((Obj[-1] - obj) / obj)
            
            Obj.append(obj)
            bound = obj + self.K * np.sum(beta * np.log(f))
            
            f_save[str(c)], w_save[str(c)], sigma_save[str(c)], bound_save[str(c)] = f, w, np.sqrt(sigmasq), bound
            
            c = c + 1    
            if(self.active_set is -1):
                break
            
        self.fhat_process, self.what_process, self.sigmahat_process, self.bound_process = f_save, w_save, sigma_save, bound_save
        self.fhat, self.what, self.sigmahat, self.bound = f, w, np.sqrt(sigmasq), bound
        
        return f, w, np.sqrt(sigmasq), bound