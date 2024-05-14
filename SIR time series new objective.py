import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_process import ArmaProcess
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))   
def sir_1(X, y, H, K):
    """Sliced Inverse Regression : Method 1
        Parameters
        ----------
        X : float, array of shape [n_samples, dimension]
            The generated samples.
        y : int, array of shape [n_samples, ]
            Labels of each generated sample.            
        y1 : repeating samples of y_{t+h}        
        H : number of slices
        K : return K estimated e.d.r directions
        Returns
        -------
        edr_est : estimated e.d.r directions
        """
    Z = X-np.mean(X, axis=0)
    width = (np.max(y) - np.min(y)) / H
    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y)+h*width <= y, y < np.min(y)+(h+1)*width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        mh = np.mean(Z[h_index, :], axis=0)
        V_hat = np.add(V_hat,ph_hat * np.matmul(mh[:, np.newaxis], mh[np.newaxis, :]))
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est =  K_largest_eigenvectors       
    return  edr_est

def sir_1_time_series(X, y, H, K):
    
    Z = X-np.mean(X, axis=0)
    width = (np.max(y) - np.min(y)) / H
    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y)+h*width <= y, y < np.min(y)+(h+1)*width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        mh = np.mean(Z[h_index, :], axis=0)
        V_hat = np.add(V_hat,ph_hat * np.matmul(mh[:, np.newaxis], mh[np.newaxis, :]))

    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est =  K_largest_eigenvectors         
    return  (V_hat, edr_est, eigenvalues ** 2)





#from 0 to Q 
from tabulate import tabulate
def upper1(ar_coeff):
    num_N = 5
    n_obs = 10000
    noise = np.zeros((num_N, n_obs))
    n = 100
    H = 48
    P = 4
    K = 1
    S = 20
    hat = [np.zeros((P, 1)) for i in range(S)]
    # ar_coeff = [0.2, 0.2, 0.2, 0.2]
    
    for w in range(n):    
        for h in range(num_N):
            noise[h] = np.random.normal(0, 1, size=n_obs)  # Normally distributed noise
        ar_series = np.zeros((num_N, n_obs))
        for t in range(0, n_obs):
            ar_series[0][t] = ar_coeff[0] * ar_series[0][t - 1]  + noise[0][t]
            ar_series[1][t] = ar_coeff[1] * ar_series[1][t - 1]  + noise[1][t]
            ar_series[2][t] = ar_coeff[2] * ar_series[2][t - 1]  + noise[2][t]
            ar_series[3][t] = ar_coeff[3] * ar_series[3][t - 1]  + noise[3][t]
            ar_series[4][t] = 2*ar_series[0][t] + 3*ar_series[1][t] + noise[4][t]
        y = ar_series[4]
        X = np.concatenate([ar_series[i].reshape(-1,1) for i in range(4)], axis = 1)
        V = []
        for a in range(0,S+1):
            M, edr, lam1 = sir_1_time_series(X[:n_obs - a + 1], y[:n_obs - a + 1], H=H, K=K)
            V.append(M)
        for q in range(1, S+1):
            Q = np.zeros((P, P))
            phi = ar_coeff
            for j in range(P):
                for k in range(P):
                    Q[j,k] = sum(phi[j]**a * V[a][j,k] * phi[k]**a for a in range(0,q))  #phi[j] and phi[k]
            eigenvalues1, eigenvectors1 = np.linalg.eig(Q)
            K_index = np.argpartition(np.abs(eigenvalues1), P-K) >= P-K
            K_largest_eigenvectors = eigenvectors1[:, K_index]
            edr_est =  K_largest_eigenvectors
            if edr_est[0]<0:
                edr_est = -edr_est
            edr_est = edr_est/np.linalg.norm(edr_est)
            hat[q-1] += edr_est       
    for i in range(S):
        hat[i] = hat[i]/n

    array = np.array(hat)
    
    print(tabulate(array, tablefmt='latex'))

ar_coeff = [0.2,0.2,0.2,0.2]
upper1(ar_coeff)
