#from 0 to Q double sum
from tabulate import tabulate
import numpy as np

def sir_1_time_series2(X, y, num_slices, K):
        n_samples, n_features = X.shape
        V_hat = np.zeros([X.shape[1], X.shape[1]])
        # Step 1: Sort the data by the response variable
        sorted_indices = np.argsort(y)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Step 2: Divide the data into slices
        slice_size = n_samples // num_slices
        ph_hat = slice_size/n_samples
        slices = []
        
        for i in range(num_slices):
            start_idx = i * slice_size
            if i < num_slices - 1:
                end_idx = (i + 1) * slice_size
            else:  # Last slice includes any remaining samples
                end_idx = n_samples
            slices.append((X_sorted[start_idx:end_idx], y_sorted[start_idx:end_idx]))
        
        # Step 3: Compute the means of the predictors within each slice
        X_means = np.array([np.mean(slice_X, axis=0) for slice_X, _ in slices])
        
        # Step 4: Center the predictor means
        X_centered = X_means - np.mean(X_means, axis=0)
        
        V_hat = np.add(V_hat,ph_hat * np.matmul(X_centered.T, X_centered))
        eigenvalues, eigenvectors = np.linalg.eig(V_hat)
        K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
        K_largest_eigenvectors = eigenvectors[:, K_index]
        edr_est =  K_largest_eigenvectors
        
        return V_hat, edr_est, eigenvalues ** 2
    
def test(ar_coeff):
    import numpy as np
    from tabulate import tabulate
    num_N = 5
    n_obs = 10000
    S = 20
    noise = np.zeros((num_N, n_obs+S))
    n = 100
    H = 50
    P = 4
    K = 1
    y = [np.zeros((num_N, n_obs+i)) for i in range(S+1)]
    hat = [np.zeros((P, 1)) for _ in range(S)]
    g = np.zeros((S, 1))
    # ar_coeff = [0.2, 0.2, 0.2, 0.2]
    n1 = 0
    l = 1  # Initialize `l` outside the loop
    while n1 < 100:
        for h in range(num_N):
            noise[h] = np.random.normal(0, 1, size=(n_obs+S))  # Normally distributed noise
        ar_series = np.zeros((num_N, n_obs+S))
        for t in range(0, n_obs+S):
            ar_series[0][t] = ar_coeff[0] * ar_series[0][t - 1] + noise[0][t]
            ar_series[1][t] = ar_coeff[1] * ar_series[1][t - 1] + noise[1][t]
            ar_series[2][t] = ar_coeff[2] * ar_series[2][t - 1] + noise[2][t]
            ar_series[3][t] = ar_coeff[3] * ar_series[3][t - 1] + noise[3][t]
            ar_series[4][t] = 2 * ar_series[0][t] + 3 * ar_series[1][t] + noise[4][t]
        for a in range(0, S+1):
            y[a] = ar_series[4][a:n_obs+a]
        X = np.concatenate([ar_series[i][0:n_obs].reshape(-1, 1) for i in range(4)], axis=1)
        V = []
        for a in range(0, S + 1):
            M, _, _ = sir_1_time_series2(X, y[a], H, K)
            V.append(M)
        for q in range(1, S + 1):
            Q = np.zeros((P, P))
            phi = ar_coeff
            for j in range(P):
                for k in range(P):
                    Q[j, k] = sum(sum(phi[j] ** a * V[a][j, k] * phi[k] ** a for a in range(0, l)) for l in range(1, q + 1))
            eigenvalues1, eigenvectors1 = np.linalg.eig(Q)
            K_index = np.argpartition(np.abs(eigenvalues1), P - K) >= P - K
            K_largest_eigenvectors = eigenvectors1[:, K_index]
            edr_est = K_largest_eigenvectors
            if edr_est[0] < 0:
                edr_est = -edr_est
            edr_est = edr_est / np.linalg.norm(edr_est)
            hat[q - 1] += edr_est
            n1 += 1
    
    for i in range(S):
        hat[i] = hat[i] / n
        g[i] = hat[i][0] / hat[i][1]
    array = np.array(hat)
    print(tabulate(array, tablefmt='latex'))
    print(g)

# Example usage
ar_coeff = [0.2, 0.2, 0.2, 0.2]
test(ar_coeff)

ar_coeff = [0.2, 0.5, 0.8, 0.8]
test(ar_coeff)

