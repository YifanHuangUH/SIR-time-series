from tabulate import tabulate
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
from tabulate import tabulate

# Function
def sir_11(X, y, num_slices, K):
    X = X - np.mean(X, axis = 0)
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
    X_centered = X_means - np.mean(X, axis=0)
    
    V_hat = np.add(V_hat,ph_hat * np.matmul(X_centered.T, X_centered))
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est =  K_largest_eigenvectors
    return edr_est, V_hat
    
# Experiment
def Test1(ar_coeff, a1, a2, n_obs, S, n_rep):
    num_N = 5
    H = 5
    P = 4
    K = 1
    n1 = 0
    l = 1 
    noise = np.zeros((num_N, n_obs+S))
    y = [np.zeros((num_N, n_obs+i)) for i in range(S)]
    X1 = [[np.zeros((num_N, n_obs)) for i in range(num_N - 1)] for i in range(S)]
    hat = [np.zeros((P, 1)) for _ in range(S)] # In each replicas, average for objective 1 
    Hat = [np.zeros((P, 1)) for _ in range(S)] # In each replicas, average for objective 2 
    g = np.zeros((S, 1))
    hat_ave = np.zeros((P, 1))
    Hat_ave = np.zeros((P, 1))
    while n1 < n_rep:
        for h in range(num_N):
            noise[h] = np.random.normal(0, 1, size=(n_obs+S))  
        ar_series = np.zeros((num_N, n_obs+S))
        for t in range(0, n_obs+S):
            ar_series[0][t] = ar_coeff[0] * ar_series[0][t - 1] + noise[0][t]
            ar_series[1][t] = ar_coeff[1] * ar_series[1][t - 1] + noise[1][t]
            ar_series[2][t] = ar_coeff[2] * ar_series[2][t - 1] + noise[2][t]
            ar_series[3][t] = ar_coeff[3] * ar_series[3][t - 1] + noise[3][t]
            ar_series[4][t] = a1 * ar_series[0][t] + a2 * ar_series[1][t] + noise[4][t]
        for a in range(0, S):
            y[a] = ar_series[4][a:n_obs+a]
            X1[a] = np.concatenate([ar_series[i][a:n_obs+a].reshape(-1,1) for i in range(P)], axis = 1)
        X = X1[0]
        V1 = []
        for a in range(0, S):
            _, M = sir_11(X, y[a], H, K)
            V1.append(M)
    # objective 1: experiment
        for q in range(1, S + 1):
            Q3 = np.zeros((P, P))
            phi = ar_coeff
            for j in range(P):
                for k in range(P):
                    Q3[j, k] = sum((phi[j] ** a) * (np.linalg.inv(np.cov(X.T)) @ V1[a] @ np.linalg.inv(np.cov(X.T)))[j, k] * (phi[k] ** a) for a in range(0, q))
            eigenvalues1, eigenvectors1 = np.linalg.eig(Q3)
            K_index = np.argpartition(np.abs(eigenvalues1), P - K) >= P - K
            K_largest_eigenvectors = eigenvectors1[:, K_index]
            edr_est = K_largest_eigenvectors  # or multiply np.linalg.inv(np.cov(X.T))
            if edr_est[0] < 0:
                edr_est = -edr_est
            edr_est = edr_est / np.linalg.norm(edr_est)
            hat[q - 1] += edr_est
            
        # In each replicas, average it up   
        for i in range(S):
            hat[i] = hat[i] / S   
            hat_ave += hat[i]        
    # objective 2: experiment
        for q in range(0, S):
            Q4 = np.zeros((P, P))
            Q4 = np.linalg.inv(np.cov(X1[q].T)) @ V1[q] @ np.linalg.inv(np.cov(X1[q].T))   # multiply np.linalg.inv(np.cov(X1[q].T)), by stationarity, it should cause no influence.
            
            # Q3 = np.linalg.inv(np.cov(X.T)) @ V1[q] @ np.linalg.inv(np.cov(X.T)), if we multiply np.linalg.inv(np.cov(X.T)) like this line, result for q = 0 should be the same.
            
            eigenvalues2, eigenvectors2 = np.linalg.eig(Q4)
            K_index = np.argpartition(np.abs(eigenvalues2), P - K) >= P - K
            K_largest_eigenvectors = eigenvectors2[:, K_index]
            edr_est = np.multiply(np.power(ar_coeff, -q), K_largest_eigenvectors.flatten())
            
            if edr_est[0] < 0:
                edr_est = -edr_est
            edr_est = edr_est / np.linalg.norm(edr_est)
            Hat[q] += edr_est.reshape(-1, 1)

        for i in range(S):
            Hat[i] = Hat[i] / S   
            Hat_ave += Hat[i]
        
        n1 += 1
        
    # Average by repetation for objective 1 and 2, better to use n_rep = 1 first.
    hat_ave = hat_ave/n_rep  
    
    Hat_ave = Hat_ave/n_rep

    # Index column
    index_array = list(range(len(hat)))
    
    #objective 1: exhibition
    for i in range(S):
        g[i] = abs(hat[i][0] / hat[i][1] - a1/a2)
        hat[i] = np.vstack((hat[i], g[i].reshape(1,-1)))
        hat[i] = np.vstack((np.array([[index_array[i]]]), hat[i]))    
    array1 = np.vectorize(lambda x: f"{x:.4f}")(hat)
    table = tabulate(array1, tablefmt='latex_raw')
    # Split the table into lines
    lines = table.split('\n')
    # Insert \hline after each row
    latex_table = '\n'.join([line + (' \\hline' if (idx > 1) else '') for idx, line in enumerate(lines)])
    print(latex_table)
    print(np.round(hat_ave.flatten(), 4))
    print(np.round(abs(hat_ave[0]/hat_ave[1] - a1/a2),4))
    
    #objective 2: exhibition
    for i in range(S):
        g[i] = abs(Hat[i][0] / Hat[i][1] - a1/a2)
        Hat[i] = np.vstack((Hat[i], g[i].reshape(1,-1)))
        Hat[i] = np.vstack((np.array([[index_array[i]]]), Hat[i]))    
    array2 = np.vectorize(lambda x: f"{x:.4f}")(Hat)
    table = tabulate(array2, tablefmt='latex_raw')
    # Split the table into lines
    lines = table.split('\n')
    # Insert \hline after each row
    latex_table = '\n'.join([line + (' \\hline' if (idx > 1) else '') for idx, line in enumerate(lines)])
    print(latex_table)
    print(np.round(Hat_ave.flatten(), 4))
    print(np.round(abs(Hat_ave[0]/Hat_ave[1] - a1/a2),4))
    
# Example
ar_coeff = [0.2, 0.5, 0.5, 0.8]
Test1(ar_coeff, 2, 3, 10000, 10, 1)



# Generation steps for objective 1
num_N = 5; a1 = 2; a2 = 3; H = 5; P = 4; K = 1; n1 = 0; l = 1; n_obs = 100; S = 10
y = [np.zeros((num_N, n_obs+i)) for i in range(S)]
X1 = [[np.zeros((num_N, n_obs)) for i in range(num_N - 1)] for i in range(S)]
hat = [np.zeros((P, 1)) for _ in range(S)] # In each replicas, average for objective 1 
Hat = [np.zeros((P, 1)) for _ in range(S)] # In each replicas, average for objective 2 
g = np.zeros((S, 1))
hat_ave = np.zeros((P, 1))
Hat_ave = np.zeros((P, 1))
noise = np.zeros((num_N, n_obs+S))
for h in range(num_N):
    noise[h] = np.random.normal(0, 1, size=(n_obs+S))  
ar_series = np.zeros((num_N, n_obs+S))
for t in range(0, n_obs+S):
    ar_series[0][t] = ar_coeff[0] * ar_series[0][t - 1] + noise[0][t]
    ar_series[1][t] = ar_coeff[1] * ar_series[1][t - 1] + noise[1][t]
    ar_series[2][t] = ar_coeff[2] * ar_series[2][t - 1] + noise[2][t]
    ar_series[3][t] = ar_coeff[3] * ar_series[3][t - 1] + noise[3][t]
    ar_series[4][t] = a1 * ar_series[0][t] + a2 * ar_series[1][t] + noise[4][t]
for a in range(0, S):
    y[a] = ar_series[4][a:n_obs+a]
    X1[a] = np.concatenate([ar_series[i][a:n_obs+a].reshape(-1,1) for i in range(P)], axis = 1)
X = X1[0]
V1 = []
for a in range(0, S):
    _, M = sir_11(X, y[a], H, K)
    V1.append(M)
# objective 1: experiment
for q in range(1, S + 1):
    Q3 = np.zeros((P, P))
    phi = ar_coeff
    for j in range(P):
        for k in range(P):
            Q3[j, k] = sum((phi[j] ** a) * (np.linalg.inv(np.cov(X.T)) @ V1[a] @ np.linalg.inv(np.cov(X.T)))[j, k] * (phi[k] ** a) for a in range(0, q))
    eigenvalues1, eigenvectors1 = np.linalg.eig(Q3)
    K_index = np.argpartition(np.abs(eigenvalues1), P - K) >= P - K
    K_largest_eigenvectors = eigenvectors1[:, K_index]
    edr_est = K_largest_eigenvectors  # or multiply np.linalg.inv(np.cov(X.T))
    if edr_est[0] < 0:
        edr_est = -edr_est
    edr_est = edr_est / np.linalg.norm(edr_est)
    hat[q - 1] += edr_est
    
# In each replicas, average it up   
for i in range(S):
    hat[i] = hat[i] / S   
    hat_ave += hat[i]




# Result 1
hat




# Generation steps for objective 2
for q in range(0, S):
    Q4 = np.zeros((P, P))
    Q4 = np.linalg.inv(np.cov(X1[q].T)) @ V1[q] @ np.linalg.inv(np.cov(X1[q].T))   # multiply np.linalg.inv(np.cov(X1[q].T)), by stationarity, it should cause no influence.
    
    # Q3 = np.linalg.inv(np.cov(X.T)) @ V1[q] @ np.linalg.inv(np.cov(X.T)), if we multiply np.linalg.inv(np.cov(X.T)) like this line, result for q = 0 should be the same.
    
    eigenvalues2, eigenvectors2 = np.linalg.eig(Q4)
    K_index = np.argpartition(np.abs(eigenvalues2), P - K) >= P - K
    K_largest_eigenvectors = eigenvectors2[:, K_index]
    edr_est = np.multiply(np.power(ar_coeff, -q), K_largest_eigenvectors.flatten())
    
    if edr_est[0] < 0:
        edr_est = -edr_est
    edr_est = edr_est / np.linalg.norm(edr_est)
    Hat[q] += edr_est.reshape(-1, 1)
for i in range(S):
    Hat[i] = Hat[i] / S   
    Hat_ave += Hat[i]




# Result 2
Hat






# Test 
np.power(ar_coeff, -3)
np.multiply(np.power(ar_coeff, -3), [2,3,4,5])
