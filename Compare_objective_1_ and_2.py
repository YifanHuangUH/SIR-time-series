from tabulate import tabulate
import numpy as np
from sklearn.preprocessing import StandardScaler

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
from tabulate import tabulate
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
from tabulate import tabulate
def ave(arr, N):
    for i in range(len(arr)):
        arr[i] = arr[i]/N 
    return arr    
def compute_eigen(Q4):
    eigenvalues2, eigenvectors2 = np.linalg.eig(Q4)
    K_index = np.argpartition(np.abs(eigenvalues2), P - K) >= P - K
    K_largest_eigenvectors = eigenvectors2[:, K_index]
    edr_est = K_largest_eigenvectors  
    if edr_est[0] < 0:
        edr_est = -edr_est
    edr_est = edr_est / np.linalg.norm(edr_est)
    return edr_est
    
def proj(edr_est): 
    E = edr_est @ np.linalg.inv(edr_est.T @ edr_est) @ edr_est.T
    return E

def exhi(obj1):        
    array1 = np.vectorize(lambda x: f"{x:.6f}")(obj1)
    table = tabulate(array1, tablefmt='latex_raw')
    # Split the table into lines
    lines = table.split('\n')
    # Insert \hline after each row
    latex_table = '\n'.join([line + (' \\hline' if (idx > 1) else '') for idx, line in enumerate(lines)])
    print(latex_table)

def Test1(ar_coeff, a1, a2, n_obs, H, S, n_rep):
    num_N = 5;P = 4;K = 1;n1 = 0;l = 1 
    noise = np.zeros((num_N, n_obs+S))
    y = [np.zeros((num_N, n_obs+i)) for i in range(S)]
    X1 = [[np.zeros((num_N, n_obs)) for i in range(P)] for i in range(S)]
    obj1 = [np.zeros((P, 1)) for _ in range(S)] # In each replicas, average for objective 1 
    obj2 = [np.zeros((P, 1)) for _ in range(S)] # In each replicas, average for objective 2 
    M = [np.zeros((P, 1)) for _ in range(S)] 
    proj1 = [np.zeros((P, P)) for _ in range(S)] # In each replicas, average proj for objective 1
    proj2 = [np.zeros((P, P)) for _ in range(S)] # In each replicas, average proj for objective 2
    g = np.zeros((S, 1))
    projection1_norm = np.zeros((S, 1))
    projection2_norm = np.zeros((S, 1))
    True_projection = np.array([[2],[3],[0],[0]])/(10*(np.linalg.norm(np.array([[2],[3],[0],[0]]))))
    obj2_avevec = [np.zeros((P, 1)) for _ in range(S)]
    obj2_ave = np.zeros((P, 1))
    while n1 < n_rep:
        # Data Generation
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
            phi = ar_coeff
            for j in range(P):
                for k in range(P):
                    Q3[j, k] = sum((phi[j] ** a) * (np.linalg.inv(np.cov(X.T)) @ V1[a] @ np.linalg.inv(np.cov(X.T)))[j, k] * (phi[k] ** a) for a in range(0, q))
            edr_est = compute_eigen(Q3)
            obj1[q - 1] += edr_est
            proj1[q - 1] += proj(edr_est)   
            
    # objective 2: experiment
        for q in range(0, S):
            Q4 = np.linalg.inv(np.cov(X1[q].T)) @ V1[q] @ np.linalg.inv(np.cov(X1[q].T))   # multiply np.linalg.inv(np.cov(X1[q].T)), by stationarity, it should cause no influence.
            # Q3 = np.linalg.inv(np.cov(X.T)) @ V1[q] @ np.linalg.inv(np.cov(X.T)), if we multiply np.linalg.inv(np.cov(X.T)) like this line, result for q = 0 should be the same.   
            K_largest_eigenvectors = compute_eigen(Q4)
            edr_est = np.multiply(np.power(ar_coeff, -q), K_largest_eigenvectors.flatten())   
            if edr_est[0] < 0:
                edr_est = -edr_est
            edr_est = edr_est / np.linalg.norm(edr_est)
            
            obj2[q] += edr_est.reshape(-1, 1)
            proj2[q] += proj(edr_est.reshape(-1, 1))
        
        n1 += 1    
    
    for i in range(S):
        obj2_avevec[i] = ave(obj2_avevec[i], n_rep)
        obj1[i] = obj1[i]/(n_rep)
        obj2[i] = obj2[i]/(n_rep)
        proj1[i] = ave(proj1[i], n_rep)
        proj2[i] = ave(proj2[i], n_rep)
        obj2_ave += obj2[i]   
        
    for j in range(S):
        for i in range(j, S, 1):
            obj2_avevec[i] += obj2[j]
    for j in range(S):
        obj2_avevec[j] = ave(obj2_avevec[j], S - j) 

    index_array = list(range(len(obj1)))
    
# def obj_exhibition():
    
    #objective 1: exhibition
    for i in range(S):
        projection1_norm[i] = np.linalg.norm(proj1[i] - proj(True_projection), 'fro')
        g[i] = abs(obj1[i][0] / obj1[i][1] - a1/a2)
        obj1[i] = np.vstack((obj1[i], g[i].reshape(1,-1)))
        obj1[i] = np.vstack((np.array([[index_array[i]]]), obj1[i])) 
        obj1[i] = np.vstack((np.array(obj1[i]), projection1_norm[i]))        
    
    #objective 2: exhibition
    for i in range(S):
        g[i] = abs(obj2[i][0] / obj2[i][1] - a1/a2)
        projection2_norm[i] = np.linalg.norm(proj2[i] - proj(True_projection), 'fro')
        obj2[i] = np.vstack((obj2[i], g[i].reshape(1,-1)))
        obj2[i] = np.vstack((np.array([[index_array[i]]]), obj2[i]))    
        obj2[i] = np.vstack((np.array(obj2[i]), projection2_norm[i]))
        obj2[i] = np.vstack((np.array(obj2[i]), abs(obj2_avevec[i][0]/obj2_avevec[i][1] - 2/3)))
        obj2[i] = np.vstack((np.array(obj2[i]), np.linalg.norm(proj(obj2_avevec[i]) - proj(True_projection), 'fro')))

    exhi(obj1), exhi(obj2)
    
#Test1(ar_coeff, a1, a2, n_obs, H, S, n_rep)
ar_coeff = [0.8, 0.6, 0.8, 0.8]
Test1(ar_coeff, 2, 3, 10000, 50, 10, 100)

