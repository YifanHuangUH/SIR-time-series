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
    X_centered = (X_means - np.mean(X, axis=0))
    # X_centered = (X_means - np.mean(X, axis=0))/np.linalg.norm(X_means - np.mean(X, axis=0))
    
    V_hat = np.add(V_hat,ph_hat * np.matmul(X_centered.T, X_centered))
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est =  K_largest_eigenvectors
    return edr_est, V_hat

#add risk to be averaged over replicas, add prediction error for obj2 for averaged direction
from tabulate import tabulate
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
def ave(arr, N):
    for i in range(len(arr)):
        arr[i] = arr[i]/N 
    return arr  

def compute_eigen(Q4, P, K):
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
    
def MSE(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate errors
    mse = mean_squared_error(y_test, y_pred)
    return mse
    
def Test1(ar_coeff, a1, a2, initial_value, n_gene, n_obs, H, S, n_rep):
    num_N = 5;P = 4;K = 1;n1 = 0;l = 1 
    noise1 = np.zeros((num_N, n_gene))
    noise2 = np.zeros((num_N, n_obs+S))
    y = [np.zeros((num_N, n_obs+i)) for i in range(S)]
    X1 = [[np.zeros((num_N, n_obs)) for i in range(P)] for i in range(S)]
    obj_1 = [np.zeros((P, 1)) for _ in range(S)]  
    obj_2 = [np.zeros((P, 1)) for _ in range(S)]  
    prediction_mse_1 = [np.zeros((1, 1)) for _ in range(S)]
    prediction_mse_2 = [np.zeros((1, 1)) for _ in range(S)]
    M = [np.zeros((P, 1)) for _ in range(S)] 
    proj_error_1 = [np.zeros((1, )) for _ in range(S)] 
    proj_error_2 = [np.zeros((1, )) for _ in range(S)] 
    error_1 = [np.zeros((1, )) for _ in range(S)]
    error_2 = [np.zeros((1, )) for _ in range(S)]
    g = np.zeros((S, 1))
    g1 = np.zeros((S, 1))
    g2 = np.zeros((S, 1))
    projection_1_norm = np.zeros((S, 1))
    projection_2_norm = np.zeros((S, 1))
    True_projection = np.array([[2],[3],[0],[0]])/((np.linalg.norm(np.array([[2],[3],[0],[0]]))))
    obj_2_avevec = [np.zeros((P, 1)) for _ in range(S)]
    obj2_ave = np.zeros((P, 1))
    while n1 < n_rep:  
        obj_2 = [np.zeros((P, 1)) for _ in range(S)]
        obj_2_avevec = [np.zeros((P, 1)) for _ in range(S)]
        # Data Generation
        for h in range(num_N):
            noise1[h] = np.random.normal(0, 1, size=(n_gene)) 
            noise2[h] = np.random.normal(0, 1, size=(n_obs+S))
            # noise[h] = np.random.normal(0, 0.5, size=(n_obs+S))
        ar_series = np.zeros((num_N, n_obs+S+1))
        # change initial value and burn-off?
        # for i in range(num_N):
        #     ar_series[i][0] = initial_value
            
        ############################################################### 
        #Burn-off
        AR1 = initial_value
        AR2 = initial_value
        AR3 = initial_value
        AR4 = initial_value
        for t in range(1, n_gene + 1):
            AR1 = ar_coeff[0] * AR1 + noise1[0][t-1]
            AR2 = ar_coeff[1] * AR2 + noise1[1][t-1]
            AR3 = ar_coeff[2] * AR3 + noise1[2][t-1]
            AR4 = ar_coeff[3] * AR4 + noise1[3][t-1]
        
        ar_series[0][0] = AR1 
        ar_series[1][0] = AR2 
        ar_series[2][0] = AR3 
        ar_series[3][0] = AR4 
        for t in range(1, n_obs+S+1): 
            ar_series[0][t] = ar_coeff[0] * ar_series[0][t - 1] + noise2[0][t - 1]
            ar_series[1][t] = ar_coeff[1] * ar_series[1][t - 1] + noise2[1][t - 1]
            ar_series[2][t] = ar_coeff[2] * ar_series[2][t - 1] + noise2[2][t - 1]
            ar_series[3][t] = ar_coeff[3] * ar_series[3][t - 1] + noise2[3][t - 1]
            ar_series[4][t] = a1 * ar_series[0][t-1] + a2 * ar_series[1][t-1] + noise2[4][t - 1]
        
        for a in range(0, S):
            y[a] = ar_series[4][a:n_obs+a]
            X1[a] = np.concatenate([ar_series[i][a:n_obs+a].reshape(-1, 1) for i in range(P)], axis = 1)

        ################################################################
        # original data generation
        # for t in range(0, n_obs+S):
        #     ar_series[0][t] = ar_coeff[0] * ar_series[0][t - 1] + noise[0][t]
        #     ar_series[1][t] = ar_coeff[1] * ar_series[1][t - 1] + noise[1][t]
        #     ar_series[2][t] = ar_coeff[2] * ar_series[2][t - 1] + noise[2][t]
        #     ar_series[3][t] = ar_coeff[3] * ar_series[3][t - 1] + noise[3][t]
        #     ar_series[4][t] = a1 * ar_series[0][t] + a2 * ar_series[1][t] + noise[4][t]
        # for a in range(0, S):
        #     y[a] = ar_series[4][a:n_obs+a]
        #     X1[a] = np.concatenate([ar_series[i][a:n_obs+a].reshape(-1, 1) for i in range(P)], axis = 1)
        X = X1[0]
        V1 = []
        for a in range(0, S):
            _, M = sir_11(X, y[a], H, K)
            V1.append(M)
    # objective 1: experiment
        for q in range(0, S):
            phi = ar_coeff
            Q3 = np.zeros((P, P))
            for j in range(P):
                for k in range(P):
                    Q3[j, k] = sum((phi[j] ** a) * (np.linalg.inv(np.cov(X.T)) @ V1[a] @ np.linalg.inv(np.cov(X.T)))[j, k] * (phi[k] ** a) for a in range(0, q + 1))
            edr_est = compute_eigen(Q3, P, K)
            if edr_est[0] < 0:
                edr_est = -edr_est
            edr_est = edr_est / np.linalg.norm(edr_est)
           
            error_1[q] += abs(edr_est[0] / edr_est[1] - a1/a2)
            prediction_mse_1[q] += MSE(X @ edr_est, y[0]) 
            proj_error_1[q] += np.linalg.norm((proj(edr_est) - proj(True_projection))**2, 'fro')
            # proj_error_1[q] += np.linalg.norm(proj(edr_est) - proj(True_projection), 'fro')
            
     # objective 2: experiment
        for q in range(0, S):
            Q4 = np.linalg.inv(np.cov(X1[q].T)) @ V1[q] @ np.linalg.inv(np.cov(X1[q].T))   # multiply np.linalg.inv(np.cov(X1[q].T)), by stationarity, it should cause no influence.
            # Q3 = np.linalg.inv(np.cov(X.T)) @ V1[q] @ np.linalg.inv(np.cov(X.T)), if we multiply np.linalg.inv(np.cov(X.T)) like this line, result for q = 0 should be the same.   
            K_largest_eigenvectors = compute_eigen(Q4, P, K)
            edr_est = np.multiply(np.power(ar_coeff, -q), K_largest_eigenvectors.flatten())   
            if edr_est[0] < 0:
                edr_est = -edr_est
            edr_est = edr_est / np.linalg.norm(edr_est)
            obj_2[q] += edr_est.reshape(-1, 1)    
            
        # Average among lags Q
        for j in range(S):
            for i in range(j, S, 1):
                obj_2_avevec[i] += obj_2[j]
        for j in range(S):
            obj_2_avevec[j] = ave(obj_2_avevec[j], j + 1)
            
        for q in range(0, S):
            error_2[q] += abs(obj_2_avevec[q][0] / obj_2_avevec[q][1] - a1/a2)
            prediction_mse_2[q] += MSE(X @ obj_2_avevec[q], y[0])
            proj_error_2[q] += np.linalg.norm((proj(obj_2_avevec[q]) - proj(True_projection))**2, 'fro')
            # proj_error_2[q] += np.linalg.norm(proj(obj_2_avevec[q]) - proj(True_projection), 'fro')            
        n1 += 1  
        
    error_1 = ave(error_1, n_rep)    
    error_2 = ave(error_2, n_rep)
    prediction_mse_1 = ave(prediction_mse_1, n_rep)
    prediction_mse_2 = ave(prediction_mse_2, n_rep)
    proj_error_1 = ave(proj_error_1, n_rep)
    proj_error_2 = ave(proj_error_2, n_rep)
    
    # for i in range(S):
    # obj1[i] = obj1[i]/(n_rep)
    # obj2[i] = obj2[i]/(n_rep)
    # proj1[i] = proj(obj1[i])
    # proj2[i] = proj(obj2[i])
    # mse[i] = Mse1(X @ obj1[i], y[0])
    # mse2[i] = Mse1(X @ obj2[i].reshape(-1, 1), y[0])
    # obj2_ave += obj2[i]       
    
    # index_array = list(range(len(obj_1)))
    # for i in range(S):
        # error_1[i] = abs(obj_1[i][0] / obj_1[i][1] - a1/a2)
    #     obj_1[i] = np.vstack((obj_1[i], g[i].reshape(1,-1)))
    #     obj_1[i] = np.vstack((np.array([[index_array[i]]]), obj_1[i])) 
    #     obj_1[i] = np.vstack((np.array(obj_1[i]), proj_error_1[i])) 
    #     obj_1[i] = np.vstack((np.array(obj_1[i]), mse[i]))
  
    #objective 1: exhibition
    # for i in range(S):
    #     g[i] = abs(obj_1[i][0] / obj_1[i][1] - a1/a2)
    #     obj_1[i] = np.vstack((obj_1[i], g[i].reshape(1,-1)))
    #     obj_1[i] = np.vstack((np.array([[index_array[i]]]), obj_1[i])) 
    #     obj_1[i] = np.vstack((np.array(obj_1[i]), proj_error_1[i])) 
    #     obj_1[i] = np.vstack((np.array(obj_1[i]), mse[i]))
        
    # Result1 = [g[0] - min(g), projection1_norm[0] - min(projection1_norm), mse[0] - min(mse)]
    # Result1 = [x.item() for x in Result1]
    #objective 2: exhibition
    
    # for i in range(S):
    #     g[i] = abs(obj2[i][0] / obj2[i][1] - a1/a2)
    #     projection2_norm[i] = np.linalg.norm(proj2[i] - proj(True_projection), 'fro')
    #     obj2[i] = np.vstack((obj_2_avevec[i], g[i].reshape(1,-1)))
    #     obj2[i] = np.vstack((np.array([[index_array[i]]]), obj_2_avevec[i]))    
    #     obj2[i] = np.vstack((np.array(obj2[i]), projection2_norm[i]))
    #     obj2[i] = np.vstack((np.array(obj2[i]), mse2[i]))
    #     obj2[i] = np.vstack((np.array(obj2[i]), abs(obj_2_avevec[i][0]/obj_2_avevec[i][1] - 2/3)))
    #     obj2[i] = np.vstack((np.array(obj2[i]), np.linalg.norm(proj(obj_2_avevec[i]) - proj(True_projection), 'fro')))
    #     obj2[i] = np.vstack((np.array(obj2[i]), Mse1(X @ obj_2_avevec[i], y[0])))
    
    # for i in range(S):
    #     g[i] = abs(obj_2_avevec[i][0] / obj_2_avevec[i][1] - a1/a2)
    #     g1[i] = proj_error_2[i]
    #     g2[i] = Mse1(X @ obj_2_avevec[i], y[0])
    #     obj_2_avevec[i] = np.vstack((obj_2_avevec[i], g[i].reshape(1,-1)))
    #     obj_2_avevec[i] = np.vstack((np.array([[index_array[i]]]), obj_2_avevec[i]))    
    #     obj_2_avevec[i] = np.vstack((obj_2_avevec[i], g1[i]))
    #     obj_2_avevec[i] = np.vstack((obj_2_avevec[i], g2[i]))
    
    # Result2 = [g[0] - min(g), g1[0] - min(g1), g2[0] - min(g2)] 
    # Result2 = [x.item() for x in Result2]
    
    print("error_1:",[x.item() for x in error_1])
    print("error_2:",[x.item() for x in error_2])
    print("proj_error_1:",[x.item() for x in proj_error_1]) 
    print("proj_error_2:",[x.item() for x in proj_error_2])
    # print(obj_2_avevec)
    # print(obj_2)
    # print("Obj1")
    # exhi(obj1) 
    # print("Obj2")
    # exhi(obj_2_avevec)
    # print(Result1), print(Result2)
    # return obj_2_avevec


#test
ar_coeff = [0.98, 0.95, 0.98, 0.95];a1=2; a2=3; initial_value = 0; n_gene = 10000; n_obs = 100; H = 5; S=10; n_rep = 100
Test1(ar_coeff, a1, a2, initial_value, n_gene, n_obs, H, S, n_rep)