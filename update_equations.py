import numpy as np
import math

def update_global(m_0,S_0,label,data,var_local):
    """ Implementation of (10.157) in Bishop's book
    
    Parameters:
    -----------
    m_0: array, shape(n_features,1)
        mean of the prior on w
    
    S_0: array, shape(n_features,n_features)
        covariance of the prior on w
    
    label: array, shape(n_instances,1)
        labels for all samples
    
    data: array, shape(n_instances,n_features)
        dataset

    var_local: array, shape(n_instances,1)
        local variables. ks values

    Returns:
    --------
    lambda_0: array, shape(n_features,1)
        lambda_0 = S_0^(-1)*m_0+SUM(label[n]-0.5)*data[n,:].T
    
    lambda_1: array, shape(n_features,n_features)
        lambda_1 = 0.5*S_0^(-1)+SUM(l(var_local[n]))*data[n,:].T*data[n,:]
    """
    n_instances,n_features = data.shape
    
    # Update for lambda_0
    tmp_0 = np.zeros((n_features,1))
    for n in range(n_instances):
        tmp_0 += (label[n]-0.5)*data[n,:].reshape(n_features,1)
    lambda_0 = np.dot(np.linalg.inv(S_0),m_0)+tmp_0

    # Update for lambda_1
    tmp_1 = np.zeros((n_features,n_features))
    for n in range(n_instances):
        tp = 0.5/var_local[n]*(1./(1+math.exp(-var_local[n]))-0.5)
        tmp_1 += tp*(data[n,:].reshape(n_features,1)*\
                data[n,:].reshape(1,n_features))
    lambda_1 = -0.5*np.linalg.inv(S_0)-tmp_1
    
    return lambda_0,lambda_1


if __name__ == "__main__":
    m_0 = np.zeros((2,1))
    S_0 = np.eye(2)
    label = np.ones((10,1))
    data = np.random.rand(10,2)
    var_local = [1]*10
    lambda_0,lambda_1 = update_global(m_0,S_0,label,data,var_local)

