"""This script implement Stochastic Variational Inference for Bayesian Logistic
Regression
"""
print __doc__

import numpy as np
import pickle
import copy
from time import time
from python.COPDGene.utils.sample_wr import sample_wr
import math

# Load Synthetic data
file_data = open("synthetic_1.pkl","rb")
data,label = pickle.load(file_data)
file_data.close()

# Data preprocessing
data = np.hstack((np.ones((data.shape[0],1)),data))

# Obtain the number of samples and features
n_instances,n_features = data.shape

# Apply the Variational Logistic Regression algorithm in Bishop'book.
# Page 498, 10.6

# Step 1: Initialize lambda_0 and lambda_1, whicha represent the parameters of
# posterior for w
m_0 = np.zeros((n_features,1))
S_0 = np.eye(n_features)
S_0_inv = np.linalg.inv(S_0)
lambda_0 = np.zeros((n_features,1))
lambda_1 = -0.5*np.eye(n_features)

# Step 2: Set step size p_t = 1/t
# NOTE that here we can use a more flexible form, as is described in Blei's
# paper

# Step 3: Iteration
n_iter = 0
max_iter = 1e6
tol = 0.0001
flag_ratio = np.infty

t1 = time()
while n_iter<max_iter and flag_ratio>tol:
    n_iter += 1
    
    # sample a data point x_n uniformly from the dataset
    index_row = sample_wr(range(n_instances),1)
    x_n = data[index_row,:].T

    # Derive S_N and m_N from global parameters(Just different representation)
    S_N = -0.5*np.linalg.inv(lambda_1)
    m_N = np.dot(S_N,lambda_0)

    # Update local variables
    tp = S_N+m_N.reshape(n_features,1)*m_N.reshape(1,n_features)
    ks = np.dot(np.dot(x_n.T,tp),x_n)
    var_local_n = np.sqrt(ks)

    # Compute the intermediate global parameters as though x_n is replicated N
    # times
    tp_lambda_0 = np.dot(S_0_inv,m_0)+n_instances*(label[index_row[0]]-0.5)*x_n
    tp = 0.5/var_local_n*(1./(1+math.exp(-var_local_n))-0.5)
    tp_lambda_1 = -0.5*S_0_inv-n_instances*tp*x_n*x_n.T
    
    # Store the old values of lambda_0 and lambda_1
    lambda_0_old = copy.copy(lambda_0)
    lambda_1_old = copy.copy(lambda_1)
    
    # Set updated rate
    rate = 1./(n_iter)
    
    # Update the current estimate of the global variational parameters
    lambda_0 = (1.-rate)*lambda_0 + rate*tp_lambda_0
    lambda_1 = (1.-rate)*lambda_1 + rate*tp_lambda_1
    
    flag_ratio = np.linalg.norm(lambda_1-lambda_1_old)/\
            np.linalg.norm(lambda_1_old)
t2 = time()
print m_N
print n_iter
print "RunningTime",t2-t1
