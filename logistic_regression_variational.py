import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from update_equations import update_global
import matplotlib.pyplot as plt
import copy
from time import time

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

# Initialize the parameters of prior for w
# w ~ N(m_0,S_0)
# lambda_0 = S_0^(-1)*m_0;  lambda_1 = -0.5*S_0^(-1)
m_0 = np.zeros((n_features,1))
S_0 = np.eye(n_features)
lambda_0 = np.zeros((n_features,1))
lambda_1 = -0.5*np.eye(n_features)

# Initialize the local parameters
var_local = [0]*n_instances
for n in range(n_instances):
    var_local[n] = np.linalg.norm(data[n,:])

n_iter = 0
max_iter = 10000
tol = 0.0001
flag_ratio = np.infty

# Start Iteration
t0 = time()
while n_iter<max_iter and flag_ratio>tol:
    n_iter += 1
    lambda_0_old = np.zeros((n_features,1))
    for i in range(n_features):
        lambda_0_old[i,0] = lambda_0_old[i,0]
    lambda_0,lambda_1 = update_global(m_0,S_0,label,data,var_local)
    S_N = -0.5*np.linalg.inv(lambda_1)
    m_N = np.dot(S_N,lambda_0)
    tp = S_N+m_N.reshape(n_features,1)*m_N.reshape(1,n_features)
    for n in range(n_instances):
        ks = np.dot(np.dot(data[n,:],tp),data[n,:])
        if ks >= 0:
            var_local[n] = np.sqrt(ks)
        else:
            var_local[n] = 0.1
    if np.linalg.norm(lambda_0_old) != 0:
        flag_ratio = np.linalg.norm(lambda_0-lambda_0_old)
    else:
        flag_ratio = np.infty
t1 = time()
print "RunningTime",(t1-t0)/60
print m_N


