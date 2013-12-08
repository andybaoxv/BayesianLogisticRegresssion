import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

# Generate samples of class 0
mean_0 = np.array([0,0,0,0])
cov_0 = np.eye(4)
n_samples_0 = 10000
data_0 = np.random.multivariate_normal(mean_0,cov_0,n_samples_0)
label_0 = np.zeros((n_samples_0,1))
#plt.scatter(data_0[:,0],data_0[:,1],c='b')

# Generate samples of class 1
mean_1 = np.array([1,2,3,4])
cov_1 = np.eye(4)
n_samples_1 = 10000
data_1 = np.random.multivariate_normal(mean_1,cov_1,n_samples_1)
label_1 = np.ones((n_samples_1,1))
#plt.scatter(data_1[:,0],data_1[:,1],c='r')

# Merge two classes to obtain the dataset
data = np.vstack((data_0,data_1))
label = np.vstack((label_0,label_1))

# Save this synthetic data
file_data = open("synthetic_1.pkl","wb")
pickle.dump([data,label],file_data)
file_data.close()

# Apply Logistic Regression
#clf_1 = LogisticRegression(penalty='l2',dual=False,tol=0.0001,C=1.0)
#clf_1.fit(data,label)
#print clf_1.coef_,clf_1.intercept_


