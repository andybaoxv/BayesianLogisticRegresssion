import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load dataset
file_pickle = open("breast_cancer_wisconsin_data.pkl","rb")
data_tmp = pickle.load(file_pickle)
file_pickle.close()

# Extract data and label from the original dataset
case_ids = []
data = np.zeros((data_tmp.shape[0],data_tmp.shape[1]-2))
labels = []
for i in range(data_tmp.shape[0]):
    case_ids.append(data_tmp[i,0])
    data[i,:] = data_tmp[i,1:data_tmp.shape[1]-1]
    if data_tmp[i,-1] == 2.:
        labels.append(0)
    else:
        labels.append(1)

# Apply Bayesian logistic regression for the data
clf = LogisticRegression(penalty='l1',C=1.0,fit_intercept=True,\
        intercept_scaling=1)

clf.fit(data,labels)

