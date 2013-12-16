import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from time import time

# Load Synthetic data
file_data = open("adult.pkl","rb")
data,label = pickle.load(file_data)
file_data.close()

# Apply Logistic Regression with L2 penalty
# C is the inverse of regularization strength; must be a positive float, small
# values specify stronger regularization
C_reg = 0.9
t0 = time()
clf_1 = LogisticRegression(penalty='l2',dual=False,tol=0.0001,\
        C=C_reg,fit_intercept=True)
clf_1.fit(data,label)
print "Regularization||","Coefficients||","Intercept"
print C_reg,clf_1.coef_, clf_1.intercept_
t1 = time()
print "RunningTime",t1-t0

w_l2 = [clf_1.intercept_[0]]
for i in range(clf_1.coef_.shape[1]):
    w_l2.append(clf_1.coef_[0,i])

