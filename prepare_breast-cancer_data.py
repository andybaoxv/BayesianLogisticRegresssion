import csv
import numpy as np
from python.COPDGene.utils.is_number import is_number
import pickle

# Read data from the original dataset
filename_dataset = "breast_cancer_wisconsin_data.csv"
file_dataset = open(filename_dataset,"rb")
csvreader = csv.reader(file_dataset)
lines = [line for line in csvreader]
file_dataset.close()

# Store the data into an array
data_tmp = np.zeros((len(lines),len(lines[0])))
samples_per_feature = [0]*data_tmp.shape[1]
sums_per_feature = [0]*data_tmp.shape[1]
for j in range(data_tmp.shape[1]):
    for i in range(data_tmp.shape[0]):
        if is_number(lines[i][j]) != False:
            data_tmp[i,j] = float(lines[i][j])
            samples_per_feature[j] += 1
            sums_per_feature[j] += data_tmp[i,j]

# Apply imputation for the missing value using mean imputation
for j in range(data_tmp.shape[1]):
    if samples_per_feature[j] != data_tmp.shape[0]:
        for i in range(data_tmp.shape[0]):
            if is_number(lines[i][j]) == False:
                data_tmp[i,j] = sums_per_feature[j]*1./samples_per_feature[j]

# Store the result into a pickle file
file_pickle = open("breast_cancer_wisconsin_data.pkl","wb")
pickle.dump(data_tmp,file_pickle)
file_pickle.close()

