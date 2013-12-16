import numpy as np
from matplotlib.pyplot import hist as hist
from sklearn import preprocessing
import pickle

# Read Dataset
data_headers=['age','workclass','fnlwgt','education','education_num',\
        'marital_status','occupation','relationship','race','sex',\
        'capital_gain','capital_loss','hours_per_week','native_country',\
        'salary']

data = np.array(np.genfromtxt('data/adult.data.txt', dtype=(int, 'S32',\
        int,'S32',int,'S32','S32','S32','S32','S32',int,int,int,'S32','S32'),\
        delimiter=',',autostrip=True,names=data_headers))
"""
data_test = np.array(np.genfromtxt('data/adult.test.txt', dtype=(int, 'S32',\
        int,'S32',int,'S32','S32','S32','S32','S32',int,int,int,'S32','S32'),\
        delimiter=',',autostrip=True,names=data_headers))
"""

# Handling Missing Values
# Delete all rows which contain '?' or missing values
row_idx_to_delete = []
for i in range(0,len(data)):
    if "?" in data[i]:
        row_idx_to_delete.append(i)
print len(row_idx_to_delete)," records have incomplete data and will be deleted"
data = np.delete(data,row_idx_to_delete)

# Feature 1: 'age' are integers

# Feature 2: convert 'workclass' to integers
workclass_attributes=["Private", "Self-emp-not-inc", "Self-emp-inc",\
        "Federal-gov","Local-gov", "State-gov", "Without-pay","Never-worked"]
idx_workclass={}
for workclass in workclass_attributes:
    idx_workclass[workclass]=data["workclass"]==workclass
for i in range(0,len(workclass_attributes)):
    data["workclass"][idx_workclass[workclass_attributes[i]]] = int(i)

# Feature 3: Fnlwgt
idx_wgt_0=data['fnlwgt']<=105702
idx_wgt_1=(data['fnlwgt']>=105702) & (data['fnlwgt']<=289569) 
idx_wgt_2=data['fnlwgt']>=289569
data['fnlwgt'][idx_wgt_0]=0
data['fnlwgt'][idx_wgt_1]=1
data['fnlwgt'][idx_wgt_2]=2

# Feature 4: education
education_attributes=["Bachelors", "Some-college", "11th", "HS-grad", \
        "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",\
        "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
idx_education={}
for education in education_attributes:
    idx_education[education]=data["education"]==education
for i in range(0,len(education_attributes)):
    data["education"][idx_education[education_attributes[i]]]=int(i)

# Feature 5: education_num
a=hist(data['education_num'],4)
means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
idx_enum_0=data['education_num']<=means[0]
idx_enum_1=(data['education_num']>means[0]) & (data['education_num']<=means[1]) 
idx_enum_2=(data['education_num']>=means[1]) & (data['education_num']<=means[2])
idx_enum_3=data['education_num']>means[2]
data['education_num'][idx_enum_0]=0
data['education_num'][idx_enum_1]=1
data['education_num'][idx_enum_2]=2
data['education_num'][idx_enum_3]=3

# Feature 6: marital_status
marital_status_attributes=['Married-civ-spouse', 'Divorced', 'Never-married',\
        'Separated', 'Widowed','Married-spouse-absent','Married-AF-spouse']
idx_marital_status={}
for marital_status in marital_status_attributes:
    idx_marital_status[marital_status]=data["marital_status"]==marital_status
for i in range(0,len(marital_status_attributes)):
    data["marital_status"][idx_marital_status[marital_status_attributes[i]]]=int(i)

# Feature 7: occupation
occupation_attributes=['Tech-support', 'Craft-repair', 'Other-service', \
        'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',\
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', \
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', \
        'Armed-Forces']
idx_occupation={}
for occupation in occupation_attributes:
    idx_occupation[occupation]=data["occupation"]==occupation
for i in range(0,len(occupation_attributes)):
    data["occupation"][idx_occupation[occupation_attributes[i]]]=int(i)

# Feature 8: relationship
relationship_attributes=['Wife', 'Own-child', 'Husband', 'Not-in-family', \
        'Other-relative', 'Unmarried']
idx_relationship={}
for relationship in relationship_attributes:
    idx_relationship[relationship]=data["relationship"]==relationship
for i in range(0,len(relationship_attributes)):
    data["relationship"][idx_relationship[relationship_attributes[i]]]=int(i)

# Feature 9: race
race_attributes=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', \
        'Other', 'Black']
idx_race={}
for race in race_attributes:
    idx_race[race]=data["race"]==race
for i in range(0,len(race_attributes)):
    data["race"][idx_race[race_attributes[i]]]=int(i)

# Feature 10: sex
sex_attributes=['Female', 'Male']
idx_sex={}
for sex in sex_attributes:
    idx_sex[sex]=data["sex"]==sex
for i in range(0,len(sex_attributes)):
    data["sex"][idx_sex[sex_attributes[i]]]=int(i)

# Feature 11: capital_gain
a=hist(data['capital_gain'],2)
means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
idx_cap_gain_0=data['capital_gain']<=means[0]
idx_cap_gain_1=(data['capital_gain']>means[0]) & (data['capital_gain']<=means[1]) 
data['capital_gain'][idx_cap_gain_0]=0
data['capital_gain'][idx_cap_gain_1]=1

# Feature 12: capital_loss
a=hist(data['capital_loss'],2)
means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
idx_cap_loss_0=data['capital_loss']<=means[0]
idx_cap_loss_1=(data['capital_loss']>means[0]) & (data['capital_loss']<=means[1]) 
data['capital_loss'][idx_cap_loss_0]=0
data['capital_loss'][idx_cap_loss_1]=1

# Feature 13: hours_per_week
a=hist(data['hours_per_week'],4)
means=[(a[1][i]+a[1][i+1])/2.0 for i in range(0,len(a[1])-1)]
idx_hours_0=data['hours_per_week']<=means[0]
idx_hours_1=(data['hours_per_week']>means[0]) & (data['hours_per_week']<=means[1]) 
idx_hours_2=(data['hours_per_week']>=means[1]) & (data['hours_per_week']<=means[2])
idx_hours_3=data['hours_per_week']>means[2]
data['hours_per_week'][idx_hours_0]=0
data['hours_per_week'][idx_hours_1]=1
data['hours_per_week'][idx_hours_2]=2
data['hours_per_week'][idx_hours_3]=3

# Feature 14: native_country
native_country_attributes=['United-States', 'Cambodia', 'England', \
        'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',\
        'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', \
        'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',\
        'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', \
        'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', \
        'Guatemala','Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', \
        'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong','Holand-Netherlands']
idx_native_country={}
for native_country in native_country_attributes:
    idx_native_country[native_country]=data["native_country"]==native_country
for i in range(0,len(native_country_attributes)):
    data["native_country"][idx_native_country[native_country_attributes[i]]]=int(i)

# Label: salary
salary_attributes=['<=50K','>50K']
idx_salary={}
for salary in salary_attributes:
    idx_salary[salary]=data["salary"]==salary
for i in range(0,len(salary_attributes)):
    data["salary"][idx_salary[salary_attributes[i]]]=int(i)

# Save the filtered data after converting into numerical discrete quantities
# for each attribute
np.savetxt('adult_filtered.txt',data,fmt="%s",delimiter=',')

filtered_data=np.array(np.genfromtxt('adult_filtered.txt', delimiter=',',\
        autostrip=True))

scaled_filtered_data = preprocessing.scale(filtered_data)

salary=scaled_filtered_data[:,14]
scaled_filtered_data=scaled_filtered_data[:,:-1]

salary2=np.array([0]*len(salary))
for i in range(0,len(salary2)):
    if salary[i]>0:
        salary2[i]=1

# Save this real dataset
file_data = open("adult.pkl","wb")
pickle.dump([scaled_filtered_data,salary2],file_data)
file_data.close()
