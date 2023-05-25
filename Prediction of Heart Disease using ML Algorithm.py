#!/usr/bin/env python
# coding: utf-8

# #                                          Prediction of Heart Disease using the optimal risk factors 

# In[1]:


#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ## 1. Import the data

# In[2]:


#Read the dataset
df = pd.read_csv("Analysis_of_Heartdisease.csv")
df.head()


# In[3]:


#summary of the original dataframe
df.info()


# ## 2. Pre-processing

# In[4]:


#Drop the index column
df=df.drop(['index'], axis='columns')
#convert target variable into numerical value
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2 = df
df2.HeartDisease = le.fit_transform(df2.HeartDisease)
df2.head()


# In[5]:


#size of the dataframe
df2.shape


# In[6]:


df1=df2
# Remove outliers for BP column
q1 = df1['BP'].quantile(0.25)
q3 = df1['BP'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['BP'])
for i in df1['BP']:
    if i > Upper_tail or i < Lower_tail:
            df1['BP'] = df1['BP'].replace(i, med)
# Remove outliers for Cholesterol column
q1 = df1['Cholesterol'].quantile(0.25)
q3 = df1['Cholesterol'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['Cholesterol'])
for i in df1['Cholesterol']:
    if i > Upper_tail or i < Lower_tail:
            df1['Cholesterol'] = df1['Cholesterol'].replace(i, med)

# Remove outliers for Thallium column
q1 = df1['Thallium'].quantile(0.25)
q3 = df1['Thallium'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['Thallium'])
for i in df1['Thallium']:
    if i > Upper_tail or i < Lower_tail:
            df1['Thallium'] = df1['Thallium'].replace(i, med)

# Remove outliers for ST_depression column
q1 = df1['ST_depression'].quantile(0.25)
q3 = df1['ST_depression'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['ST_depression'])
for i in df1['ST_depression']:
    if i > Upper_tail or i < Lower_tail:
            df1['ST_depression'] = df1['ST_depression'].replace(i, med)
            
# Remove outliers for fbs column
q1 = df1['FBS_over_120'].quantile(0.25)
q3 = df1['FBS_over_120'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['FBS_over_120'])
for i in df1['FBS_over_120']:
    if i > Upper_tail or i < Lower_tail:
            df1['FBS_over_120'] = df1['FBS_over_120'].replace(i, med)
# Remove outliers for Chest_pain_type column
q1 = df1['Chest_pain_type'].quantile(0.25)
q3 = df1['Chest_pain_type'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['Chest_pain_type'])
for i in df1['Chest_pain_type']:
    if i > Upper_tail or i < Lower_tail:
            df1['Chest_pain_type'] = df1['Chest_pain_type'].replace(i, med)
# Remove outliers for Numberofvesselsfluro column
q1 = df1['Numberofvesselsfluro'].quantile(0.25)
q3 = df1['Numberofvesselsfluro'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['Numberofvesselsfluro'])
for i in df1['Numberofvesselsfluro']:
    if i > Upper_tail or i < Lower_tail:
            df1['Numberofvesselsfluro'] = df1['Numberofvesselsfluro'].replace(i, med)
# Remove outliers for Max_HR column
q1 = df1['Max_HR'].quantile(0.25)
q3 = df1['Max_HR'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(df1['Max_HR'])
for i in df1['Max_HR']:
    if i > Upper_tail or i < Lower_tail:
            df1['Max_HR'] = df1['Max_HR'].replace(i, med)
plt.figure(figsize=(16,16))
sns.boxplot (data= df1)
plt.title("Box Plot after outlier removal")


# In[7]:


cleaned_data= df1
cleaned_data.head()
cleaned_data.shape


# In[8]:


scaler = StandardScaler()
categorical_vars = ['Sex','Exercise_angina','Numberofvesselsfluro','Chest_pain_type','EKG_results','Slope_of_ST','Thallium']
continuous_vars = ["Age","BP","Cholesterol","Max_HR","ST_depression",'FBS_over_120']
# encoding the categorical columns
data1 = pd.get_dummies(cleaned_data, columns = categorical_vars)
X = data1.drop(['HeartDisease'],axis=1)
y = data1[['HeartDisease']]
# Scaling the continuous columns
data1[continuous_vars] = scaler.fit_transform(X[continuous_vars])
data1.head()


# ## 3. Exploratory Data Analysis

# In[9]:


fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(cleaned_data.corr(),annot= True, linewidth=0.5)
plt.title("Correlation between variables")


# In[10]:


sns.boxplot(data=cleaned_data, y="Age",x="HeartDisease")


# In[11]:


sns.boxplot(data=cleaned_data, y="BP",x="HeartDisease")


# In[12]:


sns.boxplot(data=cleaned_data, y="Cholesterol",x="HeartDisease")


# ## 4. Feature selection using Recursive feature elimination method using Random forest classifier

# In[13]:


# defining the features and target
X1 = data1.drop(['HeartDisease'],axis=1)
y1 = data1[['HeartDisease']]

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.30)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

rfe_method = RFE(RandomForestClassifier(n_estimators=10, random_state=10),
    n_features_to_select=9,step=2)

rfe_method.fit(X_train1, y_train1)
X_train1.columns[(rfe_method.get_support())] 


# 8 features would give 81% accuracy 
# 10 features would give 88% accuracy
# Hence 9 features are selected

# ## Comparison with other ML models

# In[26]:


#Training the model with best risk factors
X_features=X1[['Age', 'BP', 'Cholesterol', 'Max_HR', 'ST_depression',
       'Numberofvesselsfluro_0', 'Chest_pain_type_4', 'Thallium_3',
       'Thallium_7']]
Y_features= y1[['HeartDisease']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_features, test_size=0.30, random_state=30)


# In[31]:


## Support vector machine classifier
from sklearn.svm import SVC
svc=SVC(C=1.0) 
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred1=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred1)))


# In[33]:


#  Deciion Tree classifer 
from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred2 = clf.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred2)))


# In[37]:


get_ipython().system('pip install --user xgboost')

# XGBClassifier
import xgboost as xgb
model = xgb.XGBClassifier()
# Training the model
model.fit(X_train, y_train)
# Making predictions on the test set
y_pred3 = model.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred3)))


# ## 5. Logistic Regression Model

# In[16]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[17]:


#To predict success rate
y_test_pred = pd.DataFrame(model.predict(X_test), index = y_test.index, columns = ['Disease_prediction'])
results_rf = y_test.join(y_test_pred)

results_rf['Success'] = (results_rf['HeartDisease'] == results_rf['Disease_prediction']).astype(int)
results_rf


# In[18]:


#Print the otput using classification_report and confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_test_pred))
confusion_matrix(y_test, y_test_pred)


# In[19]:


#Determing the accuracy score
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_test_pred)
print ("Logistic testing accuracy is ", round(accuracy,2) * 100, "%")


# In[20]:


#To display the ROC-AUC curve
from sklearn.metrics import roc_auc_score,roc_curve,auc
fpr , tpr , thresholds   = roc_curve(y_test,y_test_pred)
    
fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111)
ax.plot(fpr,tpr,label   = ["Area under curve : ",auc(fpr,tpr)],linewidth=2,linestyle="dotted")
ax.plot([0,1],[0,1],linewidth=2,linestyle="dashed")
plt.legend(loc="best")
plt.title("ROC-CURVE & AREA UNDER CURVE")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


# For your refernce, the below accuracy is obtained for the dataset without selecting the best risk factors

# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_clean = cleaned_data.drop(['HeartDisease'],axis=1)
y_clean = cleaned_data[['HeartDisease']]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_clean, y_clean, test_size=0.30, random_state=30)

model = LogisticRegression()
model.fit(X_train2,y_train2)
y_test_pred1 = model.predict(X_test2)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test2,y_test_pred1))
confusion_matrix(y_test2, y_test_pred1)


# In[ ]:




