# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:45:47 2025

@author: USER
"""

#Titanic data set work 
# Making a analysis between age,pclass,sex and survival rate 

#Imporing the libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sn
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler,LabelEncoder

#Importing data set 
df = pd.read_csv('E:/Python Practice/Data Sets/tested.csv')
f1 = ProfileReport(df)
html = 'E:\Python Practice\Report Overview\Titanic.html'
# f1.to_file(html)

# About the dataset
col = df.columns
print(col)
print(df['Pclass'].unique())

# Defining concern column
af = df[['Age']]
pf= df[['Pclass']]
sf = df[['Sex']]

# Concern column checking
print(af.isnull().sum())

# Filling the gap with mean 
n_af  = pd.DataFrame(af.fillna(0))
# print(n_af.isnull().sum())

# Merging the columns
X_unscaled = pd.concat([n_af,sf,pf],axis=1)

# Dependent Variable
Y= df['Survived']

# Train test split
from sklearn.model_selection import train_test_split
X_un_train,X_un_test,Y_train,Y_test = train_test_split(X_unscaled,Y,test_size=0.3,train_size=0.7,random_state=39)

# Scaling the age and Encoding the Sex 
f2=StandardScaler()
f3 = LabelEncoder()

X_a_train = f2.fit_transform(X_un_train[['Age']])
X_a_test = f2.transform(X_un_test[['Age']])

X_s_train = f3.fit_transform(X_un_train['Sex'])
X_s_test = f3.transform(X_un_test['Sex'])

# Combining the column
X_train = pd.DataFrame({
    'Scaled_Age':X_a_train.flatten(),
    'Labeled_Sex':X_s_train,
    'Pclass':X_un_train['Pclass'].values
    })

X_test = pd.DataFrame({
    'Scaled_Age':X_a_test.flatten(),
    'Labeled_Sex':X_s_test,
    'Pclass':X_un_test['Pclass'].values
    })

# Training the data set 
from sklearn.linear_model import LogisticRegression
f4=LogisticRegression()
f4.fit(X_train, Y_train)

# Prediction
predic =f4.predict(X_test)

# Accuracy Rate
from sklearn.metrics import accuracy_score
f5=accuracy_score(Y_test,predic)
print('Accuracy Rate: ',f5)



