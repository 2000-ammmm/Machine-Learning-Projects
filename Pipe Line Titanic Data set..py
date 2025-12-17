# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import set_config
from sklearn.metrics import accuracy_score

# Importing the data set
df= pd.read_csv('E:/Python Practice/Data Sets/tested.csv')

# Droping 
print(df.columns)
df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace = True)

# Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df.drop(['Survived'],axis=1),
                                                 df['Survived'],
                                               test_size=0.2,
                                                 random_state=42)

# Imputational Column Transformer
t1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_emb',SimpleImputer(strategy='most_frequent'),[-1])
    ],remainder='passthrough')


t2=ColumnTransformer([
    ('ohe_sex_emb',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])
    ],remainder='passthrough')

t3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
    ])

# Feature Selection
t4 = SelectKBest(score_func=chi2,k=5)

# Classification
t5 =DecisionTreeClassifier()

# Making Pipe Line
pipe=Pipeline([
    ('t1',t1),
    ('t2',t2),
    ('t3',t3),
    ('t4',t4),
    ('t5',t5)
    ])

pipe.fit(X_train,Y_train)
set_config(display='diagram')

# Prediction
y_pred= pipe.predict(X_test)

# Accuracy
accy = accuracy_score(Y_test,y_pred)
print('Accuracy:',accy)

