#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as nm 
from ydata_profiling import ProfileReport 
# import pdfkit
# from weasyprint import HTML

#Importing data set 
df= pd.read_csv('E:/Python Practice/Pandas Practice/StudentsPerformance.csv')
# print('The real shape of the data set is :',df.shape)

#Making the data set
# df.columns
new_df= df.loc[1:501, ['gender','math score', 'reading score','writing score']]
X=df.iloc[:,[-1,-2,-3]]
y=df.iloc[:,-4]

#Ydata profiling the 
re = ProfileReport(new_df)
html='E:\Python Practice\Pandas Practice/Student_Data_Profiling.html'
# re.to_file(html)

#Split for Training the data set
from sklearn.model_selection import train_test_split
X_train,X_sample,y_train,y_sample = train_test_split(X,y,test_size=0.2, train_size=0.8 )

#Scaling the data 
from sklearn.preprocessing import StandardScaler
pp = StandardScaler()
pp.fit(X_train)

X_train_scaled = pp.transform(X_train)
X_test_scaled = pp.transform(X_sample)

# print(pp.mean_)
# print(X_train_scaled)

#Transforming the to data frame 
X_train_scaled1 = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled1 = pd.DataFrame(X_test_scaled,columns=X_sample.columns)

#Numpy use 
# nm.round(X_train_scaled1.describe())

# Train the data set
from sklearn.linear_model import LogisticRegression
fun = LogisticRegression()
fun.fit(X_train_scaled1, y_train)
#Prediction 
prediction = fun.predict(X_test_scaled1)
print(prediction)
#Accuracy test 
from sklearn.metrics import accuracy_score
print('The logistic regression model answer:',accuracy_score(y_sample, prediction))


#Dummy classifier
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_scaled1, y_train)
baseline_score = baseline.score(X_test_scaled1, y_sample)
print(f"Baseline accuracy: {baseline_score}")


#Decision tree
from sklearn.tree import DecisionTreeClassifier
fun1 = DecisionTreeClassifier()
fun1.fit(X_train_scaled1,y_train)
#Prediction
prediction1= fun1.predict(X_test_scaled1)
#Accuracy
print('The decision tree accuracy :',accuracy_score(y_sample, prediction1))





