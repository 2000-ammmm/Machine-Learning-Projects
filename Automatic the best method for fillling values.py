# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Impoting dataset
data = pd.read_csv(r'E:\Python Practice\Data Sets\train.csv')
df= data.drop(columns=['PassengerId','Name','Ticket','Cabin'])

print(df.columns)

# Train test split
X = df.drop(columns=['Survived'])
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=42)


print((X_train.head(5)))
print((X.isnull().sum()))


# Processing
col_numerical = ['Pclass','Age', 'SibSp', 'Parch', 'Fare']
pipe1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


col_categorical =['Sex', 'Embarked']
pipe2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder())])


transform = ColumnTransformer(transformers=[
    ('num', pipe1, col_numerical),
    ('cat', pipe2, col_categorical)])
                  
set_config(display='diagram')
print(transform)

X_train_processed = transform.fit_transform(X_train)
X_test_processed = transform.transform(X_test)    

# Logistic Regression Model
clf = LogisticRegression(random_state=42)
clf.fit(X_train_processed, Y_train)

# Predicting the test set results
Y_pred = clf.predict(X_test_processed)

# Evaluating the model
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(Y_test, Y_pred))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))
