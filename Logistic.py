from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import pandas as pd
data=pd.read_csv('train_Processed.csv')
X,y=data,data['Survived']
X.drop(['Survived'],axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())

# Define hyperparameters to tune
param_grid = {
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__C': [0.1, 1.0, 10.0],
    'logisticregression__solver': ['liblinear', 'saga'],
    'logisticregression__max_iter': [100, 200, 300]
}

# Perform grid search to find the best hyperparameters
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
y_pred=grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Model accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)