# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets 2.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters 3.Train your model -Fit model to training data -Calculate mean salary value for each subset 4.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance 5.Tune hyperparameters -Experiment with different hyperparameters to improve performance 6.Deploy your model Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRASANNA A
RegisterNumber: 212223220078
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:

Initial dataset:

![323634513-2c16f00b-40e7-42a6-8e27-640023c732ea](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/955f5b37-4cdf-4dd9-8cac-ff9b4a811979)

Data Info:

![323634718-6b8685a3-0601-4b7b-8d7f-f5673e793bb7](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/bb0006e5-a794-48aa-9613-9b7fd893e520)

Optimization of null values:

![323634815-639cbd4c-7f0e-4a1c-b521-ad41039e751e](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/9668cb97-4421-4223-86e3-b603fe9e4343)

Converting string literals to numericl values using label encoder:

![323634932-ba9bb4eb-2825-4d93-9106-a5687337daa6](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/3aaf6e9c-a14f-49d4-9b18-d77539aa6f68)

Assigning x and y values:

![323635054-b5fece47-8a7b-465f-800a-ef67e0756cdb](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/41f0fde4-8dd2-46bb-af31-ac28a1ecac1c)

Mean Squared error:

![323635238-793178e7-37a0-4bc5-b3c5-c5a23942292d](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/4fe8837d-c973-445c-8827-6afbdbc86141)

R2 (Varience):

![323635348-a2fa15b1-5c55-4225-b30c-2b8868f96dad](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/dc521da4-ad94-4113-b06c-a5f5489b076e)

Prediction:

![323635664-4c93ad8d-8c51-4c94-bc05-e591bcb6447f](https://github.com/RamkumarGunasekaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870820/adc091da-5d31-4826-945e-2c10976ae3a8)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
