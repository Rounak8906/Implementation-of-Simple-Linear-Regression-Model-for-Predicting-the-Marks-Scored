# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rounak Singh
RegisterNumber: 212225240125 
*/
```
```py

Developed by:PRIYANGHA G 
RegisterNumber: 212223040157
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:

### Dataset
<img width="522" height="698" alt="image" src="https://github.com/user-attachments/assets/61ba7382-cf75-4008-8e31-2805ba75981b" />

### Head values
<img width="312" height="161" alt="image" src="https://github.com/user-attachments/assets/99169eb8-a783-437d-888c-0dc715e4d9d6" />

### Tail values
<img width="317" height="160" alt="image" src="https://github.com/user-attachments/assets/fd991cad-5a34-41f1-b772-efdcf31e4294" />


### X and Y values
<img width="880" height="721" alt="image" src="https://github.com/user-attachments/assets/d66bbf6b-86c8-4e89-8981-7b876f69ee2c" />

### Predication values of X and Y
<img width="886" height="96" alt="image" src="https://github.com/user-attachments/assets/5e8b40a2-1af7-4c9c-80eb-52a42a835566" />

### TRAINING SET
<img width="966" height="685" alt="image" src="https://github.com/user-attachments/assets/4cf62f5e-590c-4341-b5a3-a99df8ef0fe4" />

### TESTING SET AND MSE,MAE and RMSE

<img width="1021" height="747" alt="image" src="https://github.com/user-attachments/assets/a6b91d4e-1037-4834-bc49-e2b1adcc3954" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
