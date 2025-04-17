# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.
## Program // Output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vedagiri Indu Sree
RegisterNumber: 212223230236 
*/
```
```
import pandas as pd
import numpy as np
dataset=pd.read_csv("Placement_Data.csv")
dataset
```
![image](https://github.com/user-attachments/assets/83ac2116-7b61-406b-804c-08c5c0b738be)
```
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/09bd506c-9e91-4e26-b665-9c3cc238d539)
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/ea820ff8-c786-4358-86c7-ceb6c9dbd929)
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
![image](https://github.com/user-attachments/assets/d1edd8c2-2602-4956-9313-1cc8c7c1702c)
```
theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
```
![image](https://github.com/user-attachments/assets/c7a7edeb-ab8a-4b48-9cb9-0286ac6b98cf)
```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/84d66837-42eb-48c9-b82d-c132c5733481)

```
print(Y)
```
![image](https://github.com/user-attachments/assets/40c722e0-2725-462d-86cf-3422ed5ea9e3)
```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/8d2f586c-cc47-49fa-87dc-73c067e02e3c)
```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/27008605-ab87-4b51-b3ea-e1a55192925a)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

