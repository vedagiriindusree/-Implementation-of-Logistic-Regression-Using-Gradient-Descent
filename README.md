# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input
## Program:
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
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
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
```
print(y_pred)
```
```
print(Y)
```
```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
print("Name: Vedagiri Indu Sree")
print("Reg no:212223230236")
```
## Output:
## Dataset
![image](https://github.com/user-attachments/assets/e5142b76-e6a1-4998-a023-c282d02e5a03)

## dtypes
![image](https://github.com/user-attachments/assets/4c88ae9f-814a-408d-a92f-a82b83f803a7)

## dataset
![image](https://github.com/user-attachments/assets/dfef9844-56d7-4d90-adb5-2b0b4e019f55)

## y array
![image](https://github.com/user-attachments/assets/0f32facb-3108-451e-9fbe-dac2fc393189)

## Accuracy
![image](https://github.com/user-attachments/assets/31e6f997-35d2-4f59-bc18-202ba68db6fc)

## y_pred
![image](https://github.com/user-attachments/assets/e6a43de7-ca02-488c-9112-eb9f4a94d9c3)

## y
![image](https://github.com/user-attachments/assets/747df3eb-3efb-4f39-9829-e20ca7aea0e9)

## y_prednew
![image](https://github.com/user-attachments/assets/8e6da2aa-20fd-4081-aa7d-b33ef6011621)

## y_prednew
![image](https://github.com/user-attachments/assets/035c661b-eae1-4f59-83a4-14c02b1d9948)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

