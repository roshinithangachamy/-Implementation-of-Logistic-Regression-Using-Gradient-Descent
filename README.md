# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize parameters: Set weights and bias to small random values or zeros.
2. Compute the prediction
3. Calculate the cost
4. Update parameters 

## Program:
```
Developed by: T.Roshini
RegisterNumber: 212223230175
```
```
import pandas as pd
import numpy as np

dataset=pd.read_csv('Placement_Data.csv')
dataset
```
![367323740-5935aabf-6064-4d6b-b049-9cf7b7309c7f](https://github.com/user-attachments/assets/e6b43afd-795f-491d-a058-cd8c37d0b662)

```

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)



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
![367323983-ec9a5174-c9da-41a4-b2c4-f2f92ad06a4d](https://github.com/user-attachments/assets/44f93a46-198b-4a16-bbb7-343b016c47e6)

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
![367324151-ed08bc42-22ec-4f21-848f-acea3649aee6](https://github.com/user-attachments/assets/2a307ddd-75d7-4bcd-af8c-fa4a54035c30)

```
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
```
![367324290-c1d82e17-b5d1-4315-9715-1555b0e037e5](https://github.com/user-attachments/assets/fdb639ce-b88f-4523-af6b-0a7f90735fbe)

```
#initialize the model paramenters
theta=np.random.randn(X.shape[1])
y=Y

#define sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#define loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)) + (1-y)*np.log(1-h)

#define gradient descent algorithm
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta


#train model
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)


#make prediction
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

#evaluate model
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy);
```
![367324490-21cd3ee9-efd8-4ef6-bfa8-7c35238ded64](https://github.com/user-attachments/assets/2dff9dd5-d7f4-4831-8c0d-353fc449d972)

```
print(y_pred)
```
![367324689-5f266d7b-0518-4c60-a251-fe0b2f896c8b](https://github.com/user-attachments/assets/000f3f63-86c9-41ad-a982-2f1ff93b07db)

```
print(Y)
```
![367324854-cf690a6a-a102-45ac-8125-faa9f8ef10fd](https://github.com/user-attachments/assets/7349e894-b9ac-4db7-a2c3-12a7a8561b3f)

```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![367325080-91d44de8-414d-4122-b06d-3c2e958a91aa](https://github.com/user-attachments/assets/ffd1bf18-9f30-4a22-95fb-b741c0c9ce78)

```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![367325195-922a038d-0fce-4b9d-84e9-766b6bdf798b](https://github.com/user-attachments/assets/8123ca0c-4a6b-4e7d-9083-e87cc78b50d6)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

