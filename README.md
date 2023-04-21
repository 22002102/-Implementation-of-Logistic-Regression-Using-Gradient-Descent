# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 
 1.Use the standard libraries in python for finding linear regression.
 2.Set variables for assigning dataset values.
 3.Import linear regression from sklearn.
 4.Predict the values of array.
 5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
 6.Obtain the graph.

 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANJAY S
RegisterNumber:  212222230132
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
*/
```

## Output:

![logistic regression using gradient descent](sam.png)





![9](https://user-images.githubusercontent.com/119091638/233582005-df67a3bb-2858-4d2f-bc7f-87961aa06dd0.png)


![8](https://user-images.githubusercontent.com/119091638/233582018-661d7b33-c8f2-45b2-8e09-747416dc6ba9.png)



![7](https://user-images.githubusercontent.com/119091638/233582048-b8d7f90a-c7fe-4b95-914f-f5f7402db97d.png)



![6](https://user-images.githubusercontent.com/119091638/233582079-b0943e30-c26b-4ee7-8d93-6c57f6bd13a3.png)



![65](https://user-images.githubusercontent.com/119091638/233582100-cc19d7f9-5098-482f-b21d-2d93e55182c2.png)


![64](https://user-images.githubusercontent.com/119091638/233582146-b08d7d0c-d057-407c-8a8b-07d6ce44c6fa.png)

![image](https://user-images.githubusercontent.com/119091638/233583099-a9d2c786-b309-4c5d-823f-68dfa4b08c44.png)

![image](https://user-images.githubusercontent.com/119091638/233583637-3c3321e7-55ba-4c54-b1ee-68e567618e7e.png)


![image](https://user-images.githubusercontent.com/119091638/233583836-f0176c08-22f9-4dd0-879f-dace6af9ad6c.png)


![image](https://user-images.githubusercontent.com/119091638/233585133-d0416985-066b-4a55-9093-8a4a0e657bec.png)










## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

