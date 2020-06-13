# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:05:37 2020

@author: Yang
"""
import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# loss funciton

def mseloss(y,t):
    return 0.5 * np.sum((y- t)**2)

def cross_entropy(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    error = 1e-5
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+error)) / batch_size

                    
test = [0.1,0.3,0.6]
tag =  [0,0,1]


print(mseloss(np.array(test),np.array(tag)))
print(cross_entropy(np.array(test),np.array(tag)))
## Both loss function increase when the result fitting bad

# Mini-batch   
(x_train, t_train), (x_test,t_test) = \
    load_mnist(normalize =True, one_hot_label = True)
    
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
t = t_batch.reshape(1,100)
t_batch = t_batch.astype(int)

print(cross_entropy(x_batch,t_batch))

# Why loss function? Observe slight change of weights, derivative,gradiant.
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/2

def test_function(x):
    return x[0]*x[1]
    
def numerical_gradiant(f,x):
    h= 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        temp_val = x[idx]
        x[idx] = temp_val + h
        fxh1 = f(x)
        
        x[idx] = temp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2) / (2*h)
        x[idx] = temp_val
    return grad


## Calculate each gradiant by $\lim_{x\to 0}f((x+h)-f(x-h))/(2h)$
## The idea of gradiant is very similar to Newton-Raphson method. Estimation
## $x_1 = x_0 - f(x_0)/(\frac{\partial f}{\partial x})$

print(numerical_gradiant(test_function,np.array([3.0,4.0])))
