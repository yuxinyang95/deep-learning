# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:05:37 2020

@author: Yang
"""
import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1/(1+ np.exp(-x)) 


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))
# loss funciton

def mseloss(y,t):
    return 0.5 * np.sum((y- t)**2)

def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[(np.arange(batch_size).astype(int)), t] + 1e-7)) / batch_size


                    
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

# for all dimensions
def numerical_gradient(f, x):
    h = 1e-4 
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) 
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val 
        it.iternext()   
        
    return grad
init_x = np.array([-3.0,4.0])

## Calculate each gradiant by $\lim_{x\to 0}f((x+h)-f(x-h))/(2h)$
## The idea of gradiant is very similar to Newton-Raphson method. Estimation
## $x_1 = x_0 - f(x_0)/(\frac{\partial f}{\partial x})$

def gradiant_descent(f,init_x,lr = 0.01, step = 100):
    x = init_x
    for i in range(step):
        grad = numerical_gradiant(f,x)
        x -= lr * grad
        
    return x 

def test_function_1(x):
    return x[0]**2 + x[1]**2


print(gradiant_descent(test_function_1,init_x,lr = 0.1, step = 100))

# Try our gradient method on the weighted matrix, create our first simple network

class simplenet:
    def __init__(self):
        self.W = np.random.rand(2,3)
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy(y,t)
        return loss
net = simplenet()
print(net.W)
x = np.array([0.6,0.9])
t = np.array([0,0,1])
p = net.predict(x)

f = lambda w: net.loss(x,t)
dW = numerical_gradient(f,net.W)
print(net.W)
print(dW)
print(x_batch)

# Simple Network function with two layers (one hidden layer)

class Twolayer:
    def __init__(self, input_size,hidden_size,output_size,weight_int_std = 0.01):
        self.params  = {}
        #initialize the weight and bias
        self.params['W1'] = weight_int_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_int_std * np.random.randn(hidden_size,output_size) 
        self.params["b2"] = np.zeros(output_size)
        
    def predict(self,x):
        W1, W2 = self.params['W1'],self.params['W2']
        b1, b2 = self.params['b1'],self.params['b2']
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = sigmoid(a2)
        y = softmax(z2)
        return y
    
    def loss(self,x,t):
        y = self.predict(x)
        
        return cross_entropy(y, t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis = 1)
        t = np.argmax(t, axis = 1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W:self.loss(x,t)
        grads = {}
        self.grads["W1"] = numerical_gradient(loss_W,self.params['W1'])
        self.grads["b1"] = numerical_gradient(loss_W,self.params['b1'])
        self.grads["W2"] = numerical_gradient(loss_W,self.params['W2'])
        self.grads["b2"] = numerical_gradient(loss_W,self.params['b2'])
        return grads
    
     def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        return grads

# test by the random dataset       
twolayer = Twolayer(input_size = 784, hidden_size = 100, output_size = 10)
x = np.random.rand(100,784)
t = np.random.rand(100,10)
print(twolayer.accuracy(x,t))