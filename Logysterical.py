import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_l_bfgs_b


# MODULE FOR LOGISTIC REGRESSION

def sigmoid(z):
    return np.vectorize(lambda x: 1/(1+np.exp(-x) + 1E-11))(z)
    #return np.vectorize(lambda x: 1/(1+np.exp(-x)))(z)

def h(t, x):
    '''hypothesis function. probability of input x with params t'''
    return sigmoid(x.dot(t))

def j(t, x, y, lam):
    '''cost function J(theta)'''
    prediction = sigmoid(np.dot(x, t))
    m = x.shape[0]
    J = (-y.T.dot(np.log(prediction)) -        \
            (1-y).T.dot(np.log(1-prediction)) +    \
            (lam/2.0) * np.sum(np.power(t[1::], 2)))/m
    return J

def v(t, x, y, lam):
    '''gradient function, first partial derivation of J(theta)'''
    prediction = sigmoid(np.dot(x, t))
    regu = np.hstack([[0],t[1::,]*lam])
    grad = ((prediction - y).dot(x) + regu)/x.shape[0]
    return grad

def optimizeThetas(tinit, x, y, lam, visual=True):
    '''derive thetas using l_bfgs algorithm'''
    def f(w):
        return j(w, x, y, lam)
    def fprime(w):
        return v(w, x, y, lam)
    [thetas, f, d] = fmin_l_bfgs_b(func=f, x0=tinit, fprime=fprime, maxiter=400)
    if visual:
        print thetas[0:10]
        print f
        print d
    return thetas

def accuracy(t, x, y):
    acc = 0.0
    m= x.shape[0]
    for i in range(m):
        p = h(t, x[i,::])
        if (p>0.5 and y[i]==1) or (p<=0.5 and y[i]==0):
                acc += 1
    return acc/m
    
def mapfeature(X1, X2, degree=6):
    '''maps two input features to quadratic polinomiasl. 
    X1,X2 => X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..'''
    Xt = np.zeros((X1.shape[0],1))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            Xt = np.column_stack([Xt, np.power(X1, i-j) * np.power(X2, j)])
    return Xt[:,1::] # first column not needed
    

