import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_l_bfgs_b


# MODULE FOR SOFTMAX REGRESSION
def groundTruth(y, numLabels):
    '''Expanded Y Matrix, I.e. label: 3 => [0,0,1,0,0,0,0,0,0,0]'''
    m = y.shape[0]
    groundTruth = np.zeros((m, numLabels)).astype('float64')
    for row in range(m):
        col = y[row,0] # need to push all so index 0!
        groundTruth[row,col-1] = 1
    return groundTruth

def predict(x, thetas):
    x = np.atleast_2d(x)
    return h(thetas, x)

# COST & GRADIENT FUNCTIONS
def h(thetas, x):
    '''hypothesys f() assumes x is at least 2d array, and thetas well formed'''
    H = thetas.dot(x.T)
    H = H - np.amax(H, axis=0)
    H = np.exp(H).T
    suma = np.sum(H, axis=1).reshape((-1,1))
    H = np.divide(H, suma)
    return H

def j(thetas, x, groundTruth, numLabels, regul_lambda):
    '''cost function'''
    thetas = thetas.reshape(numLabels, -1)
    m = x.shape[0]
    hx = h(thetas, x)
    b = groundTruth*(np.log(hx))
    lambdaEffect = (regul_lambda/2)*np.sum(np.sum(thetas**2))
    J = -np.sum(np.sum(b))/m + lambdaEffect
    return J

def v(thetas, x, groundTruth, numLabels, regul_lambda):
    '''gradient function, first partial derivation of j(theta)'''
    thetas = thetas.reshape(numLabels, -1)
    hx = h(thetas, x)
    m = x.shape[0]
    grad = ((groundTruth-hx).T.dot(x))/(-m) + (regul_lambda*thetas)
    grad = grad.flatten(0)
    return grad

def grad_by_hand(thetas, x, groundTruth, numLabels, regul_lambda):
    epsilon = 1e-4
    t = thetas.flatten(0)
    n = t.size
    grad = np.ones((n,)).astype('float64')
    for i in range(n):
        t1 = np.copy(t)
        t2 = np.copy(t)
        t1[i] += epsilon
        t2[i] -= epsilon
        a = j(t1, x, groundTruth, numLabels, regul_lambda)
        b = j(t2, x, groundTruth, numLabels, regul_lambda)
        grad[i] = (a-b)/(2*epsilon)
    return grad

def optimizeThetas(tinit, X_train, GT, numLabels, l):
    '''derive thetas from training data (tr) using bfgs algorithm'''
    def f(w):
        return j(w, X_train, GT, numLabels, l)
    def fprime(w):
        return v(w, X_train, GT, numLabels, l)

    [thetas, f, d] = fmin_l_bfgs_b(func=f, x0=tinit, fprime=fprime, maxiter=400)
    print thetas[0:10]
    print f
    print d
    return thetas
    