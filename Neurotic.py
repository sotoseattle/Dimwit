import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_l_bfgs_b

# MODULE FOR NEURAL NETWORK


def groundTruth(y, numLabels):
    '''Expanded Y Matrix, I.e. label: 3 => [0,0,0,1,0,0,0,0,0,0]'''
    m = y.shape[0]
    groundTruth = np.zeros((m, numLabels))
    for row in range(m):
        groundTruth[row, y[row,0]] = 1
    return groundTruth

def sigmoid(z):
    return np.vectorize(lambda x: 1/(1+np.exp(-x) + 1E-11))(z)

def sigmoid_gradient(z):
    return (sigmoid(z)*(1-sigmoid(z)))

def unroll_thetas(n, layers, thetas_rolled):
    thetas = [None]*len(layers)
    start_cut = 0
    cols = n+1
    for i, rows in enumerate(layers):
        end_cut = start_cut + cols*rows
        thetas[i] = thetas_rolled[start_cut:end_cut].reshape(rows,cols)
        start_cut = end_cut
        cols = rows+1
    return thetas

def randomize_thetas(rows,cols):
    epsilon_init = 0.12;
    return np.random.rand(rows, cols+1) * 2 * epsilon_init - epsilon_init;

def grad_by_hand(t, layers, x, y, regul_lambda, number_of_thetitas):
    epsilon = 1e-4
    n = t.size
    grad = np.ones((n,)).astype('float64')
    for i in range(number_of_thetitas):
        t1 = np.copy(t)
        t2 = np.copy(t)
        t1[i] += epsilon
        t2[i] -= epsilon
        ja = j(t1, layers, x, y, regul_lambda)
        jb = j(t2, layers, x, y, regul_lambda)
        grad[i] = (ja-jb)/(2*epsilon)
    return grad

def feedforward(theta, input_v):
    '''one layer forward computation, asumes input_v has no bias'''
    m = input_v.shape[0]
    input_v = np.hstack([np.ones((m,1)),input_v])
    z = input_v.dot(theta.T)
    return sigmoid(z)

def h(thetas_rolled, layers, one_x):
    thetas = unroll_thetas(len(one_x), layers, thetas_rolled)
    a = one_x
    for t in thetas:
        a = sigmoid(np.hstack([1,a]) .dot(t.T))
    return a

def j(thetas_rolled, layers, x, y, lam):
    m,n = x.shape
    thetas = unroll_thetas(n, layers, thetas_rolled)
    y_bloat = groundTruth(np.array(y), 10)
    a, r = x, 0.
    for t in thetas:
        a = np.hstack([np.ones((m,1)),a])
        a = sigmoid(a.dot(t.T))
        r += np.sum(t[:,1:]**2)    
    J = (np.sum(-y_bloat*log(a) - (1-y_bloat)*log(1-a)) + (r*lam)/2)/m
    return J

def v(thetas_rolled, layers, x, y, lam):
    m,n = x.shape
    thetas = unroll_thetas(n, layers, thetas_rolled)
    num_thetas = len(layers)
    y_bloat = groundTruth(np.array(y), 10)
    
    # forward => fill lists of 'activations' and 'z'
    a, z = [x], [None]
    for i in range(num_thetas):
        a[-1] = np.hstack([np.ones((m,1)),a[-1]])
        z.append(a[-1].dot(thetas[i].T))
        a.append(sigmoid(z[-1]))
    
    # backwards => fill list of deltas
    d = [(a[-1] - y_bloat)]    
    for i in range(num_thetas, 1, -1):
        d.insert(0, (d[0].dot(thetas[i-1][:,1:])) * sigmoid_gradient(z[i-1]))
    
    # forward => compute gradient
    V = np.array([])
    for i in range(num_thetas):
        Vwip = (d[i].T.dot(a[i]) + (lam*thetas[i]))/m
        Vwip[:,0] -= (lam/m) * thetas[i][:,0]
        V = np.hstack([V, Vwip.flatten()])
    return V

def optimizeThetas(tinit, layers, x, y, lam, visual=True):
    def f(w):
        return j(w, layers, x, y, lam)
    def fprime(w):
        return v(w, layers, x, y, lam)
    
    [thetas, f, d] = fmin_l_bfgs_b(func=f, x0=tinit, fprime=fprime, maxiter=50)
    if visual:
        print thetas[0:10]
        print f
        print d
    return thetas



# Other utilities
#def checking_gradient():
#    ts = np.hstack([randomize_thetas(25,400).flatten(), randomize_thetas(10,25).flatten()])
#    ls = [25,10]
#    gradiente = v(ts, ls, x, y, 1.)
#
#    number_of_thetitas=10
#    ghand = grad_by_hand(ts, ls, x, y, 1., number_of_thetitas)
#    assert np.allclose(gradiente[0:number_of_thetitas], ghand[0:number_of_thetitas])
#
#tinit = np.hstack([randomize_thetas(25,400).flatten(), randomize_thetas(10,25).flatten()])
#layers = np.array([25,10])
#opt_thetas = optimizeThetas(tinit, layers, x, y, 0.0, visual=True)
