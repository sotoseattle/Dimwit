import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_bfgs


class Logysterical(object):
    '''Logistic Regression with regularization and BFGS optimization'''
    
    def __init__(self, train_data={}, thetas=None):
        self.tr = {}
        self.tr['m'], n = train_data['X'].shape
        self.tr['n'] = n+1
        self.tr['X'] = np.column_stack([np.ones((self.tr['m'],1)), train_data['X']])
        self.tr['y'] = train_data['y']
        self.tr['lam'] = (train_data['lam'] if 'lam' in train_data else 0.0)
        self.thetas = (thetas if thetas!=None else np.zeros((self.tr['n'], 1)))
    
    def costFunction(self, theta):
        prediction = Logysterical.sigmoid(np.dot(self.tr['X'], theta))
        J = (-self.tr['y'].T.dot(np.log(prediction)) -        \
            (1-self.tr['y']).T.dot(np.log(1-prediction)) +    \
            (self.tr['lam']/2.0) * np.sum(np.power(theta[1::], 2)))/self.tr['m']
        return J
    
    def gradientFunction(self, theta):
        prediction = Logysterical.sigmoid(np.dot(self.tr['X'], theta))
        regu = np.hstack([[0],theta[1::,]*(self.tr['lam'])])
        grad = ((prediction - self.tr['y']).dot(self.tr['X']) + regu)/self.tr['m']
        return grad
    
    def optimizeThetas(self, theta0):
        '''derive thetas from training data (tr) using bfgs algorithm'''
        self.thetas = fmin_bfgs(self.costFunction, theta0, \
            self.gradientFunction, maxiter=4000)
        return self.thetas
    
    def predict(self, x):
        '''probability of input x as vector using own thetas'''
        return Logysterical.sigmoid(x.dot(self.thetas))
    
    def accuracy(self, x, y):
        '''accuracy of set h(x) as measured against obsserved outcomes y'''
        acc = 0.0
        for i in range(self.tr['m']):
            p = self.predict(x[i,::])
            if (p>0.5 and y[i]==1) or (p<=0.5 and y[i]==0):
                    acc += 1
        return acc/self.tr['m']
        
    # CLASS METHODS ###########################################################
    
    @classmethod
    def sigmoid(cls, z):
        '''dot sigmoid of an np.array (vector or matrix)'''
        return np.vectorize(lambda x: 1/(1+np.exp(-x) + 1E-11))(z)
    
    @classmethod
    def mapfeature(cls, X1, X2, degree=6):
        '''maps two input features to quadratic polinomiasl. 
        X1,X2 => X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..'''
        Xt = np.zeros((X1.shape[0],1))
        for i in range(1,degree+1):
            for j in range(0,i+1):
                Xt = np.column_stack([Xt, np.power(X1, i-j) * np.power(X2, j)])
        return Xt[:,1::] # first column not needed
    
