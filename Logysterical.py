import math
import pandas as pd
import numpy as np
from scipy import optimize

class Logysterical(object):
    '''Logistic Regression with regularization and BFGS optimization'''
    
    def __init__(self, training_X, training_y, lam=0.0):
        self.lam = lam
        self.m, n = training_X.shape
        self.Xt = np.column_stack([np.ones((self.m,1)), training_X])
        self.yt = training_y
        self.thetas = np.zeros((n+1, 1))
    
    def costFunction(self, theta):
        '''Computes cost and gradient for logistic regression'''
        prediction = Logysterical.sigmoid(np.dot(self.Xt, theta))
        J = (1.0/self.m)*(-self.yt.T.dot(np.log(prediction)) - \
            (1-self.yt).T.dot(np.log(1-prediction))) +    \
            (self.lam / (2.0*self.m)) * np.sum(np.power(theta[1::], 2))
        return J
    
    def gradientFunction(self, theta):
        prediction = Logysterical.sigmoid(np.dot(self.Xt, theta))
        regulam = theta # REFACTOR
        regulam[0] = 0
        grad = (1.0/self.m)*((prediction - self.yt).dot(self.Xt)) + regulam*(self.lam/self.m)
        return grad
    
    def optimizeThetas(self, theta0):
        self.thetas = optimize.fmin_bfgs(f=self.costFunction, x0=theta0, \
              fprime=self.gradientFunction, maxiter=40000).flatten()
        return self.thetas
    
    def predict(self, x):
        return Logysterical.sigmoid(x.dot(self.thetas))
    
    def accuracy(self, x, y):
        acc = 0.0
        for i in range(self.m):
            p = self.predict(x[i,::])
            if (p>0.5 and y[i]==1) or (p<=0.5 and y[i]==0):
                    acc +=1
        return acc/self.m
        
    
    @classmethod
    def sigmoid(cls, z):
        '''computes dot sigmoid of a np.array as vector or matrix'''
        return np.vectorize(lambda x: 1/(1+np.exp(-x) + 1E-11))(z)
    

    

#data = pd.read_csv('./tests/log_data_basic.txt')
##data = pd.read_csv('./tests/log_data_reg.txt')
#X = np.array(data.ix[:, 0:2])
#y = np.array(data.ix[:, 2])
#l = Logysterical(X, y, 0)
#ini_thetas = np.array([0,0,0])
#print ini_thetas
#print l.optimizeThetas(ini_thetas)
#print "-"*40
#print l.thetas
#
#t0 = np.array([0,0,0])
#prob = l.predict(np.array([1.0, 45, 85]))
#print prob
#
#p = l.accuracy(l.Xt, l.yt)
#print p


