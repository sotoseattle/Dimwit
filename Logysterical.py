import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_bfgs

#### Use decorator pattern to integrate cost and gradient in a single method
#### refactor further grad - regulam in a single line
#### avoid gradient calc in test or fix method
#### print l.predict(mapfeaturenp.array([1, 45, 85]))

class Logysterical(object):
    '''Logistic Regression with regularization and BFGS optimization'''
    
    def __init__(self, training_X, training_y, lam=0.0):
        self.lam = lam
        self.training_samples, n = training_X.shape
        self.training_features = n+1
        self.Xt = np.column_stack([np.ones((self.training_samples,1)), training_X])
        self.yt = training_y
        self.thetas = np.zeros((self.training_features, 1))
    
    def costFunction(self, theta):
        '''Computes cost and gradient for logistic regression'''
        prediction = Logysterical.sigmoid(np.dot(self.Xt, theta))
        J = (1.0/self.training_samples)*(-self.yt.T.dot(np.log(prediction)) - \
            (1-self.yt).T.dot(np.log(1-prediction))) +    \
            (self.lam / (2.0*self.training_samples)) * np.sum(np.power(theta[1::], 2))
        return J
    
    def gradientFunction(self, theta):
        prediction = Logysterical.sigmoid(np.dot(self.Xt, theta))
        regulam = theta.T*(self.lam/self.training_samples)
        regulam[0] = 0
        grad = (1.0/self.training_samples)*((prediction - self.yt).dot(self.Xt)) + regulam
        return grad
    
    def optimizeThetas(self, theta0):
        self.thetas = fmin_bfgs(f=self.costFunction, x0=theta0, \
              fprime=self.gradientFunction, maxiter=4000).flatten()
        return self.thetas
    
    def predict(self, x):
        return Logysterical.sigmoid(x.dot(self.thetas))
    
    def accuracy(self, x, y):
        acc = 0.0
        for i in range(self.training_samples):
            p = self.predict(x[i,::])
            if (p>0.5 and y[i]==1) or (p<=0.5 and y[i]==0):
                    acc +=1
        return acc/self.training_samples
        
    
    @classmethod
    def sigmoid(cls, z):
        '''computes dot sigmoid of a np.array as vector or matrix'''
        return np.vectorize(lambda x: 1/(1+np.exp(-x) + 1E-11))(z)
    
    @classmethod
    def mapfeature(cls, X1, X2):
        '''maps two input features to quadratic polinomiasl. 
        X1,X2 => X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..'''
        Xt = np.zeros((X1.shape[0],1))
        degree = 6
        for i in range(1,degree+1):
            for j in range(0,i+1):
                Xt = np.column_stack([Xt, np.power(X1, i-j) * np.power(X2, j)])
        return Xt[:,1::] # first column not needed
    





#data = pd.read_csv('./tests/log_data_basic.txt')
data = pd.read_csv('./tests/log_data_reg.txt')
X = np.array(data.ix[:, 0:2])
y = np.array(data.ix[:, 2])
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


X1 = np.array(X[:,0])
X2 = np.array(X[:,1])
Xext = Logysterical.mapfeature(X1, X2)

l = Logysterical(Xext, y, 1.0)
ini_thetas = np.zeros((l.training_features, 1))
#print l.gradientFunction(ini_thetas)

print l.optimizeThetas(ini_thetas)
print l.accuracy(l.Xt, l.yt)










