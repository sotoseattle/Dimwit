import math
#import pandas as pd
import numpy as np
from scipy import optimize

class Logysterical(object):
    '''All related to Ligistic Regression'''
    
    def __init__(self, X, y):
        self.m = X.shape[0]
        self.X = np.column_stack([np.ones((self.m,1)), X]) # already has the ones!
        self.y = y
    
    def costFunction(self, theta):
        '''Computes cost and gradient for logistic regression'''
        prediction = Logysterical.sigmoid(np.sum(self.X*theta,axis=1))
        J = (1.0/self.m)*(-self.y.T.dot(np.log(prediction)) - \
            (1-self.y).T.dot(np.log(1-prediction)))
        return J
    
    @classmethod
    def sigmoid(cls, z):
        '''computes dot sigmoid of a np.array as vector or matrix'''
        return np.vectorize(lambda x: 1/(1+math.exp(-x)))(z)
    






##a = Logysterical()
##print Logysterical.sigmoid(0)
##print Logysterical.sigmoid([0, 1, 0.75, 4])
##print Logysterical.sigmoid([[0, 1, 0.75, 4],[-40, 40], 0])

#data = pd.read_csv('./tests/data_example.txt')
#l = Logysterical(data.ix[:, 0:2], data.ix[:, 2])
#theta = np.zeros_like(l.X)
#print l.costFunction(theta)


#
##y = np.array([2,2,2])
##x = np.array([3,2,1])
##theta = np.array([1,2,3])
##print Logysterical.costFunction(theta, x, y)