import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_bfgs


class Logysoft(object):
    '''Softmax Logistic Regression with regularization and BFGS optimization'''
    
    def __init__(self, train_data={}, thetas=None, numLabels=None):
        self.tr = {}
        self.tr['numLabels'] = numLabels
        if 'X' in train_data:
            self.tr['m'], n = train_data['X'].shape
            self.tr['n'] = n+1
            self.tr['X'] = np.column_stack([np.ones((self.tr['m'],1)), train_data['X']])
            self.tr['y'] = train_data['y']
            self.tr['lam'] = (train_data['lam'] if 'lam' in train_data else 0.0)
            self.tr['groundTruth'] = np.zeros((self.tr['numLabels'], self.tr['m']))
            for col, row in enumerate(self.tr['y']):
                self.tr['groundTruth'][row, col] = 1
        print '------>x', self.tr['X'].shape
        print '------>n', self.tr['n']
        print '------>m', self.tr['m']
        # BAD self.thetas = (thetas if thetas!=None else np.zeros((self.tr['n'], 1)))
        self.thetas = (thetas if thetas!=None else np.zeros((10, l.tr['n'])))
    
    def h(self, x, t=None):
        '''hypothesys function h(x; theta)'''
        
        #print '=> t', t.shape
        if t==None:
            thetas = self.thetas
        else:
            thetas = t.reshape((10, self.tr['n']))
        
        #print thetas.shape,'x',x.shape
        z = thetas.dot(x)
        
        #####
        #max = np.amax(z, axis=0)
        z = z - np.amax(z, axis=0)
        e = np.exp(z)
        #e = np.exp(z - np.amax(z))
        
        return e/sum(e)
    
    def j(self, specificThetas):
        '''cost function'''
        #print '.......start J'
        hx = self.h(np.transpose(self.tr['X']), specificThetas)
        b = self.tr['groundTruth'] * np.log(hx)
        J = -np.mean(np.sum(b))
        #print '.......end J'
        return J
    
    def v(self, specificThetas):
        '''gradient function, first partial derivation of j(theta)'''
        
        #print '.......start V'
        
        #((hx-y)*data')/m
        t = specificThetas
        x = np.transpose(self.tr['X'])
        hx = self.h(x, t)
        
        #print 'b', hx.shape, '-', self.tr['groundTruth'].shape
        
        #b = hx - self.tr['groundTruth']
        #print 'bbbb', b.shape
        #print 'x', tr['X'].shape
        
        #a = np.dot(b, self.tr['X'])
        #a = a/self.tr['m']
        #print 'a', a.shape
        grad = ((hx - self.tr['groundTruth']).dot(self.tr['X']))/self.tr['m']
        
        grad = grad.reshape(-1)
        #print '.......end V', grad.shape
        return grad
    
    
    def optimizeThetas(self, theta0):
        '''derive thetas from training data (tr) using bfgs algorithm'''
        self.thetas = fmin_bfgs(self.j, theta0, self.v, maxiter=10)
        return self.thetas


    

data = pd.read_csv('./examples/numberOCR/train.csv', header=1)

tr = {}
tr['X'] = data.ix[:, 1::]
tr['y'] = data.ix[:, 0]

t = 0.005* np.random.rand(10, 785)

l = Logysoft(tr, thetas = t,  numLabels = 10)


#print l.j(t)
#print l.v(t)
#print sol.shape
#print sol[5, 20:39]
#print sum(sum(sol))
#print l.tr['n']

#ini_thetas = np.zeros((10, l.tr['n']))
#ini_thetas = 0.005 * np.random.rand(10, l.tr['n'])
ini_thetas = 0.005 * np.random.rand(10 * l.tr['n'], 1);

print 'ini_thetas', ini_thetas.shape
opt_thetas = l.optimizeThetas(ini_thetas)
print opt_thetas
print opt_thetas[:, 1]

