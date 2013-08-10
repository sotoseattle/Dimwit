import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_bfgs


class Logysoft(object):
    '''Softmax Logistic Regression with regularization and BFGS optimization'''
    
    def __init__(self, train_data={}, thetas=None):
        self.tr = {}
        if 'X' in train_data:
            self.tr['m'], n = train_data['X'].shape
            self.tr['n'] = n+1
            self.tr['X'] = np.column_stack([np.ones((self.tr['m'],1)), train_data['X']])
            self.tr['y'] = train_data['y']
            self.tr['lam'] = (train_data['lam'] if 'lam' in train_data else 0.0)
        self.thetas = (thetas if thetas!=None else np.zeros((self.tr['n'], 1)))
    
    def hypothesis(self, x, t=None):
        thetas = self.thetas if t==None else t
        z = np.dot(thetas, x)
        e = np.exp(z - np.amax(z))
        return e/sum(e)
    




