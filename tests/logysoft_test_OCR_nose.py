import pandas as pd
import numpy as np
from Logysoft import * 
from nose import with_setup

epsilon = 0.000001

def setup_func_basic():
    global l
    data = pd.read_csv('./examples/numberOCR/train.csv', header=1)
    
    #t = np.array(data.ix[:,:])
    #l = Logysoft(thetas = t)
    
    tr = {}
    tr['X'] = data.ix[:, 1:785]
    tr['y'] = data.ix[:, 0]
    
    t = np.random.rand(10, 784)
    
    l = Logysoft(tr, thetas = t)
    
    
    x = np.array(tr['X'].ix[100:100,:])
    x = np.transpose(x)
    
    sol = l.h(x)
    np.argmax(sol)
    
    trueY = tr['y'].ix[100]
    
    l.j(t)
    
