import pandas as pd
import numpy as np
from Logysterical import * 
from nose import with_setup

epsilon = 0.000001

def setup_func_basic():
    global l
    data = pd.read_csv('./tests/log_data_basic.txt')
    X = data.ix[:, 0:2]
    y = data.ix[:, 2]
    l = Logysterical(X, y, 0.0)

def setup_func_reg():
    global train
    train = {}
    data = pd.read_csv('./tests/log_data_reg.txt')
    x = np.array(data.ix[:, 0:2])
    X1 = x[:,0]
    X2 = x[:,1]
    train['X'] = Logysterical.mapfeature(X1, X2)
    train['y'] = np.array(data.ix[:, 2])
    

@with_setup(setup_func_basic)
def test_J_with_zero_theta():
    theta = np.array([0,0,0])
    assert l.costFunction(theta) - 0.69314718056 < epsilon

@with_setup(setup_func_basic)    
def test_gradient_with_zero_theta():
    theta = np.array([0,0,0])
    diff = l.gradientFunction(theta) - np.array([-0.1, -12.009217, -11.262842])
    assert diff.max() < epsilon

@with_setup(setup_func_basic)
def test_optimize_thetas():
    ini_thetas = np.array([0,0,0])
    opt_thetas = l.optimizeThetas(ini_thetas)
    diff = opt_thetas - np.array([-25.1613364, 0.20623176, 0.2014716])
    assert diff.max() < epsilon

@with_setup(setup_func_basic)
def test_prediction():
    ini_thetas = np.array([0,0,0])
    opt_thetas = l.optimizeThetas(ini_thetas)
    prob = l.predict(np.array([1, 45, 85]))
    assert prob - 0.776290570502 < epsilon

@with_setup(setup_func_basic)    
def test_accuracy():
    ini_thetas = np.array([0,0,0])
    opt_thetas = l.optimizeThetas(ini_thetas)
    assert l.accuracy(l.Xt, l.yt) - 0.89 < epsilon


### WITH REGULARIZATION


@with_setup(setup_func_reg)
def test_J_with_lam_and_zero_theta():
    l = Logysterical(train['X'], train['y'], 1.0)
    theta = np.zeros((l.Xt.shape[1],1))
    assert l.costFunction(theta) - 0.69314718056 < epsilon

def test_mapping_features():
    data = pd.read_csv('./tests/log_data_reg.txt')
    x = np.array(data.ix[:, 0:2])
    X1 = x[:,0]
    X2 = x[:,1]
    qq = Logysterical.mapfeature(X1, X2)
    m,n = qq.shape
    for i in range(m):
        assert qq[i,0] - X1[i] == 0
        assert qq[i,1] - X2[i] == 0

@with_setup(setup_func_reg)
def test_optimize_thetas_reg():
    l = Logysterical(train['X'], train['y'], 1.0)
    ini_thetas = np.zeros((l.Xt.shape[1],1))
    diff = l.optimizeThetas(ini_thetas) - np.array([1.273005, 0.624876, 1.177376, -2.020142, -0.912616, -1.429907, 0.125668, -0.368551, -0.360033, -0.171068, -1.460894, -0.052499, -0.618889, -0.273745, -1.192301, -0.240993, -0.207934, -0.047224, -0.278327, -0.296602, -0.453957, -1.045511, 0.026463, -0.294330, 0.014381, -0.328703, -0.143796, -0.924883])
    for e in diff:
        assert e < epsilon



    


