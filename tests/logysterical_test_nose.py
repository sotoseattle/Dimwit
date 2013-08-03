import pandas as pd
import numpy as np
import Logysterical
from nose import with_setup

epsilon = 0.000001

def setup_func():
    global l
    data = pd.read_csv('./tests/log_data_basic.txt')
    X = data.ix[:, 0:2]
    y = data.ix[:, 2]
    l = Logysterical.Logysterical(X, y, 0.0)


@with_setup(setup_func)
def test_J_with_zero_theta():
    theta = np.array([0,0,0])
    assert l.costFunction(theta) - 0.69314718056 < epsilon

@with_setup(setup_func)    
def test_gradient_with_zero_theta():
    theta = np.array([0,0,0])
    diff = l.gradientFunction(theta) - np.array([-0.1, -12.009217, -11.262842])
    assert diff.max() < epsilon

@with_setup(setup_func)
def test_optimize_thetas():
    ini_thetas = np.array([0,0,0])
    opt_thetas = l.optimizeThetas(ini_thetas)
    diff = opt_thetas - np.array([-25.1613364, 0.20623176, 0.2014716])
    assert diff.max() < epsilon

@with_setup(setup_func)
def test_prediction():
    ini_thetas = np.array([0,0,0])
    opt_thetas = l.optimizeThetas(ini_thetas)
    prob = l.predict(np.array([1, 45, 85]))
    assert prob - 0.776290570502 < epsilon

@with_setup(setup_func)    
def test_accuracy():
    ini_thetas = np.array([0,0,0])
    opt_thetas = l.optimizeThetas(ini_thetas)
    assert l.accuracy(l.Xt, l.yt) - 0.89 < epsilon
    

