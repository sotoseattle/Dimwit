import pandas as pd
import numpy as np
from Logysoft import * 
from nose import with_setup

epsilon = 0.000001

def setup_func_basic():
    global abc, l
    abc = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y', 'z']
    data = pd.read_csv('./tests/data/singleton_thetas.txt', header=None)
    t = np.array(data.ix[:,:])
    l = Logysoft(thetas = t)


@with_setup(setup_func_basic)
def test_hypothesis_with_given_thetas_1():
    x = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0, \
                  0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1, \
                  1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1, \
                  0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1, \
                  0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1, \
                  0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1, \
                  0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1, \
                  0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0])
    x = np.hstack((1, x))
    sol = l.hypothesis(x)
    
    assert np.amax(sol) - 0.866184526157 < epsilon
    assert abc[np.argmax(sol)] == 'o'

@with_setup(setup_func_basic)
def test_hypothesis_with_given_thetas_2():
    x = np.array([0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0, \
                  0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0, \
                  0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0, \
                  0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    x = np.hstack((1, x))
    sol = l.hypothesis(x)
    assert np.amax(sol) - 0.635524052858 < epsilon
    assert abc[np.argmax(sol)] == 'r'

@with_setup(setup_func_basic)
def test_hypothesis_with_given_thetas_3():
    x = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, \
                  1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1, \
                  0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0, \
                  0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    x = np.hstack((1, x))
    sol = l.hypothesis(x)
    assert np.amax(sol) - 0.951043189829 < epsilon
    assert abc[np.argmax(sol)] == 'i'
