import pandas as pd
import numpy as np
import Logysterical
from nose import with_setup

epsilon = 0.00000001

def setup_func():
    global l
    data = pd.read_csv('./tests/data_example.txt')
    X = data.ix[:, 0:2]
    y = data.ix[:, 2]
    l = Logysterical.Logysterical(X, y)


@with_setup(setup_func)
def test_J_with_no_theta():
    theta = np.zeros_like(l.X)
    assert l.costFunction(theta) - 0.69314718056 < epsilon
