import Rvar
import Factor
import FactorOperations
from nose import with_setup
import numpy as np

epsilon = 1.e-11
#nosetests -v tests/factor_test_nose.py

def setup_func():
    global v, f
    v, f = {}, {}
    v[1] = Rvar.Rvar(1, 2)
    v[2] = Rvar.Rvar(2, 2)
    v[3] = Rvar.Rvar(3, 2)
    v[4] = Rvar.Rvar(4, 3)
    
    f['a'] = Factor.Factor([v[1]])
    f['a'].fill_values([0.11, 0.89])
    f['b'] = Factor.Factor([v[2], v[1]])
    f['b'].fill_values([0.59, 0.41, 0.22, 0.78])
    f['c'] = Factor.Factor([v[3], v[2]])
    f['c'].fill_values([0.39, 0.61, 0.06, 0.94])
    f['x'] = Factor.Factor([v[2], v[4]])
    f['x'].fill_values([0.5, 0.8, 0.1, 0, 0.3, 0.9])
    f['y'] = Factor.Factor([v[3], v[2]])
    f['y'].fill_values([0.5, 0.7, 0.1, 0.2])
    f['z'] = Factor.Factor([v[3], v[2], v[4]])
    f['z'].fill_values([0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18])


def test_cardinality_of_new_factor():
    a = Rvar.Rvar(1, 5)
    b = Rvar.Rvar(1, [0,1])
    x = Factor.Factor([a, b])
    assert x.cards==[5, 2]

@with_setup(setup_func)
def test_multiply_dimensions():
    c = FactorOperations.multiply(f['a'], f['b'])
    assert c.variables == [v[1], v[2]]
    assert c.cards == [2, 2]
    
    c = FactorOperations.multiply(f['x'], f['y'])
    assert c.variables == [v[2], v[3], v[4]]
    assert c.cards == [2, 2, 3]

@with_setup(setup_func)
def test_multiply_values():
    c = FactorOperations.multiply(f['a'], f['b'])
    sol = Factor.Factor(c.variables)
    sol.fill_values([0.0649, 0.1958, 0.0451, 0.6942])
    assert np.allclose(c.values, sol.values, atol=epsilon)


def test_multiply_values2():
    v_1 = Rvar.Rvar(1, 3)
    v_2 = Rvar.Rvar(2, 2)
    v_3 = Rvar.Rvar(3, 2)
    
    X = Factor.Factor([v_2, v_1])
    X.fill_values([0.5, 0.8, 0.1, 0., 0.3, 0.9])
    Y = Factor.Factor([v_3, v_2])
    Y.fill_values([0.5, 0.7, 0.1, 0.2])

    Z = FactorOperations.multiply(X, Y, False)
    sol = Factor.Factor([v_1, v_2, v_3])
    sol.fill_values([0.25, 0.05, 0.15, 0.08, 0, 0.09, 0.35, 0.07, 0.21, 0.16, 0, 0.18])
    assert np.allclose(Z.values, sol.values, atol=epsilon)

@with_setup(setup_func)
def test_marginalize():
    c = FactorOperations.marginalize(f['b'], v[2])
    assert np.allclose(c.values, [1., 1.], atol=epsilon)
    
    c = FactorOperations.marginalize(f['z'], v[2])
    sol = Factor.Factor([v[4], v[3]])
    sol.fill_values([0.33, 0.05, 0.24, 0.51, 0.07, 0.39])
    assert np.allclose(c.values, sol.values.T, atol=epsilon)
    
@with_setup(setup_func)
def test_conditioning():
    evidence = {v[2]:0, v[3]:1}
    assert np.allclose(FactorOperations.observe(f['a'], evidence).values, f['a'].values)
    assert np.allclose(FactorOperations.observe(f['b'], evidence, False).values, [[0.59, 0.22], [0., 0.]])
    assert np.allclose(FactorOperations.observe(f['c'], evidence, False).values, [[0., 0.], [0.61, 0.]])
    assert np.allclose(FactorOperations.observe(f['z'], {v[3]:0}, False).values, [[[0.25, 0.05, 0.15], [0.08, 0., 0.09]], [[0.,0.,0.],[0.,0.,0.]]])

def test_complete_1():
    v1 = Rvar.Rvar(1, 3)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 2)
    v5 = Rvar.Rvar(5, 3)
    v6 = Rvar.Rvar(6, 3)
    v7 = Rvar.Rvar(7, 2)
    v8 = Rvar.Rvar(8, 3)
        
    f1 = Factor.Factor([v1])
    f1.fill_values([1.0/3.0, 1.0/3.0, 1.0/3.0])
    f2 = Factor.Factor([v8, v2])
    f2.fill_values([0.9, 0.1, 0.5, 0.5, 0.1, 0.9])
    f3 = Factor.Factor([v3, v4, v7, v2])
    f3.fill_values([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9])
    f4 = Factor.Factor([v4])
    f4.fill_values([0.5, 0.5])
    f5 = Factor.Factor([v5, v6])
    f5.fill_values([0.75, 0.2, 0.05, 0.2, 0.6, 0.2, 0.05, 0.2, 0.75])
    f6 = Factor.Factor([v6])
    f6.fill_values([0.3333, 0.3333, 0.3333])
    f7 = Factor.Factor([v7, v5, v6])
    f7.fill_values([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9])
    f8 = Factor.Factor([v8, v4, v1])
    f8.fill_values([0.1, 0.3, 0.6, 0.05, 0.2,0.75, 0.2, 0.5, 0.3, 0.1, 0.35, 0.55, 0.8, 0.15, 0.05, 0.2, 0.6, 0.2])

    factors = [f2, f3, f4, f5, f6, f7, f8]
    a = f1
    for f in factors:
        a = FactorOperations.multiply(a, f)
    rvars = [v2,v3,v4,v5,v6,v7,v8]

    for v in rvars:
        a = FactorOperations.marginalize(a, v)
    
    assert np.allclose(a.values, [0.37414966, 0.30272109, 0.32312925])

def test_complete_2():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 3)
    v5 = Rvar.Rvar(5, 2)
        
    d = Factor.Factor([v1])
    d.fill_values([0.6, 0.4])
    i = Factor.Factor([v2])
    i.fill_values([0.7, 0.3]);
    s = Factor.Factor([v3, v2])
    s.fill_values([0.95, 0.05, 0.2, 0.8]);
    g = Factor.Factor([v4, v1, v2])
    g.fill_values([0.3, 0.4, 0.3, 0.05, 0.25, 0.7, 0.9, 0.08, 0.02, 0.5, 0.3, 0.2]);
    l = Factor.Factor([v5, v4])
    l.fill_values([0.1, 0.9, 0.4, 0.6, 0.99, 0.01]);
    
    s = FactorOperations.observe(s, {v3:1}) # we observe high SAT
    
    factors = [i, s, g, l]
    a = d
    for f in factors:
        a = FactorOperations.multiply(a, f)
        print a.variables, a.values.size
    
    rvars = [v1, v3, v4, v5]
    for v in rvars:
        a = FactorOperations.marginalize(a, v)
        print a.variables, a.values.size
    
    assert np.allclose(a.values, [0.12727273, 0.87272727])

