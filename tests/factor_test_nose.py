import Rvar
import Factor
from nose import with_setup

epsilon = 0.00000000001
# nosetests -v tests/factor_test_nose.py

def setup_func():
    global v, f
    v, f = {}, {}
    v[1] = Rvar.Rvar(1, 2)
    v[2] = Rvar.Rvar(2, 2)
    v[3] = Rvar.Rvar(3, 2)
    v[4] = Rvar.Rvar(4, 3)
    f[1] = Factor.Factor([v[1]], [0.11, 0.89])
    f[2] = Factor.Factor([v[2], v[1]], [0.59, 0.41, 0.22, 0.78])
    f['w'] = Factor.Factor([v[3], v[2], v[1]], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0.15, 0.21])
    f['z'] = Factor.Factor([v[3], v[2], v[4]], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18])
    f['a'] = Factor.Factor([v[1]], [0.11, 0.89])
    f['b'] = Factor.Factor([v[2], v[1]], [0.59, 0.41, 0.22, 0.78])
    f['c'] = Factor.Factor([v[3], v[2]], [0.39, 0.61, 0.06, 0.94])
    f['x'] = Factor.Factor([v[2], v[4]], [0.5, 0.8, 0.1, 0, 0.3, 0.9])
    f['y'] = Factor.Factor([v[3], v[2]], [0.5, 0.7, 0.1, 0.2])
    f['z'] = Factor.Factor([v[3], v[2], v[4]], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18])


def test_cardinality_of_new_factor():
    a = Rvar.Rvar(1, 5)
    b = Rvar.Rvar(1, [0,1])
    x = Factor.Factor([a, b])
    assert x.cards==[5, 2]

@with_setup(setup_func)
def test_ass2idx_univariate_binomial():
    assert f[1].ass2idx([0]) == 0
    assert f[1].ass2idx([1]) == 1

@with_setup(setup_func)
def test_idx2ass_univariate_binomial():
    assert f[1].idx2ass([1]) == [[0]]
    assert f[1].idx2ass([2]) == [[1]]

@with_setup(setup_func)
def test_ass2idx_bivariate_binomial():
    assert f[2].ass2idx([0,0]) == 0
    assert f[2].ass2idx([1,0]) == 1
    assert f[2].ass2idx([0,1]) == 2
    assert f[2].ass2idx([1,1]) == 3

@with_setup(setup_func)
def test_ass2idx_bivariate_binomial():
    assert f[2].idx2ass([1,1]) == [[0,0],[0,0]]
    assert f[2].idx2ass([2]) == [[1,0]]
    assert f[2].idx2ass([3,4]) == [[0,1],[1,1]]

@with_setup(setup_func)
def test_ass2idx_multivariate_binomial():
    assert f['w'].ass2idx([1,0,1]) == 5
    assert f['z'].ass2idx([1,0,2]) == 9
    assert f['z'].ass2idx([1,1,0]) == 3

@with_setup(setup_func)
def test_idx2ass_multivariate_binomial():
    assert f['w'].idx2ass([6]) == [[1,0,1]]
    assert f['z'].idx2ass([10,4]) == [[1,0,2],[1,1,0]]

@with_setup(setup_func)
def test_multiply_dimensions():
    c = f['a'].multiply_by(f['b'])
    assert c.variables == [v[1], v[2]]
    assert c.cards == [2, 2]
    
    c = f['x'].multiply_by(f['y'])
    assert c.variables == [v[2], v[3], v[4]]
    assert c.cards == [2, 2, 3]

@with_setup(setup_func)
def test_multiply_values():
    c = f['a'].multiply_by(f['b'])
    d = [0.0649, 0.1958, 0.0451, 0.6942]
    for e in [a - b for a, b in zip(c.values, d)]:
        assert e < epsilon
    
    c = f['x'].multiply_by(f['y'])
    print c.values
    d = [0.25, 0.08, 0.35, 0.16, 0.05, 0.0, 0.07, 0.0, 0.15, 0.09, 0.21, 0.18]
    for e in [a - b for a, b in zip(c.values, d)]:
        assert e < epsilon

@with_setup(setup_func)
def test_marginalize():
    c = f['b'].marginalize_by([v[2]])
    assert c.values == [0.5, 0.5]
    
    c = f['z'].marginalize_by([v[2]])
    print c.values
    d = [0.20754716981132074, 0.320754716981132, 0.03144654088050314, 0.0440251572327044, \
         0.15094339622641506, 0.2452830188679245]
    for e in [a - b for a, b in zip(c.values, d)]:
        assert e < epsilon

@with_setup(setup_func)
def test_conditioning():
    evidence = {v[2]:0, v[3]:1}
    assert f['a'].conditioned_by(evidence)==f['a']
    assert f['b'].conditioned_by(evidence).values == [0.59, 0, 0.22, 0]
    assert f['c'].conditioned_by(evidence).values == [0, 0.61, 0, 0]
    assert f['z'].conditioned_by({v[3]:0}).values == [0.25, 0, 0.08, 0, 0.05, 0, 0, 0, 0.15, 0, 0.09, 0]

@with_setup(setup_func)
def test_joint():
    jf = Factor.Factor.joint([f['a'], f['b'], f['c']])
    d = [0.025311, 0.076362, 0.002706, 0.041652, 0.039589, 0.119438, 0.042394, 0.652548]
    for e in [a - b for a, b in zip(jf.values, d)]:
        assert e < epsilon

def test_complete_1():
    v1 = Rvar.Rvar(1, 3)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 2)
    v5 = Rvar.Rvar(5, 3)
    v6 = Rvar.Rvar(6, 3)
    v7 = Rvar.Rvar(7, 2)
    v8 = Rvar.Rvar(8, 3)
    
    f1 = Factor.Factor([v1], [1.0/3.0, 1.0/3.0, 1.0/3.0])
    f2 = Factor.Factor([v8, v2], [0.9, 0.1, 0.5, 0.5, 0.1, 0.9])
    f3 = Factor.Factor([v3, v4, v7, v2], [0.9, 0.1, 0.8, 0.2, 0.7, \
            0.3, 0.6, 0.4, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9])
    f4 = Factor.Factor([v4], [0.5, 0.5])
    f5 = Factor.Factor([v5, v6], [0.75, 0.2, 0.05, 0.2, 0.6, 0.2, 0.05, 0.2, 0.75])
    f6 = Factor.Factor([v6], [0.3333, 0.3333, 0.3333])
    f7 = Factor.Factor([v7, v5, v6], [0.9, 0.1, 0.8, 0.2, 0.7, \
            0.3, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9])
    f8 = Factor.Factor([v8, v4, v1], [0.1, 0.3, 0.6, 0.05, 0.2,\
            0.75, 0.2, 0.5, 0.3, 0.1, 0.35, 0.55, 0.8, 0.15, 0.05, 0.2, 0.6, 0.2])
    
    jf = Factor.Factor.joint([f1, f2, f3, f4, f5, f6, f7, f8]).marginalize_by([v2,v3,v4,v5,v6,v7,v8])
    d = [0.3741, 0.3027, 0.3231]
    print jf.values
    for e in [a - b for a, b in zip(jf.values, d)]:
        assert e < 0.001
    assert jf.variables == [v1]

def test_complete_2():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 3)
    v5 = Rvar.Rvar(5, 2)
    
    d = Factor.Factor([v1], [0.6, 0.4])
    i = Factor.Factor([v2], [0.7, 0.3]);
    s = Factor.Factor([v3, v2], [0.95, 0.05, 0.2, 0.8]);
    g = Factor.Factor([v4, v1, v2], [0.3, 0.4, 0.3, 0.05, 0.25, 0.7, 0.9, 0.08, 0.02, 0.5, 0.3, 0.2]);
    l = Factor.Factor([v5, v4], [0.1, 0.9, 0.4, 0.6, 0.99, 0.01]);
    
    s = s.conditioned_by({v3:1}) # we observe high SAT
    q = Factor.Factor.joint([d, i, s, g, l]).marginalize_by([v1, v3, v4, v5])
    
    d = [0.12727, 0.87273]
    for e in [a - b for a, b in zip(q.values, d)]:
        assert e < epsilon

