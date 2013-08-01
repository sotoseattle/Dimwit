import Rvar
import Factor

epsilon = 0.00000000001
# nosetests -v tests/factor_test_nose.py

def test_number_3_4():
    assert 2+2==4
    
def test_cardinality_of_new_factor():
    a = Rvar.Rvar(1, 5)
    b = Rvar.Rvar(1, [0,1])
    x = Factor.Factor([a, b])
    assert x.cards==[5, 2]

def test_ass2idx_univariate_binomial():
    v1 = Rvar.Rvar(1, 2)
    f1 = Factor.Factor([v1], [0.11, 0.89])
    assert f1.ass2idx([0]) == 0
    assert f1.ass2idx([1]) == 1

def test_idx2ass_univariate_binomial():
    v1 = Rvar.Rvar(1, 2)
    f1 = Factor.Factor([v1], [0.11, 0.89])
    assert f1.idx2ass([1]) == [[0]]
    assert f1.idx2ass([2]) == [[1]]

def test_ass2idx_bivariate_binomial():
    v1, v2 = Rvar.Rvar(1, 2), Rvar.Rvar(2, 2)
    f2 = Factor.Factor([v2, v1], [0.59, 0.41, 0.22, 0.78])
    assert f2.ass2idx([0,0]) == 0
    assert f2.ass2idx([1,0]) == 1
    assert f2.ass2idx([0,1]) == 2
    assert f2.ass2idx([1,1]) == 3

def test_ass2idx_bivariate_binomial():
    v1, v2 = Rvar.Rvar(1, 2), Rvar.Rvar(2, 2)
    f2 = Factor.Factor([v2, v1], [0.59, 0.41, 0.22, 0.78])
    assert f2.idx2ass([1,1]) == [[0,0],[0,0]]
    assert f2.idx2ass([2]) == [[1,0]]
    assert f2.idx2ass([3,4]) == [[0,1],[1,1]]

def test_ass2idx_multivariate_binomial():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 3)
    w = Factor.Factor([v3, v2, v1], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0.15, 0.21])
    z = Factor.Factor([v3, v2, v4], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18])
    assert w.ass2idx([1,0,1]) == 5
    assert z.ass2idx([1,0,2]) == 9
    assert z.ass2idx([1,1,0]) == 3

def test_idx2ass_multivariate_binomial():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 3)
    w = Factor.Factor([v3, v2, v1], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0.15, 0.21])
    z = Factor.Factor([v3, v2, v4], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18])
    assert w.idx2ass([6]) == [[1,0,1]]
    assert z.idx2ass([10,4]) == [[1,0,2],[1,1,0]]

def test_multiply_dimensions():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 3)
    
    a = Factor.Factor([v1], [0.11, 0.89])
    b = Factor.Factor([v2, v1], [0.59, 0.41, 0.22, 0.78])    
    c = a.multiply_by(b)
    print c.variables
    assert c.variables == [v1, v2]
    assert c.cards == [2, 2]
    
    x = Factor.Factor([v2, v4], [0.5, 0.8, 0.1, 0, 0.3, 0.9])
    y = Factor.Factor([v3, v2], [0.5, 0.7, 0.1, 0.2])
    c = x.multiply_by(y)
    assert c.variables == [v2, v3, v4]
    assert c.cards == [2, 2, 3]

def test_multiply_values():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(4, 3)
    
    a = Factor.Factor([v1], [0.11, 0.89])
    b = Factor.Factor([v2, v1], [0.59, 0.41, 0.22, 0.78])
    x = Factor.Factor([v2, v4], [0.5, 0.8, 0.1, 0, 0.3, 0.9])
    y = Factor.Factor([v3, v2], [0.5, 0.7, 0.1, 0.2])
    
    c = a.multiply_by(b)
    d = [0.0649, 0.1958, 0.0451, 0.6942]
    for e in [a - b for a, b in zip(c.values, d)]:
        assert e < epsilon
    
    c = x.multiply_by(y)
    print c.values
    d = [0.25, 0.08, 0.35, 0.16, 0.05, 0.0, 0.07, 0.0, 0.15, 0.09, 0.21, 0.18]
    for e in [a - b for a, b in zip(c.values, d)]:
        assert e < epsilon


def test_marginalize():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(1, 3)
    
    a = Factor.Factor([v1], [0.11, 0.89])
    b = Factor.Factor([v2, v1], [0.59, 0.41, 0.22, 0.78])
    z = Factor.Factor([v3, v2, v4], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18])
    
    c = b.marginalize_by([v2])
    assert c.values == [0.5, 0.5]
    
    c = z.marginalize_by([v2])
    d = [0.20754716, 0.0314465, 0.15094339, 0.320754716, 0.0440251572, 0.245283018]
    print c.values
    for e in [a - b for a, b in zip(c.values, d)]:
        assert e < 0.0001

def test_conditioning():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(1, 3)
    evidence = {v2:0, v3:1}
    
    a = Factor.Factor([v1], [0.11, 0.89])
    b = Factor.Factor([v2, v1], [0.59, 0.41, 0.22, 0.78])
    c = Factor.Factor([v3, v2], [0.39, 0.61, 0.06, 0.94])
    z = Factor.Factor([v3, v2, v4], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18])
    
    assert a.conditioned_by(evidence)==a
    assert b.conditioned_by(evidence).values == [0.59, 0, 0.22, 0]
    assert c.conditioned_by(evidence).values == [0, 0.61, 0, 0]
    assert z.conditioned_by({v3:0}).values == [0.25, 0, 0.08, 0, 0.05, 0, 0, 0, 0.15, 0, 0.09, 0]

def test_joint():
    v1 = Rvar.Rvar(1, 2)
    v2 = Rvar.Rvar(2, 2)
    v3 = Rvar.Rvar(3, 2)
    v4 = Rvar.Rvar(1, 3)

    a = Factor.Factor([v1], [0.11, 0.89])
    b = Factor.Factor([v2, v1], [0.59, 0.41, 0.22, 0.78])
    c = Factor.Factor([v3, v2], [0.39, 0.61, 0.06, 0.94])

    jf = Factor.Factor.joint([a, b, c])
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
    f2 = Factor.Factor([v8, v2], [0.90000, 0.10000, 0.50000, 0.50000, 0.10000, 0.90000])
    f3 = Factor.Factor([v3, v4, v7, v2], [0.90000, 0.10000, 0.80000, 0.20000, 0.70000, 0.30000, 0.60000, 0.40000, 0.40000, 0.60000, 0.30000, 0.70000, 0.20000, 0.80000, 0.10000, 0.90000])
    f4 = Factor.Factor([v4], [0.5, 0.5])
    f5 = Factor.Factor([v5, v6], [0.750000, 0.200000, 0.050000, 0.200000, 0.600000, 0.200000, 0.050000, 0.200000, 0.750000])
    f6 = Factor.Factor([v6], [0.3333, 0.3333, 0.3333])
    f7 = Factor.Factor([v7, v5, v6], [0.90000, 0.10000, 0.80000, 0.20000, 0.70000, 0.30000, 0.60000, 0.40000, 0.50000, 0.50000, 0.40000, 0.60000, 0.30000, 0.70000, 0.20000, 0.80000, 0.10000, 0.90000])
    f8 = Factor.Factor([v8, v4, v1], [0.100000, 0.300000, 0.600000, 0.050000, 0.200000, 0.750000, 0.200000, 0.500000, 0.300000, 0.100000, 0.350000, 0.550000, 0.800000, 0.150000, 0.050000, 0.200000, 0.600000, 0.200000])
    
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

