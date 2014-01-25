import Rvar
import Factor
import numpy as np

# MODULE FOR FACTOR OPERATIONS
# Multiply = Factor product
# Marginalize = var Reduction
# Observe = Conditioning on evidence

## Helper functions

def realign(M, chaos):
    '''realign by swapping its axes'''
    order = range(len(chaos))
    for i in order:
        have_here = chaos[i]
        should_be = order[i]
        index_of_missing_should = chaos.index(should_be)
        if have_here != should_be:
            M = np.swapaxes(M, i, index_of_missing_should)
            chaos[index_of_missing_should] =  have_here
            chaos[i] = should_be
    return M

def new_axes(whole, partial):
    '''list of new axis needed for partial to become whole'''
    sol = []
    j = 0
    for e in whole:
        if e in partial:
            sol += [j]
            j += 1
        else:
            sol += [np.newaxis]
    return sol

def expand_rvars(whole, fA):
    '''expand a factor to new dimensions by inserting new axes and repeating values along them'''
    f = Factor.Factor(whole)

    chorizo = fA.values.copy()
    order = sorted(fA.variables)
    qt = [order.index(e) for e in fA.variables] # [0,1,3,2] indexes of fA.vars vs order [0,1,2,3]
    if qt!=range(len(qt)):                      # if not equal, we need to realign before proceeding
        chorizo = realign(chorizo, qt)

    sol = new_axes(whole, order)                # see which new axis we need
    for i,e in enumerate(sol):
        if e==None:                             # insert new axes and fill values
            chorizo = np.expand_dims(chorizo, axis=i)
            chorizo = np.repeat(chorizo, whole[i].totCard(), axis=i)    
    f.values = chorizo
    return f

def normalize(M):
    tot = np.sum(M)
    return M/tot

## Factor Operations

def multiply(fA, fB, norma=True):
    '''expanding factors to the whole set of variables, then we can multiply element-wise'''
    allVars = sorted(list(set(fA.variables) | set(fB.variables)))
    f = Factor.Factor(allVars)
    FA = expand_rvars(allVars, fA)
    FB = expand_rvars(allVars, fB)
    f.values = FA.values * FB.values
    if norma:
        f.values = normalize(f.values)
    return f

def marginalize(fA, v):
    new_vs = [e for e in fA.variables if e != v ]
    if new_vs==[]:
        raise Exception("Error: Resultant factor has empty scope")
    f = Factor.Factor(new_vs)
    pos_m = fA.variables.index(v)
    f.values = np.sum(fA.values, axis=pos_m)
    return f

def observe(fA, evidence, norma=True):
    f = Factor.Factor(fA.variables)
    M = fA.values.copy()
    for cond_var in evidence.keys():
        if cond_var in fA.variables:
            a = []
            for v in fA.variables:
                if v.id == cond_var.id:
                    a += [[x for x in range(v.totCard()) if x!=evidence[cond_var]]]
                else:
                    a += [slice(None)]
            M[a]=0.
    f.values = normalize(M) if norma else M
    return f

def sum(fA, fB, norma=True):
    '''expanding factors to the whole set of variables, then we can sum element-wise'''
    allVars = sorted(list(set(fA.variables) | set(fB.variables)))
    f = Factor.Factor(allVars)
    FA = expand_rvars(allVars, fA)
    FB = expand_rvars(allVars, fB)
    f.values = FA.values + FB.values
    if norma:
        f.values = normalize(f.values)
    return f

def max_marginalize(fA, v):
    new_vs = [e for e in fA.variables if e != v ]
    if new_vs==[]:
        raise Exception("Error: Resultant factor has empty scope")
    f = Factor.Factor(new_vs)
    pos_m = fA.variables.index(v)
    f.values = np.max(fA.values, axis=pos_m)
    return f






