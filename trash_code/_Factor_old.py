import Rvar
import copy
import numpy as np
from itertools import chain


class Factor_old(object):
    """Factor : CPD Conditional Probability (discrete) Distribution
      var    Vector of variables in the factor, e.g. [1 2 3] => X1, X2, X3
      card   Vector of cardinalities of var, e.g. [2 2 2] => all binomials
      val    Value table with the conditional probabilities
      size   Length of val array = prod(card)"""
    
    
    def __init__(self, variables = [None], values = [None]):
        self.variables = variables
        self.cards = [c.totCard() for c in variables]
        self.values = [0]*reduce(lambda x, y: x*y, self.cards)
        if len(self.values)==len(values):
            self.values = values
        elif values != [None]:
            raise Exception("Mismatch of values size")
    
    
    def multiply_by(self, fB):
        fC = Factor_old(sorted(list(set(self.variables) | set(fB.variables))))
        mapA = [fC.variables.index(item) for item in self.variables]
        mapB = [fC.variables.index(item) for item in fB.variables]
        assignmentsC = fC.idx2ass([])
        for i, assC in enumerate(assignmentsC):
            assA = [assC[e] for e in mapA]
            assB = [assC[e] for e in mapB]
            fC.values[i] = self.values[self.ass2idx(assA)]*fB.values[fB.ass2idx(assB)]
        return fC
    
    
    def marginalize_by(self, vs):
        if vs==[]:
            return self
        cvs = list(set(self.variables) - set(vs))
        if cvs==[]:
            raise Exception("Error: Resultant factor has empty scope")
        fC = Factor_old(sorted(cvs))
        
        assignmentsC = fC.idx2ass([])
        mapC = [self.variables.index(v) for v in fC.variables]
        for i, old_ass in enumerate(self.idx2ass([])):
            assC = [old_ass[e] for e in mapC]
            fC.values[fC.ass2idx(assC)] += self.values[i]
        z = sum(fC.values)
        fC.values = [e/z for e in fC.values]
        return fC

    def marginalize2_by(self, vs):
        if vs==[]:
            return self
        cvs = list(set(self.variables) - set(vs))
        if cvs==[]:
            raise Exception("Error: Resultant factor has empty scope")
        fC = Factor_old(sorted(cvs))
        
        srav  = self.variables[::-1]
        print srav
        #print vs.type
        pos = srav.index(vs[0])
        print 'pos', pos


        sdrac = self.cards[::-1]
        print 'sdrac', sdrac

        a = np.array(self.values).reshape(sdrac)   # reverse order of vars
        b = np.sum(a, axis=pos)
        c = list(b.flatten())
        
        fC.values = c  # reverse order again for r
        z = sum(fC.values)
        fC.values = [e/z for e in fC.values]
        return fC

    
    
    def conditioned_by(self, evidence):
        condVars = list(set(self.variables) & set(evidence.keys()))
        if condVars==[]:
            return self
        fC = copy.deepcopy(self)
        
        idx = []
        for v in self.variables:
            if v in evidence:
                idx.append(evidence[v])        
        for i, ass in enumerate(self.idx2ass([])):
            for j, v in enumerate(ass):
                if j<len(idx) and v != idx[j]:
                    fC.values[i] = 0.0
        return fC
            
    def ass2idx(self, ass):
        if ass.__class__ != list:
            raise Exception("assignment is not a list")
        suma, prod = 0, 1
        for i, e in enumerate([1] + self.cards[:-1]):
            prod *= e
            evidence = ass[i]
            if evidence > self.cards[i]-1:
                raise "assignment not valid"
            suma += prod*(evidence)
        return suma
    
    def idx2ass(self, indicesArray):
        limit = len(self.values)
        if indicesArray.__class__ != list:
            raise "assignment is not an array of indices"
        elif indicesArray == []:
            indices = range(1, limit+1)
        else:
            indices = indicesArray
        
        t, sol = [], []
        for e in indices:
            if e>limit:
                raise "index beyond scope of factor values"
            t.append([e-1]*len(self.cards))
        for row in t:
            prod, ass = 1, []
            for i, e in enumerate([1] + self.cards[:-1]):
                prod *= e
                ass.append((row[i]/prod) % self.cards[i])
            sol.append(ass)
        return sol
    
    
    
    @classmethod
    def joint(cls, factorList):
        l = [f.variables for f in factorList]
        allVars = list(set(chain.from_iterable(l)))
        fC = Factor_old(sorted(allVars))
        
        maps = []
        for f in factorList:
            maps.append([fC.variables.index(v) for v in f.variables])
        
        assignmentsC = fC.idx2ass([])
        for i, assC in enumerate(assignmentsC):
            prod = 1
            for j, f in enumerate(factorList):
                ass = [assC[e] for e in maps[j]]
                prod *= f.values[f.ass2idx(ass)]
            fC.values[i] = prod
        return fC
