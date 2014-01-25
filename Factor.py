import Rvar
#import FactorOperations
import numpy as np


class Factor(object):
    """Factor : CPD Conditional Probability (discrete) Distribution
      var    Vector of variables in the factor, e.g. [1 2 3] => X1, X2, X3. always ordered by id
      card   Vector of cardinalities of var, e.g. [2 2 2] => all binomials
      val    Value table with the conditional probabilities. n dimensional array
      size   Length of val array = prod(card)"""
    
    def __init__(self, variables):
        self.variables = variables ###   <---------------------------------------
        self.cards = [c.totCard() for c in self.variables]
        self.values = np.zeros(self.cards).astype('float64')
    
    def idx2ass(self, arr):
        '''mapping between all combinatorial ordered assignments and a flatten vector of values'''
        t = [[e]*len(self.cards) for e in range(len(arr))]
        order, dic = [], {}
        for q, row in enumerate(t):
            prod, ass = 1, []
            for i, e in enumerate([1] + self.cards[:-1]):
                prod *= e
                ass.append((row[i]/prod) % self.cards[i])
            key = tuple(ass)
            dic[key] = arr[q]
            order += [key]
        return [order, dic]

    def fill_values(self, arr):
        #order, dic = idx2ass(self, arr)
        order, dic = self.idx2ass(arr)
        for e in order:
            self.values[e] = dic[e]

