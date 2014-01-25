from UndGraph import UndGraph

class IndMarkovNet(UndGraph, object):
    '''Induced Markov Network as an undirected graph of linked random variables
       derived from a list of factors'''
    
    def __init__(self, listOfFactors):
        set_vars = sorted(set(v for fu in listOfFactors for v in fu.variables))
        super(IndMarkovNet, self).__init__([[e] for e in set_vars])
        self.factors = listOfFactors
        for fu in self.factors:
            self.connectAll([self.index_var(v) for v in fu.variables])
    
    def firstMinNeighborVar(self):
        minino, pos = float('inf'), None
        for i,a in enumerate(self.adj):
            if 0<len(a)<minino:
                minino = len(a)
                pos = i
        return list(self.box[pos])[0]
