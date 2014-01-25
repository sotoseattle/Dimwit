import copy
import numpy as np
import Factor
import Rvar
import FactorOperations
from UndGraph import UndGraph
from BFS_Paths import BFS_Paths
from IndMarkovNet import IndMarkovNet

class CliqueTree(UndGraph, object):

    def __init__ (self, listOfFactors):
        super(CliqueTree, self).__init__([])
        
        # create induced markov net from list of factors
        F = IndMarkovNet(listOfFactors)
        
        # create nodes iteratively through var elimination
        self.tau = [] # wee need this to hold the taus as we compute
        considered_cliques = 0
        while considered_cliques < F.V:
            z = F.firstMinNeighborVar()
            F = self.eliminateVar(F, z)
            considered_cliques += 1
        F, self.tau = None, []
        
        # prune, compact and initialize resulting tree
        # don't overprune or you wont be able to pass mssg!
        keepPruning = True
        #while keepPruning and self.V>2:
        while keepPruning:
            keepPruning = self.pruneNode()
            if not len([b for b in self.box if b])>2:
                keepPruning = False
        
        self.compactTree()
        self.initializePotentials(listOfFactors)
        # clone an initialized adj structure for message propagation
        self.delta = [[None]*len(self.adj[i]) for i in range(self.V)]

    
    def eliminateVar(self, F, z):
        # separate factors into two lists, those that use z (F_Cluster) and the rest
        F_Cluster, F_Rest, Cluster_vars = [], [], []
        for f in F.factors:
            if z in f.variables:
                F_Cluster += [f]
                Cluster_vars += f.variables
            else:
                F_Rest += [f]
        
        if F_Cluster!=[]:
            # add a node to clique tree with the variables involved
            position = self.V
            self.V += 1
            self.box.insert(position, set(Cluster_vars))
            self.adj.insert(position, [])
            
            # when computing tau of new node, check if it uses other nodes' taus and connect
            for i in range(position):
                if self.tau[i] in F_Cluster:
                    self.addEdge(i, position)
            
            # multiply the factors in Cluster... (lambda) ...and marginalize by z (tau)
            tau = F_Cluster.pop(0)
            for f in F_Cluster:
                tau = FactorOperations.multiply(tau, f, False)
            if tau.variables != [z]:
                tau = FactorOperations.marginalize(tau, z)
            self.tau.insert(position, tau)
            
            # update the edges of F (connect all vars inside new factor, & disconnect the eliminated variable)
            F.connectAll([F.index_var(v) for v in self.box[position]])
            F.adj[F.index_var(z)] = []
            
            # add to unused factor list the resulting tau ==> new factor list with var eliminated
            F_Rest += [tau]            
            F.factors = F_Rest
        return F

    def pruneNode(self):
        '''Start with a node (A), scan through its neighbors to find one that is a superset of variables (B). 
           Add edges between B and all of A's other neighbors and cut off all edges from A.'''
        for i,nodeA in enumerate(self.box):
            neighbors = self.adj[i]
            for j in neighbors:
                nodeB = self.box[j]
                if nodeA and nodeA < nodeB:
                    # connect nodeB to other nodeA's neighbors
                    for e in neighbors:
                        if j!=e:
                            self.addEdge(j,e)
                    # disconnect child
                    self.adj[i] = []
                    self.box[i] = set()
                    #self.V -= 1
                    return True
        return False
    
    def compactTree(self):
        '''compacts the clique tree after prunning to remove null nodes'''
        translatable = {}
        counter = 0
        for i,e in enumerate(self.box):
            if e:
                translatable[i] = counter
                counter +=1
        newV = len(translatable.keys())
        nodes, lnk = [None]*newV, [None]*newV
        for k in translatable:
            nodes[translatable[k]] = self.box[k]
            lnk[translatable[k]] = [translatable[e] for e in self.adj[k] if e in translatable]
        self.box = nodes
        self.adj = lnk
        self.V = len(nodes)
        pass
    
    def initializePotentials(self, listOfFactors):
        # create factors initialized to ones
        self.factors = [None]*self.V
        for i in range(self.V):
            fu = Factor.Factor(sorted(list(self.box[i])))
            fu.values = np.ones(fu.cards)
            self.factors[i] = fu
        
        # ... and now brutishly (FIFO) we assign the factors
        for fu in listOfFactors:
            notUsed = True
            for i,n in enumerate(self.box):
                if n.issuperset(set(fu.variables)):
                    self.factors[i] = FactorOperations.multiply(self.factors[i], fu, False)
                    notUsed = False
                    break  # to use only once
            if notUsed:
                raise NameError('factor not used in any clique!', fu.variables)
        pass
    
    def computePath(self):
        start = None # choose an arbitrary leaf
        for i in range(self.V):
            if len(self.adj[i])==1:
                start = i
                break
        forward = BFS_Paths(self, start).discoveryPath
        forward.pop() # we dont need the last node
        
        temp = []
        marked = [False]*self.V
        for from_v in forward:
            # every node in the path is ensured to have received all necessary incoming messages
            neighbors = self.adj[from_v]
            marked[from_v] = True
            potentials = [n for n in neighbors if not marked[n]]
            if len(potentials)>1:
                raise Exception("SCREW-UP: too many targets")
            to_v = potentials[0]
            temp.append([from_v, to_v])
        
        edges = temp
        for e in reversed(temp):    
            edges.append(e[::-1])
        return edges
    
    def mssg(self, from_v, to_w, isMax=False):
        # collect all mssg arriving at v
        mess = []
        neighbors = self.adj[from_v]
        for n in neighbors:
            if n!=to_w:
                pos = self.adj[n].index(from_v)
                msg = self.delta[n][pos]
                mess.append(msg)

        # take the the initial Psi (and log if needed)
        d = copy.copy(self.factors[from_v])
        if isMax==True:
            d.values = np.log(d.values)

        # multiply/sum by incoming messages
        for ms in mess:
            if isMax==True:
                d = FactorOperations.sum(d, ms, False)
            else:
                d = FactorOperations.multiply(d, ms, True)

        # marginalized to setsep vars
        for n in d.variables:
            if n not in (self.box[from_v] & self.box[to_w]):
                if isMax==True:
                    d = FactorOperations.max_marginalize(d, n)
                else:
                    d = FactorOperations.marginalize(d, n)
        return d

    def calibrate(self, isMax=False):
        self.beta = [None]*self.V
        # compute messages
        for e in self.computePath():
            from_v, to_w = e
            pos_to = self.adj[from_v].index(to_w)
            self.delta[from_v][pos_to] = self.mssg(from_v, to_w, isMax)
        
        # compute the beliefs
        for v in range(self.V):
            belief = copy.copy(self.factors[v])
            if isMax==True:
                belief.values = np.log(belief.values)
            for w in self.adj[v]:
                pos = self.adj[w].index(v)
                delta = self.delta[w][pos]
                if isMax==True:
                    belief = FactorOperations.sum(belief, delta, False)
                else:
                    belief = FactorOperations.multiply(belief, delta, False)
            self.beta[v] = belief
            
    #def exactMarginals(self):


