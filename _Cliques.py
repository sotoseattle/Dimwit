import numpy as np
import Factor
import Rvar
import FactorOperations


class FactorGraph(object):
    '''graph of linked random variables derived from a list of factors'''
    def __init__(self, listOfFactors):
        self.factors = listOfFactors

        # extract all variables. The index is the key for setting edges.
        cv = []
        for fu in self.factors:
            cv += fu.variables
        self.allVars = tuple(sorted(set(cv)))

        # create the adjacency matrix of the initial factor list
        numVars = len(self.allVars)
        self.edges = np.zeros((numVars, numVars)).astype('int32')
        for fu in self.factors:
            for vi in fu.variables:
                for vj in fu.variables:
                    self.edges[self.allVars.index(vi), self.allVars.index(vj)] = 1

    def firstMinNeighborVar(self):
        connections = tuple(np.sum(self.edges, axis=1))
        minE = float('+inf') 
        for e in connections:
            if e > 0 and e < minE:
                minE = e
        return self.allVars[connections.index(minE)]
    

class CliqueTree(object):    

    def __init__ (self, listOfFactors):  
        F = FactorGraph(listOfFactors)
        
        # create nodes iteratively through var elimination
        C = {'nodes':[], 'edges':np.zeros((0,0))}
        considered_cliques = 0
        while considered_cliques < len(F.allVars):
            z = F.firstMinNeighborVar()
            [F,C] = self.eliminateVar(F, C, z)
            considered_cliques += 1
        
        self.nodes = [set(n['vars']) for n in C['nodes']]
        self.edges = C['edges']

        # prune tree
        keepPruning = True
        while keepPruning:
            keepPruning = self.pruneNode()

        # initialize potentials first to all ones
        self.factors = []
        for i in range(len(self.nodes)):
            fu = Factor.Factor(sorted(list(self.nodes[i])))
            #fu.fill_values(np.ones(np.product(fu.cards)))
            fu.values = np.ones(fu.cards)
            self.factors += [fu]
        # ... and now brutishly (FIFO) we assign the factors
        for fu in listOfFactors:
            notUsed = True
            for i,n in enumerate(self.nodes):
                if set(fu.variables) <= n:
                    self.factors[i] = FactorOperations.multiply(self.factors[i], fu, False)
                    notUsed = False
                    break
            if notUsed:
                raise NameError('factor not used in any clique!', fu.variables)


    def eliminateVar(self, F, C, z):
        # separate factors into two lists, those that use z (F_Cluster) and the rest
        F_Cluster, F_Rest, Cluster_vars = [], [], []
        for f in F.factors:
            if z in f.variables:
                F_Cluster += [f]
                Cluster_vars += f.variables
            else:
                F_Rest += [f]
        
        if F_Cluster!=[]:
            Cluster_vars = tuple(sorted(set(Cluster_vars)))

            # when computing tau of new node, check if it uses other nodes' taus
            rows,cols = C['edges'].shape
            C['edges'] = np.vstack([C['edges'], np.zeros((1,cols))])
            C['edges'] = np.hstack([C['edges'], np.zeros((rows+1,1))])
            pos = np.zeros(cols+1)
            for n,node in enumerate(C['nodes']):
                if node['tau'] in F_Cluster:
                    pos[n]=1
            # create a new array of connecting node edges based on taus in common
            C['edges'][-1,:] = pos
            C['edges'][:,-1] = pos
            
            # multiply the factors in Cluster... (lambda) ...and marginalize by z (tau)
            tau = F_Cluster.pop(0)
            for f in F_Cluster:
                tau = FactorOperations.multiply(tau, f)
            if tau.variables != [z]:
                tau = FactorOperations.marginalize(tau, z)
            
            # add to unused factor list the resulting tau ==> new factor list with var eliminated
            F_Rest += [tau]
            
            # update the edges (connect all vars inside new cluster, & disconnect the eliminated variable)
            for vi in Cluster_vars:
                for vj in Cluster_vars:
                    F.edges[F.allVars.index(vi), F.allVars.index(vj)] = 1
            F.edges[F.allVars.index(z),:] = 0
            F.edges[:, F.allVars.index(z)] = 0
            
            C['nodes'] += [{'vars':Cluster_vars, 'tau':tau}]
            
            F.factors = F_Rest
        return [F, C]


    def pruneNode(self):
        '''Start with a node (A), scan through its neighbors to find one that is a superset of variables (B). 
           Cut off the edges conneselfed to A and add edges between B and all of A's other neighbors.'''
        for idx,nodeA in enumerate(self.nodes):
            neighbors = [j for j, e in enumerate(self.edges[idx,:]) if e==1]
            for neighbor_idx in neighbors:
                if nodeA < self.nodes[neighbor_idx]:  # striself variable subset
                    for k in neighbors:
                        if k!=neighbor_idx:
                            self.edges[neighbor_idx, k] = 1
                            self.edges[k, neighbor_idx] = 1
                    self.edges = np.delete(self.edges, idx, 0)  # delete row,col of edges for nodeA
                    self.edges = np.delete(self.edges, idx, 1)
                    self.nodes = np.delete(self.nodes, idx)      # delete nodeA
                    return True
        return False






    
###### TESTING #########
def testing():
    c = [None,3,3,3,3,3,3, 2,2,2,2,2,2]
    v = {x : Rvar.Rvar(x, c[x]) for x in range(1,13)}
    
    f = [None]*12
    f[0] = Factor.Factor([v[1]])
    f[0].fill_values([0.01, 0.18, 0.81])
    
    f[1] = Factor.Factor([v[2], v[1], v[3]])
    f[1].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])
    
    f[2] = Factor.Factor([v[3]])
    f[2].fill_values([0.01, 0.18, 0.81])
    
    f[3] = Factor.Factor([v[4], v[1], v[3]])
    f[3].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])
    
    f[4] = Factor.Factor([v[5], v[2], v[6]])
    f[4].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])
    
    f[5] = Factor.Factor([v[6]])
    f[5].fill_values([0.01, 0.18, 0.81])
    
    f[6] = Factor.Factor([v[7], v[1]])
    f[6].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
    
    f[7] = Factor.Factor([v[8], v[2]])
    f[7].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
    
    f[8] = Factor.Factor([v[9], v[3]])
    f[8].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
    
    f[9] = Factor.Factor([v[10], v[4]])
    f[9].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
    
    f[10] = Factor.Factor([v[11], v[5]])
    f[10].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
    
    f[11] = Factor.Factor([v[12], v[6]])
    f[11].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
    
    cc = CliqueTree(f)
    print cc.nodes
    print cc.edges
    for i,node in enumerate(cc.nodes):
        print node
        for j,link in enumerate(cc.edges[i,:]):
            if link==1:
                print ' --', cc.nodes[j]
    print cc.factors
    for fu in cc.factors:
        print fu.variables
        print fu.values


testing()