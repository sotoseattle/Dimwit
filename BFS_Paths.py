from UndGraph import UndGraph

class BFS_Paths(object):
    '''Breadth First Search Algorithm. Decoupling the DAG from its processing
       From Princeton Coursera Algorithms Class'''
    
    def __init__(self, G, s):
        self.G = G
        self.source = s
        self.discoveryPath = []
        self.marked = [False]*self.G.V
        self.edgeTo = [None]*self.G.V
        self.distTo = [float("inf")]*self.G.V
        self.bfs(self.source)
    
    def bfs(self, v):
        self.distTo[self.source] = 0
        self.marked[self.source] = True
        queue = [self.source]
        while len(queue)>0:
            v = queue.pop()
            self.discoveryPath.append(v)
            for w in self.G.adj[v]:
                if self.marked[w]==False:
                    self.edgeTo[w] = v
                    self.distTo[w] = self.distTo[v] + 1
                    self.marked[w] = True
                    queue.insert(0, w)
        # the reverse way of discovery is the way to pass messages
        self.discoveryPath.reverse()
        pass
    
    def distance(self, v):
        return self.distTo[v]
    
    def pathTo(self, v):
        if self.marked[v]==False:
            return None
        else:
            path = [v]
            x = v
            while self.distTo[x] != 0: 
                x = self.edgeTo[x]
                path.insert(0,x)
            return path
