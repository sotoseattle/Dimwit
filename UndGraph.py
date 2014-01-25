class UndGraph(object):
    '''Undirected Graph. We keep edges in a vertex indexed adjacency list
       From Princeton Coursera Algorithms Class'''

    def __init__(self, variable_groups):
        self.V = len(variable_groups)  # total vertices
        self.adj = [None]*self.V        # edges
        self.box = [None]*self.V        # variables
        for i,e in enumerate(variable_groups):
            self.box[i] = set(e)
        for i in range(self.V):
            self.adj[i] = []
        
    def addEdge(self, v, w):
        '''adds edge between vertices v and w'''
        if (v>=self.V) or (v<0) or (w>=self.V) or (w<0):
            raise Exception("Error: nodes out of bounds in graph", v, w, self.V)
        if v==w:
            return
        if w not in self.adj[v]:
            self.adj[v].insert(0, w)
        if v not in self.adj[w]:
            self.adj[w].insert(0, v);
        pass
            
    def index_var(self, var):
        for i in range(self.V):
            if var in self.box[i]:
                return i
        return None
        
    def connectAll(self, nodeList): # a utility method that comes up frequently
        abc = list(nodeList)
        while len(abc)>1:
            a = abc.pop()
            for b in abc:
                self.addEdge(a, b)
        pass
    
    def __str__(self):
        s = `self.V` + " vertices\n"
        for i, v in enumerate(self.box):
            s += "["+`i`+"] " + `list(v)` + ": " + `[list(self.box[e]) for e in self.adj[i]]` + "\n"
        return s
        
    