#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2014 01 08
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import SOMTools
import numpy
import scipy.spatial.distance
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228
from priodict import priorityDictionary
import itertools

class graph:
    def __init__(self, smap = None, mask = None, graph = None, umat = None):
        if smap != None:
            self.smap = smap
            self.X,self.Y,self.dim = self.smap.shape
            if umat == None:
                self.umat = SOMTools.getUmatrix(self.smap)
            else:
                self.umat = umat
        if graph == None:
            self.graph = {}
        else:
            self.graph = graph
        self.mask = mask

    def updategraph(self, n1, n2, d, graph=None):
        """
        update graph with node n1 and n2 and the distance d between n1 and n2
        """
        if graph == None:
            graph = self.graph
        i,j = n1
        u,v = n2
        try:
            graph[(i,j)].update({(u,v):d})
        except KeyError:
            graph[(i,j)] = {(u,v):d}

    def getgraph(self):
        if self.mask == None:
            self.mask = numpy.zeros((self.X, self.Y), dtype='bool')
        for i in range(self.X):
            for j in range(self.Y):
                if not self.mask[i,j]:
                    neighbors = SOMTools.getNeighbors((i,j), (self.X,self.Y))
                    for u,v in neighbors:
                        if not self.mask[u,v]:
                            d = scipy.spatial.distance.euclidean(self.smap[i,j], self.smap[u,v])
#                            d = self.umat[u,v]
                            self.updategraph((i,j), (u,v), d)


    def Dijkstra(self, G, start, end=None):
        """
        Dijkstra's algorithm for shortest paths
        David Eppstein, UC Irvine, 4 April 2002
        Find shortest paths from the start vertex to all
        vertices nearer than or equal to the end.

        The input graph G is assumed to have the following
        representation: A vertex can be any object that can
        be used as an index into a dictionary.  G is a
        dictionary, indexed by vertices.  For any vertex v,
        G[v] is itself a dictionary, indexed by the neighbors
        of v.  For any edge v->w, G[v][w] is the length of
        the edge.  This is related to the representation in
        <http://www.python.org/doc/essays/graphs.html>
        where Guido van Rossum suggests representing graphs
        as dictionaries mapping vertices to lists of neighbors,
        however dictionaries of edges have many advantages
        over lists: they can store extra information (here,
        the lengths), they support fast existence tests,
        and they allow easy modification of the graph by edge
        insertion and removal.  Such modifications are not
        needed here but are important in other graph algorithms.
        Since dictionaries obey iterator protocol, a graph
        represented as described here could be handed without
        modification to an algorithm using Guido's representation.

        Of course, G and G[v] need not be Python dict objects;
        they can be any other object that obeys dict protocol,
        for instance a wrapper in which vertices are URLs
        and a call to G[v] loads the web page and finds its links.
        
        The output is a pair (D,P) where D[v] is the distance
        from start to v and P[v] is the predecessor of v along
        the shortest path from s to v.
        
        Dijkstra's algorithm is only guaranteed to work correctly
        when all edge lengths are positive. This code does not
        verify this property for all edges (only the edges seen
        before the end vertex is reached), but will correctly
        compute shortest paths even for some graphs with negative
        edges, and will raise an exception if it discovers that
        a negative edge has caused it to make a mistake.
        """
        D = {}  # dictionary of final distances
        P = {}  # dictionary of predecessors
        Q = priorityDictionary()   # est.dist. of non-final vert.
        Q[start] = 0
        for v in Q:
            D[v] = Q[v]
            if v == end: break
            for w in G[v]:
                vwLength = D[v] + G[v][w]
                if w in D:
                    if vwLength < D[w]:
                        raise ValueError, \
      "Dijkstra: found better path to already-final vertex"
                elif w not in Q or vwLength < Q[w]:
                    Q[w] = vwLength
                    P[w] = v
        return (D,P)
                
    def shortestPath(self, start, end, graph=None):
        """
        Find a single shortest path from the given start vertex
        to the given end vertex.
        The input has the same conventions as Dijkstra().
        The output is a list of the vertices in order along
        the shortest path.
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        D,P = self.Dijkstra(G,start,end)
        Path = []
        while 1:
            Path.append(end)
            if end == start: break
            end = P[end]
        Path.reverse()
        return Path

    def getPathDist(self, path):
        """
        return the distance for a given path in the graph. Path is a list of
        node
        """
        d = 0
        for e in zip(path, path[1:]):
            d += self.graph[e[0]][e[1]]
        return d

    def getAllPathes(self):
        """
        return all pathes for all combinations of local minima
        """
        pathes = []
        pathdists = []
        self.localminima = numpy.asarray(SOMTools.detect_local_minima2(self.umat)).T
        self.localminimagraph = {}
        if self.mask != None:
            self.localminima = numpy.asarray(filter(lambda e: not self.mask[e[0],e[1]], self.localminima))
        for e in itertools.permutations(self.localminima, 2):
            path = self.shortestPath(tuple(e[0]), tuple(e[1]))
            pathes.append(path)
            pathd = self.getPathDist(path)
            pathdists.append(pathd)
            self.updategraph(tuple(e[0]), tuple(e[1]), pathd, graph=self.localminimagraph)
        self.allPathes = pathes
        self.allPathDists = pathdists
        self.localminimagraph = self.symmetrize_edges(self.localminimagraph)
        return pathes

    def getLongestPath(self, localmin=False):
        """
        return the shortest path for the two most distant local minima
        If localmin is set to True the path goes through local minima
        """
        if not hasattr(self, 'allPathDists'):
            pathes = self.getAllPathes()
        longestpath = self.allPathes[numpy.argmax(self.allPathDists)]
        if not localmin:
            return longestpath
        else:
            if not hasattr(self, 'mingraph'):
                mingraph = self.clean_graph()
            longestpath = self.shortestPath(longestpath[0], longestpath[-1], self.mingraph)
            return longestpath

    def has_edge(self, n1, n2, graph=None):
        """
        test the existence of a edge n1-n2 in a graph
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        if G.has_key(n1):
            return G[n1].has_key(n2)
        else:
            return False

    def symmetrize_edges(self, graph=None):
        """
        symmetrize the edges of a graph: If an edge n1->n2 exists and n2->n1
        does not. The function return a graph with symmetric edges n1<->n2
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        for n1 in G.keys():
            for n2 in G[n1].keys():
                if not self.has_edge(n2, n1, G):
                    self.updategraph(n2, n1, G[n1][n2], G)
        return G

    def delete_edge(self, n1, n2, graph):
        """
        delete an edge n1 -> n2 from a graph
        """
        del graph[n1][n2]

    def unsymmetrize_edges(self, graph=None):
        """
        symmetrize the edges of a graph: If an edge n1->n2 exists and n2->n1
        does not. The function return a graph with symmetric edges n1<->n2
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        for n1 in G.keys():
            for n2 in G[n1].keys():
                if self.has_edge(n2, n1, G):
                    self.delete_edge(n2, n1, G)
        return G

    def priorityGraph(self, graph=None):
        """
        return a priority graph. Each sub dictionnary of the graph is a
        priority dictionnary as defined in priorityDictionary
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        Gp = {}
        for n1 in G.keys():
            d = priorityDictionary()
            for n2 in G[n1].keys():
                d[n2] = G[n1][n2]
            Gp[n1] = d
        return Gp

    def n_edges(self, graph=None):
        """
        return the number of edges of a graph
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        nedges = 0
        for n1 in G.keys():
            for n2 in G[n1].keys():
                nedges += 1
        return nedges

    def get_vertices(self, graph=None):
        """
        return the list of vertices in a graph
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        vertlist = []
        for n1 in G.keys():
            if n1 not in vertlist:
                vertlist.append(n1)
            for n2 in G[n1].keys():
                if n2 not in vertlist:
                    vertlist.append(n2)
        return vertlist

    def get_distances(self, graph=None):
        """
        return the list of unique distances in a graph
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        d = []
        for n1 in G.keys():
            d.extend(G[n1].values())
        return numpy.unique(d)

    def get_smallest_edge(self, graph=None):
        """
        return the two vertices constituting the smallest vertex
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        min_d = numpy.inf
        for n1 in G.keys():
            for n2 in G[n1].keys():
                if G[n1][n2] < min_d:
                    min_d = G[n1][n2]
                    min_n1 = n1
                    min_n2 = n2
        return min_n1, min_n2

    def mergegraph(self, graph1, graph2):
        """
        merge two graphes
        """
        mgraph = {}
        for v1 in graph1.keys():
            for v2 in graph1[v1].keys():
                self.updategraph(v1, v2, graph1[v1][v2], mgraph)
        for v1 in graph2.keys():
            for v2 in graph2[v1].keys():
                self.updategraph(v1, v2, graph2[v1][v2], mgraph)
        return mgraph

    def select_edges(self, threshold, graph=None, min_d=None):
        """
        return edges with distance less than threshold and more than min_d if
        min_d is not None.
        """
        if graph == None:
            G = self.get_graph_iterator(self.graph)
        else:
            G = self.get_graph_iterator(graph)
        if min_d == None:
            min_d = -numpy.inf
        outgraph = {}
        for n1 in G.keys():
            n2 = G[n1].next()
            d = graph[n1][n2]
            while d <= threshold and d >= min_d:
                self.updategraph(n1, n2, d, outgraph)
                try:
                    n2 = G[n1].next()
                    d = graph[n1][n2]
                except StopIteration:
                    break
        return outgraph


    def get_graph_iterator(self, graph=None):
        """
        get an iterator as defined in priodict for each vertex
        """
        if graph == None:
            G = self.priorityGraph(self.graph)
        else:
            G = self.priorityGraph(graph)
        for key in G.keys():
            G[key] = G[key].__iter__()
        return G

    def plot_graph(self, graph, color='m', plotkeys=False, plotpath=False, plotnode=False):
        """
        plot the graph with matplotlib.pyplot. If plotpath is True plot the
        shortest path for edges
        """
        import matplotlib.pyplot
        G = graph
        if plotkeys:
            plottedkeys = []
        for n1 in G.keys():
            if plotkeys:
                if n1 not in plottedkeys:
                    plottedkeys.append(n1)
                    matplotlib.pyplot.annotate(n1, list(n1)[::-1])
            if plotnode:
                matplotlib.pyplot.scatter(n1[1], n1[0], color=color)
            for n2 in G[n1].keys():
                v = numpy.asarray((n1,n2))
                if plotpath:
                    path = numpy.asarray(self.shortestPath(n1, n2, graph=None))
                    matplotlib.pyplot.plot(path[:,1],path[:,0], color)
                else:
                    matplotlib.pyplot.plot(v[:,1],v[:,0], color)
                if plotkeys:
                    if n2 not in plottedkeys:
                        plottedkeys.append(n2)
                        matplotlib.pyplot.annotate(n2, list(n2)[::-1])
                if plotnode:
                    matplotlib.pyplot.scatter(n2[1], n2[0], color=color)

    def splitgraph(self, graph):
        """
        split a graph in not connected subgraphes
        """
        G = graph
        verts = self.get_vertices(G)
        n1 = verts[0]
        n1s = []
        visited = []
        n1s.append(n1)
        visited.append(n1)
        subgraph = {}
        subgraphes = []
        while 1:
            for n2 in G[n1]:
                d = G[n1][n2]
                self.updategraph(n1, n2, d, subgraph)
                if G.has_key(n2):
                    if n2 not in visited:
                        n1s.append(n2)
                        visited.append(n2)
            if len(n1s) == 0:
                subgraphes.append(subgraph)
                subgraph = {}
                verts = list(set(verts) - set(visited))
                if len(verts) == 0:
                    break
                n1 = verts[0]
            else:
                n1 = n1s.pop()
        return subgraphes

    def get_graph_distance(self, graph1, graph2):
        """
        return the distance between two graphes. The distance is the smallest
        distance between two nodes of each graph
        """
        if not hasattr(self, 'localminimagraph'):
            self.getAllPathes()
        verts1 = self.get_vertices(graph1)
        verts2 = self.get_vertices(graph2)
        dmin = numpy.inf
        for n1 in verts1:
            for n2 in verts2:
                try:
                    d = self.localminimagraph[n1][n2]
                except KeyError:
                    raise KeyError("Undefined edge %s -> %s"%(n1,n2))
                if d < dmin:
                    n1min, n2min, dmin = n1, n2, d
        return n1min, n2min, dmin


    def clean_graph(self, graph=None):
        """
        remove long range edges in a graph
        """
        if graph == None:
            if not hasattr(self, 'localminimagraph'):
                self.getAllPathes()
            G = self.localminimagraph
        else:
            G = graph
        ds = self.get_distances(G)
        nvertmax = len(self.get_vertices(G))
        nvert_prev = -1
        min_d = -numpy.inf
        subgraph_prev = {}
        for d in ds:
            subgraph = self.select_edges(d, G, min_d=min_d)
            nvert = len(self.get_vertices(subgraph))
            if nvert == nvert_prev:
                subgraph = subgraph_prev
                min_d = d
            else:
                subgraph = self.mergegraph(subgraph_prev, subgraph)
                subgraph_prev = subgraph
                nvert_prev = nvert
            if nvert == nvertmax:
                break
        subgraph = self.symmetrize_edges(subgraph)
        splitgraph = self.splitgraph(subgraph)
        ngraph = len(splitgraph)
        while ngraph != 1:
            for i in range(ngraph):
                dmin = numpy.inf
                for j in range(ngraph):
                    if i != j:
                        g1 = splitgraph[i]
                        g2 = splitgraph[j]
                        n1, n2, d = self.get_graph_distance(g1,g2)
                        if d < dmin:
                            n1min, n2min, dmin = n1, n2, d
                self.updategraph(n1min, n2min, dmin, subgraph)
            subgraph = self.symmetrize_edges(subgraph)
            splitgraph = self.splitgraph(subgraph)
            ngraph = len(splitgraph)
        self.mingraph = subgraph
        return subgraph

    def adjacency_matrix(self, graph):
        verts = self.get_vertices(graph)
        vertdict = {}
        for i, vert in enumerate(verts):
            vertdict[vert] = i
        A = numpy.zeros((len(verts), len(verts)))
        for n1 in graph.keys():
            for n2 in graph[n1].keys():
                i,j = vertdict[n1], vertdict[n2]
                A[i,j] = graph[n1][n2]
        return A, vertdict

    def fruchterman_reingold(self, graph, dim=2, pos=None, fixed=None, iterations=50):
        """
        Position nodes in adjacency matrix A using Fruchterman-Reingold
        Entry point for NetworkX graph is fruchterman_reingold_layout()
        fixed is a list of vertices to keep fixed
        function adapted from networkx: http://networkx.github.io/
        """
        
        A, vertdict = self.adjacency_matrix(graph) # get adjacency matrix and dictionnary of vertices
        A = numpy.asarray(A!=0, dtype=float)

        try:
            nnodes,_=A.shape
        except AttributeError:
            raise AttributeError(
                "fruchterman_reingold() takes an adjacency matrix as input")

        if fixed != None:
            select = numpy.zeros(len(vertdict.keys()), dtype=bool)
            for n in vertdict.keys():
                if n in fixed:
                    select[vertdict[n]] = True
            fixed = select

        returngraph = False
        if (numpy.asarray([type(e) for e in vertdict.keys()]) == type((0,))).all(): # If the vertices of a graph are positions,
            pos = numpy.zeros((len(vertdict.keys()), dim), dtype=float)
            for n in vertdict.keys():
                pos[vertdict[n]] = n # read the initial positions from the graph
            returngraph = True # and return a graph at the end, with the new positions
        elif pos==None:
            # random initial positions
            pos=numpy.asarray(numpy.random.random((nnodes,dim)),dtype=A.dtype)
        else:
            # make sure positions are of same type as matrix
            pos=pos.astype(A.dtype)

        # optimal distance between nodes
        area = pos.ptp(axis=0).prod()
        k=numpy.sqrt(area/nnodes)
        # the initial "temperature"  is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        t=0.1
        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        dt=t/float(iterations+1)
        delta = numpy.zeros((pos.shape[0],pos.shape[0],pos.shape[1]),dtype=A.dtype)
        # the inscrutable (but fast) version
        # this is still O(V^2)
        # could use multilevel methods to speed this up significantly
        for iteration in range(iterations):
            # matrix of difference between points
            for i in range(pos.shape[1]):
                delta[:,:,i]= pos[:,i,None]-pos[:,i]
            # distance between points
            distance=numpy.sqrt((delta**2).sum(axis=-1))
            # enforce minimum distance of 0.01
            distance=numpy.where(distance<0.01,0.01,distance)
            # displacement "force"
            displacement=numpy.transpose(numpy.transpose(delta)*\
                                      (k*k/distance**2-A*distance/k))\
                                      .sum(axis=1)
            # update positions
            length=numpy.sqrt((displacement**2).sum(axis=1))
            length=numpy.where(length<0.01,0.1,length)
            delta_pos=numpy.transpose(numpy.transpose(displacement)*t/length)
            if fixed is not None:
                # don't change positions of fixed nodes
                delta_pos[fixed]=0.0
            pos+=delta_pos
            # cool temperature
            t-=dt
        if returngraph:
            outgraph = {}
            for n1 in graph.keys():
                for n2 in graph[n1].keys():
                    nn1 = tuple(pos[vertdict[n1]]) # new node 1
                    nn2 = tuple(pos[vertdict[n2]]) # new node 2
                    d = graph[n1][n2]
                    self.updategraph(nn1, nn2, d, graph=outgraph)
            return outgraph
        else:
            return pos
