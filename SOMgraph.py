#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2013 12 11
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
    def __init__(self, smap, mask = None):
        self.smap = smap
        self.X,self.Y,self.dim = self.smap.shape
        self.graph = {}
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
        self.umat = SOMTools.getUmatrix(self.smap)
        self.localminima = numpy.asarray(SOMTools.detect_local_minima(self.umat)).T
        self.localminimagraph = {}
        if self.mask != None:
            self.localminima = numpy.asarray(filter(lambda e: not self.mask[e[0],e[1]], self.localminima))
        for e in itertools.combinations(self.localminima, 2):
            path = self.shortestPath(tuple(e[0]), tuple(e[1]))
            pathes.append(path)
            pathd = self.getPathDist(path)
            pathdists.append(pathd)
            self.updategraph(tuple(e[0]), tuple(e[1]), pathd, graph=self.localminimagraph)
        self.allPathes = pathes
        self.allPathDists = pathdists
        self.localminimagraph = self.symmetrize_edges(self.localminimagraph)
        return pathes

    def getLongestPath(self):
        """
        return the shortest path for the two most distant local minima
        """
        if not hasattr(self, 'allPathDists'):
            pathes = self.getAllPathes()
        return self.allPathes[numpy.argmax(self.allPathDists)]

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

    def priorityGraph(self, graph):
        """
        return a priority graph. Each sub dictionnary of the graph is a
        priority dictionnary as defined in priorityDictionary
        """
        G = graph
        Gp = {}
        for n1 in G.keys:
            d = priorityDictionary()
            for n2 in G[n1].keys:
                d[n2] = G[n1][n2]
            Gp[n1] = d
        return Gp
