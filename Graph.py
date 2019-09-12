#!/usr/bin/env python
# -*- coding: UTF8 -*-

import itertools

import numpy
import scipy.spatial.distance
import scipy.ndimage
import networkx
import community
import numbers

class node(object):
    def __init__(self, index, distance, parent):
        self.index = index
        self.distance = distance
        self.parent = parent

class Graph:
    def __init__(self, adjacency_matrix=None, smap=None):
        self.adjacency_matrix = adjacency_matrix
        self.smap = smap
        self.is_complex = False
        self.metric = lambda u, v: scipy.spatial.distance.euclidean(u, v)
        if self.smap is not None:
            if self.smap[0, 0].dtype == numpy.asarray(numpy.complex(1, 1)).dtype:
                self.is_complex = True
                print "Complex numbers space"
                self.metric = lambda u, v: numpy.sqrt(( numpy.abs(u - v) ** 2 ).sum())  # metric for complex numbers
        if self.adjacency_matrix is None and not (self.smap is None):
            self.get_adjacency_matrix()
        self.change_of_basis = None
        self.community_map = None
        self.unfolded_umat = None
        self.minimum_spanning_tree = None
        self.local_minima = None # Local minima of the U-matrix in the unfolded
                                 # space
        if self.smap is None:
            n = self.adjacency_matrix.shape[0]
            self.smap = numpy.zeros((n, 1))

    def get_adjacency_matrix(self):
        """

        return the adjacency matrix for the given SOM map (self.smap)
        """
        nx, ny, nz = self.smap.shape
        adjacency_matrix = numpy.ones((nx * ny, nx * ny)) * numpy.inf
        for i in range(nx):
            for j in range(ny):
                ravel_index = numpy.ravel_multi_index((i, j), (nx, ny))
                neighbor_indices = self.neighbor_dim2_toric((i, j), (nx, ny))
                for neighbor_index in neighbor_indices:
                    ravel_index_2 = numpy.ravel_multi_index(neighbor_index, (nx, ny))
                    distance = self.metric(self.smap[i, j], self.smap[neighbor_index])
                    adjacency_matrix[ravel_index, ravel_index_2] = distance
        self.adjacency_matrix = adjacency_matrix

    def get_laplacian_matrix(self, boolean=True):
        """
        return the laplacian matrix for the graph
        """
        if self.minimum_spanning_tree is None:
            self.get_minimum_spanning_tree()
        if not boolean:
            L = - 1 / self.minimum_spanning_tree
            for i in range(L.shape[0]):
                L[i,i] = - L[i,:].sum()
        else:
            A = 1 - numpy.isinf(self.minimum_spanning_tree)
            D = numpy.identity(A.shape[0])*A.sum(axis=0)
            L = D - A
        return L

    def neighbor_dim2_toric(self, p, s):
        """Efficient toric neighborhood function for 2D SOM.
        """
        x, y = p
        X, Y = s
        xm = (x - 1) % X
        ym = (y - 1) % Y
        xp = (x + 1) % X
        yp = (y + 1) % Y
        return [(xm, ym), (xm, y), (xm, yp), (x, ym), (x, yp), (xp, ym), (xp, y), (xp, yp)]

    def add_edge(self, graph, n1, n2, w):
        try:
            graph[n1].update({n2: w})
        except KeyError:
            graph[n1] = {n2: w}
        try:
            graph[n2].update({n1: w})
        except KeyError:
            graph[n2] = {n1: w}

    def get_graph(self, adjacency_matrix=None):
        graph = {}
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix
        assert isinstance(adjacency_matrix, numpy.ndarray)
        nx, ny = adjacency_matrix.shape
        for index in itertools.combinations(range(nx), 2):
            (i, j) = index
            weight = adjacency_matrix[i, j]
            if weight != numpy.inf:
                self.add_edge(graph, i, j, weight)
        return graph

    def make_sets(self):
        self.sets = []
        for v in range(len(self.adjacency_matrix)):
            self.sets.append({v})

    def find_set(self, u):
        for index, s in enumerate(self.sets):
            if u in s:
                break
        return index

    def union(self, index_set1, index_set2):
        indices = [index_set1, index_set2]
        indices.sort(reverse=True)
        sets = []
        for index in indices:
            sets.append(self.sets.pop(index))
        self.sets.append(sets[0].union(sets[1]))

    def get_minimum_spanning_tree(self):
        """
        Kruskal's algorithm
        """
        self.make_sets()
        minimum_spanning_tree = numpy.ones(self.adjacency_matrix.shape) * numpy.inf
        sorter = self.adjacency_matrix.flatten().argsort()
        nx, ny = self.adjacency_matrix.shape
        sorter = sorter[self.adjacency_matrix.flatten()[sorter] != numpy.inf]
        for i in sorter:
            (u, v) = numpy.unravel_index(i, (nx, ny))
            index_set1 = self.find_set(u)
            index_set2 = self.find_set(v)
            if index_set1 != index_set2:
                self.union(index_set1, index_set2)
                minimum_spanning_tree[u, v] = self.adjacency_matrix[u, v]
                minimum_spanning_tree[v, u] = self.adjacency_matrix[u, v]
        self.minimum_spanning_tree = minimum_spanning_tree

    @property
    def umatrix(self):
        """
        compute the umatrix from the adjacency matrix
        """
        if self.adjacency_matrix is None:
            self.get_adjacency_matrix()
        nx, ny = self.adjacency_matrix.shape
        umat_shape = self.smap.shape[:-1]
        umat = numpy.ma.filled(numpy.mean(numpy.ma.masked_array(self.adjacency_matrix,
                                                                self.adjacency_matrix == numpy.inf),
                                          axis=0)).reshape(umat_shape)
        return umat

    def bfs(self, root, pdist=None):
        """
        Breadth-First-Search on the minimum spanning tree
        See:
        https://en.wikipedia.org/wiki/Breadth-first_search
        • pdist: pairwise distance matrix between nodes (shape: n*n, with n the
        number of nodes). If not None, use the distances given in the matrix to
        compute the distance from the root.
        """
        if self.minimum_spanning_tree is None:
            self.get_minimum_spanning_tree()
        mstree = self.get_graph(adjacency_matrix=self.minimum_spanning_tree)
        node_list = range(self.minimum_spanning_tree.shape[0])
        nodes = {e:node(e, numpy.inf, None) for e in node_list}
        Q = []
        root = nodes[root]
        root.distance = 0
        Q.append(root)
        while len(Q) > 0:
            current = Q.pop()
            for i in mstree[current.index].keys():
                n = nodes[i]
                if numpy.isinf(n.distance):
                    if pdist is None:
                        n.distance = current.distance + 1
                    else:
                        n.distance = current.distance + pdist[current.index,
                                                              n.index]
                    n.parent = current
                    Q.append(n)
        # Sort nodes by ascending distances:
        nodes = numpy.asarray(nodes.values())
        selection = numpy.bool_(1-numpy.isinf([n.distance for n in nodes]))
        nodes = nodes[selection]
        sorter = numpy.argsort([n.distance for n in nodes])
        nodes = numpy.asarray(nodes)[sorter]
        return nodes

    def dijkstra(self, starting_cell = None, break_at_local_min = False,
                 get_predecessors = False, max_distance=numpy.inf):
        """

        Apply dijkstra distance transform to the SOM map. If break_at_local_min
        the dijkstra algorithm breaks at local minima: usefull to detect
        local_minima and basins

        If get_predecessors is True, the function returns also the dictionnary
        of predecessor. That is usefull to compute the shortest path.

        • max_distance: If not inf: break the dijkstra when the maximal
        distance reach the given value

        """
        if isinstance(starting_cell, numbers.Integral):
            starting_cell = (starting_cell, 0)
        if self.minimum_spanning_tree is None:
            self.get_minimum_spanning_tree()
        if break_at_local_min:
            umat = self.umatrix.flatten()
        ms_tree = self.minimum_spanning_tree
        nx, ny = self.smap.shape[:2]
        nx2, ny2 = ms_tree.shape
        visit_mask = numpy.zeros(nx2, dtype=bool)
        m = numpy.ones(nx2) * numpy.inf
        if starting_cell is None:
            cc = numpy.unravel_index(ms_tree.argmin(), (nx2, ny2))[0]  # current cell
        else:
            cc = numpy.ravel_multi_index(starting_cell, (nx, ny))
        m[cc] = 0
        n_visited_cell = -1
        P = {}  # dictionary of predecessors
        while (~visit_mask).sum() > 0:
            neighbors = [e for e in numpy.where(ms_tree[cc] != numpy.inf)[0] if not visit_mask[e]]
            for e in neighbors:
                d = ms_tree[cc, e] + m[cc]
                if d < m[e]:
                    m[e] = d
                    P[e] = cc
            if m[e] > max_distance:
                break
            visit_mask[cc] = True
            m_masked = numpy.ma.masked_array(m, visit_mask)
            if n_visited_cell == visit_mask.sum():
                break # break the loop if no more visited cells
                # Usefull when there is unreachable cells
            else:
                n_visited_cell = visit_mask.sum()
            if break_at_local_min:
                u_value_prev = umat[cc]
            cc = m_masked.argmin()
            if break_at_local_min and umat[cc] > u_value_prev:
                break
        if get_predecessors:
            return m.reshape((nx, ny)), P
        else:
            return m.reshape((nx, ny))

    def shortestPath(self, start, end):
        """
        Find a single shortest path from the given start vertex
        to the given end vertex.
        The output is a list of the vertices in order along
        the shortest path.
        """
        if isinstance(start, numbers.Integral):
            start = (start, 0)
        if isinstance(end, numbers.Integral):
            end = (end, 0)
        D,P = self.dijkstra(starting_cell=start, get_predecessors=True)
        Path = []
        nx, ny = self.smap.shape[:2]
        start = numpy.ravel_multi_index(start, (nx,ny))
        end = numpy.ravel_multi_index(end, (nx,ny))
        while 1:
            Path.append(end)
            if end == start: break
            end = P[end]
        Path.reverse()
        return Path

    @staticmethod
    def get_neighbors_of_set(indices):
        """
        return the neighboring indices of the given indices [(i1,j1), (i2,j2), (i3,j3), ...]
        """
        indices = numpy.asarray(indices)
        min_i = indices.min(axis=0)
        indices = indices - min_i + (1, 1)
        max_i = indices.max(axis=0)
        m = numpy.zeros(max_i + (2, 2))
        m[indices[:, 0], indices[:, 1]] = True
        d = scipy.ndimage.morphology.binary_dilation(m,
                                                     structure=scipy.ndimage.morphology.generate_binary_structure(2, 1))
        delta = numpy.logical_and(d, 1 - m)
        return numpy.asarray(numpy.where(delta)).T + min_i - (1, 1)

    def unfold_smap(self):
        """
        Unfold the SOM map self.smap
        """
        m = self.dijkstra()
        nx, ny = m.shape
        cc = numpy.asarray(numpy.where(m == 0)).T[0]
        change_of_basis = {}
        for i in range(nx * ny):
            d_min = numpy.inf
            change_of_basis[tuple(cc % (nx, ny))] = tuple(cc)
            m[tuple(cc % (nx, ny))] = numpy.inf
            for e in numpy.asarray([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]) + cc:
                d = m[tuple(e % (nx, ny))]
                if d < d_min:
                    d_min = d
            neighbors_of_set = self.get_neighbors_of_set(change_of_basis.values())
            cc = neighbors_of_set[m[neighbors_of_set[:, 0] % nx, neighbors_of_set[:, 1] % ny].argmin()]
        values = change_of_basis.values()
        min_values = numpy.asarray(values).min(axis=0)
        unfolded_shape = list(numpy.ptp(values, axis=0) + [1, 1]) + [self.smap.shape[-1]]
        unfolded_smap = numpy.empty(unfolded_shape, dtype=type(self.smap[0, 0, 0]))
        umat = self.umatrix
        unfolded_umat = numpy.ones(unfolded_shape[:-1]) * numpy.nan
        for k in change_of_basis.keys():
            t = change_of_basis[k]  # tuple
            t = tuple(numpy.asarray(t, dtype=int) - min_values)
            change_of_basis[k] = t
            unfolded_smap[t] = self.smap[k]
            unfolded_umat[t] = umat[k]
        self.unfolded_smap = unfolded_smap
        self.unfolded_umat = unfolded_umat
        self.change_of_basis = change_of_basis

    def detect_local_minima(self):
        """
        Compute the local minima of the U-matrix in the unfolded space
        """
        if self.change_of_basis is None:
            self.unfold_smap
        arr = self.umatrix
        X,Y = arr.shape
        lminima = []
        for i in range(X):
            for j in range(Y):
                pos = (i,j)
                neighbors = self.neighbor_dim2_toric(pos, (X,Y))
                nvalues = numpy.asarray( [ arr[e[0],e[1]] for e in neighbors] )
                if (arr[i,j] <= nvalues).all():
                    lminima.append((i,j))
        lminima = [self.change_of_basis[e] for e in lminima] # to be in the
                                                # basis of the unfolded map
        lminima = numpy.asarray(lminima)
        lminima = (lminima[:,0], lminima[:,1])
        self.local_minima = lminima

    def unfold_matrix(self, matrix):
        """
        unfold the given matrix given self.change_of_basis
        """
        if self.change_of_basis is None:
            self.unfold_smap()
        if len(matrix.shape) == 2: # 2D array
            unfolded_matrix = numpy.ones_like(self.unfolded_umat) * numpy.nan
        else: # 3D array
            out_shape = self.unfolded_umat.shape + (matrix[0,0,:].size,)
            unfolded_matrix = numpy.ones(out_shape) * numpy.nan
        for k in self.change_of_basis.keys():
            t = self.change_of_basis[k]  # tuple
            unfolded_matrix[t] = matrix[k]
        return unfolded_matrix

    def fold_matrix(self, matrix):
        """
        fold the given matrix given self.change_of_basis
        """
        if self.change_of_basis is None:
            self.unfold_smap()
        folding_dict = {v:k for (k,v) in self.change_of_basis.items()}
        if len(matrix.shape) == len(self.smap.shape[:-1]): # for 2D array
            folded_matrix = numpy.ones(self.smap.shape[:-1]) * numpy.nan
        else:
            folded_matrix = numpy.ones(self.smap.shape[:-1] + (matrix.shape[-1],)) * numpy.nan
        for k in folding_dict.keys():
            t = folding_dict[k] # tuple
            folded_matrix[t] = matrix[k]
        return folded_matrix


    def best_partition(self):
        print "computing communities maximizing modularity"
        gnx = networkx.Graph()
        nx, ny = self.adjacency_matrix.shape
        for index in itertools.combinations(range(nx), 2):
            (i, j) = index
            weight = self.adjacency_matrix[i, j]
            if weight != numpy.inf and weight != 0:
                gnx.add_edge(i, j, weight=weight)
        communities = community.best_partition(gnx)
        if self.smap is not None:
            shape = self.smap.shape[:-1]
        else:
            shape = (nx,1)
        community_map = numpy.ones(shape) * numpy.nan
        for k in communities.keys():
            ij = numpy.unravel_index(k, shape)
            community_map[ij] = communities[k]
        self.community_map = community_map

    def write_GML(self, outfilename, graph=None, directed_graph=False, **kwargs):
        """
        Write gml file for ugraph.

        - isomap_layout: embed isomap coordinates of the 2D embedded space in the gml output file

        **kwargs: data to write for each node.  Typically, these data are
        obtained from the self.project function. The keys of the kwargs are
        used as keys in the GML file
        """
        if graph is None:
            if self.minimum_spanning_tree is None:
                self.get_minimum_spanning_tree()
            ms_tree = self.minimum_spanning_tree
            graph = self.get_graph(ms_tree)
        outfile = open(outfilename, 'w')
        outfile.write('graph [\n')
        if directed_graph:
            outfile.write('directed 1\n')
        else:
            outfile.write('directed 0\n')
        nodes = graph.keys()
        for n in nodes:
            outfile.write('node [ id %d\n' % n)
            for key in kwargs.keys():
                try:
                    outfile.write('%s %.4f\n' % (key, kwargs[key][n]))
                except KeyError:
                    print "no %s for node %d" % (key, n)
                    pass
            outfile.write(']\n')
        for n1 in graph.keys():
            for n2 in graph[n1].keys():
                d = graph[n1][n2]
                outfile.write('edge [ source %d target %d weight %.4f\n' % (n1, n2, d))
                outfile.write(']\n')
        outfile.write(']')
        outfile.close()
