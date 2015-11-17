#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2015-11-06 16:40:02 (UTC+0100)

import numpy
import progress_reporting as Progress

class DDclust:
    def __init__(self, experimental_data, minimum_spanning_tree, feature_map, change_of_basis):
        """

        • experimental_data: filename containing the experimental data:
            ‣ First column: experimental intensities
            ‣ Second column: experimental standard deviation (std)

        • minimum_spanning_tree: the minimum spanning tree of the SOM map

        • feature_map: the map containing the data to compare with experimental
        ones. The map is given in the unfolded space.

        • change_of_basis: the dictionnary to fold or unfold the map

        All the calculations are made in the folded space

        Return the cluster map in the unfolded space.

        """
        self.minimum_spanning_tree = minimum_spanning_tree
        self.unfold = change_of_basis # to unfold the cell indexes
        self.fold = {v:k for k, v in self.unfold.iteritems()} # to fold the cell indexes
        self.folded_shape = tuple(numpy.asarray(self.unfold.keys()).max(axis=0) + 1)
        self.unfolded_shape = tuple(numpy.asarray(self.fold.keys()).max(axis=0) + 1)
        self.feature_dim = feature_map.shape[-1]
        self.feature_map = self.fold_matrix(feature_map)
        experimental_data = numpy.genfromtxt(experimental_data)
        self.experimental_intensities = experimental_data[:,0]
        self.experimental_std = experimental_data[:,1]
        if (self.experimental_std == 0).all():
            self.experimental_std = numpy.ones_like(self.experimental_std)

    def unfold_matrix(self, matrix):
        """
        unfold the given matrix given self.unfold
        """
        if len(matrix.shape) > 2:
            unfolded_matrix = numpy.ones(self.unfolded_shape +
                                         (self.feature_dim,)) * numpy.nan
        else:
             unfolded_matrix = numpy.ones(self.unfolded_shape) * numpy.nan
        for k in self.unfold.keys():
            t = self.unfold[k]  # tuple
            unfolded_matrix[t] = matrix[k]
        return unfolded_matrix

    def fold_matrix(self, matrix):
        """
        fold the given matrix given self.fold
        """
        folded_matrix = numpy.ones(self.folded_shape + (self.feature_dim,)) * numpy.nan
        for k in self.fold.keys():
            t = self.fold[k]  # tuple
            folded_matrix[t] = matrix[k]
        return folded_matrix

    def dijkstra(self, starting_cell = None, threshold = numpy.inf):
        """

        Apply dijkstra distance transform to the SOM map.
        threshold: interactive threshold for local clustering

        """
        ms_tree = self.minimum_spanning_tree
        nx, ny = self.folded_shape
        nx2, ny2 = ms_tree.shape
        visit_mask = numpy.zeros(nx2, dtype=bool)
        m = numpy.ones(nx2) * numpy.inf
        if starting_cell is None:
            cc = numpy.unravel_index(ms_tree.argmin(), (nx2, ny2))[0]  # current cell
        else:
            cc = numpy.ravel_multi_index(starting_cell, (nx, ny))
        m[cc] = 0
        while (~visit_mask).sum() > 0:
            neighbors = [e for e in numpy.where(ms_tree[cc] != numpy.inf)[0] if not visit_mask[e]]
            for e in neighbors:
                d = ms_tree[cc, e] + m[cc]
                if d < m[e]:
                    m[e] = d
            visit_mask[cc] = True
            m_masked = numpy.ma.masked_array(m, visit_mask)
            cc = m_masked.argmin()
            if m[m != numpy.inf].max() > threshold:
                break
        return m.reshape((nx, ny))

    def get_chi(self, feature):
        chi = numpy.sqrt(((feature - self.experimental_intensities)**2).sum()/
                    len(self.experimental_std))
        return chi

    def get_data_driven_cluster(self, starting_cell, unfolded = True):
        """

        For a given cell, find the cluster that minimize chi.

        • chi_profile: chi values along the flooding from the starting cell

        • threshold: flooding value that minimize chi

        • cluster: numpy array that gives the cell minimizing chi. This array
        is given in the unfolded space, by default. Set unfolded to False if
        you want the folded matrix.

        """
        dj = self.dijkstra(starting_cell=starting_cell)
        sorter = dj.flatten().argsort()
        feature_sorted = self.feature_map.reshape(numpy.prod(self.folded_shape),
                                                  self.feature_dim)[sorter]
        s = 0
        chi_profile = []
        for i,e in enumerate(feature_sorted):
            s += e
            m = s / (i+1)
            chi_profile.append(self.get_chi(m))
        # Get the cluster on the map
        threshold = numpy.argmin(chi_profile)
        cluster = dj.flatten()[sorter]
        selection = numpy.arange(len(cluster)) < threshold
        cluster[selection] = 1
        cluster[numpy.bool_(1-selection)] = 0
        cluster = cluster[numpy.argsort(sorter)].reshape(self.folded_shape)
        if unfolded:
            cluster = self.unfold_matrix(cluster)
        return chi_profile, threshold, cluster

    def get_minimal_chi_cluster(self):
        """

        Exhaustive search of the connected space that minimize chi

        """
        ni, nj = self.folded_shape
        progress = Progress.Progress(ni*nj, label = 'chi minimization')
        clusters = numpy.zeros((ni*nj,)+self.folded_shape)
        k = 0
        for i in range(ni):
            for j in range(nj):
                progress.count()
                chi_profile, threshold, cluster =\
                               self.get_data_driven_cluster(starting_cell=(i,j), unfolded=False)
                chi = chi_profile[threshold]
                cluster[cluster == 1] = chi
                clusters[k] = cluster
                k += 1
        return clusters

    def combine_clusters(self, clusters):
        """

        • clusters: output of get_minimal_chi_cluster. Array of shape (ni*nj,
        ni, nj), with (ni, nj) = self.folded_shape

        ‣ return: A cluster array:

            • If the clusters along axis=0 are not overlaid, the clusters are
            kept in the output.

            • If the clusters create a connected space, only the cluster with
            the minimal chi is kept.

        """
        clusters = numpy.ma.masked_array(clusters, clusters==0)
        sorter = clusters.min(axis=(1,2)).argsort()
        clusters = clusters[sorter] # sort cluster
        n = clusters.shape[0]
        progress = Progress.Progress(n, label = "Combining clusters", delta = 10)
        for k in range(n-1):
            progress.count()
            cluster = clusters[k].flatten()
            ind = numpy.where(cluster == cluster.min())[0]
            s = clusters[k+1:].shape
            overlaid = numpy.ma.count((clusters[k+1:].reshape((s[0], numpy.prod(s[1:])))[:,ind]),axis=1) > 0 # overlaid clusters
            overlaid_clusters = clusters[k+1:][overlaid]
            clusters[k+1:][overlaid] = numpy.zeros_like(clusters[k+1:][overlaid])
            clusters = numpy.ma.masked_array(clusters, clusters==0)
        n, ni, nj = clusters.shape
        clusters = clusters[numpy.ma.count(clusters.reshape(n, ni*nj), axis=1) > 0] # keep only slices with clusters
        return clusters
