#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-02-04 11:11:17 (UTC+0100)

import sys
sys.path.append("/home/bougui/lib/SOM")
import SOM
import MD
from MDAnalysis import Universe, Timeseries, collection
import numpy

class SOS:
    """
    SOM based implementation of the String Of Swarms method.
    """
    def __init__(self, pdb_1=None, pdb_2= None, smap=None, inputmat=None):
        """
        args:
        • pdb_1: filename of the pdb for the starting structure of the path
        • pdb_2: filename of the pdb for the ending structure of the path
        • smap: SOM map
        • inputmat: input matrix that has been used for SOM training
          (array of phi-psi dihedrals in complex numbers)
        attributes:
        • desc1: descriptor for pdb_1
        • desc2: descriptor for pdb_2
        • bmu_1: Best Matching Unit for pdb_1
        • bmu_2: Best Matching Unit for pdb_2
        """
        self.pdb_1 = pdb_1
        self.pdb_2 = pdb_2
        self.smap = smap
        if self.pdb_1 is not None and self.smap is not None:
            self.bmu_1 = self.find_bmu(self.get_dihedrals_from_pdb(self.pdb_1))
        else:
            self.bmu_1 = None
        if self.pdb_2 is not None and self.smap is not None:
            self.bmu_2 = self.find_bmu(self.get_dihedrals_from_pdb(self.pdb_2))
        else:
            self.bmu_2 = None
        self.inputmat = inputmat
        if self.smap is not None and self.inputmat is not None:
            self.som = SOM.SOM(inputmat, smap=self.smap)
            self.som.graph.unfold_smap()
            self.som.get_kinetic_communities()
        else:
            self.som = None

        self.metric = 'euclidean'

    def get_dihedrals(self, pdb, dcd):
        print "Computing dihedrals"
        universe = Universe(pdb, dcd)
        protein = universe.select_atoms('protein')
        n_residue = protein.n_residues
        collection.clear()
        for i in range(1, n_residue - 1):
            phi_sel = universe.residues[i].phi_selection()
            psi_sel = universe.residues[i].psi_selection()
            collection.addTimeseries(Timeseries.Dihedral(phi_sel))
            collection.addTimeseries(Timeseries.Dihedral(psi_sel))
        collection.compute(universe.trajectory)
        phi_psi_array = collection.data.T
        descriptors = numpy.asarray(
                zip(numpy.cos(phi_psi_array[:,0]) + 1j * numpy.sin(phi_psi_array[:,0]),
                    numpy.cos(phi_psi_array[:,1]) + 1j * numpy.sin(phi_psi_array[:,1]))
                )

        if self.inputmat is None:
            self.inputmat = descriptors
        else:
            self.inputmat = r_[self.inputmat, descriptors]
        print "Shape of the dihedral array: %dx%d"%inputmat
        print "done"

    def get_dihedrals_from_pdb(self, pdb):
        universe = Universe(pdb)
        protein = universe.select_atoms('protein')
        n_residue = protein.n_residues
        phi_psi_array = []
        for i in range(1, n_residue -1):
            phi_sel = universe.residues[i].phi_selection()
            psi_sel = universe.residues[i].psi_selection()
            phi_psi_array.append(numpy.deg2rad(phi_sel.dihedral.value())) # weird behaviour: dihedral.value return angle in degree whereas Timeseries.Dihedral returns angle in radians...
            phi_psi_array.append(numpy.deg2rad(psi_sel.dihedral.value()))
        phi_psi_array = numpy.asarray(phi_psi_array)
        descriptors = numpy.asarray(
                [numpy.cos(phi_psi_array[0]) + 1j * numpy.sin(phi_psi_array[0]),
                    numpy.cos(phi_psi_array[1]) + 1j * numpy.sin(phi_psi_array[1])]
                )
        return descriptors

    def find_bmu(self, v, return_distance = False):
        """
            Find the Best Matching Unit for the input vector v
        """
        self.is_complex = False
        if v.dtype == numpy.asarray(numpy.complex(1, 1)).dtype:
            self.is_complex = True
        if numpy.ma.isMaskedArray(self.smap):
            self.smap = self.smap.filled(numpy.inf)
        X, Y, cardinal = self.smap.shape
        if not self.is_complex:
            cdist = scipy.spatial.distance.cdist(numpy.reshape(v, (1, len(v))),
                                                 numpy.reshape(self.smap, (X * Y, cardinal)), self.metric)
        else:
            shape = self.smap.shape
            neurons = reduce(lambda x, y: x * y, shape[:-1], 1)
            cdist = numpy.sqrt(( numpy.abs(self.smap.reshape((neurons, shape[-1])) - v[None]) ** 2 ).sum(axis=1))
        index = cdist.argmin()
        if not return_distance:
            return numpy.unravel_index(index, (X, Y))
        else:
            return numpy.unravel_index(index, (X, Y)), cdist[0, index]

    def get_transition_path(self):
        """
        It returns the frame ids for the transition path between self.bmu_1 and
        self.bmu_2.
        """
        if self.som.kinetic_graph is None:
            self.som.get_kinetic_communities()
        path_som = self.som.kinetic_graph.shortestPath(self.bmu_1, self.bmu_2)
        if self.som.representatives is None:
            self.som.get_representatives()
        transition_path = numpy.int_([self.som.representatives.flatten()[e]\
                                         for e in path_som])
        return transition_path
