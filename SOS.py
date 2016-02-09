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
import MDAnalysis
import numpy
import os
import glob
import progress_reporting as Progress
from multiprocessing import Pool

def run_simulation(pdb):

    def exists_and_not_empty(filename):
        if os.path.exists(filename):
            if  os.stat(filename).st_size > 0:
                return True
            else:
                return False
        else:
            return False

    print "Equilibrating %s"%pdb
    pdb_eq = os.path.splitext(pdb)[0][:-3]+'_eq.pdb'
    if  exists_and_not_empty(pdb_eq):
        print "%s file exists and is not empty"%pdb_eq
    else:
        # Equilibration
        md_eq = MD.equilibration(pdb)
        md_eq.add_solvent()
        md_eq.create_system(platform_name='CPU')
        md_eq.minimize()
        log_eq = 'log/'+os.path.splitext(os.path.basename(pdb))[0]+'_eq.log'
        md_eq.equilibrate(filename_output_pdb=pdb_eq,
                          filename_output_log= log_eq)
        print "done"
    # Production
    for i in range(10):
        dcd_prod = 'dcd/'+os.path.splitext(os.path.basename(pdb))[0][:-3]+'_%d.dcd'%i
        if exists_and_not_empty(dcd_prod):
            print "%s file exists and is not empty"%dcd_prod
        else:
            print "MD %d for %s"%(i, pdb)
            md_prod = MD.production(pdb_eq)
            md_prod.create_system(platform_name='CPU')
            log_prod = 'log/'+os.path.splitext(os.path.basename(pdb))[0]+'_prod_%d.log'%i
            md_prod.run(filename_output_dcd=dcd_prod,
                        filename_output_log=log_prod)
            print "done"


class SOS:
    """
    SOM based implementation of the String Of Swarms method.
    """
    def __init__(self, pdb_1=None, pdb_2= None, dcd=None, smap=None, inputmat=None, n_process=1):
        """
        args:
        • pdb_1: filename of the pdb for the starting structure of the path
        • pdb_2: filename of the pdb for the ending structure of the path
        • dcd: Optional dcd trajectory used to generate the smap given as argument
        • smap: SOM map
        • inputmat: input matrix that has been used for SOM training
          (array of phi-psi dihedrals in complex numbers)
        • n_process: number of parallel process to run
        attributes:
        • desc1: descriptor for pdb_1
        • desc2: descriptor for pdb_2
        • bmu_1: Best Matching Unit for pdb_1
        • bmu_2: Best Matching Unit for pdb_2
        """
        self.pdb_1 = pdb_1
        self.pdb_2 = pdb_2
        self.dcd = dcd
        self.smap = smap
        self.pool = Pool(processes=n_process)
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
            self.som = SOM.SOM(inputmat, smap=self.smap, n_process = n_process)
            self.som.graph.unfold_smap()
            self.som.get_kinetic_communities()
        else:
            self.som = None

        self.metric = 'euclidean'

    def get_dihedrals(self):
        pdb = self.pdb_1
        dcd = self.dcd
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
        self.transition_path = transition_path
        return transition_path

    def get_pdbs(self):
        """
        Return 1 pdb file per frame id
        """

        dcd = self.dcd
        frame_ids = self.transition_path

        if not os.path.isdir('pdb'):
            os.mkdir('pdb')

        aa = numpy.argsort(frame_ids)

        sorted_frame_list = list(frame_ids[aa][::-1])
        u = MDAnalysis.Universe(self.pdb_1, dcd)
        i = 0
        for ts in u.trajectory:
            try:
                if ts.frame == sorted_frame_list[-1]:
                    OUTPDB = "pdb/%d_in.pdb"%aa[i]
                    pdb = MDAnalysis.Writer(OUTPDB, multiframe=False)
                    pdb.write(u.select_atoms('segid A')) # Only protein without waters and ions
                    print "got frame %d"%sorted_frame_list.pop()
                    i += 1
            except IndexError:
                break

    def run_MD(self, platform_name='OpenCL'):
        if not os.path.isdir('log'):
            os.mkdir('log')
        if not os.path.isdir('dcd'):
            os.mkdir('dcd')
        pdb_list = glob.glob('pdb/*_in.pdb')
        #progress = Progress.Progress(len(pdb_list), delta=1)
        #for pdb in pdb_list:
        #    run_simulation(pdb)
        #    progress.count()
        pools = self.pool.map(run_simulation, pdb_list)
