#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 01 30
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import SOM
import numpy
import random
import pickle
import scipy.spatial

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    import progressbar_notebook as progressbar
else:
    import progressbar

class GSOM:
    def __init__(self, inputvectors, growing_threshold, max_iterations=None, number_of_phases=2, alpha_begin = [.5,0.5], alpha_end = [.5,0.], radius_begin=[1.5,1.5], radius_end=[1.5,1], metric = 'euclidean', smap=None):
        self.growing_threshold = growing_threshold
        self.step = 0
        self.som = SOM.SOM(\
        inputvectors,\
        X = 3, Y = 3,\
        toricMap = False,\
        number_of_phases=number_of_phases,\
        radius_begin=radius_begin,\
        radius_end=radius_end,\
        alpha_begin = alpha_begin,\
        alpha_end = alpha_end,\
        metric = metric,\
        smap = smap
        )
        self.inputvectors = inputvectors
        self.number_of_phase = number_of_phases
        self.n_input, self.cardinal  = inputvectors.shape
        if max_iterations == None:
            self.iterations = [self.n_input, self.n_input]
        else:
            self.iterations = max_iterations
        if smap == None:
            self.smap = self.som.smap
            self.smap = numpy.ma.masked_array(self.smap, numpy.zeros_like(self.smap, dtype=bool))
        else:
            self.smap = self.som.smap
        self.X, self.Y, self.cardinal = self.smap.shape
        self.add_margins()

    def add_margins(self):
        """
        add a frame of masked elements around the self.smap
        """

        def add_left():
            smap_shape = numpy.asarray(self.smap.shape)
            new_shape = smap_shape + [0,1,0]
            mask = numpy.zeros(new_shape,dtype=bool)
            mask[:,0,...] = True
            new_smap = numpy.ma.masked_array(numpy.empty_like(mask, dtype=float), mask)
            new_smap[:,1:,...] = self.smap
            self.smap = new_smap
            self.X, self.Y, self.cardinal = self.smap.shape

        def add_right():
            smap_shape = numpy.asarray(self.smap.shape)
            new_shape = smap_shape + [0,1,0]
            mask = numpy.zeros(new_shape,dtype=bool)
            mask[:,-1,...] = True
            new_smap = numpy.ma.masked_array(numpy.empty_like(mask, dtype=float), mask)
            new_smap[:,:-1,...] = self.smap
            self.smap = new_smap
            self.X, self.Y, self.cardinal = self.smap.shape

        def add_top():
            smap_shape = numpy.asarray(self.smap.shape)
            new_shape = smap_shape + [1,0,0]
            mask = numpy.zeros(new_shape,dtype=bool)
            mask[0,:,...] = True
            new_smap = numpy.ma.masked_array(numpy.empty_like(mask, dtype=float), mask)
            new_smap[1:,:,...] = self.smap
            self.smap = new_smap
            self.X, self.Y, self.cardinal = self.smap.shape

        def add_bottom():
            smap_shape = numpy.asarray(self.smap.shape)
            new_shape = smap_shape + [1,0,0]
            mask = numpy.zeros(new_shape,dtype=bool)
            mask[-1,:,...] = True
            new_smap = numpy.ma.masked_array(numpy.empty_like(mask, dtype=float), mask)
            new_smap[:-1,:,...] = self.smap
            self.smap = new_smap
            self.X, self.Y, self.cardinal = self.smap.shape

        
        imin, jmin = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).min(axis=1)
        imax, jmax = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).max(axis=1)
        if imin == 0:
            add_top()
            imin, jmin = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).min(axis=1)
            imax, jmax = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).max(axis=1)
        if imax == self.X - 1:
            add_bottom()
            imin, jmin = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).min(axis=1)
            imax, jmax = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).max(axis=1)
        if jmin == 0:
            add_left()
            imin, jmin = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).min(axis=1)
            imax, jmax = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).max(axis=1)
        if jmax == self.Y-1:
            add_right()
            imin, jmin = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).min(axis=1)
            imax, jmax = numpy.asarray(numpy.where(~self.smap.mask.all(axis=2))).max(axis=1)
        self.som.X, self.som.Y, self.som.cardinal = self.smap.shape

    def grow(self, pos):
        """
        grow neighbors of cell pos=(i,j)
        """
        footprint = numpy.zeros((self.X,self.Y), dtype=bool)
        i,j = pos
        footprint[i-1:i+2,j-1:j+2] = True
        sub_smap = self.smap[footprint].reshape(3,3,self.cardinal)
        mask = numpy.ma.getmask(sub_smap)
        if mask.any():
            for u in range(3):
                for v in range(3):
                    if mask[u,v].all():
                        pos = numpy.asarray([u,v])
                        direction = pos - numpy.asarray([1,1])
                        neighbors = numpy.asarray([pos + direction, pos + 2 * direction, pos - direction, pos - 2 * direction])
                        neighbors = neighbors[numpy.logical_and((neighbors >= 0).all(axis=1), (neighbors < 3).all(axis=1))]
                        neighbors = sub_smap[neighbors[:,0], neighbors[:,1]]
                        if (1-neighbors.mask.all(axis=1)).sum() > 1:
                            sub_smap[u,v] = neighbors.sum(axis=0)
            self.smap[footprint] = sub_smap.reshape(9,self.cardinal)
            self.add_margins()
            return True
        else:
            return False

    def apoptosis(self, bmu, bmus):
        """
        remove neighbors of bmu if not present in bmus
        """
        connectivity = [(i,j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        neighbors = numpy.asarray(bmu) - connectivity
        neighbors = set(tuple(e) for e in neighbors)
        apoptotic = neighbors - set(bmus)
        for e in apoptotic:
            self.smap.mask[e] = True
>>>>>>> debug for apoptotic function

    def learn(self, verbose=False):
        self.smap_list = []
        self.n_neurons = []
        print 'Learning for %s vectors'%len(self.inputvectors)
        kdone=[]
        for trainingPhase in range(self.number_of_phase):
            kv=[]
            print '%s iterations'%self.iterations[trainingPhase]
            self.n_neurons.append([self.step, (1-self.smap.mask[:,:,0]).sum()])
            ## Progress bar
            tpn = trainingPhase + 1
            if verbose:
                widgets = ['Training phase %s; %d neurons: ' % (tpn, self.n_neurons[-1][1]), progressbar.Percentage(), progressbar.Bar(marker='=',left='[',right=']'), progressbar.ETA()]
                pbar = progressbar.ProgressBar(widgets=widgets, maxval=self.iterations[trainingPhase]-1)
                pbar.start()
            ###
            for t in range(self.iterations[trainingPhase]):
                self.step += 1
                if len(kv) > 0:
                    k = kv.pop()
                else:
                    kv = range(len(self.inputvectors))
                    random.shuffle(kv)
                    k = kv.pop()
                bmus, dists = self.som.findBMU(k, self.smap, return_distance = True, n_neighbors=9)
                bmu, dist = bmus[0], dists[0]
                is_growing = False
                if dist >= self.growing_threshold:
                    is_growing = self.grow(bmu)
                if not is_growing:
                    self.apoptosis(bmu, bmus)
                self.som.apply_learning(self.smap, k, bmu, self.som.radiusFunction(t, trainingPhase), self.som.learningRate(t, trainingPhase), geodesic=True, mask=self.smap.mask[:,:,0])
                self.n_neurons.append([self.step, (1-self.smap.mask[:,:,0]).sum()])
                self.add_margins()
                if verbose:
                    pbar.update(t)
            if verbose:
                pbar.finish()
            if trainingPhase < self.number_of_phase-1:
                self.smap_list.append(self.smap.copy()) # keep a copy of the smap after the first phase
                MapFile = open('map_phase_%s_%sx%s.dat' % (trainingPhase, self.X,self.Y), 'w')
                pickle.dump(self.smap, MapFile) # Write Map into npy file
                MapFile.close()

        MapFile = open('map_%sx%s.dat' % (self.X,self.Y), 'w')
        pickle.dump(self.smap, MapFile) # Write Map into npy file
        MapFile.close()
        self.n_neurons = numpy.asarray(self.n_neurons)
        return self.smap

    def umatrix(self, smap = None):
        if smap == None:
            smap = self.smap
        nx,ny,nz = smap.shape
        umat = numpy.empty((nx,ny))
        for i in range(nx)[1:-1]:
            for j in range(ny)[1:-1]:
                footprint = numpy.zeros((nx,ny), dtype=bool)
                footprint[i-1:i+2,j-1:j+2] = True
                sub_smap = smap[footprint].reshape(3,3,nz)
                mask = sub_smap.mask[:,:,0]
                mu = scipy.spatial.distance.pdist(sub_smap[~mask]).mean()
                umat[i,j] = mu
        umat = numpy.ma.masked_array(umat, smap.mask[:,:,0])
        self.umat = umat
        return umat

    def density(self, smap = None):
        if smap == None:
            smap = self.smap
        nx,ny,nz = smap.shape
        dmat = numpy.zeros((nx,ny), dtype=int)
        bmus = []
        for k in range(self.n_input):
            bmu = self.som.findBMU(k, self.smap)
            bmus.append(bmu)
            dmat[bmu] += 1
        dmat = numpy.ma.masked_array(dmat, numpy.logical_or(smap.mask[:,:,0],dmat==0))
        self.dmat = dmat
        self.bmus = numpy.asarray(bmus)
        return dmat
