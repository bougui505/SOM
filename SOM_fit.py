#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2015-10-26 10:33:38 (UTC+0100)

import SOM
import numpy
import itertools
from scipy.ndimage.morphology import distance_transform_edt
import sys

class fit(SOM.SOM):
    def __init__(self, inputvectors, experimental, X=50, Y=50, number_of_phase=2,
                metric='euclidean', radius_begin=None, radius_end=None,
                alpha_begin=[0.5, 0.25], alpha_end=[0.25, 0.], toricMap=True,
                optional_features=None):
        """
        experimental: standard saxs dat file or arbitrary numpy array.
        This class find out the minimum ensemble that fit the best the data.
        """
        self.inputvectors = inputvectors
        self.optional_features = optional_features
        self.is_complex = False
        self.X, self.Y = X, Y
        self.n_input, self.cardinal = self.inputvectors.shape
        self.number_of_phase = 2
        self.metric = metric
        self.iterations = [self.n_input, self.n_input * 2]
        if radius_begin == None:
            self.radius_begin = [numpy.sqrt(self.X * self.Y) / 8.,
                                numpy.sqrt(self.X * self.Y) / 16.]
        else:
            self.radius_begin = radius_begin
        if radius_end == None:
            self.radius_end = [numpy.sqrt(self.X * self.Y) / 16., 1]
        else:
            self.radius_end = radius_end
        self.alpha_begin = alpha_begin
        self.alpha_end = alpha_end
        self.toricMap = toricMap
        self.experimental = experimental
        if type(self.experimental) is str:
            self.read_experimental()
        elif type(self.experimental) is numpy.ndarray:
            self.experimental_intensities = self.experimental
            self.experimental_std = numpy.ones_like(self.experimental)
        self.smap = numpy.asarray([numpy.random.normal(scale=std,
                                    size=(self.X,self.Y)) for std\
                                    in self.experimental_std[::-1]]).T
        numpy.save('smap_init', self.smap) # save initial som map
        self.count = numpy.zeros((X,Y,1))
        self.chi_min = numpy.inf
        self.vector_ids = [] # ids of vector leading to a decrease in Chi
        self.weights = [] # weight maps for vector leading to a decrease in Chi

    def read_experimental(self):
        data = numpy.genfromtxt(self.experimental, skip_header=1)
        intensities = data[:,1]
        std = data[:,2]
        self.experimental_intensities = intensities
        self.experimental_std = std

    def get_chi(self, model):
        chi = numpy.sqrt(((self.experimental_intensities - model)**2 /
                            self.experimental_std**2).sum()/self.n_point)
        return chi

    def findBMU(self, k, smap, distKW=None, return_distance=False):
        """
            Find the Best Matching Unit for the input vector number k
        """
        if numpy.ma.isMaskedArray(smap):
            smap = smap.filled(numpy.inf)
        v = self.inputvectors[k]
        chis = (((self.experimental_intensities - (self.learning*v + self.smap)/
                (self.count + self.learning))**2
                / self.experimental_std**2).sum(axis=2) / self.cardinal)\
                .flatten()[None,:]
        index = chis.argmin()
        ij = numpy.unravel_index(index, (self.X, self.Y))
        chi_min = chis[0,index]
        if chi_min <= self.chi_min:
            self.chi_min = chi_min
        else:
            self.learning = 0.
        print "BMU= (%d,%d) Chi2= %.4g Chi2_min= %.4g"%(ij[0], ij[1], chi_min, self.chi_min)
#        numpy.savetxt(sys.stdout, self.smap[ij])
        sys.stdout.flush() # to write directly to file when stdout is redirected
        if not return_distance:
            return ij
        else:
            return ij, cdist[0, index]

    def apply_learning(self, smap, k, bmu, radius, rate):
        vector = self.inputvectors[k]
        shape = (self.X, self.Y)
        if rate > 0:
            if self.toricMap:
                bigshape = tuple(map(lambda x: 3 * x, shape))
                midselect = tuple([slice(s, 2 * s) for s in shape])
                features = numpy.ones(bigshape)
                copy_coord = lambda p, s: tuple([p + i * s for i in range(3)])
                all_coords = [copy_coord(coord, s) for coord, s in zip(bmu, shape)]
                for p in itertools.product(*all_coords):
                    features[p] = 0
                distance = distance_transform_edt(features)[midselect]
            else:
                features = numpy.ones(shape)
                features[bmu] = 0
                distance = distance_transform_edt(features)
            radmap = rate * numpy.exp(- distance ** 2 / (2. * radius) ** 2)
            self.count += radmap[..., None]
            smap += radmap[..., None] * vector
            self.vector_ids.append(k)
            self.weights.append(radmap)
