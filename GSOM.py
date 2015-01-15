#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 01 15
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import SOM
import numpy

class GSOM:
    def __init__(self, inputvectors, number_of_phases=2, alpha_begin = [.5,0.5], alpha_end = [.5,0.], radius_begin=[1.5,1.5], radius_end=[1.5,1]):
        self.som = SOM.SOM(\
        inputvectors,\
        X = 3, Y = 3,\
        toricMap = False,\
        number_of_phases=number_of_phases,\
        radius_begin=radius_begin,\
        radius_end=radius_end\
        )
        self.smap = self.som.random_map()
        self.smap = numpy.ma.masked_array(self.smap, numpy.zeros_like(self.smap, dtype=bool))
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
        self.smap[footprint] = sub_smap.reshape(9,2)
        self.add_margins()
