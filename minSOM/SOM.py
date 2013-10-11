#!/usr/bin/env python
# -*- coding: UTF8 -*-

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2013 10 11
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        

import os
import re
import sys

import numpy

from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import distance_transform_edt
import itertools
import bisect, copy
import scipy.spatial

class SOM(object):
    """A class to perform a variety of SOM-based analysis (any dimensions and shape)
    """
    def __init__(self, input_matrix=None, from_map=None):
        self.input_matrix = input_matrix
    
    def _generic_learning_rate(self, t, end_t, alpha_begin, alpha_end, shape='exp'):
        if shape == 'exp':
            timeCte = end_t / 10.
            rate = (alpha_begin - alpha_end) * numpy.exp( -t/timeCte ) + alpha_end
        return rate
    
    def _generic_learning_radius(self, t, end_t, radius_begin, radius_end, shape='exp'):
        if shape == 'exp':
            timeCte = end_t / 10.
            radius = (radius_begin - radius_end) * numpy.exp( -t/timeCte ) + radius_end
        return radius
    
    def _generic_learning_function(self, distance, rate, radius, shape='exp'):
        if shape == 'exp':
            return rate * numpy.exp( - distance**2 / (2.*radius)**2 )
        if shape == 'sinc':
            return rate * numpy.sinc(distance/radius)
        
    
    @staticmethod
    def _neighbor_dim1_toric(x, X):
        """Efficient toric neighborhood function for 1D SOM.
        """
        return [(x-1)%X, (x+1)%X]
    
    @staticmethod
    def _neighbor_dim2_toric(p, s):
        """Efficient toric neighborhood function for 2D SOM.
        """
        x, y = p
        X, Y = s
        xm = (x-1)%X
        ym = (y-1)%Y
        xp = (x+1)%X
        yp = (y+1)%Y
        return [(xm,ym), (xm,y), (xm,yp), (x,ym), (x,yp), (xp,ym), (xp,y), (xp,yp)]
    
    @staticmethod
    def _neighbor_dim3(p, s):
        """Efficient toric neighborhood function for 3D SOM.
        """
        x, y, z = p
        X, Y. Z = s
        xm = (x-1)%X
        ym = (y-1)%Y
        zm = (z-1)%Z
        xp = (x+1)%X
        yp = (y+1)%Y
        zp = (z+1)%Z
        return [(xm,ym,zm), (xm,y,zm), (xm,yp,zm), (x,ym,zm), (x,y,zm), (x,yp,zm), (xp,ym,zm), (xp,y,zm), (xp,yp,zm),
            (xm,ym,z), (xm,y,z), (xm,yp,z), (x,ym,z), (x,yp,z), (xp,ym,z), (xp,y,z), (xp,yp,z),
            (xm,ym,zp), (xm,y,zp), (xm,yp,zp), (x,ym,zp), (x,yp,zp), (xp,ym,zp), (xp,y,zp), (xp,yp,zp)]
    
    @staticmethod
    def _neighbor_general_toric(point, shape):
        """Toric neighborhood function for dimensions > 3.
        """
        plus_minus = lambda p, s: ((p-1)%s, p, (p+1)%s)
        all_coords = [ plus_minus(p, s) for p, s in zip(point, shape) ]
        neighbors = [ t for t in itertools.product(*all_coords) ]
        neighbors.remove(point)
        return neighbors
    
    def _generic_neighborhood_func(self, shape, toric):
        if toric:
            if len(shape) == 1:
                return self._neighbor_dim1_toric
            if len(shape) == 2:
                return self._neighbor_dim2_toric
            if len(shape) == 3:
                return self._neighbor_dim3_toric
            return self._neighbor_general_toric
        if len(shape) == 1:
            return self._neighbor_dim1
        if len(shape) == 2:
            return self._neighbor_dim2
        if len(shape) == 3:
            return self._neighbor_dim3
        return self._neighbor_general

    def rho(self, k,  BMUindices, Map):
        i,j = BMUindices
        rhoValue = self.rhoValue[k]
        rhoValue = max(scipy.spatial.distance.euclidean(self.input_matrix[k], Map[i,j]), rhoValue)
        self.rhoValue[k] = rhoValue
        return rhoValue

    def epsilon(self, k, BMUindices, Map):
        i,j = BMUindices
        return scipy.spatial.distance.euclidean(self.input_matrix[k], Map[i,j]) / self.rho(k, BMUindices, Map)
    
    def learn(self, **parameters):
        params = {
            'shape': (50, 50),
            'learning_subpart': None,
            'metric': 'euclidean',
            'phases': 2,
            'learning_rate': [
                lambda t, end_t, vector, bmu: self._generic_learning_rate(t, end_t, 0.5, 0.25, 'exp'),
                lambda t, end_t, vector, bmu: self._generic_learning_rate(t, end_t, 0.25, 0., 'exp')
                ],
            'learning_radius': [
                lambda t, end_t, vector, bmu: self._generic_learning_radius(t, end_t, 6.25, 3., 'exp'),
                lambda t, end_t, vector, bmu: self._generic_learning_radius(t, end_t, 4., 1., 'exp')
                ],
            'learning_function': [
                lambda dist, rate, radius: self._generic_learning_function(dist, rate, radius, shape='exp'),
                lambda dist, rate, radius: self._generic_learning_function(dist, rate, radius, shape='exp')
                ],
            'iterations': [
                self.input_matrix.shape[0],
                2*self.input_matrix.shape[0],
                ],
            'toric': True,
            'random_init': True,
            'autoparam' : False,
            'n_cpu': 1,
            'seed': None, # used to randomize the input vectors
            'verbose': False
            }
        params.update(parameters) # TODO: put default learning parameters after update if not present
        self.parameters = params
        
        show_umatrices = 'show_umatrices' in params # DEBUG show_umatrices=(imshow,draw)
        
        numpy.random.seed(params['seed'])
        # initialize SOM using given parameters (size, size of input vectors, choice of init mode)
        smap = self.map_init(self.input_matrix, params['shape'], params['random_init'])
        data_len = smap.shape[-1]
        nvec = self.input_matrix.shape[0]
        shape, subpart, verbose, n_cpu = params['shape'], params['learning_subpart'], params['verbose'], params['n_cpu']
        #neighborhood_func = self._generic_neighborhood_func(shape, params['toric'])
        # get the view of the SOM that will be used to find bmus
        if subpart is None:
            bmu_map = smap.view()
            inp_mat = self.input_matrix.view()
        else:
            mask = list(subpart.nonzero()[0])
            bmu_map = smap[..., mask]
            inp_mat = self.input_matrix[:, mask]
        if params['autoparam']:
            self.rhoValue = numpy.zeros(nvec)
        for phase, end_t in enumerate(params['iterations']): # loop on phases
            order = numpy.arange(nvec)
            numpy.random.shuffle(order)
            order = order[:min(end_t, nvec)]
            if params['autoparam'] and phase>0:
                order = numpy.argsort(self.rhoValue)
            func = params['learning_function'][phase]
            for t in range(end_t): # loop on iterations
                i = order[t % len(order)]
                vector = self.input_matrix[i] # get the vector
                bmu = self.findbmu(bmu_map, inp_mat[i], n_cpu=n_cpu) # find the bmu
                if not params['autoparam']:
                    radius = params['learning_radius'][phase](t, end_t, vector, smap[bmu]) # get the radius
                    rate = params['learning_rate'][phase](t, end_t, vector, smap[bmu]) # and rate
                else:
                    eps = self.epsilon(i, bmu, smap)
                    radius = eps*params['learning_radius'][phase](0, end_t, vector, smap[bmu])
                    rate = eps*params['learning_rate'][phase](0, end_t, vector, smap[bmu])
                self.apply_learning(smap, vector, bmu, radius, rate, func, params) # apply the gaussian to 
                if verbose and (t%100 == 0):
                    print phase, t, end_t, '%.2f%%'%((100.*t)/end_t), radius, rate, bmu
                    if show_umatrices:
                        imshow, draw = params['show_umatrices']
                        imshow(self.umatrix(smap, toric=True), interpolation='nearest')
                        draw()
        self.smap = smap
        return self.smap

    def batchlearn(self, **parameters):
        params = {
            'shape': (50, 50),
            'learning_subpart': None,
            'metric': 'euclidean',
            'phases': 2,
            'learning_radius': [
                lambda t, end_t, bmu: self._generic_learning_radius(t, end_t, 6.25, 3., 'exp'),
                lambda t, end_t, bmu: self._generic_learning_radius(t, end_t, 4., 1., 'exp')
                ],
            'learning_function': [
                lambda dist, rate, radius: self._generic_learning_function(dist, rate, radius, shape='exp'),
                lambda dist, rate, radius: self._generic_learning_function(dist, rate, radius, shape='exp')
                ],
            'iterations': [
                self.input_matrix.shape[0]/200,
                self.input_matrix.shape[0]/100,
                ],
            'toric': True,
            'random_init': True,
            'n_cpu': 1,
            'seed': None, # used to randomize the input vectors
            'verbose': False
            }
        params.update(parameters) # TODO: put default learning parameters after update if not present
        self.parameters = params
        numpy.random.seed(params['seed'])
        # initialize SOM using given parameters (size, size of input vectors, choice of init mode)
        smap = self.map_init(self.input_matrix, params['shape'], params['random_init'])
        data_len = smap.shape[-1]
        nvec = self.input_matrix.shape[0]
        shape, subpart, verbose, n_cpu = params['shape'], params['learning_subpart'], params['verbose'], params['n_cpu']
        if subpart is None:
            bmu_map = smap.view()
            inp_mat = self.input_matrix.view()
        else:
            mask = list(subpart.nonzero()[0])
            bmu_map = smap[..., mask]
            inp_mat = self.input_matrix[:, mask]
        for phase, end_t in enumerate(params['iterations']): # loop on phases
            order = numpy.arange(nvec)
            numpy.random.shuffle(order)
#            order = order[:min(end_t, nvec)]
            func = params['learning_function'][phase]
            batch = nvec/end_t
#            print batch, order.shape, order.min(), order.max()
            t_prime = 0
            end_t_prime = end_t * batch
            for t in range(end_t): # loop on iterations
                ind = order[t*batch:(t+1)*batch]
                vectors = self.input_matrix[ind] # get the vectors
                bmus = self.get_allbmus(smap=smap, vectors=vectors)
#                print bmus.shape, vectors.shape
                neighborhoods = numpy.zeros((shape[0],shape[1],1))
                prods = numpy.zeros((shape[0],shape[1],data_len))
                for i_bmu, bmu in enumerate(bmus):
                    radius = params['learning_radius'][phase](t_prime, end_t_prime, smap[bmu])
                    neighborhood = self.apply_learning(smap, vectors[i_bmu], bmu, radius, 1, func, params, batchlearn=True)
                    neighborhoods += neighborhood
                    prods += neighborhood*vectors[i_bmu]
                    t_prime += 1
                smap = prods / neighborhoods
                if verbose and (t%(end_t/100) == 0):
                    print phase, t, end_t, '%.2f%%'%((100.*t)/end_t), t_prime, radius
#                    if show_umatrices:
#                        imshow, draw = params['show_umatrices']
#                        imshow(self.umatrix(smap, toric=True), interpolation='nearest')
#                        draw()
        self.smap = smap
        return self.smap

    def findbmu(self, smap, vector, n_cpu=1):
        shape = list(smap.shape)
        neurons = reduce(lambda x,y: x*y, shape[:-1], 1)
        d = cdist(smap.reshape((neurons, shape[-1])), vector[None])[:,0]
        return numpy.unravel_index(numpy.argmin(d), tuple(shape[:-1]))
    
    def get_allbmus(self, smap=None, vectors=None, **parameters):
        if smap is None:
            smap = self.smap
        if vectors is None:
            vectors = self.input_matrix
        try:
            subpart = parameters['learning_subpart']
        except KeyError:
            subpart = None
        s = reduce(lambda x,y: x*y, list(smap.shape)[:-1], 1)
        if subpart is None:
            dist = cdist(smap.reshape((s, smap.shape[-1])), vectors)
        else:
            mask = list(subpart.nonzero()[0])
            dist = cdist(smap[..., mask].reshape((s, mask.sum())), vectors[:, mask])
        return numpy.asarray(numpy.unravel_index(dist.argmin(axis=0), smap.shape[:-1])).T

    def get_allbmus_kdtree(self, smap=None, **parameters): # Don't use this function for high dimension data: greater than 20 !!!
        if smap is None:
            smap = self.smap
        try:
            subpart = parameters['learning_subpart']
        except KeyError:
            subpart = None
        s = reduce(lambda x,y: x*y, list(smap.shape)[:-1], 1)
        tree = scipy.spatial.cKDTree(smap.reshape((s, smap.shape[-1])))
        return numpy.asarray(numpy.unravel_index(tree.query(self.input_matrix)[1], smap.shape[:-1])).T
    
    def apply_learning(self, smap, vector, bmu, radius, rate, func, params, batchlearn=False):
        toric, shape = params['toric'], params['shape']
        if toric:
            bigshape = tuple(map(lambda x: 3*x, shape))
            midselect = tuple([ slice(s, 2*s) for s in shape ])
            features = numpy.ones(bigshape)
            copy_coord = lambda p, s: tuple([p+i*s for i in range(3)])
            all_coords = [ copy_coord(coord, s) for coord, s in zip(bmu, shape) ]
            for p in itertools.product(*all_coords):
                features[p] = 0
            distance = distance_transform_edt(features)[midselect]
        else:
            features = numpy.ones(shape)
            features[bmu] = 0
            distance = distance_transform_edt(features)
        #radmap = numpy.exp( -sqdistance / (2.*radius)**2 )
        radmap = func(distance, rate, radius)
        if not batchlearn:
            adjmap = (smap - vector) * radmap[..., None]
            smap -= adjmap
        else:
            adjmap = radmap[..., None]
            return adjmap

    
    def map_init(self, input_matrix, map_shape, random_init):
        nvec = input_matrix.shape[1]
        shape = list(map_shape)+[nvec]
        smap = numpy.empty(tuple(shape))
        if random_init:
            mmin = numpy.min(input_matrix, axis=0)
            mmax = numpy.max(input_matrix, axis=0)
            #smallshape = tuple([1]*len(map_shape)+[nvec])
            for i in range(nvec):
                smap[..., i] = numpy.random.uniform(mmin[i], mmax[i], map_shape)
        else:
            raise NotImplementedError('non-random init is not implemented yet')
        return smap
    
    def umatrix(self, smap=None, **parameters):
        if smap is None:
            smap = self.smap
        params = { 'toric': True }
        params.update(parameters)
        shape = list(smap.shape)[:-1]
        toric = params['toric']
        umatrix = numpy.zeros(shape)
        neighborhood = self._generic_neighborhood_func(shape, toric)
        for point in itertools.product(*[ range(s) for s in shape ]):
            neuron = smap[point]
            neighbors = tuple(numpy.asarray(neighborhood(point, shape), dtype='int').T)
            #print neighbors
            umatrix[point] = cdist(smap[neighbors], neuron[None]).mean()
        return umatrix
    
