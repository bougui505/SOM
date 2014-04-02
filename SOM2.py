#!/usr/bin/env python
# -*- coding: UTF8 -*-

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2014 04 02
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""


import os
import struct
import re
import sys

import numpy

from scipy.ndimage.morphology import distance_transform_edt
import itertools
import bisect, copy
import scipy.spatial

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

if run_from_ipython():
    try:
        from IPython.display import clear_output
    except ImportError:
        pass

class SOM(object):
    """A class to perform a variety of SOM-based analysis (any dimensions and shape)
    """
    def __init__(self, input_matrix=None, from_map=None):
        self.input_matrix = input_matrix
        mmin = numpy.min(input_matrix, axis=0)
        mmax = numpy.max(input_matrix, axis=0)
        self.k = numpy.sqrt((((mmax-mmin)[:3])**2).sum())/numpy.pi # coefficient to scale geodesic distance with Euclidean distance
        if input_matrix != None:
            self.ncom = self.input_matrix.shape[1] / 7 #number of center of mass
            print "%d rigid bodies"%self.ncom
        try:
            __IPYTHON__
            self.ipython = True
        except NameError:
            self.ipython = False

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
                if verbose:
                    if self.ipython:
                        try:
                            clear_output()
                        except NameError:
                            self.ipython = False
                        print phase, t, end_t, '%.2f%%'%((100.*t)/end_t), radius, rate, bmu
                    elif (t%100 == 0):
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
                    try:
                        clear_output()
                    except NameError:
                        self.ipython = False
#                    if show_umatrices:
#                        imshow, draw = params['show_umatrices']
#                        imshow(self.umatrix(smap, toric=True), interpolation='nearest')
#                        draw()
        self.smap = smap
        return self.smap

    def rigidbody_dist(self, smap, vector):
        k = self.k
        ncoords = self.ncom * 3
        shape = list(smap.shape)
        neurons = reduce(lambda x,y: x*y, shape[:-1], 1)
        sqeucl = scipy.spatial.distance.cdist(smap.reshape((neurons, shape[-1]))[:,:ncoords], vector[None,:ncoords], 'sqeuclidean')
        qdist = 0
        for i in range(self.ncom):
            a = ncoords + i*4
            b = ncoords + (i+1)*4
            qdist += (2*numpy.arccos(abs(1-scipy.spatial.distance.cdist(smap.reshape(neurons,shape[-1])[:,a:b], vector[None,a:b], 'cosine'))))**2
        return sqeucl[:,0] + k*qdist[:,0]

    def findbmu(self, smap, vector, n_cpu=1, returndist=False):
        if numpy.ma.isMaskedArray(vector):
            smap = smap[:,:,numpy.asarray(1-vector.mask, dtype=bool)]
            vector = numpy.asarray(vector[numpy.asarray(1-vector.mask, dtype=bool)])
        shape = list(smap.shape)
        neurons = reduce(lambda x,y: x*y, shape[:-1], 1)
        d = self.rigidbody_dist(smap, vector)
        if returndist:
            r = list(numpy.unravel_index(numpy.argmin(d), tuple(shape[:-1])))
            r.append(d.min())
            return tuple(r)
        else:
            return numpy.unravel_index(numpy.argmin(d), tuple(shape[:-1]))

    def q_mult(self, q1, q2):
        w1, x1, y1, z1 = q1.T
        w2, x2, y2, z2 = q2.T
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        prod = numpy.asarray([w, x, y, z]).T
        return prod

    def q_conjugate(self, q):
        w, x, y, z = q
        return numpy.asarray([w, -x, -y, -z])

    def qv_mult(self, q1, v1):
        n = v1.shape[0]
        q2 = numpy.hstack( ( numpy.zeros((n,1)),v1 ) )
        return numpy.asarray(self.q_mult(self.q_mult(q1, q2), self.q_conjugate(q1))[1:])

    def q_transform(self, coords, quat, vect):
        """
        Apply a transformation to the coordinates (coords) : a quaternion
        rotation from quat and a translation from vect
        Parameters:
            coords: initial coordinates; numpy.array of shape (n,3)
            quat: quaternion defining the rotation of shape (4,)
            vect: vector defining the translation of shape (3,)
        Returns:
            coords: the transformed coordinates; numpy.array of shape (n,3)
        """
        coords -= coords.mean(axis=0)
        coords = self.qv_mult(quat, coords)[:,1:] # rotation
        coords += vect # translation
        return coords

    def get_quats_vects_from_map(self, smap=None):
        """
        Get quaternions and centers of mass from self.smap
        Parameters:
            self
        Returns:
            quats: numpy.array of shape (self.ncom, X, Y, 4)
            coms: numpy.array of shape (self.ncom, X, Y, 3)
        """
        if smap == None:
            smap = self.smap
        quats = []
        coms = []
        com_ind = numpy.asarray([0,3])
        quat_ind = numpy.asarray([self.ncom * 3, self.ncom * 3 + 4])
        for i in range(self.ncom):
            coms.append(smap[...,com_ind[0]:com_ind[1]])
            quats.append(smap[..., quat_ind[0]:quat_ind[1]])
            com_ind += 3
            quat_ind += 4
        quats = numpy.asarray(quats)
        coms = numpy.asarray(coms)
        return quats, coms


    def slerp(self, t, q0, q1):
        """SLERP: Spherical Linear intERPolation between two quaternions.

        The return value is an interpolation between q0 and q1. For t=0.0 the
        return value equals q0, for t=1.0 it equals q1.  q0 and q1 must be unit
        quaternions.  Always picks the shortest path.

        Adapted from Python Computer Graphics Kit implementation, GPL v2.
        expanded to work on arrays of quaternions.

        Parameters:
            t  : displacement array, of shape (a, b)
            q0 : quaternion array, of shape (a, b, 4)
            q1 : quaternion, of shape (4)
        Returns:
            qfinal : the interpolated quaternion array of shape (a, b, 4)
        """
        ca = numpy.dot(q0,q1)
        neg_q1 = (ca < 0)
        ca = numpy.abs(ca)
        o = numpy.arccos(numpy.clip(ca,0,1))
        so = numpy.sin(o)
        a = numpy.sin(o*(1.-t)) / so
        b = numpy.sin(o*t) / so

        retmat = numpy.empty(q0.shape)
        #t close to 0 is q0
        nullloc = numpy.abs(so)<=1e-8

        retmat[nullloc] = q0[nullloc]
        #perform direct path
        poscase = numpy.logical_not(numpy.logical_or(neg_q1, nullloc))
        retmat[poscase] = q0[poscase,:]*a[poscase][...,None] \
                          + q1[None,:]*b[poscase][...,None]
        #direct path would be long, negate
        negcase = numpy.logical_and(neg_q1, numpy.logical_not(nullloc))
        retmat[negcase] = q0[negcase,:]*a[negcase][...,None] \
                            - q1[None,:]*b[negcase][...,None]
        return retmat

    def apply_learning(self, smap, vector, bmu, radius, rate, func, params, batchlearn=False):
        toric, shape = params['toric'], params['shape']
        if numpy.ma.isMaskedArray(vector):
            vector = vector.filled(0)
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
            #centers of mass, euclidian distance
            smap[:,:,:3*self.ncom] += radmap[..., None] \
                *(vector[:3*self.ncom] - smap[:,:,:3*self.ncom])
            #quaternions, use slerp
            for rb in xrange(self.ncom):
                qbegin = self.ncom*3+rb*4
                smap[...,qbegin:qbegin+4] = self.slerp(radmap,
                                               smap[...,qbegin:qbegin+4],
                                               vector[qbegin:qbegin+4])
        else:
            adjmap = radmap[..., None]
            return adjmap

    def normalize_quaternion(self, q):
        return q/numpy.sqrt((q**2).sum())

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
            #normalize quaternions, TODO use linalg.norm instead
            for i,j in numpy.ndindex(*map_shape):
                for rb in xrange(self.ncom):
                    qbegin = self.ncom*3 + rb*4
                    smap[i,j,qbegin:qbegin+4]\
                        = self.normalize_quaternion(smap[i,j,qbegin:qbegin+4])
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
            umatrix[point] = self.rigidbody_dist(smap[neighbors], neuron).mean()
        return umatrix

    def getmatindex(self):
        """
        return a masked array with the same shape than U-matrix. Each element
        gives the index of the input matrix of the best matching input vector.
        """
        print "computing distances and bmus"
        bmudists = numpy.asarray([self.findbmu(self.smap, e, returndist=True) for e in self.input_matrix])
        X,Y,Z = self.smap.shape
        indexmap = -numpy.ones((X,Y), dtype=int)
        for i in range(X):
            for j in range(Y):
                indices = numpy.nonzero((bmudists[:,:2] == (i,j)).all(axis=1))[0]
                if len(indices) > 0:
                    distances = bmudists[:,2][indices]
                    minindex = numpy.argmin(distances)
                    index = indices[minindex]
                    indexmap[i,j] = index
        mask = indexmap == -1
        self.indexmap = numpy.ma.masked_array(indexmap, mask)
        return self.indexmap

    def cluster_umatrix(self, umatrix, connectivity=2, gradient_connectivity=None, verbose=False):
        """Do a hierarchical clustering of the given umatrix.
        """

        if len(umatrix.shape) != 2:
            raise ValueError('cluster_umatrix only takes 2D U-matrices as argument.')

        # define the subprocesses that will be used later

        def getValues(neighbors, mat):
            return [mat[i,j] for i,j in neighbors]

        def down_gradient(umatrix, start, clusters, connectivity):
            """Start from a point and go down the gradient.
            Returns the arriving point (which should be a local minimum).
            """
            s = umatrix.shape
            point = start
            val = umatrix[point[0],point[1]]
            neighbors = [ n for n in self._neighbor_dim2_toric(point, s) if clusters[n] == 0 ]
            if len(neighbors) == 0:
                return -1
            nvals = getValues(neighbors, umatrix)
            path = {point}
            while val >= numpy.min(nvals):
                idx = numpy.argmin(nvals)
                point = neighbors[idx]
                val = nvals[idx]
                neighbors = [ n for n in self._neighbor_dim2_toric(point, s) if (n not in path) and (clusters[n] == 0) ]
                nvals = getValues(neighbors, umatrix)
                path.add(point)
                #print path
                #if len(path) > 30:
                #   break
            return point

        def fill_bassin(umatrix, start, clusters, connectivity):
            s = umatrix.shape
            bassin = start # so we can start from merged bassins
            bassin_vals = getValues(bassin, umatrix)
            bassin_ord = argsort(bassin_vals)
            bassin = [ bassin[i] for i in bassin_ord ]
            bassin_vals = [ bassin_vals[i] for i in bassin_ord ]
            neighbors = { item for nb in bassin for item in set(self._neighbor_dim2_toric(nb, s)) }
            neighbors.difference_update(set(bassin))
            neighbors = list(neighbors)
            neighbors_vals = getValues(neighbors, umatrix)
            neighbors_ord = argsort(neighbors_vals)
            neighbors = [ neighbors[i] for i in neighbors_ord ]
            neighbors_vals = [ neighbors_vals[i] for i in neighbors_ord ]
            #print " beginning 'fill_bassin' with %d points in bassin and %d in neighbors."%(len(bassin), len(neighbors))
            while True:
                c=0
                # if neighbor list is empty, we finished the flooding (at least for this part :))
                if len(neighbors) == 0:
                    return -1, bassin
                # next point is lowest neighbor
                point = neighbors.pop(0)
                val = neighbors_vals.pop(0)
                if clusters[point] < 0: # touching outside : ignore this point
                    c += 1
                    continue
                if clusters[point] != 0: # touching previous : stop
                    # but before, check
                    return clusters[point], point, val, bassin
                if val < bassin_vals[-1]: # spillage : stop
                    return clusters[point], point, val, bassin
                # if we're here, next point is neither from another cluster nor under the current waterlevel (nor outside)
                # add the point to the bassin and add its neighbors to the neighbor list
                idx = bisect.bisect(bassin_vals, val)
                bassin_vals.insert(idx, val)
                bassin.insert(idx, point)
                nb = self._neighbor_dim2_toric(point, s)
                nb_vals = getValues(nb, umatrix)
                for i, p in enumerate(nb):
                    if p in bassin or p in neighbors:
                        continue
                    idx = bisect.bisect(neighbors_vals, nb_vals[i])
                    neighbors_vals.insert(idx, nb_vals[i])
                    neighbors.insert(idx, p)
                #print "  finished one loop of 'fill_bassin'. Bassin has %d neurons and neighbors %d"%(len(bassin), len(neighbors))

        def fill_clusters(clusters, bassin, current, s):
            X, Y = s
            for a, b in bassin:
                for i in range(5):
                    for j in range(5):
                        clusters[(a+i*X)%(5*X), (b+j*Y)%(5*Y)] = -1 # fill with outside value in every copy
                clusters[a, b] = current # then, fill with correct cluster color just in the bassin

        # now we start with the main process
        if gradient_connectivity is None:
            gradient_connectivity = connectivity
        s = umatrix.shape
        X, Y = s
        current = 1
        umat_big = numpy.zeros((5*X, 5*Y))
        for i in range(5):
            for j in range(5):
                umat_big[i*X:(i+1)*X, j*Y:(j+1)*Y] = umatrix
        cl_list = []
        clusters = numpy.zeros(umat_big.shape, dtype='int')
        merges = []
        leaves = [1]
        i, j = numpy.unravel_index(numpy.argmin(umatrix), s)
        start = [ (i+2*X, j+2*Y) ]
        leaves_val = [ umat_big[start[0]] ]
        bassins = {}
        cluster_number_list = []
        while True:
            if verbose: print "Starting fill-up of bassin #%d with %d starting points."%(current, len(start))
            result = fill_bassin(umat_big, start, clusters, connectivity)
            if result[0] == -1: # completed clustering of this part of the map
                if verbose: print "New bassin completed (%d neurons). Filled whole space."%(len(result[1]))
                bassin = result[1]
                bassins[current] = bassin
                fill_clusters(clusters, result[1], current, s)
                mask = clusters == 0
                if mask.sum() == 0:
                    if verbose: print "Clustering completed (%d neurons)."%((clusters != -1).sum())
                    cl_list.append(clusters)
                    break
                else:
                    if verbose: print "Still some missing parts..."
                    start = [ numpy.unravel_index(numpy.argmin(umatrix[mask[2*X:3*X, 2*Y:3*Y]]), s) ] # TODO: get the real index 'through' mask
            elif result[0] == 0:
                a, point, val, bassin = result 
                if verbose: print "New bassin completed (%d neurons). Spilled in another, non-explored bassin."%len(bassin)
                bassins[current] = bassin
                leaves.append(current+1)
                fill_clusters(clusters, bassin, current, s)
                start = down_gradient(umat_big, point, clusters, gradient_connectivity)
                if start == -1:
                    #if verbose: print "down_gradient failed to attain anything."
                    #return cl_list
                    mask = clusters == 0
                    if mask.sum() == 0:
                        if verbose: print "The map is filled. Clustering completed."
                        cl_list.append(clusters)
                        del leaves[-1]
                        break
                    else:
                        if verbose: print "Filling from the spilling point..."
                        start = [ point ]
                        leaves_val.append(umat_big[point])
                else:
                    leaves_val.append(umat_big[start])
                    start = [ start ]
            elif result[0] > 0:
                cl, point, val, bassin = result
                if verbose: print "New bassin completed (%d neurons). Touched bassin #%d."%(len(bassin), cl)
                if len(bassin) < 3:
                    if verbose: print "New bassin is too small. It will be merged into cluster #%d."%(cl)
                    if current in leaves:
                        idx = leaves.index(current)
                        del leaves[idx]
                        del leaves_val[idx]
                    #print "DEBUG %d %d"%(current, cl)
                    #print current, cl
                    #current -= 1
                    current = cl -1
                    bassins[cl] += bassin
                    fill_clusters(clusters, bassin, cl, s)
                    start = bassins[cl]
                else:
                    #if True:
                    if verbose: print "Both bassins will be merged into cluster #%d."%(current+1)
                    fill_clusters(clusters, bassin, current, s)
                    bassins[current] = bassin
                    #if len(merges) > 0: print merges[-1]
                    merges.append((current, cl, current+1, point, val))
                    #print merges[-1]
                    cl_list.append(clusters)
                    mask = (clusters == cl) | (clusters == current)
                    clusters = numpy.zeros_like(clusters)
                    clusters[~mask] = cl_list[-1][~mask]
                    clusters[mask] = current + 1
                    start = [ (i, j) for i, j in numpy.asarray(mask.nonzero()).T ]
            if current in leaves and len(bassin) < 3:
                idx = leaves.index(current)
                del leaves[idx]
                del leaves_val[idx]
            current += 1
        # end
        stack = numpy.zeros_like(clusters)
        stack[clusters == -1] = -1
        for i, c in enumerate(reversed(cl_list)):
            m = c > 0
            stack[m] = c[m]
        old_leaves = leaves
        linkage, old_to_new, new_to_old, leaves, old_leaves, leaves_val = self.getLinkage(cl_list, umat_big, stack, verbose=verbose)
        leaves_cl = -1*numpy.ones_like(clusters)
        leaves_cl[clusters == -1] = -2
        #leaves_cl[clusters == 0] = -2 # shouldn't happen anyway
        for i, l in enumerate(leaves):
            for p in bassins[new_to_old[l]]:
                leaves_cl[p] = l
        x_offset = numpy.any(cl_list[-1] >= 0, axis=1)
        y_offset = numpy.any(cl_list[-1] >= 0, axis=0)
        view = lambda ar: ar[x_offset][:,y_offset]
        to_big = numpy.zeros((X, Y, 2), dtype='int')
        to_verybig = numpy.zeros((X, Y, 2), dtype='int')
        for x in range(X):
            for y in range(Y):
                for i in range(5):
                    for j in range(5):
                        p = x + i*X, y + j*Y
                        if stack[p] != -1:
                            to_verybig[x, y, :] = p
        xmin, ymin = to_verybig.min(axis=0).min(axis=0)
        to_big[:] = to_verybig
        to_big[:,:, 0] -= xmin
        to_big[:,:, 1] -= ymin
        to_leaves = { l: [l] for l in leaves }
        N = len(leaves)
        for i, t in enumerate(linkage):
            a, b, val, n = t
            c = N + i
            tl = [int(a), int(b)]
            allleaves = False
            while not allleaves:
                allleaves = True
                for j, l in enumerate(tl):
                    if l not in leaves:
                        allleaves = False
                        del tl[j]
                        tl += to_leaves[l]
                        break
            to_leaves[c] = tl
        return leaves, [ self.translate(c, old_to_new) for c in cl_list ], linkage, to_leaves, self.translate(stack, old_to_new), leaves_cl, leaves_val, (x_offset.sum(), y_offset.sum()), view, to_big, to_verybig


    @staticmethod
    def getLinkage(cl_list, umat_big, stack, verbose=False):
        # get real merging from cl_list

        prec_u = numpy.unique(cl_list[0])
        blarg = 0
        if 0 in prec_u:
            blarg += 1
        if -1 in prec_u:
            blarg += 1
        prec_u = prec_u[blarg:]
        prec_cl = cl_list[0]

        m = []
        l = []
        leaves = []
        for i, cl in enumerate(cl_list[1:]):
            mask = prec_cl > 0
            N0p = prec_cl[mask]
            U0 = numpy.unique(N0p)
            N1p = cl[mask]
            U1p = numpy.unique(N1p)
            c = 0
            for col in U1p:
                if col not in U0:
                    c += 1
                    mask2 = N1p == col
                    cols = numpy.unique(N0p[mask2])
                    if len(cols) != 2:
                        raise ValueError("cols != 2: %s"%(cols))
                    else:
                        m.append((cols[0], cols[1], col, umat_big[(prec_cl == cols[0]) | (prec_cl == cols[1])].max() ))
                        #print "enumerate step %d event %d between cl %d and %d to give %d."%(i, c, cols[0], cols[1], col)
                        if cols[0] not in l:
                            l.append(cols[0])
                            leaves.append(cols[0])
                        if cols[1] not in l:
                            l.append(cols[1])
                            leaves.append(cols[1])
                        l.append(col)
            #if c == 0:
            #    print "enumerate step %d no event:"%(i), numpy.unique(prec_cl), numpy.unique(cl)
            prec_cl = cl

        leaves_val = []
        for i, l in enumerate(leaves):
            #a, b, c, point, val = l
            leaves_val.append(umat_big[stack == l].min())
        leaves = numpy.asarray(leaves)
        leaves_val = numpy.asarray(leaves_val)
        # now reorder leaves and merging events so that it fits "linkage" description.
        order = numpy.argsort(leaves_val)
        old_to_new = { p: i for i, p in enumerate(leaves[order]) }
        new_to_old = { i: p for i, p in enumerate(leaves[order]) }
        N = len(leaves)
        if verbose: print "Number of leaves: %d"%N
        merge_ord = []
        count = { p:1 for p in range(N) }
        for i, l in enumerate(m):
            a, b, c, val = l
            try:
                na = old_to_new[a]
                nb = old_to_new[b]
            except:
                print "Cluster number not found", a, b, old_to_new
                raise
            nc = N+i
            old_to_new[c] = nc
            new_to_old[nc] = c
            count[nc] = count[na] + count[nb]
            merge_ord.append((na,nb,val,count[nc]))
        linkage = numpy.asarray(merge_ord, dtype='double')
        return linkage, old_to_new, new_to_old, order, leaves, leaves_val #[order], order

    @staticmethod
    def translate(arr, old_to_new, add_one=False):
        new = numpy.zeros_like(arr)
        u = numpy.unique(arr)
        new[arr == -1] = -1
        new[arr == -2] = -2
        for color in u:
            #print color, trans[color], (arr==color).sum(), trans[color] + (1 if add_one else 0)
            if color <= 0:
                continue
            new[arr == color] = old_to_new[color] + (1 if add_one else 0)
        return new

    @staticmethod
    def get_bigger(arr, view, cl):
        mask = cl == -1
        X, Y = arr.shape
        big_arr = numpy.zeros((5*X, 5*Y), arr.dtype)
        for i in range(5):
            for j in range(5):
                big_arr[i*X:(i+1)*X, j*Y:(j+1)*Y] = arr
        return view(numpy.ma.masked_array(big_arr, mask=mask))


#   def getNeighborsList(pos, shape, connectivity):
#       i, j = pos
#       n, m = shape
#       im = (i-1)%n
#       ip = (i+1)%n
#       jm = (j-1)%m
#       jp = (j+1)%m
#       if connectivity == 2: return [(im,jm),(im,j),(im,jp),(i,jm),(i,jp),(ip,jm),(ip,j),(ip,jp)]
#       return [(im,j),(i,jm),(i,jp),(ip,j)]
#
#   def getNeighbors(pos, shape, connectivity):
#       i, j = pos
#       n, m = shape
#       im = (i-1)%n
#       ip = (i+1)%n
#       jm = (j-1)%m
#       jp = (j+1)%m
#       if connectivity == 2: return {(im,jm),(im,j),(im,jp),(i,jm),(i,jp),(ip,jm),(ip,j),(ip,jp)}
#       return {(im,j),(i,jm),(i,jp),(ip,j)}


#   c = cluster_umatrix(uMatrix, connectivity=2, gradient_connectivity=2, verbose=True)
#   leaves, cl_list, linkage, to_leaves, stack, leaves_cl, leaves_val, newshape, view, to_big, to_verybig = c

