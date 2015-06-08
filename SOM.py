#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 08
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import numpy
import random
import pickle
import itertools
import scipy.spatial
from scipy.ndimage.morphology import distance_transform_edt
from multiprocessing import Pool
import Graph
import copy


def is_interactive():
    import __main__ as main

    return not hasattr(main, '__file__')


if is_interactive():
    import progressbar_notebook as progressbar
else:
    import progressbar


def get_bmus(v_smap_iscomplex):
    a, smap, is_complex = v_smap_iscomplex
    X, Y, cardinal = smap.shape
    if not is_complex:
        cdist = scipy.spatial.distance.cdist(a, smap.reshape(X * Y, cardinal))
        bmus = numpy.asarray(numpy.unravel_index(cdist.argmin(axis=1), (X, Y))).T  # new bmus
    else:
        bmus = []
        for vector in a:
            cdist = numpy.sqrt(( numpy.abs(smap.reshape((X * Y, cardinal)) - vector[None]) ** 2 ).sum(axis=1))
            b = numpy.asarray(numpy.unravel_index(cdist.argmin(axis=0), (X, Y))).T  # new bmus
            bmus.append(b)
        bmus = numpy.asarray(bmus)
    return bmus


class SOM:
    """
    Main class.
        Attributes:
            cardinal         : integer, length of input vectors
            inputvectors     : list of lists of objects, input vectors
            inputnames       : list of integers, vector names
            X                : integer, width of Kohonen map
            Y                : integer, height of Kohonen map
            number_of_phases : integer, number of training phases
    """

    def __init__(self, inputvectors, X=50, Y=50, number_of_phases=2, iterations=None, alpha_begin=[.50, .25],
                 alpha_end=[.25, 0.], radius_begin=None, radius_end=None, inputnames=None,
                 randomUnit=None, smap=None, metric='euclidean', toricMap=True,
                 randomInit=True, autoSizeMap=False, n_process=1):
        self.n_process = n_process
        self.pool = Pool(processes=self.n_process)
        if inputnames == None:
            inputnames = range(inputvectors.shape[0])
        self.n_input, self.cardinal = inputvectors.shape
        self.inputvectors = inputvectors
        self.inputnames = inputnames
        self.toricMap = toricMap
        self.randomInit = randomInit
        self.X = X
        self.Y = Y
        self.number_of_phase = number_of_phases
        self.alpha_begin = alpha_begin
        self.alpha_end = alpha_end
        self.bmus = None
        self.transition_matrix = None
        if radius_begin == None:
            self.radius_begin = [numpy.sqrt(self.X * self.Y) / 8., numpy.sqrt(self.X * self.Y) / 16.]
        else:
            self.radius_begin = radius_begin
        if radius_end == None:
            self.radius_end = [numpy.sqrt(self.X * self.Y) / 16., 1]
        else:
            self.radius_end = radius_end
        if iterations == None:
            self.iterations = [self.n_input, self.n_input * 2]
        else:
            self.iterations = iterations
        self.is_complex = False
        if self.inputvectors.dtype == numpy.asarray(numpy.complex(1, 1)).dtype:
            self.is_complex = True
            print "Complex numbers space"
        if not self.is_complex:
            self.metric = metric
        else:
            self.metric = lambda u, v: numpy.sqrt(( numpy.abs(u - v) ** 2 ).sum())  # metric for complex numbers
        if randomUnit is None:
            # Matrix initialization
            if smap is None:
                if randomInit:
                    self.smap = self.random_map()
                else:
                    inputarray = numpy.asarray(self.inputvectors)
                    inputmean = inputarray.mean(axis=0)
                    M = inputarray - inputmean
                    if numpy.argmin(M.shape) == 0:
                        # (min,max) * (max,min) -> (min,min)
                        # (0,1) * (1,0) -> (0,0)
                        mmt = True
                        covararray = numpy.dot(M, M.T)
                    else:
                        # (1,0) * (0,1) -> (1,1)
                        mmt = False
                        covararray = numpy.dot(M.T, M)
                    eival, eivec = numpy.linalg.eigh(covararray)
                    args = eival.argsort()[::-1]
                    eival = eival[args]
                    eivec = eivec[:, args]
                    sqev = numpy.sqrt(eival)[:2]
                    if autoSizeMap:
                        self.X, self.Y = map(lambda x: int(round(x)), sqev / (
                            (numpy.prod(sqev) / (self.X * self.Y)) ** (
                                1. / 2)))  # returns a size with axes size proportional to the eigenvalues and so that the total number of neurons is at least the number of neurons given in SOM.conf (X*Y)
                        print "Size of map will be %dx%d." % (self.X, self.Y)
                    # (1,0)*(0,0) if mmt else (0,1)*(1,1)
                    proj = numpy.dot(M.T, eivec) if mmt else numpy.dot(M, eivec)
                    Cmin = proj.min(axis=0)
                    Cmax = proj.max(axis=0)
                    Xmin, Ymin = Cmin[:2]
                    Xmax, Ymax = Cmax[:2]
                    origrid = numpy.mgrid[Xmin:Xmax:self.X * 1j, Ymin:Ymax:self.Y * 1j]
                    restn = inputarray.shape[1] - 2
                    if restn > 0:
                        # now do the rest
                        restptp = Cmax[2:] - Cmin[2:]
                        rest = numpy.random.random((restn, self.X, self.Y)) * restptp[:, numpy.newaxis,
                                                                              numpy.newaxis] + Cmin[2:, numpy.newaxis,
                                                                                               numpy.newaxis]
                        origrid = numpy.r_[origrid, rest]
                    self.smap = numpy.dot(origrid.transpose([1, 2, 0]), eivec.T) + inputmean
            else:
                self.loadMap(smap)
            print "Shape of the SOM:%s" % str(self.smap.shape)
        self.graph = Graph.Graph(smap=self.smap)

    def random_map(self):
        print "Map initialization..."
        if not self.is_complex:
            maxinpvalue = self.inputvectors.max(axis=0)
            mininpvalue = self.inputvectors.min(axis=0)
        else:
            maxinpvalue_real = numpy.real(self.inputvectors).max(axis=0)
            mininpvalue_real = numpy.real(self.inputvectors).min(axis=0)
            maxinpvalue_imag = numpy.imag(self.inputvectors).max(axis=0)
            mininpvalue_imag = numpy.imag(self.inputvectors).min(axis=0)
        somShape = [self.X, self.Y]
        vShape = numpy.array(self.inputvectors[0]).shape
        for e in vShape:
            somShape.append(e)
        if not self.is_complex:
            smap = numpy.random.uniform(0, 1, (self.X, self.Y, self.cardinal)) * (
                maxinpvalue - mininpvalue) + mininpvalue
        else:
            smap_real = numpy.random.uniform(0, 1, (self.X, self.Y, self.cardinal)) * (
                maxinpvalue_real - mininpvalue_real) + mininpvalue_real
            smap_imag = numpy.random.uniform(0, 1, (self.X, self.Y, self.cardinal)) * (
                maxinpvalue_imag - mininpvalue_imag) + mininpvalue_imag
            smap = smap_real + 1j * smap_imag
        return smap

    def save_data(self, outfile='som.dat', **kwargs):
        print 'saving data in %s'%outfile
        data = copy.copy(self.__dict__)
        del data['graph']
        del data['pool'] # pool objects cannot be passed between processes or pickled
        for key, value in kwargs.iteritems():
            data[key] = value
        f = open(outfile,'wb')
        pickle.dump(data, f, 2)
        f.close()
        print 'done'

    def load_data(self, infile='som.dat'):
        print 'loading data from %s'%infile
        f = open(infile,'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        print 'done'

    def loadMap(self, smap):
        self.smap = smap
        shape = numpy.shape(self.smap)
        self.X = shape[0]
        self.Y = shape[1]

    def findBMU(self, k, smap, distKW=None, return_distance=False):
        """
            Find the Best Matching Unit for the input vector number k
        """
        if numpy.ma.isMaskedArray(smap):
            smap = smap.filled(numpy.inf)
        if not self.is_complex:
            cdist = scipy.spatial.distance.cdist(numpy.reshape(self.inputvectors[k], (1, self.cardinal)),
                                                 numpy.reshape(smap, (self.X * self.Y, self.cardinal)), self.metric)
        else:
            vector = self.inputvectors[k]
            shape = self.smap.shape
            neurons = reduce(lambda x, y: x * y, shape[:-1], 1)
            cdist = numpy.sqrt(( numpy.abs(smap.reshape((neurons, shape[-1])) - vector[None]) ** 2 ).sum(axis=1))
        index = cdist.argmin()
        if not return_distance:
            return numpy.unravel_index(index, (self.X, self.Y))
        else:
            return numpy.unravel_index(index, (self.X, self.Y)), cdist[0, index]

    def radiusFunction(self, t, trainingPhase=0):
        timeCte = float(self.iterations[trainingPhase]) / 10
        self.radius = ( self.radius_begin[trainingPhase] - self.radius_end[trainingPhase] ) * numpy.exp(-t / timeCte) + \
                      self.radius_end[trainingPhase]
        return self.radius

    def learningRate(self, t, trainingPhase=0):
        timeCte = float(self.iterations[trainingPhase]) / 10
        self.learning = ( self.alpha_begin[trainingPhase] - self.alpha_end[trainingPhase] ) * numpy.exp(-t / timeCte) + \
                        self.alpha_end[trainingPhase]
        return self.learning

    def apply_learning(self, smap, k, bmu, radius, rate):
        vector = self.inputvectors[k]
        shape = (self.X, self.Y)
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
        # radmap = numpy.exp( -sqdistance / (2.*radius)**2 )
        radmap = rate * numpy.exp(- distance ** 2 / (2. * radius) ** 2)
        adjmap = (smap - vector) * radmap[..., None]
        smap -= adjmap

    def learn(self, verbose=False):
        Map = self.smap
        print 'Learning for %s vectors' % len(self.inputvectors)
        for trainingPhase in range(self.number_of_phase):
            kv = []
            print '%s iterations' % self.iterations[trainingPhase]
            ## Progress bar
            tpn = trainingPhase + 1
            if verbose:
                widgets = ['Training phase %s : ' % tpn, progressbar.Percentage(),
                           progressbar.Bar(marker='=', left='[', right=']'), progressbar.ETA()]
                pbar = progressbar.ProgressBar(widgets=widgets, maxval=self.iterations[trainingPhase] - 1)
                pbar.start()
            ###
            for t in range(self.iterations[trainingPhase]):
                if len(kv) > 0:
                    k = kv.pop()
                else:
                    kv = range(len(self.inputvectors))
                    random.shuffle(kv)
                    k = kv.pop()
                self.apply_learning(Map, k, self.findBMU(k, Map), self.radiusFunction(t, trainingPhase),
                                    self.learningRate(t, trainingPhase))
                if verbose:
                    pbar.update(t)
            if verbose:
                pbar.finish()
        self.smap = Map
        MapFile = open('map_%sx%s.dat' % (self.X, self.Y), 'w')
        pickle.dump(Map, MapFile)  # Write Map into file map.dat
        MapFile.close()
        self.find_bmus()
        self.save_data()
        return self.smap

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

    @property
    def umatrix(self):
        shape = list(self.smap.shape)[:-1]
        umatrix = numpy.zeros(shape)
        for point in itertools.product(*[range(s) for s in shape]):
            neuron = self.smap[point]
            neighbors = tuple(numpy.asarray(self.neighbor_dim2_toric(point, shape), dtype='int').T)
            if not self.is_complex:
                cdist = scipy.spatial.distance.cdist(self.smap[neighbors], neuron[None])
            else:
                cdist = numpy.sqrt(( numpy.abs(self.smap[neighbors] - neuron[None]) ** 2 ).sum(axis=1))
            umatrix[point] = cdist.mean()
        return umatrix

    def find_bmus(self):
        n_split = self.cardinal / 100
        if n_split < self.n_process:
            n_split = self.n_process
        sub_arrays = numpy.array_split(self.inputvectors, n_split)
        sub_arrays = [a for a in sub_arrays if a.size > 0]
        pools = self.pool.map(get_bmus, [(a, self.smap, self.is_complex) for a in sub_arrays])
        bmus = []
        for a in pools:
            bmus.extend(list(a))
        self.bmus = numpy.asarray(bmus)
        return self.bmus

    def get_representatives(self):
        """
        Find representative for each neuron. The representative is defined as the input element with the smallest
        distance to the neuron.
        :return:
        """
        if self.bmus is None:
            self.find_bmus()
        error_map = numpy.ones(self.smap.shape[:-1]) * numpy.inf
        representatives = numpy.ones_like(error_map) * numpy.nan
        for i, bmu in enumerate(self.bmus):
            bmu = tuple(bmu)
            d = scipy.spatial.distance.euclidean(self.inputvectors[i], self.smap[bmu])
            if d < error_map[bmu]:
                error_map[bmu] = d
                representatives[bmu] = i
        return representatives

    def get_transition_matrix(self, lag=1, dwell_time=None):
        """
        Compute the transition matrix. Relevant only for time series input data.
        lag is the lag time.
        dwell_time is the period of time that a system remains in a given state
        :type dwell_time: int
        :type lag: int
        :return:
        """
        if self.bmus is None:
            self.find_bmus()
        shape = self.smap.shape[:-1]
        n = numpy.product(shape)
        transition_matrix = numpy.zeros((n, n))
        density = numpy.zeros(shape, dtype=int)
        modulo = 1
        for k1, k2 in zip(range(self.n_input), range(lag, self.n_input)):
            bmu1, bmu2 = self.bmus[k1], self.bmus[k2]
            density[tuple(bmu1)] += 1
            if dwell_time is not None:
                modulo = k2 % dwell_time
            if modulo != 0:
                transition_matrix[numpy.ravel_multi_index(bmu1, shape), numpy.ravel_multi_index(bmu2, shape)] += 1
        transition_matrix = transition_matrix / density.flatten()
        transition_matrix[:, density.flatten() == 0] = 0
        self.transition_matrix = transition_matrix

    def get_kinetic_communities(self, lag=1, dwell_time=None):
        """
        Compute the kinetic communities. Relevant only for time series input
        :return:
        """
        if self.transition_matrix is None:
            self.get_transition_matrix(lag=lag, dwell_time=dwell_time)
        self.kinetic_graph = Graph.Graph(adjacency_matrix=self.transition_matrix)
        self.kinetic_graph.best_partition()
        self.kinetic_graph.community_map = self.kinetic_graph.community_map.reshape(self.smap.shape[:-1])
