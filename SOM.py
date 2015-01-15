#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 01 15
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import numpy
import random
import progressbar
import pickle
import itertools
import scipy.spatial

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
    def __init__(self, inputvectors, X=50, Y=50, number_of_phases=2, iterations=None, alpha_begin = [.50,.25], alpha_end = [.25,0.], radius_begin = None, radius_end = None, inputnames=None, distFunc=None, randomUnit=None, mapFileName=None, metric = 'euclidean', autoParam = False, sort2ndPhase=False, toricMap=True, randomInit=True, autoSizeMap=False):
        if inputnames == None:
            inputnames = range(inputvectors.shape[0])
        self.metric = metric
        self.n_input, self.cardinal  = inputvectors.shape
        self.inputvectors = inputvectors
        self.inputnames = inputnames
        self.autoParam = autoParam
        self.sort2ndPhase = sort2ndPhase
        self.toricMap = toricMap
        self.randomInit = randomInit
        self.X = X
        self.Y = Y
        self.number_of_phase = number_of_phases
        i = 1
        self.alpha_begin = alpha_begin
        self.alpha_end = alpha_end
        if radius_begin == None:
            self.radius_begin = [self.X/8.,self.X/16.]
        else:
            self.radius_begin = radius_begin
        if radius_end == None:
            self.radius_end = [self.X/16.,1]
        else:
            self.radius_end = radius_end
        if iterations == None:
            self.iterations = [self.n_input, self.n_input*2]
        else:
            self.iterations = iterations
        if randomUnit is None:
            # Matrix initialization
            if mapFileName == None:
                if randomInit:
                    self.smap = self.random_map()
                else:
                    inputarray=numpy.asarray(self.inputvectors)
                    inputmean=inputarray.mean(axis=0)
                    M=inputarray-inputmean
                    if numpy.argmin(M.shape) == 0:
                        # (min,max) * (max,min) -> (min,min)
                        # (0,1) * (1,0) -> (0,0)
                        mmt=True
                        covararray=numpy.dot(M,M.T)
                    else:
                        # (1,0) * (0,1) -> (1,1)
                        mmt=False
                        covararray=numpy.dot(M.T,M)
                    eival,eivec=numpy.linalg.eigh(covararray)
                    args=eival.argsort()[::-1]
                    eival=eival[args]
                    eivec=eivec[:,args]
                    sqev=numpy.sqrt(eival)[:2]
                    if autoSizeMap:
                        self.X,self.Y=map(lambda x: int(round(x)),sqev/((numpy.prod(sqev)/(self.X*self.Y))**(1./2))) # returns a size with axes size proportional to the eigenvalues and so that the total number of neurons is at least the number of neurons given in SOM.conf (X*Y)
                        print "Size of map will be %dx%d."%(self.X,self.Y)
                    # (1,0)*(0,0) if mmt else (0,1)*(1,1)
                    proj=numpy.dot(M.T,eivec) if mmt else numpy.dot(M,eivec)
                    Cmin=proj.min(axis=0)
                    Cmax=proj.max(axis=0)
                    Xmin,Ymin=Cmin[:2]
                    Xmax,Ymax=Cmax[:2]
                    origrid=numpy.mgrid[Xmin:Xmax:self.X*1j,Ymin:Ymax:self.Y*1j]
                    restn=inputarray.shape[1]-2
                    if restn > 0:
                        # now do the rest
                        restptp=Cmax[2:]-Cmin[2:]
                        rest=numpy.random.random((restn,self.X,self.Y))*restptp[:,numpy.newaxis,numpy.newaxis]+Cmin[2:,numpy.newaxis,numpy.newaxis]
                        origrid=numpy.r_[origrid,rest]
                    self.smap=numpy.dot(origrid.transpose([1,2,0]),eivec.T)+inputmean
            else:
                self.smap = self.loadMap(mapFileName)
            print "Shape of the SOM:%s"%str(self.smap.shape)
        self.distFunc = distFunc

    def random_map(self):
        print "Map initialization..."
        maxinpvalue = self.inputvectors.max(axis=0)
        mininpvalue = self.inputvectors.min(axis=0)
        somShape = [self.X, self.Y]
        vShape = numpy.array(self.inputvectors[0]).shape
        for e in vShape:
            somShape.append(e)
        smap = numpy.random.uniform(mininpvalue[0], maxinpvalue[0], (self.X,self.Y,1))
        for e in zip(mininpvalue[1:],maxinpvalue[1:]):
            smap = numpy.concatenate( (smap,numpy.random.uniform(e[0],e[1],(self.X,self.Y,1))), axis=2 )
        return smap

    def loadMap(self, MapFile):
        MapFileFile = open(MapFile, 'r')
        self.smap = pickle.load(MapFileFile)
        MapFileFile.close()
        shape = numpy.shape(self.smap)
        self.X = shape[0]
        self.Y = shape[1]
        return self.smap
        
    def findBMU(self, k, Map, distKW=None, return_distance=False):
        """
            Find the Best Matching Unit for the input vector number k
        """
        if numpy.ma.isMaskedArray(Map):
            Map = Map.filled(numpy.inf)
        cdist = scipy.spatial.distance.cdist(numpy.reshape(self.inputvectors[k], (1,self.cardinal)), numpy.reshape(Map, (self.X*self.Y,self.cardinal)), self.metric)
        index = cdist.argmin()
        if not return_distance:
            return numpy.unravel_index(index, (self.X,self.Y))
        else:
            return numpy.unravel_index(index, (self.X,self.Y)), cdist[0,index]
        
    def radiusFunction(self, t, trainingPhase=0):
        timeCte = float(self.iterations[trainingPhase])/10
        self.radius = ( self.radius_begin[trainingPhase] - self.radius_end[trainingPhase] ) * numpy.exp( -t/timeCte ) + self.radius_end[trainingPhase]
        return self.radius
        
    def learningRate(self, t, trainingPhase=0):
        timeCte = float(self.iterations[trainingPhase])/10
        self.learning = ( self.alpha_begin[trainingPhase] - self.alpha_end[trainingPhase] ) * numpy.exp( -t/timeCte ) + self.alpha_end[trainingPhase]
        return self.learning
        
    def rho(self, k,  BMUindices, Map):
        i,j = BMUindices
        rhoValue = max(scipy.spatial.distance.euclidean(self.inputvectors[k], Map[i,j]), self.rhoValue)
        self.rhoValue = rhoValue
        return rhoValue

    def epsilon(self, k, BMUindices, Map):
        i,j = BMUindices
        return scipy.spatial.distance.euclidean(self.inputvectors[k], Map[i,j]) / self.rho(k, BMUindices, Map)


    def BMUneighbourhood(self, t, BMUindices, trainingPhase, Map = None, k = None):
        i,j = BMUindices
        if self.toricMap:
            i2 = i + self.X
            j2 = j + self.Y
            X,Y=numpy.mgrid[-i2:3*self.X-i2:1,-j2:3*self.Y-j2:1]
        else:
            X,Y=numpy.mgrid[-i:self.X-i,-j:self.Y-j]
        if not self.autoParam:
            adjMap = numpy.exp( -(X**2+Y**2)/ (2.*self.radiusFunction(t, trainingPhase))**2 )
        elif self.autoParam:
            self.epsilon_value = self.epsilon(k,BMUindices,Map)
            radius =self.epsilon_value * self.radius_begin[trainingPhase]
            self.epsilon_values.append(self.epsilon_value)
            adjMap = numpy.exp(-(X**2+Y**2)/ ( 2.* radius )**2 )
        if self.toricMap:
            adjMapR = numpy.zeros((self.X,self.Y,9))
            c = itertools.count()
            for i in range(3):
                for j in range(3):
                    adjMapR[:,:,c.next()] = adjMap[i*self.X:(i+1)*self.X,j*self.Y:(j+1)*self.Y]
            return numpy.max(adjMapR, axis=2)
        else:
            return adjMap

    def adjustment(self, k, t, trainingPhase, Map, BMUindices):
        self.adjustMap = numpy.zeros(Map.shape)
        if not self.autoParam:
            learning = self.learningRate(t, trainingPhase)
            self.adjustMap = numpy.reshape(self.BMUneighbourhood(t, BMUindices, trainingPhase), (self.X, self.Y, 1)) * learning * (self.inputvectors[k] - Map)
        elif self.autoParam:
            radius_map = self.BMUneighbourhood(t, BMUindices, trainingPhase, Map=Map, k=k)
            learning = self.epsilon_value * self.alpha_begin[trainingPhase]
            self.adjustMap = numpy.reshape(radius_map, (self.X, self.Y, 1)) * learning * (self.inputvectors[k] - Map)
        return self.adjustMap
    
    def learn(self, jobIndex='', verbose='False'):
        if self.autoParam:
            self.epsilon_values = []
        Map = self.smap
        print 'Learning for %s vectors'%len(self.inputvectors)
        firstpass=0
        kdone=[]
        for trainingPhase in range(self.number_of_phase):
            kv=[]
            if self.autoParam:
                self.rhoValue = 0
            print '%s iterations'%self.iterations[trainingPhase]
            ## Progress bar
            tpn = trainingPhase + 1
            if verbose:
                widgets = ['Training phase %s : ' % tpn, progressbar.Percentage(), progressbar.Bar(marker='=',left='[',right=']'), progressbar.ETA()]
                pbar = progressbar.ProgressBar(widgets=widgets, maxval=self.iterations[trainingPhase]-1)
                pbar.start()
            ###
            for t in range(self.iterations[trainingPhase]):
                if self.sort2ndPhase and tpn > 1:
                    if len(kv) > 0:
                        k = kv.pop()
                    else:
                        asarkd=numpy.asarray(kdone)
                        print "Computing epsilon values for the current map..."
                        epsvalues=[ self.epsilon(k,self.findBMU(k,Map),Map) for k in asarkd ]
                        indx=numpy.argsort(epsvalues)[::1 if self.autoParam else -1]
                        kv = list(asarkd[indx])
                        k = kv.pop()
                else:
                    if len(kv) > 0:
                        k = kv.pop()
                        if firstpass==1: kdone.append(k)
                    else:
                        firstpass+=1
                        kv = range(len(self.inputvectors))
                        random.shuffle(kv)
                        k = kv.pop()
                        if firstpass==1: kdone.append(k)
                Map = Map + self.adjustment(k, t, trainingPhase, Map, self.findBMU(k, Map))
                if verbose:
                    pbar.update(t)
            if verbose:
                pbar.finish()
        self.smap = Map
        if jobIndex == '':
            MapFile = open('map_%sx%s.dat' % (self.X,self.Y), 'w')
        else:
            MapFile = open('map_%sx%s_%s.dat' % (self.X,self.Y,jobIndex), 'w')
        pickle.dump(Map, MapFile) # Write Map into file map.dat
        MapFile.close()
        if self.autoParam:
            numpy.savetxt('epsilon_values.txt', self.epsilon_values, fmt='%10.5f')
        return self.smap
        
