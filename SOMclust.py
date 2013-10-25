#!/usr/bin/env pyth  
# -*- c ding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2013 10 25
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import copy
import numpy
import scipy.ndimage
import SOMTools
import SOM2

class clusters:

    def __init__(self, umatrix, bmus, smap, waterstop=None):
        self.umatrix = umatrix
        self.umat_cont, self.x_offset, self.y_offset, self.mask, self.waterlevels, self.flooding = self.flood(umatrix, verbose = True, waterstop=waterstop)
        self.bmus = bmus
        self.som = SOM2.SOM()
        self.som.smap = smap

    def flood(self, inputmat, x_offset=None, y_offset=None, mask=None, verbose=False, waterstop = None, startingpoint = None, floodgate = False):
        def arrange(outmatrix):
            x_offset = -(outmatrix==0).all(axis=1)
            y_offset = -(outmatrix==0).all(axis=0)
            mask = outmatrix == 0
            a = mask[x_offset]
            mask = a[:,y_offset]
            return mask, x_offset, y_offset
        if waterstop == None:
            waterstop = inputmat.max()
        mat = copy.deepcopy(inputmat)
        if startingpoint == None:
            matexpand = self.expandMatrix(mat,5)
        else:
            matexpand = mat
        if (x_offset, y_offset, mask) == (None,None,None):
            X, Y = mat.shape
            sortneighbors = lambda n: numpy.asarray(n)[numpy.asarray([mat[e[0]%X,e[1]%Y] for e in n]).argsort()]
            circummat = numpy.zeros_like(matexpand)
            if startingpoint == None:
                u,v = numpy.unravel_index(mat.argmin(), mat.shape)
                u,v = u+2*X, v+2*Y
            else:
                u,v = startingpoint
            circummat[u,v] = mat[u%X,v%Y]
            mat[u%X,v%Y] = numpy.inf
            bayou = [(u,v)]
            neighbors = [item for sublist in [self.getNeighbors(e, matexpand.shape) for e in bayou]
                         for item in sublist]
            i,j = sortneighbors(neighbors)[0]
            count = 0
            waterlevels = []
            n = mat.size - 1
            stopflooding = False
            while count < n:
                flooding = True
                if stopflooding:
                    break
                while flooding:
                    flooding = False
                    neighbors = [item for sublist in [self.getNeighbors(e, matexpand.shape) for e in bayou]
                                 for item in sublist]
                    neighbors = list(set(neighbors) - set(bayou))
                    neighbors = sortneighbors(neighbors)
                    i,j = neighbors[0]
                    waterlevel = mat[i%X,j%Y]
                    if len(waterlevels) > 0:
                        if floodgate and waterlevel < waterlevels[-1]:
                            flooding = False
                            stopflooding = True
                            break
                    if waterlevel > waterstop:
                        flooding = False
                        stopflooding = True
                        break
                    for neighbor in neighbors:
                        i, j = neighbor
                        if mat[i%X,j%Y] <= waterlevel and count < mat.size:
                            waterlevels.append(waterlevel)
                            count += 1
                            u, v = i, j
                            bayou.append((u,v))
                            if verbose:
                                if count % (n / 100) == 0:
                                    print "%.2f/100: flooding: %d/%d, %.2f, (%d, %d)"%(count / (n/100.), count, n, waterlevel,u,v)
                            circummat[u,v] = mat[u%X,v%Y]
                            mat[u%X,v%Y] = numpy.inf
                            flooding = True
                        else:
                            break
            if not floodgate:
                mask, x_offset, y_offset = arrange(circummat)
                a = matexpand[x_offset]
                out = a[:,y_offset]
                flooding = [(e[0]%X, e[1]%Y) for e in bayou]
                return out, x_offset, y_offset, mask, waterlevels, flooding
            else:
                return circummat, circummat == 0
        else:
            a = matexpand[x_offset]
            out = a[:,y_offset]
            return out,x_offset,y_offset,mask

    def expandMatrix(self, matrix, expansionfactor = 3):
     if len(matrix.shape) == 2:
      n,p=matrix.shape
      outMatrix = numpy.zeros((expansionfactor*n,expansionfactor*p))
      for i in range(expansionfactor):
       for j in range(expansionfactor):
        outMatrix[i*n:(i+1)*n,j*p:(j+1)*p] = matrix
     elif len(matrix.shape) == 3:
      n,p,k=matrix.shape
      outMatrix = numpy.zeros((expansionfactor*n,expansionfactor*p,k))
      for i in range(expansionfactor):
       for j in range(expansionfactor):
        outMatrix[i*n:(i+1)*n,j*p:(j+1)*p] = matrix
     return outMatrix

    def getNeighbors(self, pos,shape):
        X,Y = shape
        i,j = pos
        neighbors = []
        for k in range(i-1,i+2):
            for l in range(j-1,j+2):
                if k != i or l != j:
                    neighbors.append((k%X,l%Y))
        return neighbors

    def detect_local_minima2(self, arr, toricMap=False):
        X,Y = arr.shape
        lminima = []
        for i in range(X):
            for j in range(Y):
                pos = (i,j)
                neighbors = self.getNeighbors(pos, (X,Y))
                nvalues = numpy.asarray( [ arr[e[0],e[1]] for e in neighbors] )
                if (arr[i,j] <= nvalues).all():
                    lminima.append((i,j))
        lminima = numpy.asarray(lminima)
        lminima = (lminima[:,0], lminima[:,1])
        return lminima

    def continuousMap(self, clusters):
     for i in range(clusters.shape[0]):
      if clusters[i,0] != 0 and clusters[i,-1] != 0:
       clusters[clusters == clusters[i,-1]] = clusters[i,0]
     for j in range(clusters.shape[1]):
      if clusters[0,j] != 0 and clusters[-1,j] != 0:
       clusters[clusters == clusters[-1,j]] = clusters[0,j]
     c = 1
     for e in numpy.unique(clusters)[1:]:
      clusters[clusters==e] = c
      c+=1
     return clusters

    def getclusters(self):
        self.localminima = self.detect_local_minima2( numpy.ma.masked_array(self.umat_cont, self.mask))
        self.filteredumat = copy.deepcopy(self.umat_cont)
###Sort local minima
        minimasorter = numpy.asarray([self.filteredumat[(u,self.localminima[1][i])] for i,u in enumerate(self.localminima[0])]).argsort()
        self.localminima = list(self.localminima)
        self.localminima[0], self.localminima[1] = self.localminima[0][minimasorter], self.localminima[1][minimasorter]
###
        i,j = self.umat_cont.shape
        k = self.localminima[0].size
        cmats = numpy.zeros((i,j,k), dtype='int')
        for i, u in enumerate(self.localminima[0]):
            v = self.localminima[1][i]
            lake, masklake = self.flood(self.filteredumat, startingpoint=(u,v), floodgate=True, verbose=False)
            self.cmat = 1 - masklake
            self.filteredumat[self.cmat==1] = numpy.inf
            self.cmat[self.cmat==1] = i + 1
            cmats[:,:,i] = self.cmat
        cmats = cmats[:,:,numpy.argsort(cmats.sum(axis=0).sum(axis=0))][:,:,::-1]
        self.cmat = numpy.zeros((cmats.shape)[:2], dtype='int')
        for m in cmats.T:
            m = m.T
            n, p = self.cmat.shape
            for i in range(n):
                for j in range(p):
                    if self.cmat[i,j] == 0:
                        self.cmat[i,j] = m[i,j]
        self.cmat = self.continuousMap(self.cmat)
        self.offsetmat = (((numpy.asarray(numpy.meshgrid(range(self.x_offset.size),range(self.y_offset.size))).T)[self.x_offset][:,self.y_offset]))
        X,Y = self.umatrix.shape
        self.offsetmat[:,:,0] = self.offsetmat[:,:,0]%X
        self.offsetmat[:,:,1] = self.offsetmat[:,:,1]%Y
        self.labels = []
        for e in self.bmus:
            i,j = e
            u,v = numpy.nonzero(((self.offsetmat - numpy.asarray([i,j])[None, None, :]) == 0).all(axis=2))
            self.labels.append(self.cmat[u,v].max())
        self.labels = numpy.asarray(self.labels)

        #fill up clusters
        smallclustmat = numpy.zeros_like(self.umatrix, dtype=int)
        for i in range(self.umat_cont.shape[0]):
            for j in range(self.umat_cont.shape[1]):
                if self.cmat[i,j] != 0:
                    ip,jp = self.offsetmat[i,j]
                    smallclustmat[ip,jp] = self.cmat[i,j]
        smallclustmat = SOMTools.continuousMap(scipy.ndimage.label(smallclustmat!=0)[0])
        clustmatori = copy.deepcopy(smallclustmat)
        tofill = smallclustmat==0
        fillupindex = self.som.get_allbmus(vectors=self.som.smap[tofill], smap=self.som.smap[smallclustmat!=0])
        for k,e in enumerate(numpy.asarray(numpy.nonzero(tofill)).T):
            i,j = e
            c = clustmatori[clustmatori!=0][fillupindex[k]]
            smallclustmat[i,j] = c
        erodedmap = numpy.zeros_like(smallclustmat)
        for e in numpy.unique(smallclustmat):
            emap = SOMTools.condenseMatrix(scipy.ndimage.binary_erosion(SOMTools.expandMatrix(smallclustmat==e)))
            erodedmap[emap==1] = emap[emap==1]
        self.cmat = self.flood(smallclustmat, x_offset=self.x_offset, y_offset=self.y_offset, mask=self.mask)[0]
        self.erodedmap = self.flood(erodedmap, x_offset=self.x_offset, y_offset=self.y_offset, mask=self.mask)[0]
        return self.cmat, self.erodedmap
