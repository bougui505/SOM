#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import numpy
import sys
import scipy.spatial
import SOMTools
from scipy import ndimage
import copy
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters

class StepAndSlide:
    def __init__(self, matrix, step, tol, start, stop, dist_tol = 0.):
        self.matrix = matrix
        (self.X,self.Y) = matrix.shape
        self.step = step
        self.tol = tol
        self.start = start
        self.stop = stop
        self.dist_tol = dist_tol
        self.count=0
        self.path = []
        if not self.areIsoEnergetic(start, stop):
            print "Error: Start and stop points are not isoenergetic!"
            sys.exit(0)

    def findIsoEnergeticPoints(self,point):
        endValue = self.matrix[point[0],point[1]]
        isoEnergeticPoints = numpy.logical_and(self.matrix >= endValue - self.tol, self.matrix <= endValue + self.tol)
        return isoEnergeticPoints

    def areIsoEnergetic(self, point1, point2):
        isoEnergeticPoints = self.findIsoEnergeticPoints(point1)
        return isoEnergeticPoints[point2[0],point2[1]]

    def getDistance(self, point1, point2):
        X = self.X
        Y = self.Y
        ds = []
        bs = []
        for i in [-X,0,X]: # for periodicity
            for j in [-Y,0,Y]:
                a = numpy.asarray(point1)
                b = numpy.asarray(point2)+[i,j]
                bs.append(b)
                d = scipy.spatial.distance.euclidean(a, b)
                ds.append(d)
        d = min(ds)
        b = bs[numpy.argmin(ds)]
        return d, a, b

    def getNeighbors(self, point, matrix):
        point = numpy.asarray(point)
        X,Y = matrix.shape
        neighbors = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if i != 0 or j != 0:
                    neighbor = tuple([(point[0] + i)%X, (point[1] + j)%Y])
                    neighbors.append(neighbor)
        return neighbors

    def getMinNeighbor(self, point, matrix):
        neighbors = self.getNeighbors(point, matrix)
        v = []
        for neighbor in neighbors:
            i,j = neighbor
            v.append(matrix[i,j])
        minNeighbor = neighbors[numpy.argmin(v)]
        return minNeighbor

    def getInitialGuess(self, point1, point2):
        X = self.X
        Y = self.Y
        matrix = numpy.ones((X,Y), dtype=int)
        a = point1
        b = point2
        matrix[b[0],b[1]] = 0
        matrix_expand = SOMTools.expandMatrix(matrix)
        distMat = ndimage.distance_transform_edt(matrix_expand)
        pathMat = numpy.zeros_like(distMat,dtype=int)
        a = tuple(numpy.asarray(point1)+numpy.array([self.X,self.Y]))
        b = tuple(numpy.asarray(point2)+numpy.array([self.X,self.Y]))
        a_mod, b_mod = (a[0]%X,a[1]%Y), (b[0]%X,b[1]%Y)
        while a_mod != b_mod:
            a = self.getMinNeighbor(a,distMat)
            i,j = a
            pathMat[i,j] = 1
            a_mod, b_mod = (a[0]%X,a[1]%Y), (b[0]%X,b[1]%Y)
        pathMat = SOMTools.condenseMatrix(pathMat)
        pathMat = numpy.bool_(pathMat)
        return pathMat

    def Step(self, point1, point2):
        stepPath1 = []
        stepPath2 = []
        v1_ori, v2_ori = self.matrix[point1, point2]
        v1, v2 = v1_ori, v2_ori
        test1, test2 = (v1-v1_ori < self.step), (v2-v2_ori < self.step)
        while (test1 or test2) and scipy.spatial.distance.euclidean(point1,point2)>numpy.sqrt(2):
            self.count += 1
            guess = self.getInitialGuess(point1, point2)
            neighbors1 = numpy.asarray(self.getNeighbors(point1,self.matrix))
            neighbors1 = (neighbors1[:,0], neighbors1[:,1])
            neighbors2 = numpy.asarray(self.getNeighbors(point2,self.matrix))
            neighbors2 = (neighbors2[:,0], neighbors2[:,1])
            step1 = tuple(numpy.asarray(neighbors1).T[guess[neighbors1]][0])
            step2 = tuple(numpy.asarray(neighbors2).T[guess[neighbors2]][0])
            if test1:
                v1 = self.matrix[step1]
                point1 = step1
            if test2:
                v2 = self.matrix[step2]
                point2 = step2
            test1, test2 = (v1-v1_ori < self.step), (v2-v2_ori < self.step)
            if point1 not in stepPath1 and test1: stepPath1.append(point1)
            if point2 not in stepPath2 and test2: stepPath2.append(point2)
        return stepPath1, stepPath2

    def getIsoNeighbors(self,point,matrix):
        neighbors= numpy.array(self.getNeighbors(point,matrix))
#        print neighbors
        (i,j) = point
        z = matrix[i,j]
        v = []
        for neighbor in neighbors:
            (i,j) = neighbor
            v.append(matrix[i,j])
        gv = numpy.abs(numpy.array(v)-z)
#        print gv<=self.tol
        neighbors_iso = neighbors[gv<=self.tol]
        neighbors_iso = (neighbors_iso[:,0], neighbors_iso[:,1])
        return neighbors_iso

    def Slide(self, point1, point2):
        slidePath1, slidePath2 = [], []
        matrix = copy.deepcopy(self.matrix)
        d = scipy.spatial.distance.euclidean(point1,point2)
        (i,j),(k,l) = point1, point2
        d_prev = d
        test = (d.min() <= d_prev.min())
        while test:
            self.count += 1
            neighbors1, neighbors2 = numpy.asarray(self.getIsoNeighbors(point1, matrix)).T, numpy.asarray(self.getIsoNeighbors(point2, matrix)).T
            for e in slidePath1:
                if neighbors1.size != 0: neighbors1 = numpy.delete(neighbors1, numpy.where(numpy.all(neighbors1==e,axis=1))[0], axis=0)
            for e in slidePath2:
                if neighbors2.size != 0: neighbors2 = numpy.delete(neighbors2, numpy.where(numpy.all(neighbors2==e,axis=1))[0], axis=0)
#            matrix[(i,k),(j,l)] += self.count*matrix.max()
            d_prev = d
            d = scipy.spatial.distance.cdist(neighbors1, neighbors2)
            if d.size > 0:
                test = (d.min() <= d_prev.min() + self.dist_tol)
                if test:
                    d_min = d.min()
                    arg_min1, arg_min2 = numpy.where(d == d.min())
                    point1, point2 = tuple(neighbors1[arg_min1][0]), tuple(neighbors2[arg_min2][0])
                    (i,j),(k,l) = point1, point2
            else:
                point1, point2 = None, None
                test = False
            if (point1 not in slidePath1 and point1 != None): slidePath1.append(point1)
            if (point2 not in slidePath2 and point2 != None): slidePath2.append(point2)
        return slidePath1, slidePath2


    def run(self):
        path1, path2 = [], []
        A = self.start
        B = self.stop
        path1.append(A)
        path2.append(B)
        d = scipy.spatial.distance.euclidean(A, B)
        while d > numpy.sqrt(2):
            steps = self.Step(A,B)
            path1.extend(steps[0])
            path2.extend(steps[1])
            if steps[0] != []:
                A = steps[0][-1]
            if steps[1] != []:
                B = steps[1][-1]
            print "iter: %d, step: %s,%s,%d steps,%.2f"%(self.count, A,B,len(steps[0])+len(steps[1]),d)
            slides = self.Slide(A,B)
#            if slides[0][-1] == A and slides[1][-1] == B: break
            A_prev = A
            B_prev = B
            if slides[0] != []:
                A = slides[0][-1]
            if slides[1] != []:
                B = slides[1][-1]
            if A == A_prev and B == B_prev: break
            print "iter: %d, slide: %s,%s,%d slides,%.2f"%(self.count, A,B,len(slides[0])+len(slides[1]),d)
            path1.extend(slides[0])
            path2.extend(slides[1])
            self.path.extend(path1)
            self.path.extend(path2)
            d = scipy.spatial.distance.euclidean(A, B)
        return path1, path2
