#!/usr/bin/env python
import numpy
import itertools
import cPickle
import random
import sys

class analysis:
 def __init__(self, map=None, mapFileName=None, distFunc = None):
  self.distFunc = distFunc
  if mapFileName != None:
   self.Map = self.loadMap(mapFileName)
  else:
   self.Map = map
  shape = numpy.shape(self.Map)
  self.X = shape[0]
  self.Y = shape[1]
  clustersMap = self.clusterDiscovery(0)
  distanceMat = self.clusterDistance(clustersMap)
  self.distanceMatrix = distanceMat

 def loadMap(self, MapFile):
  MapFileFile = open(MapFile, 'r')
  Map = cPickle.load(MapFileFile)
  MapFileFile.close()
  return Map

 def clusterDiscovery(self,T):
  borderMat = self.borderline()
  Nv = [[i,j] for i in range(self.X) for j in range(self.Y)]
  random.shuffle(Nv)
  Cv = []
  while Nv != []:
   N = Nv[0] # locate an arbitrary node N
   C = [] # start a new cluster C
   A = [] # Adjacent nodes
   A, C, Nv = self.cluster(borderMat,Nv,N,T,C,A) # call procedure cluster for N, C and T
   while A != []:
    N = A[0]
    A.remove(N)
    A, C, Nv = self.cluster(borderMat,Nv,N,T,C,A) # call procedure cluster for N, C and T
   Cv.append(C)
  clustersMap = numpy.zeros((self.X,self.Y))
  CId = 0
  for v in Cv:
   CId = CId + 1
   for e in v:
    clustersMap[e[0],e[1]] = CId
  return clustersMap
   
 def cluster(self,borderMat,Nv,N,T,C,A):
  C.append(N) # assign N to C
  #print 'Nv%s'%Nv
  Nv.remove(N) # mark N as visited
  c = 0
  for i in range(N[0]*2,N[0]*2+2):
   for j in range(N[1]*2,N[1]*2+2):
    c = c + 1
    if borderMat[i,j] <= T: # distance < Threshold
     if (c == 1 and N[1] >= 1):
      fc = [N[0],N[1]-1]
      if (fc in Nv and fc not in A): # unvisited node
       A.append(fc)
     if (c == 2 and N[0] >= 1):
      fc = [N[0]-1,N[1]]
      if (fc in Nv and fc not in A): # unvisited node
       A.append(fc)
     if (c == 3 and N[0] <= self.X - 1):
      fc = [N[0]+1,N[1]]
      if (fc in Nv and fc not in A): # unvisited node
       A.append(fc)
     if (c == 4 and N[1] <= self.Y - 1):
      fc = [N[0],N[1]+1]
      if (fc in Nv and fc not in A): # unvisited node
       A.append(fc)
  return A, C, Nv

 def borderline(self):
  Map = self.Map
  borderMat = numpy.zeros((self.X*2,self.Y*2))
  initMat = numpy.repeat(numpy.repeat(Map,2,axis=0),2,axis=1)
  for i in range(0,self.X*2,2):
   for j in range(0,self.Y*2,2):
    if j > 1:
     if self.distFunc is None:
      borderMat[i,j] = ( numpy.dot( initMat[i,j]-initMat[i,j-1], numpy.transpose( initMat[i,j]-initMat[i,j-1] ) ) )**0.5
     else:
      borderMat[i,j] = self.distFunc(initMat[i,j], initMat[i,j-1])
    else:
     borderMat[i,j] = 0
  for i in range(0,self.X*2,2):
   for j in range(1,self.Y*2,2):
    if i > 1:
     if self.distFunc is None:
      borderMat[i,j] = ( numpy.dot( initMat[i,j]-initMat[i-1,j], numpy.transpose( initMat[i,j]-initMat[i-1,j] ) ) )**0.5
     else:
      borderMat[i,j] = self.distFunc(initMat[i,j], initMat[i-1,j])
    else:
     borderMat[i,j] = 0
  for i in range(1,self.X*2,2):
   for j in range(0,self.Y*2,2):
    try:
     if self.distFunc is None:
      borderMat[i,j] = ( numpy.dot( initMat[i,j]-initMat[i+1,j], numpy.transpose( initMat[i,j]-initMat[i+1,j] ) ) )**0.5
     else:
      borderMat[i,j] = self.distFunc(initMat[i,j], initMat[i+1,j])
    except IndexError:
     #borderMat[i,j] = ( numpy.dot( initMat[i,j]-initMat[0,j], numpy.transpose( initMat[i,j]-initMat[0,j] ) ) )**0.5
     borderMat[i,j] = 0
  for i in range(1,self.X*2,2):
   for j in range(1,self.Y*2,2):
    try:
     if self.distFunc is None:
      borderMat[i,j] = ( numpy.dot( initMat[i,j]-initMat[i,j+1], numpy.transpose( initMat[i,j]-initMat[i,j+1] ) ) )**0.5
     else:
      borderMat[i,j] = self.distFunc(initMat[i,j], initMat[i,j+1])
    except IndexError:
     #borderMat[i,j] = ( numpy.dot( initMat[i,j]-initMat[i,0], numpy.transpose( initMat[i,j]-initMat[i,0] ) ) )**0.5
     borderMat[i,j] = 0
  #self.borderMat = borderMat[1:self.X*2,1:self.Y*2]/borderMat[1:self.X*2,1:self.Y*2].max()
  borderMat = borderMat/borderMat.max() #Normalize borderMat values between 0 and 1
  return borderMat
 
 def clusterLooseness(self, clustersMap):
  """
  return an self.X * self.Y matrix with cluster containing cluster looseness values
  """
  Map = self.Map
  clusterLoosenessMat = numpy.zeros((self.X,self.Y))
  for CId in range(int(clustersMap.min()), int(clustersMap.max() + 1)):
   C = [[numpy.where(clustersMap == CId)[0][i], numpy.where(clustersMap == CId)[1][i]] for i in range(len(numpy.where(clustersMap == CId)[0]))]
   loosenessMat = numpy.zeros((len(C),len(C)))
   i = 0
   for N1 in C:
    j = 0
    for N2 in C:
     loosenessMat[i,j] = (numpy.dot( Map[N1[0],N1[1]]-Map[N2[0],N2[1]], numpy.transpose(Map[N1[0],N1[1]]-Map[N2[0],N2[1]]) ))**0.5
     j = j + 1
    i = i + 1
   #print loosenessMat.sum()
   if len(C) > 1:
    looseness = (loosenessMat.sum()/2) / ( len(C)*(len(C) - 1)/2 )
   else:
    looseness = loosenessMat.sum()/2
   for N in C:
     clusterLoosenessMat[N[0],N[1]] = looseness
  return clusterLoosenessMat

 def clusterDistance(self, clustersMap):
  Map = self.Map
  m = int(clustersMap.max()) # Total number of clusters
  #print m
  clusterDistanceMat = numpy.zeros((m,m))
  for i in range(m):
   for j in range(m):
    CId1 = i + 1
    CId2 = j + 1
    C1 = [[numpy.where(clustersMap == CId1)[0][k], numpy.where(clustersMap == CId1)[1][k]] for k in range(len(numpy.where(clustersMap == CId1)[0]))]
    C2 = [[numpy.where(clustersMap == CId2)[0][l], numpy.where(clustersMap == CId2)[1][l]] for l in range(len(numpy.where(clustersMap == CId2)[0]))]
    distanceMat = numpy.zeros((len(C1),len(C2)))
    i2 = 0
    for N1 in C1:
     j2 = 0
     for N2 in C2:
      if CId1 != CId2:
       if self.distFunc is None:
        distanceMat[i2,j2] = (numpy.dot( Map[N1[0],N1[1]]-Map[N2[0],N2[1]], numpy.transpose(Map[N1[0],N1[1]]-Map[N2[0],N2[1]]) ))**0.5
       else:
        distanceMat[i2,j2] = self.distFunc(Map[N1[0],N1[1]], Map[N2[0],N2[1]])
      j2 = j2 + 1
     i2 = i2 + 1
    #print CId1, CId2
    #print distanceMat
    clusterDistanceMat[i,j] = distanceMat.mean()
  return clusterDistanceMat

 def xi(self, loosenessMat, distanceMat):
  m = numpy.shape(distanceMat)[0]
  if m == 1:
   return None
  else:
   khi = numpy.mean(loosenessMat)
   delta = numpy.mean(distanceMat)
   xi_value = m*khi / delta
   return xi_value

 def xiT(self):
  print 'Finding the best clustering ...'
  Tv = numpy.unique(self.distanceMatrix)/numpy.max(self.distanceMatrix)
  xiPrev = 0.
  Tprev = 0.
  gradXiMax = 0
  nClustPrev = 0
  for T in Tv:
   clustersMap = self.clusterDiscovery(T)
   nClust = len(numpy.unique(clustersMap))
   if nClust != nClustPrev:
    nClustPrev = nClust
    loosenessMat = self.clusterLooseness(clustersMap)
    distanceMat = self.clusterDistance(clustersMap)
    xi = self.xi(loosenessMat, distanceMat)
    if xi == None:
     break
    gradXi = (xi - xiPrev) / (T - Tprev)
    sys.stdout.write('Threshold:'+('%.2f'%T).rjust(6)+'; xi:'+('%.2f'%xi).rjust(7)+'; grad(xi):'+('%.2f'%gradXi).rjust(12)+'; nClust:'+('%s'%nClust).rjust(4))
    sys.stdout.flush()
    sys.stdout.write("\r")
    if abs(gradXi) >= abs(gradXiMax):
#    if gradXi > 0:
#     T_best = Tprev
#     xi_best = xiPrev
     if gradXi < 0:
      gradXiMax = gradXi
      T_best = T
      xi_best = xi
    xiPrev = xi
    Tprev = T
  print '\nBest clustering for threshold : %.2f; xi: %.2f' % (T_best, xi_best)
  return T_best
