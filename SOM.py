#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import RandomArray
import re
import math
import matplotlib.pyplot
import random
import progressbar
import pickle
import ROCSOM
import sys
from multiprocessing import Process, Queue
import itertools
import scipy.spatial
import tarfile
import os

class SOM3D:
 """
 Main class.
  Attributes:
   cardinal         : integer, length of input vectors
   inputvectors     : list of lists of objects, input vectors
   inputnames       : list of integers, vector names
   X                : integer, width of Kohonen map
   Y                : integer, height of Kohonen map
   Z                : integer, depth of Kohonen map
   number_of_phases : integer, number of training phases
 """
 def __init__(self, inputvectors, inputnames, confname = 'SOM3D.conf',simplify_vectors=False, distFunc=None, randomUnit=None, mapFileName=None, metric = 'euclidean', autoParam = False):
  self.metric = metric
  self.cardinal = len(inputvectors[0])
  self.inputvectors = inputvectors
  self.inputnames = inputnames
  self.autoParam = autoParam
  conffile = open(confname, 'r')
  lines = conffile.readlines()
  conffile.close()
  test = False
  for line in lines:
   if re.findall('<TreeSOM>', line):
    test = True
   if test:
    if re.findall('#', line):
     line = line.split('#')[0]
    if re.findall(r'X\s*=', line):
     self.X = int(line.split('=')[1]) # Number of neurons in X dimension
    if re.findall(r'Y\s*=', line):
     self.Y = int(line.split('=')[1]) # Number of neurons in Y dimension
    if re.findall(r'Z\s*=', line):
     self.Z = int(line.split('=')[1]) # Number of neurons in Z dimension
    if re.findall(r'number_of_phase\s*=', line):
     self.number_of_phase = int(line.split('=')[1]) # Number of training phase
  i = 1
  self.alpha_begin = []
  self.alpha_end = []
  self.radius_begin = []
  self.radius_end = []
  self.iterations = []
  while i <= self.number_of_phase:
   test = False
   for line in lines:
    if re.findall('<TreeSOM>', line):
     test = True
    if test:
     if re.findall(r'#', line):
      line = line.split('#')[0]
     if re.findall(r'alpha_begin_%s\s*='%i, line):
      self.alpha_begin.append(float(line.split('=')[1]))
     if re.findall(r'alpha_end_%s\s*='%i, line):
      self.alpha_end.append(float(line.split('=')[1]))
     if re.findall(r'radius_begin_%s\s*='%i, line):
      self.radius_begin.append(float(line.split('=')[1]))
     if re.findall(r'radius_end_%s\s*='%i, line):
      self.radius_end.append(float(line.split('=')[1]))
     if re.findall(r'iterations_%s\s*='%i, line):
      self.iterations.append(int(line.split('=')[1]))
   i=i+1
  # Vector simplification
  if simplify_vectors:
   self.inputvectors = self.simplifyVectors()
   self.cardinal = len(self.inputvectors[0])
  if randomUnit is None:
   # Matrix initialization
   if mapFileName == None:
    maxinpvalue = self.inputvectors.max()
    mininpvalue = self.inputvectors.min()
    somShape = [self.X, self.Y, self.Z]
    vShape = numpy.array(self.inputvectors[0]).shape
    for e in vShape:
     somShape.append(e)
    self.M = RandomArray.uniform(mininpvalue,maxinpvalue,somShape)
   else:
    self.M = self.loadMap(mapFileName)
   print "Shape of the SOM:%s"%str(self.M.shape)
  self.distFunc = distFunc

 def loadMap(self, MapFile):
  MapFileFile = open(MapFile, 'r')
  self.Map = pickle.load(MapFileFile)
  MapFileFile.close()
  shape = numpy.shape(self.Map)
  self.X = shape[0]
  self.Y = shape[1]
  self.Z = shape[2]
  return self.Map
  
 def makeSubInputvectors(self, n, jobIndex=''):
  """
  Make a self.inputvectors variable with all known ligands and a random set of n unknown ligands
  """
  if jobIndex == '':
   rocsom = ROCSOM.ROCSOM(MapFile = 'map_%sx%sx%s.dat' % (self.X,self.Y,self.Z))
  else:
   rocsom = ROCSOM.ROCSOM(MapFile = 'map_%sx%sx%s_%s.dat' % (self.X,self.Y,self.Z,jobIndex))
  KL = rocsom.KL
  UL = rocsom.UL
  rmL = random.sample(UL, len(UL)-n) # list of randomly chosen ligands to remove
  for U in rmL:
   self.inputvectors.pop(self.inputnames.index(U))
   self.inputnames.remove(U)
   
 def makeNewInputvectors(self, ligands_list):
  """
  Make a new input vectors variable with all ligands contains in ligands_list
  """
  for L in self.inputnames:
   if L not in ligands_list:
    self.inputvectors.pop(self.inputnames.index(L))
    self.inputnames.remove(L)

 def simplifyVectors(self):
  """
   Remove systematic zeros in vectors
  """
  sumV = []
  for Eindex in range(self.cardinal):
   S = 0
   for Vindex in range(len(self.inputvectors)):
    S = S + self.inputvectors[Vindex][Eindex]
   sumV.append(S)
  c = 0
  nonzerosindex = []
  for e in sumV:
   if e!=0:
    nonzerosindex.append(c)
   c = c + 1
  simplifiedVectors = []
  for V in self.inputvectors:
   simplifiedVector = [V[i] for i in nonzerosindex]
   simplifiedVectors.append(simplifiedVector)
  self.simplifiedVectors = simplifiedVectors
  return self.simplifiedVectors

 def findBMU(self, k, Map, distKW=None):
  """
   Find the Best Matching Unit for the input vector number k
  """
  return numpy.unravel_index(scipy.spatial.distance.cdist(numpy.reshape(self.inputvectors[k], (1,self.cardinal)), numpy.reshape(Map, (self.X*self.Y*self.Z,self.cardinal)), self.metric).argmin(), (self.X,self.Y,self.Z))
  
 def defaultDist(self, vector, Map, distKW):
  X,Y,Z,cardinal=distKW['X'],distKW['Y'],distKW['Z'],distKW['cardinal']
  V=numpy.ones((X,Y,Z,cardinal))*vector
  return numpy.sum( (V - Map) * ( (V - Map)* (numpy.ones([X,Y,Z,cardinal])*numpy.ones([1,cardinal])) ) , axis=3 )**0.5
  
 def radiusFunction(self, t, trainingPhase=0):
  timeCte = float(self.iterations[trainingPhase])/10
  self.radius = ( self.radius_begin[trainingPhase] - self.radius_end[trainingPhase] ) * math.exp( -t/timeCte ) + self.radius_end[trainingPhase]
  return self.radius
  
 def learningRate(self, t, trainingPhase=0):
  timeCte = float(self.iterations[trainingPhase])/10
  self.learning = ( self.alpha_begin[trainingPhase] - self.alpha_end[trainingPhase] ) * math.exp( -t/timeCte ) + self.alpha_end[trainingPhase]
  return self.learning
  
 def rho(self, k,  BMUindices, Map):
  i,j,z = BMUindices
  dist=scipy.spatial.distance.euclidean(self.inputvectors[k], Map[i,j,z])
  rhoValue = max(dist, self.rhoValue)
  self.rhoValue = rhoValue
  return rhoValue

 def epsilon(self, k, BMUindices, Map):
  i,j,z = BMUindices
  eps=min(scipy.spatial.distance.euclidean(self.inputvectors[k], Map[i,j,z]) / self.rho(k, BMUindices, Map),0.5)
#  print "%.4f %.4f"%(eps,eps*self.radius_begin[0])
  return eps


 def BMUneighbourhood(self, t, BMUindices, trainingPhase, Map = None, k = None):
  i,j,z = BMUindices
  i2 = i + self.X
  j2 = j + self.Y
  z2 = z + self.Z
  X,Y,Z=numpy.mgrid[-i2:3*self.X-i2:1,-j2:3*self.Y-j2:1,-z2:3*self.Z-z2:1]
  if not self.autoParam:
   adjMap = numpy.exp( -(X**2+Y**2+Z**2)/ (2.*self.radiusFunction(t, trainingPhase))**2 )
  elif self.autoParam:
   self.epsilon_value = self.epsilon(k,BMUindices,Map)
   radius_auto =self.epsilon_value * self.radius_begin[trainingPhase]
   radius = min(self.radiusFunction(t, trainingPhase), radius_auto)
   self.epsilon_values.append(self.epsilon_value)
   adjMap = numpy.exp(-(X**2+Y**2+Z**2)/ ( 2.* radius )**2 )
  adjMapR = numpy.zeros((self.X,self.Y,self.Z,27))
  c = itertools.count()
  for i in range(3):
   for j in range(3):
    for z in range(3):
     adjMapR[:,:,:,c.next()] = adjMap[i*self.X:(i+1)*self.X,j*self.Y:(j+1)*self.Y,z*self.Z:(z+1)*self.Z]
  return numpy.max(adjMapR, axis=3)

 def adjustment(self, k, t, trainingPhase, Map, BMUindices):
  self.adjustMap = numpy.zeros(Map.shape)
  if not self.autoParam:
   learning = self.learningRate(t, trainingPhase)
   self.adjustMap = numpy.reshape(self.BMUneighbourhood(t, BMUindices, trainingPhase), (self.X, self.Y, self.Z, 1)) * learning * (self.inputvectors[k] - Map)
  elif self.autoParam:
   radius_map = self.BMUneighbourhood(t, BMUindices, trainingPhase, Map=Map, k=k)
   learning = self.epsilon_value
   self.adjustMap = numpy.reshape(radius_map, (self.X, self.Y, self.Z, 1)) * learning * (self.inputvectors[k] - Map)
  return self.adjustMap
 
 def learn(self, jobIndex='', nSnapshots = 50):
  if self.autoParam:
   self.epsilon_values = []
  Map = self.M
  kv = range(len(self.inputvectors))
  print 'Learning for %s vectors'%len(self.inputvectors)
  for trainingPhase in range(self.number_of_phase):
   if self.autoParam:
    self.rhoValue = 0
   print '%s iterations'%self.iterations[trainingPhase]
   ## Progress bar
   tpn = trainingPhase + 1
   widgets = ['Training phase %s : ' % tpn, progressbar.Percentage(), progressbar.Bar(marker='=',left='[',right=']'), progressbar.ETA()]
   pbar = progressbar.ProgressBar(widgets=widgets, maxval=self.iterations[trainingPhase]-1)
   pbar.start()
   ###
   snapshots = range(0, self.iterations[trainingPhase], self.iterations[trainingPhase]/nSnapshots)
   for t in range(self.iterations[trainingPhase]):
    try:
     k = random.choice(kv)
     kv.remove(k)
    except IndexError:
     kv = range(len(self.inputvectors))
     k = random.choice(kv)
     kv.remove(k)
    Map = Map + self.adjustment(k, t, trainingPhase, Map, self.findBMU(k, Map))
    if t in snapshots:
     snapFileName = 'MapSnapshot_%s_%s.npy'%(trainingPhase,t)
     numpy.save(snapFileName, Map)
     tar = tarfile.open('MapSnapshots.tar', 'a')
     tar.add(snapFileName)
     tar.close()
     os.remove(snapFileName)
    pbar.update(t)
   pbar.finish()
  self.Map = Map
  if jobIndex == '':
   MapFile = open('map_%sx%sx%s.dat' % (self.X,self.Y,self.Z), 'w')
  else:
   MapFile = open('map_%sx%sx%s_%s.dat' % (self.X,self.Y,self.Z,jobIndex), 'w')
  pickle.dump(Map, MapFile) # Write Map into file map.dat
  MapFile.close()
  if self.autoParam:
   numpy.savetxt('epsilon_values.txt', self.epsilon_values, fmt='%10.5f')
  return self.Map
  
 def distmapPlot(self,k,Map):
  # TODO: make this a .map generator for use with vmd (3D) - shouldn't be functional right now
  V=numpy.ones((self.X,self.Y,self.Z,self.cardinal))*self.inputvectors[k]
  distmat = numpy.sum( (V - Map) * ( (V-Map)* (numpy.ones([self.X,self.Y,self.Z,self.cardinal])*numpy.ones([1,self.cardinal])) ) , axis=3 )**0.5
  matplotlib.pyplot.imshow(numpy.ones((self.X,self.Y,self.Z))-distmat/distmat.max(), interpolation='nearest') # Normalized Map for color plot
  matplotlib.pyplot.show()
  
 def mapPlot(self,Map):
  matplotlib.pyplot.imshow(Map/Map.max(), interpolation='nearest') # Normalized Map for color plot
  matplotlib.pyplot.show()

 def borderline(self,Map):
  # useless (dixit guillaume)
  borderMat = numpy.zeros((self.X*2,self.Y*2,self.Z*2))
  initMat = numpy.repeat(numpy.repeat(numpy.repeat(Map,2,axis=0),2,axis=1),2,axis=2)
  for i in range(0,self.X*2,2):
   for j in range(0,self.Y*2,2):
    for k in range(0,self.Z*2,2):
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
  self.borderMat = borderMat/borderMat.max() #Normalize borderMat values between 0 and 1
  return self.borderMat

 def clusterDiscovery(self,Map,T):
  # useless (dixit guillaume)
  borderMat = self.borderline(Map)
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
  self.clustersMap = clustersMap
  return self.clustersMap
   
 def cluster(self,borderMat,Nv,N,T,C,A):
  # useless (dixit guillaume)
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

 def clusterLooseness(self, clustersMap, Map):
  # useless (dixit guillaume)
  """
  return an self.X * self.Y matrix with cluster containing cluster looseness values
  """
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
  self.clusterLoosenessMat = clusterLoosenessMat
  return self.clusterLoosenessMat
  
 def clusterDistance(self, clustersMap, Map):
  # useless (dixit guillaume)
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
  self.clusterDistanceMat = clusterDistanceMat
  return self.clusterDistanceMat
    
 def calibration(self, Map, clustersMap, BMUs = None, name = False):
  # useless (dixit guillaume)
  dataClusters = {}
  if name:
   nameClusters = {}
  CIds = numpy.sort(numpy.unique(clustersMap.ravel()))
  for CId in CIds:
   dataClusters.update({CId:[]})
   if name:
    nameClusters.update({CId:[]})
  for k in range(len(self.inputvectors)):
   if BMUs == None:
    BMU = self.findBMU(k, Map)
   else:
    BMU = BMUs[k]
   CId = clustersMap[BMU[0],BMU[1]]
   dataClusters[CId].append(self.inputvectors[k])
   if name:
    nameClusters[CId].append(self.inputnames[k])
  for CId in CIds:
   if dataClusters[CId] == []:
    dataClusters.pop(CId)
    if name:
     nameClusters.pop(CId)
  self.dataClusters = dataClusters
  if name:
   self.nameClusters = nameClusters
   return self.dataClusters, self.nameClusters
  else:
   return self.dataClusters
    
 def dataLooseness(self, dataClusters, clustersMap, q=None):
  # useless (dixit guillaume)
  #m = len(dataClusters.keys())
  CIds = dataClusters.keys()
  dataLoosenessMat = numpy.ones((self.X,self.Y))*(-1)
  for CId in CIds:
   loosenessMat = numpy.zeros((len(dataClusters[CId]),len(dataClusters[CId])))
   i = 0
   for data1 in dataClusters[CId]:
    j = 0
    n = len(dataClusters[CId])
    data1 = numpy.array(data1)
    for data2 in dataClusters[CId]:
     data2 = numpy.array(data2)
     if self.distFunc is None:
      loosenessMat[i,j] = (numpy.dot( (data1-data2), numpy.transpose( (data1-data2) ) ))**0.5
     else:
      loosenessMat[i,j] = self.distFunc(data1,data2)
     j = j + 1
#     sys.stdout.write('dataLooseness calculation for cluster %s (%s/%s)'%(str(int(CId)).rjust(4), str(j).rjust(4), str(n).rjust(4)))
#     sys.stdout.flush()
#     sys.stdout.write("\r")
    i = i + 1
   if len(dataClusters[CId]) > 1:
    looseness = (loosenessMat.sum()/2) / ( len(dataClusters[CId])*(len(dataClusters[CId]) - 1)/2 )
   else:
    looseness = loosenessMat.sum()/2
   C = [[numpy.where(clustersMap == CId)[0][i], numpy.where(clustersMap == CId)[1][i]] for i in range(len(numpy.where(clustersMap == CId)[0]))]
   for N in C:
    dataLoosenessMat[N[0],N[1]] = looseness
  numpy.putmask(dataLoosenessMat, dataLoosenessMat==-1, -dataLoosenessMat.max())
  self.dataLoosenessMat = dataLoosenessMat
#  sys.stdout.write("\n")
  if q == None:
   return self.dataLoosenessMat
  else:
   q.put(self.dataLoosenessMat)
  
 def dataDistance(self, dataClusters, q=None):
  # useless (dixit guillaume)
  CIds = dataClusters.keys()
  dataDistanceMat = numpy.zeros((len(CIds),len(CIds)))
  i = 0
  for CId1 in CIds:
   j = 0
   for CId2 in CIds:
    if j > i:
#     sys.stdout.write('dataDistance calculation for cluster %s, %s'%(str(int(CId1)).rjust(4),str(int(CId2)).rjust(4)))
#     sys.stdout.flush()
#     sys.stdout.write("\r")
     dataC1 = dataClusters[CId1]
     dataC2 = dataClusters[CId2]
     distanceMat = numpy.zeros((len(dataC1),len(dataC2)))
     i2 = 0
     for data1 in dataC1:
      data1 = numpy.array(data1)
      j2 = 0
      for data2 in dataC2:
       data2 = numpy.array(data2)
       if CId1 != CId2:
        if self.distFunc is None:
         distanceMat[i2,j2] = (numpy.dot(data1-data2, numpy.transpose(data1-data2)))**0.5
        else:
         distanceMat[i2,j2] = self.distFunc(data1,data2)
       j2 = j2 + 1
      i2 = i2 + 1
     dataDistanceMat[i,j] = distanceMat.mean()
    j = j + 1
   i = i + 1
  shape = numpy.shape(dataDistanceMat)
  for i in range(shape[0]):
   for j in range(i):
    dataDistanceMat[i,j] = dataDistanceMat[j,i]
  self.dataDistanceMat = dataDistanceMat
#  sys.stdout.write("\n")
  if q == None:
   return self.dataDistanceMat
  else:
   q.put(self.dataDistanceMat)
  
 def xi(self, dataLoosenessMat, dataDistanceMat):
  # useless (dixit guillaume)
  m = numpy.shape(dataDistanceMat)[0]
  if m == 1:
   return None
  else:
   khi = numpy.mean(dataLoosenessMat)
   delta = numpy.mean(dataDistanceMat)
   self.xi_value = khi / delta
   return self.xi_value
   
 def xiT(self, Map):
  # useless (dixit guillaume)
  """
  Find the best clustering.
  """
  print 'Finding the best clustering ...'
#  step = 0.01
#  Tv = numpy.arange(0,1+step,step)
#  clustersMap = self.clusterDiscovery(Map, 0)
#  distMatrix = self.clusterDistance(clustersMap, Map)
#  Tv = numpy.unique(distMatrix)[1:]/max(numpy.unique(distMatrix)[1:])
  borderMat = self.borderline(Map)
  Tv = [e for e in numpy.unique(borderMat) if e<=1]
  gradXi_value = 0.
  BMUs = []
  for k in range(len(self.inputvectors)):
   BMUs.append(self.findBMU(k, Map))
  start = True
  T_old = 0.
  xis = []
  q1 = Queue(maxsize=-1)
  q2 = Queue(maxsize=-1)
  for T in Tv:
   clustersMap = self.clusterDiscovery(Map,T)
   dataClusters = self.calibration(Map, clustersMap)
   p1 = Process(target=self.dataLooseness, args=(dataClusters, clustersMap,q1))
   p1.start()
   p2 = Process(target=self.dataDistance, args=(dataClusters,q2))
   p2.start()
   dataLoosenessMat = q1.get()
   dataDistanceMat = q2.get()
   xi = self.xi(dataLoosenessMat, dataDistanceMat)
   if start:
    xi_new = xi
    xi_old = xi_new
    start = False
   else:
    xi_old = xi_new
    xi_new = xi
    if xi_new == None:
     break
   if xi_new >= 1:
    break
   gradXi = (xi_new-xi_old)/(T-T_old)
   sys.stdout.write('Threshold: %.2f; xi: %.2f; grad(xi): %.2f'%(T, xi_new,gradXi))
   sys.stdout.flush()
   sys.stdout.write("\r")
   if abs(gradXi) > abs(gradXi_value):
    gradXi_value = gradXi
    if gradXi > 0:
     T_best = T_old
     xi_value = xi_old
    else:
     T_best = T
     xi_value = xi_new
   T_old = T
  self.xi_best = xi_value
  self.T_best = T_best
  print '\nBest clustering for threshold : %.2f; xi: %.2f' % (T_best, xi_value)
  return self.T_best
    
 def tree(self, Map, jobIndex=''):
  # useless (dixit guillaume)
  print 'Building tree from Kohonen map'
  ### finding BMUs for Map
  BMUs = []
  for k in range(len(self.inputvectors)):
   BMUs.append(self.findBMU(k, Map))
  ###
  step_value = 1E-1
  step = step_value
  Tv = numpy.arange(0,1+step,step)
  clustersMap = self.clusterDiscovery(Map,0)
  dataClusters = self.calibration(Map, clustersMap, BMUs, name = True)[1]
  clusters_old = dataClusters.values() # ensemble of clusters
  elementary_clusters = clusters_old
  test = len(dataClusters)
  dist = {} # distances mapping ( {[cluster]:dist} )
  uniqueGroupTest = 0
  #for T in Tv:
  T = 0
  while 1:
   T = T + step
   clustersMap = self.clusterDiscovery(Map,T)
   dataClusters = self.calibration(Map, clustersMap, BMUs, name = True)[1]
   clusters = dataClusters.values() # ensemble of clusters
   keys = dataClusters.keys()
   ngrp = len(dataClusters) # number of groups
   #print 'ngrp=%s'%ngrp
   sys.stdout.write('Threshold: %.2f'%T)
   sys.stdout.flush()
   sys.stdout.write("\r")
   if (ngrp == 1 and ngrp == test):
    uniqueGroupTest = uniqueGroupTest + 1
   if uniqueGroupTest == 1:
    break
   indexes = []
   #print 'test=%s'%test
   if (ngrp == test - 1): # Adaptative step according to number of groups
    #print T
    #print 'ngrp == test - 1'
    for cluster in clusters:
     if cluster not in clusters_old:
      for e in cluster:
       for elementary_cluster in elementary_clusters:
        if e in elementary_cluster:
         indexes.append(elementary_clusters.index(elementary_cluster))
         #print elementary_cluster, elementary_clusters.index(elementary_cluster)
    indexes = numpy.unique(numpy.array(indexes))
    indexes = [int(el) for el in indexes] # to format int type and not numpy.int32 type
    #print indexes
    dist.update({tuple(indexes):T})
    clusters_old = clusters
    test = ngrp
   elif ngrp == test:
    #print 'ngrp == test'
    step = step_value
    test = ngrp
   else:
    #print 'ngrp == test - x with x > 1'
    T = T - step
    step = step / 10
  #print dist
  
  simplekeys_old = [1]
  while 1:
   keys = dist.keys()
   simplekeys = [key for key in keys if len(key) == 2]
   if simplekeys == simplekeys_old:
    break
   #print simplekeys
   for key in keys:
    keyv = list(key)
    for simplekey in simplekeys:
     c = 0
     for e in simplekey:
      #print e, key
      #print type(e), type(key)
      if e in key:
       c = c + 1
     if c==2 and len(key) > 2:
      simple = ()
      for e in simplekey:
       simple = simple + (keyv.pop(keyv.index(e)),)
      keyv.append(simple)
    newkey = tuple(keyv)
    value = dist.pop(key)
    dist.update({newkey:value})
    simplekeys_old = simplekeys
#  print dist
  
  keys = dist.keys()
  #print keys
  length = [len(key) for key in keys]
  if max(length) > 2:
   print 'Problem with tree building. More than two groups are found at the end of iterations'
  distItems = dist.items()
  #print distItems
  sortList = []
  while len(distItems) != 0:
   maxDist = 0
   for e in distItems:
    if e[1] > maxDist:
     maxDist = e[1]
     biggestTree = e
   sortList.append(biggestTree)
   distItems.remove(sortList[-1])
  #print sortList
  #print list(sortList[0][0])
  newick = str(sortList[0][0])
  self.sortList = sortList[0][0]
  #print newick
  for i in range(len(elementary_clusters)):
   #newick = newick.replace('%s' % i, '%s' % elementary_clusters[i])
   mols = str(tuple(elementary_clusters[i])).lstrip('(').rstrip(')').rstrip(',')
   #print mols
   newick = re.sub(r'(?<=\()%s(?=,)' % i, r'(%s)'%mols,newick)
   newick = re.sub(r'(?<=\b)%s(?=\))' % i, r'(%s)'%mols,newick)
  #print newick
  for subnewick in sortList:
   subnewick = list(subnewick)
   for i in range(len(elementary_clusters)):
    mols = str(tuple(elementary_clusters[i])).lstrip('(').rstrip(')').rstrip(',')
    #print mols
    subnewick[0] = re.sub(r'%s(?=,)' % i, r'(%s)'%mols,str(subnewick[0]))
    subnewick[0] = re.sub(r'%s(?=\))' % i, r'(%s)'%mols,str(subnewick[0]))
   newick = newick.replace('%s' % str(subnewick[0]), '%s:%s' % (subnewick[0],subnewick[1]) )
  self.elementary_clusters = elementary_clusters
  #print newick
  if jobIndex == '':
   newickFile = open('map.tree', 'w')
  else:
   newickFile = open('map%s.tree'%jobIndex, 'w')
  newickFile.write(newick)
  newickFile.close()
  def sortClusters(elementary_clusters, newick):
   def flatten(list):
    for e in list:
     if hasattr(e,'__getitem__'):
      for i in flatten(e):
       yield i
     else:
      yield e
   newick = newick[0]
   print '\nSOM hierarchical clustering: %s'%str(newick)
   l = list(flatten(newick))
   sortedClusters = []
   for e in l:
    sortedClusters.append(elementary_clusters[e])
   return sortedClusters
  return sortClusters(elementary_clusters, sortList[0])
    
#som = SOM()
#Map = som.loadMap('map_5x4.dat')
#Map = som.learn()
#som.tree(Map)
#som.mapPlot(Map)
#clusterMap = som.clusterDiscovery(Map,0.5)
#dataClusters = som.calibration(Map, clusterMap)
#dataLoosenessMat = som.dataLooseness(dataClusters, clusterMap)
#som.mapPlot(Map)
#som.mapPlot(dataLoosenessMat)
#clusterMap = som.clusterDiscovery(Map,0.3)
#print dataLoosenessMat
#som.mapPlot(dataLoosenessMat)
#dataDistanceMat = som.dataDistance(dataClusters)
#T_best = som.xiT(Map)
#clusterLoosenessMap = som.clusterLooseness(clusterMap, calibratedMap)
#som.mapPlot(clusterLoosenessMap)
#som.xiT(Map)
#clusterMap = som.clusterDiscovery(Map,0.3)
#print clusterLoosenessMap
#clusterDistanceMap = som.clusterDistance(clusterMap, Map)
#print som.xi(clusterLoosenessMap, clusterDistanceMap)
