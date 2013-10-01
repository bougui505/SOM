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
import itertools
import pdbReader
import pdbWriter
import SOM
import somAnalysis
import math

class matrix:
 def __init__(self, matrix):
  self.matrix = matrix
  self.shape = numpy.shape(matrix)

 def reduceCoor(self, ceil=False):
  """
  for a square matrix
  """
  if self.shape[0] != self.shape[1]:
   print 'Warning: the matrix is not a square matrix'
   return self.matrix
  else:
   s = self.shape[0]/3
   hs = numpy.hsplit(self.matrix, s)
   vhs = []
   for e in hs:
    vhs.append(numpy.vsplit(e,s))
   means = []
   for m in vhs:
    for e in m:
     if ceil:
      means.append(int(math.ceil(e.trace()/3)))
     else:
      means.append(e.trace()/3)
   rm = numpy.zeros((s,s))
   ii = itertools.cycle(range(s))
   ij = itertools.cycle(range(s))
   for e in means:
    i = ii.next()
    if i == 0:
     j = ij.next()
    rm[i,j]=e
   return rm


 def reduceMean(self, factor=3, ceil=False):
  """
  for a square matrix
  """
  if self.shape[0] != self.shape[1]:
   print 'Warning: the matrix is not a square matrix'
   return self.matrix
  else:
   s = self.shape[0]/factor
   hs = numpy.hsplit(self.matrix, s)
   vhs = []
   for e in hs:
    vhs.append(numpy.vsplit(e,s))
   means = []
   for m in vhs:
    for e in m:
     if ceil:
      means.append(int(math.ceil(numpy.mean(e))))
     else:
      means.append(numpy.mean(e))
   rm = numpy.zeros((s,s))
   ii = itertools.cycle(range(s))
   ij = itertools.cycle(range(s))
   for e in means:
    i = ii.next()
    if i == 0:
     j = ij.next()
    rm[i,j]=e
   return rm

 def eigen(self):
  val,vec=numpy.linalg.eig(self.matrix)
  args=val.argsort()[::-1]
  eigenvalues=val[args]
  eigenvectors=vec[:,args]
  return eigenvalues, eigenvectors

 def projection(self, eigenvalue, eigenvector, pdbFileName, clusterVarComparison=None, outPdbFile='projection.pdb', append=False, modelNumber=0, continuousScale=False, ca = True, bb = False):
  pdbFile = open(pdbFileName)
  pdbR = pdbReader.PdbReader(pdbFile)
  if clusterVarComparison is None:
    p = abs(eigenvalue*eigenvector)
  else:
    cl1,cl2,var=clusterVarComparison
#    print var.shape,eigenvector.shape
    p = var*( (1.*(eigenvector == cl1)-1.*(eigenvector == cl2)))
  pdbFile = open(pdbFileName)
  pdbW = pdbWriter.PdbWriter(pdbFile)
  pdbW.addBFactor(p, outPdbFile, append=append, modelNumber=modelNumber, ca = ca, bb = bb)
  pymolScript = open(outPdbFile.replace('.pdb', '.pml'), 'w')
  max = numpy.max(eigenvector)
  if not continuousScale:
   pymolScript.write("""
load %s
hide everything
show cartoon
spectrum b, minimum=0, maximum=%s
   """%(outPdbFile, max))
  else:
   pymolScript.write("""
load %s
hide everything
show cartoon
spectrum b, blue_white_red, minimum=-1, maximum=1
alter all, b=str(abs(b))
cartoon putty
unset cartoon_smooth_loops
unset cartoon_flat_sheets
   """%(outPdbFile))
  pymolScript.close()

 def projectionRms(self, pdbFileName, varxyz, outPdbFileName='projectionClusterRms.pdb', ca = True, bb = False):
  f = open(outPdbFileName, 'w')
  f.close()
  diagonal=self.matrix.diagonal().astype('int') #clusterMatrix
#  bins=list(numpy.bincount(diagonal).argsort())[1:]
#  bins.reverse()
  bins = list(numpy.unique(diagonal))
  var=numpy.sqrt(varxyz.reduceCoor().diagonal())
  var/=max(var)
  c = itertools.count()
  for i in range(len(bins)):
   for j in range(i+1, len(bins)):
    self.projection(1, diagonal, pdbFileName,clusterVarComparison=(bins[i],bins[j],var), outPdbFile='%s_%s_%s.pdb'%(outPdbFileName.replace('.pdb', ''), i, j), continuousScale=True, ca = ca, bb = bb)

 def symmetryZeros(self):
  """
  Replace one part of a symmetric matrice with zeros
  """
  rm = numpy.zeros((self.shape[0],self.shape[1]))
  for i in range(self.shape[0]):
   for j in range(self.shape[1]):
    if j < i:
     rm[i,j] = self.matrix[i,j]
  return rm

 def fourierTransform(self):
  from numpy.fft import fft2
  return fft2(self.matrix)

 def matrix2vectors(self, matrix, groupedXYZ=False):
  vectors = []
  vectorNames = []
  if groupedXYZ:
   x,y=matrix.shape
   vectors=[ numpy.hsplit(i,x/3) for i in numpy.vsplit(matrix,y/3) ]
   vectorNames=range(x)
  else:
   for i in range(matrix.shape[0]):
    vectors.append(list(matrix[i:i+1][0]))
    vectorNames.append(i)
  print 'Shape of the input data: %s'%(str(numpy.array(vectors).shape))
  return vectorNames, vectors

 def distFunc(self, vector, map):
  """
  vector = [array([[...],[...],...]), ...]
  """
#  print numpy.shape(vector)
#  print numpy.shape(map)
  def interleave(listOfArrays):
   v = numpy.array(listOfArrays)
#   return sum([ list(e[i]) for e in v for i in range(v.shape[1]) ],[])
   return list(v.flat)
  def euclideanDistance(vector, Map):
   X, Y, cardinal = Map.shape
   V=numpy.ones((X,Y,cardinal))*vector
   return numpy.sum( (V - Map) * ( (V - Map)* (numpy.ones([X,Y,cardinal])*numpy.ones([1,cardinal])) ) , axis=2 )**0.5
  def euclideanDistanceVectors(vector1, vector2):
   a = numpy.array(vector1)
   b = numpy.array(vector2)
   return numpy.linalg.norm(a-b)
#   iV2 = itertools.chain(vector2)
#   d = 0
#   for e in vector1:
#    d = d + (e - iV2.next())**2
#   return d
  inputVector = interleave(vector)
  somShape = map.shape
  if len(somShape) == 5:
   newMap = numpy.zeros((somShape[0], somShape[1], somShape[2]*somShape[3]*somShape[4]))
  elif len(somShape) == 3:
   newMap = []
  if len(somShape) == 5:
   for i in range(somShape[0]):
    for j in range(somShape[1]):
      v = interleave(map[i,j])
      newMap[i,j] = v
  elif len(somShape) == 3:
   v = interleave(map)
   newMap.extend(v)
  if len(somShape) == 5:
   distMat = euclideanDistance(inputVector, newMap)
   return distMat
  elif len(somShape) == 3:
   d = euclideanDistanceVectors(inputVector, newMap)
   return d

 def somClustering(self,groupedXYZ=False,mapFileName=None,confFileName="SOM.conf",threshold=None):
  matrix = self.matrix
  vectorNames, vectors = self.matrix2vectors(matrix,groupedXYZ=groupedXYZ)
  if groupedXYZ:
   som=SOM.SOM(vectors, vectorNames, distFunc=self.distFunc,mapFileName=mapFileName,confname=confFileName)
  else:
   som=SOM.SOM(vectors, vectorNames,mapFileName=mapFileName,confname=confFileName)
  if mapFileName == None:
   map = som.learn()
  else:
   map = som.M
  somA = somAnalysis.analysis(map=map)
  if threshold == None:
   threshold = somA.xiT()
  clustersMap = som.clusterDiscovery(map,threshold)
#  print clustersMap
  distMatrix = som.clusterDistance(clustersMap, map)
  order = self.upgma(distMatrix, reverse = True)
#  print order
  nameClusters =  som.calibration(map, clustersMap, name=True)[1]
  clusters = nameClusters.values()
#  print clusters
  print 'Number of clusters found: %s'%len(clusters)
#  clusters = som.calibration(map, clustersMap, name = True)
#  clusterValues = clusters[0]
#  clusterNames = clusters[1]
#  clusters = som.tree(map)
#  print clusters
#  clusters = [clusters[e] for e in order]
  sortedClusters = []
  for e in order:
   try:
    sortedClusters.append(clusters[e])
   except IndexError:
    pass
  return sortedClusters

 def upgma(self, distanceMatrix, reverse = False):
  """
  perform upgma algorithm for the given distance matrix
  """
  shape = numpy.shape(distanceMatrix)
  ##################################################################
  def locate():
   if reverse:
    m = numpy.where(distanceMatrix == distanceMatrix.max())
   else:
    m = numpy.where(distanceMatrix.min() == distanceMatrix.min())
   return list(numpy.unique(numpy.array(m)))
  ##############################################################
  def newDistMatrix(clusterIndexes):
   shape = numpy.shape(distanceMatrix)
   newMatrix = distanceMatrix
   for i in range(shape[0]):
    for j in range(shape[1]):
     if i in clusterIndexes and j in clusterIndexes:
      newMatrix[i,j] = 0
     elif (i in clusterIndexes):
      newMatrix[i,j] = ( distanceMatrix[clusterIndexes[0], j] + distanceMatrix[clusterIndexes[1], j] )/2
     elif (j in clusterIndexes):
      newMatrix[i,j] = ( distanceMatrix[i, clusterIndexes[0]] + distanceMatrix[i, clusterIndexes[1]] )/2
     else:
      newMatrix[i,j] = distanceMatrix[i,j]
   return newMatrix
  ##################################################################
  clusterIds = []
  while len(clusterIds) < shape[0]:
   clusterId = locate()
   distanceMatrix = newDistMatrix(clusterId)
   for e in clusterId:
    if e not in clusterIds:
     clusterIds.append(e)
  return clusterIds

 def plotSomClustering(self, clusters):
  n = max(sum(clusters, [])) + 1
  rm = numpy.zeros( ( n,n ) )
  count = itertools.count(1)
  for cluster in clusters:
   c = count.next()
   for e1 in cluster:
    for e2 in cluster:
     rm[e1,e2] = c
  return rm

 def plotSomBmus(self,groupedXYZ=False,mapFileName=None, confFileName='SOM.conf'):
  matrix = self.matrix
  vectorNames, vectors = self.matrix2vectors(matrix, groupedXYZ=groupedXYZ)
  if groupedXYZ:
   som = SOM.SOM(vectors, vectorNames, distFunc=self.distFunc,mapFileName=mapFileName, confname=confFileName)
  else:
   som = SOM.SOM(vectors, vectorNames,mapFileName=mapFileName, confname=confFileName)
  if mapFileName == None:
   map = som.loadMap('map_%sx%s.dat'%(som.X, som.Y))
  else:
   map = som.M
  cardinal = som.cardinal
  k = itertools.count()
  rm = []
  for v in vectors:
   ij = som.findBMU(k.next(), map)
   bmu = map[ij[0],ij[1]]
   if groupedXYZ:
    flattenBmu = list(bmu.flat)
    rm.extend([flattenBmu[0:3*cardinal], flattenBmu[3*cardinal:2*3*cardinal], flattenBmu[2*3*cardinal:3*3*cardinal]])
   else:
    rm.append(bmu)
  return numpy.array(rm)

 def plotSortedClusters(self, clusterMatrix, groupedXYZ=False):
  permut = numpy.diag(clusterMatrix).argsort()
  if not groupedXYZ:
   return self.matrix[permut,][:,permut]
  if groupedXYZ:
   permutX = list(3*numpy.array(permut))
   permutY = [e+1 for e in permutX]
   permutZ = [e+2 for e in permutX]
   permutXYZ = list(numpy.array(zip(permutX,permutY,permutZ)).flatten())
   return self.matrix[permutXYZ,][:,permutXYZ]
