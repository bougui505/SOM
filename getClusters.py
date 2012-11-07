#!/usr/bin/env python
import numpy
import scipy.ndimage
import sys
import SOMTools as SOM3DTools
#import SOM3DTools
import splitDockMap


class Clust3D(object):
 @staticmethod
 def getClusters(Map, uMatrix, threshold, toricMap=True):
  clusterMat = scipy.ndimage.label(uMatrix<threshold)[0]
  if toricMap: clusterMat = SOM3DTools.continuousMap(clusterMat)
  cIds = numpy.unique(clusterMat)[1:]
  uMeans = []
  uMins = []
  uMedians = []
  for i in cIds:
   sel = (clusterMat == i)
   uMean = uMatrix[sel].mean()
   uMin = uMatrix[sel].min()
   uMedian = numpy.median(uMatrix[sel])
   uMeans.append(uMean)
   uMins.append(uMin)
   uMedians.append(uMedian)
  sortedCids = cIds[numpy.argsort(uMeans)]
  sortedClusterMat = numpy.zeros_like(clusterMat)
# mapCom_global, mapNorm1_global, mapNorm2_global, mapNorm3_global, mapVectors1_global,mapVectors2_global,mapVectors3_global = splitDockMap.splitMap(map,'global')
  c = 0
  for i in sortedCids:
   c+=1
   sel = (clusterMat == i)
#  splitDockMap.get3Dvectors(mapCom_global[sel], mapVectors1_global[sel], mapNorm1_global[sel], 'global_c%s'%c)
   sortedClusterMat[sel] = c
   cId = numpy.where(cIds==i)[0][0]
   print 'cId: %s, mean:%.2f, median: %.2f, min:%.2f'%(c, uMeans[cId], uMedians[cId], uMins[cId])
  numpy.save('clusterMat.npy', sortedClusterMat)
  SOM3DTools.vmdMap(sortedClusterMat, 'clusterMat.map')
  return sortedClusterMat

 @staticmethod
 def toricMinimizeToCluster(uMatrix,clusterMat,index):
  x,y,z=index
  shape=uMatrix.shape
  directNeighbors=([x,(x+1) % shape[0],(x-1) % shape[0],x,x,x,x],[y,y,y,(y+1) % shape[1],(y-1) % shape[1],y,y],[z,z,z,z,z,(z+1) % shape[1],(z-1) % shape[2]])
  zipDirectNeighbors=zip(directNeighbors[0],directNeighbors[1],directNeighbors[2])
  nextstep=numpy.argmin(uMatrix[directNeighbors])
  nextstepidx=zipDirectNeighbors[nextstep]
  cl=clusterMat[nextstepidx]
  if cl > 0 or nextstep==0:
   return cl
  else:
   return Clust3D.minimizeToCluster(uMatrix,clusterMat,nextstepidx)

 @staticmethod
 def minimizeToCluster(uMatrix,clusterMat,index):
  x,y,z=index
  shape=uMatrix.shape
  X,Y,Z=shape
  rawNeighbors=([x,(x+1) % shape[0],(x-1) % shape[0],x,x,x,x],[y,y,y,(y+1) % shape[1],(y-1) % shape[1],y,y],[z,z,z,z,z,(z+1) % shape[1],(z-1) % shape[2]])
  zipRawNeighbors=zip(rawNeighbors[0],rawNeighbors[1],rawNeighbors[2])
  zipDirectNeighbors=[]
  xd=[]
  yd=[]
  zd=[]
  for i,j,k in zipRawNeighbors:
   if 0 <= i < X and 0 <= j < Y and 0 <= k < Z:
    zipDirectNeighbors.append((i,j,k))
    xd.append(i)
    yd.append(j)
    zd.append(k)
  directNeighbors=(xd,yd,zd)
  nextstep=numpy.argmin(uMatrix[directNeighbors])
  nextstepidx=zipDirectNeighbors[nextstep]
  cl=clusterMat[nextstepidx]
  if cl > 0 or nextstep==0:
   return cl
  else:
   return Clust3D.minimizeToCluster(uMatrix,clusterMat,nextstepidx)


 @staticmethod
 def getAllClustersMini(map, uMatrix, threshold=0.5, toricMap=True):
  clusterMat = scipy.ndimage.label(uMatrix<threshold)[0]
  if toricMap: clusterMat = SOM3DTools.continuousMap(clusterMat)
  cIds = numpy.unique(clusterMat)[1:]
  # right there we have enough to quit
  # now in this method, we don't want any point in cluster 0.
  for index, clust in numpy.ndenumerate(clusterMat): # surely not the most efficient method
   if clust == 0:
    if toricMap: clusterMat[index]=Clust3D.toricMinimizeToCluster(uMatrix,clusterMat,index)
    else: clusterMat[index]=Clust3D.minimizeToCluster(uMatrix,clusterMat,index)
  uMeans = []
  uMins = []
  uMedians = []
  for i in cIds:
   sel = (clusterMat == i)
   uMean = uMatrix[sel].mean()
   uMin = uMatrix[sel].min()
   uMedian = numpy.median(uMatrix[sel])
   uMeans.append(uMean)
   uMins.append(uMin)
   uMedians.append(uMedian)
  sortedCids = cIds[numpy.argsort(uMeans)]
  sortedClusterMat = numpy.zeros_like(clusterMat)
# mapCom_global, mapNorm1_global, mapNorm2_global, mapNorm3_global, mapVectors1_global,mapVectors2_global,mapVectors3_global = splitDockMap.splitMap(map,'global')
  c = 0
  for i in sortedCids:
   c+=1
   sel = (clusterMat == i)
#  splitDockMap.get3Dvectors(mapCom_global[sel], mapVectors1_global[sel], mapNorm1_global[sel], 'global_c%s'%c)
   sortedClusterMat[sel] = c
   cId = numpy.where(cIds==i)[0][0]
   print 'cId: %s, mean:%.2f, median: %.2f, min:%.2f'%(c, uMeans[cId], uMedians[cId], uMins[cId])
  numpy.save('clusterMat.npy', sortedClusterMat)
  SOM3DTools.vmdMap(sortedClusterMat, 'clusterMat.map')
  return sortedClusterMat

 @staticmethod
 def getAllClustersFind(Map, uMatrix, threshold=0.5, toricMap=True):
  clusterMat = scipy.ndimage.label(uMatrix<threshold)[0]
  if toricMap: clusterMat = SOM3DTools.continuousMap(clusterMat)
  cIds = numpy.unique(clusterMat)[1:]
  # now in this method, we don't want any point in cluster 0.
  # here we used findBMU on a "custom map" to find BMU on the thresholded clusters
  maxval=numpy.max(Map[clusterMat == 0],axis=0)
  nMap=numpy.ones(Map.shape)*maxval[numpy.newaxis,numpy.newaxis,numpy.newaxis,:]*1000
  nMap[clusterMat!=0]=Map[clusterMat!=0]
  # directly ported from SOM.py
  def findBMU(Map,vector):
   X,Y,Z,cardinal=Map.shape
   return numpy.unravel_index(scipy.spatial.distance.cdist(numpy.reshape(vector, (1,cardinal)), numpy.reshape(Map, (X*Y*Z,cardinal)), "euclidean").argmin(), (X,Y,Z))
  for index, clust in numpy.ndenumerate(clusterMat): # surely not the most efficient method
   if clust == 0:
    clusterMat[index]=clusterMat[findBMU(nMap,Map[index])]
  uMeans = []
  uMins = []
  uMedians = []
  for i in cIds:
   sel = (clusterMat == i)
   uMean = uMatrix[sel].mean()
   uMin = uMatrix[sel].min()
   uMedian = numpy.median(uMatrix[sel])
   uMeans.append(uMean)
   uMins.append(uMin)
   uMedians.append(uMedian)
  sortedCids = cIds[numpy.argsort(uMeans)]
  sortedClusterMat = numpy.zeros_like(clusterMat)
# mapCom_global, mapNorm1_global, mapNorm2_global, mapNorm3_global, mapVectors1_global,mapVectors2_global,mapVectors3_global = splitDockMap.splitMap(map,'global')
  c = 0
  for i in sortedCids:
   c+=1
   sel = (clusterMat == i)
#  splitDockMap.get3Dvectors(mapCom_global[sel], mapVectors1_global[sel], mapNorm1_global[sel], 'global_c%s'%c)
   sortedClusterMat[sel] = c
   cId = numpy.where(cIds==i)[0][0]
   print 'cId: %s, mean:%.2f, median: %.2f, min:%.2f'%(c, uMeans[cId], uMedians[cId], uMins[cId])
  numpy.save('clusterMat.npy', sortedClusterMat)
  SOM3DTools.vmdMap(sortedClusterMat, 'clusterMat.map')
  return sortedClusterMat


class Clust(object):
 @staticmethod
 def getClusters(map, uMatrix, relative_threshold):
  uMax = uMatrix.max()
  threshold = relative_threshold*uMax
  clusterMat = SOMTools.continuousMap(scipy.ndimage.label(uMatrix<threshold)[0])
  cIds = numpy.unique(clusterMat)[1:]
  uMeans = []
  uMins = []
  uMedians = []
  for i in cIds:
   sel = (clusterMat == i)
   uMean = uMatrix[sel].mean()
   uMin = uMatrix[sel].min()
   uMedian = numpy.median(uMatrix[sel])
   uMeans.append(uMean)
   uMins.append(uMin)
   uMedians.append(uMedian)
  sortedCids = cIds[numpy.argsort(uMeans)]
  sortedClusterMat = numpy.zeros_like(clusterMat)
# mapCom_global, mapNorm1_global, mapNorm2_global, mapNorm3_global, mapVectors1_global,mapVectors2_global,mapVectors3_global = splitDockMap.splitMap(map,'global')
  c = 0
  for i in sortedCids:
   c+=1
   sel = (clusterMat == i)
#  splitDockMap.get3Dvectors(mapCom_global[sel], mapVectors1_global[sel], mapNorm1_global[sel], 'global_c%s'%c)
   sortedClusterMat[sel] = c
   cId = numpy.where(cIds==i)[0][0]
   print 'cId: %s, mean:%.2f, median: %.2f, min:%.2f'%(c, uMeans[cId], uMedians[cId], uMins[cId])
  numpy.save('clusterMat.npy', sortedClusterMat)
  SOMTools.plotMat(sortedClusterMat, 'clusterMat.pdf', interpolation='nearest')
  return sortedClusterMat

 @staticmethod
 def minimizeToCluster(uMatrix,clusterMat,index):
  x,y=index
  shape=uMatrix.shape
  directNeighbors=([x,(x+1) % shape[0],(x-1) % shape[0],x,x],[y,y,y,(y+1) % shape[1],(y-1) % shape[1]])
  zipDirectNeighbors=zip(directNeighbors[0],directNeighbors[1])
  nextstep=numpy.argmin(uMatrix[directNeighbors])
  nextstepidx=zipDirectNeighbors[nextstep]
  cl=clusterMat[nextstepidx]
  if cl > 0 or nextstep==0:
   return cl
  else:
   return minimizeToCluster(uMatrix,clusterMat,nextstepidx)


 @staticmethod
 def getAllClusters(map, uMatrix, relative_threshold):
  uMax = uMatrix.max()
  threshold = relative_threshold*uMax
  clusterMat = SOMTools.continuousMap(scipy.ndimage.label(uMatrix<threshold)[0])
  cIds = numpy.unique(clusterMat)[1:]
  # right there we have enough to quit
  # now in this method, we don't want any point in cluster 0.
  for index, clust in numpy.ndenumerate(clusterMat): # surely not the most efficient method
   if clust == 0:
    clusterMat[index]=minimizeToCluster(uMatrix,clusterMat,index)
  uMeans = []
  uMins = []
  uMedians = []
  for i in cIds:
   sel = (clusterMat == i)
   uMean = uMatrix[sel].mean()
   uMin = uMatrix[sel].min()
   uMedian = numpy.median(uMatrix[sel])
   uMeans.append(uMean)
   uMins.append(uMin)
   uMedians.append(uMedian)
  sortedCids = cIds[numpy.argsort(uMeans)]
  sortedClusterMat = numpy.zeros_like(clusterMat)
# mapCom_global, mapNorm1_global, mapNorm2_global, mapNorm3_global, mapVectors1_global,mapVectors2_global,mapVectors3_global = splitDockMap.splitMap(map,'global')
  c = 0
  for i in sortedCids:
   c+=1
   sel = (clusterMat == i)
#  splitDockMap.get3Dvectors(mapCom_global[sel], mapVectors1_global[sel], mapNorm1_global[sel], 'global_c%s'%c)
   sortedClusterMat[sel] = c
   cId = numpy.where(cIds==i)[0][0]
   print 'cId: %s, mean:%.2f, median: %.2f, min:%.2f'%(c, uMeans[cId], uMedians[cId], uMins[cId])
  numpy.save('clusterMat.npy', sortedClusterMat)
  SOMTools.plotMat(sortedClusterMat, 'clusterMat.pdf', interpolation='nearest')
  return sortedClusterMat

def main():
 mapFileName = sys.argv[1]
 relative_threshold = float(sys.argv[2])
 map = numpy.load(mapFileName)
 uMatrix = numpy.load('uMatrix.npy')
 getClusters(map, uMatrix, relative_threshold)

if __name__ == '__main__':
 main()
