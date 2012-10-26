#!/usr/bin/env python
import matplotlib.pyplot
import IO
import numpy
import itertools
import scipy.spatial
import scipy.stats
import scipy.ndimage.measurements
import SOM
import glob
#from newProtocolModule import *
from SOMTools import *
import cPickle
import os

relearn = False

if glob.glob('inputMatrix.dat') == []:
 struct = IO.Structure('struct.pdb')
 fd=open('resList')
 reslist=[ line[:-1].split(' ') for line in fd ]
 reslist=[ (int(x),y) for x,y in reslist ]

# dico={}
# mask=numpy.zeros((struct.atoms.shape[0]),dtype="bool")
# for x,y in reslist:
#  if y not in dico:
#   dico[y]=struct.getSelectionIndices([y],'segid')
#  mask=numpy.logical_or(mask,numpy.logical_and(dico[y],struct.getSelectionIndices([x],'resid')))
 mask = numpy.ones((struct.atoms.shape[0]),dtype="bool")
 traj = IO.Trajectory('traj.dcd', struct, selectionmask=mask, nframe=11731)

 restraints = readRestraints()
 dists = []
 dotProducts = []
# i = itertools.count()
 shapeTraj = traj.array.reshape(traj.nframe,traj.natom,3)
 for r1, r2 in restraints:
  try:
   atom1 =(mask.nonzero()[0]==numpy.logical_and(traj.struct.getSelectionIndices([r1[0]],"resid"),traj.struct.getSelectionIndices([r1[1]],"segid")).nonzero()[0][0]).nonzero()[0][0]
   atom2 =(mask.nonzero()[0]==numpy.logical_and(traj.struct.getSelectionIndices([r2[0]],"resid"),traj.struct.getSelectionIndices([r2[1]],"segid")).nonzero()[0][0]).nonzero()[0][0]
   trajA1 = shapeTraj[:,atom1]
   trajA1m = shapeTraj[:,atom1-1] #for Calpha i-1
   trajA1p = shapeTraj[:,atom1+1] #for Calpha i+1
   trajA2 = shapeTraj[:,atom2]
   trajA2m = shapeTraj[:,atom2-1] #for Calpha i-1
   trajA2p = shapeTraj[:,atom2+1] #for Calpha i+1
   v_A1_1 = trajA1p - trajA1
   v_A1_2 = trajA1m - trajA1
   crossA1 = numpy.cross(v_A1_1, v_A1_2)
   v_A2_1 = trajA2p - trajA2
   v_A2_2 = trajA2m - trajA2
   crossA2 = numpy.cross(v_A2_1, v_A2_2)
   dotA1A2 = numpy.dot(crossA1/numpy.linalg.norm(crossA1),crossA2.T/numpy.linalg.norm(crossA2)).diagonal()
   distA1A2 = numpy.sqrt(((trajA1 - trajA2)**2).sum(axis=1))
   dists.append(distA1A2)
   dotProducts.append(dotA1A2)
  except IndexError:
   pass
 inputMatrix = numpy.dstack((numpy.asarray(dists).T, numpy.asarray(dotProducts).T))
 x, y, z = inputMatrix.shape
 inputMatrix = inputMatrix.reshape(x,y*z)

#remove systematic zeros
# mask = 1-(inputMatrix == 0).all(axis=0)
# inputMatrix = inputMatrix.compress(mask, axis=1)

 inMfile = open('inputMatrix.dat', 'w')
 cPickle.dump(inputMatrix, inMfile)
 inMfile.close()
else:
 inMfile = open('inputMatrix.dat')
 inputMatrix = cPickle.load(inMfile)
 inMfile.close()

#Learning #############################################################################################################
mapFileName = 'map1.dat'
if glob.glob(mapFileName) == []:
 som = SOM.SOM(inputMatrix, range(inputMatrix.shape[0]), metric='euclidean', autoParam = False)
 som.learn()
 os.system('mv map_%sx%s.dat map1.dat'%(som.X,som.Y))
else:
 som = SOM.SOM(inputMatrix, range(inputMatrix.shape[0]), mapFileName=mapFileName, metric='euclidean', autoParam = False)
 if relearn:
  som.learn()
  os.system('mv map_%sx%s.dat map1.dat'%(som.X,som.Y))
#######################################################################################################################

#Plot Maps ###############################################################
allMaps = allKohonenMap2D(som.Map, inputMatrix, metric='euclidean')
allMasks = findMinRegionAll(allMaps)
allMins = findMinAll(allMaps)
bmuDensity = numpy.reshape(allMins.sum(axis=1), (som.X,som.Y))
plotMat(bmuDensity, 'density.pdf', contour=False, interpolation='nearest')
density = numpy.reshape(allMasks.sum(axis=1), (som.X,som.Y))
plotMat(density, 'density2.pdf', contour=False)
#plot potential
pMap = restraintsPotential(som.Map[:,:,0:-1:2], 10, 28, 36)
stds = pMap.std(axis=0).std(axis=0)
varCoef = numpy.nan_to_num(scipy.stats.variation(scipy.stats.variation(pMap, axis=0), axis=0))
averagepMap = numpy.average(pMap, axis=2, weights=varCoef)
sumpMap = pMap.sum(axis=2)
plotMat(averagepMap, 'averageRestraintPotentialMap.pdf', contour=True)
plotMat(sumpMap, 'restraintPotentialMap.pdf', contour=True)
logHmap = numpy.log((som.Map[:,:,0:-1:2]/15)**2).sum(axis=2) # target distance = 15 A
plotMat(logHmap, 'logHmap.pdf', contour=True)
#Number of violated restraints
violationMap = (som.Map[:,:,0:-1:2] > 36).sum(axis=2)
plotMat(violationMap, 'violationMap.pdf', interpolation='nearest')


#EM map correlation
correlations = numpy.atleast_2d(numpy.genfromtxt('correlationEM.dat')[:,1])
meanCorrelationMatrix = getEMmapCorrelationMatrix(correlations, allMins, som)
plotMat(meanCorrelationMatrix, 'meanCorrelationMatrix.pdf', interpolation='nearest')
meanCorrelationRegions = getEMmapCorrelationMatrix(correlations, allMasks, som)
plotMat(meanCorrelationRegions, 'meanCorrelationRegions.pdf', contour=True)
#outside map correlation
outside = numpy.atleast_2d(numpy.genfromtxt('outsideEM.dat')[:,1])
meanOutsideMatrix = getEMmapCorrelationMatrix(outside, allMins, som)
plotMat(meanOutsideMatrix, 'meanOutsideMatrix.pdf', interpolation='nearest')
meanOutsideRegions = getEMmapCorrelationMatrix(outside, allMasks, som)
plotMat(meanOutsideRegions, 'meanOutsideRegions.pdf', contour=True)

##########################################################################
#uMatrix #############################################################
uMatrix = getUmatrix(som)
plotMat(uMatrix, 'uMatrix.pdf', contour=False)
clusterMatrix, nClusters = scipy.ndimage.measurements.label(findMinRegion(uMatrix, scale = 0.75))
plotMat(clusterMatrix, 'clusterMatrix.pdf', interpolation='nearest')
for i in range(1,nClusters+1):
 indices = (allMins * numpy.atleast_2d((clusterMatrix == i).flatten()).T).any(axis=0)
 cluster = numpy.array(som.inputnames)[indices]
 outfile = open('cluster_%s.out'%i, 'w')
 [outfile.write('%s\n'%(e+1)) for e in cluster] # start from 1
 outfile.write('\n')
 outfile.close()
vmdMap(sliceMatrix(uMatrix), 'uMatrix.map')
