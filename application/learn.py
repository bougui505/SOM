#!/usr/bin/env python
import matplotlib.pyplot
import IO
import numpy
import itertools
import scipy.spatial
import scipy.stats
import scipy.ndimage.measurements
import SOM
import parallelSOM
import glob
#from newProtocolModule import *
from SOMTools import *
import cPickle
import os
import ConfigParser
import sys

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

inputMatrixFileName = Config.get('learn', 'inputMatrixFileName')
mapFileName = Config.get('learn', 'mapFileName')
relearn = Config.getboolean('learn', 'relearn')
parallelLearning = Config.getboolean('learn', 'parallelLearning')
nSnapshots = Config.getint('learn', 'nSnapshots')
autoParam = Config.getboolean('learn', 'autoParam')


if glob.glob(inputMatrixFileName) == []:
 print 'No inputMatrix.dat file!'
else:
 if inputMatrixFileName.split('.')[1] == 'npy':
  inputMatrix = numpy.load(inputMatrixFileName)
 else:
  inMfile = open(inputMatrixFileName)
  inputMatrix = cPickle.load(inMfile)
  inMfile.close()

#Learning #############################################################################################################
if glob.glob(mapFileName) == []:
 som = SOM.SOM(inputMatrix, range(inputMatrix.shape[0]), metric='euclidean', autoParam = autoParam)
 if parallelLearning:
  parallelSOM.learn(som)
 else:
  som.learn(nSnapshots = nSnapshots)
 os.system('mv map_%sx%s.dat map1.dat'%(som.X,som.Y))
else:
 som = SOM.SOM(inputMatrix, range(inputMatrix.shape[0]), mapFileName=mapFileName, metric='euclidean', autoParam = autoParam)
 if relearn:
  if parallelLearning:
   parallelSOM.learn(som)
  else:
   som.learn(nSnapshots = nSnapshots)
  os.system('mv map_%sx%s.dat map1.dat'%(som.X,som.Y))
#######################################################################################################################

som = SOM.SOM(inputMatrix, range(inputMatrix.shape[0]), mapFileName=mapFileName, metric='euclidean', autoParam = False)
bmuCoordinates = []
bmuProb = []
sys.stdout.write('Computing density\n')
n = inputMatrix.shape[0]
c = 0
for l in inputMatrix:
 c = c + 1
 bmu, p = getBmuProb(som, l)
 bmuCoordinates.append(bmu)
 bmuProb.append(p)
 sys.stdout.write('%s/%s: %.4f'%(c,n,p))
 sys.stdout.write('\r')
 sys.stdout.flush()
sys.stdout.write('\ndone\n')
bmuCoordinates = numpy.array(bmuCoordinates)
bmuProb = numpy.array(bmuProb)
numpy.save('bmuCoordinates.npy', bmuCoordinates)
numpy.save('bmuProb.npy', bmuProb)

density = numpy.zeros((som.X,som.Y))
densityProb = numpy.zeros((som.X,som.Y))
iterbmuProb = itertools.chain(bmuProb)
for l in bmuCoordinates:
 i,j = l
 density[i,j] = density[i,j] + 1
 densityProb[i,j] += numpy.log(iterbmuProb.next())
densityProb = numpy.exp(densityProb / density)
numpy.save('density.npy', density)
numpy.save('densityProb.npy', densityProb)


##########################################################################
#uMatrix #############################################################
uMatrix = getUmatrix(som.Map)
numpy.save('uMatrix.npy', uMatrix)
#clusterMatrix, nClusters = scipy.ndimage.measurements.label(findMinRegion(uMatrix, scale = 0.75))
#plotMat(clusterMatrix, 'clusterMatrix.pdf', interpolation='nearest')
#for i in range(1,nClusters+1):
# indices = (allMins * numpy.atleast_2d((clusterMatrix == i).flatten()).T).any(axis=0)
# cluster = numpy.array(som.inputnames)[indices]
# outfile = open('cluster_%s.out'%i, 'w')
# [outfile.write('%s\n'%(e+1)) for e in cluster] # start from 1
# outfile.write('\n')
# outfile.close()
vmdMap(sliceMatrix(uMatrix), 'uMatrix.map')
plotMat(uMatrix, 'uMatrix.pdf', contour=False)
plotMat(density, 'density.pdf', interpolation='nearest')
plotMat(densityProb, 'densityProb.pdf', interpolation='nearest')
