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
nSnapshots = Config.getint('learn', 'nSnapshots')
autoParam = Config.getboolean('learn', 'autoParam')
sort2ndPhase = Config.getboolean('learn', 'sort2ndPhase')
toricMap = Config.getboolean('learn', 'toricMap')
randomInit = Config.getboolean('learn', 'randomInit')
autoSizeMap = Config.getboolean('learn', 'autoSizeMap')


if glob.glob(inputMatrixFileName) == []:
 print 'No inputMatrix.dat file!'
else:
 if inputMatrixFileName.split('.')[1] == 'npy':
  inputMatrix = numpy.load(inputMatrixFileName)
 else:
  inMfile = open(inputMatrixFileName)
  inputMatrix = cPickle.load(inMfile)
  inMfile.close()

mapFileName = 'map_*.dat'

#Learning #############################################################################################################
if glob.glob(mapFileName) == []:
 som = SOM.SOM3D(inputMatrix, range(inputMatrix.shape[0]), metric='euclidean', autoParam = autoParam, sort2ndPhase=sort2ndPhase, toricMap=toricMap, randomInit=randomInit, autoSizeMap=autoSizeMap)
 som.learn(nSnapshots = nSnapshots)
else:
 mapFileName = glob.glob(mapFileName)[0]
 print "Map file: %s"%mapFileName
 som = SOM.SOM3D(inputMatrix, range(inputMatrix.shape[0]), mapFileName=mapFileName, metric='euclidean', autoParam = autoParam, sort2ndPhase=sort2ndPhase, toricMap=toricMap, randomInit=randomInit, autoSizeMap=autoSizeMap)
 if relearn:
  som.learn(nSnapshots = nSnapshots)
#######################################################################################################################

#som = SOM.SOM3D(inputMatrix, range(inputMatrix.shape[0]), mapFileName=mapFileName, metric='euclidean', autoParam = False)
bmuCoordinates = []
#bmuProb = []
sys.stdout.write('Computing density\n')
n = inputMatrix.shape[0]
c = 0
for l in inputMatrix:
 c = c + 1
# bmu, p = getBmuProb(som, l)
 bmu = getBMU(som,l)
 bmuCoordinates.append(bmu)
# bmuProb.append(p)
 sys.stdout.write('%s/%s'%(c,n))
 sys.stdout.write('\r')
 sys.stdout.flush()
sys.stdout.write('\ndone\n')
bmuCoordinates = numpy.array(bmuCoordinates)
#bmuProb = numpy.array(bmuProb)
numpy.save('bmuCoordinates.npy', bmuCoordinates)
#numpy.save('bmuProb.npy', bmuProb)

density = numpy.zeros((som.X,som.Y,som.Z))
densityProb = numpy.zeros((som.X,som.Y,som.Z))
#iterbmuProb = itertools.chain(bmuProb)
for l in bmuCoordinates:
 i,j,k = l
 density[i,j,k] = density[i,j,k] + 1
# densityProb[i,j] += numpy.log(iterbmuProb.next())
#densityProb = numpy.exp(densityProb / density)
numpy.save('density.npy', density)
#numpy.save('densityProb.npy', densityProb)


##########################################################################
#uMatrix #############################################################
uMatrix = getUmatrix(som.Map, toricMap=toricMap)
numpy.save('uMatrix.npy', uMatrix)
uMatrix_flatten = numpy.concatenate((som.Map.reshape((som.X*som.Y*som.Z,som.cardinal)),numpy.atleast_2d(uMatrix.flatten()).T), axis=1)
numpy.savetxt('uMatrix.txt', uMatrix_flatten)
