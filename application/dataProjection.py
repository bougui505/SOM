#!/usr/bin/env python
from SOMTools import *
import numpy
import pickle
import ConfigParser
import sys
import itertools

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

infile = Config.get('dataProjection', 'dataFile')
outFileName = Config.get('dataProjection', 'outFileName')
bmuFileName = Config.get('dataProjection', 'bmuFileName')
smap = Config.get('dataProjection', 'map')
vmin = Config.get('dataProjection', 'vmin')
vmax = Config.get('dataProjection', 'vmax')
try:
 vmin = float(vmin)
except ValueError:
 vmin = None
try:
 vmax = float(vmax)
except ValueError:
 vmax = None
smap = numpy.load(smap)
X,Y,Z,cardinal = smap.shape
dataMap = numpy.zeros((X,Y,Z))
data = numpy.genfromtxt(infile)
idata = itertools.chain(data)
bmuCoordinates = numpy.load(bmuFileName)
if data.shape[0] == bmuCoordinates.shape[0]:
 density = numpy.zeros((X,Y,Z))
 for bmu in bmuCoordinates:
  i,j,k = bmu
  dataMap[i,j,k] += idata.next()
  density[i,j,k] += 1
 dataMap = dataMap / density
 pickle.dump(dataMap, open('%s.dat'%outFileName, 'w'))
 flatten_map = numpy.concatenate((smap.reshape(X*Y*Z,cardinal), numpy.atleast_2d(dataMap.reshape(X*Y*Z)).T), axis=1)
 numpy.savetxt('%s.txt'%outFileName, flatten_map)
else:
 print 'Shape mismatch between data (%s) and bmuCoordinates (%s)!'%(data.shape[0], bmuCoordinates.shape[0])
