#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
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
outPDFFileName = Config.get('dataProjection', 'outPDFFileName')
bmuFileName = Config.get('dataProjection', 'bmuFileName')
X = Config.getint('dataProjection', 'X')
Y = Config.getint('dataProjection', 'Y')
interpolation = Config.get('dataProjection', 'interpolation')
contour = Config.getboolean('dataProjection', 'contour')
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

dataMap = numpy.zeros((X,Y))
data = numpy.genfromtxt(infile)
idata = itertools.chain(data)
bmuCoordinates = numpy.load(bmuFileName)
if data.shape[0] == bmuCoordinates.shape[0]:
 density = numpy.zeros((X,Y))
 for bmu in bmuCoordinates:
  i,j = bmu
  dataMap[i,j] += idata.next()
  density[i,j] += 1
 dataMap = dataMap / density
 pickle.dump(dataMap, open('%s.dat'%outPDFFileName.split('.')[0], 'w'))
 plotMat(dataMap, outPDFFileName, interpolation = interpolation, contour = contour, vmin = vmin, vmax = vmax)
else:
 print 'Shape mismatch between data (%s) and bmuCoordinates (%s)!'%(data.shape[0], bmuCoordinates.shape[0])
