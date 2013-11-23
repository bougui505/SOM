#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
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
import ConfigParser
import sys
import PCA

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

nframe = Config.getint('makeVectors', 'nframes')
getAngles = Config.getboolean('makeVectors', 'getAngles')
structFile = Config.get('makeVectors', 'structFile')
trajFile = Config.get('makeVectors', 'trajFile')
squareDistMat = Config.getboolean('makeVectors', 'squareDistMat')
pca = Config.getboolean('makeVectors', 'pca')
projection = Config.getboolean('makeVectors', 'projection')


if glob.glob('inputMatrix.dat') == []:
 struct = IO.Structure(structFile)
# fd=open('resList')
# reslist=[ line[:-1].split(' ') for line in fd ]
# reslist=[ (int(x),y) for x,y in reslist ]

 mask = numpy.ones((struct.atoms.shape[0]),dtype="bool")
 traj = IO.Trajectory(trajFile, struct, selectionmask=mask, nframe=nframe)

 restraints = readRestraints()
 dists = []
 if getAngles:
  dotProducts = []
 shapeTraj = traj.array.reshape(traj.nframe,traj.natom,3)
 n = len(restraints)
 counter = itertools.count()
 sys.stdout.write('computing distances\n')
 for r1, r2 in restraints:
  sys.stdout.write('%s/%s'%(counter.next(),n))
  sys.stdout.write('\r')
  sys.stdout.flush()
  try:
   atom1 =(mask.nonzero()[0]==numpy.logical_and(traj.struct.getSelectionIndices([r1[0]],"resid"),traj.struct.getSelectionIndices([r1[1]],"segid")).nonzero()[0][0]).nonzero()[0][0]
   atom2 =(mask.nonzero()[0]==numpy.logical_and(traj.struct.getSelectionIndices([r2[0]],"resid"),traj.struct.getSelectionIndices([r2[1]],"segid")).nonzero()[0][0]).nonzero()[0][0]
   trajA1 = shapeTraj[:,atom1]
   trajA2 = shapeTraj[:,atom2]
   if squareDistMat:
    distA1A2 = ((trajA1 - trajA2)**2).sum(axis=1)
   else:
    distA1A2 = numpy.sqrt(((trajA1 - trajA2)**2).sum(axis=1))
   dists.append(distA1A2)
   if getAngles:
    trajA1m = shapeTraj[:,atom1-1] #for Calpha i-1
    trajA1p = shapeTraj[:,atom1+1] #for Calpha i+1
    trajA2m = shapeTraj[:,atom2-1] #for Calpha i-1
    trajA2p = shapeTraj[:,atom2+1] #for Calpha i+1
    v_A1_1 = trajA1p - trajA1
    v_A1_2 = trajA1m - trajA1
    crossA1 = numpy.cross(v_A1_1, v_A1_2)
    v_A2_1 = trajA2p - trajA2
    v_A2_2 = trajA2m - trajA2
    crossA2 = numpy.cross(v_A2_1, v_A2_2)
    dotA1A2 = numpy.dot(crossA1/numpy.linalg.norm(crossA1),crossA2.T/numpy.linalg.norm(crossA2)).diagonal()
    dotProducts.append(dotA1A2)
  except IndexError:
   pass
 if getAngles:
  inputMatrix = numpy.dstack((numpy.asarray(dists).T, numpy.asarray(dotProducts).T))
  x, y, z = inputMatrix.shape
  inputMatrix = inputMatrix.reshape(x,y*z)
 else:
  inputMatrix = numpy.asarray(dists).T
  if pca:
   eigenVectorsList = []
   ref = True
   counter = itertools.count()
   n = inputMatrix.shape[0]
   sys.stdout.write('PCA\n')
   for dist in inputMatrix:
    sys.stdout.write('%s/%s'%(counter.next(),n))
    sys.stdout.write('\r')
    sys.stdout.flush()
    distMat = scipy.spatial.distance.squareform(dist)
    eigenVectors,eigenValues = PCA.princomp(distMat,numpc=4)
    if ref:
     eigenVectors_ref = eigenVectors
     ref = False
    eigenVectors = eigenVectors*numpy.sign(numpy.dot(eigenVectors.T,eigenVectors_ref).diagonal())
    if projection:
     descriptor = numpy.dot(eigenVectors.T,distMat).flatten()
    else:
     descriptor = eigenVectors.T.flatten()
    eigenVectorsList.append(descriptor)
   inputMatrix = numpy.asarray(eigenVectorsList)
 numpy.save('inputMatrix.npy', inputMatrix)

# inMfile = open('inputMatrix.dat', 'w')
# cPickle.dump(inputMatrix, inMfile)
# inMfile.close()
else:
 inMfile = open('inputMatrix.dat')
 inputMatrix = cPickle.load(inMfile)
 inMfile.close()

