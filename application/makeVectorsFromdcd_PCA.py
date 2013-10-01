#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
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
from multiprocessing import Pool

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

nframe = Config.getint('makeVectors', 'nframes')
structFile = Config.get('makeVectors', 'structFile')
trajFile = Config.get('makeVectors', 'trajFile')
projection = Config.getboolean('makeVectors', 'projection')
nProcess = Config.getint('makeVectors', 'nProcess')

pool = Pool(processes=nProcess)

ref = True
descriptorsList = []
eigenVectorsList = []
eigenValuesList = []
meansList = []

struct = IO.Structure(structFile)

mask = numpy.ones((struct.atoms.shape[0]),dtype="bool")
traj = IO.Trajectory(trajFile, struct, selectionmask=mask, nframe=nframe)

trajIndex = 0
while trajIndex < nframe:
 distMats = []
 proc = 0
 while proc < nProcess and trajIndex < nframe:
  sys.stdout.write('%s/%s'%(trajIndex+1,nframe))
  sys.stdout.write('\r')
  sys.stdout.flush()
  traj_i = traj.array[trajIndex]
  shapeTraj = traj_i.reshape(traj.natom,3)
  dist = scipy.spatial.distance.pdist(shapeTraj)
  distMat = scipy.spatial.distance.squareform(dist)**2
  mean = numpy.mean(distMat,axis=1)
  meansList.append(mean)
  distMats.append(distMat)
  proc += 1
  trajIndex += 1
 eigenVectors_eigenValues = pool.map(PCA.princomp, distMats)
 eigenVectorsPool = [e[0] for e in eigenVectors_eigenValues]
 eigenValuesPool = [e[1] for e in eigenVectors_eigenValues]
 for eigenValues in eigenValuesPool:
  eigenValuesList.append(eigenValues)
 for eigenVectors in eigenVectorsPool:
  if ref:
   eigenVectors_ref = eigenVectors
   ref = False
  eigenVectors = eigenVectors*numpy.sign(numpy.dot(eigenVectors.T,eigenVectors_ref).diagonal())
  eigenVectorsList.append(eigenVectors.flatten())
  if projection:
   descriptor = numpy.dot(eigenVectors.T,distMat).flatten()
  else:
   descriptor = eigenVectors.T.flatten()
  descriptorsList.append(descriptor)
projections = numpy.asarray(descriptorsList)
numpy.save('projections.npy', projections)
eigenVectorsList = numpy.asarray(eigenVectorsList)
numpy.save('eigenVectorsList', eigenVectorsList)
eigenValuesList = numpy.asarray(eigenValuesList)
numpy.save('eigenValues', eigenValuesList)
meansList = numpy.asarray(meansList)
numpy.save('meansList', meansList)
reconstruction = numpy.concatenate((eigenVectorsList,projections,meansList),axis=1)
numpy.save('reconstruction',reconstruction)
