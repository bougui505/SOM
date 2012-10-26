#!/usr/bin/env python
import pickle
import numpy
import scipy.spatial
import sys
import ConfigParser

if __name__ == '__main__':
 configFileName = sys.argv[1]
 Config = ConfigParser.ConfigParser()
 Config.read(configFileName)

 Map = pickle.load(open(Config.get('snapshotIdMap', 'map')))
 inputMatrix = numpy.load(Config.get('snapshotIdMap', 'inputMatrix'))
 bmuCoordinates = numpy.load(Config.get('snapshotIdMap', 'bmuCoordinates'))

 X,Y,Z = Map.shape
 snapIds = numpy.ma.masked_all((X,Y,2))
 n = X*Y
 k = 0
 nframe = bmuCoordinates.shape[0]
 for bmu in bmuCoordinates:
  sys.stdout.write('%s/%s'%(k+1,nframe))
  sys.stdout.write('\r')
  sys.stdout.flush()
  i,j = bmu
  dist = scipy.spatial.distance.euclidean(inputMatrix[k], Map[i,j])
  distPrec = snapIds[i,j,1]
  if dist < distPrec or numpy.ma.isMaskedArray(distPrec):
   snapIds[i,j,0] = k
   snapIds[i,j,1] = dist
  k += 1
 snapIds = snapIds[:,:,0]
 snapIds = snapIds.filled(numpy.nan)
 numpy.save('snapshotIdMap.npy', snapIds)
 sys.stdout.write('\ndone\n')
