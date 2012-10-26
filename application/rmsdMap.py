#!/usr/bin/env python
import numpy
import IO
import ConfigParser
import sys
import SOMTools

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

nframe = Config.getint('rmsdMap', 'nframes')
structFile = Config.get('rmsdMap', 'structFile')
trajFile = Config.get('rmsdMap', 'trajFile')
snapshotIdMap = Config.get('rmsdMap', 'snapshotIdMap')
multipleStructures = Config.getboolean('rmsdMap', 'multipleStructures')
bmuCoordinatesFileName = Config.get('rmsdMap', 'bmuCoordinates')

struct = IO.Structure(structFile)
mask = numpy.ones((struct.atoms.shape[0]),dtype="bool")
traj = IO.Trajectory(trajFile, struct, selectionmask=mask, nframe=nframe)
shapeTraj = traj.array.reshape(nframe, traj.natom, 3)

snapshotIdMap = numpy.load(snapshotIdMap)
X,Y = snapshotIdMap.shape
snapshotIdMap_expand = SOMTools.expandMatrix(snapshotIdMap)

rmsdMap = numpy.ma.masked_all((X,Y))

n = X*Y
if multipleStructures:
 k = 0
 bmuCoordinates = numpy.load(bmuCoordinatesFileName)
 for i in range(100,200):
  for j in range(100,200):
   k+=1
   sys.stdout.write('%s/%s'%(k,n))
   sys.stdout.write('\r')
   sys.stdout.flush()
   snapIds = []
   sRef = snapshotIdMap_expand[i,j]
   snapIds.append(sRef)
   snapRefIds = numpy.where((bmuCoordinates == numpy.array([i%X, j%Y])).all(axis=1))[0].tolist()
   snapIds.extend(snapRefIds)
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([(i-1)%X, (j-1)%Y])).all(axis=1))[0].tolist())
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([(i-1)%X, j%Y])).all(axis=1))[0].tolist())
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([(i-1)%X, (j+1)%Y])).all(axis=1))[0].tolist())
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([i%X, (j-1)%Y])).all(axis=1))[0].tolist())
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([i%X, (j+1)%Y])).all(axis=1))[0].tolist())
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([(i+1)%X, (j-1)%Y])).all(axis=1))[0].tolist())
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([(i+1)%X, j%Y])).all(axis=1))[0].tolist())
   snapIds.extend(numpy.where((bmuCoordinates == numpy.array([(i+1)%X, (j+1)%Y])).all(axis=1))[0].tolist())
   if snapRefIds != []:
    snapTraj = shapeTraj[snapIds]
#    rmsd = numpy.sqrt((((snapTraj - snapTraj[0])**2).sum(axis=2).sum(axis=1) / traj.natom))[1:].mean()
    rmsd = numpy.sqrt((((snapTraj - snapTraj[0])**2).sum(axis=2).sum(axis=1) / traj.natom))
    rmsd = numpy.ma.masked_array(rmsd, rmsd==0)
    rmsd = rmsd.mean()
    rmsdMap[i%X,j%Y] = rmsd
 rmsdMap = rmsdMap.filled(numpy.nan)
 numpy.save('rmsdMap_multiple.npy', rmsdMap)
 sys.stdout.flush()



else:
 k = 0
 for i in range(100,200):
  for j in range(100,200):
   k+=1
   sys.stdout.write('%s/%s'%(k+1,n))
   sys.stdout.write('\r')
   sys.stdout.flush()
   sRef = snapshotIdMap_expand[i,j]
   if not numpy.isnan(sRef):
    snapIds = [sRef]
    i1,j1 = i-1, j-1
    s1 = snapshotIdMap_expand[i1,j1]
    if not numpy.isnan(s1):
     snapIds.append(s1)
    i2,j2 = i-1,j
    s2 = snapshotIdMap_expand[i2,j2]
    if not numpy.isnan(s2):
     snapIds.append(s2)
    i3,j3 = i-1,j+1
    s3 = snapshotIdMap_expand[i3,j3]
    if not numpy.isnan(s3):
     snapIds.append(s3)
    i4,j4 = i,j-1
    s4 = snapshotIdMap_expand[i4,j4]
    if not numpy.isnan(s4):
     snapIds.append(s4)
    i5,j5 = i,j+1
    s5 = snapshotIdMap_expand[i5,j5]
    if not numpy.isnan(s5):
     snapIds.append(s5)
    i6,j6 = i+1,j-1
    s6 = snapshotIdMap_expand[i6,j6]
    if not numpy.isnan(s6):
     snapIds.append(s6)
    i7,j7 = i+1,j
    s7 = snapshotIdMap_expand[i7,j7]
    if not numpy.isnan(s7):
     snapIds.append(s7)
    i8,j8 = i+1,j+1
    s8 = snapshotIdMap_expand[i8,j8]
    if not numpy.isnan(s8):
     snapIds.append(s8)
    snapTraj = shapeTraj[snapIds]
    rmsd = numpy.sqrt((((snapTraj - snapTraj[0])**2).sum(axis=2).sum(axis=1) / traj.natom))[1:].mean()
    rmsdMap[i%X,j%Y] = rmsd
 rmsdMap = rmsdMap.filled(numpy.nan)
 numpy.save('rmsdMap.npy', rmsdMap)
 sys.stdout.flush()
