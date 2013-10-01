# IPython log file

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        


import SOMTools
import numpy
import os
import pickle
import IO

def getCoordFromNeuron(n):
    naa = n.size/9
    eigenVectors = n[:4*naa].reshape(naa,4)
    projections = n[4*naa:8*naa].reshape(4,naa)
    means = n[8*naa:]
    m_r = numpy.dot(eigenVectors,projections) + numpy.atleast_2d(means).T
    w,v = numpy.linalg.eig(m_r-numpy.atleast_2d(numpy.mean(m_r,axis=1)).T)
    w = numpy.sign(w)*w
    coords = numpy.real( numpy.sqrt( numpy.sign(w[:3])*w[:3] ) * numpy.sign(w[:3]) * v[:,:3] )
    return w[:3].real,v[:,:3].real,coords

def getChirality(coord1,coord2,coord3,coord4):
    v0 = coord2 - coord1
    v1 = coord3 - coord2
    v2 = coord4 - coord3
    chirality = numpy.dot(numpy.cross(v0,v1),v2)/(numpy.linalg.norm(v1))**3
    return chirality

def getChiralities(coords):
    zippedCoords = zip(coords[0:],coords[1:],coords[2:],coords[3:])
    chiralities = []
    for e in zippedCoords:
        chirality = getChirality(e[0],e[1],e[2],e[3])
        chiralities.append(chirality)
    return chiralities

#smap = numpy.load('map_50x50.dat')
getPath = False
smap = pickle.load(open('map_50x50.dat'))
if getPath:
    uMatrix = numpy.load('uMatrix.npy')
    energy = numpy.log(uMatrix/uMatrix.max())
    outPath, clusterPathMat, grads = SOMTools.minPath(energy, 0.13)
    clusterPathMat[outPath[0]] = 1
    outPath = outPath
else:
    outPath = []
    for i in range(smap.shape[0]):
        for j in range(smap.shape[1]):
            outPath.append((i,j))
traj = []
nframes = len(outPath)
for i in range(nframes):
    w,v,coords = getCoordFromNeuron(smap[outPath[i]])
    if i == 0:
        vref = v
        coordsRef = coords
    v1 = v*(numpy.atleast_2d(numpy.sign(numpy.dot(vref.T,v).diagonal())))
    v2 = v
#    vs = [ v, v*numpy.atleast_2d([-1,1,1]), v*numpy.atleast_2d([1,-1,1]), v*numpy.atleast_2d([1,1,-1]), v*numpy.atleast_2d([-1,-1,1]), v*numpy.atleast_2d([-1,1,-1]), v*numpy.atleast_2d([1,-1,-1]), v*numpy.atleast_2d([-1,-1,-1]) ]
    coordsN = [ numpy.sqrt(w)*e for e in [v1,v2,-v1,-v2] ] / numpy.sqrt(2)
    rmsds = [ ((e - coordsRef)**2).sum(axis=1).mean() for e in coordsN ]
    chiralities = [ numpy.sum(numpy.sign(getChiralities(e))) for e in coordsN ]
    print chiralities
    coords = coordsN[numpy.argmax(chiralities)]
    traj.append(coords)
    print "%d/%d %d"%(i+1,len(outPath),numpy.max(chiralities))
traj = numpy.asarray(traj)
numpy.save('SOM_traj', traj)
