# IPython log file


import SOMTools
import numpy
import os
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

smap = numpy.load('map_50x50.dat')
uMatrix = numpy.load('uMatrix.npy')
outPath, clusterPathMat, grads = SOMTools.minPath(uMatrix, 4000)
os.mkdir('PDB')
for i in range(len(outPath)):
    w,v,coords = getCoordFromNeuron(smap[outPath[i]])
    if i == 0:
        vref = v
    v = v*(numpy.atleast_2d(numpy.sign(numpy.dot(vref.T,v).diagonal())))
    coords = numpy.sqrt(w)*v
    numpy.savetxt('PDB/s_%d.txt'%i, coords)
    print "%d/%d"%(i,len(outPath))
