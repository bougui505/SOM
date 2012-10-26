import numpy
bmuCoordinates = numpy.load('bmuCoordinates.npy')
density = numpy.load('density.npy')
networkFile = open('SOMTrajNetwork.net', 'w')
nVertices = density.size
networkFile.write('*Vertices %s\n'%nVertices)
#nodeNames = numpy.load('rmsMap.npy')
nodeNames = numpy.arange(bmuCoordinates.shape[0])
for i in range(nVertices):
# networkFile.write('%s "%s" %s\n'%(i+1,i,density.flatten()[i]))
 bmuI, bmuJ = numpy.unravel_index(i, density.shape)
 nodeName = nodeNames.flatten()[i]
# networkFile.write('%s "%s" 1.0\n'%(i+1,nodeName))
 networkFile.write('%s "%s,%s" 1.0\n'%(i+1,bmuI, bmuJ))
nArcs = bmuCoordinates.shape[0] - 1
networkFile.write('*Arcs %s\n'%nArcs)
ravelCoordinates = [numpy.ravel_multi_index(bmu, density.shape) for bmu in bmuCoordinates]
arcs = zip(ravelCoordinates[:-1], ravelCoordinates[1:])
for arc in arcs:
 a1,a2 = arc
 networkFile.write('%s %s 1.0\n'%(a1+1,a2+1))
networkFile.close()
