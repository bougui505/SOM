#!/usr/bin/env python
import numpy
import sys
import SOMTools


def getNorm(map):
 X,Y,Z = map.shape
 norm = numpy.sqrt(numpy.dot( numpy.reshape(map, (X*Y,3)), numpy.reshape(map, (X*Y,3)).T ).diagonal().reshape(X,Y))
 return norm

def splitMap(map, type):
 """
 i=0 : global
 i=12: plus
 i=24: minus
 i=36: polar
 i=48: hydrophobic
 """
 twoD = False
 if map.ndim == 2:
  twoD = True
  Y,Z = map.shape
  map = map.reshape(1,Y,Z)
 X,Y,Z = map.shape
 if type == 'global':
  i=0
 elif type == 'plus':
  i=12
 elif type == 'minus':
  i=24
 elif type == 'polar':
  i=36
 elif type == 'hydrophobic':
  i=48
 mapCom = map[:,:,i:i+3]
 mapVectors1 = map[:,:,i+3:i+6] - mapCom
 mapVectors2 = map[:,:,i+6:i+9] - mapCom
 mapVectors3 = map[:,:,i+9:i+12] - mapCom
 mapNorm1 = getNorm(mapVectors1)
 mapNorm2 = getNorm(mapVectors2)
 mapNorm3 = getNorm(mapVectors3)
 mapVectors1 = mapVectors1 / numpy.atleast_3d(mapNorm1)
 mapVectors2 = mapVectors2 / numpy.atleast_3d(mapNorm2)
 mapVectors3 = mapVectors3 / numpy.atleast_3d(mapNorm3)
 numpy.save('mapNorm1_%s.npy'%type, mapNorm1)
 if twoD:
  mapCom = mapCom.reshape(Y,3)
  mapNorm1 = mapNorm1.reshape(Y,1)
  mapNorm2 = mapNorm2.reshape(Y,1)
  mapNorm3 = mapNorm3.reshape(Y,1)
  mapVectors1 = mapVectors1.reshape(Y,3)
  mapVectors2 = mapVectors2.reshape(Y,3)
  mapVectors3 = mapVectors3.reshape(Y,3)
 return mapCom,mapNorm1,mapNorm2,mapNorm3,mapVectors1,mapVectors2,mapVectors3

def projection(map1,map2,basename):
 X,Y = (map1.shape[0], map1.shape[1])
 proj = numpy.dot(map1.reshape(X*Y,3), map2.reshape(X*Y,3).T).diagonal().reshape(X,Y)
 numpy.save('projection_%s.npy'%basename, proj)
# SOMTools.plotMat(proj, 'projection_%s.pdf'%basename, interpolation='nearest')
 return proj

def get3Dcom(mapCom, basename, reshape = True):
 if reshape:
  X,Y = (mapCom.shape[0], mapCom.shape[1])
  coords = mapCom.reshape(X*Y,3)
  n = X*Y
 else:
  coords = mapCom
  n = coords.shape[0]
 outFile = open('3Dmap_%s.xyz'%(basename), 'w')
 outFile.write('%s\n'%(n))
 outFile.write('SOM of center of mass for %s\n'%basename)
 [ outFile.write('X %.3f %.3f %.3f\n'%(e[0], e[1], e[2])) for e in coords ]

def get3Dvectors(mapCom, mapVector, mapNorm, basename, bidirectional = False):
 X,Y = (mapCom.shape[0], mapCom.shape[1])
 if mapCom.ndim == 3:
  coms = mapCom.reshape(X*Y,3)
 else:
  coms = mapCom
 if mapVector.ndim == 3:
  vectors = mapVector.reshape(X*Y,3)
 else:
  vectors = mapVector
 if mapNorm.ndim == 2:
  norms = mapNorm.reshape(X*Y)
 else:
  norms = mapNorm
# vectors = 2*numpy.atleast_2d(norms).T*vectors+coms-numpy.atleast_2d(norms)
 Lambda = numpy.atleast_2d(norms).T
 vectors_Trans = coms + Lambda * vectors
 if bidirectional:
  coms = coms - Lambda * vectors
 vectors = vectors_Trans
 outFileName = '3Dvectors_%s.pdb'%basename
 outFile = open(outFileName, 'w')
 n = coms.shape[0]
 c=1
 for i in range(n):
  com = coms[i]
  comX, comY, comZ = com
  outFile.write("ATOM  %5.0f  BOX BOX     1    %8.3f%8.3f%8.3f\n"%(c,comX, comY, comZ))
  l = vectors[i]
  x,y,z = l
  c+=1
  outFile.write('ATOM  %5.0f  BOX BOX     1    %8.3f%8.3f%8.3f\n'%(c, x, y, z))
  c+=1
 i=1
 while i < 2*n:
  outFile.write('CONECT%5.0f'%i)
  i=i+1
  outFile.write('%5.0f\n'%i)
  i=i+1

def main():
 mapFileName = sys.argv[1]
 map = numpy.load(mapFileName)
 if len(map.shape) == 3:
  X,Y,Z = map.shape
 elif len(map.shape) == 2:
  map = numpy.atleast_3d(map.T).T
  X,Y,Z = map.shape

 mapCom_global, mapNorm1_global, mapNorm2_global, mapNorm3_global, mapVectors1_global,mapVectors2_global,mapVectors3_global = splitMap(map,'global')

 mapCom_plus, mapNorm1_plus, mapNorm2_plus, mapNorm3_plus, mapVectors1_plus, mapVectors2_plus , mapVectors3_plus = splitMap(map,'plus')

 mapCom_minus, mapNorm1_minus, mapNorm2_minus, mapNorm3_minus, mapVectors1_minus, mapVectors2_minus, mapVectors3_minus = splitMap(map,'minus')

 mapCom_polar, mapNorm1_polar, mapNorm2_polar, mapNorm3_polar, mapVectors1_polar, mapVectors2_polar, mapVectors3_polar = splitMap(map,'polar')

 mapCom_hydrophobic, mapNorm1_hydrophobic, mapNorm2_hydrophobic, mapNorm3_hydrophobic, mapVectors1_hydrophobic, mapVectors2_hydrophobic, mapVectors3_hydrophobic = splitMap(map, 'hydrophobic')

 projection(mapVectors1_plus,mapVectors1_global, 'plus')
 projection(mapVectors1_minus,mapVectors1_global, 'minus')
 projection(mapVectors1_polar,mapVectors1_global, 'polar')
 projection(mapVectors1_hydrophobic,mapVectors1_global, 'hydrophobic')
 get3Dcom(mapCom_global, 'global')
 get3Dcom(mapCom_plus, 'plus')
 get3Dcom(mapCom_minus, 'minus')
 get3Dcom(mapCom_polar, 'polar')
 get3Dcom(mapCom_hydrophobic, 'hydrophobic')
 get3Dvectors(mapCom_global, mapVectors1_global, mapNorm1_global, 'global1')
 get3Dvectors(mapCom_global, mapVectors2_global, mapNorm2_global, 'global2')
 get3Dvectors(mapCom_global, mapVectors3_global, mapNorm3_global, 'global3')
 get3Dvectors(mapCom_plus, mapVectors1_plus, mapNorm1_plus, 'plus')
 get3Dvectors(mapCom_minus, mapVectors1_minus, mapNorm1_minus, 'minus')
 get3Dvectors(mapCom_polar, mapVectors1_polar, mapNorm1_polar, 'polar')
 get3Dvectors(mapCom_hydrophobic, mapVectors1_hydrophobic, mapNorm1_hydrophobic, 'hydrophobic')

if __name__ == "__main__":
 main()

