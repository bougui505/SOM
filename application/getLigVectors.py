#!/usr/bin/env python
import numpy
import PCA
import sys
import pickle
import glob
import ConfigParser
import splitDockMap
import math

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

ligCharges = numpy.load(Config.get('pc', 'charges'))
ligAtomIds = numpy.load(Config.get('pc', 'atomIds'))
coordMat = numpy.load(Config.get('pc','coordMat'))
align = Config.getboolean('pc', 'align')
residueDescription = Config.getboolean('pc', 'residue')
if residueDescription:
 resTypes = numpy.load(Config.get('pc','resTypes'))

def pdbBoxWriter(com, vectors, outFileName='ligBox.pdb'):
 outFile = open(outFileName, 'w')
 comX, comY, comZ = com
 outFile.write("""
ATOM  %5.0f  BOX BOX     1    %8.3f%8.3f%8.3f
"""%(1,comX, comY, comZ))
 c=1
 for l in vectors:
  x,y,z = l
#  outFile.write('BOX %.3f %.3f %.3f\n'%(x, y, z))
  c+=1
  outFile.write('ATOM  %5.0f  BOX BOX     1    %8.3f%8.3f%8.3f\n'%(c, x, y, z))
 outFile.write('CONECT    1    2    3    4')

def getPC(coords, outFileName='ligBox.pdb'):
 size = coords.size
 shape = coords.shape
 if shape != (1,3):
  if size != 0 :
   eigenVectors, eigenValues = PCA.princomp(coords.T, numpc=3, getEigenValues=True)
   com = coords.mean(axis=0)
   projection = numpy.dot(coords-com,eigenVectors)
   signs = numpy.sign(numpy.sign(projection).sum(axis=0))
   signs2 = numpy.sign(projection[numpy.abs(projection).argmax(axis=0)].diagonal())
   signs[signs==0] = signs2[signs==0]
   eigenVectors = eigenVectors*signs
   vectors = com + eigenVectors.T * numpy.atleast_2d(numpy.sqrt(eigenValues)).T
  elif size == 0:
   com = numpy.zeros((3))
   vectors = numpy.zeros((3,3))
 else:
  com = coords.flatten()
  vectors = numpy.zeros((3,3))
# pdbBoxWriter(com, vectors, outFileName)
 return com, vectors

def getDescriptors(inputs):
 coords, charges = inputs
 com_global, vectors_global = getPC(coords, 'ligBox_global.pdb')
 coords_plus = coords[charges>=1]
 coords_minus = coords[charges<=-1]
 coords_polar = coords[numpy.logical_or(numpy.logical_and(charges>-1,charges<=-0.5), numpy.logical_and(charges>=0.5,charges<1))]
 coords_hydrophobic = coords[numpy.logical_and(charges>-0.5,charges<0.5)]
 com_plus, vectors_plus = getPC(coords_plus, 'ligBox_plus.pdb')
 com_minus, vectors_minus = getPC(coords_minus, 'ligBox_minus.pdb')
 com_polar, vectors_polar = getPC(coords_polar, 'ligBox_polar.pdb')
 com_hydrophobic, vectors_hydrophobic = getPC(coords_hydrophobic, 'ligBox_hydrophobic.pdb')
 descriptors = []
 [ descriptors.extend(e.flatten().tolist()) for e in [com_global, vectors_global, com_plus, vectors_plus, com_minus, vectors_minus, com_polar, vectors_polar, com_hydrophobic, vectors_hydrophobic]]
 return descriptors

def getResidueDescriptors(inputs):
 aaGroups = {'LYS': 'plus', 'ARG': 'plus', 'ASP': 'minus', 'GLU': 'minus', 'GLN': 'polar', 'ASN':'polar', 'SER':'polar', 'CYS':'polar', 'THR':'polar', 'ILE':'hydrophobic', 'VAL':'hydrophobic', 'LEU':'hydrophobic', 'MET':'hydrophobic', 'PHE':'hydrophobic', 'TYR':'hydrophobic', 'TRP':'hydrophobic'}
 coords, resTypes = inputs
 aas = resTypes
 resProperties = []
 for aa in aas:
  if aa in aaGroups.keys():
   resProperties.append(aaGroups[aa])
  else:
   resProperties.append('nd')
 com_global, vectors_global = getPC(coords, 'ligBox_global.pdb')
 sel_plus = numpy.array([e=='plus' for e in resProperties])
 sel_minus = numpy.array([e=='minus' for e in resProperties])
 sel_polar = numpy.array([e=='polar' for e in resProperties])
 sel_hydrophobic = numpy.array([e=='hydrophobic' for e in resProperties])
# print 'plus: %s'%resTypes[sel_plus]
# print 'minus: %s'%resTypes[sel_minus]
# print 'polar: %s'%resTypes[sel_polar]
# print 'hydrophobic: %s'%resTypes[sel_hydrophobic]
 coords_plus = coords[sel_plus]
 coords_minus = coords[sel_minus]
 coords_polar = coords[sel_polar]
 coords_hydrophobic = coords[sel_hydrophobic]
 com_plus, vectors_plus = getPC(coords_plus, 'ligBox_plus.pdb')
 com_minus, vectors_minus = getPC(coords_minus, 'ligBox_minus.pdb')
 com_polar, vectors_polar = getPC(coords_polar, 'ligBox_polar.pdb')
 com_hydrophobic, vectors_hydrophobic = getPC(coords_hydrophobic, 'ligBox_hydrophobic.pdb')
 descriptors = []
 [ descriptors.extend(e.flatten().tolist()) for e in [com_global, vectors_global, com_plus, vectors_plus, com_minus, vectors_minus, com_polar, vectors_polar, com_hydrophobic, vectors_hydrophobic]]
 return descriptors

def rotate_descriptors(rotation_matrix, descriptor):
 card = descriptor.size
 descriptor = descriptor.reshape(card/3,3)
 rotate_descriptor = []
 for d in descriptor:
  rotate_descriptor.extend(numpy.dot(rotation_matrix, d).tolist())
#  rotate_descriptor.extend(d.tolist())
 return rotate_descriptor

def rotation_matrix(angle, direction, point=None):
 """Return matrix to rotate about axis defined by point and direction.

 >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
 >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
 True
 >>> angle = (random.random() - 0.5) * (2*math.pi)
 >>> direc = numpy.random.random(3) - 0.5
 >>> point = numpy.random.random(3) - 0.5
 >>> R0 = rotation_matrix(angle, direc, point)
 >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
 >>> is_same_transform(R0, R1)
 True
 >>> R0 = rotation_matrix(angle, direc, point)
 >>> R1 = rotation_matrix(-angle, -direc, point)
 >>> is_same_transform(R0, R1)
 True
 >>> I = numpy.identity(4, numpy.float64)
 >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
 True
 >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
 ...                                               direc, point)))
 True

 """
 sina = math.sin(angle)
 cosa = math.cos(angle)
# direction = unit_vector(direction[:3])
 # rotation matrix around unit vector
 R = numpy.diag([cosa, cosa, cosa])
 R += numpy.outer(direction, direction) * (1.0 - cosa)
 direction *= sina
 R += numpy.array([[ 0.0,         -direction[2],  direction[1]],
                   [ direction[2], 0.0,          -direction[0]],
                   [-direction[1], direction[0],  0.0]])
 M = numpy.identity(4)
 M[:3, :3] = R
 if point is not None:
  # rotation not around origin
  point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
  M[:3, 3] = point - numpy.dot(R, point)
 return M

def getNorm(v):
 return numpy.sqrt(numpy.dot(v,v))

def getRotationParameters(v1,v2):
 norm1 = getNorm(v1)
 norm2 = getNorm(v2)
 v1 = v1/norm1
 v2 = v2/norm2
 angle = numpy.arccos(numpy.dot(v1,v2))
 direction = numpy.cross(v1,v2)
 direction = direction/getNorm(direction)
 return angle, direction

index = numpy.where(ligAtomIds==1)[0].tolist()
index.append(coordMat.shape[0])
c = 0
coords_split = []
charges_split = []
if residueDescription:
 resTypes_split = []
for i in index[:-1]:
 b = i
 c+=1
 e = index[c]
 coords_split.append(coordMat[b:e])
 charges_split.append(ligCharges[b:e])
 if residueDescription:
  resTypes_split.append(resTypes[b:e])

k=0
n = len(coords_split)
descriptors = []
while k < n:
 sys.stdout.write('%9.f/%9.f'%(k+1,n))
 sys.stdout.write('\r')
 sys.stdout.flush()
 coords_k = coords_split[k]
 charges_k = charges_split[k]
 if residueDescription:
  resTypes_k = resTypes_split[k]
  input = [coords_k, resTypes_k]
  descriptors.append(getResidueDescriptors(input))
 else:
  input = [coords_k, charges_k]
  descriptors.append(getDescriptors(input))
 k+=1
descriptors = numpy.array(descriptors)
descriptors[numpy.isnan(descriptors)] = 0
if align:
 origins = descriptors[:,:3]
 translateDescriptors = numpy.zeros_like(descriptors)
 for i in range(0,60,3):
  translateDescriptors[:,i:i+3] = descriptors[:,i:i+3] - descriptors[:,:3]
 descriptors = translateDescriptors
 mapCom,mapNorm1,mapNorm2,mapNorm3,mapVectors1,mapVectors2,mapVectors3 = splitDockMap.splitMap(descriptors, 'global')
 #vRef = numpy.append(mapNorm1[0]*mapVectors1[0], mapNorm2[0]*mapVectors2[0]).reshape(2,3)
 vRef = descriptors[0,3:12].reshape(3,3)
# vRef[2]=0
 rotateDescriptors = []
 rotateCoords = []
 for i in range(mapVectors1.shape[0]):
#  v = numpy.append(mapVectors1[i], mapVectors2[i], mapVectors3[i]).reshape(3,3)
  v = descriptors[i,3:12].reshape(3,3)
  angle, direction = getRotationParameters(v[0],vRef[0])
  if numpy.isnan(getNorm(direction)):
   direction = v[2]/getNorm(v[2])
  rotMat1 = rotation_matrix(angle,direction)[:3,:3]
  angle, direction = getRotationParameters(numpy.dot(rotMat1, v[1]),vRef[1])
  rotMat2 = rotation_matrix(angle,direction)[:3,:3]
  rotMat = numpy.dot(rotMat2, rotMat1)
  rotateDescriptors.append(rotate_descriptors(rotMat, descriptors[i]))
  coords_split[i] = coords_split[i] - origins[i]
  rotateCoords.extend(numpy.dot(rotMat, coords_split[i].T).T.tolist())
 rotateDescriptors = numpy.array(rotateDescriptors)
 rotateCoords = numpy.array(rotateCoords)
 descriptors = rotateDescriptors
  
numpy.save('inputMatrix.npy', descriptors)
numpy.save('rotateCoords.npy', rotateCoords)
sys.stdout.write('\ndone\n')
#charges = params['charge']
#getDescriptors([coords, params])
#eigenVectorsPool = pool.map(PCA.princomp, distMats)
