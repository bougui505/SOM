#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        

import numpy
import os
import glob

if glob.glob('PDBs') == []:
 os.mkdir('PDBs')

names = numpy.load('names.npy')
atomIds = numpy.load('atomIds.npy')
atomType = numpy.load('atomType.npy')
charges = numpy.load('charges.npy')
resTypes = numpy.load('resTypes.npy')
resIds = numpy.load('resIds.npy')
coordMat = numpy.load('rotateCoords.npy')

index = numpy.where(atomIds==1)[0].tolist()
index.append(coordMat.shape[0])
c = 0
coords_split = []
charges_split = []
resTypes_split = []
resIds_split = []
atomType_split = []
moleculeNames = []

for i in index[:-1]:
 b = i
 c+=1
 e = index[c]
 moleculeNames.append(names[b])
 coords_split.append(coordMat[b:e])
 charges_split.append(charges[b:e])
 resTypes_split.append(resTypes[b:e])
 resIds_split.append(resIds[b:e])
 atomType_split.append(atomType[b:e])

i=0
for name in moleculeNames:
 file=open('PDBs/%s.pdb'%name, 'w')
 for j in range(resIds_split[i].size):
  file.write('ATOM'.ljust(6) + ('%d'%(j+1)).rjust(5) + ' ' + ('%s'%atomType_split[i][j]).ljust(4) + ' ' + ('%s'%resTypes_split[i][j]).rjust(3) + ' ' + ' ' + ('%d'%resIds_split[i][j]).rjust(4) + ' ' + '   ' + ('%.3f'%coords_split[i][j,0]).rjust(8)  + ('%.3f'%coords_split[i][j,1]).rjust(8) + ('%.3f'%coords_split[i][j,2]).rjust(8) + '\n')
 i+=1
 file.close()



