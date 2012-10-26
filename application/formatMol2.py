#!/usr/bin/env python
import sys
import numpy

frFileName = sys.argv[1]

fr=open(frFileName)
basename = frFileName.split('.')[0]
fw=open('%s.coord'%basename,'w')

record=False
i=0
recordName = False
ligNames = []
ligCharges = []
ligAtomIds = []
atomName = []
atomType = []
resTypes = []
resIds = []
coordMat = []
print 'Reading parameters...'
for line in fr:
 if recordName:
  name=line.strip()
  recordName = False
 if line.find('@<TRIPOS>MOLECULE')!=-1:
  recordName = True

 if line.find('@<TRIPOS>BOND')!=-1:
  record=False

 if record:
  sp=line.split()
  fw.write(name+'\t'+sp[0]+'\t'+sp[1]+'\t'+sp[2]+'\t'+sp[3]+'\t'+sp[4]+'\t'+sp[8]+'\n')
  coordMat.append([ sp[2],sp[3],sp[4] ])
  ligNames.append(name)
  ligCharges.append(sp[8])
  ligAtomIds.append(sp[0])
  atomName.append(sp[1])
  atomType.append(sp[5])
  resTypes.append(sp[7][:3])
  resIds.append(sp[6])

 if line.find('@<TRIPOS>ATOM')!=-1:
  i=i+1
  record=True
fr.close()
fw.close()

print 'Array conversion...'
ligNames = numpy.array(ligNames)
ligCharges = numpy.array(ligCharges, dtype=float)
ligAtomIds = numpy.array(ligAtomIds, dtype=int)
resTypes = numpy.array(resTypes)
atomName = numpy.array(atomName)
resIds = numpy.array(resIds, dtype=int)
coordMat = numpy.array(coordMat, dtype=float)
atomType = numpy.array(atomType)
print 'Array filtering...'
filter = numpy.bool_(1-numpy.isnan(coordMat).any(axis=1))
print 'Writing parameter files...'
#numpy.save('%s_names.npy'%basename, ligNames[filter])
#numpy.save('%s_charges.npy'%basename, ligCharges[filter])
#numpy.save('%s_atomIds.npy'%basename, ligAtomIds[filter])
numpy.save('%s_coordMat.npy'%basename, coordMat[filter])
#numpy.save('%s_resTypes.npy'%basename, resTypes[filter])
#numpy.save('%s_resIds.npy'%basename, resIds[filter])
#numpy.save('%s_atomName.npy'%basename, atomName[filter])
#numpy.save('%s_atomType.npy'%basename, atomType[filter])
numpy.savez('%s_parameters.npz'%basename, names=ligNames[filter], charges=ligCharges[filter], atomIds=ligAtomIds[filter], resTypes=resTypes[filter], resIds=resIds[filter], atomNames=atomName[filter], atomTypes=atomType[filter])
