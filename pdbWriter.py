#!/usr/bin/env python
from pdbReader import PdbReader
import itertools
import numpy

class PdbWriter(PdbReader):
 def writeChain(self, chainId, outPdbFilename):
  outPdbFile = open(outPdbFilename, 'w')
  for line in self.pdbLines:
   if self.isAtom(line):
    if self.getChainId(line) == chainId:
     outPdbFile.write(line)
  outPdbFile.close()

 def writeChains(self, chainIds, outPdbFilename, hetatm=True):
  outPdbFile = open(outPdbFilename, 'w')
  for line in self.pdbLines:
   if self.isAtom(line, hetatm):
    if self.getChainId(line) in chainIds:
     outPdbFile.write(line)
  outPdbFile.close()

 def addChainId(self, chainId, outPdbFilename):
  lines = self.pdbLines
  outPdbFile = open(outPdbFilename, 'w')
  for line in self.pdbLines:
   if self.isAtom(line):
    outPdbFile.write('%s%s%s'%(line[0:21],chainId, line[22:]))
  outPdbFile.close()

 def addBFactor(self, data, outPdbFilename, append = False, modelNumber=0, ca = True, bb = False):
  """
  data = [d1,d2,...,dn], with n the number of residues if ca == True
  """
  lines = self.pdbLines
  if append == False:
   outPdbFile = open(outPdbFilename, 'w')
  else:
   outPdbFile = open(outPdbFilename, 'a')
   outPdbFile.write('MODEL'+('%s'%modelNumber).rjust(9)+'\n')
  resSeqOld = 0
  idata = itertools.chain(data)
  for line in self.pdbLines:
   try:
    if self.isAtom(line):
     resSeq = self.getResSeq(line)
     if resSeq != resSeqOld:
      resSeqOld = resSeq
      value = ('%.2f'%idata.next()).rjust(6)
     if ca:
      outPdbFile.write('%s%s\n'%(line[0:61],value))
     elif bb:
      if self.getAtomName(line) == 'N':
       outPdbFile.write('%s%s\n'%(line[0:61],value))
       value = ('%.2f'%idata.next()).rjust(6)
      if self.getAtomName(line) == 'CA':
       outPdbFile.write('%s%s\n'%(line[0:61],value))
       value = ('%.2f'%idata.next()).rjust(6)
      if self.getAtomName(line) == 'C':
       outPdbFile.write('%s%s\n'%(line[0:61],value))
#      else:
#       value = '0.00'.rjust(6)
#       outPdbFile.write('%s%s\n'%(line[0:61],value))
   except StopIteration:
    break
  if append:
   outPdbFile.write('ENDMDL\n')

 def alterCoordinates(self, data, outPdbFilename, append = False, modelNumber=0, ca = True, bb = False, allAtom=False, add=True):
  """
  data = [[x1,y1,z1], [x2,y2,z2], ...], with n the number of residues if ca == True
  """
  def alter(vector):
   coords = numpy.array(self.getCoord(line))
   if add:
    rCoords = coords + vector
   else:
    rCoords = vector
   xValue = ('%.3f'%rCoords[0]).rjust(8)
   yValue = ('%.3f'%rCoords[1]).rjust(8)
   zValue = ('%.3f'%rCoords[2]).rjust(8)
   outPdbFile.write('%s%s%s%s\n'%(line[0:30],xValue,yValue,zValue))
  lines = self.pdbLines
  if append == False:
   outPdbFile = open(outPdbFilename, 'w')
  else:
   outPdbFile = open(outPdbFilename, 'a')
   outPdbFile.write('MODEL'+('%s'%modelNumber).rjust(9)+'\n')
  resSeqOld = 0
  idata = itertools.chain(data)
  for line in self.pdbLines:
   try:
    if self.isAtom(line):
     resSeq = self.getResSeq(line)
     if not allAtom:
      if resSeq != resSeqOld:
       resSeqOld = resSeq
       vector = idata.next()
     if ca:
      alter(vector)
     elif bb:
      if self.getAtomName(line) == 'N':
       alter(vector)
       vector = idata.next()
      if self.getAtomName(line) == 'CA':
       alter(vector)
       vector = idata.next()
      if self.getAtomName(line) == 'C':
       alter(vector)
     elif allAtom:
      vector = idata.next()
      alter(vector)
#      else:
#       value = '0.00'.rjust(6)
#       outPdbFile.write('%s%s\n'%(line[0:61],value))
   except StopIteration:
    break
  if append:
   outPdbFile.write('ENDMDL\n')
