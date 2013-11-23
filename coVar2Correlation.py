#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import re
import numpy
import itertools
from pylab import *
import math
import pdbReader
import matriceManipulation

class Correlation():
 def __init__(self, covarMatrixFileName=None):
  """
  input: 
  covar: *.mat
  rms: *.anal
  """
  if not covarMatrixFileName is None:
   self.covarFile = open(covarMatrixFileName)
   self.extractRms()
   rmsFileName = 'extractedRms.anal'
   self.rmsFileRow = open(rmsFileName)
   self.rmsFileRow.readline()
   self.rmsFileRow.readline()
   self.rmsFileLine = open(rmsFileName)
   self.rmsFileLine.readline()
   self.rmsFileLine.readline()
   self.offsetInit = self.rmsFileLine.tell()
   self.rmsXYZ1 = itertools.chain(self.readRmsFile(self.rmsFileRow))

 def initFile(self, file, offset = 0):
  file.seek(offset)

 def loadCovarMatrix(self):
  self.initFile(self.covarFile)
  iter = True
  m = []
  while iter:
   v = self.readCovarMatrixLine()
   if v == 'EOF':
    iter = False
   else:
    m.append(v)
  return numpy.array(m)

 def readCovarMatrixLine(self):
  """
  read line i of self.covarFile
  """
  l = self.covarFile.readline().strip()
  try:
   v = [float(e) for e in re.split('\s+', l)]
   return v
  except ValueError:
   return 'EOF'

 def extractRms(self):
  """
  read the covarience matrix and extract rms from it
  """
  def writeToRmsFile(outFile):
   outFile.write( ('%s'%atomNum).rjust(10)+('%.3f'%l[0]).rjust(11)+('%.3f'%l[1]).rjust(11)+('%.3f'%l[2]).rjust(11)+'\n' )

  rms = []
  i = itertools.count()
  iter = True
  while iter:
   v = self.readCovarMatrixLine()
   if v != 'EOF':
    rms.append(v[i.next()])
   else:
    iter = False
  i = itertools.cycle(range(3))
  j = itertools.count()
  outFile = open('extractedRms.anal', 'w')
  outFile.write('Analysis of modes: RMS FLUCTUATIONS\n')
  outFile.write('  Atom no.       rmsX       rmsY       rmsZ\n')
  for e in rms:
   if i.next() == 0:
    atomNum = j.next()
    if atomNum > 0:
     writeToRmsFile(outFile)
    l = []
   l.append(math.sqrt(e))
  atomNum = j.next()
  writeToRmsFile(outFile)
  outFile.close()
  self.initFile(self.covarFile)

 def readRmsFile(self, rmsFile):
  l = rmsFile.readline().strip()
  if l != '':
#   print l
   v = [float(e) for e in re.split('\s+', l)]
   rmsX = v[1]
   rmsY = v[2]
   rmsZ = v[3]
   return [rmsX, rmsY, rmsZ]
  else:
   return 'EOF'

 def correlation(self,cov,rms1,rms2):
#  print cov, rms1, rms2
  return cov/(rms1*rms2)

 def nextRms(self):
  try:
   return self.rmsXYZ1.next()
  except StopIteration:
   self.rmsXYZ1 = itertools.chain(self.readRmsFile(self.rmsFileRow))
   return self.rmsXYZ1.next()

 def matrixLine(self):
  self.initFile(self.rmsFileLine, self.offsetInit)
  covars = self.readCovarMatrixLine()
  if covars != 'EOF':
   rmsXYZ1 = self.nextRms()
   corrs = []
   counter = itertools.cycle(range(3))
   for covar in covars:
    c = counter.next()
    if c == 0:
     rms = self.readRmsFile(self.rmsFileLine)
     rmsXYZ2 = itertools.cycle(rms)
    rms2 = rmsXYZ2.next()
    corr = self.correlation(covar, rmsXYZ1, rms2)
#    print covar, rmsXYZ1, rms2, corr
    corrs.append( corr )
   return corrs
  else:
   return 'EOF'

 def matrix(self):
  cm = []
  ml = self.matrixLine()
#  print ml
  while ml != 'EOF':
   cm.append(ml)
   ml = self.matrixLine()
#   print ml
  return numpy.array(cm)

 def plot(self, matrix, outfileName='correlation.pdf',normalize=False):
  figure()
  if normalize:
   matrix = numpy.log(abs(matrix/matrix.max()))
  imshow(matrix, interpolation='nearest')
  colorbar()
  savefig(outfileName)

 def write3Dcorr(self, matrix, pdbFileName, outFileName='corrCoords.txt', threshold = 0.0, anticorr = True):
  mm = matriceManipulation.matrix(matrix)
  matrix = mm.symmetryZeros()
  def sorter(matrix, anticorr):
   l = numpy.ravel(matrix)
   sl = numpy.sort(l)
   lsl = list(sl)
   if not anticorr:
    lsl.reverse()
   return lsl

  def replace(pn, matrix):
   """
   pn = +1|-1 
   """
   for i in range(numpy.shape(matrix)[0]):
    for j in range(numpy.shape(matrix)[1]):
     e = matrix[i,j]
     if pn*e > 0:
      matrix[i,j] = 0
   return matrix

  pdbFile = open(pdbFileName)
  pdbR = pdbReader.PdbReader(pdbFile)
  caCoords = pdbR.getCAcoord()
  resSeqs = pdbR.getResSeqs()
  if anticorr == True:
   matrix = replace(1, matrix)
  else:
   matrix = replace(-1, matrix)
  lCorrSorted = sorter(matrix, anticorr)

  outFile = open(outFileName, 'w')
  outFile.write('Sorted Ca coordinates from the most correlated\n')
  outFile.write('resSeq1'.rjust(10)+'resSeq2'.rjust(10)+'Pearson'.rjust(10)+'x1'.rjust(10)+'y1'.rjust(10)+'z1'.rjust(10)+'x2'.rjust(10)+'y2'.rjust(10)+'z2'.rjust(10)+'\n')
  for e in lCorrSorted:
   if abs(e) > abs(threshold):
    ij = numpy.where(matrix==e)
    jValues = itertools.chain(ij[1])
    for i in ij[0]:
     j = jValues.next()
     iCoord = caCoords[i]
     jCoord = caCoords[j]
     outFile.write( ('%s'%resSeqs[i]).rjust(10)+('%s'%resSeqs[j]).rjust(10)+('%.3f'%e).rjust(10)+('%.3f'%iCoord[0]).rjust(10)+('%.3f'%iCoord[1]).rjust(10)+('%.3f'%iCoord[2]).rjust(10)+('%.3f'%jCoord[0]).rjust(10)+('%.3f'%jCoord[1]).rjust(10)+('%.3f'%jCoord[2]).rjust(10)+'\n' )
   else:
    break

 def plot3Dcorr(self, fileName='corrCoords.txt', outFileName='corr3Dplot.py'):
  file = open(fileName)
  file.readline()
  file.readline()
  def readCoord(file):
   l = file.readline()
   if l == '':
    return 'EOF'
   else:
#    print l
    v = re.split('\s+', l)
#    print v
    coord1 = v[4:7]
    coord2 = v[7:10]
    return coord1, coord2
  outFile = open(outFileName, 'w')
  outFile.write('from pymol.cgo import *\n')
  outFile.write('from pymol import cmd\n')
  outFile.write('obj = []\n')
  def writePymolCgoScript(file, coords, index=1):
#   print coords
   file.write('corr = [\n')
   file.write('LINEWIDTH, 1.0,\n')
   file.write('BEGIN, LINES,\n')
   file.write('COLOR,    0.100,    1.000,    0.000,\n')
   file.write('VERTEX, %s, %s, %s,\n'%(coords[0][0], coords[0][1], coords[0][2]))
   file.write('VERTEX, %s, %s, %s,\n'%(coords[1][0], coords[1][1], coords[1][2]))
   file.write('END\n')
   file.write(']\n')
   file.write('obj = obj + corr\n')
   file.write("cmd.load_cgo(obj,'correlation', %s)\n"%index)
  coords = readCoord(file)
  count = itertools.count(1)
  while coords != 'EOF':
   writePymolCgoScript(outFile, coords, count.next())
   coords = readCoord(file)
