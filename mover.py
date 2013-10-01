
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import numpy
import pdbReader
import pdbWriter
import itertools

class Mover:
 def __init__(self, matrix):
  self.matrix = matrix
  self.shape = numpy.shape(matrix)

 def move(self, eigenvalue, eigenvector, pdbFileName, clusterVarComparison=None, outPdbFile='move.pdb', append=False, modelNumber=0, continuousScale=False, ca = True, bb = False):
  pdbFile = open(pdbFileName)
  pdbR = pdbReader.PdbReader(pdbFile)
  if clusterVarComparison is None:
    p = abs(eigenvalue*eigenvector)
  else:
    cl1,cl2,var=clusterVarComparison
#    print var.shape,eigenvector.shape
    c = (1.*(eigenvector == cl1)+1.*(eigenvector == cl2))
    p = var* (numpy.array([c,c,c]).transpose())
  pdbFile = open(pdbFileName)
  pdbW = pdbWriter.PdbWriter(pdbFile)
  pdbW.alterCoordinates(p, outPdbFile, append=append, modelNumber=modelNumber, ca = ca, bb = bb)

 def moveRms(self, pdbFileName, varxyz, outPdbFileName='moveRms.pdb', ca = True, bb = False):
  f = open(outPdbFileName, 'w')
  f.close()
  diagonal=self.matrix.diagonal().astype('int') #clusterMatrix
#  bins=list(numpy.bincount(diagonal).argsort())[1:]
#  bins.reverse()
  bins = list(numpy.unique(diagonal))
#  print varxyz.matrix
  var=numpy.sqrt(varxyz.matrix.diagonal())
#  var/=max(var)
  xs = itertools.islice(var,0,len(var),3)
  ys = itertools.islice(var,1,len(var),3)
  zs = itertools.islice(var,2,len(var),3)
  var = numpy.array([ [xs.next(),ys.next(),zs.next()] for i in range(len(var)/3) ])
  c = itertools.count()
  for i in range(len(bins)):
   for j in range(i+1, len(bins)):
    self.move(1, diagonal, pdbFileName,clusterVarComparison=(bins[i],bins[j],var), outPdbFile='%s_%s_%s.pdb'%(outPdbFileName.replace('.pdb', ''), i, j), continuousScale=True, ca = ca, bb = bb)
