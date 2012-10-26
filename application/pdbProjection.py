#!/usr/bin/env/python
import pdbReader
import pdbWriter
import numpy

def projection(eigenvalue, eigenvector, pdbFileName, clusterVarComparison=None, outPdbFile='projection.pdb', append=False, modelNumber=0, continuousScale=False, ca = True, bb = False):
 pdbFile = open(pdbFileName)
 pdbR = pdbReader.PdbReader(pdbFile)
 if clusterVarComparison is None:
  p = abs(eigenvalue*eigenvector)
 else:
  cl1,cl2,var=clusterVarComparison
  print var.shape,eigenvector.shape
  p = var*( (1.*(eigenvector == cl1)-1.*(eigenvector == cl2)))
 pdbFile = open(pdbFileName)
 pdbW = pdbWriter.PdbWriter(pdbFile)
 pdbW.addBFactor(p, outPdbFile, append=append, modelNumber=modelNumber, ca = ca, bb = bb)
 pymolScript = open(outPdbFile.replace('.pdb', '.pml'), 'w')
 max = numpy.max(eigenvector)
 if not continuousScale:
  pymolScript.write("""
load %s
hide everything
show cartoon
spectrum b, minimum=0, maximum=%s
  """%(outPdbFile, max))
 else:
  pymolScript.write("""
load %s
hide everything
show cartoon
spectrum b, blue_white_red, minimum=-1, maximum=1
alter all, b=str(abs(b))
cartoon putty
unset cartoon_smooth_loops
unset cartoon_flat_sheets
  """%(outPdbFile))
 pymolScript.close()
