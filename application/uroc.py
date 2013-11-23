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
import re
import matplotlib.pyplot
import SOMTools
import pickle
import sys

def getMinValue(inputValues, bmus, ligNames):
 minIndex = [] 
 uValueList = []
 for k in range(bmus.shape[0]):
  bmu = bmus[k]
  i,j = bmu
  if len(inputValues.shape)==2:
   uValue = inputValues[i,j]
  else:
   uValue = inputValues[k]
  if ligNames[k] != ligNames[k-1] or k == bmus.shape[0] - 1: 
   if k != 0:
    MinPoz = numpy.argmin(uValueList)
    MinList = numpy.bool_(numpy.zeros_like(uValueList)).tolist()
    MinList[MinPoz] = True
    minIndex.extend(MinList)
    uValueList = []
  uValueList.append(uValue)
 minIndex = numpy.array(minIndex)
 return minIndex

def getPathROC(uMatrix, bmus, ligNames, motif, FileName): 
 outPath, clusterPathMat, grads = SOMTools.minPath(uMatrix, 0.1*numpy.max(uMatrix))
 pathLigNames = []
 for e in outPath:
  pathLigNames.append(ligNames[numpy.where((bmus==e).all(axis=1))[0]])
 R = getROC(ligNames, motif, FileName)
 return R

def getCompounds(e, motif):
 if re.findall(motif, e):
  return True
 else:
  return False

def getActiveMap(uMatrix, bmus, ligNames, motif, getCompounds = getCompounds):
 getCompounds = numpy.vectorize(getCompounds)
 X, Y = uMatrix.shape
 density =  numpy.zeros((X,Y))
 for bmu in bmus:
  i,j = bmu
  density[i,j] += 1
 f = getCompounds(ligNames, motif)
 activeBMUs = bmus[f]
 activeMap = numpy.zeros((X,Y))
 for bmu in activeBMUs:
  i,j = bmu
  activeMap[i,j]+=1
 activeMap = numpy.ma.masked_array(activeMap, activeMap == 0)
 pickle.dump(activeMap, open('activeMap.dat', 'w'))
 tprMap = activeMap/numpy.sum(activeMap)
 activeProportionMap = activeMap/density
 numpy.save('densityBestPose.npy', density)
 SOMTools.plotMat(density, 'densityBestPose.pdf', interpolation='nearest')
 return activeMap, tprMap, activeProportionMap


def getROC(ligNames, motif, FileName, uValues, getCompounds = getCompounds):
 getCompounds = numpy.vectorize(getCompounds)
 f = getCompounds(ligNames, motif)
 n1 = numpy.size(uValues)
 n2 = numpy.size(numpy.unique(uValues))
 multipleValues = ( n1 != n2 )
 if multipleValues:
  print 'WARNING: Same sorting value for multiple compounds'
 f = f[numpy.argsort(uValues)]
 uValues = uValues[numpy.argsort(uValues)]
 fprec = -1
 uValue_prec = -1
 FP = 0.
 TP = 0.
 R = []
 N = (1-f).sum()
 P = f.sum()
 print 'card(N):%s; card(P):%s'%(N,P)
 for i in range(f.size):
  if multipleValues:
   test = (uValues[i] != uValue_prec)
  else:
   test = (f[i]!=fprec)
  if test:
   sys.stdout.write('TP: %9.0f;FP: %9.0f'%(TP, FP))
   sys.stdout.write('\r')
   sys.stdout.flush()
   R.append((FP/N, TP/P))
   fprec = f[i]
   uValue_prec = uValues[i]
  if f[i]:
   TP+=1
  else:
   FP+=1
 sys.stdout.write('TP: %9.0f;FP: %9.0f\nNumber of ROC points: %s\n'%(TP, FP, len(R)))
 R.append((FP/N, TP/P))
 
 x = [e[0] for e in R]
 y = [e[1] for e in R]
 AUC = numpy.trapz(y,x,dx=1./len(x))
 matplotlib.pyplot.clf()
 matplotlib.pyplot.plot([0,1],[0,1])
 matplotlib.pyplot.plot(x,y)
 matplotlib.pyplot.savefig(FileName)

 return R, AUC

def main():
 uMatrix= numpy.load('uMatrix.npy')
 bmuProb = 1 - numpy.load('bmuProb.npy')
 bmuCoordinates = numpy.load('bmuCoordinates.npy')
 ligNames = numpy.load('names.npy')
 ligAtomIds = numpy.load('atomIds.npy')
 ligNames=ligNames[ligAtomIds==1]
 Scores= numpy.genfromtxt('scores.txt')

 activeMap, tprMap, activeProportionMap = getActiveMap(uMatrix, bmuCoordinates, ligNames, 'CHEMBL')
 SOMTools.plotMat(activeMap, 'activeMap.pdf', interpolation='nearest')
 SOMTools.plotMat(tprMap, 'tprMap.pdf', interpolation='nearest')
 SOMTools.plotMat(activeProportionMap, 'activeProportionMap.pdf', interpolation='nearest')

 sel = getMinValue(uMatrix, bmuCoordinates, ligNames)
 activeMap, tprMap, activeProportionMap = getActiveMap(uMatrix, bmuCoordinates[sel], ligNames[sel], 'CHEMBL')
 SOMTools.plotMat(activeMap, 'activeMap_bestPose.pdf', interpolation='nearest')
 SOMTools.plotMat(tprMap, 'tprMap_bestPose.pdf', interpolation='nearest')
 SOMTools.plotMat(activeProportionMap, 'activeProportionMap_bestPose.pdf', interpolation='nearest')

 selBmuProb = getMinValue(bmuProb, bmuCoordinates, ligNames)
 selScore = getMinValue(Scores, bmuCoordinates, ligNames)

 bmus = bmuCoordinates[sel]
 uValues = []
 for bmu in bmus:
  i,j=bmu
  uValues.append(uMatrix[i,j])
 uValues = numpy.array(uValues)
 Rmin, AUCmin = getROC(ligNames[sel], 'CHEMBL', 'ROC_Min.pdf', uValues)
 Rbmu, AUCbmu = getROC(ligNames[selBmuProb], 'CHEMBL', 'ROC_BMUprob.pdf', bmuProb[selBmuProb])
 Rscores, AUCscores = getROC(ligNames[selScore], 'CHEMBL', 'ROC_Scores.pdf', Scores[selScore])
 print 'AUCbmu=%f; AUCmin=%f; AUCscores:%f'%(AUCbmu, AUCmin, AUCscores)


if __name__ == "__main__":
 main()
