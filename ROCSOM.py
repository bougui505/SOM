#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import SOM
import numpy
import re
import matplotlib.pyplot
import progressbar
import cPickle

# Import Psyco if available
#try:
# import psyco
# psyco.full()
#except ImportError:
# print 'psyco not available'
# pass

class ROCSOM:
 def __init__(self, MapFile = None, som=None):
  self.MapFile = MapFile
  if som == None:
   self.som = SOM.SOM(filename='ALL_vectors')
  else:
   self.som = som
  inputFile = open(self.som.AuPosSOM_inputFile, 'r')
  lines = inputFile.readlines()
  inputFile.close()
  test = 2
  KL = []
  for line in lines:
   if re.findall('<Known_ligands>', line):
        test = test - 1
   if test == 1 and not re.findall('<Known_ligands>', line) and len(line.split('#')) != 2:
        KL.append(line.strip())
  self.KL = KL # list of known ligands
  test = 2
  UL = []
  for line in lines:
   if re.findall('<Unknown_ligands>', line):
        test = test - 1
   if test == 1 and not re.findall('<Unknown_ligands>', line) and len(line.split('#')) != 2:
        UL.append(line.strip())
  self.UL = UL # list of unknown ligands
  if MapFile == None:
   self.som.learn()
  else:
   MapFileFile = open(MapFile, 'r')
   self.som.Map = cPickle.load(MapFileFile)
   MapFileFile.close()
  self.BMUs = []
  for k in range(len(self.som.inputvectors)):
   self.BMUs.append(self.som.findBMU(k, self.som.Map))
  
 def roc(self, bestSeAnalysis = False, findBMUs = False, jobIndex=''):
  """
  Find the best threshold according to ROC analysis. The best clustering is the clustering with minimal gamma value. If bestSeAnalysis = True the best clustering corresponds to the clustering with the highest sensibility with maximal specificity.
  """
  step = 0.1
  step_value = step
  Tv = numpy.arange(0,1+step,step)
  cardK = float(len(self.KL))
  cardU = float(len(self.UL))
  X = []
  Y = []
  ## Progress bar
  widgets = ['ROC best clustering', progressbar.Percentage(), progressbar.Bar(marker='=',left='[',right=']'), progressbar.ETA()]
  pbar = progressbar.ProgressBar(widgets=widgets, maxval=1+step)
  pbar.start()
  ###
  gammaROCmin = 2**0.5
  SeOptMax = 0
  gammaROCmin2 = 2**0.5
  T = 0
  uniqueGroupTest = 0
  self.som.clusterDiscovery(self.som.Map,0)
  if findBMUs:
   self.som.calibration(self.som.Map, self.som.clustersMap, BMUs = None, name = True)
  else:
   self.som.calibration(self.som.Map, self.som.clustersMap, BMUs = self.BMUs, name = True)
  E = self.som.nameClusters
  test = len(E)
  emergencyStop = False
  while 1:
   T = T + step
   self.som.clusterDiscovery(self.som.Map,T)
   self.som.calibration(self.som.Map, self.som.clustersMap, self.BMUs, name = True)
   E = self.som.nameClusters
   ngrp = len(E)
   if (ngrp == 1 and ngrp == test):
        uniqueGroupTest = uniqueGroupTest + 1
   if uniqueGroupTest == 1:
        break
   if (ngrp == test - 1 or T == 0):
        CIds = E.keys()
        gammaMin = 2**0.5
        gammaMin2 = 2**0.5
        SeMax = 0
        for CId in CIds:
         Se = float(len([e for e in E[CId] if e in self.KL])) / cardK
         Sp = 1 - float(len([e for e in E[CId] if e in self.UL])) / cardU
         gamma = ((1-Sp)**2 + (1-Se)**2)**0.5
         if bestSeAnalysis == False:
          if gamma < gammaMin:
           gammaMin = gamma
           SeOpt = Se
           SpOpt = Sp
           CIdOpt = CId
         elif bestSeAnalysis:
          if Se > SeMax:
           SeMax = Se
           gammaMin2 = gamma
          if (Se == SeMax and gamma <= gammaMin2):
           gammaMin2 = gamma
           SeOpt = Se
           SpOpt = Sp
           CIdOpt = CId
        X.append(1-SpOpt)
        Y.append(SeOpt)
        gammaROC = ((1-SpOpt)**2 + (1-SeOpt)**2)**0.5
        if bestSeAnalysis == False:
         if gammaROC < gammaROCmin:
          gammaROCmin = gammaROC
          Topt = T
          Xopt = 1-SpOpt
          Yopt = SeOpt
          Copt = E[CIdOpt] # list of clustered compounds <<============================
        elif bestSeAnalysis:
         if SeOpt > SeOptMax:
          SeOptMax = SeOpt
          gammaROCmin2 = gammaROC
          if (SeOpt == SeOptMax and gammaROC <= gammaROCmin2):
           gammaROCmin2 = gammaROC
           Topt = T
           Xopt = 1-SpOpt
           Yopt = SeOpt
           Copt = E[CIdOpt] # list of clustered compounds <<============================
        pbar.update(T)
        test = ngrp
        if T == 0:
         emergencyStop = True
   elif ngrp == test:
        step = step_value
        test = ngrp
   else:
        T = T - step
        step = step / 10
   if emergencyStop:
        break
  pbar.finish()
  print 'Best clustering for Threshold : %s' % Topt
  print 'Sensitivity : %s' % Yopt
  self.Se = Yopt
  print 'Specificity : %s' % (1-Xopt)
  self.Sp = 1-Xopt
  #matplotlib.pyplot.axis([0,1,0,1])
  #matplotlib.pyplot.plot(X,Y,'+')
  #matplotlib.pyplot.plot(X,Y)
  #matplotlib.pyplot.plot([Xopt],[Yopt],'ro')
  #matplotlib.pyplot.plot([0,1],[0,1],'+')
  #matplotlib.pyplot.show()
  if self.MapFile != None:
   XY = [[0,1],[0,1]]
   XY[0].extend(X)
   XY[1].extend(Y)
   if jobIndex == '':
    rocfile = open('roc_%s.xy'%self.MapFile, 'w')
   else:
    rocfile = open('roc_som%s.dat'%jobIndex, 'w')
   cPickle.dump(XY,rocfile)
   rocfile.close()
  self.Topt = Topt
  self.Copt = Copt
  #return self.Topt
  
 def rocNlearn(self):
  self.som = SOM.SOM(filename='ALL_vectors') # instance class SOM
  cardK = len(self.KL)
  cardU = len(self.UL)
  step = cardK # step for the ROC analysis. Typically the number of known molecules
  numOfMol = range(0,cardU,step) # number of molecules for learning
   ## Progress bar
  widgets = ['ROC analysis', progressbar.Percentage(), progressbar.Bar(marker='=',left='[',right=']'), progressbar.ETA()]
  pbar = progressbar.ProgressBar(widgets=widgets, maxval=numOfMol[-1])
  pbar.start()
  ###
  XY = [[0,1],[0,1]]
  gammaMin = 2**0.5
  gammaMin2 = 2**0.5
  SeMax = 0
  for n in numOfMol:
   self.som = SOM.SOM(filename='ALL_vectors') # instance class SOM for taking all vectors
   self.som.makeSubInputvectors(n)
   #print som.inputnames
   self.som.learn()
   self.som = SOM.SOM(filename='ALL_vectors') # instance class SOM for taking all vectors for calibration
   self.__init__(MapFile = 'map_%sx%s.dat' % (self.som.X,self.som.Y)) # for reloading map
   self.roc()
   XY[0].append(1-self.Sp)
   XY[1].append(self.Se)
   gamma = ((1-self.Sp)**2 + (1-self.Se)**2)**0.5
   if gamma < gammaMin:
        bestMap = self.som.Map
        gammaMin = gamma
        bestSp = self.Sp
        bestSe = self.Se
        bestn = n
   if (self.Se >= SeMax):
        SeMax = self.Se
        if gamma < gammaMin2:
         gammaMin2 = gamma
         print 'Best sensitivity : %s'%SeMax
         bestSeMap = self.som.Map
   pbar.update(n)
  rocfile = open('roc.xy', 'w')
  cPickle.dump(XY,rocfile)
  rocfile.close()
  bestMapFile = open('bestMap.dat', 'w')
  cPickle.dump(bestMap,bestMapFile)
  bestMapFile.close()
  bestSeMapFile = open('bestSeMap.dat', 'w')
  cPickle.dump(bestSeMap,bestSeMapFile)
  bestSeMapFile.close()
  
  print 'Best map for learning with all known ligands and %s molecules from the database' %bestn
  print 'Best Sensitivity : %s' %bestSe
  print 'Best Specificity : %s' %bestSp
  self.plotROC('roc.xy')
  pbar.finish()
  
 def plotROC(self, rocfilename, verbose=False):
  rocfile = open(rocfilename, 'r')
  XY = cPickle.load(rocfile)
  if verbose:
   for i in range(len(XY[2])):
        print '%s ; %s ; %s ; %s'%(XY[0][i],XY[1][i],XY[2][i],XY[3][i])
  rocfile.close()
  matplotlib.pyplot.grid()
  matplotlib.pyplot.plot([0,1],[0,1])
  matplotlib.pyplot.plot(XY[0],XY[1],'o')
  matplotlib.pyplot.xlabel('1-Sp (background noise)')
  matplotlib.pyplot.ylabel('Se (detected signal)')
  matplotlib.pyplot.show()
  
 def iterativeTrees(self):
  """
  Make iterative trees to reduce the number of possible active compounds
  """
  eMol = 0
  eMolOld = 1
  self.Copt = []
  while eMol != eMolOld:
   eMolOld = eMol
   Copt_old = self.Copt
   self.som.learn()
   self.roc(bestSeAnalysis = True, findBMUs = True)
   self.som.makeNewInputvectors(self.Copt)
   #self.BMUs = []
   #for k in range(len(self.som.inputvectors)):# for new BMU 
        #self.BMUs.append(self.som.findBMU(k, self.som.Map))
   if self.Se != 1:
        self.Copt = Copt_old
        break
   eMol = len(self.Copt)
   print eMol
   #print self.Copt
   #print self.som.inputnames
   
  print self.Copt
  
#roc = ROCSOM(MapFile = 'map_5x4.dat')
#roc = ROCSOM()
#Topt = roc.roc()
#Map = roc.som.Map
#clusterMap = roc.som.clusterDiscovery(Map, Topt)
#dataClusters = roc.som.calibration(Map, clusterMap)
#dataLoosenessMat = roc.som.dataLooseness(dataClusters, clusterMap)
#roc.som.distmapPlot(0,Map)
#clusterLoosenessMap = roc.som.clusterLooseness(clusterMap, Map)
#roc.som.mapPlot(clusterLoosenessMap)
