#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import SOM
import threading
import progressbar
import random
import numpy
import itertools
import pickle

class adjustmentThread(threading.Thread):
 def __init__(self, k, t, trainingPhase, Map, som):
#  print threading.currentThread().getName()
  threading.Thread.__init__(self)
  self.bmus = som.findBMU(k, Map)
  self.k = k
  self.t = t
  self.trainingPhase = trainingPhase
  self.Map = Map
  self.som = som
#  self.adjustMap = 0
 def run(self):
  self.adjustMap = self.som.adjustment(self.k, self.t, self.trainingPhase, self.Map, self.bmus)

def learn(som):
 Map = som.M
 kv = range(len(som.inputvectors))
 print 'Learning for %s vectors'%len(som.inputvectors)
 for trainingPhase in range(som.number_of_phase):
  print '%s iterations'%som.iterations[trainingPhase]
  ## Progress bar
  tpn = trainingPhase + 1
  widgets = ['Training phase %s : ' % tpn, progressbar.Percentage(), progressbar.Bar(marker='=',left='[',right=']'), progressbar.ETA()]
  pbar = progressbar.ProgressBar(widgets=widgets, maxval=som.iterations[trainingPhase]-1)
  pbar.start()
  ###
  itert = itertools.chain(range(som.iterations[trainingPhase]))
  for t in itert:
   try:
    k = random.choice(kv)
    kv.remove(k)
    kSlave = random.choice(kv)
    kv.remove(kSlave)
   except IndexError:
    kv = range(len(som.inputvectors))
    k = random.choice(kv)
    kv.remove(k)
    kSlave = random.choice(kv)
    kv.remove(kSlave)
   master = adjustmentThread(k, t, trainingPhase, Map, som)
   slave = adjustmentThread(kSlave, t, trainingPhase, Map, som)
   master.start()
   slave.start()
   master.join()
   slave.join()
   diff = ((master.adjustMap - slave.adjustMap)**2).sum(axis=2).mean()
   threshold = ((master.adjustMap**2).sum(axis=2) + (slave.adjustMap**2).sum(axis=2)).mean()
   test = diff / threshold
#   print diff, threshold, diff / threshold
   if test >= 0.99:
    Map = Map + master.adjustMap + slave.adjustMap
    try:
     t = itert.next()
    except StopIteration:
     pass
   else:
    Map = Map + master.adjustMap
    kv.append(kSlave)
   pbar.update(t)
  pbar.finish()
 som.Map = Map
 MapFile = open('map_%sx%s.dat' % (som.X,som.Y), 'w')
 pickle.dump(Map, MapFile) # Write Map into file map.dat
 MapFile.close()
 if som.autoParam:
  som.epsilonFile.close()
 return som.Map
