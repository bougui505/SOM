#!/usr/bin/env python
import glob
import re

inFiles = glob.glob('*/*.out')
outFile = open('allPockets.mol2', 'w')

for fn in inFiles:
 f = open(fn)
 d = fn.split('/')[0]
 print d
 mol2f = open('%s/dp.mol2'%d)
 outFile.write('@<TRIPOS>MOLECULE\n%s\n@<TRIPOS>ATOM'%d)
 c = 0
 record = False
 motif = {}
 for l in f:
  if re.findall('^ATOM', l):
   atomType = l[12:16].strip()
   resName = l[17:20].strip()
   resId = l[22:26].strip()
   try:
    motif['%s%s'%(resName,resId)].append(atomType)
   except KeyError:
    motif['%s%s'%(resName,resId)] = [atomType]
 for l2 in mol2f:
  if re.findall('@<TRIPOS>BOND', l2):
   record = False
  if record:
   v = l2.split()
   if v[7] in motif.keys():
    if v[1] in motif[v[7]]:
     c+=1
     outFile.write('\n%s\t'%c)
     [outFile.write('%s\t'%e) for e in v[1:]]
  if re.findall('@<TRIPOS>ATOM', l2):
   record = True
 outFile.write('\n@<TRIPOS>BOND\n')
 mol2f.close()
 f.close()
