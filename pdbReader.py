class PdbReader:
 def __init__(self, pdbFile):
  self.pdbFile = pdbFile
  self.pdbLines = pdbFile.readlines()

 def isAtom(self, line, hetatm=False):
  """
  return True if the given line is an atom coordinates
  """
  recordName = line[0:6].strip()
  if hetatm:
   if recordName == 'ATOM' or recordName == 'HETATM':
    return True
   else:
    return False
  else:
   if recordName == 'ATOM':
    return True
   else:
    return False
 
 def getAtomNum(self, line):
  """
  return the AtomNum of the given line
  """
  atomNum = int(line[6:11].strip())
  return atomNum

 def getAtomName(self, line):
  """
  return the AtomName of the given line
  """
  atomName = line[12:16].strip()
  return atomName

 def getResSeq(self, line):
  """
  return the ResSeq of the given line
  """
  resSeq = int(line[22:26].strip())
  return resSeq

 def getResName(self, line):
  """
  return the ResSeq of the given line
  """
  resName = line[17:20].strip()
  return resName

 def getChainId(self, line):
  """
  return the chain Id of the given line
  """
  chainId = line[21]
  return chainId

 def getCoord(self, line):
  """
  return the coordinates for the given line
  """
  x = float(line[30:38].strip())
  y = float(line[38:46].strip())
  z = float(line[46:54].strip())
  return [x, y, z]

 def getIndices(self,selection):
  """
  return the indices of the given selection
  """
  indices = []
  count = 0
  for line in self.pdbLines:
   if self.isAtom(line):
    if selection is None or self.getAtomName(line) in selection:
     indices.append(count)
    count += 1
  return indices

 def getBBcoord(self):
  """
  return the BB coordinates : N-CA-C-O as a vector: [[N1,CA1,C1,O1],[N2,CA2,C2,O2],...]
  """
  coord = []
  i = 0
  for line in self.pdbLines:
   if self.isAtom(line):
    if self.getAtomName(line) == 'N':
     i = i + 1
     n = self.getCoord(line)
    if self.getAtomName(line) == 'CA':
     i = i + 1
     ca = self.getCoord(line)
    if self.getAtomName(line) == 'C':
     i = i + 1
     c = self.getCoord(line)
    if self.getAtomName(line) == 'O':
     i = i + 1
     o = self.getCoord(line)
    if i == 4:
     i = 0
     coord.append([n,ca,c,o])
  return coord

 def getCAcoord(self):
  """
  return the BB coordinates : N-CA-C-O as a vector: [[N1,CA1,C1,O1],[N2,CA2,C2,O2],...]
  """
  coord = []
  for line in self.pdbLines:
   if self.isAtom(line):
    if self.getAtomName(line) == 'CA':
     ca = self.getCoord(line)
     coord.append(ca)
  return coord

 def getResSeqs(self):
  resSeqs = []
  for line in self.pdbLines:
   if self.isAtom(line):
    if self.getAtomName(line) == 'CA':
     resSeqs.append( self.getResSeq(line) )
  return resSeqs

 def retrieveCoord(self, atomNumG, chainIdG):
  """
  Retrieve coordinates for the given atom number (atomNumG) and the given chainId (chainIdG)
  """
  for line in self.pdbLines:
   if self.isAtom(line):
    atomNum = self.getAtomNum(line)
    chainId = self.getChainId(line)
    if atomNum == atomNumG and chainId == chainIdG:
     coord = self.getCoord(line)
     break
  try:
   return coord
  except UnboundLocalError:
   print "No coordinate for the given atom"

 def getChainIds(self):
  """
  Give all the chain Ids
  """
  chainIds = []
  for line in self.pdbLines:
   if self.isAtom(line):
    chainId = self.getChainId(line)
    if chainId not in chainIds:
     chainIds.append(chainId)
  return chainIds
