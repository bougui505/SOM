import numpy,struct
import pdbReader

class DcdReader:
 def __init__(self, dcdfile, pdbfile, selection=None, verbose=False):
  self.file=dcdfile
  self.verbose=verbose
  self.selectDict={
      None:    None,
      "all":   None,
      "bb":    ['N','CA','C','O'],
      "backbone": ['N','CA','C','O'],
      "ca":    ['CA'],
      "trace": ['CA']
    }
  self.sel=selection if hasattr(selection,"pop") else self.selectDict[selection]
  if hasattr(pdbfile,"pdbfile"):
   self.pdbfile=pdbfile
  else:
   self.pdbfile=pdbReader.PdbReader(pdbfile if hasattr(pdbfile,"read") else open(pdbfile,'r'))
  self.array=None
  self.mean=None
  self.covariance=None
  self.correlation=None
  self.indices=self.pdbfile.getIndices(self.sel)
  self.natom=len(self.indices)
  self.nframe=0
  self.header={}
  self.load()

 def load(self,verbose=False):
  """
  Load the dcd file into self.array.
  As of yet, only CHARMM format in big-endian form is supported (with both 32b and 64b support).
  """
  if verbose or self.verbose:
   print "Loading dcd file..."

  def unpackRead(string,fd):
   """ Read the next bytes from a binary file corresponding to the given format string """
   size=struct.calcsize(string)
   return struct.unpack(string,fd.read(size))

  def skipFormat(string,fd):
   """ Skip the next bytes of the file corresponding to the size of the given format string """
   size=struct.calcsize(string)
   fd.seek(size,1)

  trj=open(self.file,'rb')
# READ HEADER
# block header size
  hsize=0
  while trj.read(1) != 'C':
   hsize+=1
   if hsize > 8:
    raise IOError('CHARMM "CORD" head flag not found. This file may not be a properly formatted dcd file.')
  if hsize == 4:
   i='i'
   self.header['long']=False
  elif hsize == 8:
   i='l'
   self.header['long']=True
  else:
   raise IOError('Funny header size. This file may not be a properly formatted dcd file.')
  trj.seek(0)
  skipFormat(i,trj)
  (self.header['flag'],)=unpackRead('4s',trj)
  if self.header['flag'] != "CORD":
   raise IOError('CHARMM "CORD" head flag not found. This file may not be a properly formatted dcd file.')
  self.header['consts']=unpackRead('20i',trj)
  nframe=self.header['consts'][0]
  if nframe != 0:
   self.nframe=nframe
#  Loading nframe frames
   self.box=[[0.,0.,0.,0.,0.,0.] for x in range(nframe)]
   skipFormat(i,trj)
#  second block
   skipFormat(i,trj)
   (self.header['descn'],)=unpackRead('i',trj)
   descn=self.header['descn']
   self.header['desc']=unpackRead(descn*'80s',trj)
   skipFormat(i,trj)
#  third block
   skipFormat(i,trj)
   (self.header['natom'],)=unpackRead('i',trj)
   natom=self.header['natom']
   self.array=numpy.zeros((nframe,3*self.natom))
   xindices=range(0,3*self.natom,3)
   yindices=range(1,3*self.natom,3)
   zindices=range(2,3*self.natom,3)
   skipFormat(i,trj)
   temparray=numpy.zeros((natom))
#    READ COORDS
   for frame in range(nframe):
    if self.header['consts'][10]:
     skipFormat(i,trj)
     self.box[frame]=list(unpackRead('6d',trj))
     skipFormat(i,trj)
    skipFormat(i,trj)
    temparray[:]=list(unpackRead(str(natom)+'f',trj))
    self.array[frame,xindices]=temparray[self.indices]
    skipFormat(i,trj)
    skipFormat(i,trj)
    temparray[:]=list(unpackRead(str(natom)+'f',trj))
    self.array[frame,yindices]=temparray[self.indices]
    skipFormat(i,trj)
    skipFormat(i,trj)
    temparray[:]=list(unpackRead(str(natom)+'f',trj))
    self.array[frame,zindices]=temparray[self.indices]
    skipFormat(i,trj)

 def __len__(self):
  return self.natom

 def getMean(self):
  """
  Compute the mean coordinates of all frames.
  """
  if self.verbose:
   print "Computing mean structure..."
  self.mean=numpy.mean(self.array,axis=0)
  return self.mean

 def getCovariance(self):
  """
  Compute the covariance matrix of the coordinates array.
  This compute the mean if not done before.
  """
  if self.mean is None:
   self.getMean()
  if self.verbose:
   print "Computing covariance matrix..."
  M=self.array-self.mean[None]
  self.covariance=numpy.dot(M.T,M)
  return self.covariance

 def getCorrelation(self):
  """
  Compute the correlation matrix of the coordinates array.
  This compute the covariance matrix if not done before.
  """
  if self.covariance is None:
   self.getCovariance()
  if self.verbose:
   print "Computing correlation matrix..."

#  stdev=numpy.sqrt(numpy.diag(self.covariance))
#  self.correlation=numpy.empty(self.covariance.shape)
#  for item,cov in numpy.ndenumerate(self.covariance):
#   x,y=item
#   self.correlation[x,y]=cov/(stdev[x]*stdev[y])
  stdev=numpy.sqrt(numpy.diag(self.covariance))
  stdevmat=numpy.outer(stdev,stdev)
  self.correlation=self.covariance/stdevmat
  return self.correlation

 def align(self,template="mean"):
  """
  Align all the structures of the trajectory to a single template and center everything on 0,0,0
  This uses Kabsch algorithm (Kabsch, Wolfgang, A solution of the best rotation to relate two sets of vectors. 1976, Acta Crystallographica 32:922).
  """
  if template == "mean":
   if self.mean is None:
    self.getMean()
   tar=self.mean
  else:
   tar=self.array[template,:]
  if self.verbose:
   print "Aligning..."
  xi=range(0,3*self.natom,3)
  yi=range(1,3*self.natom,3)
  zi=range(2,3*self.natom,3)
  tempar=numpy.empty((self.natom,3))
  tempar[:,0]=tar[xi]-numpy.mean(tar[xi])[None]
  tempar[:,1]=tar[yi]-numpy.mean(tar[yi])[None]
  tempar[:,2]=tar[zi]-numpy.mean(tar[zi])[None]
  targar=numpy.empty((self.natom,3))
  for frame in range(self.nframe):
   if frame == template:
    self.array[frame,xi]=tempar[:,0]
    self.array[frame,yi]=tempar[:,1]
    self.array[frame,zi]=tempar[:,2]
    continue
   targar[:,0]=self.array[frame,xi]-numpy.mean(self.array[frame,xi])
   targar[:,1]=self.array[frame,yi]-numpy.mean(self.array[frame,yi])
   targar[:,2]=self.array[frame,zi]-numpy.mean(self.array[frame,zi])
   V,s,tW=numpy.linalg.svd(numpy.dot(targar.T,tempar))
#  change sign of the last vector if needed to assure similar orientation of bases
   if numpy.linalg.det(V)*numpy.linalg.det(tW) < 0:
    V[:,2]*=-1
   rot=numpy.dot(V,tW)
   targrot=numpy.dot(targar,rot)
   self.array[frame,xi]=targrot[:,0]
   self.array[frame,yi]=targrot[:,1]
   self.array[frame,zi]=targrot[:,2]
  if template=="mean":
    self.mean[xi]=tempar[:,0]
    self.mean[yi]=tempar[:,1]
    self.mean[zi]=tempar[:,2]

def rmsd(a,b):
# Compute rmsd between two arrays of the same shape
 return numpy.square(numpy.mean((a-b)**2))
