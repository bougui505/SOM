#!/usr/bin/env python
# -*- coding: UTF8 -*-

import os,sys,struct
import numpy
import array
import itertools

class Structure(object):
 
 def __init__(self,path=None,type='pdb',array=None,hetatm=True):
  self.atoms=array
  self.path=path
  self.type=type
  if (self.atoms is None) and (not self.path is None):
   self.load(self.path,self.type)
 
 def load(self,file,stype,hetatm=True):
  self.atoms=[]
  app=self.atoms.append
  if stype == 'pdb':
   fd=file if hasattr(file,"read") else open(file,'r')
   c=1
   index=-1
   for line in fd:
    c+=1
    if line[:3] == 'END':
     break
    if not line[:6] in ['ATOM  ','HETATM']:
     continue
    index+=1
    count=int(line[6:11])
    aname=line[12:16].strip()
    rname=line[17:20].strip()
    chain=line[21]
    rnum=int(line[22:26])
    x=float(line[30:38])
    y=float(line[38:46])
    z=float(line[46:54])
    beta=float(line[60:66])
    segid=line[72:76].strip()
    app((index,count,aname,rname,chain,rnum,(x,y,z),beta,segid))
   self.atoms=numpy.asarray(self.atoms,dtype=[('index','<i4'), ('count', '<i4'), ('atomname', '|S4'), ('resname', '|S3'), ('chain', '|S1'), ('resid', '<i2'), ('coord', '<f4', (3,)), ('beta', '<f4'), ('segid', '|S4')])
   fd.close()
  
  elif stype == 'psf':
   raise ValueError('psf structure file type is not supported yet.')
 
 def getSameAs(self,atoms,field):
  """
  Return all indices of atoms whose 'field' is the same as the specified atom's
  """
  if hasattr(atoms,'__getitem__'):
   sel=numpy.zeros(self.atoms['index'].shape,dtype="bool")
   for atom in atoms:
    sel=numpy.logical_or(sel,self.getSameAs(atom,field))
   return sel
  else:
   what=self.atoms[field][atom]
   return (self.atoms[field] == what)
 
 def getFieldValue(self,atom,field):
  return self.atoms[field][atom]
 
 def write(self,file=None,type=None):
  if type is None:
   type=self.type
  if file != None:
   if hasattr(file,"write"):
    f=file
    wfunc=f.write
    wendl='\n'
   else:
    f=open(file,'w+')
    wfunc=f.write
    wendl='\n'
  else:
   wfunc=print_func
   wendl=''
  
  if type=='pdb':
   for index,count,aname,rname,chain,rnum,coord,beta,segid in self.atoms:
    x,y,z=coord
    wfunc("ATOM  %(natom) 5d %(aname)s %(rname)s %(chain)s%(rnum) 4i    %(x) 8.3f%(y) 8.3f%(z) 8.3f  1.00%(beta)6.2f%(wendl)s"%{ \
     'natom': count, \
     'aname': aname, \
     'rname': rname, \
     'chain': chain, \
     'rnum':  rnum,  \
     'x':x,          \
     'y':y,          \
     'z':z,          \
     'beta':beta,    \
     'wendl':wendl   \
     })
   wfunc('END')
  
  elif type=='grd':
   raise NotImplemented()
  
  if file != None:
   f.close()
 
 def getSelectionIndices(self,selection,field):
  """
  Return a mask of all indices whose field value is in selection.
  If selection is None, 
  """
  if selection is None:
   return numpy.ones(self.atoms.shape,dtype="bool")
  else:
   return numpy.asarray([ value in selection for value in self.atoms[field] ])
 
 def __getitem__(self,item):
  return self.atoms.__getitem__(item)

class Trajectory(object):
 def __init__(self, dcdfile=None, struct=None, array=None, selection=None, selectionmask=None, verbose=False, nframe=0):
  self.file = dcdfile
  self.verbose = verbose
  self.selectDict = {
      None:    (None,'atomname'),
      "all":   (None,'atomname'),
      "bb":    (['N','CA','C','O'],'atomname'),
      "backbone": (['N','CA','C','O'],'atomname'),
      "ca":    (['CA'],'atomname'),
      "trace": (['CA'],'atomname')
    }
  self.sel,self.selfield=selection if hasattr(selection,"__getitem__") else self.selectDict[selection]
  self.natom = 0
  if not struct is None:
   if hasattr(struct,"atoms"):
    self.struct=struct
   else:
    self.struct=Structure(struct if hasattr(struct,"read") else open(struct,'r'))
   self.indices = self.struct.getSelectionIndices(self.sel,self.selfield) if selectionmask is None else selectionmask
   self.natom=self.indices.sum()
  self.array=None
  self.mean=None
  self.covariance=None
  self.correlation=None
  self.nframe=0
  self.header={'long': False, }
  if not array is None:
   self.array = array
   self.nframe, self.natom = self.array.shape[0], self.array.shape[1]/3
  self.header['flag'] = 'CORD'
  self.header['descn'] = 1
  self.header['desc'] = ('dcd IO.py'+'\x00'*71,)
  self.header['consts'] = (self.nframe,1,1,self.nframe,0,0,0,1,0,1017614563,0,0,0,0,0,0,0,0,0,35)
  self.header['natom'] = self.natom
  elif not self.file is None:
   self.load(nframe=nframe,verbose=verbose)

 def load(self,nframe=0,verbose=False):
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
  self.nframe=self.header['consts'][0]
  if nframe > 0:
   self.nframe=nframe
  if self.nframe != 0:
#  Loading nframe frames
   self.box=[[0.,0.,0.,0.,0.,0.] for x in range(self.nframe)]
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
   self.array=numpy.zeros((self.nframe,3*self.natom))
   #print natom,self.natom,self.array.shape
   xindices=range(0,3*self.natom,3)
   yindices=range(1,3*self.natom,3)
   zindices=range(2,3*self.natom,3)
   skipFormat(i,trj)
   temparray=numpy.zeros((natom))
   #   print self.indices,xindices,self.natom
   #   READ COORDS
   for frame in range(self.nframe):
    if self.header['consts'][10]:
     skipFormat(i,trj)
     self.box[frame]=list(unpackRead('6d',trj))
     skipFormat(i,trj)
    skipFormat(i,trj)
    temparray[:]=list(unpackRead(str(natom)+'f',trj))
    #print self.array[frame,xindices].shape,temparray[self.indices].shape
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
  else:
   raise ValueError("No frame number found in header nor as parameter. Cannot load trajectory. Please specify a correct number of frames using the 'nframe' parameter.")

 def write(self,path,type='dcd',bkeep=False):
  if type == 'pdb': # FIXME : pdb seems to be a bit broken, need to work on it.
   if self.struct is None:
    raise ValueError('Impossible to write a pdb file without loading a structure first. Load a structure first using loadStruct().')
   if path != 'stdout':
    f=open(path,'w+')
    wfunc=f.write
    wendl='\n'
   else:
    wfunc=print_func
    wendl=''
   
   nframe,natom=self.array.shape
   for frame in range(nframe):
    wfunc('MODEL % 8d'%(frame+1))
    for coord in self.array[frame,:]:
     count,aname,rname,chain,rnum,oricoords,b,segid=struct[count] # FIXME : count !??
     orix,oriy,oriz=oricoords
     beta=b if bkeep else 0.
     x,y,z=coord
     # FIXME : shouldn't work properly now because of the "stripped" fields in Structure.load()
     wfunc("ATOM  %(natom) 5d %(aname)s %(rname)s %(chain)s%(rnum) 4i    %(x) 8.3f%(y) 8.3f%(z) 8.3f  1.00%(beta)6.2f     %(segid)s%(wendl)s"%{ \
      'natom':count, \
      'aname':aname, \
      'rname':rmame, \
      'chain':chain, \
      'rnum':rnum,   \
      'x':x, \
      'y':y, \
      'z':z, \
      'beta':beta, \
      'segid': segid, \
      'wendl':wendl \
      })
    wfunc('ENDMDL'+wendl)
   if file != None:
    f.close()
  
  elif type == 'dcd':
   
   def packWrite(string,fd,*args):
    """
    Write arguments to a file using the given format string
    """
    fd.write(struct.pack(string,*args))
   
   def sizeWrite(strg,fd,long=False):
    """
    Write the byte size of the given format string to a file.
    Useful to write binary Fortran-formatted file.
    """
    if long:
     i='l'
    else:
     i='i'
    fd.write(struct.pack(i,struct.calcsize(strg)))
   
   fd=open(path,'wb')
   long=self.header['long']
   
   nframe,natom=self.array.shape
   # WRITE HEADER
   sizeWrite('4s20i',fd,long)
   packWrite('4s',fd,'CORD')
   array.array('i',self.header['consts']).tofile(fd)
   sizeWrite('4s20i',fd,long)
   descn=self.header['descn']
   sizeWrite('i'+descn*'80s',fd,long)
   packWrite('i',fd,descn)
   for desc in self.header['desc']:
    packWrite('80s',fd,desc)
   sizeWrite('i'+descn*'80s',fd,long)
   sizeWrite('i',fd,long)
   natom=self.header['natom']
   packWrite('i',fd,natom)
   sizeWrite('i',fd,long)
   
   xindices=range(0,3*self.natom,3)
   yindices=range(1,3*self.natom,3)
   zindices=range(2,3*self.natom,3)
   
   size_str=str(natom)+'f'
   # WRITE TRAJ
   for frame in range(nframe):
    if self.header['consts'][10]:
     sizeWrite('6d',fd,long)
     array.array('d',self.box[frame]).tofile(fd)
     sizeWrite('6d',fd,long)
    sizeWrite(size_str,fd,long)
    array.array('f',self.array[frame,xindices]).tofile(fd)
    sizeWrite(size_str,fd,long)
    sizeWrite(size_str,fd,long)
    array.array('f',self.array[frame,yindices]).tofile(fd)
    sizeWrite(size_str,fd,long)
    sizeWrite(size_str,fd,long)
    array.array('f',self.array[frame,zindices]).tofile(fd)
    sizeWrite(size_str,fd,long)
   # CLOSE FILE
   fd.truncate()
   fd.close()
  
  else:
   raise ValueError('type cannot be of type '+type)
  


 def __len__(self):
  return self.nframe

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

 def getDistance(self):
  xindices=range(0,3*self.natom,3)
  yindices=range(1,3*self.natom,3)
  zindices=range(2,3*self.natom,3)
  X=self.array[:,xindices]
  Y=self.array[:,yindices]
  Z=self.array[:,zindices]
  coord=numpy.asarray([X,Y,Z])
  distmat=numpy.zeros((self.natom,self.natom,self.nframe),dtype="float32")
  couple=itertools.combinations(xrange(self.natom),2)
  for i,j in couple:
   val=numpy.sqrt(numpy.sum((coord[:,:,i]-coord[:,:,j])**2,axis=0))
   distmat[i,j]=val
   distmat[j,i]=val
  self.distmat=distmat
  return distmat


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

