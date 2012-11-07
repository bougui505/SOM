#!/usr/bin/env python
import matplotlib.pyplot
import IO
import numpy
import itertools
import scipy.spatial
import scipy.interpolate
import scipy.stats
import SOM
import glob
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
import scipy.ndimage.measurements
import sys
import tarfile
import os

def readRestraints(fileName='restraintsList'):
 f = open(fileName)
 restraints = []
 for l in f:
  list = l.split()
  r1 = (int(list[0]), list[1])
  r2 = (int(list[2]), list[3])
  restraints.append([r1, r2])
 return restraints

def restraintsPotential(matrix, r0, r1, r2, k=1):
 mask1 = 1 - (matrix < r0)
 mask2 = 1 - numpy.logical_and( (matrix >= r0), (matrix <= r1) )
 mask3 = 1 - numpy.logical_and( (matrix > r1), (matrix <= r2) )
 mask4 = 1 - (matrix > r2)
 pMap1 = numpy.ma.filled(.5*k*(numpy.ma.masked_array(matrix, mask1) - r0)**2, 0)
 pMap2 = numpy.ma.filled(0*numpy.ma.masked_array(matrix, mask2), 0)
 pMap3 = numpy.ma.filled(.5*k*(numpy.ma.masked_array(matrix, mask3) - r1)**2, 0)
 pMap4 = numpy.ma.filled(.5*k*(r2-r1)*(2*numpy.ma.masked_array(matrix, mask4) - r2 - r1), 0)
 return pMap1 + pMap2 + pMap3 + pMap4

def energyMaps(som, energyFileName, frameMasks):
 energies = numpy.genfromtxt(energyFileName)[:frameMasks.shape[1]]
 energyMatrix = numpy.reshape((frameMasks * energies), (som.X,som.Y,frameMasks.shape[1]))
 return numpy.ma.masked_array(energyMatrix, energyMatrix == 0)

def plotMat(matrix, outFileName, colorbar=True, cbarTicks = None, cmap = None, contour=False, clabel = True, texts=None, interpolation=None, vmin = None, vmax = None):
 matplotlib.pyplot.clf()
 matplotlib.pyplot.imshow(matrix, cmap = cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
 if colorbar:
  if cbarTicks == None:
   matplotlib.pyplot.colorbar()
  else:
   matplotlib.pyplot.colorbar(ticks = cbarTicks)
 if contour:
  CS = matplotlib.pyplot.contour(matrix, colors = 'k')
  if clabel:
   matplotlib.pyplot.clabel(CS, inline=1, fontsize=10)
 if texts != None:
  for text in texts:
   matplotlib.pyplot.text(text[0][1], text[0][0], text[1], horizontalalignment='center', verticalalignment='center', fontsize = 10, bbox=dict(facecolor='red', alpha=0.5))
 matplotlib.pyplot.savefig(outFileName)

def plotHistogram(vector, outFileName, title='', xlabel='', ylabel = '', cumulative=0, range=None, bins = 10):
 matplotlib.pyplot.clf()
 n, bins, patches = matplotlib.pyplot.hist(vector,bins=bins,range=range,cumulative=cumulative)
 for i in range(n.shape[0]):
  matplotlib.pyplot.text((bins[i]+bins[i+1])/2, n[i], n[i])
 matplotlib.pyplot.grid()
 matplotlib.pyplot.xlabel(xlabel)
 matplotlib.pyplot.ylabel(ylabel)
 matplotlib.pyplot.title(title)
 matplotlib.pyplot.savefig(outFileName)

def kohonenMap2D(map, inputVector, cosine = False):
 kohonenPlot = []
 if len(map.shape) == 4:
  if map.shape[3] == 3:
   kohonenPlot = threeDspaceDistance(inputVector, map)
 else:
  for line in map:
   for v in line:
    if cosine:
     kohonenPlot.append(scipy.spatial.distance.cosine(v,inputVector))
    else:
     kohonenPlot.append(scipy.spatial.distance.euclidean(v,inputVector))
 return numpy.reshape(kohonenPlot, map.shape[0:2])

def threeDspaceDistance(m,somMap):
 sum1axis = len(somMap.shape) - 1
 sum2axis = sum1axis - 1
 return numpy.sqrt(((m - somMap)**2).sum(axis=sum1axis)).sum(axis=sum2axis)

def allKohonenMap2D(map, inputVectors, metric='euclidean'):
 if len(map.shape) == 4:
  if map.shape[3] == 3:
   allMaps = numpy.zeros((inputVectors.shape[0], map.shape[0]*map.shape[1]))
   n = inputVectors.shape[0]
   for k in range(n):
    sys.stdout.write('%s/%s'%(k,n))
    sys.stdout.write('\r')
    sys.stdout.flush()
    allMaps[k,:]= threeDspaceDistance(inputVectors[k], numpy.reshape(map, (map.shape[0]*map.shape[1], map.shape[2], 3)))/inputVectors.shape[1]
 else:
  allMaps = scipy.spatial.distance.cdist(inputVectors, numpy.reshape(map, (map.shape[0]*map.shape[1], map.shape[2])), metric).transpose()
 sys.stdout.write('\n')
 return allMaps

def findMinRegion(matrix, scale=1):
# min = numpy.min(matrix)
# ptp = numpy.ptp(matrix)/20
# return matrix < min + ptp
 return matrix < numpy.mean(matrix) - scale*numpy.std(matrix)

def findMaxRegion(matrix, scale=1):
 return matrix > numpy.mean(matrix) + scale*numpy.std(matrix)

def findMinRegionAll(matrix, scale = 1):
# mins = numpy.min(matrix, axis=0)
# ptps = numpy.ptp(matrix, axis = 0)/20
# return matrix < mins + ptps
 return matrix < numpy.mean(matrix, axis=0) - scale*numpy.std(matrix, axis=0)

def findMinAll(matrix):
 mins = numpy.min(matrix, axis=0)
 return matrix == mins

def writeClusters(bmuMats, clusters, outFileName='clusters.txt'):
 clist = numpy.dot(bmuMats.T , numpy.reshape(clusters, (clusters.shape[0]*clusters.shape[1],)))
 clustersDict = {}
 for i in range(1,clist.max() + 1):
  clustersDict[i] = []
 c = 0
 for e in clist:
  if e != 0:
   clustersDict[e].append(c)
  c = c+1
 outFile = open(outFileName, 'w')
 for i in range(1, clist.max() + 1):
  for e in clustersDict[i]:
   outFile.write('%s '%e)
  outFile.write('\n')
 return clustersDict

def writePdbs(clustersDict):
 for key in clustersDict.keys():
  outPdbFile = open('cluster_%s.pdb'%key, 'w')
  for fn in clustersDict[key]:
   infile = open('Pdbs/frame%s.pdb'%fn)
   for l in infile:
    outPdbFile.write(l)
  outPdbFile.close()


def detect_peaks(image):
 """
 from: http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
 Takes an image and detect the peaks usingthe local maximum filter.
 Returns a boolean mask of the peaks (i.e. 1 when
 the pixel's value is the neighborhood maximum, 0 otherwise)
 """

 # define an 8-connected neighborhood
 neighborhood = generate_binary_structure(2,2)

 #apply the local maximum filter; all pixel of maximal value 
 #in their neighborhood are set to 1
 local_max = maximum_filter(image, footprint=neighborhood)==image
 #local_max is a mask that contains the peaks we are 
 #looking for, but also the background.
 #In order to isolate the peaks we must remove the background from the mask.

 #we create the mask of the background
 background = (image==0)

 #a little technicality: we must erode the background in order to 
 #successfully subtract it form local_max, otherwise a line will 
 #appear along the background border (artifact of the local maximum filter)
 eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

 #we obtain the final mask, containing only peaks, 
 #by removing the background from the local_max mask
 detected_peaks = local_max - eroded_background

 return detected_peaks

def cmap_discretize(cmap, N):
 """Return a discrete colormap from the continuous colormap cmap.
 
     cmap: colormap instance, eg. cm.jet. 
     N: Number of colors.
 
 Example
     x = resize(arange(100), (5,100))
     djet = cmap_discretize(cm.jet, 5)
     imshow(x, cmap=djet)
 """

 cdict = cmap._segmentdata.copy()
 # N colors
 colors_i = numpy.linspace(0,1.,N)
 # N+1 indices
 indices = numpy.linspace(0,1.,N+1)
 for key in ('red','green','blue'):
  # Find the N colors
  D = numpy.array(cdict[key])
  I = scipy.interpolate.interp1d(D[:,0], D[:,1])
  colors = I(colors_i)
  # Place these colors at the correct indices.
  A = numpy.zeros((N+1,3), float)
  A[:,0] = indices
  A[1:,1] = colors
  A[:-1,2] = colors
  # Create a tuple for the dictionary.
  L = []
  for l in A:
   L.append(tuple(l))
  cdict[key] = tuple(L)
 # Return colormap object.
 return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def minPath(matrix, gradThreshold):
 if type(matrix) == numpy.ma.core.MaskedArray:
  matrix = matrix.filled()
 X,Y = matrix.shape
 iStart,jStart = scipy.ndimage.measurements.minimum_position(matrix)
 i,j = iStart,jStart
 outPath = [(iStart,jStart)]
 path = [(iStart,jStart)]
 pathMax = []
 start = True
 maxPosition = scipy.ndimage.measurements.maximum_position(matrix)
 ic = itertools.count()
 c = ic.next()
 r = 0
 grads = []
 zScores = []
 grad = 0
 clusterPathMat = numpy.zeros(matrix.shape, dtype=int)
 cIndex = 1
 while (i,j) != maxPosition or start:
  c = ic.next()
  start = False
  labels = numpy.zeros(matrix.shape, dtype=int)
  for x in [i-1,i,i+1]:
   for y in [j-1,j,j+1]:
    if (x%X,y%Y) not in path:
     labels[x%X,y%Y] = 1
  if labels.sum() != 0:
   reverse = False
   ip,jp = i,j
   i,j = scipy.ndimage.measurements.minimum_position(matrix, labels = labels)
   gradp = grad
   grad = matrix[i,j] - matrix[ip,jp]
   if grads != []:
    gradMax = max(numpy.abs(grads))
   else:
    gradMax = 9999.
   path.append((i,j))
   zScore = (grad - numpy.mean(grads))/numpy.std(grads)
   zScores.append(zScore)
   grads.append(grad)
   sys.stdout.write('path: grad=%10.2f ; reverse: %10.0f; clusters #: %10.0f'%(grad,r,cIndex))
   sys.stdout.write('\r')
   sys.stdout.flush()
   if grad > gradThreshold:
    cIndex = cIndex + 1
   clusterPathMat[i,j] = cIndex
  else:
   if reverse:
    r = r - 1
   else:
    r = -1
   reverse = True
   try:
    i,j = path[r]
   except IndexError:
    print 'BREAK !!!!!'
    break
  if (i,j) not in outPath:
   outPath.append((i,j))
 sys.stdout.write('\ndone\n')
 clusterPathMat = numpy.ma.masked_array(clusterPathMat, clusterPathMat==0)
 return outPath, clusterPathMat, grads

def getEMmapCorrelationMatrix(correlations, allMins, som):
 correlationsMatrix = allMins * correlations
 correlationsMatrix = numpy.ma.masked_array(correlationsMatrix, correlationsMatrix == 0.)
 meanCorrelationMatrix = numpy.reshape(correlationsMatrix.mean(axis=1),(som.X, som.Y))
 return meanCorrelationMatrix

def getUmatrix(Map,toricMap=True):
 X,Y,cardinal = Map.shape
 distanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Map.reshape(X*Y,cardinal)))
 uMatrix = numpy.zeros((X,Y))
 if toricMap:
  for i in range(X):
   for j in range(Y):
    iRef = numpy.ravel_multi_index((i%X,j%Y),(X,Y))
    iS=[(i-1)%X,i%X,(i+1)%X]
    jS=[(j-1)%Y,j%Y,(j+1)%Y]
    neighbors=[]
    for a in range(2):
     for b in range(2):
      if not (a==b==1): neighbors.append((iS[a],jS[b]))
    jRefs = [ numpy.ravel_multi_index(tup,(X,Y)) for tup in neighbors ]
    mean=numpy.mean([ distanceMatrix[iRef,jRefs[idx]] for idx in range(8)  ])
    uMatrix[i,j,k] = mean
 else:
  for i in range(X):
   for j in range(Y):
    iRef = numpy.ravel_multi_index((i%X,j%Y),(X,Y))
    iS=[(i-1),i,(i+1)]
    jS=[(j-1),j,(j+1)]
    neighbors=[]
    for a in range(3):
     for b in range(3):
      if not (a==b==1) and 0 <= iS[a] < X and 0 <= jS[b] < Y:
       neighbors.append((iS[a],jS[b]))
    jRefs = [ numpy.ravel_multi_index(tup,(X,Y)) for tup in neighbors ]
    mean=numpy.mean([ distanceMatrix[iRef,idc] for idc in jRefs ])
    uMatrix[i,j] = mean
 return uMatrix

def sliceMatrix(matrix, nslice = 100):
 nx, ny = matrix.shape
 min = numpy.min(matrix)
 ptp = numpy.ptp(matrix)
 step = ptp/nslice
 outMatrix = numpy.zeros((nx,ny,nslice))
 for slice in range(nslice):
  mask = matrix > (min+(slice+1)*step)
  outMatrix[:,:,slice] = numpy.ma.masked_array(matrix, mask).filled(0)
 return outMatrix

def sliceMatrix2(matrix, nslice = 100):
 nx, ny = matrix.shape
 min = numpy.min(matrix)
 ptp = numpy.ptp(matrix)
 step = ptp/nslice
 outMatrix = numpy.zeros((nx,ny,nslice))
 for slice in range(nslice):
  mask = 1 - numpy.logical_and(matrix < (min+(slice+1)*step), matrix > (min+(slice)*step))
  outMatrix[:,:,slice] = numpy.ma.masked_array(matrix, mask).filled(0)
 return outMatrix


def vmdMap(matrix, outfilename, spacing=1, center=(0.,0.,0.)):
 center = tuple(numpy.array(center) + .5*spacing)
 outfile = open(outfilename, 'w')
 nx,ny,nz=matrix.shape
 outfile.write("GRID_PARAMETER_FILE NONE\n")
 outfile.write("GRID_DATA_FILE NONE\n")
 outfile.write("MACROMOLECULE NONE\n")
 outfile.write("SPACING %s\n"%spacing)
 outfile.write("NELEMENTS %d %d %d\n"%(nx-1,ny-1,nz-1))
 outfile.write("CENTER %.2f %.2f %.2f\n"%center)
 for z in range(nz):
  for y in range(ny):
   for x in range(nx):
    outfile.write("%.3f\n"%(matrix[x,y,z]))

def xyzMap(matrix, outfilename, spacing=1, center=(0.,0.,0.)):
 outfile = open(outfilename, 'w')
 offset = (numpy.asarray(matrix.shape) / 2.) * spacing - .5*spacing
 index = numpy.asarray(matrix.nonzero(), dtype=float).T * spacing + numpy.asarray(center) - offset
 outfile.write('%d\nmade by guillaume\n'%index.shape[0])
 numpy.savetxt(outfile, index, fmt='  C %10.5f%10.5f%10.5f')
 outfile.truncate()
 outfile.close()

def expandMatrix(matrix):
 n,p=matrix.shape
 outMatrix = numpy.zeros((3*n,3*p))
 for i in range(3):
  for j in range(3):
   outMatrix[i*n:(i+1)*n,j*p:(j+1)*p] = matrix
 return outMatrix

def condenseMatrix(matrix):
 n,p=matrix.shape
 n2,p2 = n/3, p/3
 outMatrix = numpy.zeros_like(matrix)
 outMatrix = numpy.reshape(outMatrix, (n2,p2,9))
 c = 0
 for i in range(3):
  for j in range(3):
   outMatrix[:,:,c] = matrix[i*n2:(i+1)*n2,j*p2:(j+1)*p2]
   c = c + 1
 outMatrix = numpy.ma.masked_array(outMatrix, outMatrix == 0)
 return numpy.ma.masked_array(outMatrix, 1-numpy.logical_and( outMatrix>=outMatrix[:,:,5].min() , outMatrix<=outMatrix[:,:,5].max())).min(axis=2).filled(0)

def continuousMap(clusters):
 for i in range(clusters.shape[0]):
  if clusters[i,0] != 0 and clusters[i,-1] != 0:
   clusters[clusters == clusters[i,-1]] = clusters[i,0]
 for j in range(clusters.shape[1]):
  if clusters[0,j] != 0 and clusters[-1,j] != 0:
   clusters[clusters == clusters[-1,j]] = clusters[0,j]
 return clusters

def uDensity(bmuDensity,uMatrix,nslice = 100,clip=True):
 if clip:
  uMatrix_sliced = sliceMatrix2(uMatrix, nslice = nslice)
 else:
  uMatrix_sliced = sliceMatrix(uMatrix, nslice = nslice)
 uDensity = numpy.zeros_like(uMatrix_sliced)
 for z in range(uMatrix_sliced.shape[2]):
  uMatrix_z = uMatrix_sliced[:,:,z]
  clusters = continuousMap(scipy.ndimage.measurements.label(uMatrix_z > 0)[0])
  for cId in numpy.unique(clusters)[1:]:
   density_cId = bmuDensity[clusters == cId].sum()/(clusters == cId).sum()
   uDensity[:,:,z][clusters == cId] = density_cId
 uDensity = numpy.ma.masked_array(uDensity, uDensity == 0)
 uDensity = uDensity.mean(axis=2)
 return uDensity

def densityProb(som, allMaps, t = None):
 if t == None:
  t = som.iterations[1]
 densityProb = numpy.ma.masked_all_like(allMaps)
 c = 0
 for dMap in allMaps.T:
  bmuRavel = numpy.where(dMap==dMap.min())
  bmu = numpy.unravel_index(bmuRavel, (som.X,som.Y))
  pMap = som.BMUneighbourhood(t,bmu,1).flatten()*(dMap**2)
  beta = 1/((dMap**2).max())
  p = -(beta*pMap).sum()
  densityProb[bmuRavel,c] = p
  c = c + 1
 densityProb = numpy.exp(densityProb.mean(axis=1).reshape(som.X,som.Y))
 return densityProb

def getBMU(som, vector):
 dMap = scipy.spatial.distance.cdist(som.Map.reshape(som.X*som.Y,som.cardinal), numpy.atleast_2d(vector)).flatten()
 bmuCoordinates = numpy.unravel_index(numpy.argmin(dMap), (som.X, som.Y))
 return bmuCoordinates

def getBmuProb(som, vector):
 t = som.iterations[1]
 dMap = scipy.spatial.distance.cdist(som.Map.reshape(som.X*som.Y,som.cardinal), numpy.atleast_2d(vector)).flatten()
 bmuCoordinates = numpy.unravel_index(numpy.argmin(dMap), (som.X, som.Y))
 pMap = som.BMUneighbourhood(t,bmuCoordinates,1).flatten()*(dMap**2)
 beta = 1/((dMap**2).max())
 p = numpy.exp(-(beta*pMap).sum())
 return bmuCoordinates, p

def uMovie(fileName = 'MapSnapshots.tar'):
 infile = tarfile.open('MapSnapshots.tar')
 members = infile.getmembers()
 os.mkdir('uMatrices')
 c = itertools.count()
 for member in members:
  mapFile = infile.extractfile(member)
  map = numpy.load(mapFile)
  mapFile.close()
  uMatrix = getUmatrix(map)
  plotMat(uMatrix, 'uMatrices/uMat_%0*d.png'%(4,c.next()))
