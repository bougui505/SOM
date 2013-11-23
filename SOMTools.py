#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2013 10 23
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
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
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters
import scipy.misc
import copy

def plot3Dmat(mat, contourScale = True, rstride=1, cstride=1):
 from mpl_toolkits.mplot3d import Axes3D
 from matplotlib import cm
 from matplotlib.ticker import LinearLocator, FormatStrFormatter
 import matplotlib.pyplot as plt
 import numpy as np
 fig = plt.figure()
 ax = fig.gca(projection='3d')
 x,y = mat.shape
 X, Y = np.meshgrid(np.arange(x), np.arange(y))
 Z = mat
 surf = ax.plot_surface(X, Y, Z, rstride=rstride, cstride=cstride, linewidth=0., cmap=cm.jet, alpha=0.75)
 cset = ax.contour(X, Y, Z, zdir='z', offset=mat.min(), cmap=cm.gray)
 if contourScale:
  fig.colorbar(cset)
 cset = ax.contour(X, Y, Z, zdir='x', offset=0, cmap=cm.gray)
 cset = ax.contour(X, Y, Z, zdir='y', offset=y, cmap=cm.gray)
# surf = ax.contour(X, Y, Z, cmap=cm.hot)
# surf = ax.contourf(X, Y, Z, cmap=cm.hot)
 fig.colorbar(surf)
 plt.show()

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

def minPath(matrix, gradThreshold=None, startingPoint = None, nsteps=None):
 if nsteps == None:
  nsteps = matrix.size
 if gradThreshold == None:
  gradThreshold = numpy.mean(matrix)
 if type(matrix) == numpy.ma.core.MaskedArray:
  matrix = matrix.filled()
 X,Y = matrix.shape
 if startingPoint == None:
  iStart,jStart = scipy.ndimage.measurements.minimum_position(matrix)
 else:
  iStart,jStart = startingPoint
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
  if c > nsteps:
   break
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
   sys.stdout.write('flood: grad=%10.2f ; reverse: %10.0f; clusters #: %10.0f'%(grad,r,cIndex))
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
    jRef1 = numpy.ravel_multi_index(((i-1)%X,(j-1)%Y),(X,Y))
    jRef2 = numpy.ravel_multi_index(((i-1)%X,(j)%Y),(X,Y))
    jRef3 = numpy.ravel_multi_index(((i-1)%X,(j+1)%Y),(X,Y))
    jRef4 = numpy.ravel_multi_index(((i)%X,(j-1)%Y),(X,Y))
    jRef5 = numpy.ravel_multi_index(((i)%X,(j+1)%Y),(X,Y))
    jRef6 = numpy.ravel_multi_index(((i+1)%X,(j-1)%Y),(X,Y))
    jRef7 = numpy.ravel_multi_index(((i+1)%X,(j)%Y),(X,Y))
    jRef8 = numpy.ravel_multi_index(((i+1)%X,(j+1)%Y),(X,Y))
    mean = numpy.mean([distanceMatrix[iRef,jRef1], distanceMatrix[iRef,jRef2], distanceMatrix[iRef,jRef3], distanceMatrix[iRef,jRef4], distanceMatrix[iRef,jRef5], distanceMatrix[iRef,jRef6], distanceMatrix[iRef,jRef7], distanceMatrix[iRef,jRef8]])
    uMatrix[i,j] = mean
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

def expandMatrix(matrix, expansionfactor = 3):
 if len(matrix.shape) == 2:
  n,p=matrix.shape
  outMatrix = numpy.zeros((expansionfactor*n,expansionfactor*p))
  for i in range(expansionfactor):
   for j in range(expansionfactor):
    outMatrix[i*n:(i+1)*n,j*p:(j+1)*p] = matrix
 elif len(matrix.shape) == 3:
  n,p,k=matrix.shape
  outMatrix = numpy.zeros((expansionfactor*n,expansionfactor*p,k))
  for i in range(expansionfactor):
   for j in range(expansionfactor):
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
 c = 1
 for e in numpy.unique(clusters)[1:]:
  clusters[clusters==e] = c
  c+=1
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

def getBMUfromMap(smap, vector):
 X,Y,cardinal = smap.shape
 dMap = scipy.spatial.distance.cdist(smap.reshape(X*Y,cardinal), numpy.atleast_2d(vector)).flatten()
 bmuCoordinates = numpy.unravel_index(numpy.argmin(dMap), (X, Y))
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

def detect_local_minima(arr, toricMap=False, getFilteredArray=False):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    if toricMap:
        X,Y = arr.shape
        arr = expandMatrix(arr)
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    arr_filtered = filters.minimum_filter(arr, footprint=neighborhood)
    local_min = (arr_filtered==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    if toricMap:
        detected_minima = detected_minima[X:2*X,Y:2*Y]
    if not getFilteredArray:
        return numpy.where(detected_minima)
    else:
        if toricMap:
            return numpy.where(detected_minima), condenseMatrix(arr_filtered)
        else:
            return numpy.where(detected_minima), arr_filtered

def detect_local_minima2(arr, toricMap=False):
    X,Y = arr.shape
    lminima = []
    for i in range(X):
        for j in range(Y):
            pos = (i,j)
            neighbors = getNeighbors(pos, (X,Y))
            nvalues = numpy.asarray( [ arr[e[0],e[1]] for e in neighbors] )
            if (arr[i,j] <= nvalues).all():
                lminima.append((i,j))
    lminima = numpy.asarray(lminima)
    lminima = (lminima[:,0], lminima[:,1])
    return lminima

def detect_local_maxima(arr, toricMap=False):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    if toricMap:
        X,Y = arr.shape
        arr = expandMatrix(arr)
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_max = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_maxima = local_max - eroded_background
    if toricMap:
        detected_maxima = detected_maxima[X:2*X,Y:2*Y]
    return numpy.where(detected_maxima)

def getSaddlePoints(matrix, gaussian_filter_sigma=0., low=None, high=None):
    if low == None:
        low = matrix.min()
    if high == None:
        high = matrix.max()
    matrix = expandMatrix(matrix)
    neighborhood = morphology.generate_binary_structure(len(matrix.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    matrix = filters.minimum_filter(matrix, footprint=neighborhood)
    matrix = condenseMatrix(matrix)
    outPath, clusterPathMat, grad = minPath(matrix)
    flood = numpy.asarray(outPath)
    potential = []
    for e in flood:
        i,j = e
        potential.append(matrix[i,j])
    potential = numpy.asarray(potential)
    potential = scipy.ndimage.filters.gaussian_filter(potential, gaussian_filter_sigma)
    derivative = lambda x: numpy.array(zip(-x,x[1:])).sum(axis=1)
    signproduct = lambda x: numpy.array(zip(x,x[1:])).prod(axis=1)
    potential_prime = derivative(potential)
    signproducts = numpy.sign(signproduct(potential_prime))
    extrema = flood[2:][numpy.where(signproducts<0)[0],:]
    bassinlimits = derivative(signproducts)
    saddlePoints = numpy.asarray(outPath[3:])[bassinlimits==-2]
    saddlePointValues = numpy.asarray(map(lambda x: matrix[x[0],x[1]], saddlePoints))
    saddlePoints = saddlePoints[numpy.logical_and(saddlePointValues>=low, saddlePointValues<=high),:]
    return saddlePoints

def getVectorField(Map, sign=True, colorMatrix=None):
 X,Y,cardinal = Map.shape
 uMatrix = getUmatrix(Map)
 uMatrix_ravel = uMatrix.flatten()
 distanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Map.reshape(X*Y,cardinal)))
 vectorsField = numpy.zeros((X,Y,2))
 vectors_unit = [(-1/numpy.sqrt(2),-1/numpy.sqrt(2)),(-1,0),(-1/numpy.sqrt(2),1/numpy.sqrt(2)),(0,-1),(0,1),(1/numpy.sqrt(2),-1/numpy.sqrt(2)),(1,0),(1/numpy.sqrt(2),1/numpy.sqrt(2))]
 for i in range(X):
  for j in range(Y):
   iRef = numpy.ravel_multi_index((i%X,j%Y),(X,Y))
   jRef1 = numpy.ravel_multi_index(((i-1)%X,(j-1)%Y),(X,Y))
   jRef2 = numpy.ravel_multi_index(((i-1)%X,(j)%Y),(X,Y))
   jRef3 = numpy.ravel_multi_index(((i-1)%X,(j+1)%Y),(X,Y))
   jRef4 = numpy.ravel_multi_index(((i)%X,(j-1)%Y),(X,Y))
   jRef5 = numpy.ravel_multi_index(((i)%X,(j+1)%Y),(X,Y))
   jRef6 = numpy.ravel_multi_index(((i+1)%X,(j-1)%Y),(X,Y))
   jRef7 = numpy.ravel_multi_index(((i+1)%X,(j)%Y),(X,Y))
   jRef8 = numpy.ravel_multi_index(((i+1)%X,(j+1)%Y),(X,Y))
   norms = [distanceMatrix[iRef,jRef1], distanceMatrix[iRef,jRef2], distanceMatrix[iRef,jRef3], distanceMatrix[iRef,jRef4], distanceMatrix[iRef,jRef5], distanceMatrix[iRef,jRef6], distanceMatrix[iRef,jRef7], distanceMatrix[iRef,jRef8]]
   if sign:
    signs = [numpy.sign(uMatrix_ravel[iRef]-uMatrix_ravel[e]) for e in [jRef1,jRef2,jRef3,jRef4,jRef5,jRef6,jRef7,jRef8]]
    norms = numpy.array(signs)*numpy.array(norms)
   vectors = numpy.atleast_2d(norms).T*numpy.array(vectors_unit)
   vectorsField[i,j] = vectors.sum(axis=0)
 if colorMatrix == None:
  vectorsFieldPlot = matplotlib.pyplot.quiver(vectorsField[:,:,1], vectorsField[:,:,0], uMatrix, units='xy', pivot='tail')
 else:
  vectorsFieldPlot = matplotlib.pyplot.quiver(vectorsField[:,:,1], vectorsField[:,:,0], colorMatrix, units='xy', pivot='tail')
 return vectorsField

def getFlowMap(bmus,smap,colorByUmatrix=True,colorByPhysicalTime=False, colorByDensity=False, normByDensity=False,timeStep=None, inFlow = False, colorbar = True, colormap=None, colorMatrix=None):
    X,Y,cardinal = smap.shape
    pivot = 'tail'
    if inFlow:
        bmus = bmus[::-1]
        pivot = 'tip'
    bmuLinks = numpy.array( zip( bmus,bmus[1:],bmus[1:]+numpy.array([X,0]),bmus[1:]+numpy.array([0,Y]),bmus[1:]+numpy.array([X,Y]),bmus[1:]+numpy.array([-X,0]),bmus[1:]+numpy.array([0,-Y]),bmus[1:]+numpy.array([-X,Y]),bmus[1:]+numpy.array([X,-Y]),bmus[1:]+numpy.array([-X,Y]) ))
    getMinDistIndex = lambda x: x[1:][scipy.spatial.distance.cdist(numpy.atleast_2d(x[0]),x[1:]).argmin()] # To take into account periodicity
    bmuLinks = numpy.array(map(getMinDistIndex, bmuLinks))
    vectors = bmuLinks - bmus[:bmuLinks.shape[0]]
    n = vectors.shape[0]
    vectorsMap = numpy.zeros((X,Y,2))
    normsMap = numpy.zeros((X,Y))
    counterMap = numpy.zeros((X,Y), dtype=int)
    c = 1
    quiverkeyLength = 50
    for k in range(n):
        i,j = bmus[k]
        normOfVect_k = numpy.linalg.norm(vectors[k])
        if normOfVect_k != 0:
            vectorsMap[i,j] += vectors[k] / normOfVect_k
            normsMap[i,j]+=1
        counterMap[i,j] = c
        c+=1
    if normByDensity:
        vectorsMap = vectorsMap / numpy.atleast_3d(normsMap)
        quiverkeyLength = 1
    coords = numpy.zeros((X*Y,4))
    c = 0
    for i in range(X):
        for j in range(Y):
            u,v = vectorsMap[i,j]
            coords[c] = [i,j,u,v]
            c+=1
    numpy.savetxt('vectorsField.txt', coords, fmt='%d %d %.2f %.2f')
    matplotlib.pyplot.axis([-X/20,X+X/20,-Y/20,Y+Y/20])
    if colorByUmatrix and not colorByPhysicalTime and not colorByDensity and colorMatrix == None:
        uMatrix = getUmatrix(smap)
        vectorsMapPlot = matplotlib.pyplot.quiver(coords[:,0], coords[:,1], coords[:,2], coords[:,3], uMatrix, units='xy', pivot=pivot, cmap=colormap)
    if colorByPhysicalTime:
        if timeStep ==None:
            vectorsMapPlot = matplotlib.pyplot.quiver(coords[:,0], coords[:,1], coords[:,2], coords[:,3], counterMap, units='xy', pivot=pivot, cmap=colormap)
        else:
            vectorsMapPlot = matplotlib.pyplot.quiver(coords[:,0], coords[:,1], coords[:,2], coords[:,3], counterMap*timeStep, units='xy', pivot=pivot, cmap=colormap)
    if colorByDensity:
        vectorsMapPlot = matplotlib.pyplot.quiver(coords[:,0], coords[:,1], coords[:,2], coords[:,3], normsMap, units='xy', pivot=pivot, cmap=colormap)
    if colorMatrix != None:
        vectorsMapPlot = matplotlib.pyplot.quiver(coords[:,0], coords[:,1], coords[:,2], coords[:,3], colorMatrix, units='xy', pivot=pivot, cmap=colormap)
    matplotlib.pyplot.quiverkey(vectorsMapPlot, 0.9, 0.01, quiverkeyLength, 'flow: %d'%(quiverkeyLength), coordinates = 'axes')
    if colorbar:
        cb = matplotlib.pyplot.colorbar()
    if colorByPhysicalTime and timeStep != None:
        cb.set_label('Physical time in ns')
    zeroFlow = numpy.where(normsMap==0)
    matplotlib.pyplot.plot(zeroFlow[0], zeroFlow[1], linestyle='.', markerfacecolor='black', marker='o')
    return vectorsMap, normsMap

def getNeighbors(pos,shape):
    X,Y = shape
    i,j = pos
    neighbors = []
    for k in range(i-1,i+2):
        for l in range(j-1,j+2):
            if k != i or l != j:
                neighbors.append((k%X,l%Y))
    return neighbors

def metropolis_acceptance(matrix, pos1, pos2, k, T):
    p_pos2 = numpy.exp( -matrix[pos2] / (k*T) )
    p_pos1 = numpy.exp( -matrix[pos1] / (k*T) )
    acceptance = min( [ 1, p_pos2 / p_pos1  ] )
    return numpy.random.rand() < acceptance

def mcpath(matrix, start, nstep, T=298.0, stop = None, k = None, x_offset=None, y_offset=None, mask=None):
    #matrix,x_offset,y_offset,mask = contourSOM(matrix, x_offset, y_offset, mask)
    if k == None:
        k = numpy.median(matrix) / (numpy.log(100)*298.0) # acceptance of 0.01 for the median energy at 298 K
    X,Y = matrix.shape
    if stop == None:
        target = matrix.min()
        stop = scipy.ndimage.minimum_position(matrix)
        print 'Minimal value for (%d,%d) position'%stop
        minpos = numpy.asarray(numpy.where(matrix == target)).T
    else:
        minpos = numpy.asarray(stop)
        minpos.resize((1,2))
    #print minpos
    grid = numpy.ones_like(matrix)
    k_grid = numpy.median(grid) / (numpy.log(100)*T) # acceptance of 0.01 for the median energy at 298 K
    for e in minpos:
        i,j = e
        grid[i,j] = 0
    grid = scipy.ndimage.morphology.distance_transform_edt(grid)
    pos = start
    neighbors = getNeighbors(pos, (X,Y))
    numpy.random.shuffle(neighbors)
    path = []
    energies = []
    pathMat = numpy.zeros((X,Y), dtype='bool')
    path.append(pos)
    pathMat[pos] = True
    energies.append(matrix[pos])
    for i in range(nstep):
        for pos2 in neighbors:
            pos2 = tuple(pos2)
            isAccepted = metropolis_acceptance(grid, pos, pos2, k_grid, T)
            if isAccepted:
                isAccepted = metropolis_acceptance(matrix, pos, pos2, k, T)
                if isAccepted:
                    pos = pos2
                    break
        if pos not in path:
            path.append(pos)
            energies.append(matrix[pos])
            pathMat[pos] = True
        if pos == stop:
            break
        neighbors = getNeighbors(pos, (X,Y))
        numpy.random.shuffle(neighbors)
    return matrix, path, pathMat, energies, grid

def histeq(im,nbr_bins=256):
    """Histogram equalization with Python and NumPy """
    #get image histogram
    imhist,bins = numpy.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    #use linear interpolation of cdf to find new pixel values
    im2 = numpy.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

def ddmap(mat):
    """
    Data driven colormapping from: http://graphics.tu-bs.de/publications/Eisemann11DDC/
    """
    scoresfrommap = numpy.unique(mat)
    scoresfrommap = scoresfrommap[numpy.asarray(1-numpy.isnan(scoresfrommap), dtype=bool)]
#plot(scoresfrommap)
    v = numpy.asarray(zip(range(len(scoresfrommap)), scoresfrommap))
    vdiag = v[-1] - v[0]
    proj = numpy.asarray(map(lambda x: numpy.dot(x, vdiag) / numpy.linalg.norm(vdiag)**2, v))
    dictproj = dict(zip(scoresfrommap, proj))
    projmat = numpy.reshape(numpy.asarray([dictproj[e] if not numpy.isnan(e) else e for e in mat.flatten()]), mat.shape)
    projmat = projmat * scoresfrommap.ptp() + scoresfrommap.min()
    return projmat
