#!/usr/bin/env python
import numpy
#import pylab

def princomp(A,numpc=4,reconstruct=False,getEigenValues=True):
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A - numpy.atleast_2d(numpy.mean(A,axis=1)).T) # subtract the mean (along columns)
# print 'A:%s'%A
# print 'M:%s'%M
# print 'cov:%s'%numpy.cov(M)
 [eigenValues,eigenVectors] = numpy.linalg.eig(numpy.cov(M))
 p = numpy.size(eigenVectors,axis=1)
 idx = numpy.argsort(eigenValues) # sorting the eigenvalues
 idx = idx[::-1]       # in ascending order
 # sorting eigenvectors according to the sorted eigenvalues
 eigenVectors = eigenVectors[:,idx]
 eigenValues = eigenValues[idx] # sorting eigenvalues
 if numpc < p or numpc >= 0:
  eigenVectors = eigenVectors[:,range(numpc)] # cutting some PCs
#  eigenValues = eigenValues[range(numpc)]
 #data reconstruction
 if reconstruct:
#  A_r = numpy.zeros_like(A)
#  for i in range(numpc):
#   A_r = A_r + eigenValues[i]*numpy.dot(numpy.atleast_2d(eigenVectors[:,i]).T,numpy.atleast_2d(eigenVectors[:,i]))
  score = numpy.dot(eigenVectors.T,M) # projection of the data in the new space
  Ar = (numpy.dot(eigenVectors,score)+numpy.mean(A,axis=0)).T # image reconstruction
  return eigenVectors.real,eigenValues.real,Ar
 else:
  if getEigenValues:
   return eigenVectors.real,eigenValues.real
  else:
   return eigenVectors.real
