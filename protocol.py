#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import coVar2Correlation
import matriceManipulation
#import mover
import numpy
import dcdReader,pdbReader

from optparse import OptionParser
parser = OptionParser()

parser.add_option("--CA", action="store_true", dest="ca", default=True, help="For a C-alpha covariance matrix (default)")
parser.add_option("--BB", action="store_true", dest="bb", default=False, help="For a backbone covariance matrix (backbone: N-CA-C)")
parser.add_option("--conf", dest="confFileName", metavar="SOM.conf", default="SOM.conf", help="Input covariance matrix file in ptraj format")
parser.add_option("-c", "--covar", dest="covarMatrixFileName", metavar="covariance.mat", help="Input covariance matrix file in ptraj format")
parser.add_option("-s", "--pdb", dest="pdbFileName", metavar="struct.pdb", help="Input pdb File for projections. Necessary to load a dcd.")
parser.add_option("--noreduce", action="store_false", dest="reduce", default=True, help="Do not reduce the xyz coordinates to an unique coordinate")
parser.add_option("--groupXYZ", action="store_true", dest="groupedXYZ", default=False, help="Group XYZ so that the coordinates of a single residue cannot be separated into different clusters.")
parser.add_option("-m", "--map", dest="mapFileName", metavar="map.dat", default = None, help="Kohonen map file")
parser.add_option("-d", "--dcd", dest="dcdFile", metavar="trajectory.dcd", default = None, help="DCD trajectory")
parser.add_option("-a", "--align", dest="align", metavar="'mean'|<int>", default = None, help="Align the frames of given dcd file on the mean ('mean'), the nth frame (<int> = n). No alignment is done if the option is not specified.")
parser.add_option("-t", "--threshold", dest="threshold", default=None, help="Threshold for SOM clustering")

(options, args) = parser.parse_args()
if options.bb:
 options.ca = False

if options.dcdFile:
 corr = coVar2Correlation.Correlation()
 dcd=dcdReader.DcdReader(options.dcdFile,options.pdbFileName,selection="ca" if options.ca else "bb",verbose=True)
 if options.align:
  dcd.align(options.align)
 m=dcd.getCorrelation()
else:
 corr = coVar2Correlation.Correlation(options.covarMatrixFileName)
 m=corr.matrix()
corr.plot(m, outfileName='correlationXYZ.pdf', normalize=False)
mm = matriceManipulation.matrix(m)
if options.reduce and not options.groupedXYZ:
 rm=mm.reduceCoor()
 corr.plot(rm, outfileName='correlationReduce.pdf', normalize=False)
else:
 rm = m

#SOM clustering
mm = matriceManipulation.matrix(rm)
clusters = mm.somClustering(groupedXYZ=options.groupedXYZ, mapFileName=options.mapFileName, confFileName=options.confFileName, threshold=float(options.threshold))
clusterMatrix = mm.plotSomClustering(clusters)
corr.plot(clusterMatrix, outfileName='correlationSOM.pdf', normalize=False)
sortedClusterMatrix = mm.plotSortedClusters(clusterMatrix, groupedXYZ=options.groupedXYZ)
corr.plot(sortedClusterMatrix, outfileName='sortedCorrel.pdf', normalize=False)
bmuMatrix = mm.plotSomBmus(groupedXYZ=options.groupedXYZ, mapFileName=options.mapFileName, confFileName=options.confFileName)
corr.plot(bmuMatrix, outfileName='correlationBMU.pdf', normalize=False)
mm = matriceManipulation.matrix(bmuMatrix)
sortedBmuMatrix =  mm.plotSortedClusters(clusterMatrix, groupedXYZ= options.groupedXYZ)
corr.plot(sortedBmuMatrix, outfileName='sortedCorrelBMU.pdf', normalize=False)
mm = matriceManipulation.matrix(clusterMatrix)
if options.dcdFile:
 covM=dcd.covariance
else:
 covM=corr.loadCovarMatrix()
varxyz=matriceManipulation.matrix(covM)
if options.reduce:
 mm.projection(1, clusterMatrix.diagonal(), options.pdbFileName, outPdbFile='projectionCluster.pdb', ca = options.ca, bb = options.bb)
 mm.projectionRms(options.pdbFileName, varxyz, ca = options.ca, bb = options.bb)
# if options.bb:
#  mov = mover.Mover(clusterMatrix)
#  mov.moveRms(options.pdbFileName, varxyz, ca = options.ca, bb = options.bb)
else:
 reducedClusterMatrix = mm.reduceCoor(ceil=True)
 corr.plot(reducedClusterMatrix, outfileName='reducedCorrelationSOM.pdf', normalize=False)
# With mean
 traceReducedClusterMatrix = mm.reduceMean(ceil=True)
 print traceReducedClusterMatrix.shape
 corr.plot(traceReducedClusterMatrix, outfileName='reducedCorrelationSOM_mean.pdf', normalize=False)
 rmm = matriceManipulation.matrix(reducedClusterMatrix)
 rmm.projection(1, reducedClusterMatrix.diagonal(), options.pdbFileName, outPdbFile='projectionCluster.pdb', ca = options.ca, bb = options.bb)
 rmm.projectionRms(options.pdbFileName, varxyz, ca = options.ca, bb = options.bb)

#Covariance
corr.plot(covM, outfileName='covarianceXYZ.pdf', normalize=False)
mm = matriceManipulation.matrix(covM)
rcovM = mm.reduceCoor()
corr.plot(rcovM, outfileName='covarianceReduce.pdf', normalize=False)

