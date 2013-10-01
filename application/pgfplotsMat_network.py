#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import numpy
import itertools
import sys
import pickle
import scipy.spatial

matrixFileName = sys.argv[1]
network = numpy.genfromtxt('SOMTrajNetwork.uniq.bmu')
bmus = numpy.int_(network[:,2:])
nodeIds = numpy.int_(network[:,0])
sStates = network[:,1]
arcs = numpy.genfromtxt('SOMTrajNetwork.uniq.arcs')
links = numpy.int_(arcs[:,:2])
flux = arcs[:,2]
links = links[numpy.argsort(flux)]
flux = flux[numpy.argsort(flux)]
bmuTraj = numpy.load('bmuCoordinates.npy')

if matrixFileName.split('.')[1] == 'npy':
 matrix = numpy.load(matrixFileName)
elif matrixFileName.split('.')[1] == 'dat':
 matrix = pickle.load(open(matrixFileName))
else:
 print "File extension must be 'dat' for a python pickle file or 'npy' for a numpy file object"
X,Y = matrix.shape

outFile = open('%s.tex'%matrixFileName.split('.')[0], 'w')
header = """
\documentclass{article}
\usepackage{pgfplots}
\usepgfplotslibrary{external}
\\tikzexternalize
\\newlength{\\figwidth}
\setlength{\\figwidth}{\\textwidth}
\\newlength{\\figheight}
\setlength{\\figheight}{%s\\figwidth}
\definecolor{skyblue1}{rgb}{0.447,0.624,0.812}
\\begin{document}
\\begin{figure}
\\begin{tikzpicture}
\\tikzset{
 every pin/.style={font=\\tiny}
}
\\begin{axis}[colorbar, colormap/bluered, width=\\figwidth, height=\\figheight, unbounded coords=jump, point meta min=%.3f, point meta max=%.3f, mark size=%s\\figwidth, clip marker paths=true, xmin = -5, xmax=%s,ymin=-5,ymax=%s,pin distance=0pt]
"""%(Y/X, numpy.nanmin(matrix), numpy.nanmax(matrix), 1/(2.0*X), X+5, Y+5)
outFile.write(header)
# for matrix plot
outFile.write("\\addplot[only marks, mark=square*, scatter/use mapped color={draw opacity=0,fill=mapped color}, scatter, scatter src=explicit] coordinates{\n")
imatrix = itertools.chain(matrix.flatten())
for i in range(X):
 for j in range(Y):
  value = imatrix.next()
  if not numpy.isnan(value):
   outFile.write("(%s,%s) [%.3f]\n"%(i,j,value))
outFile.write("};\n")
###
#plot links
iflux = itertools.chain(flux)
bins = numpy.histogram(flux, 6)[1]
b = numpy.array( [[0,0], [-X,Y], [0,Y], [X,Y], [-X,0], [X,0], [-X,-Y], [0,-Y], [X,-Y]] )
fluxMax = flux.max()
selectedLinks = []
for link in links:
 fluxValue = iflux.next()
 if fluxValue > bins[1]:
  bmu1, bmu2 = bmus[link[0]-1], bmus[link[1]-1]
  bmus2 = bmu2*numpy.ones((9,2))
  bmus2 = bmus2 + b
  dist = scipy.spatial.distance.cdist(bmus2, numpy.atleast_2d(bmu1))
  bmu2_2 = bmus2[numpy.argmin(dist)]

  bmus1 = bmu1*numpy.ones((9,2))
  bmus1 = bmus1 + b
  dist = scipy.spatial.distance.cdist(bmus1, numpy.atleast_2d(bmu2))
  bmu1_2 = bmus1[numpy.argmin(dist)]

  if fluxValue >= bins[-5] and fluxValue <= bins[-4]:
   color = 'white'
   linestyle = 'solid'
  elif fluxValue > bins[-4] and fluxValue <= bins[-3]:
   color = 'white'
   linestyle = 'solid'
  elif fluxValue > bins[-3] and fluxValue <= bins[-2]:
   color = 'white'
   linestyle = 'solid'
  elif fluxValue > bins[-2] and fluxValue <= bins[-1]:
   color = 'white'
   linestyle = 'solid'
  else:
   color = 'white'
   linestyle = 'loosely dotted'
  lineWidth = 2.*numpy.exp(-1.8*(1-fluxValue/fluxMax))
#  lineWidth = 2.*numpy.exp(-1.*(1-fluxValue/fluxMax))
  x1,y1 = bmu1
  x2,y2 = numpy.int_(bmu2_2)
  outFile.write("\\addplot[color=%s, mark=none, line width=%.2f, style=%s] coordinates{(%s,%s) (%s,%s)};\n"%(color,lineWidth,linestyle,x1,y1,x2,y2))
  x1,y1 = numpy.int_(bmu1_2)
  x2,y2 = bmu2
  outFile.write("\\addplot[color=%s, mark=none, line width=%.2f, style=%s] coordinates{(%s,%s) (%s,%s)};\n"%(color,lineWidth,linestyle,x1,y1,x2,y2))
  selectedLinks.extend([bmu1,bmu2,bmu1_2,bmu2_2])
selectedLinks = numpy.array(selectedLinks)
###
# for network plot
inodeId = itertools.chain(nodeIds)
isState = itertools.chain(sStates)
bins = numpy.histogram(sStates, bins=5)[1]
for bmu in bmus:
 i,j = bmu
 nodeId = inodeId.next()
 sState = isState.next()
 if sState >= bins[0] and sState <= bins[1]:
  color = 'blue'
 elif sState > bins[1] and sState <= bins[2]:
  color = 'blue'
 elif sState > bins[2] and sState <= bins[3]:
  color = 'green'
 elif sState > bins[3] and sState <= bins[4]:
  color = 'orange'
 else:
  color = 'red'
 if (bmu == selectedLinks).sum(axis=1).max() == 2:
  snapshotIds = numpy.where(((bmu == bmuTraj).all(axis=1)))[0]
  snapshotId = snapshotIds.min()
  outFile.write("\\addplot[only marks, mark=oplus*, color=black, fill=%s, mark size=%.2f] coordinates{"%(color,(sState/sStates.max())*6))
  outFile.write("(%s,%s)\n"%(i,j))
  outFile.write("};\n")
  outFile.write("\\node[pin=120:{%s}] at (axis cs:%s,%s) {};\n"%(snapshotId,i,j))
#  outFile.write("\\addplot[nodes near coords, point meta=explicit symbolic, only marks, mark=none, color=black, font=\\tiny, node/.style={fill=yellow!50!white, rectangle,rounded corners=3pt}] coordinates{")
#  outFile.write("(%s,%s) [%s]\n"%(i,j,snapshotId))
#  outFile.write("};\n")
  outFile_snapshots = open('%s-%s.txt'%tuple(bmu), 'w')
  [outFile_snapshots.write('%s\n'%(e+1)) for e in snapshotIds]
  outFile_snapshots.close()
###
closing = """
\end{axis}
\end{tikzpicture}
\end{figure}
\end{document}
"""
outFile.write(closing)
outFile.close()
