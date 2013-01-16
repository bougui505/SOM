#!/usr/bin/env python
import numpy
import itertools
import sys
import pickle
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="matrixFileName", help="input matrix file to plot", metavar="file.npy", default=None)
parser.add_option("-n", "--name", dest="namesFileName", help="names to write on the map", metavar="names.npy", default=None)
parser.add_option("-b", "--bmuCoordinates", dest="bmuCoordsFileName", help="BMU Coordinates. Required only for plotting names", default=None)
parser.add_option("-c", "--contour", action="store_false", dest="use_mapped_color", help="Add contour to pixels", default=True)
parser.add_option("-o", "--offset", type="int", nargs=2, dest="offset", help="Add x and y offset", metavar= "0 0", default=(0,0))
(options, args) = parser.parse_args()

matrixFileName = options.matrixFileName
plotNames = False
if options.namesFileName != None:
 if options.bmuCoordsFileName == None:
  raise IOError('No bmuCoordinates file')
 plotNames = True
 namesFileName = options.namesFileName
 names = numpy.load(namesFileName)
 names = numpy.unique(names)
 bmuCoordinates = numpy.load(options.bmuCoordsFileName)

x_offset = options.offset[0]
y_offset = options.offset[1]

if matrixFileName.split('.')[1] == 'npy':
 matrix = numpy.load(matrixFileName)
elif matrixFileName.split('.')[1] == 'dat':
 matrix = pickle.load(open(matrixFileName))
else:
 raise IOError("wrong format for matrix data file. Must be 'dat' for a python pickle file or 'npy' for a numpy file object")
X,Y = matrix.shape

if x_offset !=0 or y_offset != 0:
 matrix_o = numpy.ma.masked_all_like(matrix)
 for i in range(X):
  for j in range(Y):
   matrix_o[i,j] = matrix[(i+x_offset)%X,(j+y_offset)%Y]
 matrix = matrix_o

if numpy.ma.isMaskedArray(matrix):
 matrix = matrix.filled(numpy.nan)

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
\\begin{document}
\\begin{figure}
\\begin{tikzpicture}
\\begin{axis}[colorbar, colormap/bluered, width=\\figwidth, height=\\figheight, unbounded coords=jump, point meta min=%.3f, point meta max=%.3f, mark size=%s\\figwidth, clip marker paths=true, pin distance=0pt, xmin=-1, ymin=-1, xmax=%d, ymax=%d]
"""%(Y/X, numpy.nanmin(matrix), numpy.nanmax(matrix), 1/(2.25*X),X,Y)
#\\begin{axis}[colorbar, colormap/bluered, width=\\figwidth, height=\\figheight, unbounded coords=jump, point meta min=%.3f, point meta max=%.3f, mark size=%s\\figwidth]
outFile.write(header)
# for matrix plot
if options.use_mapped_color:
 outFile.write("\\addplot[only marks, mark=square*, scatter/use mapped color={draw opacity=0,fill=mapped color}, scatter, scatter src=explicit] coordinates{\n")
else:
 outFile.write("\\addplot[only marks, mark=square*, scatter, scatter src=explicit] coordinates{\n")
imatrix = itertools.chain(matrix.flatten())
if plotNames:
 outNames = []
for i in range(X):
 for j in range(Y):
  value = imatrix.next()
  if plotNames:
   bmuIndex = numpy.where((bmuCoordinates == [i,j]).all(axis=1))[0]
   if numpy.size(bmuIndex) == 0:
    plotNames2 = False
   else:
    plotNames2 = True
    bmuIndex = bmuIndex[0]
    name = names[bmuIndex]
  if not numpy.isnan(value):
   outFile.write("(%s,%s) [%.3f]\n"%(i,j,value))
   if plotNames and plotNames2:
    outNames.append("\\node[color=orange] at (axis cs:%s,%s) {%s};\n"%(i,j, name))
outFile.write("};\n")
if plotNames:
 outFile.writelines(outNames)
###
closing = """
\end{axis}
\end{tikzpicture}
\end{figure}
\end{document}
"""
outFile.write(closing)
outFile.close()
