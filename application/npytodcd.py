#!/usr/bin/env python
import numpy
import sys
import IO
npyfilename = sys.argv[1]
traj = numpy.load(npyfilename)
trajobj = IO.Trajectory()
nframes, natoms, dim = traj.shape
trajobj.natom = natoms
trajobj.header['natom'] = natoms
trajobj.array = traj.reshape(nframes, natoms*3)
trajobj.write('%s.dcd'%npyfilename.split('.')[0])
