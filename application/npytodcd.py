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
