#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2013 12 04
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
arraydim = len(traj.shape)
if arraydim == 3:
    nframes, natoms, dim = traj.shape
elif arraydim == 2:
    nframes, natoms3 = traj.shape
    natoms = natoms3/3
    dim = 3
    traj = traj.reshape(nframes, natoms, dim)
trajobj.natom = natoms
trajobj.header['natom'] = natoms
trajobj.array = traj.reshape(nframes, natoms*dim)
trajobj.write('%s.dcd'%npyfilename.split('.')[0])
