#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 01 19
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import IO
import numpy
import itertools
import scipy.spatial
import ConfigParser
import sys

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

nframe = Config.getint('drid', 'nframes')
structFile = Config.get('drid', 'structFile')
trajFile = Config.get('drid', 'trajFile')

struct = IO.Structure(structFile)

mask = numpy.ones((struct.atoms.shape[0]),dtype="bool")
traj = IO.Trajectory(trajFile, struct, selectionmask=mask, nframe=nframe)
drids = []

trajIndex = 0
for traj_i in traj.array:
    sys.stdout.write('%s/%s'%(trajIndex+1,nframe))
    sys.stdout.write('\r')
    sys.stdout.flush()
    shapeTraj = traj_i.reshape(traj.natom,3)
    distMat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(shapeTraj))
    distmatinv = numpy.ma.masked_invalid( 1 / scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(shapeTraj)) )
    mu = distmatinv.mean(axis=0)
    nu = numpy.sqrt(((distmatinv - mu)**2).mean(axis=0))
    xi = (((distmatinv - mu)**3).mean(axis=0))**(1./3)
    mu, nu, xi = mu.filled(), nu.filled(), xi.filled()
    drid = numpy.asarray(zip(mu,nu,xi)).flatten()
    drids.append(drid)
    trajIndex += 1

numpy.save('drids', drids)
