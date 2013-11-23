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

def getparam(pdbf, nAtoms):
 pdbData = numpy.genfromtxt(pdbf, delimiter=(6,5,5,1,3,2,4,1,11,8,8,6,6,10,2,2), invalid_raise=False, autostrip=True, dtype=numpy.str_)[:nAtoms]
 params = {}
 params.update({'record_name': pdbData[:,0]})
 params.update({'atom_number': numpy.int_(pdbData[:,1])})
 params.update({'atom_name': pdbData[:,2]})
 params.update({'alternate_location': pdbData[:,3]})
 params.update({'residue_name': pdbData[:,4]})
 params.update({'chain_id': pdbData[:,5]})
 params.update({'residue_number': numpy.int_(pdbData[:,6])})
 params.update({'residue_insertion': pdbData[:,7]})
 params.update({'occupancy': numpy.float_(pdbData[:,11])})
 params.update({'temperature_factor': numpy.float_(pdbData[:,12])})
 params.update({'segment_identifier': pdbData[:,13]})
 params.update({'element_symbol': pdbData[:,14]})
 params.update({'atom_charge': pdbData[:,15]})
 return params
