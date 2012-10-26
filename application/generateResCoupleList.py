#!/usr/bin/env python
import getparam
import itertools
import numpy
nAtoms = 605
#resList = [[int(e[0]),e[1]] for e in numpy.genfromtxt('resList', dtype=str)]
pdbf = open('protein_ca.pdb')
params = getparam.getparam(pdbf, nAtoms)
residue_numbers = params['residue_number'][1:-1]
segment_identifiers = params['segment_identifier'][1:-1]
iSeg = itertools.chain(segment_identifiers)
resList = [[e,iSeg.next()] for e in residue_numbers]
#resSeg = [[residue_numbers[i], segment_identifiers[i]] for i in range(len(residue_numbers))]
couples = [e for e in itertools.combinations(resList, 2)]
of = open('restraintsList', 'w')
[of.write('%s %s %s %s\n'%(e[0][0], e[0][1], e[1][0], e[1][1])) for e in couples]
of.close()
