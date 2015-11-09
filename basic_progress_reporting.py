#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2015-11-09 11:01:18 (UTC+0100)

import sys

class Progress:
    def __init__(self, n_step, delta = 1):
        """

        • n_step: total number of step

        • delta: delta in percent (default 1%)

        """
        self.n_step = n_step
        self.progress = set([ int(self.n_step*p/100) for p in range(0,100,delta)[1:] ])
        self.c = 0

    def count(self):
        self.c += 1
        if self.c in self.progress:
            print "%d %%"%(float(self.c)*100/(self.n_step))
            sys.stdout.flush()
