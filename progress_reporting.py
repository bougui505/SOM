#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2015-11-09 11:01:18 (UTC+0100)

import sys
from datetime import datetime

class Progress:
    def __init__(self, n_step, delta = 1):
        """

        • n_step: total number of step

        • delta: delta in percent (default 1%)

        """
        self.n_step = n_step
        self.progress = set([ int(self.n_step*p/100) for p in range(0,100,delta)[1:] ])
        self.c = 0
        self.delta = delta
        self.t1 = datetime.now()

    def count(self):
        self.c += 1
        if self.c in self.progress:
            t2 = datetime.now()
            percent = float(self.c)*100/(self.n_step)
            eta = (t2 - self.t1) * int((100 - percent) / self.delta)
            print "%d %% ETA: %s"%(percent, eta)
            sys.stdout.flush()
            self.t1 = t2
