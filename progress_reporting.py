#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2015-11-09 11:01:18 (UTC+0100)

import sys
from datetime import datetime

class Progress:
    def __init__(self, n_step, delta = 1, label = None):
        """

        • n_step: total number of step

        • delta: delta in percent (default 1%)

        • label: Optional string to display

        """
        self.n_step = n_step
        self.progress = set([ int(self.n_step*p/100) for p in range(0,100,delta)[1:] ])
        self.progress.update([self.n_step,])
        self.c = 0
        self.delta = delta
        self.t1 = datetime.now()
        self.label = label

    def count(self, report=None):
        """
        • report: A string or value you want to report
        """
        self.c += 1
        if self.c in self.progress or self.c == 1:
            t2 = datetime.now()
            percent = float(self.c)*100/(self.n_step)
            if self.c > 1:
                eta = (t2 - self.t1) * int((100 - percent) / self.delta)
            else:
                eta = (t2 - self.t1) * int((100 - percent) / percent)
            if self.label is None:
                string = "%d %% ETA: %s"%(percent, eta)
            else:
                string = "%s: %d %% ETA: %s"%(self.label, percent, eta)
            if report is not None:
                string+=" | %s"%report
            print string
            sys.stdout.flush()
            self.t1 = t2
