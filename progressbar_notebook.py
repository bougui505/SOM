#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 01 16
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""


from IPython.display import HTML, Javascript, display
import uuid

class ProgressBar():

    def __init__(self, widgets, maxval):
        self.maxval = maxval
    
    def start(self):
        self.divid = str(uuid.uuid4())
        pb = HTML(
        """
        <div style="border: 1px solid black; width:500px">
          <div id="%s" style="background-color:blue; width:0%%">&nbsp;</div>
        </div> 
        """ % self.divid)
        display(pb)

    def update(self, value):
        modulo = self.maxval / 100
        if value % modulo == 0:
            display(Javascript("$('div#%s').width('%i%%')" % (self.divid, value*101/self.maxval)))

    def finish(self):
        print 'done'
        return None

def Percentage():
    return None

def Bar(marker,left,right):
    return None

def ETA():
    return None

