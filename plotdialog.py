#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 01
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

from chimera.baseDialog import ModelessDialog 
class PlotDialog(ModelessDialog): 
    "PlotDialog is a Chimera dialog whose content is a matplotlib figure" 
    buttons = ('Close',) 
    title = "matplotlib figure" 
    def __init__(self, showToolbar=True, **kw): 
        self.showToolbar = showToolbar 
        ModelessDialog.__init__(self, **kw) 
    def fillInUI(self, parent): 
        from matplotlib.figure import Figure 
        self.figure = Figure() 
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg 
        fc = FigureCanvasTkAgg(self.figure, master=parent) 
        fc.get_tk_widget().pack(side="top", fill="both", expand=True) 
        self.figureCanvas = fc 
        if self.showToolbar: 
            nt = NavigationToolbar2TkAgg(fc, parent) 
            nt.update() 
            self.navToolbar = nt 
        else: 
            self.navToolbar = None 
    def add_subplot(self, *args): 
        return self.figure.add_subplot(*args) 
    def delaxes(self, ax): 
        self.figure.delaxes(ax) 
    def draw(self): 
        self.figureCanvas.draw() 
    def registerPickHandler(self, func): 
        self.figureCanvas.mpl_connect("pick_event", func) 
