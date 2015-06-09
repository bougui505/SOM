#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 09
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

from chimera.mplDialog import MPLDialog
class PlotDialog(MPLDialog):
    "PlotDialog is a Chimera dialog whose content is a matplotlib figure" 
    buttons = ('Close',) 
    title = "Self-Organizing Map"
    def __init__(self, showToolbar=True, **kw): 
        self.showToolbar = showToolbar 
        MPLDialog.__init__(self, **kw)
    def fillInUI(self, parent): 
        import Pmw, Tkinter
        # Option menu for map type
        self.mapChains = []
        self.mapTypeOption = Pmw.RadioSelect(parent,
                        labelpos='w',
                        label_text='Display: ',
                        buttontype="radiobutton",
                        command=self.switch_matrix)
        self.mapTypeOption.add("U-matrix")
        self.mapTypeOption.add("Closest frame id")
        self.mapTypeOption.pack()

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
