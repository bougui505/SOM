#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 23
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

from chimera.mplDialog import MPLDialog


class PlotDialog(MPLDialog):
    "PlotDialog is a Chimera dialog whose content is a matplotlib figure"
    buttons = ('Close',)
    title = "Self-Organizing Map"
    provideStatus = True

    def __init__(self, min_value, max_value, max_path_value, showToolbar=True, **kw):
        """
        min_value and max_value are the limits for the slider on the U-matrix
        max_path_value maximum path distance
        """
        self.min_value = min_value
        self.max_value = max_value
        self.max_path_value = max_path_value
        self.showToolbar = showToolbar
        MPLDialog.__init__(self, **kw)

    def fillInUI(self, parent):
        import Pmw, Tkinter

        # Main figure
        from matplotlib.figure import Figure
        self.figure = Figure()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
        fc = FigureCanvasTkAgg(self.figure, master=parent)
        fc.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.figureCanvas = fc
        if self.showToolbar:
            nt = NavigationToolbar2TkAgg(fc, parent)
            nt.update()
            self.navToolbar = nt
        else:
            self.navToolbar = None

        # Sliders for clustering
        resolution = (self.max_value - self.min_value)/1000
        self.slider = Tkinter.Scale(parent, from_=self.min_value+resolution, to=self.max_value, command=self.get_clusters, orient='horizontal', label='Threshold on U-matrix', length=500, resolution=resolution)
        self.slider.pack(side='top')
        resolution = (self.max_path_value)/100000
        self.slider2 = Tkinter.Scale(parent, from_=0, to=self.max_path_value, command=self.get_basin, orient='horizontal', label='Threshold on U-matrix for basin flooding', length=500, resolution=resolution)
        self.slider2.pack(side='top')

        # Option menu for map type
        self.display_option_items = ["U-matrix", "Density", "Closest frame id", "RMSD"]
        self.display_option = Pmw.OptionMenu(parent,
                                             labelpos='w',
                                             label_text='Display: ',
                                             items = self.display_option_items,
                                             command=self.switch_matrix)
        self.display_option.pack(side='left')

        # Button to add projection map
        self.add_projection_button = Tkinter.Button(parent, text='Data projection',
                                                    command=self.project_data)
        self.add_projection_button.pack(side='left')

        # Option menu for selection mode
        self.selection_mode_menu = Pmw.OptionMenu(parent,
                                             labelpos='w',
                                             label_text='Selection mode: ',
                                             items = ['Cell', 'Cluster'],
                                             command=self.update_selection_mode)
        self.selection_mode_menu.pack(side='left')

        # Option to select a slice to display
        self.slice_items = [0, ]
        self.slice_selection = Pmw.OptionMenu(parent,
                                                labelpos='w',
                                                label_text='Slice: ',
                                                items = self.slice_items,
                                                command=self.slice_matrix)
        self.slice_selection.pack(side='left')

        # Option to select features to display in 1D
        self.feature_items = [None, ]
        self.feature_selection = Pmw.OptionMenu(parent,
                                                labelpos='w',
                                                label_text='Display features: ',
                                                items = self.feature_items,
                                                command=self.display_features)
        self.feature_selection.pack(side='left')

    def add_subplot(self, *args):
        return self.figure.add_subplot(*args)

    def delaxes(self, ax):
        self.figure.delaxes(ax)

    def draw(self):
        self.figureCanvas.draw()

    def registerPickHandler(self, func):
        self.figureCanvas.mpl_connect("pick_event", func)


from chimera.baseDialog import ModalDialog
from tkFileDialog import askopenfilename

import Tkinter
class Projection(ModalDialog):
    buttons = ('Cancel', 'Apply')

    def __init__(self):
        self.name = None
        self.filename = None
        ModalDialog.__init__(self)

    def fillInUI(self, parent):
        from chimera.tkoptions import StringOption
        Tkinter.Label(parent, text='Data projection',
                        relief="ridge", bd=4).grid(row=0, column=0)
        self.name = StringOption(parent, 1, "Name for the projection", 'Projection', None)
        self.filename = Tkinter.Button(parent, text='Open data file',
                                        command=lambda: self.get_filename(parent)).grid(row=2, column=0)

    def get_filename(self, parent):
        self.filename = askopenfilename()
        Tkinter.Label(parent, text='Filename: %s'%self.filename,
                            relief="ridge", bd=4).grid(row=2, column=1)

    def Apply(self):
        name = self.name.get()
        ModalDialog.Cancel(self, value=(name, self.filename))

    def destroy(self):
        ModelessDialog.destroy(self)



class RMSD(ModalDialog):
    buttons = ('Cancel', 'Apply')

    def __init__(self, movie):
        self.movie = movie
        self.ref = None
        # self.ref2 = None # for an other value
        ModalDialog.__init__(self)

    def fillInUI(self, parent):
        import Tkinter
        from chimera.tkoptions import IntOption, BooleanOption, FloatOption

        Tkinter.Label(parent, text="Compute RMSD from a reference frame.\nThis calculation can take SEVERAL MINUTES...",
                      relief="ridge", bd=4).grid(row=0, column=0, columnspan=2, sticky="ew")
        startFrame = self.movie.startFrame
        endFrame = self.movie.endFrame
        self.ref = IntOption(parent, 1, "Frame of reference", startFrame, None, min=startFrame, max=endFrame, width=6)
        # self.ref2 = IntOption(parent, 2, "Frame of reference 2", startFrame, None, min=startFrame, max=endFrame, width=6)

    def Apply(self):
        ref = self.ref.get()
        # ref2 = self.ref2.get()
        ModalDialog.Cancel(self, value=(ref, ))

    # def Cancel(self):
    # ModalDialog.Cancel(self, value=None)

    def destroy(self):
        self.movie = None
        ModelessDialog.destroy(self)


class Density(ModalDialog):
    buttons = ('OK', 'Cancel')

    def __init__(self):
        self.perform_calculation = False
        ModalDialog.__init__(self)

    def fillInUI(self, parent):
        import Tkinter

        Tkinter.Label(parent,
                      text="Compute the number of structures per neurons.\nThis calculation can take SEVERAL MINUTES...",
                      relief="ridge", bd=4).pack()
        self.perform_calculation = True

    def OK(self):
        ModalDialog.Cancel(self, value=(self.perform_calculation, ))

class Plot1D(MPLDialog):
    """
    plot 1D profile of the selected cell
    """
    title = "Self-Organizing Map"
    provideStatus = True

    def __init__(self, **kw):
        """
        min_value and max_value are the limits for the slider on the U-matrix
        max_path_value maximum path distance
        """
        MPLDialog.__init__(self, **kw)

    def fillInUI(self, parent):
        from matplotlib.figure import Figure
        self.figure = Figure()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
        fc = FigureCanvasTkAgg(self.figure, master=parent)
        fc.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.figureCanvas = fc

    def add_subplot(self, *args):
        return self.figure.add_subplot(*args)

    def delaxes(self, ax):
        self.figure.delaxes(ax)

    def draw(self):
        self.figureCanvas.draw()
