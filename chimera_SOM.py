#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 04
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys; sys.path.append('.') 
from plotdialog import PlotDialog 
import numpy
import Combine
from chimera import openModels
from collections import OrderedDict

class UmatPlot(PlotDialog): 
    def __init__(self, movie): 
        PlotDialog.__init__(self) 
        self.movie = movie
        self.matrix = numpy.load('umat.npy')
        self.frame_map = numpy.load('frame_map.npy')
        self.selected_neurons = OrderedDict([])
        self.colors = [] # colors of the dot in the map
        self.subplot = self.add_subplot(1,1,1) 
        self._displayData() 
        self.registerPickHandler(self.onPick)
        self.figureCanvas.mpl_connect("key_press_event", self.onKey)
        self.figureCanvas.mpl_connect("key_release_event", self.offKey)
        self.keep_selection = False
        self.init_models = set(openModels.list())

    def _displayData(self): 
        ax = self.subplot
        ax.clear() 
        for i, neuron in enumerate(self.selected_neurons):
            y, x = neuron
            y+=.5
            x+=.5
            if self.keep_selection and self.colors[i] == 'g':
                ax.scatter(x, y, c=self.colors[i], edgecolors='white')
            elif not self.keep_selection:
                ax.scatter(x, y, c=self.colors[i], edgecolors='white')
        nx,ny = self.matrix.shape
        fig = ax.imshow(self.matrix, interpolation='nearest', extent=(0,ny,nx,0), picker=True)
        #fig.colorbar
        self.figure.canvas.draw()

    def close_current_models(self):
        self.selected_neurons = OrderedDict([])
        self.colors = []
        current_models = set(openModels.list())
        models_to_close = current_models - self.init_models
        openModels.close(models_to_close)

    def display_frame(self, frame_id):
        self.movie.currentFrame.set(frame_id)
        self.movie.LoadFrame()

    def add_model(self, name):
        mol = self.movie.model.Molecule()
        Combine.cmdCombine([mol], name=name)

    def onPick(self, event):
        x,y = event.mouseevent.xdata, event.mouseevent.ydata
        j,i = int(x), int(y)
        if not self.keep_selection:
            self.close_current_models()
        frame_id = self.frame_map[i,j]
        if not numpy.isnan(frame_id):
            frame_id = int(frame_id)
            if self.keep_selection:
                self.colors.append('g')
            else:
                self.colors.append('r')
            self.display_frame(frame_id)
            self.selected_neurons[(i,j)] = openModels.list()[-1]
            self._displayData()
            if self.keep_selection:
                self.add_model(name='%d,%d'%(i,j))

    def onKey(self, event):
        if event.key == 'control':
            self.keep_selection = True

    def offKey(self, event):
        if event.key == 'control':
            self.keep_selection = False

from chimera.extension import manager
from Movie.gui import MovieDialog
movies = [inst for inst in manager.instances if isinstance(inst, MovieDialog)]
if len(movies) != 1:
    raise AssertionError("not exactly one MD Movie")
movie = movies[0]
UmatPlot(movie) 
