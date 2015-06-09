#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 05
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys;

sys.path.append('.')
from plotdialog import PlotDialog
import numpy
import Combine
from chimera import update
from chimera import openModels
from collections import OrderedDict
import Midas


class UmatPlot(PlotDialog):
    def __init__(self, movie):
        PlotDialog.__init__(self)
        self.movie = movie
        data = numpy.load('som.dat')
        self.matrix = data['unfolded_umat']
        self.change_of_basis = data['change_of_basis']
        self.rep_map = self.unfold_matrix(data['representatives'])  # map of representative structures
        self.bmus = numpy.asarray([self.change_of_basis[tuple(e)] for e in data['bmus']])
        self.selected_neurons = OrderedDict([])
        self.colors = []  # colors of the dot in the map
        self.subplot = self.add_subplot(1, 1, 1)
        self._displayData()
        movie.triggers.addHandler(self.movie.NEW_FRAME_NUMBER, self.update_bmu, None)
        self.registerPickHandler(self.onPick)
        self.figureCanvas.mpl_connect("key_press_event", self.onKey)
        self.figureCanvas.mpl_connect("key_release_event", self.offKey)
        self.keep_selection = False
        self.init_models = set(openModels.list())
        self.i, self.j = None, None # current neuron

    def unfold_matrix(self, matrix):
        """
        unfold the given matrix given self.change_of_basis
        """
        unfolded_matrix = numpy.ones_like(self.matrix) * numpy.nan
        for k in self.change_of_basis.keys():
            t = self.change_of_basis[k]  # tuple
            unfolded_matrix[t] = matrix[k]
        return unfolded_matrix

    def update_bmu(self, event_name, empty, frame_id):
        bmu = self.bmus[frame_id - 1]
        y, x = bmu
        y += .5
        x += .5
        if not self.keep_selection:
            ax = self.subplot
            ax.clear()
            ax.scatter(x, y, c='r', edgecolors='white')
            nx, ny = self.matrix.shape
            ax.imshow(self.matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
            self.figure.canvas.draw()

    def _displayData(self):
        ax = self.subplot
        ax.clear()
        for i, neuron in enumerate(self.selected_neurons):
            y, x = neuron
            y += .5
            x += .5
            if self.keep_selection:
                ax.scatter(x, y, c=self.colors[i], edgecolors='white')
            elif not self.keep_selection:
                ax.scatter(x, y, c=self.colors[i], edgecolors='white')
        nx, ny = self.matrix.shape
        ax.imshow(self.matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
        self.figure.canvas.draw()

    def close_current_models(self):
        self.selected_neurons = OrderedDict([])
        self.colors = []
        current_models = set(openModels.list())
        models_to_close = current_models - self.init_models
        openModels.close(models_to_close)
        update.checkForChanges() # to avoid memory leaks (see: http://www.cgl.ucsf.edu/chimera/docs/ProgrammersGuide/faq.html)

    def display_frame(self, frame_id):
        self.movie.currentFrame.set(frame_id)
        self.movie.LoadFrame()

    def add_model(self, name):
        mol = self.movie.model.Molecule()
        Combine.cmdCombine([mol], name=name)

    def update_model_color(self):
        model_id = openModels.listIds()[-1][0]
        if not self.keep_selection:
            Midas.color('orange red', '#%d'%model_id)
        else:
            Midas.color('forest green', '#%d'%model_id)
            for model in self.init_models:
                Midas.color('forest green', '#%d'%model.id)

    def onPick(self, event):
        if not self.keep_selection:
            self.close_current_models()
        else:
            if len(self.selected_neurons) == 1:
                if self.i is not None and self.j is not None:
                    self.add_model(name='%d,%d'%(self.i, self.j))
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        self.j, self.i = int(x), int(y)
        if (self.i, self.j) not in self.selected_neurons.keys():
            frame_id = self.rep_map[self.i, self.j] + 1
            if not numpy.isnan(frame_id):
                frame_id = int(frame_id)
                if self.keep_selection:
                    self.colors.append('g')
                else:
                    self.colors.append('r')
                self.display_frame(frame_id)
                if self.keep_selection:
                    self.add_model(name='%d,%d' % (self.i, self.j))
                self.selected_neurons[(self.i, self.j)] = openModels.list()[-1]
                self.update_model_color()
        else:
            model_to_del = self.selected_neurons[(self.i, self.j)]
            if model_to_del not in self.init_models:
                openModels.close([model_to_del])
                del self.selected_neurons[(self.i, self.j)]
        self._displayData()

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
