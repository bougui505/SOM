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

import sys;

sys.path.append('.')
from plotdialog import PlotDialog
import numpy
import Combine
from chimera import update
from chimera import openModels
from chimera import numpyArrayFromAtoms
from chimera.match import matchPositions
from Movie.analysis import analysisAtoms, AnalysisError
from collections import OrderedDict
import Midas


class UmatPlot(PlotDialog):
    def __init__(self, movie):
        PlotDialog.__init__(self)
        self.movie = movie
        data = numpy.load('som.dat')
        self.matrix = data['unfolded_umat']
        self.displayed_matrix = self.matrix
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
        self.rmsd_list = None
        self.rep_rmsd = None
        self.ctrl_pressed = False


    def switch_matrix(self, value):
        if self.mapTypeOption.getvalue() == "U-matrix" or self.mapTypeOption.getvalue() is None:
            self.displayed_matrix = self.matrix
        elif self.mapTypeOption.getvalue() == "Closest frame id":
            self.displayed_matrix = self.rep_map
        elif self.mapTypeOption.getvalue() == "RMSD from first frame":
            if self.rep_rmsd is None:
                self.rmsd_per_representative()
            self.displayed_matrix = self.rep_rmsd
        self._displayData()


    def unfold_matrix(self, matrix):
        """
        unfold the given matrix given self.change_of_basis
        """
        unfolded_matrix = numpy.ones_like(self.matrix) * numpy.nan
        for k in self.change_of_basis.keys():
            t = self.change_of_basis[k]  # tuple
            unfolded_matrix[t] = matrix[k]
        return unfolded_matrix

    def rmsd_per_representative(self):
        """
        compute the RMSD from the first frame for each representative
        :return:
        """
        rep_frames = self.rep_map.flatten()
        self.rep_rmsd = []
        total = len(rep_frames)
        for i, frame_id in enumerate(rep_frames):
            if not numpy.isnan(frame_id):
                frame_id = int(frame_id)
                rmsd = self.compute_rmsd(frame_id+1)
            else:
                rmsd = numpy.nan
            self.status('Computing RMSD per cell: %.4f/1.0000, RMSD=%.2f'%(float(i)/total, rmsd))
            self.rep_rmsd.append(rmsd)
        self.rep_rmsd = numpy.asarray(self.rep_rmsd)
        nx, ny = self.rep_map.shape
        self.rep_rmsd = self.rep_rmsd.reshape((nx,ny))

    def rmsd_per_frame(self):
        """
        compute the RMSD from the first frame for each frame of the loaded trajectory
        :return:
        """
        self.rmsd_list = []
        for i in range(self.movie.startFrame, self.movie.endFrame+1):
            self.rmsd_list.append(self.compute_rmsd(i))

    def compute_rmsd(self, frame):
        """
        compute the rmsd from the first frame for the given frame
        :param frame:
        :return:
        """
        useSel = False
        ignoreBulk = True
        ignoreHyds = True
        metalIons = False
        atoms = analysisAtoms(self.movie.model.Molecule(), useSel, ignoreBulk, ignoreHyds, metalIons)
        frame_ref = 1
        self.movie._LoadFrame(frame_ref, makeCurrent=False)
        ref = numpyArrayFromAtoms(atoms, self.movie.findCoordSet(frame_ref))
        self.movie._LoadFrame(frame, makeCurrent=False)
        current = numpyArrayFromAtoms(atoms, self.movie.findCoordSet(frame))
        rmsd = matchPositions(ref, current)[1]
        return rmsd

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
            ax.imshow(self.displayed_matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
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
        ax.imshow(self.displayed_matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
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


        if event.mouseevent.button==3 or self.ctrl_pressed:
            self.keep_selection = True
        else:
            self.keep_selection = False

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
            self.ctrl_pressed = True

    def offKey(self, event):
        if event.key == 'control':
            self.ctrl_pressed = False


from chimera.extension import manager
from Movie.gui import MovieDialog

movies = [inst for inst in manager.instances if isinstance(inst, MovieDialog)]
if len(movies) != 1:
    raise AssertionError("not exactly one MD Movie")
movie = movies[0]
UmatPlot(movie) 
