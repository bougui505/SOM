#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 17
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys;

sys.path.append('.')
from plotdialog import PlotDialog
from plotdialog import RMSD
from plotdialog import Density
import numpy
import Combine
from chimera import update
from chimera import openModels
from chimera import numpyArrayFromAtoms
from chimera.match import matchPositions
from Movie.analysis import analysisAtoms, AnalysisError
from collections import OrderedDict
import Midas
import matplotlib


class UmatPlot(PlotDialog):
    def __init__(self, movie):
        self.movie = movie
        self.data = numpy.load('som.dat')
        self.init_som_shape = self.data['representatives'].shape # initial som shape
        self.matrix = self.data['unfolded_umat']
        self.minimum_spanning_tree = self.data['minimum_spanning_tree']
        self.min_uvalue = numpy.nanmin(self.matrix)
        PlotDialog.__init__(self, self.min_uvalue, numpy.nanmax(self.matrix),
                            self.dijkstra().max())
        self.master = self._master
        self.displayed_matrix = self.matrix
        self.selection_mode = 'Cell'
        self.unfold = self.data['change_of_basis'] # to unfold the cell indexes
        self.fold = {v:k for k, v in self.unfold.iteritems()} # to fold the cell indexes
        self.rep_map = self.unfold_matrix(self.data['representatives'])  # map of representative structures
        self.bmus = numpy.asarray([self.unfold[tuple(e)] for e in self.data['bmus']])
        self.selected_neurons = OrderedDict([])
        self.colors = []  # colors of the dot in the map
        self.subplot = self.add_subplot(1, 1, 1)
        self.colorbar = None
        self.cluster_map = None
        self.highlighted_cluster = None
        self._displayData()
        movie.triggers.addHandler(self.movie.NEW_FRAME_NUMBER, self.update_bmu, None)
        self.registerPickHandler(self.onPick)
        self.figureCanvas.mpl_connect("key_press_event", self.onKey)
        self.figureCanvas.mpl_connect("key_release_event", self.offKey)
        self.keep_selection = False
        self.init_models = set(openModels.list())
        self.i, self.j = None, None  # current neuron
        self.rmsd_list = None
        self.rep_rmsd = None
        self.density = None
        self.ctrl_pressed = False
        self.motion_notify_event = None

    def switch_matrix(self, value):
        if self.display_option.getvalue() == "U-matrix" or self.display_option.getvalue() is None:
            self.displayed_matrix = self.matrix
        elif self.display_option.getvalue() == "Density":
            if self.density is None:
                dlg = Density()  # Dialog
                if dlg.run(self.master):
                    self.get_density()
            if self.density is not None:
                self.displayed_matrix = self.density
        elif self.display_option.getvalue() == "Closest frame id":
            self.displayed_matrix = self.rep_map
        elif self.display_option.getvalue() == "RMSD":
            if self.rep_rmsd is None:
                dlg = RMSD(self.movie)  # the frame of reference to compute RMSD on
                user_input = dlg.run(self.master)
                if user_input is not None:
                    frame_ref = user_input[0]
                    self.rmsd_per_representative(frame_ref=frame_ref)
            if self.rep_rmsd is not None:
                self.displayed_matrix = self.rep_rmsd
        self._displayData()

    def update_selection_mode(self, value):
        """

        Change the mouse behaviour for selection

        """
        self.selection_mode = self.selection_mode_menu.getvalue() # 'Cell' or 'Cluster'
        if self.selection_mode == 'Cell':
            self.figureCanvas.mpl_disconnect(self.motion_notify_event)
            self.highlighted_cluster = None
        elif self.selection_mode == 'Cluster':
            self.motion_notify_event = self.figureCanvas.mpl_connect("motion_notify_event", self.highlight_cluster)

    def get_clusters(self, value):
        """
        Define clusters with the threshold given by the slider dialog (Cluster())
        """
        threshold = self.slider.get()
        self.cluster_map = self.matrix <= threshold
        self._displayData()

    def get_density(self):
        """
        Compute the number of structures per neuron
        """
        density = numpy.zeros_like(self.matrix)
        total = len(self.bmus)
        for i, e in enumerate(self.bmus):
            self.status('Computing population per cell: %.4f/1.0000' % (float(i + 1) / total, ))
            i, j = e
            density[i, j] += 1
        density[density == 0] = numpy.nan
        self.density = density


    def unfold_matrix(self, matrix):
        """
        unfold the given matrix given self.unfold
        """
        unfolded_matrix = numpy.ones_like(self.matrix) * numpy.nan
        for k in self.unfold.keys():
            t = self.unfold[k]  # tuple
            unfolded_matrix[t] = matrix[k]
        return unfolded_matrix

    def fold_matrix(self, matrix):
        """
        fold the given matrix given self.fold
        """
        folded_matrix = numpy.ones(self.init_som_shape) * numpy.nan
        for k in self.fold.keys():
            t = self.fold[k]  # tuple
            folded_matrix[t] = matrix[k]
        return folded_matrix

    def rmsd_per_representative(self, frame_ref=1):
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
                rmsd = self.compute_rmsd(frame_id + 1, frame_ref)
            else:
                rmsd = numpy.nan
            self.status('Computing RMSD per cell: %.4f/1.0000, RMSD=%.2f' % (float(i + 1) / total, rmsd))
            self.rep_rmsd.append(rmsd)
        self.rep_rmsd = numpy.asarray(self.rep_rmsd)
        nx, ny = self.rep_map.shape
        self.rep_rmsd = self.rep_rmsd.reshape((nx, ny))

    def rmsd_per_frame(self):
        """
        compute the RMSD from the first frame for each frame of the loaded trajectory
        :return:
        """
        self.rmsd_list = []
        for i in range(self.movie.startFrame, self.movie.endFrame + 1):
            self.rmsd_list.append(self.compute_rmsd(i))

    def compute_rmsd(self, frame, frame_ref=1):
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
            if self.cluster_map is not None:
                ax.contour(self.cluster_map, 1, colors='white', linewidths=2.5, extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
                ax.contour(self.cluster_map, 1, colors='red', extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
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
        heatmap = ax.imshow(self.displayed_matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
        if self.cluster_map is not None:
            ax.contour(self.cluster_map, 1, colors='white', linewidths=2.5, extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
            ax.contour(self.cluster_map, 1, colors='red', extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
        if self.highlighted_cluster is not None:
            ax.contour(self.highlighted_cluster, 1, colors='white', linewidths=2.5,
                        extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
            ax.contour(self.highlighted_cluster, 1, colors='green', extent=(0, ny, 0, nx),
                        origin='lower') # display the contours for cluster

        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(heatmap)
        else:
            self.colorbar.update_bruteforce(heatmap)
        self.figure.canvas.draw()

    def close_current_models(self):
        self.selected_neurons = OrderedDict([])
        self.colors = []
        current_models = set(openModels.list())
        models_to_close = current_models - self.init_models
        openModels.close(models_to_close)
        update.checkForChanges()  # to avoid memory leaks (see: http://www.cgl.ucsf.edu/chimera/docs/ProgrammersGuide/faq.html)

    def display_frame(self, frame_id):
        self.movie.currentFrame.set(frame_id)
        self.movie.LoadFrame()

    def add_model(self, name):
        mol = self.movie.model.Molecule()
        Combine.cmdCombine([mol], name=name)

    def update_model_color(self):
        model_id = openModels.listIds()[-1][0]
        if not self.keep_selection:
            Midas.color('orange red', '#%d' % model_id)
        else:
            Midas.color('forest green', '#%d' % model_id)
            for model in self.init_models:
                Midas.color('forest green', '#%d' % model.id)

    def onPick(self, event):
        resolution= self.max_path_value/100 # 10x chosen arbitrarily
        if self.selection_mode == 'Cell': # Select unique cells
            if event.mouseevent.button == 3 or self.ctrl_pressed:
                self.keep_selection = True
            else:
                self.keep_selection = False

            if not self.keep_selection:
                self.close_current_models()
            else:
                if len(self.selected_neurons) == 1:
                    if self.i is not None and self.j is not None:
                        self.add_model(name='%d,%d' % (self.i, self.j))
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
            self.get_basin(None) # to display the basin around the selected cell
            #self._displayData() # commented as it's already done by self.get_basin(None) above
        elif self.selection_mode == 'Cluster' and event.mouseevent.button == 1:
            self.close_current_models()
            if self.highlighted_cluster is not None:
                self.highlighted_cluster[numpy.isnan(self.highlighted_cluster)] = False
                self.highlighted_cluster = numpy.bool_(self.highlighted_cluster)
                frame_ids = self.rep_map[self.highlighted_cluster] + 1
                frame_ids = frame_ids[~numpy.isnan(frame_ids)]
                n = len(frame_ids)
                if n > 10:
                    frame_ids = frame_ids[::n/10] # take only ten representatives
                for frame_id in frame_ids:
                    frame_id = int(frame_id)
                    self.display_frame(frame_id)
                    self.add_model(name='cluster')
        #Scrolling to modify cluster size            
        elif self.selection_mode == 'Cluster' and event.mouseevent.button == "up":  
            self.slider2.set(self.slider2.get()+resolution)
            self.get_basin(None,display=False)
            self.highlight_cluster(event.mouseevent)
            self._displayData()
        elif self.selection_mode == 'Cluster' and event.mouseevent.button == "down":
            self.slider2.set(self.slider2.get()-resolution)
            self.get_basin(None,display=False)
            self.highlight_cluster(event.mouseevent)
            self._displayData()

    def highlight_cluster(self, event):
        x, y = event.xdata, event.ydata
        threshold = self.slider2.get() # threshold for slider 2
        if x is not None and y is not None:
            self.j, self.i = int(x), int(y)
            if self.fold.has_key((self.i,self.j)):
                if threshold > 0:
                    self.get_basin(None, display=False)
                    self.highlighted_cluster = self.cluster_map
                else:
                    cell = self.fold[(self.i, self.j)]
                    if self.cluster_map[self.i,self.j]:
                        self.highlighted_cluster = self.pick_up_cluster((self.i,self.j))
                    else:
                        self.highlighted_cluster = None
                self._displayData()

    def neighbor_dim2_toric(self, p, s):
        """Efficient toric neighborhood function for 2D SOM.
        """
        x, y = p
        X, Y = s
        xm = (x - 1) % X
        ym = (y - 1) % Y
        xp = (x + 1) % X
        yp = (y + 1) % Y
        return [(xm, ym), (xm, y), (xm, yp), (x, ym), (x, yp), (xp, ym), (xp, y), (xp, yp)]

    def get_neighbors_of_area(self, cluster_map):
        """
        return the neighboring indices of an area defined by a boolean array
        """
        neighbors = []
        shape = cluster_map.shape
        for cell in numpy.asarray(numpy.where(cluster_map)).T:
            for e in self.neighbor_dim2_toric(cell, shape):
                if not cluster_map[e]:
                    neighbors.append(e)
        return neighbors


    def pick_up_cluster(self, starting_cell):
        """

        pick up a cluster according to connexity

        """
        cluster_map = self.fold_matrix(self.cluster_map)
        cell = self.fold[starting_cell]
        visit_mask = numpy.zeros(self.init_som_shape, dtype=bool)
        visit_mask[cell] = True
        checkpoint = True
        while checkpoint:
            checkpoint = False
            for e in self.get_neighbors_of_area(visit_mask):
                if cluster_map[e]:
                    visit_mask[e] = True
                    checkpoint = True
        return self.unfold_matrix(visit_mask)

    def onKey(self, event):
        if event.key == 'control':
            self.ctrl_pressed = True

    def offKey(self, event):
        if event.key == 'control':
            self.ctrl_pressed = False

    def get_basin(self, value, display=True):
        """
        Define basin with the threshold given by the slider dialog
        """
        threshold = self.slider2.get()
        if self.i is not None and self.j is not None\
            and threshold > 0 and self.fold.has_key((self.i, self.j)):
            cell = self.fold[(self.i, self.j)]
            self.cluster_map = self.unfold_matrix(self.dijkstra(starting_cell=cell, threshold=threshold) != numpy.inf)
        else:
            self.get_clusters(None)
        if display:
            self._displayData()

    def dijkstra(self, starting_cell = None, threshold = numpy.inf):
        """

        Apply dijkstra distance transform to the SOM map.
        threshold: interactive threshold for local clustering

        """
        ms_tree = self.minimum_spanning_tree
        nx, ny = self.init_som_shape
        nx2, ny2 = ms_tree.shape
        visit_mask = numpy.zeros(nx2, dtype=bool)
        m = numpy.ones(nx2) * numpy.inf
        if starting_cell is None:
            cc = numpy.unravel_index(ms_tree.argmin(), (nx2, ny2))[0]  # current cell
        else:
            cc = numpy.ravel_multi_index(starting_cell, (nx, ny))
        m[cc] = 0
        while (~visit_mask).sum() > 0:
            neighbors = [e for e in numpy.where(ms_tree[cc] != numpy.inf)[0] if not visit_mask[e]]
            for e in neighbors:
                d = ms_tree[cc, e] + m[cc]
                if d < m[e]:
                    m[e] = d
            visit_mask[cc] = True
            m_masked = numpy.ma.masked_array(m, visit_mask)
            cc = m_masked.argmin()
            if m[m != numpy.inf].max() > threshold:
                break
        return m.reshape((nx, ny))

from chimera.extension import manager
from Movie.gui import MovieDialog

movies = [inst for inst in manager.instances if isinstance(inst, MovieDialog)]
if len(movies) != 1:
    raise AssertionError("not exactly one MD Movie")
movie = movies[0]
UmatPlot(movie)
