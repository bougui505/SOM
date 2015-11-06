#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 25
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys;

sys.path.append('.')
from plotdialog import PlotDialog
from plotdialog import RMSD
from plotdialog import SelectClusterMode
from plotdialog import Density
from plotdialog import Projection
from plotdialog import Add_experimental_data
from plotdialog import Plot1D
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
import pickle
import chimera

from chimera import runCommand



class UmatPlot(PlotDialog):
    def __init__(self, movie):
        self.movie = movie
        self.projections = {} # Dictionnary containing all the data projections made by the user
        self.data = numpy.load('som.dat')
        self.init_som_shape = self.data['representatives'].shape # initial som shape
        self.matrix = self.data['unfolded_umat']
        self.minimum_spanning_tree = self.data['minimum_spanning_tree']
        self.local_minima = self.data['local_minima'] # Local minima positions of
                                                      # the unfolded U-matrix
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
        self.movie_id = movie.model.id
        self.i, self.j = None, None  # current neuron
        self.rmsd_list = None
        self.rep_rmsd = None
        self.density = self.data['density']
        self.ctrl_pressed = False
        self.motion_notify_event = None
        self.slice_id = 0 # slice of the matrix to display for high dimensional data
        self.plot1D = None # 1D plot for multidimensional features
        self.feature_item = self.feature_selection.getvalue() # 1D feature to display
        self.load_projections() # Loading user defined projections into plugin
        self.clustermode = (1,'Frames') # to display either Density map or ensemble of frames
        self.experimental_intensities = None

    def switch_matrix(self, value):
        if self.display_option.getvalue() == "U-matrix" or self.display_option.getvalue() is None:
            self.displayed_matrix = self.matrix
            self.slice_matrix(None) # to update the slicer menu
        elif self.display_option.getvalue() == "Density":
            if self.density is None:
                dlg = Density()  # Dialog
                if dlg.run(self.master):
                    self.get_density()
            if self.density is not None:
                self.displayed_matrix = self.density
            self.slice_matrix(None) # to update the slicer menu
        elif self.display_option.getvalue() == "Closest frame id":
            self.displayed_matrix = self.rep_map
            self.slice_matrix(None) # to update the slicer menu
        elif self.display_option.getvalue() == "RMSD":
            if self.rep_rmsd is None:
                dlg = RMSD(self.movie)  # the frame of reference to compute RMSD on
                user_input = dlg.run(self.master)
                if user_input is not None:
                    frame_ref = user_input[0]
                    self.rmsd_per_representative(frame_ref=frame_ref)
            if self.rep_rmsd is not None:
                self.displayed_matrix = self.rep_rmsd
            self.slice_matrix(None) # to update the slicer menu
        else: # user defined projection
            self.displayed_matrix = self.projections[self.display_option.getvalue()][0] # 0 is the projection, 1 is the standard deviation
            self.slice_matrix(None) # to update the slicer menu
        self._displayData()

    def project_data(self):
        """

        project data onto the map and add an entry to self.display_option to
        visualize the resulting map

        """
        dlg = Projection() # dialog
        user_input = dlg.run(self.master)
        if user_input is not None:
            item, filename, read_header = user_input
        else:
            item, filename = None, None
        if item is not None and filename is not None:
            if self.density is None:
                dlg = Density()  # Dialog
                if dlg.run(self.master):
                    self.get_density()
            if self.density is not None: # the user didn't cancel the density computation:
                if not read_header:
                    data = numpy.genfromtxt(filename)
                    feature_names = OrderedDict([(str(i), i) for i in range(data[0].size)]) # feature names for slice
                else:
                    data = numpy.genfromtxt(filename, names=True)
                    feature_names = OrderedDict([(name, i) for i, name in enumerate(data.dtype.names)]) # feature names for slice
                try:
                    n_features = len(data[0])
                except TypeError:
                    n_features = 1
                if n_features == 1:
                    projection_map = numpy.zeros_like(self.matrix)
                    std_map = numpy.zeros_like(self.matrix) # for standard deviation
                else:
                    nx, ny = self.matrix.shape
                    projection_map = numpy.zeros((nx, ny, n_features))
                    std_map = numpy.zeros((nx, ny, n_features)) # for standard deviation
                total = len(self.bmus)
                for i, bmu in enumerate(self.bmus):
                    self.status('Data projection: %.4f/1.0000' % (float(i + 1) / total, ))
                    bmu = tuple(bmu)
                    if n_features > 1:
                        projection_map[bmu] += numpy.asarray(list(data[i]), dtype=float)
                    else:
                        projection_map[bmu] += data[i]
                if n_features == 1:
                    projection_map = projection_map / self.density
                else:
                    projection_map = projection_map / self.density[:,:,None]
                    self.feature_items.append(item) # add a 1D feature to display
                    self.feature_selection.setitems(self.feature_items)
                for i, bmu in enumerate(self.bmus):
                    self.status('Computing standard deviation: %.4f/1.0000' % (float(i + 1) / total, ))
                    bmu = tuple(bmu)
                    if n_features > 1:
                        std_map[bmu] += (numpy.asarray(list(data[i]), dtype=float)-projection_map[bmu])**2
                    else:
                        std_map[bmu] += (data[i]-projection_map[bmu])**2
                if n_features == 1:
                    std_map = numpy.sqrt(std_map / self.density)
                else:
                    std_map = numpy.sqrt(std_map / self.density[:,:,None])
                self.display_option_items.append(item)
                self.display_option.setitems(self.display_option_items)
                self.projections[item] = (projection_map, std_map, feature_names)
        self.save_projections()

    def add_experimental_data(self):
        """

        Add experimental data to perform data driven clustering. The clustering
        threshold is based on the deviation between the selected neurons of the
        map and the experimental values (chi based).

        When experimental data are loaded, the feature_map is used to obtain
        the chi_map. The chi value between each cell of the chi_map is computed
        against the experimental_data.

        """
        dlg = Add_experimental_data() # dialog
        user_input = dlg.run(self.master) # (filename,)
        filename = user_input[0]

        # The filename containing the experimental data should be a 2-column
        # filename containing data for the first column and standard deviation
        # for the second one.

        experimental_data = numpy.genfromtxt(filename)
        self.experimental_intensities = experimental_data[:,0]
        self.experimental_std = experimental_data[:,1]
        feature_map = self.data['projections']['feature_map'][0]
        delta_map = self.experimental_intensities - feature_map
        if (self.experimental_std == numpy.zeros_like(self.experimental_std)).all(): # zero std
            std = numpy.ones_like(self.experimental_std)
        else:
            std = self.experimental_std
        chi_map = numpy.sqrt( (delta_map**2 / std**2).sum(axis=2)/\
                    len(self.experimental_intensities) )
        feature_names = OrderedDict([(str(i), i) for i in range(self.experimental_intensities.size)])
        self.display_option_items.append('chi_map')
        self.display_option.setitems(self.display_option_items)
        self.projections['chi_map'] = (chi_map, numpy.zeros_like(chi_map), feature_names)

    def save_projections(self, outfile='som.dat'):
        """

        Save projections into som.dat file

        """
        self.status('saving data in %s'%outfile)
        self.data['projections'] = self.projections
        self.data['density'] = self.density
        f = open(outfile,'wb')
        pickle.dump(self.data, f, 2)
        f.close()
        self.status('done')

    def load_projections(self):
        """

        Load projection into plugin. Read data from self.data.

        """
        if self.data.has_key('density'):
            self.density = self.data['density']
        if self.data.has_key('projections'):
            self.projections = self.data['projections']
            for item in self.projections.keys():
                self.status('Loading projection: %s'%item)
                # update the display menu
                self.display_option_items.append(item)
                self.display_option.setitems(self.display_option_items)
                if len(self.projections[item][0].shape) > 2:
                    self.feature_items.append(item) # add a 1D feature to display
                    self.feature_selection.setitems(self.feature_items)

    def update_selection_mode(self, value):
        """

        Change the mouse behaviour for selection

        """
        self.selection_mode = self.selection_mode_menu.getvalue() # 'Cell' or 'Cluster'
        if self.selection_mode == 'Cell':
            self.figureCanvas.mpl_disconnect(self.motion_notify_event)
            self.highlighted_cluster = None
        elif self.selection_mode == 'Cluster':
            #dialog box to select cluster mode
            dlg = SelectClusterMode(self.movie)  
            params=dlg.run(self.master)
            if params is not None:
                self.clustermode = params
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
            if len(self.displayed_matrix.shape) == 2: # two dimensional array
                ax.imshow(self.displayed_matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
                ax.scatter(self.local_minima[1]+.5, self.local_minima[0]+.5,
                           c='#FFA700', alpha=.5) # Plot the local minima
                                                  # of the U-matrix
            else: # we must slice the matrix
                ax.imshow(self.displayed_matrix[:,:,self.slice_id], interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
                ax.scatter(self.local_minima[1]+.5, self.local_minima[0]+.5,
                           c='#FFA700', alpha=.5) # Plot the local minima
                                                  # of the U-matrix
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
        if len(self.displayed_matrix.shape) == 2: # two dimensional array
            heatmap = ax.imshow(self.displayed_matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
            ax.scatter(self.local_minima[1]+.5, self.local_minima[0]+.5,
                       c='#FFA700', alpha=.5) # Plot the local minima
                                              # of the U-matrix
        else: # we must slice the matrix
            heatmap = ax.imshow(self.displayed_matrix[:,:,self.slice_id], interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
            ax.scatter(self.local_minima[1]+.5, self.local_minima[0]+.5,
                       c='#FFA700', alpha=.5) # Plot the local minima
                                              # of the U-matrix
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
        self.display_features(None)


    def display_features(self, value):
        """

        To display 1D features in a seperate plot

        """
        self.feature_item = self.feature_selection.getvalue() # 1D feature to display
        if self.feature_item is not None and self.projections.has_key(self.feature_item):
            feature_map = self.projections[self.feature_item][0] # 0 is the projection, 1 is the standard deviation
            std_map = self.projections[self.feature_item][1] # 0 is the projection, 1 is the standard deviation
            feature_names = self.projections[self.feature_item][2] # 2 is the names of the features

            if self.i is not None and self.j is not None and self.density[self.i, self.j] > 0:
                if self.plot1D is None:
                    self.plot1D = Plot1D()
                    self.subplot1D = self.plot1D.add_subplot(1, 1, 1)
                self.subplot1D.clear()
                ax = self.subplot1D
                if self.selection_mode == 'Cell':
                    width = .8 # width of the bar of the barplot
                    n = len(self.selected_neurons)
                    if self.experimental_intensities is not None:
                        n+=1 # One more bar plot
                    for i, neuron in enumerate(self.selected_neurons):
                        features = feature_map[neuron].flatten()
                        std_features = std_map[neuron].flatten()
                        nx = features.size
                        if self.experimental_intensities is None:
                            x = numpy.arange(nx) + i*width/n
                        else: # plot experimental data
                            x = numpy.arange(nx)
                            ax.bar(x, self.experimental_intensities,
                                   yerr=self.experimental_std, align='center',
                                   width=width/n, color='#D62D20')
                            x = numpy.arange(nx) + (i+1)*width/n
                        if i == 0:
                            ax.bar(x, features, yerr=std_features,
                                    align='center', width=width/n, color='r') # red barplot for the first selected neuron
                        else:
                            ax.bar(x, features, yerr=std_features,
                                    align='center', width=width/n, color='g') # green for the other
                    ax.set_xticks(numpy.arange(features.size))
                    ax.set_xticklabels(feature_names.keys(), rotation=75)
                    self.plot1D.draw()
                elif self.selection_mode == 'Cluster':
                    if self.highlighted_cluster is not None:
                        self.highlighted_cluster[numpy.isnan(self.highlighted_cluster)] = False
                        self.highlighted_cluster = numpy.logical_and(numpy.bool_(self.highlighted_cluster),
                                                                    self.density > 0)
                        mean_features = feature_map[self.highlighted_cluster].mean(axis=0)
                        std_features = std_map[self.highlighted_cluster].mean(axis=0)
                        ax.bar(numpy.arange(mean_features.size), mean_features, yerr=std_features,
                                            align='center')
                        ax.set_xticks(numpy.arange(mean_features.size))
                        ax.set_xticklabels(feature_names.keys(), rotation=75)
                        self.plot1D.draw()

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
            Midas.color('orange red', '#%d' % self.movie_id)
        else:
            Midas.color('forest green', '#%d' % model_id)
            Midas.color('byhet', '#%d' % model_id)
            Midas.color('forest green', '#%d' % self.movie_id)
        Midas.color('byhet', '#%d' % self.movie_id)

    def onPick(self, event):
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
            if event.mouseevent.button == 1 or event.mouseevent.button == 3:
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
                if self.clustermode[1] == "Frames":
                    if n > 10:
                        frame_ids = frame_ids[::n/10] # take only ten representatives
                    for frame_id in frame_ids:
                        frame_id = int(frame_id)
                        self.display_frame(frame_id)
                        self.add_model(name='cluster')
                elif self.clustermode[1] == 'Density':
                    if n > 100:
                        frame_ids = frame_ids[::n/100] # take only 100 representatives
                    trajMol = self.movie.model._mol

                    if chimera.selection.currentEmpty():
                        #something to select all atoms
                        runCommand("select #%d" % self.movie_id)
                    atoms = [a for a in chimera.selection.currentAtoms() if a.molecule == trajMol]    
                    name="ClusterDensityMap"
                    self.computeVolume(atoms, frame_ids=frame_ids,volumeName=name, spacing=self.clustermode[0])
                    model_id=openModels.listIds()[-1][0]
                    #Midas.color('aquamarine,s', '#%d' %model_id)
                    runCommand("volume #%d level 50. color aquamarine style surface" %model_id)



    def slice_matrix(self, value):
        """
        slice matrix when its dimension is larger than 2
        """
        n_dim = len(self.displayed_matrix.shape) # dimension of the displayed array
        if n_dim > 2: # high dimensional array, we must slice in dimensions !
            feature_names = self.projections[self.display_option.getvalue()][2]
            n_features = self.displayed_matrix.shape[-1]
            if len(self.slice_items) != n_features:
                self.slice_items = feature_names.keys()
                self.slice_selection.setitems(self.slice_items)
            self.slice_id = feature_names[self.slice_selection.getvalue()] # to get the slice index of the corresponding slice name
            self._displayData()
        else:
            self.slice_items = [0, ]
            self.slice_selection.setitems(self.slice_items)


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

    
    def computeVolume(self, atoms, frame_ids, volumeName=None, spacing=0.5, radiiTreatment="ignored"):
        #function taken from Movie/gui.py and tweaked to compute volume based on an array of frame_ids
        from Matrix import xform_matrix
        gridData = {}
        from math import floor
        from numpy import array, float32, concatenate
        from _contour import affine_transform_vertices
        insideDeltas = {}
        include = {}
        sp2 = spacing * spacing
        for fn in frame_ids:
            cs = self.movie.findCoordSet(fn)
            if not cs:
                self.movie.status("Loading frame %d" % fn)
                self.movie._LoadFrame(int(fn), makeCurrent=False)
                cs = self.movie.findCoordSet(fn)

            self.movie.status("Processing frame %d" % fn)
            pts = array([a.coord(cs) for a in atoms], float32)
            if self.movie.holdingSteady:
                if bound is not None:
                    steadyPoints = array([a.coord(cs)
                        for a in steadyAtoms], float32)
                    closeIndices = find_close_points(
                        BOXES_METHOD, steadyPoints,
                        #otherPoints, bound)[1]
                        pts, bound)[1]
                    pts = pts[closeIndices]
                try:
                    xf, inv = self.movie.transforms[fn]
                except KeyError:
                    xf, inv = self.movie.steadyXform(cs=cs)
                    self.movie.transforms[fn] = (xf, inv)
                xf = xform_matrix(xf)
                affine_transform_vertices(pts, xf)
                affine_transform_vertices(pts, inverse)
            if radiiTreatment != "ignored":
                ptArrays = [pts]
                for pt, radius in zip(pts, [a.radius for a in atoms]):
                    if radius not in insideDeltas:
                        mul = 1
                        deltas = []
                        rad2 = radius * radius
                        while mul * spacing <= radius:
                            for dx in range(-mul, mul+1):
                                for dy in range(-mul, mul+1):
                                    for dz in range(-mul, mul+1):
                                        if radiiTreatment == "uniform" \
                                        and min(dx, dy, dz) > -mul and max(dx, dy, dz) < mul:
                                            continue
                                        key = tuple(sorted([abs(dx), abs(dy), abs(dz)]))
                                        if key not in include.setdefault(radius, {}):
                                            include[radius][key] = (dx*dx + dy*dy + dz*dz
                                                    ) * sp2 <= rad2
                                        if include[radius][key]:
                                            deltas.append([d*spacing for d in (dx,dy,dz)])
                            mul += 1
                        insideDeltas[radius] = array(deltas)
                        if len(deltas) < 10:
                            print deltas
                    if insideDeltas[radius].size > 0:
                        ptArrays.append(pt + insideDeltas[radius])
                pts = concatenate(ptArrays)
            # add a half-voxel since volume positions are
            # considered to be at the center of their voxel
            from numpy import floor, zeros
            pts = floor(pts/spacing + 0.5).astype(int)
            for pt in pts:
                center = tuple(pt)
                gridData[center] = gridData.get(center, 0) + 1

        # generate volume
        self.movie.status("Generating volume")
        axisData = zip(*tuple(gridData.keys()))
        minXyz = [min(ad) for ad in axisData]
        maxXyz = [max(ad) for ad in axisData]
        # allow for zero-padding on both ends
        dims = [maxXyz[axis] - minXyz[axis] + 3 for axis in range(3)]
        from numpy import zeros, transpose
        volume = zeros(dims, int)
        for index, val in gridData.items():
            adjIndex = tuple([index[i] - minXyz[i] + 1
                            for i in range(3)])
            volume[adjIndex] = val
        from VolumeData import Array_Grid_Data
        gd = Array_Grid_Data(volume.transpose(),
                    # the "cushion of zeros" means d-1...
                    [(d-1) * spacing for d in minXyz],
                    [spacing] * 3)
        if volumeName is None:
            volumeName = self.movie.ensemble.name
        gd.name = volumeName

        # show volume
        self.movie.status("Showing volume")
        import VolumeViewer
        dataRegion = VolumeViewer.volume_from_grid_data(gd)
        vd = VolumeViewer.volumedialog.volume_dialog(create=True)
        vd.message("Volume can be saved from File menu")
        self.movie.status("Volume shown")

from chimera.extension import manager
from Movie.gui import MovieDialog

movies = [inst for inst in manager.instances if isinstance(inst, MovieDialog)]
if len(movies) != 1:
    raise AssertionError("not exactly one MD Movie")
movie = movies[0]
UmatPlot(movie)
