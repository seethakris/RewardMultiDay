import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats
from _collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from copy import copy

# For plotting styles
MainFolder = '/Users/seetha/Box Sync/MultiDayData/'
PlottingFormat_Folder = os.path.join(MainFolder, 'Scripts/PlottingTools/')
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

DataDetailsFolder = os.path.join(MainFolder, 'Scripts/AnimalDetails/')
sys.path.append(DataDetailsFolder)
import AnimalDetailsWT

class GetData(object):
    def __init__(self, FolderName):
        self.FolderName = FolderName
        self.animals = [f for f in os.listdir(self.FolderName) if
                        f not in ['.DS_Store']]
        self.trackbins = 5
        self.tracklength = 200
        self.plpc = PlotPCs()

    def get_data_folders(self, animalname):
        imgfilename = [f for f in os.listdir(os.path.join(self.FolderName, animalname)) if f.endswith('.mat')]
        parsed_behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                  allow_pickle=True)
        pf_data = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

        pf_params = np.load(
            os.path.join(self.FolderName, 'PlaceCells', f'%s_placecell_data.npz' % animalname), allow_pickle=True)

        pf_remapping_dict = np.load(
            os.path.join(self.FolderName, animalname, 'PlaceCells', '%s_pcs_sortedbyTask1', animalname),
            allow_pickle=True)
        return imgfilename, parsed_behavior, pf_data, pf_params, pf_remapping_dict

    def get_example_cell(self, ax, cellnumber, animalname, taskstoplot, tasktocompare):
        pf_data = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

        pf_params = pd.read_csv(
            os.path.join(self.FolderName, animalname, 'PlaceCells', f'%s_placecellparams_df.csv' % animalname), index_col=0)

        # Sort cells with high reliability
        cells = pf_params[(pf_params['Reliability']>0.75) & (pf_params['Task']==tasktocompare)]['CellNumber'].to_list()
        # fs, ax = plt.subplots(len(cells), 2, sharex=True, figsize=(20, 20))
        print(np.shape(cells))
        mean_data = []
        num_laps = [33, 25]
        for n1, t in enumerate(taskstoplot):
            filename = [f for f in pf_data if t in f][0]
            data = scipy.io.loadmat(os.path.join(self.FolderName, animalname, 'Behavior', filename))
            # for n2, c in enumerate(cells):
            pc_activity = data['Allbinned_F'][0][cells[cellnumber]].T #[-num_laps[n1]:, :]
                # pc_activity = data['Allbinned_F'][0][c].T
            normalise_data = pc_activity / np.nanmax(pc_activity, 1)[:, np.newaxis]
            normalise_data = np.nan_to_num(normalise_data)
            if n1 > 0:
                mean_data.append(np.nanmean(pc_activity[1, :]))
                ax[-1].plot(pc_activity[0, :])
                ax[-1].plot(pc_activity[1, :])
            else:
                mean_data.append(pc_activity[-1, :])
                ax[-1].plot(pc_activity[-1, :])
            print(t, np.shape(normalise_data))
            ax[n1].imshow(normalise_data, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=1)

        # print(np.corrcoef(mean_data[0], mean_data[1]))

    def combine_placecells_withtask(self, fig, axis, taskstoplot, tasktocompare='Task1'):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        for a in self.animals:
            animalinfo = AnimalDetailsWT.MultiDaysAnimals(a)
            pf_remapping = np.load(
                os.path.join(self.FolderName, a, 'PlaceCells', '%s_pcs_sortedby%s.npy' % (a, tasktocompare)),
                allow_pickle=True).item()

            for t in taskstoplot:
                pc_activity_dict[t] = np.vstack((pc_activity_dict[t], pf_remapping[t])) if pc_activity_dict[
                    t].size else pf_remapping[t]

        pcsortednum = {keys: [] for keys in taskstoplot}
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[tasktocompare], 1))
        for t in taskstoplot:
            pcsortednum[t] = pcsorted

        task_data = pc_activity_dict[tasktocompare][pcsorted, :]
        normalise_data = np.nanmax(task_data, 1)[:, np.newaxis]
        print(np.shape(normalise_data))
        self.plpc.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum,
                                          normalise_data=normalise_data)

    def get_reliable_examples(self, ax, animalname, taskstoplot):

        pf_data = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
        pf_params = pd.read_csv(
            os.path.join(self.FolderName, animalname, 'PlaceCells', f'%s_placecellparams_df.csv' % animalname), index_col=0)

        # print(pf_params.head())
        # Cells with high reliability
        cells = pf_params[(pf_params['FiringRatio']>0.6) & (pf_params['Task']==taskstoplot) & (pf_params['NumPlacecells']==1)]['CellNumber'].to_list()
        cells = cells[0:20]
        print(np.shape(cells))

        filename = [f for f in pf_data if taskstoplot in f][0]
        data = scipy.io.loadmat(os.path.join(self.FolderName, animalname, 'Behavior', filename))
        # fs, ax = plt.subplots(1, len(cells), sharex=True, figsize=(20, 5))
        for n2, c in enumerate(cells[-1:]):
            pc_activity = data['Allbinned_F'][0][c].T
            normalise_data = pc_activity / np.nanmax(pc_activity, 1)[:, np.newaxis]
            normalise_data = np.nan_to_num(normalise_data)
            ax.imshow(normalise_data, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=1)
            reliability = pf_params[(pf_params['CellNumber']==c) & (pf_params['Task']==taskstoplot)]['Reliability']
            firingratio = pf_params[(pf_params['CellNumber']==c) & (pf_params['Task']==taskstoplot)]['FiringRatio']
            print(reliability, firingratio, np.size(pc_activity, 0))
            # ax.set_title('Reliability %0.2f, Firing ratio %0.2f, laps %d' %(reliability, firingratio, np.size(pc_activity, 0)))

    def combine_byreliability(self, fig, axis, taskstoplot, tasktocompare, reliability):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        pcsortednum = {keys: [] for keys in taskstoplot}

        for a in self.animals:
            pf_data = \
                [f for f in os.listdir(os.path.join(self.FolderName, a, 'Behavior')) if
                 (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

            pf_params = pd.read_csv(
                os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecellparams_df.csv' % a), index_col=0)

            if reliability=='High':
                cells_to_plot = pf_params[(pf_params['Task']==tasktocompare) & (pf_params['Reliability']>=0.5)]['CellNumber']
            else:
                cells_to_plot = pf_params[(pf_params['Task']==tasktocompare) & (pf_params['Reliability']<0.5)]['CellNumber']
            for i in pf_data:
                ft = i.find('Task')
                taskname = i[ft:ft + i[ft:].find('_')]
                if taskname not in taskstoplot:
                    continue
                x = scipy.io.loadmat(os.path.join(self.FolderName, a,  'Behavior', i))
                pc_activity = []
                for n, c in enumerate(cells_to_plot):
                    pc_activity.append(np.nanmean((x['Allbinned_F'][0, c]), 1))
                pc_activity = np.asarray(pc_activity)
                pc_activity_dict[taskname] = np.vstack((pc_activity_dict[taskname], pc_activity)) if pc_activity_dict[
                    taskname].size else pc_activity

        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[tasktocompare], 1))
        for t in taskstoplot:
            pcsortednum[t] = pcsorted

        task_data = pc_activity_dict[tasktocompare][pcsorted, :]
        normalise_data = np.nanmax(task_data, 1)[:, np.newaxis]

        self.plpc.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum,normalise_data=normalise_data)

        return pc_activity_dict

    def combine_commoncells(self, fig, axis, taskstoplot, tasktocompare):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        pcsortednum = {keys: [] for keys in taskstoplot}
        celldf = self.get_cells_pertask_peranimal(taskstoplot)

        for a in self.animals:
            pf_data = \
                [f for f in os.listdir(os.path.join(self.FolderName, a, 'Behavior')) if
                 (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

            commoncells = celldf[celldf[taskstoplot].sum(axis=1)==len(taskstoplot)]
            cells_to_plot = commoncells[commoncells['AnimalName']==a]['CellNum']

            for i in pf_data:
                ft = i.find('Task')
                taskname = i[ft:ft + i[ft:].find('_')]
                if taskname not in taskstoplot:
                    continue
                x = scipy.io.loadmat(os.path.join(self.FolderName, a,  'Behavior', i))
                pc_activity = []
                for n, c in enumerate(cells_to_plot):
                    pc_activity.append(np.nanmean((x['Allbinned_F'][0, c]), 1))
                pc_activity = np.asarray(pc_activity)
                pc_activity_dict[taskname] = np.vstack((pc_activity_dict[taskname], pc_activity)) if pc_activity_dict[
                    taskname].size else pc_activity

        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[tasktocompare], 1))
        for t in taskstoplot:
            pcsortednum[t] = pcsorted

        task_data = pc_activity_dict[tasktocompare][pcsorted, :]
        normalise_data = np.nanmax(task_data, 1)[:, np.newaxis]

        self.plpc.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum,normalise_data=normalise_data)

        return pc_activity_dict

    def get_cells_pertask_peranimal(self, taskstoplot):
        numcells_dict = {keys: [] for keys in taskstoplot + ['AnimalName', 'CellNum']}
        for a in self.animals:
            pfparams = np.load(
                os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecell_data.npz' % a), allow_pickle=True)
            numcells = pfparams['numcells']
            sig_PFs = pfparams['sig_PFs_cellnum_revised'].item()
            numPFs = pfparams['numPFs_incells'].item()
            for c in np.arange(numcells):
                numcells_dict['CellNum'].append(c)
                numcells_dict['AnimalName'].append(a)

                for t in taskstoplot:
                    if c in sig_PFs[t]:
                        numcells_dict[t].append(1)
                    else:
                        numcells_dict[t].append(0)
        numcells_df = pd.DataFrame.from_dict(numcells_dict)
        return numcells_df


class PlotPCs(object):
    @staticmethod
    def plot_placecells_pertask(fig, axis, taskstoplot, pc_activity, sorted_pcs, controlflag=0, **kwargs):
        for n, taskname in enumerate(taskstoplot):
            task_data = pc_activity[taskname][sorted_pcs[taskname], :]
            if 'normalise_data' in kwargs.keys():
                normalise_data = task_data / kwargs['normalise_data']
            else:
                normalise_data = task_data / np.nanmax(task_data, 1)[:, np.newaxis]
            # normalise_data = np.nan_to_num(normalise_data)

            img = axis[n].imshow(np.nan_to_num(normalise_data),
                                 aspect='auto', cmap='jet', vmin=0, vmax=1.0)

            axis[n].set_xticks([0, 20, 39])
            axis[n].set_xticklabels([0, 100, 200])
            axis[n].set_xlim((0, 39))
            if controlflag:
                axis[n].set_title('Cntrl: %s' % taskname)
            else:
                axis[n].set_title('Exp: %s' % taskname)

            pf.set_axes_style(axis[n], numticks=4)
        axis[0].set_xlabel('Track Length (cm)')
        axis[0].set_ylabel('Cell')
        axins = PlotPCs.add_colorbar_as_inset(axis[-1])
        cb = fig.colorbar(img, cax=axins, pad=0.2, ticks=[0, 1])
        cb.set_label('Delta f/f')
        cb.ax.tick_params(size=0)

    @staticmethod
    def add_colorbar_as_inset(axes):
        axins = inset_axes(axes,
                           width="5%",  # width = 5% of parent_bbox width
                           height="40%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axes.transAxes,
                           borderpad=0.5,
                           )
        return axins
