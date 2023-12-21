import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import seaborn as sns
from collections import OrderedDict
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import scipy.stats
import h5py
from scipy.optimize import curve_fit

# For plotting styles
PlottingFormat_Folder = '/home/seethakris/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class GetData:
    def __init__(self, animalinfo, FolderName, noreward_task):
        self.FolderName = FolderName
        self.FigureFolder = os.path.join(self.FolderName, 'Figures')
        self.SaveFolder = os.path.join(self.FolderName, 'PlaceCells')
        self.TaskDict = animalinfo['task_dict']
        self.Task_Numframes = animalinfo['task_numframes']
        self.tracklength = animalinfo['tracklength']
        self.trackbins = animalinfo['trackbins']
        self.animalname = animalinfo['animal']
        self.noreward_task = noreward_task

        if not os.path.exists(self.FigureFolder):
            os.mkdir(self.FigureFolder)
        if not os.path.exists(self.SaveFolder):
            os.mkdir(self.SaveFolder)

        self.get_data_folders()
        if animalinfo['v73_flag']:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()

        self.get_lapframes_numcells()
        self.lickstoplap = self.Parsed_Behavior['lick_stop'].item()[self.noreward_task]
        self.lickstopframe = np.where(self.good_lapframes['Task3'] == self.lickstoplap + 1)[0][
            0]  # Task3 is no reward for multiday animal
        print(self.lickstoplap, self.lickstopframe)

        # Find significant place cells
        self.find_sig_PFs_cellnum_bytask()
        self.beginning_cells = self.revise_sig_PFs()
        self.create_populationvector()
        self.calculate_pfparameters()
        # self.correlate_acivity_of_allcellsbytask()
        # # self.common_droppedcells_withTask1()
        # # self.save_analyseddata()

    def create_data_dict(self, TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict

    def get_data_folders(self):
        self.ImgFileName = [f for f in os.listdir(self.FolderName) if f.endswith('.mat')]
        self.Parsed_Behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                       allow_pickle=True)
        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(self.FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

    def find_sig_PFs_cellnum_bytask(self):
        self.sig_PFs_cellnum = self.create_data_dict(self.TaskDict)
        self.numPFS_incells = self.create_data_dict(self.TaskDict)
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            tempx = np.squeeze(np.asarray(np.nan_to_num(x['number_of_PFs'])).T).astype(int)
            print(f'%s : Place cells: %d PlaceFields: %d' % (
                taskname, np.size(np.where(tempx > 0)[0]), np.sum(tempx[tempx > 0])))

            self.sig_PFs_cellnum[taskname] = np.where(tempx > 0)[0]
            self.numPFS_incells[taskname] = tempx[np.where(tempx > 0)[0]]-1

    def revise_sig_PFs(self):
        # Get_gaussianfit
        x = np.arange(0, np.int(self.tracklength / self.trackbins))
        y = self.gaussian(x, 1, 0.01, 5) + np.random.normal(0, 0.2, x.size)
        best_vals, covar = curve_fit(self.gaussian, x, y, p0=[1, 0, 1])
        gaussfit = self.gaussian(x, *best_vals)
        beginning_cell_dict = self.create_data_dict(self.TaskDict)
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            beginning_cell = np.zeros_like(self.sig_PFs_cellnum[taskname])
            count = 0
            for n in np.arange(np.size(self.sig_PFs_cellnum[taskname])):
                    placecell_num = self.numPFS_incells[taskname][n]
                    pc_activity = np.nanmean(
                        x['sig_PFs'][placecell_num][self.sig_PFs_cellnum[taskname][n]], 1)
                    start_bins = x['PF_start_bins'][placecell_num][self.sig_PFs_cellnum[taskname][n]]
                    end_bins = x['PF_end_bins'][placecell_num][self.sig_PFs_cellnum[taskname][n]]
                    tailflag = self.check_for_beginning_transients(pc_activity, start_bins, end_bins, gaussfit)
                    if tailflag:
                        beginning_cell[n] += 1
            beginning_cell_dict[taskname] = beginning_cell
            plt.title(taskname)
            plt.show()
        self.update_PFs(beginning_cell_dict)
        return beginning_cell_dict

    def update_PFs(self, beginning_cells):
        self.sig_PFs_cellnum_revised = self.create_data_dict(self.TaskDict)
        self.sig_PFs_beginning = self.create_data_dict(self.TaskDict)
        self.numPFs_incells_revised = self.create_data_dict(self.TaskDict)
        self.cellid_beg_multiplepfs = self.create_data_dict(self.TaskDict)
        for taskname in self.TaskDict.keys():
            for n in np.arange(np.size(self.sig_PFs_cellnum[taskname])):
                if beginning_cells[taskname][n] > 0 and self.numPFS_incells[taskname][n] == 1:
                    self.sig_PFs_beginning[taskname].append(self.sig_PFs_cellnum[taskname][n])
                else:
                    if beginning_cells[taskname][n] > 0 and self.numPFS_incells[taskname][n] > 1:
                        self.cellid_beg_multiplepfs[taskname].append(self.sig_PFs_cellnum[taskname][n])
                    self.sig_PFs_cellnum_revised[taskname].append(self.sig_PFs_cellnum[taskname][n])
                    self.numPFs_incells_revised[taskname].append(self.numPFS_incells[taskname][n])

    def check_for_beginning_transients(self, data, start_bins, end_bins, gaussfit, threshold=2):
        if start_bins == 1 and np.argmax(data) <= threshold:
            correlation_withgauss = self.check_if_tail(data, gaussfit)
            if correlation_withgauss > 0.7:
                plt.plot(data)
                plt.plot(np.argmax(data), np.max(data), '*', markersize=10)
                return True
            else:
                return False

    def check_if_tail(self, data, gaussfit):
        corr = np.corrcoef(data, gaussfit)[0, 1]
        return corr

    def gaussian(self, x, amp, cen, wid):
        return amp * np.exp(-(x - cen) ** 2 / wid)

    def load_fluorescentdata(self):
        self.Fc3data_dict = self.create_data_dict(self.TaskDict)
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        print(np.shape(data['Fc3']))
        count = 0
        for i in self.TaskDict.keys():
            self.Fc3data_dict[i] = data['Fc3'].T[:,
                                   count:count + self.Task_Numframes[i]]
            print(f'%s : Number of Frames: %d' % (i, np.size(self.Fc3data_dict[i], 1)))
            count += self.Task_Numframes[i]
        if count != np.size(data['Fc3'], 0):
            raise Exception('Error!! File size doesnt match')
        else:
            print('All good')

    def get_lapframes_numcells(self):
        self.good_lapframes = self.create_data_dict(self.TaskDict)
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['E'].T for p in
                                      self.PlaceFieldData if t in p][0]

        self.numcells = np.size(self.Fc3data_dict['Task1'], 0)
        print(f'Total number of cells: %d' % self.numcells)

    def load_v73_Data(self):
        self.Fc3data_dict = self.create_data_dict(self.TaskDict)
        f = h5py.File(os.path.join(self.FolderName, self.ImgFileName[0]), 'r')
        for k, v in f.items():
            print(k, np.shape(v))

        count = 0
        for i in self.TaskDict.keys():
            self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

        if count != np.size(f['Fc3'], 1):
            raise Exception('Error!! File size doesnt match')
        else:
            print('All good')

    def get_and_sort_placeactivity(self):
        pc_activity_dict = {keys: [] for keys in self.TaskDict.keys()}
        pcsortednum = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            pc_activity = np.asarray([])
            for n in np.arange(np.size(self.sig_PFs_cellnum_revised[taskname])):
                if self.sig_PFs_cellnum_revised[taskname][n] in self.cellid_beg_multiplepfs[taskname]:
                    beg = 1
                else:
                    beg = 0
                placecell_num = self.numPFS_incells[taskname][n]
                pc_temp = np.nanmean(x['sig_PFs'][placecell_num][self.sig_PFs_cellnum_revised[taskname][n]], 1)
                pc_activity = np.vstack((pc_activity, pc_temp)) if pc_activity.size else pc_temp
            pcsortednum[taskname] = np.argsort(np.nanargmax(pc_activity, 1))
            pc_activity_dict[taskname] = pc_activity
        return pc_activity_dict, pcsortednum

    def calculate_remapping_with_task(self, taskA):
        pc_activity_dict = {keys: [] for keys in self.TaskDict.keys()}
        pcsortednum = {keys: [] for keys in self.TaskDict.keys()}
        cells_to_plot = list(self.sig_PFs_cellnum[taskA])
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            pc_activity = []
            for n, c in enumerate(cells_to_plot):
                if taskname == self.noreward_task:
                    pc_activity.append(np.nanmean((x['Allbinned_F'][0, c][:, :self.lickstoplap]), 1))
                else:
                    pc_activity.append(np.nanmean((x['Allbinned_F'][0, c]), 1))
            pc_activity_dict[taskname] = np.asarray(pc_activity)

        # Sort by taskA
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[taskA], 1))
        for taskname in self.TaskDict.keys():
            pcsortednum[taskname] = pcsorted

        return pc_activity_dict, pcsortednum

    def correlate_acivity_of_allcellsbytask(self, TaskA='Task1'):
        data_formapping = [i for i in self.PlaceFieldData if TaskA in i][0]
        data_formapping = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', data_formapping))['Allbinned_F']

        correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            laps = np.size(x['Allbinned_F'][0, 0], 1)
            corr = np.zeros((self.numcells, laps))

            for c in range(0, self.numcells):
                data1 = np.nanmean(data_formapping[0, c], 1)
                data2 = np.nan_to_num(x['Allbinned_F'][0, c])
                for l in range(0, laps):

                    temp = np.corrcoef(data2[:, l], data1)[0, 1]
                    if ~np.isnan(temp):
                        corr[c, l] = temp
            correlation_per_task[taskname] = corr
        return correlation_per_task

    def create_populationvector(self):
        pc_activity_dict = {keys: [] for keys in self.TaskDict.keys()}
        pcsortednum = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            if taskname == 'Task2b':
                continue
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            numlaps = self.Parsed_Behavior['numlaps'].item()[taskname]
            #numcells = np.size(self.sig_PFs_cellnum_revised[taskname])
            numcells = self.numcells
            pc_activity = np.zeros((numlaps, numcells, 40))
            #for n in np.arange(np.size(self.sig_PFs_cellnum_revised[taskname])):
            for n in np.arange(numcells):
                pc_temp = x['Allbinned_F'][0, n]
                for l in range(numlaps):
                    pc_activity[l, n, :] = pc_temp[:, l]
            pc_activity_dict[taskname] = pc_activity
            print(taskname, np.shape(pc_activity))
        self.save_pcs(pc_activity_dict, 'PopulationVectorsAllCells')
        return pc_activity_dict

    def create_populationvector_placecell(self):
        pc_activity_dict = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            if taskname == 'Task2b':
                continue
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            numlaps = self.Parsed_Behavior['numlaps'].item()[taskname]
            numcells = np.size(self.sig_PFs_cellnum_revised[taskname])
            pc_activity = np.zeros((numlaps, numcells, 40))
            for n in np.arange(np.size(self.sig_PFs_cellnum_revised[taskname])):
                pc_temp = x['Allbinned_F'][0, n]
                for l in range(numlaps):
                    pc_activity[l, n, :] = pc_temp[:, l]
            pc_activity_dict[taskname] = pc_activity
            print(taskname, np.shape(pc_activity))
        self.save_pcs(pc_activity_dict, 'PopulationVectors_Placecells')
        return pc_activity_dict

    def calculate_pfparameters(self):
        # Go through place cells for each task and get center of mass for each lap traversal
        # Algorithm from Marks paper
        self.pfparams_df = pd.DataFrame(
            columns=['Task', 'CellNumber', 'PlaceCellNumber', 'NumPlacecells', 'COM', 'WeightedCOM', 'Precision',
                     'Precision_rising', 'Width', 'FiringRatio', 'Firingintensity', 'Reliability'])
        for t in self.PlaceFieldData:
            ft = t.find('Task')
            taskname = t[ft:ft + t[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', t))
            for n in np.arange(np.size(self.sig_PFs_cellnum_revised[taskname])):
                if self.sig_PFs_cellnum_revised[taskname][n] in self.cellid_beg_multiplepfs[taskname]:
                    beg = 1
                else:
                    beg = 0
                placecell_num = self.numPFS_incells[taskname][n]
                data = x['sig_PFs'][placecell_num][self.sig_PFs_cellnum_revised[taskname][n]]
                COM = np.zeros(np.size(data, 1))
                weighted_com_num = np.zeros(np.size(data, 1))
                weighted_com_denom = np.zeros(np.size(data, 1))
                xbin = np.linspace(0, 40, 40, endpoint=False)
                for i in np.arange(np.size(data, 1)):
                    f_perlap = data[:, i]
                    f_perlap = np.nan_to_num(f_perlap)
                    # Skip laps without fluorescence
                    if not np.any(f_perlap):
                        continue
                    num_com = np.sum(np.multiply(f_perlap, xbin))
                    denom_com = np.sum(f_perlap)
                    COM[i] = num_com / denom_com
                    weighted_com_num[i] = np.max(f_perlap) * COM[i]
                    weighted_com_denom[i] = np.max(f_perlap)

                weighted_com = np.sum(weighted_com_num) / np.sum(weighted_com_denom)
                precision_num = np.zeros(np.size(data, 1))
                precision_num_rising = np.zeros(np.size(data, 1))
                precision_denom = np.zeros(np.size(data, 1))
                precision_denom_rising = np.zeros(np.size(data, 1))
                for i in np.arange(np.size(data, 1)):
                    f_perlap = data[:, i]
                    f_perlap = np.nan_to_num(f_perlap)
                    f_per_lap_rising = np.zeros_like(f_perlap)
                    rise = int(np.round(COM[i]))
                    f_per_lap_rising[:rise] = f_perlap[:rise]
                    # For precision, only try to use half transients
                    # Calculate two precisions
                    # Skip laps without fluorescence
                    if not np.any(f_perlap):
                        continue
                    precision_num_rising[i] = np.max(f_per_lap_rising) * np.square(rise - weighted_com)
                    precision_denom_rising[i] = np.max(f_per_lap_rising)
                    precision_num[i] = np.max(f_perlap) * np.square(COM[i] - weighted_com)
                    precision_denom[i] = np.max(f_perlap)

                precision = 1 / (np.sqrt((np.sum(precision_num) / np.sum(precision_denom))))
                precision_rising = 1 / (np.sqrt((np.sum(precision_num_rising) / np.sum(precision_denom_rising))))

                if precision > 5:
                    precision = np.nan
                if precision_rising > 5:
                    precision_rising = np.nan

                # Calculate stability
                stability = np.zeros((np.size(data, 1), np.size(data, 1)))
                stability[:] = np.nan
                for i in np.arange(np.size(data, 1)):
                    for j in np.arange(np.size(data, 1)):
                        if i != j:
                            corrcoef = np.corrcoef(data[:, i], data[:, j])[0, 1]
                            if not np.isnan(corrcoef):
                                stability[i, j] = corrcoef

                # Calculate number of laps where there is firing
                data_bw = data > 0
                numlaps_withfiring = np.size(np.where(np.max(data_bw, 0))) / np.size(data, 1)
                # print(f'Cell %d COM %0.2f, precision all %0.2f presicion rising face %0.2f' % (
                #     self.sig_PFs_cellnum[taskname][n], np.mean(weighted_com), precision, precision_rising))
                firingratio, infield_f = self.calculate_inoutfield_firing(x, n, placecell_num, taskname)
                self.pfparams_df = self.pfparams_df.append({'Task': taskname,
                                                            'CellNumber':
                                                                self.sig_PFs_cellnum_revised[taskname][n],
                                                            'PlaceCellNumber': placecell_num + 1,
                                                            'NumPlacecells': self.numPFs_incells_revised[taskname][
                                                                n],
                                                            'COM': COM,
                                                            'WeightedCOM': weighted_com,
                                                            'Precision': precision,
                                                            'Precision_rising': precision_rising,
                                                            'Reliability': np.nanmean(stability) * numlaps_withfiring,
                                                            'Width': x['PF_width'][placecell_num][
                                                                self.sig_PFs_cellnum_revised[taskname][n]],
                                                            'FiringRatio': firingratio,
                                                            'Firingintensity': infield_f},
                                                           ignore_index=True)

    def calculate_inoutfield_firing(self, data, pcnum, pfnum, taskname):

        pc_activity = np.nanmean(data['sig_PFs_with_noise'][pfnum][self.sig_PFs_cellnum_revised[taskname][pcnum]], 1)
        pc_start = np.int(
            data['PF_start_bins'][pfnum][self.sig_PFs_cellnum_revised[taskname][pcnum]]) - 1  # Pythonic(-1)
        pc_end = np.int(data['PF_end_bins'][pfnum][self.sig_PFs_cellnum_revised[taskname][pcnum]])
        # print(pcnum, pfnum, pc_start, pc_end, taskname, self.sig_PFs_cellnum[taskname][pcnum])
        infieldbins = np.zeros(pc_activity.shape, dtype=np.bool_)
        infieldbins[pc_start:pc_end] = True
        infield_activity = np.nanmean(pc_activity[infieldbins])
        infield_maxdff = np.nanmax(pc_activity[infieldbins])
        outfield_activity = np.nanmean(pc_activity[~infieldbins])
        ratio = outfield_activity / infield_activity
        if ratio > 1:
            ratio = 0
        return ratio, infield_maxdff

    def save_pcs(self, pcdict, savename):
        np.save(os.path.join(self.SaveFolder, '%s_%s' % (self.animalname, savename)), pcdict)

    def save_analyseddata(self):
        self.pfparams_df['animalname'] = self.animalname
        self.pfparams_df.to_csv(
            os.path.join(self.SaveFolder, f'%s_placecellparams_df.csv' % self.animalname))

        np.savez(os.path.join(self.SaveFolder, f'%s_placecell_data.npz' % self.animalname),
                 sig_PFs_cellnum=self.sig_PFs_cellnum, numPFs_incells=self.numPFS_incells,
                 sig_PFs_cellnum_revised=self.sig_PFs_cellnum_revised,
                 numPFs_incells_revised=self.numPFs_incells_revised,
                 sig_PFs_beginning=self.sig_PFs_beginning,
                 cellid_beg_multiplepfs=self.cellid_beg_multiplepfs,
                 numcells=self.numcells,
                 animalname=self.animalname, Fc3data=self.Fc3data_dict)


class PlottingFunctions(GetData):
    def plot_placecells_with_track_pertask(self, pc_activity, sorted_pcs, figsize=(10, 4)):
        taskaxis = {'Task1': 0, 'Task2': 1, 'Task3': 2, 'Task4': 3, 'Task5': 4}
        fs, ax1 = plt.subplots(1, len(self.TaskDict), figsize=figsize, dpi=100, sharex='all', sharey='all')
        for taskname in self.TaskDict.keys():
            task_data = pc_activity[taskname][sorted_pcs[taskname], :]
            normalise_data = task_data / np.nanmax(task_data, 1)[:, np.newaxis]
            ax1[taskaxis[taskname]].imshow(np.nan_to_num(normalise_data),
                                           aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=1)

            ax1[taskaxis[taskname]].set_title(self.TaskDict[taskname])
            ax1[taskaxis[taskname]].set_xticks([0, 20, 39])
            ax1[taskaxis[taskname]].set_xticklabels([0, 100, 200])
            ax1[taskaxis[taskname]].set_xlim((0, 39))
            pf.set_axes_style(ax1[taskaxis[taskname]], numticks=4)
        ax1[0].set_xlabel('Track Length (cm)')
        ax1[0].set_ylabel('Cell')

        ax1[0].locator_params(nbins=4)
        fs.tight_layout()
        plt.show()

    def plot_correlation_by_task(self, data_to_plot, placecell_flag=0, taskA='Task1', figsize=(10, 6)):
        numlaps = self.Parsed_Behavior['numlaps'].item()
        numlicks_withinreward = self.Parsed_Behavior['numlicks_withinreward'].item()
        fs, axes = plt.subplots(2, len(self.TaskDict.keys()), sharex='col', sharey='row',
                                gridspec_kw={'height_ratios': [2, 1]},
                                figsize=figsize)

        count_axis = 0
        if placecell_flag:
            cells_to_plot = self.sig_PFs_cellnum[taskA]
            celldata = data_to_plot[taskA][cells_to_plot, :]
            sort_cells = np.argsort(np.nanmean(celldata, 1))[::-1]
        else:
            celldata = data_to_plot[taskA]
            sort_cells = np.argsort(np.nanmean(celldata, 1))[::-1]

        for n, i in enumerate(self.TaskDict.keys()):
            if placecell_flag:
                celldata = data_to_plot[i][cells_to_plot, :]
            else:
                celldata = data_to_plot[i]
            celldata = celldata[sort_cells, :]

            ax1 = axes[0, count_axis]
            ax1.imshow(celldata, interpolation='nearest', aspect='auto', cmap='viridis', vmin=0,
                       vmax=1)
            ax1.set_xlim((0, numlaps[i]))
            ax1.set_title(self.TaskDict[i])

            ax2 = axes[1, count_axis]
            ax2.plot(np.nanmean(celldata, 0), '-o', linewidth=2, color='b')
            # print(np.shape(celldata))
            print(f'%s : %0.3f' % (i, np.nanmean(celldata[:, 1:])))
            ax2.set_xlabel('Lap number')
            ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
            ax3.plot(numlicks_withinreward[i], '-o', color='r', alpha=0.5,
                     label='Lap time')
            ax3.set_ylim((0, 7))

            if n == len(self.TaskDict.keys()) - 1:
                ax3.set_ylabel('Pre Licks', color='r')
            else:
                ax3.set_yticklabels([])

            for l in range(0, numlaps[i] - 1):
                if numlicks_withinreward[i][l]:
                    ax2.axvline(l, linewidth=0.25, color='k')
            count_axis += 1
            for a in [ax1, ax2, ax3]:
                pf.set_axes_style(a)
        axes[0, 0].set_ylabel('Cell Number')
        axes[1, 0].set_ylabel('Mean Correlation', color='b')
        fs.subplots_adjust(wspace=0.1, hspace=0.1)
