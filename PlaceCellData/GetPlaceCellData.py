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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import sklearn.cluster

# For plotting styles
if sys.platform == 'darwin':
    MainFolder = '/Users/seetha/Box Sync/MultiDayData/'
else:
    MainFolder = '/home/sheffieldlab/Desktop/NoReward/'

PlottingFormat_Folder = os.path.join(MainFolder, 'Scripts/PlottingTools/')
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class GetData:
    def __init__(self, animalinfo, FolderName, v73_flag=0, controlflag=0, noreward_task='Task2', rewardflag=0):
        self.FolderName = FolderName
        self.FigureFolder = os.path.join(self.FolderName, 'Figures')
        self.SaveFolder = os.path.join(self.FolderName, 'PlaceCells')
        self.RewardFolder = os.path.join(self.FolderName, 'RewardCells')
        self.controlflag = controlflag
        self.noreward_task = noreward_task
        self.rewardflag = rewardflag
        self.nsecondsroundrew = 2
        self.framerate = 30.98
        if self.controlflag:
            self.SaveDataframeFolder = os.path.join(
                '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/PlaceCellResults_All/')
        else:
            self.SaveDataframeFolder = os.path.join(
                '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/PlaceCellResults_All/')
            self.RewardDataframeFolder = os.path.join(
                '/home/sheffieldlab/Desktopß/NoReward/ImagingData/Good_behavior/Dataused/RewardCellResults_All/')

        self.TaskDict = animalinfo['task_dict']
        self.Task_Numframes = animalinfo['task_numframes']
        self.tracklength = animalinfo['tracklength']
        self.trackbins = animalinfo['trackbins']
        self.animalname = animalinfo['animal']

        if not os.path.exists(self.FigureFolder):
            os.mkdir(self.FigureFolder)
        if not os.path.exists(self.SaveFolder):
            os.mkdir(self.SaveFolder)
        if not os.path.exists(self.RewardFolder):
            os.mkdir(self.RewardFolder)

        if self.controlflag:
            self.new_taskDict = copy(self.TaskDict)
        else:
            self.new_taskDict = copy(self.TaskDict)
            self.new_taskDict[f'%sb' % self.noreward_task] = '2 No Rew No Lick'
        self.new_taskDict = OrderedDict(sorted(self.new_taskDict.items()))

        self.get_data_folders()
        if v73_flag:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()

        self.get_lapframes_numcells()

        if self.controlflag == 0:
            self.lickstoplap = self.Parsed_Behavior['lick_stop'].item()[self.noreward_task]
            self.lickstopframe = np.where(self.good_lapframes[self.noreward_task] == self.lickstoplap + 1)[0][0]
            print(self.lickstoplap, self.lickstopframe)
        # Find significant place cells
        # Add no lick data where exists
        self.find_sig_PFs_cellnum_bytask()
        self.beginning_cells = self.revise_sig_PFs()

        if not self.controlflag and self.rewardflag:
            self.collect_data_around_rewardzone()
            self.reward_data_correlation(self.reward_imaging_data)
            self.calculate_reward_cell_parameters()
        else:
            self.create_populationvector()
            self.calculate_pfparameters()
            self.correlate_acivity_of_allcellsbytask()
            self.common_droppedcells_withTask1()

        self.save_analyseddata()

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
        self.sig_PFs_cellnum = self.create_data_dict(self.new_taskDict)
        self.numPFS_incells = self.create_data_dict(self.new_taskDict)
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            tempx = np.squeeze(np.asarray(np.nan_to_num(x['number_of_PFs'])).T).astype(int)
            print(f'%s : Place cells: %d PlaceFields: %d' % (
                taskname, np.size(np.where(tempx > 0)[0]), np.sum(tempx[tempx > 0])))

            self.sig_PFs_cellnum[taskname] = np.where(tempx > 0)[0]
            self.numPFS_incells[taskname] = tempx[np.where(tempx > 0)[0]]

    def revise_sig_PFs(self):
        # Get_gaussianfit
        x = np.arange(0, np.int(self.tracklength / self.trackbins))
        y = self.gaussian(x, 1, 0.01, 5) + np.random.normal(0, 0.2, x.size)
        best_vals, covar = curve_fit(self.gaussian, x, y, p0=[1, 0, 1])
        gaussfit = self.gaussian(x, *best_vals)
        beginning_cell_dict = self.create_data_dict(self.new_taskDict)
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            beginning_cell = np.zeros_like(self.sig_PFs_cellnum[taskname])
            count = 0
            for n in np.arange(np.size(self.sig_PFs_cellnum[taskname])):
                for i in np.arange(self.numPFS_incells[taskname][n]):
                    pc_activity = np.nanmean(
                        x['sig_PFs'][i][self.sig_PFs_cellnum[taskname][n]], 1)
                    start_bins = x['PF_start_bins'][i][self.sig_PFs_cellnum[taskname][n]]
                    end_bins = x['PF_end_bins'][i][self.sig_PFs_cellnum[taskname][n]]
                    tailflag = self.check_for_beginning_transients(pc_activity, start_bins, end_bins, gaussfit)
                    if tailflag:
                        beginning_cell[n] += 1
            beginning_cell_dict[taskname] = beginning_cell
            plt.title(taskname)
            plt.show()
        self.update_PFs(beginning_cell_dict)
        return beginning_cell_dict

    def update_PFs(self, beginning_cells):
        self.sig_PFs_cellnum_revised = self.create_data_dict(self.new_taskDict)
        self.sig_PFs_beginning = self.create_data_dict(self.new_taskDict)
        self.numPFs_incells_revised = self.create_data_dict(self.new_taskDict)
        self.cellid_beg_multiplepfs = self.create_data_dict(self.new_taskDict)
        for taskname in self.new_taskDict.keys():
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

    def common_droppedcells_withTask1(self):
        self.droppedcells = self.create_data_dict(self.new_taskDict)
        self.commoncells = self.create_data_dict(self.new_taskDict)
        for i in self.new_taskDict.keys():
            if self.controlflag:
                if i not in 'Task1a':
                    self.droppedcells[i] = np.asarray(
                        [l for l in self.sig_PFs_cellnum_revised['Task1a'] if l not in self.sig_PFs_cellnum_revised[i]])
                    self.commoncells[i] = list(
                        set(self.sig_PFs_cellnum_revised['Task1a']).intersection(self.sig_PFs_cellnum_revised[i]))
            else:
                if i not in 'Task1':
                    self.droppedcells[i] = np.asarray(
                        [l for l in self.sig_PFs_cellnum_revised['Task1'] if l not in self.sig_PFs_cellnum_revised[i]])
                    self.commoncells[i] = list(
                        set(self.sig_PFs_cellnum_revised['Task1']).intersection(self.sig_PFs_cellnum_revised[i]))

    def load_fluorescentdata(self):
        self.Fc3data_dict = self.create_data_dict(self.TaskDict)
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        count = 0
        for i in self.TaskDict.keys():
            self.Fc3data_dict[i] = data['data'].item()[2].T[:,
                                   count:count + self.Task_Numframes[i]]
            print(f'%s : Number of Frames: %d' % (i, np.size(self.Fc3data_dict[i], 1)))
            count += self.Task_Numframes[i]

    def get_lapframes_numcells(self):
        self.good_lapframes = self.create_data_dict(self.new_taskDict)
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['E'].T for p in
                                      self.PlaceFieldData if t in p and 'Task2a' not in p][0]

        if self.controlflag:
            self.numcells = np.size(self.Fc3data_dict['Task1a'], 0)
        else:
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
                for i in np.arange(beg, self.numPFs_incells_revised[taskname][n]):
                    # pc_temp = np.nanmean(x['sig_PFs_with_noise'][i][self.sig_PFs_cellnum_revised[taskname][n]], 1)
                    pc_temp = np.nanmean((x['Allbinned_F'][0, self.sig_PFs_cellnum_revised[taskname][n]]), 1)
                    pc_activity = np.vstack((pc_activity, pc_temp)) if pc_activity.size else pc_temp
            pcsortednum[taskname] = np.argsort(np.nanargmax(pc_activity, 1))
            pc_activity_dict[taskname] = pc_activity
        return pc_activity_dict, pcsortednum

    def calculate_pfparameters(self):
        # Go through place cells for each task and get center of mass for each lap traversal
        # Algorithm from Marks paper
        self.pfparams_df = pd.DataFrame(
            columns=['Task', 'CellNumber', 'PlaceCellNumber', 'NumPlacecells', 'COM', 'WeightedCOM', 'Precision',
                     'Precision_rising', 'Width', 'FiringRatio', 'Firingintensity', 'Stability'])
        for t in self.PlaceFieldData:
            ft = t.find('Task')
            taskname = t[ft:ft + t[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', t))
            for n in np.arange(np.size(self.sig_PFs_cellnum_revised[taskname])):
                if self.sig_PFs_cellnum_revised[taskname][n] in self.cellid_beg_multiplepfs[taskname]:
                    beg = 1
                else:
                    beg = 0
                for p1 in np.arange(beg, self.numPFs_incells_revised[taskname][n]):
                    data = x['sig_PFs'][p1][self.sig_PFs_cellnum_revised[taskname][n]]
                    COM = np.zeros(np.size(data, 1))
                    weighted_com_num = np.zeros(np.size(data, 1))
                    weighted_com_denom = np.zeros(np.size(data, 1))
                    xbin = np.linspace(0, 40, 40, endpoint=False)
                    # fs, ax = plt.subplots(1)
                    # ax.imshow(data.T, aspect='auto', cmap='jet', vmin=0, vmax=0.5)
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
                        # plt.plot(COM[i], i, '*', markersize=10)
                        # plt.plot(COM[i], 1, '*', markersize=10)

                    weighted_com = np.sum(weighted_com_num) / np.sum(weighted_com_denom)
                    precision_num = np.zeros(np.size(data, 1))
                    precision_num_rising = np.zeros(np.size(data, 1))
                    precision_denom = np.zeros(np.size(data, 1))
                    precision_denom_rising = np.zeros(np.size(data, 1))
                    # ax1 = ax.twinx()
                    for i in np.arange(np.size(data, 1)):
                        f_perlap = data[:, i]
                        f_perlap = np.nan_to_num(f_perlap)
                        f_per_lap_rising = np.zeros_like(f_perlap)
                        rise = int(np.round(COM[i]))
                        f_per_lap_rising[:rise] = f_perlap[:rise]
                        # For precision, only try to use half transients
                        # ax1.plot(f_perlap, 'r', alpha=0.5)
                        # ax1.plot(f_per_lap_rising, 'k', alpha=0.5, linewidth=2)

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
                    # ax.set_title(f'Cell %d COM %0.2f, precision all %0.2f presicion rising face %0.2f' % (
                    #     self.sig_PFs_cellnum[taskname][n], np.mean(weighted_com), precision, precision_rising))
                    # plt.show()
                    firingratio, infield_f = self.calculate_inoutfield_firing(x, n, p1, taskname)
                    self.pfparams_df = self.pfparams_df.append({'Task': taskname,
                                                                'CellNumber':
                                                                    self.sig_PFs_cellnum_revised[taskname][n],
                                                                'PlaceCellNumber': p1 + 1,
                                                                'NumPlacecells': self.numPFs_incells_revised[taskname][
                                                                    n],
                                                                'COM': COM,
                                                                'WeightedCOM': weighted_com,
                                                                'Precision': precision,
                                                                'Precision_rising': precision_rising,
                                                                'Stability': np.nanmean(stability) * numlaps_withfiring,
                                                                'Width': x['PF_width'][p1][
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

    def calculate_remapping_with_task(self, taskA):
        pc_activity_dict = {keys: [] for keys in self.new_taskDict.keys()}
        pcsortednum = {keys: [] for keys in self.new_taskDict.keys()}
        cells_to_plot = list(self.sig_PFs_cellnum_revised[taskA])
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            pc_activity = np.asarray([])
            for n, c in enumerate(cells_to_plot):
                if c in self.cellid_beg_multiplepfs[taskA]:
                    beg = 1
                else:
                    beg = 0
                for l in np.arange(beg, self.numPFs_incells_revised[taskA][n]):
                    # Separate lick and no lick in task2
                    if taskname == self.noreward_task:
                        pc_temp = np.nanmean((x['Allbinned_F'][0, c][:, :self.lickstoplap]), 1)
                    else:
                        pc_temp = np.nanmean((x['Allbinned_F'][0, c]), 1)
                    pc_activity = np.vstack((pc_activity, pc_temp)) if pc_activity.size else pc_temp
            pc_activity_dict[taskname] = pc_activity

        # Sort by taskA
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[taskA], 1))
        for taskname in self.new_taskDict.keys():
            pcsortednum[taskname] = pcsorted

        return pc_activity_dict, pcsortednum

    def correlate_acivity_of_allcellsbytask(self, TaskA='Task1'):
        data_formapping = [i for i in self.PlaceFieldData if TaskA in i][0]
        data_formapping = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', data_formapping))['Allbinned_F']

        self.correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
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

            self.correlation_per_task[taskname] = corr

    def collect_data_around_rewardzone(self):
        print('Collecting reward zone data')
        self.reward_imaging_data = {k: [] for k in self.TaskDict.keys()}
        self.reward_Fc3_pertask = {k: [] for k in self.TaskDict.keys()}
        for nt, t in enumerate(self.TaskDict.keys()):
            pffile = [p for p in self.PlaceFieldData if t in p and 'Task2b' not in p][0]
            print(pffile)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', pffile))
            lapframes = x['E'].T
            print(t, np.max(lapframes))
            reward_data = self.Parsed_Behavior['reward_data'].item()[t]
            numlaps = self.Parsed_Behavior['numlaps'].item()[t]
            good_running_index = self.Parsed_Behavior['good_running_index'].item()[t]
            time_around_rew = np.int(self.nsecondsroundrew * self.framerate)  # seconds before and after reward
            rewarddata_percell = np.zeros((self.numcells, numlaps, time_around_rew + time_around_rew * 2))
            print(np.shape(rewarddata_percell))
            reward_Fc3_data = np.zeros_like(self.Fc3data_dict[t])
            for n1 in np.arange(self.numcells):
                for n2, (this, next) in enumerate(zip(range(1, np.max(lapframes)), range(2, np.max(lapframes) + 1))):
                    # print(np.shape(lapframes), np.shape(good_running_index))
                    [thislap, nextlap] = [np.where(lapframes == this)[0], np.where(lapframes == next)[0]]
                    [thislap_start, thislap_end] = [good_running_index[thislap[0]],
                                                    good_running_index[thislap[-1]]]
                    nextlap_start = good_running_index[nextlap[0]]

                    if t not in 'Task2':
                        reward_frame = reward_data[thislap_start:nextlap_start]
                        reward_frame = np.where(np.diff(reward_frame, axis=0) > 4)[0][0]
                        reward_frame += thislap_start
                        rewarddata_percell[n1, n2, :] = self.Fc3data_dict[t][n1,
                                                        reward_frame - time_around_rew:reward_frame + time_around_rew * 2]

                        reward_Fc3_data[n1, reward_frame - time_around_rew:reward_frame + time_around_rew * 2] = \
                            self.Fc3data_dict[t][n1,
                            reward_frame - time_around_rew:reward_frame + time_around_rew * 2]
                    else:
                        rewarddata_percell[n1, n2, :] = self.Fc3data_dict[t][n1,
                                                        thislap_end - time_around_rew:thislap_end + time_around_rew * 2]
                        reward_Fc3_data[n1, thislap_end - time_around_rew:thislap_end + time_around_rew * 2] = \
                            self.Fc3data_dict[t][n1,
                            thislap_end - time_around_rew:thislap_end + time_around_rew * 2]
            self.reward_imaging_data[t] = rewarddata_percell
            self.reward_Fc3_pertask[t] = reward_Fc3_data

    def reward_data_correlation(self, reward_data, task_to_correlate='Task1'):
        self.reward_correlation_data = {k: [] for k in self.TaskDict.keys()}
        correlate_with = np.mean(reward_data[task_to_correlate], 1)
        print(np.shape(correlate_with))
        for nt, t in enumerate(self.TaskDict.keys()):
            numlaps = self.Parsed_Behavior['numlaps'].item()[t]
            per_cell_corr = np.zeros((self.numcells, numlaps))
            for c in np.arange(self.numcells):
                for l in np.arange(numlaps):
                    corrcoef = np.corrcoef(reward_data[t][c, l, :], correlate_with[c, :])[0, 1]
                    if not np.isnan(corrcoef):
                        per_cell_corr[c, l] = corrcoef
            self.reward_correlation_data[t] = per_cell_corr

    def plot_reward_data_per_cell(self, reward_data):
        pdf = PdfPages(os.path.join(self.RewardFolder, 'Rewardzonefiring.pdf'))
        for c in np.arange(self.numcells):
            print('Saving Cell %s \r' % c)
            fs, ax = plt.subplots(1, len(self.TaskDict), sharex='all', sharey='all', figsize=(10, 3))
            for nt, t in enumerate(self.TaskDict.keys()):
                numlaps = self.Parsed_Behavior['numlaps'].item()[t]
                ax[nt].imshow(reward_data[t][c], aspect='auto', cmap='jet', vmin=0, vmax=0.5)
                ax[nt].set_xticks(np.linspace(0, np.size(reward_data[t], 2) + 1, 4))
                ax[nt].set_xticklabels(np.rint(np.linspace(0, np.size(reward_data[t], 2) + 1, 4) / self.framerate) - 1)
                ax[nt].vlines(self.nsecondsroundrew * self.framerate, ymin=0, ymax=numlaps, color='k')
                data_cell = self.reward_df[(self.reward_df['CellNumber'] == c) & (self.reward_df['Task'] == t)]
                ax[nt].set_title(f'COM %f\nPrecision %f\nStability %f' % (
                    data_cell['WeightedCOM'], data_cell['Precision'], data_cell['Stability']), fontsize=8)
                pf.set_axes_style(ax[nt])
            pdf.savefig(fs, bbox_inches='tight')
            plt.close()
        pdf.close()

    def calculate_reward_cell_parameters(self):
        # Go through place cells for each task and get center of mass for each lap traversal
        # Algorithm from Marks paper
        self.reward_df = pd.DataFrame(
            columns=['Task', 'CellNumber', 'COM', 'WeightedCOM', 'Precision', 'Firingintensity', 'Stability'])
        for t in self.TaskDict.keys():
            for c in np.arange(self.numcells):
                data = self.reward_imaging_data[t][c].T
                # print(np.shape(data))
                COM = np.zeros(np.size(data, 1))
                weighted_com_num = np.zeros(np.size(data, 1))
                weighted_com_denom = np.zeros(np.size(data, 1))
                xbin = np.linspace(0, np.size(data, 0), np.size(data, 0), endpoint=False)
                # plt.figure()
                # plt.imshow(data.T, aspect='auto', cmap='jet', vmin=0, vmax=0.5)
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
                    # plt.plot(COM[i], i, '*', markersize=10)

                weighted_com = np.sum(weighted_com_num) / np.sum(weighted_com_denom)
                # Calculate precision
                precision_num = np.zeros(np.size(data, 1))
                precision_denom = np.zeros(np.size(data, 1))
                for i in np.arange(np.size(data, 1)):
                    f_perlap = data[:, i]
                    f_perlap = np.nan_to_num(f_perlap)
                    # Skip laps without fluorescence
                    if not np.any(f_perlap):
                        continue
                    precision_num[i] = np.max(f_perlap) * np.square(COM[i] - weighted_com)
                    precision_denom[i] = np.max(f_perlap)

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

                precision = 1 / (np.sqrt((np.sum(precision_num) / np.sum(precision_denom))))

                if precision > 5:
                    precision = np.nan

                # plt.title(f'Cell %d COM %0.2f, precision %0.2f, stability %0.2f' % (
                #     c, np.mean(weighted_com), precision, np.mean(ut)))
                # plt.show()
                self.reward_df = self.reward_df.append({'Task': t,
                                                        'CellNumber': c,
                                                        'COM': COM,
                                                        'WeightedCOM': weighted_com,
                                                        'Precision': precision,
                                                        'Stability': np.nanmean(stability) * numlaps_withfiring,
                                                        'Firingintensity': np.mean(data, 1)},
                                                       ignore_index=True)

    def save_analyseddata(self):

        if not self.controlflag and self.rewardflag:
            self.reward_df['animalname'] = self.animalname
            self.reward_df.to_csv(
                os.path.join(self.RewardDataframeFolder, f'%s_rewardcellparams_df.csv' % self.animalname))
        else:
            # Save params dataframe after appending animal name as column
            self.pfparams_df['animalname'] = self.animalname
            self.pfparams_df.to_csv(
                os.path.join(self.SaveDataframeFolder, f'%s_placecellparams_df.csv' % self.animalname))

        # Save pfnum and correlation datasets
        if self.controlflag:
            np.savez(os.path.join(self.SaveFolder, f'%s_placecell_data.npz' % self.animalname),
                     sig_PFs_cellnum=self.sig_PFs_cellnum, numPFs_incells=self.numPFS_incells,
                     sig_PFs_cellnum_revised=self.sig_PFs_cellnum_revised,
                     numPFs_incells_revised=self.numPFs_incells_revised,
                     sig_PFs_beginning=self.sig_PFs_beginning,
                     cellid_beg_multiplepfs=self.cellid_beg_multiplepfs,
                     dropped_cells=self.droppedcells, common_cells=self.commoncells,
                     correlation_withTask1=self.correlation_per_task, numcells=self.numcells,
                     animalname=self.animalname, Fc3data=self.Fc3data_dict, framerate=self.framerate)
        elif not self.controlflag and self.rewardflag:
            np.savez(os.path.join(self.RewardFolder, f'%s_reward_data.npz' % self.animalname),
                     sig_PFs_cellnum=self.sig_PFs_cellnum, numPFs_incells=self.numPFS_incells,
                     sig_PFs_cellnum_revised=self.sig_PFs_cellnum_revised,
                     numPFs_incells_revised=self.numPFs_incells_revised,
                     sig_PFs_beginning=self.sig_PFs_beginning,
                     cellid_beg_multiplepfs=self.cellid_beg_multiplepfs, numcells=self.numcells,
                     animalname=self.animalname, Fc3data=self.Fc3data_dict, rewarddata_percell=self.reward_imaging_data,
                     rewardzone_Fc3=self.reward_Fc3_pertask, rewardcorrelation=self.reward_correlation_data,
                     nsecondsroundrew=self.nsecondsroundrew, framerate=self.framerate)
        else:
            np.savez(os.path.join(self.SaveFolder, f'%s_placecell_data.npz' % self.animalname),
                     sig_PFs_cellnum=self.sig_PFs_cellnum, numPFs_incells=self.numPFS_incells,
                     sig_PFs_cellnum_revised=self.sig_PFs_cellnum_revised,
                     numPFs_incells_revised=self.numPFs_incells_revised,
                     sig_PFs_beginning=self.sig_PFs_beginning,
                     cellid_beg_multiplepfs=self.cellid_beg_multiplepfs,
                     dropped_cells=self.droppedcells, common_cells=self.commoncells,
                     correlation_withTask1=self.correlation_per_task, numcells=self.numcells,
                     animalname=self.animalname, Fc3data=self.Fc3data_dict, framerate=self.framerate)

    def save_pcs(self, pcdict, savename):
        np.save(os.path.join(self.FolderName, 'PlaceCells', '%s_%s' % (self.animalname, savename)), pcdict)


class PlottingFunctions(GetData):
    def plot_population_vector_withtask(self):
        # Calculate population vector correlation
        population_vec = self.create_populationvector()
        fs, ax1 = plt.subplots(1, len(self.TaskDict), figsize=(12, 4), dpi=100)
        for n, t in enumerate(self.TaskDict.keys()):
            data = population_vec[t]
            c = np.zeros((np.size(data, 0), np.size(data, 0)))
            for l1 in range(np.size(data, 0)):
                for l2 in range(np.size(data, 0)):
                    d1 = np.nanmean(data[l1], 1)
                    d2 = np.nanmean(data[l2], 1)
                    c[l1, l2] = np.corrcoef(d1, d2)[0, 1]
            ax1[n].imshow(c, aspect='auto', cmap='jet', interpolation='nearest', vmin=-0.06, vmax=1)
            ax1[n].set_title(self.TaskDict[t])

        ax1[0].set_xlabel('Lap#')
        ax1[0].set_ylabel('Lap#')

        ax1[0].locator_params(nbins=4)
        fs.tight_layout()
        plt.show()

    def plot_placecells_with_track_pertask(self, pc_activity, sorted_pcs, figsize=(10, 4), **kwargs):
        taskaxis = {'Task1': 0, 'Task1a': 0, 'Task1b': 1, 'Task2': 1, 'Task2b': 2, 'Task3': 3, 'Task4': 4}
        fs, ax1 = plt.subplots(1, len(self.new_taskDict), figsize=figsize, dpi=100, sharex='all', sharey='all')
        for taskname in self.new_taskDict.keys():
            task_data = pc_activity[taskname][sorted_pcs[taskname], :]
            if 'normalise_basetask' in kwargs.keys():
                normalise_data = task_data / np.nanmax(
                    pc_activity[kwargs['normalise_basetask']][sorted_pcs[taskname], :], 1)[:, np.newaxis]
            else:
                normalise_data = task_data / np.nanmax(task_data, 1)[:, np.newaxis]

            ax1[taskaxis[taskname]].imshow(np.nan_to_num(normalise_data),
                                           aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=1)

            ax1[taskaxis[taskname]].set_title(self.new_taskDict[taskname])
            ax1[taskaxis[taskname]].set_xticks([0, 20, 39])
            ax1[taskaxis[taskname]].set_xticklabels([0, 100, 200])
            ax1[taskaxis[taskname]].set_xlim((0, 39))
            pf.set_axes_style(ax1[taskaxis[taskname]], numticks=4)
        ax1[0].set_xlabel('Track Length (cm)')
        ax1[0].set_ylabel('Cell')

        ax1[0].locator_params(nbins=4)
        fs.tight_layout()
        plt.show()

    def plot_common_PF_heatmap(self):
        PF_sort = {keys: [] for keys in self.new_taskDict.keys()}
        PF_sort.update({'Cellnum': []})
        for i in range(0, self.numcells):
            PF_sort['Cellnum'].append(i)
            for t in self.new_taskDict.keys():
                if i in self.sig_PFs_cellnum_revised[t]:
                    PF_sort[t].append(1)
                else:
                    PF_sort[t].append(0)

        self.PF_sort_df = pd.DataFrame.from_dict(PF_sort)
        # Dont use Task2 for now
        if not self.controlflag:
            self.PF_sort_df = self.PF_sort_df.drop(['Task2'], axis=1)
            self.PF_sort_df = self.PF_sort_df.sort_values(by=['Task1', 'Task2b', 'Task3'], ascending=False)
        else:
            self.PF_sort_df = self.PF_sort_df.sort_values(by=['Task1a', 'Task1b'], ascending=False)
        self.PF_sort_df = self.PF_sort_df.reset_index(drop=True)

        sns.heatmap(self.PF_sort_df.drop(['Cellnum'], axis=1), cbar=False)

    def plot_percent_PFs_bytracklength(self, tasks_to_plot, bins=10):
        fs, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=100, sharey='all', sharex='all')
        # Plot COM of place cells along track
        if self.animalname == 'CFC4':
            normfactor = 678
        else:
            normfactor = self.numcells

        with sns.color_palette('deep'):
            for i in tasks_to_plot:
                com = self.pfparams_df[self.pfparams_df.Task == i]['WeightedCOM'] * self.trackbins
                hist_com, bins_com, center, width = CommonFunctions.make_histogram(com, bins, normfactor,
                                                                                   self.tracklength)
                ax[0].bar(center, hist_com, align='center', width=width, alpha=0.5, label=self.new_taskDict[i])
        ax[0].set_title('Fields in %s and %s' % (tasks_to_plot[0], tasks_to_plot[1]))
        ax[0].set_ylabel('Percentage of fields')

        # Plot COM of place cells that are no longer firing - compare with first task
        droppedcells = np.asarray(
            [l for l in self.sig_PFs_cellnum[tasks_to_plot[0]] if l not in self.sig_PFs_cellnum[tasks_to_plot[1]]])
        print('Number of cells without place fields in %s : %d' % (tasks_to_plot[1], np.size(droppedcells)))
        df_plot = self.pfparams_df[self.pfparams_df['Task'] == tasks_to_plot[0]]
        com = df_plot[df_plot['CellNumber'].isin(droppedcells)]['WeightedCOM'] * self.trackbins
        hist_com, bins_com, center, width = CommonFunctions.make_histogram(com, bins, normfactor, self.tracklength)
        ax[1].bar(center, hist_com, align='center', width=width, alpha=0.5, label=self.new_taskDict[i])
        ax[1].set_title('Fields in %s but not in %s' % (tasks_to_plot[0], tasks_to_plot[1]))

        # Plot COM of place cells precent in both tasks
        commoncells = list(
            set(self.sig_PFs_cellnum[tasks_to_plot[0]]).intersection(self.sig_PFs_cellnum[tasks_to_plot[1]]))
        print('Number of cells with place fields in both: %d' % np.size(commoncells))
        for i in tasks_to_plot:
            df_plot = self.pfparams_df[self.pfparams_df['Task'] == i]
            com = df_plot[df_plot['CellNumber'].isin(commoncells)]['WeightedCOM'] * self.trackbins
            hist_com, bins_com, center, width = CommonFunctions.make_histogram(com, bins, normfactor,
                                                                               self.tracklength)
            ax[2].bar(center, hist_com, align='center', width=width, alpha=0.5)
            ax[2].set_title('Fields present in both %s and %s' % (tasks_to_plot[0], tasks_to_plot[1]))

        for a in ax:
            a.set_xlabel('Weighted center of mass (cm)')
            a.set_aspect(10)
            pf.set_axes_style(a)
        ax[0].legend(bbox_to_anchor=(0, -0.6), loc=2, borderaxespad=0., ncol=2)

    def plot_pfparams(self, tasks_to_plot, plottype='bar'):
        columns_to_plot = ['Precision', 'Precision_rising', 'Width', 'FiringRatio', 'Firingintensity']
        color = sns.color_palette('dark', len(tasks_to_plot))
        df_plot = self.pfparams_df[self.pfparams_df['Task'].isin(tasks_to_plot)]
        with sns.color_palette('dark'):
            fs, ax = plt.subplots(1, len(columns_to_plot) + 1, figsize=(15, 4))
            if plottype == 'hist':
                axins = []
                for a in ax[1:]:
                    axins.append(inset_axes(a, width=1.5, height=1.0))
            if plottype == 'bar':
                sns.countplot(x='Task', data=df_plot, order=tasks_to_plot, ax=ax[0])
            for n, columns in enumerate(columns_to_plot):
                if plottype == 'bar':
                    sns.barplot(y=columns, x='Task', data=df_plot, order=tasks_to_plot, ax=ax[n + 1], errwidth=3)
                else:
                    for n1, t in enumerate(tasks_to_plot):
                        y = np.nan_to_num(df_plot[df_plot['Task'] == t][columns])
                        sns.distplot(y, kde=False, ax=ax[n + 1])
                        self.add_inset_with_cdf(axins[n], y, color[n1])

        if plottype == 'bar':
            ax[0].set_ylabel('Number of place fields')
        else:
            ax[0].axis('off')
        for a in ax:
            pf.set_axes_style(a, numticks=4)
        fs.tight_layout()

    def plot_pfparams_commoncells(self, taskA, taskB, plottype='bar'):
        columns_to_plot = ['Precision', 'Precision_rising', 'Width', 'FiringRatio', 'Firingintensity']
        # Get dataframe for common cells between two tasks
        t1 = self.pfparams_df[self.pfparams_df['Task'] == taskA]
        t2 = self.pfparams_df[self.pfparams_df['Task'] == taskB]
        combined = pd.merge(t1, t2, how='inner', on=['CellNumber'],
                            suffixes=(f'_%s' % taskA, f'_%s' % taskB))
        combined.to_csv(
            os.path.join(self.SaveDataframeFolder,
                         f'%s_commonpfparams_%s_and%s.csv' % (self.animalname, taskA, taskB)))
        color = sns.color_palette('dark', 2)
        fs, ax = plt.subplots(1, len(columns_to_plot), figsize=(15, 4))
        axins = []
        if plottype == 'hist':
            for a in ax:
                axins.append(inset_axes(a, width=2, height=1.5))
        for n1, t in enumerate([taskA, taskB]):
            for n2, columns in enumerate(columns_to_plot):
                y = np.nan_to_num(combined[f'%s_%s' % (columns, t)])
                if plottype == 'bar':
                    m, ci = CommonFunctions.mean_confidence_interval(y)
                    ax[n2].bar(n1, m, yerr=ci, error_kw={'lw': 3}, color=color[n1])
                    ax[n2].set_ylabel(columns)
                else:
                    sns.distplot(y, kde=False, ax=ax[n2], color=color[n1])
                    ax[n2].set_xlabel(columns)
                    self.add_inset_with_cdf(axins[n2], y, color[n1])

        for a in ax:
            if plottype == 'bar':
                a.set_xticks([0, 1])
                a.set_xticklabels([taskA, taskB])
            pf.set_axes_style(a, numticks=4)

        fs.tight_layout()
        if plottype == 'hist':
            self.scatter_of_common_center_of_mass(combined, taskA, taskB)

    def add_inset_with_cdf(self, axins, y, color):
        axins.hist(y, bins=1000, density=True, cumulative=True, label='CDF',
                   histtype='step', color=color, linewidth=2)
        axins.set_yticks([0, 1])
        pf.set_axes_style(axins)

    def plot_rewardcell_scatter(self, taskA, taskB):
        t1 = self.pfparams_df[self.pfparams_df['Task'] == taskA]
        t2 = self.pfparams_df[self.pfparams_df['Task'] == taskB]
        combined = pd.merge(t1, t2, how='inner', on=['CellNumber'],
                            suffixes=(f'_%s' % taskA, f'_%s' % taskB))
        combined.to_csv(
            os.path.join(self.SaveDataframeFolder,
                         f'%s_commonpfparams_%s_and%s.csv' % (self.animalname, taskA, taskB)))

    def scatter_of_common_center_of_mass(self, combined_dataset, taskA, taskB, bins=10):
        fs, ax = plt.subplots(1, 2, figsize=(8, 4))
        x = combined_dataset[f'%s_%s' % ('WeightedCOM', taskA)] * self.trackbins
        y = combined_dataset[f'%s_%s' % ('WeightedCOM', taskB)] * self.trackbins
        # Scatter plot
        ax[0].scatter(y, x, color='k')
        ax[0].plot([0, self.tracklength], [0, self.tracklength], linewidth=2, color=".3")
        ax[0].set_xlabel(taskB)
        ax[0].set_ylabel(taskA)
        ax[0].set_title('Center of Mass')
        # Heatmap of scatter plot
        heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        img = ax[1].imshow(heatmap.T, cmap='gray_r', extent=extent, interpolation='bilinear', origin='lower', vmin=0,
                           vmax=np.max(heatmap))
        ax[1].plot([0 + bins, self.tracklength - bins], [0 + bins, self.tracklength - bins], linewidth=2, color=".3")
        ax[1].set_xlim((0, self.tracklength))
        ax[1].set_ylim((0, self.tracklength))
        axins = CommonFunctions.add_colorbar_as_inset(axes=ax[1])
        cb = fs.colorbar(img, cax=axins, pad=0.2, ticks=[0, np.int(np.max(heatmap))])
        cb.set_label('Field Density', rotation=270, labelpad=12)

        for a in ax:
            pf.set_axes_style(a, numticks=5)

    def plot_correlation_by_task(self, data_to_plot, plot_top_cells=100, placecell_flag=0, taskA='Task1',
                                 figsize=(10, 6)):
        numlaps = self.Parsed_Behavior['numlaps'].item()
        # if self.animalname == 'CFC3' and self.controlflag:
        #     numlaps['Task1a'] -= 13
        numlicks_withinreward = self.Parsed_Behavior['numlicks_withinreward'].item()
        fs, axes = plt.subplots(2, len(self.TaskDict.keys()), sharex='col', sharey='row',
                                gridspec_kw={'height_ratios': [2, 1]},
                                figsize=figsize)

        count_axis = 0
        if placecell_flag:
            cells_to_plot = self.sig_PFs_cellnum_revised[taskA]
            celldata = data_to_plot[taskA][cells_to_plot, :]
            sort_cells = np.argsort(np.nanmean(celldata, 1))[::-1]
        else:
            celldata = data_to_plot[taskA]
            sort_cells = np.argsort(np.nanmean(celldata, 1))[::-1]
            sort_cells = sort_cells[0:np.int((plot_top_cells / 100) * np.size(sort_cells))]

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
            ax2.plot(np.mean(celldata, 0), '-o', linewidth=2, color='b')
            ax2.set_xlabel('Lap number')
            ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
            ax3.plot(numlicks_withinreward[i], '-o', color='r', alpha=0.5,
                     label='Lap time')

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

    def plot_pfparams_bytracklength(self, tasks_to_plot, nbins):
        columns_to_plot = ['Precision', 'Precision_rising', 'Width', 'FiringRatio', 'Firingintensity']
        color = sns.color_palette('deep', len(tasks_to_plot))
        fs, ax = plt.subplots(1, len(columns_to_plot), figsize=(10, 3), dpi=100)

        for n1, t in enumerate(tasks_to_plot):
            com = self.pfparams_df[self.pfparams_df['Task'] == t]['WeightedCOM'] * self.trackbins
            bins = np.linspace(0, self.tracklength, nbins + 1)
            ind = np.digitize(com, bins)
            for n2, c in enumerate(columns_to_plot):
                y = np.asarray(self.pfparams_df[self.pfparams_df['Task'] == t][c])
                mean_binned = np.zeros(nbins)
                error = np.zeros(nbins)
                for b in np.arange(1, nbins + 1):
                    m, ci = CommonFunctions.mean_confidence_interval(y[ind == b])
                    mean_binned[b - 1], error[b - 1] = m, ci
                width = np.diff(bins)
                center = (bins[:-1] + bins[1:]) / 2
                ax[n2].bar(center, mean_binned, width=width, alpha=0.5, color=color[n1])
                ax[n2].set_ylabel(c)

        for a in ax:
            a.set_xlabel('Track Length (cm)')
            pf.set_axes_style(a, numticks=4)

        fs.tight_layout()


class CommonFunctions:
    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, h

    @staticmethod
    def best_fit_slope_and_intercept(xs, ys):
        m = (((np.nanmean(xs) * np.nanmean(ys)) - np.nanmean(xs * ys)) /
             ((np.nanmean(xs) * np.nanmean(xs)) - np.nanmean(xs * xs)))

        b = np.nanmean(ys) - m * np.nanmean(xs)
        regression_line = [(m * x) + b for x in xs]

        return regression_line

    @staticmethod
    def make_histogram(com, bins, normalisefactor, tracklength):
        hist_com, bins_com = np.histogram(com, bins=np.arange(0, tracklength + 5, bins))
        hist_com = (hist_com / np.sum(normalisefactor)) * 100
        width = np.diff(bins_com)
        center = (bins_com[:-1] + bins_com[1:]) / 2
        return hist_com, bins_com, center, width

    @staticmethod
    def add_colorbar_as_inset(axes):
        axins = inset_axes(axes,
                           width="5%",  # width = 5% of parent_bbox width
                           height="50%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axes.transAxes,
                           borderpad=0.5,
                           )
        return axins
