import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats

# For plotting styles
PlottingFormat_Folder = '/Users/seetha/Box Sync/MultiDayData/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

DataDetailsFolder = '/Users/seetha/Box Sync/MultiDayData/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import AnimalDetailsWT

class GetData(object):
    def __init__(self, FolderName, base_task):
        self.FolderName = FolderName
        self.animals = [f for f in os.listdir(self.FolderName) if f not in ['.DS_Store']][1:]

        self.corr_animal = {k: [] for k in self.animals}
        for a in self.animals:
            print(a)
            self.corr_animal[a] = self.correlate_acivity_of_allcellsbytask(a, base_task)

    def get_place_cellinfo(self, animal):
        pcdata = np.load(
            os.path.join(self.FolderName, animal, 'PlaceCells', '%s_placecell_data.npz' % animal),
            allow_pickle=True)
        return pcdata

    def correlate_acivity_of_allcellsbytask(self, animal, TaskA='Task1'):
        animalinfo = AnimalDetailsWT.MultiDaysAnimals(animal)
        TaskDict = animalinfo['task_dict']
        numcells = self.get_place_cellinfo(animal)['sig_PFs_cellnum_revised'].item()

        PlaceFieldData = \
            [f for f in os.listdir(os.path.join(self.FolderName, animal, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

        data_formapping = [i for i in PlaceFieldData if TaskA in i][0]
        data_formapping = scipy.io.loadmat(os.path.join(self.FolderName, animal, 'Behavior', data_formapping))['Allbinned_F']

        correlation_per_task = {keys: [] for keys in TaskDict.keys()}
        for i in PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, animal, 'Behavior', i))
            laps = np.size(x['Allbinned_F'][0, 0], 1)
            corr = np.empty((len(numcells[TaskA]), laps))
            corr[:] = np.nan

            for n, c in enumerate(numcells[TaskA]):
                data1 = np.nanmean(data_formapping[0, c][:, :], 1)
                # plt.plot(data1)
                # plt.plot(np.nanmean(data_formapping[0, c][:, -5:], 1))
                # plt.show()
                data2 = np.nan_to_num(x['Allbinned_F'][0, c])
                for l in range(0, laps):
                    temp = np.corrcoef(data2[:, l], data1)[0, 1]
                    corr[n, l] = temp
            # return
            correlation_per_task[taskname] = corr
        return correlation_per_task

    def plot_first_vs_lastlap(self, tasklist):
        data = self.corr_animal.copy()
        corr_data_base_all, corr_data_task_all = [], []
        for a in self.animals:
            corr_data_base, corr_data_task = [], []
            for n, t in enumerate(tasklist):
                if n == 0:
                    corr_data_base.append(np.nanmean(data[a][t][:, -1], 0))
                else:
                    corr_data_task.append(np.nanmean(data[a][t][:, 0:10], 0))

            corr_data_base_all.append(corr_data_base)
            corr_data_task_all.append(corr_data_task)

        return np.squeeze(np.asarray(corr_data_base_all)), np.squeeze(np.asarray(corr_data_task_all))
