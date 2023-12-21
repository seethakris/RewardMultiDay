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

class GetData(object):
    def __init__(self, FolderName, LickFolder, taskdict, createresultflag=False, get_commoncells=False):
        self.FolderName = FolderName
        self.LickFolder = LickFolder
        self.TaskDict = taskdict
        self.animals = [f for f in os.listdir(self.FolderName) if f not in ['.DS_Store']][1:]
        self.spatialbins = 40
        self.get_commoncells = get_commoncells
        self.corr_animal = {k: [] for k in self.animals}

        for i in self.animals:
            print(i)
            self.corr_animal[i] = self.combine_pop_vec(i)

    def get_pop_vec(self, animalname):
        pop_vec = np.load(
            os.path.join(self.FolderName, animalname, 'PlaceCells', '%s_PopulationVectorsAllCells.npy' % animalname),
            allow_pickle=True)
        return pop_vec.item()

    def combine_pop_vec(self, animalname):
        population_vec = self.get_pop_vec(animalname)
        lickstop_df = pd.read_csv(os.path.join(self.LickFolder, 'Lickstops.csv'), index_col=0)
        lickstop = lickstop_df.loc[animalname, lickstop_df.columns[1]]
        lap_vline = []
        compileddata = []
        for n, t in enumerate(self.TaskDict):
            taskdata = population_vec[t][:, :, :]
            taskdata = (taskdata > 0).astype(np.int_)
            if t in ['Task3']:
                laps_required = lickstop
                print(laps_required, np.shape(taskdata))
                taskdata = taskdata[-5:, :, :]
            # print(np.shape(taskdata))
            if self.get_commoncells:
                celldf = self.get_commoncells_peranimal(animalname)
                commoncells = celldf[celldf[self.TaskDict.keys()].sum(axis=1)==len(self.TaskDict.keys())]['CellNum'].to_list()
                commoncells = np.unique(commoncells)
                taskdata = taskdata[:, commoncells, :]
                # print(np.shape(taskdata))
            taskdata = np.nanmean(taskdata, 0)
            compileddata.append(taskdata)
            
        print(np.shape(compileddata))
        correlation = self.correlate_pop_vec(compileddata)
        return correlation

    def get_commoncells_peranimal(self, a):
        numcells_dict = {keys: [] for keys in list(self.TaskDict.keys()) + ['CellNum']}
        pfparams = np.load(
            os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecell_data.npz' % a), allow_pickle=True)
        numcells = pfparams['numcells']
        sig_PFs = pfparams['sig_PFs_cellnum_revised'].item()
        numPFs = pfparams['numPFs_incells'].item()
        for c in np.arange(numcells):
            numcells_dict['CellNum'].append(c)
            for t in self.TaskDict.keys():
                if c in sig_PFs[t]:
                    numcells_dict[t].append(1)
                else:
                    numcells_dict[t].append(0)
        numcells_df = pd.DataFrame.from_dict(numcells_dict)
        return numcells_df

    def correlate_pop_vec(self, compileddata):
        # Correlate each task with others
        allcompiled = np.asarray([])
        for d1 in compileddata:
            compiled_corr = np.asarray([])
            for d2 in compileddata:
                c = np.zeros((self.spatialbins, self.spatialbins))
                for p1 in range(self.spatialbins):
                    for p2 in range(self.spatialbins):
                        c[p1, p2] = np.corrcoef(d1[:, p1], d2[:, p2])[0, 1]
                compiled_corr = np.hstack((compiled_corr, c)) if compiled_corr.size else c
            allcompiled = np.vstack((allcompiled, compiled_corr)) if allcompiled.size else compiled_corr
        return allcompiled

    def average_pv_corr(self):
        allmatrix = []
        for k, v in self.corr_animal.items():
            allmatrix.append(v)
        average_corr = np.nanmedian(np.asarray(allmatrix), 0)
        return average_corr

    def calculate_mean_trace_bettasks(self):
        tasklist = list(self.TaskDict.keys())
        mean_corr = {'%s_%s' % (k, j): [] for k in tasklist for j in tasklist}
        sem_corr = {'%s_%s' % (k, j): [] for k in tasklist for j in tasklist}
        for k, v in self.corr_animal.items():
            for n1, i in enumerate(range(0, np.size(v, 0), self.spatialbins)):
                for n2, j in enumerate(range(0, np.size(v, 0), self.spatialbins)):
                    data = v[i:i + self.spatialbins, j:j + self.spatialbins]
                    mean_corr['%s_%s' % (tasklist[n1], tasklist[n2])].append(
                        np.nanmean(np.diag(data)))
                    sem_corr['%s_%s' % (tasklist[n1], tasklist[n2])].append(
                        scipy.stats.sem(data, nan_policy='omit'))
        mean_corr = pd.DataFrame.from_dict(mean_corr)
        sem_corr = pd.DataFrame.from_dict(sem_corr)
        drop_columns = ['%s_%s' % (k, k) for k in tasklist]
        mean_corr = mean_corr.drop(columns=drop_columns)
        sem_corr = sem_corr.drop(columns=drop_columns)
        return mean_corr, sem_corr

    def collect_pop_vec_diag(self):
        tasklist = list(self.TaskDict.keys())
        diag_corr = {'%s_%s' % (k, j): [] for k in tasklist for j in tasklist}
        for k, v in self.corr_animal.items():
            for n1, i in enumerate(range(0, np.size(v, 0), self.spatialbins)):
                for n2, j in enumerate(range(0, np.size(v, 0), self.spatialbins)):
                    data = v[i:i + self.spatialbins, j:j + self.spatialbins]
                    diag_corr['%s_%s' % (tasklist[n1], tasklist[n2])].extend(
                        np.nan_to_num(np.diag(data)).tolist())
        print(np.shape(diag_corr['Task1_Task2']))
        diag_corr = pd.DataFrame.from_dict(diag_corr)
        drop_columns = ['%s_%s' % (k, k) for k in tasklist]
        diag_corr = diag_corr.drop(columns=drop_columns)
        return diag_corr

    def plot_diag_by_pos(self):
        tasklist = list(self.TaskDict.keys())
        diag_corr = {'%s_%s' % (k, j): [] for k in tasklist for j in tasklist}
        for k, v in self.corr_animal.items():
            for n1, i in enumerate(range(0, np.size(v, 0), self.spatialbins)):
                for n2, j in enumerate(range(0, np.size(v, 0), self.spatialbins)):
                    data = np.diag(v[i:i + self.spatialbins, j:j + self.spatialbins])
                    diag_corr['%s_%s' % (tasklist[n1], tasklist[n2])].append(data)
        return diag_corr

    def get_mean_error_withtrack(self, ax, data, tasklist):
        bins = np.linspace(0, 40, 5, dtype=int)
        df = pd.DataFrame(columns=['Task', 'Track', 'Average'])
        tracktype = ['Beg', 'Mid', 'End']

        for i in tasklist:
            for n, a in enumerate(self.animals):
                beg = np.nanmean(data[i][n][bins[0]:bins[1]])
                mid = np.nanmean(data[i][n][bins[1]:bins[2]])
                end = np.nanmean(data[i][n][bins[3]:bins[4]])
                for ttype, tdata in zip(tracktype, [beg, mid, end]):
                    df = df.append({'Task': i,
                                    'Track': ttype,
                                    'Average': tdata,
                                    'Animal': a}, ignore_index=True)

        sns.pointplot(data=df, x='Track', y='Average', hue='Task', ci=95, dodge=True, ax=ax)
        sns.stripplot(data=df, x='Track', y='Average', hue='Task', s=5, dodge=True, ax=ax)
        ax.legend_.remove()
        pf.set_axes_style(ax, numticks=3)
        return df

    def get_mean_error_withtrack_all(self, data, tasklist):
        bins = np.linspace(0, 40, 5, dtype=np.int)
        df = pd.DataFrame(columns=['Task', 'Track', 'Average'])
        tracktype = ['Beg', 'Mid', 'End']
        trackdict = {k:[] for k in tracktype+['Task']}

        for i in tasklist:
            for n, a in enumerate(self.animals):
                trackdict['Beg'].extend(data[i][n][bins[0]:bins[1]])
                trackdict['Mid'].extend(data[i][n][bins[1]:bins[2]])
                trackdict['End'].extend(data[i][n][bins[3]:bins[4]])
                trackdict['Task'].extend([i]*len(data[i][n][bins[0]:bins[1]]))


        return trackdict


    def pop_vec_allanimals(self):
        compileddata = {k: [] for k in self.TaskDict.keys()}
        for a in self.animals:
            print(a)
            population_vec = self.get_pop_vec(a)
            lickstop_df = pd.read_csv(os.path.join(self.LickFolder, 'Lickstops.csv'), index_col=0)
            lickstop = lickstop_df.loc[a, lickstop_df.columns[1]]
            for n, t in enumerate(self.TaskDict):
                taskdata = population_vec[t]
                taskdata = (taskdata > 0).astype(np.int_)
                if t == 'Task3':
                    laps_required = lickstop
                    taskdata = taskdata[laps_required:, :, :]
                taskdata = np.mean(taskdata, 0)
                compileddata[t].extend(taskdata)
        for t in compileddata:
            compileddata[t] = np.asarray(compileddata[t])
            print(t, np.shape(compileddata[t]))
        return compileddata
