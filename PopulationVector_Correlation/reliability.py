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
    def __init__(self, FolderName, CombinedDataFolder, basetask):
        self.FolderName = FolderName
        self.CombinedDataFolder = CombinedDataFolder
        self.reliabilitythresh =0.5
        self.animals = [f for f in os.listdir(self.FolderName) if f not in ['.DS_Store']][1:]
        self.combineddf = self.get_placecell_csv()

        self.updatedf, self.lapwisedf = pd.DataFrame(), pd.DataFrame()

        for a in self.animals:
            print(a)
            animalinfo = AnimalDetailsWT.MultiDaysAnimals(a)
            TaskDict = animalinfo['task_dict']
            df = self.correlate_acivity_of_allcellsbytask(a, basetask, TaskDict)
            self.updatedf = pd.concat([self.updatedf, df])

            df = self.correlate_acivity_of_allcellsbytask_lapwise(a, basetask, TaskDict)
            self.lapwisedf = pd.concat([self.lapwisedf, df])

    def get_place_cellinfo(self, animal):
        pcdata = np.load(
            os.path.join(self.FolderName, animal, 'PlaceCells', '%s_placecell_data.npz' % animal),
            allow_pickle=True)
        return pcdata

    def get_placefielddata(self, animal, alltasks):
        PlaceFieldData = \
            [f for f in os.listdir(os.path.join(self.FolderName, animal, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
        data = {k:[] for k in alltasks}

        for i in PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            if taskname in alltasks.keys():
                data[taskname] = scipy.io.loadmat(os.path.join(self.FolderName, animal, 'Behavior', i))
        return data

    def get_placecell_csv(self):
        csvfiles = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv') if
                        'common' not in f and 'reward' not in f]
        for n, f in enumerate(csvfiles):
            df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
            if n == 0:
                combined_dataframe = df
            else:
                combined_dataframe = combined_dataframe.append(df, ignore_index=True)

            idx = combined_dataframe.index[(combined_dataframe['Width'] > 120)]
            # print(idx)
            combined_dataframe = combined_dataframe.drop(idx)
        print(combined_dataframe.columns)
        return combined_dataframe[['Task', 'CellNumber', 'PlaceCellNumber', 'NumPlacecells', 'Reliability', 'FiringRatio','animalname']]

    def correlate_acivity_of_allcellsbytask(self, animal, TaskA, alltasks):
        numcells = self.get_place_cellinfo(animal)['sig_PFs_cellnum_revised'].item()
        pcdata = self.get_placefielddata(animal, alltasks)
        basedata = pcdata[TaskA]
        placecelldf = self.combineddf[self.combineddf['animalname']==animal]

        newdf = placecelldf.copy()
        for index, row in placecelldf.iterrows():
            data1 = np.nanmean(pcdata[TaskA]['Allbinned_F'][0, row['CellNumber']][:, -5:], 1)
            data2 = np.nanmean(pcdata[row['Task']]['Allbinned_F'][0, row['CellNumber']][:, :5], 1)
            # data1 = pcdata[TaskA]['Allbinned_F'][0, row['CellNumber']][:, -1]
            # data2 = pcdata[row['Task']]['Allbinned_F'][0, row['CellNumber']][:, 0]
            temp = np.corrcoef(data2, data1)[0, 1]
            newdf.loc[index, 'Correlation'] = temp

        return newdf
    
    def correlate_acivity_of_allcellsbytask_lapwise(self, animal, TaskA, alltasks):
        numcells = self.get_place_cellinfo(animal)['sig_PFs_cellnum_revised'].item()
        pcdata = self.get_placefielddata(animal, alltasks)
        basedata = pcdata[TaskA]
        placecelldf = self.combineddf[self.combineddf['animalname']==animal]

        newdf = placecelldf.copy()
        for index, row in placecelldf.iterrows():
            data1 = np.nanmean(pcdata[TaskA]['Allbinned_F'][0, row['CellNumber']][:, -5:], 1)

            for i in range(10):
                data2 = np.nan_to_num(pcdata[row['Task']]['Allbinned_F'][0, row['CellNumber']][:, i])
                temp = np.corrcoef(data2, data1)[0, 1]
                newdf.loc[index, 'Correlation_%d'%i] = temp

        return newdf


    def update_pfparams_withcommoncells(self, celldf, taskstoplot):
        updated_pfparam = self.updatedf.copy()
        updated_pfparam['CommonCells'] = False

        for a in self.animals:
            commoncells = celldf[celldf[taskstoplot].sum(axis=1)==len(taskstoplot)]
            commoncells = commoncells[commoncells['AnimalName']==a]['CellNum']
            updated_pfparam.loc[(updated_pfparam['animalname']==a) & (updated_pfparam['CellNumber'].isin(commoncells)), 'CommonCells'] = True

        return updated_pfparam

    def plot_regressionline(self, ax, taskstoplot, commoncells=False, **kwargs):
        if commoncells:
            commondf = self.update_pfparams_withcommoncells(kwargs['celldata'], taskstoplot)
            thisdata = commondf[commondf['CommonCells']]
        else:
            thisdata = self.updatedf.copy()
        thisdata = thisdata.dropna(subset=['Reliability', 'Correlation'])
        for n, t in enumerate(taskstoplot):
            data = thisdata[thisdata['Task']==t]
            print(len(data))
            sns.regplot(y='Reliability', x='Correlation', data= data, marker='o',
                        scatter_kws={'s':10}, ax=ax[n])
            c, p = scipy.stats.pearsonr(data['Reliability'], data['Correlation'])
            print('Task %s:, Corr=%0.3f, P=%0.3f' %(t, c, p))
            ax[n].set_title(t)
            pf.set_axes_style(ax[n])

    def divide_by_reliability(self, taskstoplot):
        datadict = {k:[] for k in ['Correlation', 'Type', 'Task', 'Animal']}
        for n, t in enumerate(taskstoplot):
            thisdata = self.updatedf[self.updatedf['Task']==t]
            for i in thisdata['animalname'].unique():
                datadict['Correlation'].append(np.nanmean(thisdata[(thisdata['Reliability']>=self.reliabilitythresh) & (thisdata['animalname']==i)]['Correlation']))
                datadict['Type'].append('high reliability')
                datadict['Task'].append(t)
                datadict['Animal'].append(i)

                datadict['Correlation'].append(np.nanmean(thisdata[(thisdata['Reliability']<self.reliabilitythresh) & (thisdata['animalname']==i)]['Correlation']))
                datadict['Type'].append('low reliability')
                datadict['Task'].append(t)
                datadict['Animal'].append(i)
        df = pd.DataFrame.from_dict(datadict)
        return df

    def plot_by_reliability(self, ax, data, taskstoplot):
        for n, t in enumerate(taskstoplot):
            sns.barplot(data=data[data['Task']==t], x='Type', y='Correlation', ax=ax[n])
            thisdata = data[data['Task']==t].pivot(index='Animal', columns='Type', values='Correlation').reset_index()
            for i, r in thisdata.iterrows():
                ax[n].plot([0.25, 0.75], [thisdata['high reliability'], thisdata['low reliability']], 'k.-')
            ax[n].set_title(t)

            thisdata = thisdata.dropna()
            tstat, p = scipy.stats.ttest_rel(thisdata['high reliability'], thisdata['low reliability'])
            print('Task: %s, t=%0.3f, p=%0.3f' %(t, tstat, p))
            pf.set_axes_style(ax[n])
            # ax[n].set_xlim((0, 1))

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

    def plot_lapwise_correlation(self, ax, taskstoplot):
        df = self.lapwisedf.copy()
        column_name = ['Correlation_%d' %c for c in range(10)]
        x_axis = [-1, +1]
        color = ['black', 'grey']
        marker = ['o', 'x']
        for ax_n, t in enumerate(taskstoplot):
            animalwisedf_high, animalwisedf_low = pd.DataFrame(), pd.DataFrame()
            for a in self.animals:
                thisdf = df[(df['Task']==t) & (df['Reliability']>=self.reliabilitythresh) & (df['animalname']==a)][column_name] #High reliability
                animalwisedf_high = pd.concat([animalwisedf_high, thisdf.mean(axis=0)], axis=1)
                if a == 'NR34' and t=='Task5':
                    print(a, t, thisdf)
                thisdf = df[(df['Task']==t) & (df['Reliability']<self.reliabilitythresh)& (df['animalname']==a)][column_name] #Low reliability
                animalwisedf_low = pd.concat([animalwisedf_low, thisdf.mean(axis=0)], axis=1)
                

            for n1, x in enumerate([animalwisedf_high, animalwisedf_low]):
                ax[ax_n].errorbar(np.arange(10), x.mean(axis=1), yerr=x.sem(axis=1))
                for n2, (index, row) in enumerate(x.iterrows()):
                    ax[ax_n].plot([n2+(x_axis[n1]*0.25)]*len(row), row, linewidth=0, marker=marker[n1], markersize=3, color=color[n1])
            ax[ax_n].set_title(t)

            # print(animalwisedf_high)
            # print(animalwisedf_low)
            # print('P-values for task %s' %t)
            # for (i1, d1), (_, d2) in zip(animalwisedf_high.iterrows(), animalwisedf_low.iterrows()):
            #     d1, d2 = d1.to_numpy(), d2.to_numpy()
            #     if t=='Task5':
            #         d1 = d1[1:]
            #         d2 = d2[1:]
            #     t1, p = scipy.stats.ttest_rel(d1, d2)
            #     print('Lap: %s: t=%0.3f, p=%0.3f' %(i1, t1, p))
            # print('')