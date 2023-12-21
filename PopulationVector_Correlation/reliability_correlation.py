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
    def __init__(self, FolderName, CombinedDataFolder, basetask, taskstoplot):
        self.FolderName = FolderName
        self.CombinedDataFolder = CombinedDataFolder
        self.reliabilitythresh =0.5
        self.taskstoplot = taskstoplot
        self.basetask = basetask
        self.animals = [f for f in os.listdir(self.FolderName) if f not in ['.DS_Store']][1:]
        self.combineddf = self.get_placecell_csv()

        self.animal_highrel = {k:[] for k in taskstoplot}
        self.animal_lowrel = {k:[] for k in taskstoplot}
        self.corrdf = pd.DataFrame()

        for a in self.animals:
            print(a)
            temp_dict = self.correlate_acivity_of_allcellsbytask_lapwise(animal=a, TaskA=self.basetask, 
                                                             taskstoplot=self.taskstoplot,
                                                             reliability_type='High')
            for t in taskstoplot:
                self.animal_highrel[t].append(np.nanmean(temp_dict[t], 0))

            temp_dict = self.correlate_acivity_of_allcellsbytask_lapwise(animal=a, TaskA=self.basetask, 
                                                             taskstoplot=self.taskstoplot,
                                                             reliability_type='Low')
            
            for t in taskstoplot:
                self.animal_lowrel[t].append(np.nanmean(temp_dict[t], 0))

            df = self.reliability_vs_correlation(animal=a, TaskA=self.basetask, 
                                                taskstoplot=self.taskstoplot)
            self.corrdf = pd.concat([self.corrdf, df])
            

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
            if taskname in alltasks:
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

    def correlate_acivity_of_allcellsbytask_lapwise(self, animal, TaskA, taskstoplot, reliability_type='High'):
        pcdata = self.get_placefielddata(animal, TaskA + taskstoplot)
        placecelldf = self.combineddf[self.combineddf['animalname']==animal]

        if reliability_type == 'High':
            reliablecells = placecelldf[(placecelldf['Reliability']>=self.reliabilitythresh) 
                                            & (placecelldf['Task']==TaskA[0])]['CellNumber']
        else:
            reliablecells = placecelldf[(placecelldf['Reliability']<self.reliabilitythresh) 
                                            & (placecelldf['Task']==TaskA[0])]['CellNumber']
        print(reliability_type, np.shape(reliablecells))
        
        dict_corr = {k:np.array([]) for k in taskstoplot}
        for t in taskstoplot:
            for c in reliablecells:
                data1 = np.nanmean(pcdata[TaskA[0]]['Allbinned_F'][0, c][:, -5:], 1)
                lapwise = []
                for i in range(10):
                    data2 = np.nan_to_num(pcdata[t]['Allbinned_F'][0, c][:, i])
                    temp = np.corrcoef(data2, data1)[0, 1]
                    lapwise.append(temp)
                lapwise = np.asarray(lapwise)
                dict_corr[t] = np.vstack((lapwise, dict_corr[t])) if dict_corr[t].size else lapwise
        
        return dict_corr
    
    def reliability_vs_correlation(self, animal, TaskA, taskstoplot):
        pcdata = self.get_placefielddata(animal, TaskA + taskstoplot)
        placecelldf = self.combineddf[self.combineddf['animalname']==animal]

        cells = placecelldf[placecelldf['Task']==TaskA[0]]['CellNumber']
        corr_dict = {k:[] for k in ['Task', 'Reliability', 'Correlation', 'CellNumber', 'AnimalName']}

        for t in taskstoplot:
            for c in cells:
                data1 = np.nanmean(pcdata[TaskA[0]]['Allbinned_F'][0, c][:, -5:], 1)
                data2 = np.nanmean(pcdata[t]['Allbinned_F'][0, c][:, :5], 1)
                corr_dict['Correlation'].append(np.corrcoef(data2, data1)[0, 1])
                corr_dict['Task'].append(t)
                corr_dict['AnimalName'].append(animal)
                corr_dict['CellNumber'].append(c)
                reliability = placecelldf[(placecelldf['Task']==TaskA[0]) & (placecelldf['CellNumber']==c)]['Reliability'].to_numpy()

                if len(reliability)>1:
                    corr_dict['Reliability'].append(np.max(reliability))
                else:
                    corr_dict['Reliability'].append(reliability[0])
        
        return pd.DataFrame.from_dict(corr_dict)

    
    def plot_lapwise_correlation(self, ax):
        for ax_n, t in enumerate(self.taskstoplot):
            x_axis = [-1, +1]
            color = ['black', 'grey']
            marker = ['o', 'x']

            for n1, x in enumerate([self.animal_highrel[t], self.animal_lowrel[t]]):
                to_plot = np.array(x)
                ax[ax_n].errorbar(np.arange(10), np.mean(x, 0), scipy.stats.sem(x, 0))
                print(np.shape(to_plot))
                row, col = to_plot.shape
                for n2 in range(col):
                    ax[ax_n].plot([n2+(x_axis[n1]*0.25)]*row, to_plot[:, n2], 
                                  linewidth=0, marker=marker[n1], markersize=3, color=color[n1])
            
            d1, d2 = np.array(self.animal_highrel[t]), np.array(self.animal_lowrel[t])
            for i in range(d1.shape[1]):
                t1, p = scipy.stats.ttest_rel(d1[:, i], d2[:, i])
                print('Lap: %s: t=%0.3f, p=%0.3f' %(i, t1, p))

    def plot_by_reliability(self, ax, taskstoplot):
        self.corrdf['Reliability_Type'] = 'High'
        self.corrdf.loc[self.corrdf['Reliability']<0.5, 'Reliability_Type'] = 'Low'

        for n, t in enumerate(taskstoplot):
            data = self.corrdf[self.corrdf['Task']==t]
            data = data.dropna()
            sns.regplot(y='Reliability', x='Correlation', data=data, marker='o',
                            scatter_kws={'s':10, 'alpha':0.5}, ax=ax[0, n], color='Grey')
            c, p = scipy.stats.pearsonr(data['Reliability'], data['Correlation'])
            print('Task %s:, Corr=%0.3f, P=%0.3f' %(t, c, p))
            ax[0, n].set_title(t)
            pf.set_axes_style(ax[0, n])

            sns.barplot(data=data, x='Reliability_Type', y='Correlation', ax=ax[1, n], order=['High', 'Low'])
            thisdata = data.groupby(by=['AnimalName', 'Reliability_Type'])['Correlation'].mean().reset_index()
            thisdata = thisdata.pivot(index='AnimalName', columns='Reliability_Type', values='Correlation').reset_index()

            for i, r in thisdata.iterrows():
                ax[1, n].plot([0.25, 0.75], [thisdata['High'], thisdata['Low']], 'k.-')

            tstat, p = scipy.stats.ttest_rel(thisdata['High'], thisdata['Low'])
            print('Task: %s, t=%0.3f, p=%0.3f' %(t, tstat, p))
