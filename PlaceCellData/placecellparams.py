import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
from copy import copy
import scipy.stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dabest
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multicomp as mc


# For plotting styles
PlottingFormat_Folder = '/Users/seetha/Box Sync/MultiDayData/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

PvaluesFolder = '/Users/seetha/Box Sync/NoReward/Scripts/Figure1/'
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues

class Combinedplots:
    def __init__(self, FolderName, CombinedDataFolder):
        self.CombinedDataFolder = CombinedDataFolder
        self.FolderName = FolderName

        csvfiles_pfs = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv') if
                        'common' not in f and 'reward' not in f]
        self.npzfiles = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.npz')]
        self.trackbins = 5
        self.tracklength = 200
        self.numanimals = len(csvfiles_pfs)
        # Combined pf dataframes into one big dataframe
        self.pfparam_combined = self.combineanimaldataframes(csvfiles_pfs)
        self.animals = self.pfparam_combined['animalname'].unique()

        print(csvfiles_pfs)

    def combineanimaldataframes(self, csvfiles, common_flag=False):
        for n, f in enumerate(csvfiles):
            df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
            if n == 0:
                combined_dataframe = df
            else:
                combined_dataframe = combined_dataframe.append(df, ignore_index=True)

            idx = combined_dataframe.index[(combined_dataframe['Width'] > 120)]
            print(idx)
            combined_dataframe = combined_dataframe.drop(idx)
        return combined_dataframe

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

    def update_pfparams_withcommoncells(self, celldf, taskstoplot):
        updated_pfparam = self.pfparam_combined.copy()
        updated_pfparam['CommonCells'] = False

        for a in self.animals:
            commoncells = celldf[celldf[taskstoplot].sum(axis=1)==len(taskstoplot)]
            commoncells = commoncells[commoncells['AnimalName']==a]['CellNum']
            updated_pfparam.loc[(updated_pfparam['animalname']==a) & (updated_pfparam['CellNumber'].isin(commoncells)), 'CommonCells'] = True

        return updated_pfparam

    def plot_pfparams(self, ax, tasks_to_plot, columns_to_plot, alltaskpresent=False, commoncellflag=False):
        # Plot a combined historgram and a boxplot of means
        if commoncellflag:
            celldf = self.get_cells_pertask_peranimal(tasks_to_plot)
            updatedpf_param = self.update_pfparams_withcommoncells(celldf, tasks_to_plot)
            df_plot = updatedpf_param[(updatedpf_param['Task'].isin(tasks_to_plot)) & (updatedpf_param['CommonCells'])]
        else:
            df_plot = self.pfparam_combined[self.pfparam_combined['Task'].isin(tasks_to_plot)]

        print(len(df_plot))
        for n1, c in enumerate(columns_to_plot):
            # Plot boxplot
            group = df_plot.groupby(by=['animalname', 'Task']).mean()[c].reset_index()
            group = group.pivot(index='animalname', columns='Task')
            group.columns = group.columns.droplevel()

            x = [0.25, 1.25, 2.25, 3.25, 4.25]
            for n, row in group.iterrows():
                ax[n1, 0].plot(x, row, 'ko-', markerfacecolor='none', zorder=2, color='lightgrey')

            group = group.melt(value_name=c)
            comp1 = mc.MultiComparison(group[c], group['Task'])
            tbl, a1, a2 = comp1.allpairtest(scipy.stats.ttest_rel, method= "bonf")
            print(tbl)
            sns.boxplot(x='Task', y=c, data=group, ax=ax[n1, 0], order=tasks_to_plot, showfliers=False)

            ax[n1, 0].set_xlabel('')
            for n2, t in enumerate(tasks_to_plot):
                d = df_plot[df_plot['Task'] == t][c]
                d = d[~np.isnan(d)]
                ax[n1, 1].hist(d, bins=1000, density=True, cumulative=True, label='CDF',
                               histtype='step', linewidth=0.5)

        for a in ax.flatten():
            pf.set_axes_style(a, numticks=3)


    def plot_com_hist(self, ax, combined_dataframe, tasks_to_plot, bins=20):
        for n, taskname in enumerate(tasks_to_plot):
            task_data = combined_dataframe[(combined_dataframe.Task == taskname)]['WeightedCOM'] * self.trackbins
            weights = np.ones_like(task_data) / float(len(task_data))
            ax1 = ax[n].twinx()
            #
            sns.distplot(task_data, ax=ax1, bins=40, color='black',
                         kde=True, hist=False,
                         kde_kws={'kernel': 'gau', 'bw_adjust': 0.2, 'shade': True, 'cut': 0, 'lw': 0,
                                  'color': [0.6, 0.6, 0.6],
                                  'alpha': 0.9})

            sns.distplot(task_data, ax=ax[n], bins=40, color='black',
                         kde=False, hist=True, hist_kws={'histtype': 'step', 'color': 'k', 'lw': 1, 'weights': weights})

            dist = np.array([])
            for i in range(1000):
                uniform = np.random.uniform(0, 200, len(task_data))
                dist = np.hstack((uniform, dist)) if dist.size else uniform
                # print(np.max(dist))
            weights_shuffle = np.ones_like(dist) / float(len(dist))
            hist_shuffle, bin_edges = np.histogram(dist, 40, density=True, weights=weights_shuffle)
            sns.distplot(dist, ax=ax[n], bins=40, color='black',
                         kde=False, hist=True, hist_kws={'histtype': 'step', 'color': 'lightgrey', 'lw': 1, 'weights': weights_shuffle})

            hist_taskdata, bin_edges = np.histogram(task_data, 40, density=True, weights=weights)
            print(np.shape(hist_taskdata))
            for i in ['Beg', 'Mid', 'End']:
                if i == 'Beg':
                    data1 = hist_shuffle[:10]
                    data2 = hist_taskdata[:10]
                elif i == 'Mid':
                    data1 = hist_shuffle[10:30]
                    data2 = hist_taskdata[10:30]
                else:
                    data1 = hist_shuffle[35:]
                    data2 = hist_taskdata[35:]
                tstat, pvalue = scipy.stats.ttest_ind(data1, data2)
                print('P-Value : %s : tstat: %0.3f, p %0.3f' %(i, tstat, pvalue))

            ax1.set_ylim((0, 0.015))
            ax[n].set_ylim((0, 0.07))
            # ax[n].set_xlim((0, 200))
            # ax[n].set_yticks((0, 0.03, 0.06))
            # ax[n].set_yticklabels((0, 3, 6.5))
            ax[n].set_xlabel('COM')
            ax[n].set_title(taskname)
            ax1.set_yticks(())
            sns.despine(right=True, top=True, left=True, bottom=True, ax=ax1)
            sns.despine(right=True, top=True, ax=ax[n])
        ax[0].set_ylabel('Percent of fields')

    def calculate_ratiofiring_atrewzone(self, ax, combined_dataframe, tasks_to_compare, ranges, rows_to_plot=['Mid', 'End']):
        cellratio_df = pd.DataFrame(columns=['Beg', 'Mid', 'End', 'Diff', 'Animal', 'TaskName'])
        cellratio_dict = {k: [] for k in tasks_to_compare}

        animalnames = np.unique(combined_dataframe.animalname)

        for n1, a in enumerate(animalnames):
            for n2, taskname in enumerate(tasks_to_compare):
                normfactor = np.sum(
                    np.load(os.path.join(self.CombinedDataFolder,
                                         [f for f in self.npzfiles if animalnames[n1] in f][0]),
                            allow_pickle=True)['numcells'].item())
                data = combined_dataframe[(combined_dataframe.Task == taskname) & (combined_dataframe.animalname == a)]

                if len(data):
                    g = data.groupby(pd.cut(data.WeightedCOM * self.trackbins, ranges)).count()['WeightedCOM'].tolist()

                    # Uniform distribution
                    shuffle = []
                    shuffle_track = []
                    for i in range(100):
                        data['Shuffle'] = np.random.uniform(0, 40, len(data))
                        g_shuffle = data.groupby(pd.cut(data.Shuffle * self.trackbins, ranges)).count()[
                            'Shuffle'].tolist()
                        shuffle_track.append((g_shuffle[0] / sum(g_shuffle)) * 100)
                        shuffle.append(
                            ((g_shuffle[-1] / sum(g_shuffle)) - (np.mean(g_shuffle[1:3]) / sum(g_shuffle))) * 100)

                    cellratio_df = cellratio_df.append({'Beg': (g[0] / sum(g)) * 100,
                                                        'Mid': (np.mean(g[1:3]) / sum(g)) * 100,
                                                        'End': (g[-1] / sum(g)) * 100,
                                                        'Diff': ((g[-1] / sum(g)) - (np.mean(g[1:3]) / sum(g))) * 100,
                                                        'Shuffle_Diff': np.mean(shuffle),
                                                        'Shuffle_End' : np.mean(shuffle_track),
                                                        'Animal': a, 'TaskName': taskname},
                                                       ignore_index=True)
                    cellratio_dict[taskname].append(g / normfactor)

        x = [0.5, 1.5, 2.5, 3.5, 4.5]
        sns.boxplot(x='TaskName', y='Diff', data=cellratio_df, width=0.6, showfliers=False, zorder=1, ax=ax[-1])
        plot_ind = cellratio_df.loc[cellratio_df['TaskName'].isin(tasks_to_compare)]
        plot_ind = plot_ind.pivot(columns='TaskName', values='Diff', index='Animal')
        for index, rows in plot_ind.iterrows():
            ax[-1].plot(x, rows, 'o-', markerfacecolor='none', zorder=2, color='lightgrey')
        ax[-1].set_xticklabels(tasks_to_compare)
        ax[-1].axhline(np.mean(cellratio_df['Shuffle_Diff']), color='k', linestyle='--')
        print(plot_ind.describe())

        df = cellratio_df.melt(id_vars=['Animal', 'TaskName'], var_name='Track', value_name='Ratio')
        for n, taskname in enumerate(tasks_to_compare):
            print(taskname)
            sns.boxplot(y='Ratio', x='Track',
                        data=df.loc[(df['TaskName'] == taskname) & (df['Track'].isin(rows_to_plot))],
                        width=0.6, showfliers=False, zorder=1, ax=ax[n])

            plot_ind = cellratio_df.loc[cellratio_df['TaskName'] == taskname][rows_to_plot]
            for index, rows in plot_ind.iterrows():
                ax[n].plot(x[:len(rows_to_plot)], rows, 'o-', markerfacecolor='none', zorder=2, color='lightgrey')
            ax[n].set_xlim((-0.5, x[len(rows_to_plot) - 1] + 0.5))
            ax[n].axhline(np.mean(cellratio_df['Shuffle_End']), color='k', linestyle='--')
            pf.set_axes_style(ax[n])
            ax[n].set_ylabel('Percentage of place fields')
            ax[n].set_title(taskname)

        # mannwhitney test
        for t in tasks_to_compare[1:]:
            data1 = cellratio_df[cellratio_df.TaskName == tasks_to_compare[0]]['Diff']
            data2 = cellratio_df[cellratio_df.TaskName == t]['Diff']

            print(scipy.stats.ttest_rel(np.asarray(data1, dtype='float32'), np.asarray(data2, dtype='float32')))

        return cellratio_df

class CommonFunctions:
    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m-h, m+h
