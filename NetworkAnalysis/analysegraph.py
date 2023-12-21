import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats

class PlotData(object):
    def __init__(self, FolderName):
        self.FolderName = FolderName
        
    def concat_files(self, reliability_flag='high'):
        files = [f for f in os.listdir(self.FolderName) if reliability_flag in f]
        reldf = pd.DataFrame()
        for i in files:
            animalname = i[:i.find('_')]
            df = pd.read_csv(os.path.join(self.FolderName, i))
            df['Id'] = animalname
            reldf = pd.concat((reldf, df))

        reldf['Reliability'] = reliability_flag
        return reldf

    def group_df_bylocation(self, df, column_name):
        rel_g = df.groupby(by='binnedlocation')[column_name].agg(['mean', 'sem']).reset_index()
        return rel_g
    
    def group_df_byanimal(self, df, column_name):
        rel_g = df.groupby(by=['Id', 'Reliability'])[column_name].agg(['mean', 'sem', 'count']).reset_index()
        return rel_g
    
    def select_graph_node_bylocation(self, df, column_name):
        rel_g = df.groupby(by='binnedlocation')[column_name].count()
        print(rel_g)
        num_to_sample = rel_g.min()

        subsample_df = pd.DataFrame()
        for i in range(1, np.max(df['binnedlocation'])+1):
            newdf = df[df['binnedlocation']==i].sample(num_to_sample)
            subsample_df = pd.concat((subsample_df, newdf))

        rel_g = subsample_df.groupby(by='binnedlocation')[column_name].agg(['mean', 'sem', 'count']).reset_index()
        # print(rel_g)
        return rel_g
    
    def normalize_graph_node_bylocation(self, df, column_name):
        rel_g = df.groupby(by='binnedlocation')[column_name].agg(['mean', 'sem', 'count']).reset_index()
        rel_g['mean'] = rel_g['mean']/rel_g['count']
        rel_g['sem'] = rel_g['sem']/rel_g['count']
        print(rel_g)
        return rel_g
    
    def plot_column(self, ax, df_high, high_group, df_low, low_group, column_name):
        for i in [high_group, low_group]:
            ax[0].plot(i['mean'])
            ax[0].fill_between(np.arange(i['mean'].shape[0]), i['mean'] - i['sem'], i['mean'] + i['sem'], alpha=0.5)
            
        # df_task1_task2 = pd.concat((df_high, df_low))
        # sns.barplot(x='Reliability', y=column_name, data=df_task1_task2, ax=ax[1])
        # group = self.group_df_byanimal(df_task1_task2, column_name=column_name)
        # group = group.pivot(index='Id', values='mean', columns='Reliability')

        df_task1_task2 = pd.concat((df_high, df_low))
        group = self.group_df_byanimal(df_task1_task2, column_name=column_name)
        sns.barplot(x='Reliability', y='mean', data=group, order=['high', 'low'], ax=ax[1])
        group = group.pivot(index='Id', values='mean', columns='Reliability')
        print(group.sem(axis=0))
        for i, r in group.iterrows():
            ax[1].plot([0, 1], r, 'k.-')

        t1, p = scipy.stats.ttest_rel(group['high'].to_numpy(), group['low'].to_numpy());
        print('t=%0.3f, p=%0.3f' %(t1, p))

        if column_name=='Weighted Degree':
            for a in ax:
                a.set_ylim((0, 45))
        else:
            ax[0].set_ylim((0.5, 0.8))
            ax[1].set_ylim((0, 0.8))

    def plot_normalized_column(self, ax, df_high, high_group, df_low, low_group, column_name):
        for i in [high_group, low_group]:
            ax[0].plot(i['mean'])
            ax[0].fill_between(np.arange(i['mean'].shape[0]), i['mean'] - i['sem'], i['mean'] + i['sem'], alpha=0.5)
            
        df_task1_task2 = pd.concat((df_high, df_low))
        # 
        group = self.group_df_byanimal(df_task1_task2, column_name=column_name)
        group['mean'] = group['mean']/group['count']
        group['sem'] = group['sem']/group['count']
        print(group)
        sns.barplot(x='Reliability', y='mean', data=group, order=['high', 'low'], ax=ax[1])
        group = group.pivot(index='Id', values='mean', columns='Reliability')
        

        for i, r in group.iterrows():
            ax[1].plot([0, 1], r, 'k.-')

        t1, p = scipy.stats.ttest_rel(group['high'].to_numpy(), group['low'].to_numpy());
        print('t=%0.3f, p=%0.3f' %(t1, p))
        