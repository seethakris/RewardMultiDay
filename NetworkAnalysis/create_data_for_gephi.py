import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats
import networkx as nx
import warnings
warnings.filterwarnings("ignore")


class GetData(object):
    def __init__(self, FolderName, CombinedDataFolder):
        self.FolderName = FolderName
        self.CombinedDataFolder = CombinedDataFolder
        self.animals = [f for f in os.listdir(self.FolderName) if f not in ['.DS_Store']][1:]

        self.placecelldf = self.get_placecell_csv()

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
        # print(combined_dataframe.columns)
        return combined_dataframe[['Task', 'CellNumber', 'PlaceCellNumber', 'NumPlacecells', 'Reliability',
                                   'WeightedCOM', 'animalname']]
    
    def find_numcells(self, animalname, basetask, reliability_type):
        if reliability_type == 'High':
            numcells = self.placecelldf[(self.placecelldf['Reliability']>=0.5) &
                                        (self.placecelldf['animalname']==animalname) & 
                                        (self.placecelldf['Task']==basetask)]['CellNumber'].to_list()
            
        elif reliability_type == 'Low':
            numcells = self.placecelldf[(self.placecelldf['Reliability']<0.5) &
                                        (self.placecelldf['animalname']==animalname) & 
                                        (self.placecelldf['Task']==basetask)]['CellNumber'].to_list()
        
        else:
            numcells = self.placecelldf[(self.placecelldf['animalname']==animalname) & 
                                        (self.placecelldf['Task']==basetask)]['CellNumber'].to_list()
        return numcells
    
    def get_similarnumbers_acrosstrack(self, animalname, basetask, reliability_type):
        if reliability_type == 'High':
            df = self.placecelldf[(self.placecelldf['Reliability']>=0.5) &
                                        (self.placecelldf['animalname']==animalname) & 
                                        (self.placecelldf['Task']==basetask)]
            
        else:
            df = self.placecelldf[(self.placecelldf['Reliability']<0.5) &
                                        (self.placecelldf['animalname']==animalname) & 
                                        (self.placecelldf['Task']==basetask)]
            
        df['Binnedloc'] = np.digitize(df['WeightedCOM'], bins=np.arange(0, 45, 8))
        group = df.groupby(by='Binnedloc')['CellNumber'].count()
        min_bin, arg_bin = group.min(), group.argmin()
        # print(group)

        subsample_df =pd.DataFrame()
        for i in df['Binnedloc'].unique():
            # print(group.iloc[i-1])
            if min_bin<5: #for really small minimas
                new_bin = 10
                if i ==arg_bin+1 or group.iloc[i-1]<=10:
                    subsample_df = pd.concat((subsample_df, df[df['Binnedloc']==i]))
                else:
                    newdf = df[df['Binnedloc']==i].sample(new_bin)
                    subsample_df = pd.concat((subsample_df, newdf))
            else:
                newdf = df[df['Binnedloc']==i].sample(min_bin)
                subsample_df = pd.concat((subsample_df, newdf))

        numcells = subsample_df['CellNumber'].to_list()
        group = subsample_df.groupby(by='Binnedloc')['CellNumber'].count()
        # print(group)
        return numcells

    def get_adjacency_matrix(self, tasktocompare, SaveFolder, basetask='Task1', 
                             corr_thresh=0.1, reliability_type='High', subsample=False,
                            shuffle_flag=False, **kwargs):
        
        if not os.path.exists(SaveFolder):
            os.mkdir(SaveFolder)
        if not os.path.exists(os.path.join(SaveFolder, '%s_%s' % (basetask, tasktocompare))):
            os.mkdir(os.path.join(SaveFolder, '%s_%s' % (basetask, tasktocompare)))

        numcell_list = {k:[] for k in self.animals}
        graph_all_animals = pd.DataFrame()
        for a in self.animals:
            print(a)
            PlaceFieldData = [f for f in os.listdir(os.path.join(self.FolderName, a, 'Behavior')) if
                              (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
            pf_file = [f for f in PlaceFieldData if tasktocompare in f or basetask in f]
            if tasktocompare==basetask:
                pf_file.append(pf_file[0])
            pf_data = [scipy.io.loadmat(os.path.join(self.FolderName, a, 'Behavior', f)) for f in pf_file]

            if subsample:
                numcells = self.get_similarnumbers_acrosstrack(a, basetask=basetask, reliability_type=reliability_type)
            else:
                numcells = self.find_numcells(a, basetask=basetask, reliability_type=reliability_type)

            if reliability_type=='High':
                numcell_list[a] = len(numcells)
            if reliability_type=='Low' and 'high_rel_numcells' in kwargs:
                print(np.shape(numcells))
                rand_numcell = kwargs['high_rel_numcells'][a]
                numcells = np.random.choice(numcells, rand_numcell)
                print(np.shape(numcells))
            

            numlaps = np.min((np.size(pf_data[0]['Allbinned_F'][0, 0], 1), np.size(pf_data[1]['Allbinned_F'][0, 0], 1)))
            numlaps = int(numlaps)

            print('Analysing..%d cells' % len(numcells))
            corr_matrix = np.zeros((np.size(numcells), np.size(numcells)))

            for n1, cell1 in enumerate(numcells):
                task1_filtered = np.nan_to_num(pf_data[0]['sig_PFs'][0][cell1])
                if task1_filtered.size == 0:
                    continue
                task1_raw = np.nan_to_num(pf_data[0]['Allbinned_F'][0][cell1])
                for n2, cell2 in enumerate(numcells):
                    task2_raw = np.nan_to_num(pf_data[1]['Allbinned_F'][0][cell2])
                    # print(task2_raw.shape)
                    if shuffle_flag:
                        task2_raw = task2_raw[np.random.permutation(task2_raw.shape[0]), :]
                    c, p = scipy.stats.pearsonr(np.nanmean(task1_raw, 1), np.nanmean(task2_raw, 1))

                    if ~np.isnan(c) and p < 0.05:
                        data_bw = task1_filtered > 0
                        numlaps_withfiring = np.size(np.where(np.max(data_bw, 0))) / np.size(task1_filtered, 1)
                        corr_matrix[n2, n1] = c * numlaps_withfiring
            
            np.save(os.path.join(SaveFolder, '%s_%s' % (basetask, tasktocompare), '%s_%s_with_%s_AdjMatrix.npy' % (a, basetask, tasktocompare)), corr_matrix)
            self.define_edges(SaveFolder, a, corr_matrix, corr_thresh, task1=basetask, task2=tasktocompare)
            self.define_nodes(SaveFolder, a, corr_matrix, numcells, corr_thresh=corr_thresh, task1=basetask, task2=tasktocompare)
        
        # return numcell_list

    def define_nodes(self, SaveFolder, animalname, adj_matrix, numcells, corr_thresh, task1, task2):
        com_array = np.zeros((np.size(numcells), 6))

        csv_file = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'PlaceCells')) if f.endswith('.csv')][0]
        # print(csv_file)
        
        pc_csv = pd.read_csv(os.path.join(self.FolderName, animalname, 'PlaceCells', csv_file))

        pc_csv = pc_csv[pc_csv['Task'] == task1]
        inds_to_keep = np.where(adj_matrix < corr_thresh)
        adj_matrix[inds_to_keep[0], inds_to_keep[1]] = 0

        for n, i in enumerate(numcells):
            com_array[n, 0] = n
            com = pc_csv.loc[pc_csv['CellNumber'] == i]['WeightedCOM'].values
            # print(com)
            # Get correlation of cell with itself
            com_array[n, 3] = adj_matrix[n, n]

            com_array[n, 4] = np.mean(adj_matrix[n, :])
            nonzeros = adj_matrix[n, :]
            nonzeros = nonzeros[nonzeros != 0]
            com_array[n, 5] = np.mean(nonzeros)
            # print(adj_matrix[n, n])
            try:
                if len(com) > 1:
                    com_array[n, 1] = com[0]
                else:
                    com_array[n, 1] = com
            except:
                print(com)
                print('WHAT')
                com_array[n, 1] = 0

        bin_com = np.digitize(com_array[:, 1], bins=np.arange(0, 45, 5))
        print(np.unique(bin_com))
        com_array[:, 2] = bin_com

        print('Nodes', com_array.shape)
        save_fn = os.path.join(SaveFolder, '%s_%s' % (task1, task2), '%s_%s_with_%s_Nodes.csv' % (animalname, task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Id,Location,Binnedlocation,AutoCorrelation,Meancorr_withzero,Meancorr_without\n")
            for row in com_array:
                f.write(f"{int(row[0])},{row[1]},{int(row[2])},{row[3]},{row[4]},{row[5]}\n")
        return com_array

    def define_edges(self, SaveFolder, animalname, adj_matrix, corr_thresh, task1, task2):
        print('Corr matrix min %0.1f and max %0.1f' % (np.amin(adj_matrix), np.amax(adj_matrix)))
        # output edge list
        # Diagonal indices
        # di = np.diag_indices(adj_matrix.shape[0])
        # adj_matrix[di] = 0
        inds_to_keep = np.where(adj_matrix > corr_thresh)
        print(f"N edges: {len(inds_to_keep[0])}")
        save_fn = os.path.join(SaveFolder, '%s_%s' % (task1, task2), '%s_%s_with_%s_Edges.csv' % (animalname, task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Source,Target,Weight,Type\n")
            for x, y in zip(inds_to_keep[0], inds_to_keep[1]):
                f.write(f"{x},{y},{adj_matrix[x,y]:.4f},Undirected\n")

    

    
if __name__=='__main__':
    main()


        


        
