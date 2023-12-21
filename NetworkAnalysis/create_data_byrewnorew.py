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
    
    def find_numcells(self, animalname, basetask):
        numcells = self.placecelldf[(self.placecelldf['animalname']==animalname) & 
                                    (self.placecelldf['Task']==basetask)]['CellNumber'].to_list()
        return numcells
    
    def get_similarnumbers_acrosstrack(self, animalname, basetask,  subsample_cells=False, **kwargs):
        cells = self.placecelldf[(self.placecelldf['animalname']==animalname) & 
                                            (self.placecelldf['Task']==basetask)]['CellNumber'].to_list()
        
        if subsample_cells:
            df = df.sample(n=kwargs['subsample_number'])
            print(len(df))
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
                             corr_thresh=0.1, subsample=False, subsample_cells=False, **kwargs):
        
        if not os.path.exists(SaveFolder):
            os.mkdir(SaveFolder)
        if not os.path.exists(os.path.join(SaveFolder, '%s_%s' % (basetask, tasktocompare))):
            os.mkdir(os.path.join(SaveFolder, '%s_%s' % (basetask, tasktocompare)))

        graph_all_animals = pd.DataFrame()
        for a in self.animals:
            # if a =='CFC16':
            #     continue
            print(a)
            PlaceFieldData = [f for f in os.listdir(os.path.join(self.FolderName, a, 'Behavior')) if
                              (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
            pf_file = [f for f in PlaceFieldData if tasktocompare in f or basetask in f]
            pf_data = [scipy.io.loadmat(os.path.join(self.FolderName, a, 'Behavior', f)) for f in pf_file]

            if subsample:
                if subsample_cells:
                    numcells = self.get_similarnumbers_acrosstrack(a, basetask=basetask, 
                                                                subsample_cells=subsample_cells,
                                                                subsample_number=kwargs['subsample_numcells'][a])
                else:
                    numcells = self.get_similarnumbers_acrosstrack(a, basetask=basetask, 
                                                                subsample_cells=subsample_cells)
            else:
                numcells = self.find_numcells(a, basetask=basetask)

            if 'high_rel_numcells' in kwargs:
                print(np.shape(numcells))
                rand_numcell = kwargs['subsample_numcells'][a]
                numcells = np.random.choice(numcells, rand_numcell)
                print(np.shape(numcells))

            numlaps = np.min((np.size(pf_data[0]['Allbinned_F'][0, 0], 1), np.size(pf_data[1]['Allbinned_F'][0, 0], 1)))
            numlaps = int(numlaps)
            # print(numlaps)

            print('Analysing..%d cells' % len(numcells))
            corr_matrix = np.zeros((np.size(numcells), np.size(numcells)))

            for n1, cell1 in enumerate(numcells):
                task1_filtered = np.nan_to_num(pf_data[0]['sig_PFs'][0][cell1])
                if task1_filtered.size == 0:
                    continue
                task1_raw = np.nan_to_num(pf_data[0]['Allbinned_F'][0][cell1])
                for n2, cell2 in enumerate(numcells):
                    task2_raw = np.nan_to_num(pf_data[1]['Allbinned_F'][0][cell2])
                    c, p = scipy.stats.pearsonr(np.nanmean(task1_raw, 1), np.nanmean(task2_raw, 1))

                    if ~np.isnan(c) and p < 0.05:
                        data_bw = task1_filtered > 0
                        numlaps_withfiring = np.size(np.where(np.max(data_bw, 0))) / np.size(task1_filtered, 1)
                        corr_matrix[n2, n1] = c * numlaps_withfiring
            
            # np.save(os.path.join(SaveFolder, '%s_%s' % (basetask, tasktocompare), '%s_%s_with_%s_AdjMatrix.npy' % (a, basetask, tasktocompare)), corr_matrix)
            # self.define_edges(SaveFolder, a, corr_matrix, corr_thresh, task1=basetask, task2=tasktocompare)
            # self.define_nodes(SaveFolder, a, corr_matrix, numcells, corr_thresh=corr_thresh, task1=basetask, task2=tasktocompare)
            graph = self.create_graph_object(adj_matrix=corr_matrix,
                                             corr_thresh=corr_thresh,
                                             numcells=numcells,
                                             animalname=a)
            graph_all_animals = pd.concat((graph_all_animals, graph))

        group_by_location = graph_all_animals.groupby(by=['BinnedLocation'])['Degree', 'Clustering'].agg(['mean', 'sem', 'count'])
        group_by_animals = graph_all_animals.groupby(by=['AnimalName'])['Degree', 'Clustering'].agg(['mean', 'sem', 'count'])
        return graph_all_animals, group_by_location, group_by_animals
        

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

    
    def create_graph_object(self, adj_matrix, corr_thresh, numcells, animalname):
        csv_file = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'PlaceCells')) if f.endswith('.csv')][0]
        # print(csv_file)
        pc_csv = pd.read_csv(os.path.join(self.FolderName, animalname, 'PlaceCells', csv_file))
        inds_to_keep = np.where(adj_matrix > corr_thresh)
        
        g = nx.Graph()
        for i in range(len(numcells)):
            g.add_node(i)
            
        for x, y in zip(inds_to_keep[0], inds_to_keep[1]):
            g.add_edge(x, y, weight=adj_matrix[x, y])
        
        degree = g.degree(range(len(numcells)), weight='weight')
        clustering = nx.clustering(g, range(len(numcells)), weight='weight')
        avg_clustering = nx.average_clustering(g, range(len(numcells)), weight='weight')

        graph_df = pd.DataFrame(columns=['Node', 'Degree', 'Clustering', 'Location', 'AnimalName'])
        for i, (n, d) in enumerate(zip(numcells, degree)):
            com = pc_csv.loc[pc_csv['CellNumber'] == n]['WeightedCOM'].values
            if not com.size:
                continue
            graph_df = graph_df.append({'Node':i, 'Degree':d[1],
                                        'Clustering': clustering[i],
                                        'AnimalName':animalname,
                                        'Location':com[0]}, ignore_index=True)
        
        graph_df['BinnedLocation'] = np.digitize(graph_df['Location'], bins=np.arange(0, 45, 8))
        return graph_df
    
    def plot_graph_factors_bytrack_iterated(self, ax, combineddf, column_name):
        all_mean = np.array([])
        for i in combineddf:
            if column_name == 'Degree':
                norm_mean = i[column_name]['mean']/i[column_name]['count']
                all_mean = np.vstack((norm_mean, all_mean)) if all_mean.size else norm_mean
            else:
                try:
                    norm_mean = i[column_name]['mean']
                    all_mean = np.vstack((norm_mean, all_mean)) if all_mean.size else norm_mean
                except:
                    print('Whatever')
        
        norm_mean = np.mean(all_mean, 0)
        norm_sem = scipy.stats.sem(all_mean, 0)

        ax.plot(np.arange(len(norm_mean)), norm_mean)
        ax.fill_between(np.arange(len(norm_mean)), norm_mean-norm_sem, norm_mean+norm_sem, alpha=0.5)

    def plot_graph_bylocation_separate(self, ax, df, column_name):
        if column_name == 'Degree':
            norm_mean = df[column_name]['mean']/df[column_name]['count']
            norm_sem = df[column_name]['sem']/df[column_name]['count']
        else:
            norm_mean = df[column_name]['mean']
            norm_sem = df[column_name]['sem']

        ax.plot(np.arange(len(norm_mean)), norm_mean)
        ax.fill_between(np.arange(len(norm_mean)), norm_mean-norm_sem, norm_mean+norm_sem, alpha=0.5)

    def plot_factors_bylocation(self, ax, high_df, low_df, column_name):
        for i in [high_df, low_df]:
            if column_name == 'Degree':
                norm_mean = i[column_name]['mean']/i[column_name]['count']
                norm_sem = i[column_name]['sem']/i[column_name]['count']
            else:
                norm_mean = i[column_name]['mean']
                norm_sem = i[column_name]['sem']

            ax.plot(np.arange(len(norm_mean)), norm_mean)
            ax.fill_between(np.arange(len(norm_mean)), norm_mean-norm_sem, norm_mean+norm_sem, alpha=0.5)

    def plot_factors_by_animal(self, ax, high_df, low_df, column_name, iterated=False):
        x =[]
        if iterated:
            if column_name == 'Degree':
                norm_mean = high_df[column_name]['mean']/high_df[column_name]['count']
            else:
                norm_mean = high_df[column_name]['mean']
            x = [norm_mean, low_df]
            for n, i in enumerate(x):
                ci = scipy.stats.norm.interval(alpha=0.99, loc=np.mean(i), scale=scipy.stats.sem(i))
                ci = np.mean(i)-ci[0]
                ax.bar(n, np.mean(i), yerr=ci)
        else:
            for n, i in enumerate([high_df, low_df]):
                if column_name == 'Degree':
                    norm_mean = i[column_name]['mean']/i[column_name]['count']
                else:
                    norm_mean = i[column_name]['mean']
                x.append(norm_mean)   

            ci = scipy.stats.norm.interval(alpha=0.99, loc=np.mean(norm_mean), scale=scipy.stats.sem(norm_mean))
            ci = np.mean(norm_mean)-ci[0]
            ax.bar(n, np.mean(norm_mean), yerr=ci)

        t1, p = scipy.stats.ttest_rel(x[0], x[1]);
        print('t=%0.3f, p=%0.3f' %(t1, p))

        if iterated:
            for n, (i1, i2) in enumerate(zip(high_df[column_name]['mean'], low_df)):
                if column_name == 'Degree':
                    norm1 = i1/high_df[column_name]['count'].to_list()[n]
                    norm2 = i2
                    ax.plot([0, 1], [norm1, norm2], 'ko-')
                else:
                    ax.plot([0, 1], [i1, i2], 'ko-')
        else:
            for n, (i1, i2) in enumerate(zip(high_df[column_name]['mean'], low_df[column_name]['mean'])):
                if column_name == 'Degree':
                    norm1 = i1/high_df[column_name]['count'].to_list()[n]
                    norm2 = i2/low_df[column_name]['count'].to_list()[n]
                    ax.plot([0, 1], [norm1, norm2], 'ko-')
                else:
                    ax.plot([0, 1], [i1, i2], 'ko-')
        

def main():
    DirectoryName = '/Users/seetha/Box Sync/MultiDayData/Data/'
    CombinedFolderName = '/Users/seetha/Box Sync/MultiDayData/PlaceCellResultsAll/'
    SaveFolder = '/Users/seetha/Box Sync/MultiDayData/NetworkAnalysis/'

    network = GetData(DirectoryName, CombinedFolderName)

    #Run iterations on sampling same cells from all locations
    taskstocompare = 'Task2'
    basetask = 'Task1'
    sample = 50
    by_location_rew, by_animal_rew = [], []
    for iteration in range(sample):
        print(iteration)
        temp, group1, group2 = network.get_adjacency_matrix(taskstocompare, basetask=basetask, subsample=True,
                                        SaveFolder=os.path.join(SaveFolder, 'Allcells_subsampled'))
        by_location_rew.append(group1)
        by_animal_rew.append(group2)

    savefile = os.path.join(SaveFolder, 'Allcells_subsampled', '%s_%s' % (basetask, taskstocompare), 'by_location_iterated.csv')
    pd.concat(by_location_rew).to_csv(savefile)
    savefile = os.path.join(SaveFolder, 'Allcells_subsampled', '%s_%s' % (basetask, taskstocompare), 'by_animal_iterated.csv')
    pd.concat(by_animal_rew).to_csv(savefile)


    taskstocompare = 'Task4'
    basetask = 'Task3'
    by_location_norew, by_animal_norew = [], []
    sample = 50
    # numcells = {'NR34':200, 'CFC17':200, 'NR32':183, 'CFC16':200, 'CFC18':200}
    for iteration in range(sample):
        print(iteration)
        # Subsample location
        temp, group1, group2 = network.get_adjacency_matrix(taskstocompare, basetask=basetask,subsample=True,
                                        SaveFolder=os.path.join(SaveFolder, 'Allcells_subsampled'))
        by_location_norew.append(group1)
        by_animal_norew.append(group2)

    savefile = os.path.join(SaveFolder, 'Allcells_subsampled', '%s_%s' % (basetask, taskstocompare), 'by_location_iterated.csv')
    pd.concat(by_location_norew).to_csv(savefile)
    savefile = os.path.join(SaveFolder, 'Allcells_subsampled', '%s_%s' % (basetask, taskstocompare), 'by_animal_iterated.csv')
    pd.concat(by_animal_norew).to_csv(savefile)

    
if __name__=='__main__':
    main()


        


        
