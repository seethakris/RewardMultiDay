import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats

class GetData(object):
    def __init__(self, FolderName, SaveFolder):
        self.FolderName = FolderName
        self.SaveFolder = SaveFolder
        self.animals = [f for f in os.listdir(self.FolderName) if f not in ['.DS_Store']][1:]

    def find_numcells(self, animalname, basetask, placecellflag):
        if placecellflag:
            pf_params = np.load(os.path.join(self.FolderName, animalname, 'PlaceCells',
                                             '%s_placecell_data.npz' % animalname), allow_pickle=True)
            pf_number = np.asarray(pf_params['numPFs_incells_revised'].item()[basetask])
            singlepfs = np.where(pf_number == 1)[0]
            print('Number of pfs %d Number of single pfs %d' % (np.size(pf_number), np.size(singlepfs)))
            numcells = list(np.asarray(pf_params['sig_PFs_cellnum_revised'].item()[basetask])[singlepfs])

        else:
            PlaceFieldData = [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
                              (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
            pf_file = [f for f in PlaceFieldData if basetask in f][0]
            numcells = np.size(
                scipy.io.loadmat(os.path.join(self.FolderName, animalname, 'Behavior', pf_file))['Allbinned_F'])
            numcells = np.arange(numcells)
        return numcells

    def get_min_placecells(self, animalname, basetask):
        pf_params = np.load(os.path.join(self.FolderName, animalname, 'PlaceCells',
                                         '%s_placecell_data.npz' % animalname), allow_pickle=True)
        placecells_thistask = pf_params['sig_PFs_cellnum_revised'].item()[basetask]
        mincells = np.min((len(placecells_thistask), len(pf_params['sig_PFs_cellnum_revised'].item()['Task3b'])))

        if mincells != len(placecells_thistask):
            print('Picking %d random cells' % mincells)
            random_placecells = np.random.choice(placecells_thistask, mincells, replace=False)
            return random_placecells
        else:
            return placecells_thistask

    def get_adjacency_matrix(self, tasktocompare, basetask='Task1', corr_thresh=0.1, placecellflag=True, shuffle_flag=False):
        for a in self.animals:
            print(a)
            PlaceFieldData = [f for f in os.listdir(os.path.join(self.FolderName, a, 'Behavior')) if
                              (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
            # Get correlation with task
            pf_file = []
            for t in [tasktocompare, basetask]:
                pf_file.append([f for f in PlaceFieldData if t in f][0])
            pf_data = [scipy.io.loadmat(os.path.join(self.FolderName, a, 'Behavior', f)) for f in pf_file]

            numcells = self.find_numcells(a, basetask=basetask, placecellflag=placecellflag)
            numlaps = int(np.size(pf_data[1]['Allbinned_F'][0, 0], 1) / 2)
            print(numlaps)
            # if self.placecellflag:
            #     numcells = self.get_min_placecells(a, basetask)

            print('Analysing..%d cells' % np.size(numcells))
            corr_matrix = np.zeros((np.size(numcells), np.size(numcells)))
            for n1, cell1 in enumerate(numcells):
                if basetask == tasktocompare:
                    # print('Tasks are similar')
                    # print(np.shape(pf_data[0]['sig_PFs'][0][cell1]))
                    task1 = np.nan_to_num(pf_data[0]['sig_PFs'][0][cell1][:, 5:numlaps])
                else:
                    task1 = np.nan_to_num(pf_data[0]['Allbinned_F'][0, cell1])

                if shuffle_flag:
                    # plt.plot(np.mean(task1, 1))
                    task1 = task1[np.random.permutation(task1.shape[0]), :]
                    # plt.plot(np.mean(task1, 1))
                    # plt.show()
                    # np.random.shuffle(task1)

                for n2, cell2 in enumerate(numcells):
                    if basetask == tasktocompare:
                        task2 = np.nan_to_num(pf_data[1]['sig_PFs'][0][cell2][:, numlaps:])
                    else:
                        task2 = np.nan_to_num(pf_data[1]['sig_PFs'][0][cell2])

                    # if shuffle_flag:
                        # np.random.shuffle(task2)
                    c, p = scipy.stats.pearsonr(np.nanmean(task1, 1), np.nanmean(task2, 1))

                    if ~np.isnan(c) and p < 0.05:
                        data_bw = task1 > 0
                        # print(np.shape(data_bw))
                        if basetask == tasktocompare:
                            numlaps_withfiring = 1
                        else:
                            numlaps_withfiring = np.size(np.where(np.max(data_bw, 0))) / np.size(task1, 1)
                        corr_matrix[n2, n1] = c * numlaps_withfiring
                        # plt.imshow(task1.T, aspect='auto')
                        # plt.title('Correlation %0.2f, Corrected %0.2f, Numlaps %d' % (c, c * numlaps_withfiring, np.size(np.where(np.max(data_bw, 0)))))
                        # plt.show()
            # return corr_matrix
            np.save(os.path.join(self.SaveFolder, '%s_%s' % (basetask, tasktocompare), '%s_%s_with_%s_AdjMatrix.npy' % (a, basetask, tasktocompare)), corr_matrix)
            self.define_edges(a, corr_matrix, corr_thresh, task1=basetask, task2=tasktocompare)
            self.define_nodes(a, corr_matrix, numcells, corr_thresh=corr_thresh, task1=basetask, task2=tasktocompare)

    def define_nodes(self, animalname, adj_matrix, numcells, corr_thresh, task1, task2):
        com_array = np.zeros((np.size(numcells), 6))

        csv_file = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'PlaceCells')) if f.endswith('.csv')][0]
        print(csv_file)
        
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

            if len(com) > 1:
                com_array[n, 1] = com[0]
            else:
                com_array[n, 1] = com

        bin_com = np.digitize(com_array[:, 1], bins=np.arange(0, 45, 5))
        print(np.unique(bin_com))
        com_array[:, 2] = bin_com

        print('Nodes', com_array.shape)
        save_fn = os.path.join(self.SaveFolder, '%s_%s' % (task1, task2), '%s_%s_with_%s_Nodes.csv' % (animalname, task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Id,Location,Binnedlocation,AutoCorrelation,Meancorr_withzero,Meancorr_without\n")
            for row in com_array:
                f.write(f"{int(row[0])},{row[1]},{int(row[2])},{row[3]},{row[4]},{row[5]}\n")
        return com_array

    def define_edges(self, animalname, adj_matrix, corr_thresh, task1, task2):
        print('Corr matrix min %0.1f and max %0.1f' % (np.amin(adj_matrix), np.amax(adj_matrix)))
        # output edge list
        # Diagonal indices
        # di = np.diag_indices(adj_matrix.shape[0])
        # adj_matrix[di] = 0
        inds_to_keep = np.where(adj_matrix > corr_thresh)
        print(f"N edges: {len(inds_to_keep[0])}")
        save_fn = os.path.join(self.SaveFolder, '%s_%s' % (task1, task2), '%s_%s_with_%s_Edges.csv' % (animalname, task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Source,Target,Weight,Type\n")
            for x, y in zip(inds_to_keep[0], inds_to_keep[1]):
                f.write(f"{x},{y},{adj_matrix[x,y]:.4f},Undirected\n")
        
