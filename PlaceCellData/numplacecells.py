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
    def __init__(self, FolderName, CombinedDataFolder, taskstoplot):
        self.FolderName = FolderName
        self.CombinedDataFolder = CombinedDataFolder
        self.animals = [f for f in os.listdir(self.FolderName) if f not in ['.DS_Store']][1:]
        self.combineddf = self.get_placecell_csv()

        self.celldf = self.get_cells_pertask_peranimal(taskstoplot)
        self.updateddf = self.update_pfparams_withcommoncells(self.celldf, taskstoplot)

    def get_place_cellinfo(self, animal):
        pcdata = np.load(
            os.path.join(self.FolderName, animal, 'PlaceCells', '%s_placecell_data.npz' % animal),
            allow_pickle=True)
        return pcdata


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
        updated_pfparam = self.combineddf.copy()
        updated_pfparam['CommonCells'] = False

        for a in self.animals:
            commoncells = celldf[celldf[taskstoplot].sum(axis=1)==len(taskstoplot)]
            commoncells = commoncells[commoncells['AnimalName']==a]['CellNum']
            updated_pfparam.loc[(updated_pfparam['animalname']==a) & (updated_pfparam['CellNumber'].isin(commoncells)), 'CommonCells'] = True

        return updated_pfparam
    
    def get_placecell_perc_peranimal(self, taskstoplot):
        numcells_dict = {keys: [] for keys in ['AnimalName', 'Task', 'Numplacecells']}
        for a in self.animals:
            pfparams = np.load(
                os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecell_data.npz' % a), allow_pickle=True)
            numPFs = pfparams['sig_PFs_cellnum_revised'].item()
            for t in taskstoplot:
                numcells_dict['Task'].append(t)
                numcells_dict['AnimalName'].append(a)
                numcells_dict['Numplacecells'].append(np.size(numPFs[t]))
        return pd.DataFrame.from_dict(numcells_dict)