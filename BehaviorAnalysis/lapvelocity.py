import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io


class GetData(object):
    def __init__(self, ExpFolderName, TaskDict):
        self.ExpFolderName = ExpFolderName
        self.ExpAnimals = [f for f in os.listdir(self.ExpFolderName) if f not in ['.DS_Store']][1:]
        self.TaskDict = TaskDict
        self.laptime = self.load_behdata()

    def create_data_dict(self):
        data_dict = {keys: [] for keys in self.TaskDict}
        return data_dict
    
    def load_behdata(self):
        laptime = np.zeros((3, len(self.ExpAnimals)))
        n=0
        for i in self.ExpAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'), allow_pickle=True)
            for t in self.TaskDict:
                if t in ['Task1', 'Task3']:
                    laptime[0, n] = np.mean(data['goodlaps_laptime'].item()[t][-5:])
                else: 
                    laptime[1, n] = np.mean(data['goodlaps_laptime'].item()[t][0:2])
                    laptime[2, n] = np.mean(data['goodlaps_laptime'].item()[t][3:10])
            n+=1
        return laptime