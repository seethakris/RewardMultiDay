import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io


class GetData(object):
    def __init__(self, FolderName, TaskDict):
        self.FolderName = FolderName
        self.TaskDict = TaskDict
        self.BehFileName = [f for f in os.listdir(os.path.join(self.FolderName, 'Behavior')) if
                                    f.endswith('.mat') and 'PlaceFields' not in f and 'plain1' not in f and 'Lick' not in f]
        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]

        self.running_data = self.create_data_dict()
        self.reward_data = self.create_data_dict()
        self.lick_data = self.create_data_dict()
        self.load_behdata()

    def create_data_dict(self):
        data_dict = {keys: [] for keys in self.TaskDict}
        return data_dict
    
    def load_behdata(self):
        for i in self.BehFileName:
            print(i)
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            
            self.reward_data[taskname] = x['session'].item()[0][0][0][1]
            self.running_data[taskname] = x['session'].item()[0][0][0][0]
            self.lick_data[taskname] = x['session'].item()[0][0][0][3]

    def plot_laps_pertask(self, ax, taskstoplot, axistoplot, numlaps = 5):
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            if taskname in taskstoplot:
                x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
                lapframes = x['bad_E'].T
                if taskname == 'Task1':
                    print('Bla')
                    framestart = np.where(lapframes==15)[0][0]
                    frameend = np.where(lapframes==15+numlaps+1)[0][0]
                else:
                    framestart = np.where(lapframes==1)[0][0]
                    frameend = np.where(lapframes==numlaps+1)[0][0]
                print(framestart, frameend)
                ax[axistoplot[taskname]].plot(self.running_data[taskname][framestart:frameend], linewidth=2)
                ax[axistoplot[taskname]].plot(self.lick_data[taskname][framestart:frameend]/5, alpha=0.5)
            
    def plot_spectask_lap(self, ax, taskstoplot, laptoplot):
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            if taskname == taskstoplot:
                x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
                lapframes = x['bad_E'].T
                framestart = np.where(lapframes==laptoplot)[0][0]
                frameend = np.where(lapframes==laptoplot)[0][-1]
                ax.plot(self.running_data[taskname][framestart:frameend], linewidth=2)
                ax.plot(self.lick_data[taskname][framestart:frameend]/5, alpha=0.5)

    def plot_all(self, ax, taskstoplot, axistoplot):
        for t in taskstoplot:
            ax[axistoplot[t]].plot(self.running_data[t], linewidth=2)
            ax[axistoplot[t]].plot(self.lick_data[t]/5, alpha=0.5)

