import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats

sns.set_context('paper', font_scale=1.3)
import pandas as pd
import warnings


class LickStop(object):
    def __init__(self, ParentDataFolder, norewardtask='Task3'):
        self.ParentDataFolder = ParentDataFolder
        self.SaveFolder = os.path.normpath(self.ParentDataFolder + os.sep + os.pardir)
        self.norewardtask = norewardtask
        self.animalfiles = [f for f in os.listdir(self.ParentDataFolder) if f not in ['.DS_Store', 'LickData']]

        self.lickstopdf = pd.DataFrame(index=[self.animalfiles],
                                       columns=['First0', 'First2', 'First4', 'First6'])
        self.lickstopdf_corrected = pd.DataFrame(index=[self.animalfiles],
                                                 columns=['First0', 'First2', 'First4', 'First6'])
        self.get_behavior_data()
        self.save_lick_dataframes()

    def get_behavior_data(self):
        for i in self.animalfiles:
            behaviorfiles = np.load(os.path.join(self.ParentDataFolder, i, 'SaveAnalysed', 'behavior_data.npz'),
                                    allow_pickle=True)
            self.get_different_lickstops(i, behaviorfiles)

    def get_different_lickstops(self, animalname, behaviorfiles):
        # Get different lick stops based on different criteria
        # 1. Get original lick stop with first 0
        licks_within_rewardzone = behaviorfiles['numlicks_withinreward_alllicks'].item()[self.norewardtask]
        numlaps = behaviorfiles['numlaps'].item()[self.norewardtask]
        lick_stop = np.where(licks_within_rewardzone == 0)[0][0]
        self.lickstopdf.loc[animalname, 'First0'] = lick_stop
        # 2. First consecutive zeros in lickk
        self.get_consecutivelickstop(animalname, 2, licks_within_rewardzone, lick_stop, numlaps)
        self.get_consecutivelickstop(animalname, 4, licks_within_rewardzone, lick_stop, numlaps)
        self.get_consecutivelickstop(animalname, 6, licks_within_rewardzone, lick_stop, numlaps)

    def get_consecutivelickstop(self, animalname, consecutivenolick, licks_within_reward, lick_stop, numlaps,
                                correct_flag=0):
        if not np.all(licks_within_reward[lick_stop:lick_stop + consecutivenolick] == 0):
            for i in np.arange(lick_stop, numlaps):
                if np.all(licks_within_reward[i:i + consecutivenolick] == 0):
                    if correct_flag:
                        self.lickstopdf_corrected.loc[animalname, f'First%d' % consecutivenolick] = i
                    else:
                        self.lickstopdf.loc[animalname, f'First%d' % consecutivenolick] = i
                    break
        else:
            if correct_flag:
                self.lickstopdf_corrected.loc[animalname, f'First%d' % consecutivenolick] = lick_stop
            else:
                self.lickstopdf.loc[animalname, f'First%d' % consecutivenolick] = lick_stop

    def get_different_licks_corrected(self, error_margin):
        for i in self.animalfiles:
            behaviorfiles = np.load(os.path.join(self.ParentDataFolder, i, 'SaveAnalysed', 'behavior_data.npz'),
                                    allow_pickle=True)
            licks_noreward_new = behaviorfiles['numlicks_withinreward_alllicks'].item()[self.norewardtask]

            licks_noreward_new[licks_noreward_new < error_margin] = 0

            numlaps = behaviorfiles['numlaps'].item()[self.norewardtask]
            lick_stop = np.where(licks_noreward_new == 0)[0][0]
            self.lickstopdf_corrected.loc[i, 'First0'] = lick_stop
            # 2. First consecutive zeros in lickk

            self.get_consecutivelickstop(i, 2, licks_noreward_new, lick_stop, numlaps, correct_flag=1)
            self.get_consecutivelickstop(i, 4, licks_noreward_new, lick_stop, numlaps, correct_flag=1)
            self.get_consecutivelickstop(i, 6, licks_noreward_new, lick_stop, numlaps, correct_flag=1)

    def save_lick_dataframes(self):
        SaveFolder = os.path.join(self.SaveFolder, 'LickData')
        self.lickstopdf.to_csv(os.path.join(SaveFolder, 'Lickstops.csv'))
        self.lickstopdf_corrected.to_csv(os.path.join(SaveFolder, 'NormalizedLickstops.csv'))
