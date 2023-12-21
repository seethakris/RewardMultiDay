import os
import sys
from collections import OrderedDict

def MultiDaysAnimals(animalname):
    CommonFolder = '/Users/seetha/Box Sync/MultiDayData/Data/'
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                'Task2': '2 Fam Rew',
                                'Task3': '3 No Rew',
                                'Task4': '4 No Rew',
                                'Task5': '5 Fam Rew'}

    if animalname == 'CFC16':
        Detailsdict['foldername'] = os.path.join(CommonFolder, animalname)
        Detailsdict['v73_flag'] = 1 # If matfile was saved as v7.3
        Detailsdict['task_numframes'] = {'Task1': 18000,
                                         'Task2': 15000,
                                         'Task3': 20000,
                                         'Task4': 20000,
                                         'Task5': 15000}


    if animalname == 'CFC17':
        Detailsdict['foldername'] = os.path.join(CommonFolder, animalname)
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['task_numframes'] = {'Task1': 18000,
                                         'Task2': 15000,
                                         'Task3': 20000,
                                         'Task4': 20000,
                                         'Task5': 15000}

    if animalname == 'CFC18':
        Detailsdict['foldername'] = os.path.join(CommonFolder, animalname)
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['task_numframes'] = {'Task1': 18000,
                                         'Task2': 15000,
                                         'Task3': 20000,
                                         'Task4': 20000,
                                         'Task5': 15000}

    if animalname == 'DG11':
        Detailsdict['foldername'] = os.path.join(CommonFolder, animalname)
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['task_numframes'] = {'Task1': 25000,
                                         'Task2': 15000,
                                         'Task3': 20000,
                                         'Task4': 20000,
                                         'Task5': 15000}

    if animalname == 'NR31':
        Detailsdict['foldername'] = os.path.join(CommonFolder, animalname)
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['task_numframes'] = {'Task1': 20000,
                                         'Task2': 15000,
                                         'Task3': 20000,
                                         'Task4': 15000,
                                         'Task5': 10000}

    if animalname == 'NR32':
        Detailsdict['foldername'] = os.path.join(CommonFolder, animalname)
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['task_numframes'] = {'Task1': 20000,
                                         'Task2': 15000,
                                         'Task3': 25000,
                                         'Task4': 20000,
                                         'Task5': 15000}
    if animalname == 'NR34':
        Detailsdict['foldername'] = os.path.join(CommonFolder, animalname)
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['task_numframes'] = {'Task1': 20000,
                                         'Task2': 15000,
                                         'Task3': 25000,
                                         'Task4': 20000,
                                         'Task5': 25000}

    Detailsdict['NoRewardTasks'] = ['Task3', 'Task4']
    Detailsdict['animal'] = animalname

    return Detailsdict
