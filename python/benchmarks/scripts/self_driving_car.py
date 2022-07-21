import os
from operator import itemgetter

import pandas as pd

PREFIX = os.getcwd() + '/../self_driving_car_sameloc/logs'


def trim_indices(step_list):
    trim_interval = 2048
    count = 1
    result = []
    for i in range(len(step_list)):
        if step_list[i] >= count * trim_interval:
            count += 1
            result.append(i - 1)
    result.append(len(step_list) - 1)
    return result


def trim(scalar, indices):
    return list(itemgetter(*indices)(scalar))


def get_self_driving_car_scalars(prefix, number: int):
    train_csv = pd.read_csv(f'{prefix}/TRAIN/{number}.csv', header=1)
    evaluation_csv = pd.read_csv(f'{prefix}/TEST/{number}.csv', header=1)
    indices = trim_indices(train_csv[' #all_steps'])

    train_time = list(pd.to_datetime(train_csv[' total_duration']).map(lambda its: its.timestamp()))

    return {'step': trim(list(train_csv[' #all_steps']), indices),
            'stats/episodes': trim(list(train_csv['episode']), indices),
            'crash_episodes': trim(list(train_csv[' #of_accidents']), indices),
            'eval/mean_reward': evaluation_csv[' score'],
            'eval/success_rate': list(evaluation_csv[' #steps'].map(lambda x: float(x == 45))),
            'eval/safe_rate': list(evaluation_csv[' #steps'].map(lambda x: float(x == 45))),
            'eval/training_episodes': evaluation_csv['episode'],
            'eval/training_crash_episodes': evaluation_csv[' #of_accidents'],
            'eval/wall_time': list(pd.to_datetime(evaluation_csv[' total_duration'])
                                   .map(lambda ts: ts.timestamp() - train_time[0])),
            'wall_time': train_time}


def get_self_driving_car_no_shield_scalars():
    return {i: get_self_driving_car_scalars(f'{PREFIX}/NO_SHIELD/', i) for i in range(30)}


def get_self_driving_car_safe_padding_scalars():
    return {i: get_self_driving_car_scalars(f'{PREFIX}/PADDING/', i) for i in range(30)}


def get_self_driving_car_dynamic_shield_scalars():
    return {i: get_self_driving_car_scalars(f'{PREFIX}/SHIELD_60/', i) for i in range(30)}
