import os
from copy import deepcopy
from glob import glob

from .common import get_scalars

PREFIX = os.getcwd() + '/../water_tank/logs/aloha*.group-mmm.org/WaterTank-c100-i50-v0/Apr*'


def append_crashes(scalars):
    for key in scalars.keys():
        scalars[key]['crash_episodes'] = deepcopy(scalars[key]['stats/episodes'])
        for i in range(len(scalars[key]['stats/episodes'])):
            scalars[key]['crash_episodes'][i] -= scalars[key]['stats/successful_episodes'][i]
    return scalars


def get_water_tank_no_shield_scalars():
    return append_crashes(get_scalars(glob(f'{PREFIX}/WaterTank-c100-i50-v0-Standard-penalty1.0/ppo_*')))


def get_water_tank_safe_padding_scalars():
    return append_crashes(get_scalars(glob(f'{PREFIX}/WaterTankSafePaddingWrapper/ppo_*')))


def get_water_tank_dynamic_shield_scalars():
    return append_crashes(get_scalars(glob(f'{PREFIX}/WaterTankAdaptiveDynamicPreemptiveShieldWrapper/ppo_*')))
