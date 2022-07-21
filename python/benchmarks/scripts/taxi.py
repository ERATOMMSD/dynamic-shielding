import os
from copy import deepcopy
from glob import glob

from .common import get_scalars

PREFIX = os.getcwd() + '/../taxi/logs/aloha*.group-mmm.org/TaxiFixStart-v3/Apr*'
DYNAMIC_PREFIX = os.getcwd() + '/../taxi/logs/aloha*.group-mmm.org/TaxiFixStart-v3/Apr08*'
#DYNAMIC_PREFIX1 = os.getcwd() + '/../taxi/logs/aloha*.group-mmm.org/TaxiFixStart-v3/Mar13_23*'
#DYNAMIC_PREFIX2 = os.getcwd() + '/../taxi/logs/aloha*.group-mmm.org/TaxiFixStart-v3/Mar14*'

NUM_TIMESTEP: int = 200000

def append_crashes(scalars):
    for key in scalars.keys():
        scalars[key]['crash_episodes'] = deepcopy(scalars[key]['stats/wrong_dropoff_count'])
        for i in range(len(scalars[key]['stats/wrong_dropoff_count'])):
            scalars[key]['crash_episodes'][i] += scalars[key]['stats/wrong_pickup_count'][i]
            scalars[key]['crash_episodes'][i] += scalars[key]['stats/broken_count'][i]
    return scalars


def get_taxi_no_shield_scalars():
    return append_crashes(get_scalars(glob(f'{PREFIX}/TaxiFixStart-v3-Standard-penalty0.0/ppo_*'),
                                      NUM_TIMESTEP=NUM_TIMESTEP))


def get_taxi_safe_padding_scalars():
    return append_crashes(get_scalars(glob(f'{PREFIX}/TaxiSafePaddingWrapper/ppo_*'), NUM_TIMESTEP=NUM_TIMESTEP))


def get_taxi_dynamic_shield_scalars():
    return {**append_crashes(get_scalars(glob(f'{DYNAMIC_PREFIX}/TaxiAdaptiveDynamicPreemptiveShieldWrapper/ppo_*'),
                                         NUM_TIMESTEP=NUM_TIMESTEP))}
#            **append_crashes(get_scalars(glob(f'{DYNAMIC_PREFIX2}/TaxiAdaptiveDynamicPreemptiveShieldWrapper/ppo_*')))}
