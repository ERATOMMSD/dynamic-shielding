import os
from glob import glob

from .common import get_scalars

PREFIX = os.getcwd() + '/../cliffwalking/logs/aloha*.group-mmm.org/CliffWalkingExt-v0/Apr*'


def append_success_rate(scalars):
    for key in scalars.keys():
        for mean_ep_length in scalars[key]['mean_ep_length']:
            scalars[key]['eval/success_rate'].append(mean_ep_length)
            scalars[key]['eval/success_rate'][-1].value = scalars[key]['eval/success_rate'][-1] < 100
    return scalars


def get_cliffwalking_no_shield_scalars():
    return get_scalars(glob(f'{PREFIX}/CliffWalkingExt-v0-Standard-penalty0.0/ppo_*'))


def get_cliffwalking_safe_padding_scalars():
    return get_scalars(glob(f'{PREFIX}/CliffWalkingSafePaddingWrapper/ppo_*'))


def get_cliffwalking_dynamic_shield_scalars():
    return get_scalars(glob(f'{PREFIX}/CliffWalkingAdaptiveDynamicPreemptiveShieldWrapper/ppo_*'))
