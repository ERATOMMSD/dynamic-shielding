import os
from glob import glob

from .common import get_scalars

PREFIX = os.getcwd() + '/../car_racing/logs/aloha*.group-mmm.org/discrete_car_racing-v4/Apr*'
PREFIX2 = os.getcwd() + '/../car_racing/logs/aloha*.group-mmm.org/discrete_car_racing-v4/Mar*'

NO_SHIELD_PATH = [
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_16:23:08/discrete_car_racing-v4_no_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_15:20:44/discrete_car_racing-v4_no_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_15:16:45/discrete_car_racing-v4_no_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_14:25:09/discrete_car_racing-v4_no_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_14:07:45/discrete_car_racing-v4_no_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_14:01:37/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_13:31:06/discrete_car_racing-v4_no_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_13:15:00/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_13:10:09/discrete_car_racing-v4_no_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_12:55:02/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_12:54:48/discrete_car_racing-v4_no_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_11:24:04/discrete_car_racing-v4_no_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_11:03:09/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_09:18:58/discrete_car_racing-v4_no_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_08:52:06/discrete_car_racing-v4_no_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_07:05:56/discrete_car_racing-v4_no_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_06:37:39/discrete_car_racing-v4_no_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_06:28:43/discrete_car_racing-v4_no_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_05:20:32/discrete_car_racing-v4_no_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_04:47:41/discrete_car_racing-v4_no_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_04:16:11/discrete_car_racing-v4_no_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_04:15:01/discrete_car_racing-v4_no_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_03:49:59/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_03:29:42/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_03:16:40/discrete_car_racing-v4_no_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_03:09:33/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_02:34:14/discrete_car_racing-v4_no_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_02:27:07/discrete_car_racing-v4_no_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_02:24:06/discrete_car_racing-v4_no_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_01:15:22/discrete_car_racing-v4_no_shield', ]

SAFE_PADDING_PATH = [
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb21_11:51:29/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_18:03:35/discrete_car_racing-v4_safe_padding',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_17:33:27/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_17:01:00/discrete_car_racing-v4_safe_padding',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_16:37:41/discrete_car_racing-v4_safe_padding',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_16:20:03/discrete_car_racing-v4_safe_padding',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_16:05:47/discrete_car_racing-v4_safe_padding',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_15:35:26/discrete_car_racing-v4_safe_padding',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_15:10:04/discrete_car_racing-v4_safe_padding',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_15:08:08/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_14:52:51/discrete_car_racing-v4_safe_padding',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_13:45:34/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_12:42:31/discrete_car_racing-v4_safe_padding',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_11:26:41/discrete_car_racing-v4_safe_padding',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_11:01:24/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_08:44:33/discrete_car_racing-v4_safe_padding',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_08:30:32/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_08:15:39/discrete_car_racing-v4_safe_padding',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_07:06:14/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_07:03:44/discrete_car_racing-v4_safe_padding',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_06:36:20/discrete_car_racing-v4_safe_padding',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_06:11:19/discrete_car_racing-v4_safe_padding',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb18_06:04:23/discrete_car_racing-v4_safe_padding',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_05:41:09/discrete_car_racing-v4_safe_padding',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_05:25:23/discrete_car_racing-v4_safe_padding',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_05:20:38/discrete_car_racing-v4_safe_padding',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_04:49:06/discrete_car_racing-v4_safe_padding',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb18_04:33:36/discrete_car_racing-v4_safe_padding',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb18_04:05:25/discrete_car_racing-v4_safe_padding',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb18_03:29:31/discrete_car_racing-v4_safe_padding', ]

DYNAMIC_SHIELD_PATH = [
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb21_19:59:44/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb21_19:52:37/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb21_19:50:21/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb21_19:47:41/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_19:36:02/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_19:15:15/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_19:06:17/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb21_18:39:40/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb21_16:26:56/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha02.group-mmm.org/discrete_car_racing-v4/Feb21_16:25:06/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb21_16:24:51/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb21_16:22:20/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha01.group-mmm.org/discrete_car_racing-v4/Feb21_16:20:28/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_16:03:51/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_15:43:09/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_15:16:00/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha03.group-mmm.org/discrete_car_racing-v4/Feb21_15:07:25/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_15:00:56/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_11:50:57/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_11:47:10/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_11:32:08/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_11:06:09/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_07:45:37/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_07:45:06/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_07:39:35/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_07:21:01/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_03:38:04/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_03:24:32/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_03:21:00/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha00.group-mmm.org/discrete_car_racing-v4/Feb21_03:14:29/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield', ]

NUM_TIMESTEP: int = 200000


def get_car_racing_no_shield_scalars():
    return {**get_scalars(glob(f'{PREFIX}/discrete_car_racing-v4_no_shield/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP),
            **get_scalars(glob(f'{PREFIX2}/discrete_car_racing-v4_no_shield/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP)}
#    return get_scalars(glob(f'{PREFIX}/discrete_car_racing-v4_no_shield/ppo_*'),
#                       NUM_TIMESTEP=NUM_TIMESTEP)


def get_car_racing_safe_padding_scalars():
    return {**get_scalars(glob(f'{PREFIX}/discrete_car_racing-v4_safe_padding/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP),
            **get_scalars(glob(f'{PREFIX2}/discrete_car_racing-v4_safe_padding/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP)}
#    return get_scalars(glob(f'{PREFIX}/discrete_car_racing-v4_safe_padding/ppo_*'),
#                       NUM_TIMESTEP=NUM_TIMESTEP)


def get_car_racing_dynamic_shield_scalars():
    return {**get_scalars(glob(f'{PREFIX}/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP),
            **get_scalars(glob(f'{PREFIX2}/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP)}
#    return get_scalars(glob(f'{PREFIX}/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield/ppo_*'),
#                       NUM_TIMESTEP=NUM_TIMESTEP)
