import os
from glob import glob

from .common import get_scalars

PREFIX = os.getcwd() + '/../sidewalk/logs/aloha*.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Apr*'
PREFIX_DYNAMIC = os.getcwd() + '/../sidewalk/logs/aloha*.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Apr10*'
NO_SHIELD_PREFIX = os.getcwd() + '/../sidewalk/logs/aloha*.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Apr11*'
PREFIX2 = os.getcwd() + '/../sidewalk/logs/aloha*.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Mar*'

NO_SHIELD_PATH = [
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:56:38/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:46:41/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:44:49/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:35:08/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:22:27/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:09:48/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:45:39/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:45:10/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:37:34/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:33:49/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:19:44/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:12:13/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:54:51/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:46:05/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:21:29/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:01:48/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:54:53/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:41:06/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:28:46/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:27:52/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:19:18/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:13:38/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:11:58/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:11:50/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:08:44/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_22:24:36/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_21:54:30/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_21:50:27/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_20:18:50/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_20:00:18/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_1', ]

SAFE_PADDING_PATH = [
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_03:08:58/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:59:07/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:57:22/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:49:10/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:36:13/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:22:39/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:59:10/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:57:29/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:49:13/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:46:56/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:32:54/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:25:50/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:07:25/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:59:14/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:33:28/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:14:46/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:06:59/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:54:50/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:41:58/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:41:49/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:30:49/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:26:01/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:25:51/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:25:07/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:22:26/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_22:39:36/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_22:08:49/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_22:05:29/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_20:30:58/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_20:14:38/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_1', ]

DYNAMIC_SHIELD_PATH = [
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_14:58:08/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_14:50:05/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_14:35:12/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_14:26:33/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_14:26:32/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_14:26:29/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_14:26:25/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_11:12:37/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_11:12:36/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_11:12:33/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_03:27:26/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:53:19/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:27:31/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:17:32/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_02:02:07/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:38:24/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_01:01:04/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:37:06/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:24:33/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha02.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:11:52/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:11:08/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb22_00:01:08/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha03.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_23:55:37/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_22:40:06/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_22:36:03/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha01.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_20:48:18/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_02:25:35/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_01:06:35/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_00:57:03/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1',
    'logs/aloha00.group-mmm.org/RangedDeterministicSidewalk-DomainRand-v0/Feb21_00:53:03/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_1', ]

NUM_TIMESTEP: int = int(1e5)
MAX_SHIELD_LIFE: float = 75
LEARNING_RATE: float = 0.00005


def get_sidewalk_no_shield_scalars():
    return get_scalars(glob(f'{NO_SHIELD_PREFIX}/RangedDeterministicSidewalk-DomainRand-v0_no_shield/ppo_*'))


def get_sidewalk_safe_padding_scalars():
    return {**get_scalars(glob(f'{PREFIX}/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE),
            **get_scalars(glob(f'{PREFIX2}/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_*'),
                          NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE)}


#    return get_scalars(glob(f'{PREFIX}/RangedDeterministicSidewalk-DomainRand-v0_safe_padding/ppo_*'),
#                       NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE)


def get_sidewalk_dynamic_shield_scalars():
    return {
        **get_scalars(glob(f'{PREFIX_DYNAMIC}/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_*'),
                      NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE)}
#    return get_scalars(glob(f'{PREFIX}/RangedDeterministicSidewalk-DomainRand-v0_adaptive_dynamic_shield/ppo_*'),
#                       NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE)
