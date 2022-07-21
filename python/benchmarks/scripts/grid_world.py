import os
from glob import glob

from .common import get_scalars

NO_SHIELD_PATH = [
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634921.372014',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634922.0199647',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634922.9261622',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634923.0889668',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634923.454476',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634923.4622524',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634923.476143',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634923.4789686',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634924.4504151',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634924.4824588',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634924.4838274',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634924.4989483',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634943.5145602',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634943.5412498',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634943.5696783',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648634943.5816796',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648636427.3804193',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648640176.9126742',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648640461.108265',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648642766.9364576',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648646805.0605266',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648649028.0788734',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648650941.6566231',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648653033.2336514',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648653493.6333885',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648654915.7225983',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648655205.1231387',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648655305.4786603',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648658909.4673681',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1648660091.2449875', ]

SAFE_PADDING_PATH = [
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635097.8298864',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635098.915749',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635099.1640449',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635099.9276319',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635100.2735868',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635100.8680737',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635100.9312773',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635101.9448168',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635102.7107444',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635103.1446056',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635103.9812634',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635103.987568',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635130.1432416',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635130.6091373',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635131.6366072',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648635134.6676538',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648636606.3734732',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648640401.6618352',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648640650.7088656',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648642955.6600983',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648646989.1575434',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648649217.2116945',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648651125.4370763',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648653214.975491',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648653679.7746723',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648655097.9534252',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648655409.8366592',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648655480.9197927',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648659101.0655978',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1648660318.8409357', ]

DYNAMIC_SHIELD_PATH = [
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649017886.5196564',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649018019.6740541',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649018952.1504288',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649018953.0353355',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649019199.7185156',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649019361.6863458',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649019363.443366',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649019933.994355',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649020903.7398813',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649021246.9818609',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649021739.8255048',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649021887.9585922',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649022146.0924504',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649022282.0310526',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649022841.866495',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649022870.9336085',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649023054.6601675',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649023382.984525',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649023551.3209786',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649023822.1712675',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649024014.7026122',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649024015.9430652',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649024524.209279',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649025468.2073486',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649025915.9243975',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649026622.3532813',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649027073.87354',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649027307.8053248',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649027493.9999387',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1649027561.871166', ]

NUM_TIMESTEP: int = 100 * 1000
MAX_SHIELD_LIFE: float = 200.0
LEARNING_RATE: float = 3e-4


def get_grid_world_no_shield_scalars():
    return get_scalars(
        sum(map(lambda path: glob(os.getcwd() + '/../grid_world/' + path + '/*.PPO_*'), NO_SHIELD_PATH), []),
        NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE)


def get_grid_world_safe_padding_scalars():
    return get_scalars(
        sum(map(lambda path: glob(os.getcwd() + '/../grid_world/' + path + '/*.PPO_*'), SAFE_PADDING_PATH), []),
        NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE)


def get_grid_world_dynamic_shield_scalars():
    return get_scalars(
        sum(map(lambda path: glob(os.getcwd() + '/../grid_world/' + path + '/*.PPO_*'), DYNAMIC_SHIELD_PATH), []),
        NUM_TIMESTEP=NUM_TIMESTEP, MAX_SHIELD_LIFE=MAX_SHIELD_LIFE, LEARNING_RATE=LEARNING_RATE)
