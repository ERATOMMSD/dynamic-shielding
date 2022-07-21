import os.path as osp
import random
import sys
from typing import List

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.common.evaluate import evaluate_list_repeat, print_results, evaluate_repeat
from benchmarks.grid_world.grid_world_arena import evaluate_output
from benchmarks.grid_world.grid_world_safe_padding import player1_alphabet, player2_alphabet, output_alphabet
from benchmarks.grid_world.grid_world_specifications import safe, safe_with_no_crash_duration, no_wall, safe2
from benchmarks.grid_world.scripts.helper import TASK_STRING

num_repetition = 30


def evaluate_no_shield_list(env, path_list):
    random.seed(0)
    model_list: List[PPO] = list(map(lambda path: PPO.load(osp.join(osp.dirname(__file__), '..',
                                                                    path, 'best_model', 'best_model.zip')), path_list))
    results = evaluate_list_repeat(env, model_list,
                                   num_repetition=num_repetition,
                                   max_episode_steps=env.unwrapped.MAX_STEP)
    print_results(results)


def evaluate_with_shield_list(env, path_list):
    random.seed(0)
    result_list = []
    specifications: List[str] = [safe, safe_with_no_crash_duration(5), safe_with_no_crash_duration(3),
                                 safe_with_no_crash_duration(1), safe2, no_wall]
    for path in path_list:
        best_model_path = osp.join(osp.dirname(__file__), '..', path, 'best_model')
        model: PPO = PPO.load(osp.join(best_model_path, 'best_model.zip'))
        result_list += evaluate_repeat(env, model,
                                       num_repetition=num_repetition,
                                       max_episode_steps=env.unwrapped.MAX_STEP,
                                       path=best_model_path,
                                       ltl_formula=specifications,
                                       player1_alphabet=player1_alphabet,
                                       player2_alphabet=player2_alphabet,
                                       output_alphabet=output_alphabet,
                                       evaluate_output=evaluate_output)
    print_results(result_list)


# configure the training environment
env = Monitor(gym.make(f'{TASK_STRING}'))

print('dynamic shield with adaptive min_depth')
evaluate_with_shield_list(env, [
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638255476.4750552',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256318.7493055',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256331.7297418',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256369.9151819',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256511.505128',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256866.7839973',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256869.1014495',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256919.2560768',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256921.8488636',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256934.4217854',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256953.0002127',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638256955.2476757',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638257009.9418504',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638257059.9476898',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638257186.9843526',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638257205.5105467',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638259101.0656395',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638259499.1651115',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638259718.5128624',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638260711.7525818',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638260728.9203649',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638260749.7422748',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638260812.7860966',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638261370.817823',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638261488.0523424',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638261659.1993673',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638261783.2796018',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638261844.614889',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638262016.362468',
    'logs/grid_world2-v1_PPO_adaptive_dynamic_shielding_factor_1-100000-0-1638262040.1023064',
])

print('safe padding')
evaluate_with_shield_list(env, [
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508465.861675',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508467.7524276',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508468.4341578',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508468.8310163',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508470.069747',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508470.343056',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508471.613135',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508473.024384',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508475.017282',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508475.1464653',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508513.25009',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508514.006192',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508515.0176315',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508516.686989',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508523.5172946',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637508523.9259338',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637510702.2727308',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637514695.2198699',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637516125.2691326',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637516134.1316857',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637517428.1169462',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637517790.3112366',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637518111.4447196',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637518484.2414436',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637518811.3597248',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637519169.4927478',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637519441.729724',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637520879.324246',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637520899.9818616',
    'logs/grid_world2-v1_PPO_safe_padding-100000-0-1637521084.319661',
])

print('no shield')
evaluate_no_shield_list(env, [
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508283.329474',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508285.0868664',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508285.2190287',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508285.3399954',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508285.9554791',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508285.978334',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508286.0658362',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508286.2818723',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508287.3217905',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508287.6245158',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508331.7188447',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508331.8536627',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508331.877794',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508332.3473008',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508333.5678983',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637508333.6983688',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637510520.3543284',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637514507.1614807',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637515934.6426814',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637515946.1295943',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637517244.199243',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637517602.488713',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637517928.1202042',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637518297.2867255',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637518632.7924762',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637518978.2882562',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637519254.8387592',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637520692.3194363',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637520718.3673928',
    'logs/grid_world2-v1_PPO_no_shield-100000-0-1637520893.563242',
])
