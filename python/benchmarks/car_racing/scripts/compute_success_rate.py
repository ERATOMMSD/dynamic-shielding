import os.path as osp
import sys
from typing import List

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.common.evaluate import evaluate_list_repeat, print_results, evaluate_repeat
from benchmarks.car_racing.scripts.helper import TASK_STRING
from benchmarks.car_racing.car_racing_safe_padding import player1_alphabet, player2_alphabet, output_alphabet
from benchmarks.car_racing.car_racing_specifications import no_consecutive_grass
from benchmarks.car_racing.discrete_car_racing import evaluate_output

num_repetition = 1


def evaluate_no_shield_list(env, path_list):
    model_list: List[PPO] = list(map(lambda path: PPO.load(osp.join(osp.dirname(__file__), '..',
                                                                    path, 'best_model', 'best_model.zip')), path_list))
    results = evaluate_list_repeat(env, model_list,
                                   num_repetition=num_repetition,
                                   max_episode_steps=env.spec.max_episode_steps)
    print_results(results)


def evaluate_with_shield_list(env, path_list):
    result_list = []
    for path in path_list:
        best_model_path = osp.join(osp.dirname(__file__), '..', path, 'best_model')
        model: PPO = PPO.load(osp.join(best_model_path, 'best_model.zip'))
        result_list += evaluate_repeat(env, model,
                                       num_repetition=num_repetition,
                                       max_episode_steps=env.spec.max_episode_steps,
                                       path=best_model_path,
                                       ltl_formula=[no_consecutive_grass],
                                       player1_alphabet=player1_alphabet,
                                       player2_alphabet=player2_alphabet,
                                       output_alphabet=output_alphabet,
                                       evaluate_output=evaluate_output)
    print_results(result_list)


# configure the training environment
env = Monitor(gym.make(f'{TASK_STRING}'))
env.reset()

print('dynamic shield with adaptive min_depth')
 # evaluate_no_shield_list(env, [
evaluate_with_shield_list(env, [
    'logs/aloha/discrete_car_racing-v4/Dec02_03:20:11/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_03:03:12/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_02:57:39/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_02:52:26/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_02:45:23/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_02:45:04/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_02:22:19/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_02:11:21/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_02:05:36/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec02_01:53:22/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:58:30/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:58:24/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:53:23/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:50:29/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:45:06/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:44:13/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:43:49/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:33:20/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:21:29/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_23:16:16/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:56:14/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:55:30/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:48:20/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:47:37/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:41:01/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:40:32/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:37:39/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:37:33/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:34:48/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
    'logs/aloha/discrete_car_racing-v4/Dec01_20:33:43/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield',
])

print('safe padding')
evaluate_with_shield_list(env, [
    'logs/aloha/discrete_car_racing-v4/Nov22_19:23:48/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_19:18:47/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_19:08:16/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_19:04:17/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_18:53:16/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_17:55:14/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_17:55:13/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_17:45:12/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_17:29:11/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_16:59:42/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_16:23:40/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_16:07:09/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_16:00:38/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_15:51:36/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_15:44:05/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_15:38:06/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_15:30:33/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_14:33:31/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_13:09:30/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_13:05:59/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:59:58/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:49:27/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:37:26/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:33:55/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:28:52/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:19:51/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:15:50/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_12:05:49/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov22_11:14:49/discrete_car_racing-v4_safe_padding',
    'logs/aloha/discrete_car_racing-v4/Nov08_10:48:26/discrete_car_racing-v4_safe_padding',
])

print('no shield')
evaluate_no_shield_list(env, [
    'logs/aloha/discrete_car_racing-v4/Nov08_09:12:27/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_08:59:34/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_08:28:18/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_08:14:28/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_08:08:45/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_07:38:20/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_07:21:56/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_06:47:45/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_06:41:33/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_04:22:31/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_03:48:16/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_03:24:12/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_03:23:44/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov08_02:38:20/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov07_21:50:11/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Nov07_21:50:09/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_05:04:28/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_05:00:17/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_05:00:04/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_04:58:21/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_04:43:56/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_04:37:55/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_04:21:54/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_03:28:22/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_03:24:00/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_03:18:34/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_03:18:23/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_03:17:06/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_03:01:39/discrete_car_racing-v4_no_shield',
    'logs/aloha/discrete_car_racing-v4/Oct16_02:56:03/discrete_car_racing-v4_no_shield', ])
