"""
This is a script to run the experiment generating samples

Usage: python generate_samples.py [OPTIONS]
"""
import os.path as osp
import sys
from enum import Flag, auto
from typing import Dict, Tuple

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)

from benchmarks.grid_world.scripts.grid_world_runner import Algorithms, DynamicShieldGridWorldRunner, ShieldKind


class Benchmarks(Flag):
    GridWorld2 = auto()
    GridWorld3 = auto()


class Shields(Flag):
    StaticShield = auto()
    DynamicShield = auto()
    NoShield = auto()
    HugeNegativeReward = auto()
    DynamicShieldWithHugeNegativeReward = auto()


# Constants
NUM_TIMESTEPS: Dict[Benchmarks, int] = {Benchmarks.GridWorld2: int(1e5),
                                        Benchmarks.GridWorld3: int(5e5)}
LEARNING_RATES: Dict[Tuple[Benchmarks, Algorithms], float] = {(Benchmarks.GridWorld2, Algorithms.PPO): 3e-4,
                                                              (Benchmarks.GridWorld3, Algorithms.PPO): 3e-4,
                                                              (Benchmarks.GridWorld2, Algorithms.DQN): 0.00001,
                                                              (Benchmarks.GridWorld3, Algorithms.DQN): 0.00001}
GAMMAS: Dict[Benchmarks, float] = {Benchmarks.GridWorld2: 0.99,
                                   Benchmarks.GridWorld3: 0.99}
TASK_STRINGS: Dict[Benchmarks, str] = {Benchmarks.GridWorld2: 'grid_world2-v1',
                                       Benchmarks.GridWorld3: 'grid_world3-v2'}
MIN_DEPTH: Dict[Benchmarks, int] = {Benchmarks.GridWorld2: 5,
                                    Benchmarks.GridWorld3: 5}

if __name__ == "__main__":
    benchmark = Benchmarks.GridWorld2
    task_string = TASK_STRINGS[benchmark]
    algorithm = Algorithms.PPO
    shield_string = 'dynamic'
    runner = DynamicShieldGridWorldRunner(task_string, algorithm, ShieldKind.Preemptive, MIN_DEPTH[benchmark],
                                          skip_mealy_size=0, save_pickle_filename='training_data.pickle',
                                          max_shield_life=100, concurrent_reconstruction=True)
    runner.train(NUM_TIMESTEPS[benchmark], num=0, lr=LEARNING_RATES[benchmark, algorithm], gamma=GAMMAS[benchmark],
                 tb_log_name=f'{task_string}_{shield_string}_shield_{str(algorithm)}')
