import os.path as osp
import time
from enum import Flag, auto
from typing import Optional, List

import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

from benchmarks.common.generic import launch_gateway
from benchmarks.common.train import EvalCallback
from benchmarks.grid_world.deterministic_grid_world_dynamic_shield import GridWorldDynamicShield
from benchmarks.grid_world.grid_world_crash_logger import GridWorldCrashLogger
from benchmarks.grid_world.grid_world_specifications import safe, safe_with_no_crash_duration, safe2, no_wall
from benchmarks.grid_world.grid_world_wrappers import GridWorldPostDynamicShieldWrapper, GridWorldPreDynamicShieldWrapper, \
    GridWorldSafePaddingWrapper
from benchmarks.grid_world.scripts.helper import configure_logger, FACTOR
from src.shields import UpdateShield
from src.shields.adaptive_dynamic_shield import AdaptiveDynamicShield
from src.shields.evaluation_shield import EvaluationShield
from src.wrappers.crash_logger import CrashLoggingCallback
from src.wrappers.safe_padding_callbacks import SafePaddingCallback
from src.wrappers.shield_callbacks import ShieldingCallback, SaveBestShieldCallback
from src.wrappers.shield_policy import shield_policy
from src.wrappers.shield_wrappers import PostposedShieldWrapper, PreemptiveShieldWrapper

LOG_ROOT = osp.join(osp.dirname(__file__), '../logs/')


class Algorithms(Flag):
    PPO = auto()
    DQN = auto()


class ShieldKind(Flag):
    Preemptive = auto()
    Postposed = auto()


class GridWorldRunner:
    def __init__(self, task_string: str, algorithm: Algorithms):
        self.algorithm: Algorithms = algorithm
        self.experiment_string: str = f'{self._get_algorithm_string()}_no_shield'
        self.env = Monitor(GridWorldCrashLogger(gym.make(f'{task_string}')))
        self.task_string: str = task_string
        self.preemptive_shielding = False
        self.log_dir = None
        self.eval_env = None

    def _get_algorithm_string(self) -> str:
        if self.algorithm == Algorithms.PPO:
            return 'PPO'
        else:
            return 'DQN'

    def _make_model(self, lr: float, gamma: float, log_dir) -> BaseAlgorithm:
        if self.algorithm == Algorithms.PPO:
            return PPO('MlpPolicy', self.env, verbose=1, gamma=gamma, learning_rate=lr, tensorboard_log=log_dir)
        else:
            return DQN('MlpPolicy', self.env, verbose=1, gamma=gamma, learning_rate=lr, tensorboard_log=log_dir)

    def _crashes(self) -> List:
        return [CrashLoggingCallback()]

    def train(self, num_timesteps: int, num: int, lr: float, gamma: float,
              tb_log_name='tensorboard.log') -> BaseAlgorithm:
        self.log_dir = osp.join(LOG_ROOT,
                                f'{self.task_string}_{self.experiment_string}-{num_timesteps}-{num}-{time.time()}/')
        configure_logger(self.log_dir)
        print(f'Training {self._get_algorithm_string()}')
        model: BaseAlgorithm = self._make_model(lr, gamma, self.log_dir)
        if self.preemptive_shielding:
            shield_policy(model, self.eval_env)
        if self.eval_env is None:
            self.eval_env = self.env
        eval_callback = EvalCallback(eval_env=self.eval_env, n_eval_episodes=30,
                                     callback_on_new_best=SaveBestShieldCallback(),
                                     best_model_save_path=osp.join(self.log_dir, 'best_model'),
                                     eval_freq=10000, verbose=1,
                                     deterministic=True, render=False)
        model.learn(total_timesteps=num_timesteps, log_interval=1, tb_log_name=tb_log_name,
                    callback=self._crashes() + [eval_callback])
        # Save the trained model
        model.save(osp.join(self.log_dir, 'final_model'))
        return model


class SafePaddingGridWorldRunner(GridWorldRunner):
    def __init__(self, task_string: str, algorithm: Algorithms, shield_kind: ShieldKind):
        super().__init__(task_string, algorithm)
        self.experiment_string = f'{self._get_algorithm_string()}_safe_padding'
        if shield_kind == ShieldKind.Postposed:
            raise RuntimeError("Postposed safe padding is not implemented")
        else:
            self.env = GridWorldSafePaddingWrapper(self.env)
            self.preemptive_shielding = True

    def _crashes(self) -> List:
        return [CrashLoggingCallback(), SafePaddingCallback()]


class DynamicShieldGridWorldRunner(GridWorldRunner):

    def __init__(self, task_string: str, algorithm: Algorithms, shield_kind: ShieldKind, min_depth: int,
                 concurrent_reconstruction: bool = True, max_shield_life: int = 100, skip_mealy_size: int = 5000,
                 load_pickle_filename: Optional[str] = None, save_pickle_filename: Optional[str] = None,
                 save_full_pickle_filename: Optional[str] = None):
        super().__init__(task_string, algorithm)
        self.experiment_string = f'{self._get_algorithm_string()}_dynamic_shielding'
        self.save_pickle_filename = save_pickle_filename
        self.save_full_pickle_filename = save_full_pickle_filename

        # Set up py4j gateway
        self.gateway = launch_gateway()
        if shield_kind == ShieldKind.Postposed:
            self.env = GridWorldPostDynamicShieldWrapper(self.env, self.gateway, min_depth=min_depth,
                                                         concurrent_reconstruction=concurrent_reconstruction,
                                                         max_shield_life=max_shield_life,
                                                         skip_mealy_size=skip_mealy_size,
                                                         pickle_filename=load_pickle_filename)
        else:
            self.env = GridWorldPreDynamicShieldWrapper(self.env, self.gateway, min_depth=min_depth,
                                                        concurrent_reconstruction=concurrent_reconstruction,
                                                        max_shield_life=max_shield_life,
                                                        skip_mealy_size=skip_mealy_size,
                                                        pickle_filename=load_pickle_filename)
            self.preemptive_shielding = True
        eval_shield = EvaluationShield(training_env=self.env)
        self.eval_env = PreemptiveShieldWrapper(env=Monitor(gym.make(f'{task_string}')), shield=eval_shield)

    def _crashes(self):
        return [CrashLoggingCallback(), ShieldingCallback()]

    def train(self, num_timesteps: int, num: int, lr: float, gamma: float,
              tb_log_name='tensorboard.log') -> BaseAlgorithm:
        result = super().train(num_timesteps, num, lr, gamma, tb_log_name)
        if self.save_pickle_filename is not None:
            self.env.shield.save_samples(osp.join(self.log_dir, self.save_pickle_filename))
        if self.save_full_pickle_filename is not None:
            self.env.shield.save_full_samples(osp.join(self.log_dir, self.save_full_pickle_filename))
        self.gateway.shutdown()
        return result


class AdaptiveDynamicShieldGridWorldRunner(DynamicShieldGridWorldRunner):

    def __init__(self, task_string: str, algorithm: Algorithms, shield_kind: ShieldKind,
                 concurrent_reconstruction: bool = True, max_shield_life: int = 100, skip_mealy_size: int = 5000,
                 load_pickle_filename: Optional[str] = None, save_pickle_filename: Optional[str] = None,
                 save_full_pickle_filename: Optional[str] = None):
        super().__init__(task_string, algorithm, shield_kind, concurrent_reconstruction=concurrent_reconstruction,
                         max_shield_life=max_shield_life, skip_mealy_size=skip_mealy_size,
                         load_pickle_filename=load_pickle_filename, save_pickle_filename=save_pickle_filename,
                         save_full_pickle_filename=save_full_pickle_filename, min_depth=1)
        self.experiment_string = f'{self._get_algorithm_string()}_adaptive_dynamic_shielding_factor_1'
        self.env = GridWorldCrashLogger(gym.make(f'{task_string}'))
        specifications: List[str] = [safe, safe_with_no_crash_duration(5), safe_with_no_crash_duration(3),
                                     safe_with_no_crash_duration(1), safe2, no_wall]
        shield = AdaptiveDynamicShield(ltl_formula=specifications, gateway=self.gateway,
                                       alphabet_start=0, alphabet_end=24,
                                       max_episode_length=self.env.unwrapped.MAX_STEP,
                                       alphabet_mapper=GridWorldDynamicShield.alphabet_mapper,
                                       evaluate_output=GridWorldDynamicShield.evaluate_output,
                                       reverse_alphabet_mapper=GridWorldDynamicShield.reverse_alphabet_mapper,
                                       reverse_output_mapper=GridWorldDynamicShield.reverse_output_mapper,
                                       update_shield=UpdateShield.RESET,
                                       skip_mealy_size=skip_mealy_size,
                                       concurrent_reconstruction=concurrent_reconstruction,
                                       max_shield_life=max_shield_life, factor=FACTOR,
                                       not_use_deviating_shield=False)
        if load_pickle_filename is not None:
            shield.load_transition_cover(load_pickle_filename)
        if shield_kind == ShieldKind.Postposed:
            self.env = PostposedShieldWrapper(env=self.env, shield=shield, punish=False)
        else:
            self.env = PreemptiveShieldWrapper(env=self.env, shield=shield)
        eval_shield = EvaluationShield(training_env=self.env)
        self.eval_env = PreemptiveShieldWrapper(env=Monitor(gym.make(f'{task_string}')), shield=eval_shield)
