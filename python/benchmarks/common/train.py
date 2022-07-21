import os
import socket
from datetime import datetime
from typing import Union, Optional, Dict, Any

import gym
import numpy as np
import stable_baselines3.common.callbacks as callbacks
from py4j.java_gateway import JavaGateway
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv, VecEnv, sync_envs_normalization

from src.shields.evaluation_shield import EvaluationShield
from src.wrappers.shield_callbacks import SaveBestShieldCallback
from src.wrappers.shield_policy import shield_policy
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper


class EvalCallback(callbacks.EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(self,
                 eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[callbacks.BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: Optional[str] = None,
                 best_model_save_path: Optional[str] = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 warn: bool = True,
                 ):
        super(EvalCallback, self).__init__(eval_env, callback_on_new_best, n_eval_episodes, eval_freq, log_path,
                                           best_model_save_path, deterministic, render, verbose, warn)
        # For computing success rate
        self._is_crash_buffer = []

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            maybe_is_crash = info.get("is_crash")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            if maybe_is_crash is not None:
                self._is_crash_buffer.append(maybe_is_crash)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []
            self._is_crash_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)
                self.logger.record("eval/this_episodes", len(self._is_success_buffer))
                self.logger.record("eval/success_episodes", sum(self._is_success_buffer))

            if len(self._is_crash_buffer) > 0:
                self.logger.record("eval/crash_episodes", sum(self._is_crash_buffer))
                crash_rate = np.mean(self._is_crash_buffer)
                self.logger.record("eval/crash_rate", crash_rate)
                self.logger.record("eval/safe_rate", 1.0 - crash_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

def find_available_port() -> int:
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def launch_gateway_on_available_port() -> JavaGateway:
    port = find_available_port()
    return JavaGateway.launch_gateway(
        port=port,
        jarpath="../../../java/target/learnlib-py4j-example-1.0-SNAPSHOT.jar",
        die_on_exit=True,
    )


def run(shield, ltl_formula, depths, game, penalties, total_steps, learning_rate, callback, shield_life):
    if shield:
        # Using shielding wrapper name class as the label
        label = f'{shield.__name__}'
        # quick hack, probably better to use is instance here..
        if 'Adaptive' in label:
            gateway = launch_gateway_on_available_port()
            dlabel = f'{label}'

            env = shield(env=Monitor(gym.make(game)),
                         ltl_formula=ltl_formula,
                         gateway=gateway,
                         max_shield_life=shield_life)

            # Creating second environment to be used in evaluation callback
            eval_shield = EvaluationShield(training_env=env)
            eval_env = PreemptiveShieldWrapper(env=Monitor(gym.make(game)), shield=eval_shield)

            train(env=env,
                  label=dlabel,
                  game=game,
                  total_steps=total_steps,
                  learning_rate=learning_rate,
                  callback=callback,
                  eval_env=eval_env)

            gateway.close()
        elif 'Dynamic' in label:
            for depth in depths:
                gateway = launch_gateway_on_available_port()
                dlabel = f'{label}-depth{depth}'
                env = shield(env=Monitor(gym.make(game)),
                             ltl_formula=ltl_formula,
                             gateway=gateway,
                             min_depth=depth,
                             max_shield_life=shield_life)

                # Creating second environment to be used in evaluation callback
                eval_shield = EvaluationShield(training_env=env)
                eval_env = PreemptiveShieldWrapper(env=Monitor(gym.make(game)), shield=eval_shield)

                train(env=env,
                      label=dlabel,
                      game=game,
                      total_steps=total_steps,
                      learning_rate=learning_rate,
                      callback=callback,
                      eval_env=eval_env)
                gateway.close()
        else:
            env = shield(env=Monitor(gym.make(game)), ltl_formula=ltl_formula)
            train(env=env,
                  label=label,
                  game=game,
                  total_steps=total_steps,
                  learning_rate=learning_rate,
                  callback=callback)
    else:
        label = f'{game}-Standard'
        for penalty in penalties:
            env = Monitor(gym.make(game))
            env.set_penalty(penalty)
            plabel = f"{label}-penalty{penalty}"
            train(env=env,
                  label=plabel,
                  game=game,
                  total_steps=total_steps,
                  learning_rate=learning_rate,
                  callback=callback)


def train(env, label, game, callback, total_steps=int(5e5), learning_rate=1e-4, policy='MlpPolicy', root_dir='.',
          eval_env=None):
    """
    :param env:
    :param label:
    :param game:
    :param callback:
    :param total_steps:
    :param learning_rate:
    :param policy:
    :param root_dir:
    :param eval_env: We can give a dedicated environment for evaluation if the evaluation causes a concurrency issue.
    :return: the trained model
    """
    timestamp = datetime.now().strftime("%b%d_%H:%M:%S")
    host = socket.gethostname()
    log_dir = os.path.join(f"{root_dir}/logs", host, game, timestamp, label)
    model_dir = os.path.join(f"{root_dir}/models", host, game, timestamp, label)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model = PPO(policy, env, verbose=1, tensorboard_log=log_dir, learning_rate=learning_rate)
    # Shielding the policy for preemptive shielding
    if isinstance(env, PreemptiveShieldWrapper):
        shield_policy(model, eval_env=eval_env)
    # Set up evaluation environment
    if eval_env is not None and policy == 'CnnPolicy':
        eval_env = VecTransposeImage(DummyVecEnv([lambda: eval_env]))
    eval_callback = EvalCallback(eval_env=model.get_env() if eval_env is None else eval_env,
                                 callback_on_new_best=SaveBestShieldCallback(),
                                 best_model_save_path=os.path.join(log_dir, 'best_model'),
                                 n_eval_episodes=30,
                                 eval_freq=10000, verbose=1,
                                 deterministic=True, render=False)
    if isinstance(callback, list):
        callback_list = callback + [eval_callback]
    else:
        callback_list = [callback(), eval_callback]
    # Starting the learning process
    model.learn(total_timesteps=total_steps, tb_log_name=model.__class__.__name__.lower(),
                callback=callback_list)
    model.save(os.path.join(model_dir, model.__class__.__name__.lower()))
    return model
