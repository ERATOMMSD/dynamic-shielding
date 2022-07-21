import logging
import os.path
from enum import Enum, auto
from typing import List, Optional, Union

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv

from src.shields import StaticShield, SafePadding
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_callbacks import SHIELD_PICKLE_NAME, SAFE_PADDING_PICKLE_NAME
from src.wrappers.shield_policy import shield_policy
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', level=logging.WARN)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class Result(Enum):
    SUCCESS = auto()
    CRASH = auto()
    NONE = auto()


def is_crash(result: Result) -> bool:
    return result == Result.CRASH


def is_success(result: Result) -> bool:
    return result == Result.SUCCESS


def evaluate(env: GymEnv, model: PPO, max_episode_steps: int) -> Result:
    # Enjoy trained agent
    obs = env.reset()
    states = None
    total_reward = 0.0
    for i in range(max_episode_steps):
        if hasattr(obs, 'copy'):
            obs = obs.copy()
        action, states = model.predict(obs, state=states, deterministic=True)
        obs, rewards, done, info = env.step(action)
        total_reward += rewards
        if done and i + 1 < max_episode_steps:
            LOGGER.debug(total_reward)
            if rewards > 0:
                return Result.SUCCESS
            else:
                return Result.CRASH
    LOGGER.debug(f'total_reward: {total_reward}')
    return Result.NONE


def evaluate_repeat(env: GymEnv, model: PPO, num_repetition: int, max_episode_steps: int,
                    path: str = None,
                    ltl_formula: Optional[Union[str, List[str]]] = None,
                    player1_alphabet: List[int] = None,
                    player2_alphabet: List[int] = None,
                    output_alphabet: List[int] = None,
                    evaluate_output=None) -> List[Result]:
    if path is not None and ltl_formula is not None and len(ltl_formula) > 0:
        shield_path = os.path.join(path, SHIELD_PICKLE_NAME)
        safe_padding_path = os.path.join(path, SAFE_PADDING_PICKLE_NAME)
        if os.path.exists(shield_path):
            shield = StaticShield.create_from_pickle(ltl_formula=ltl_formula,
                                                     pickle_filename=shield_path,
                                                     evaluate_output=evaluate_output)
            assert (player1_alphabet == shield.safety_game.p1_alphabet)
            assert (player2_alphabet == shield.safety_game.p2_alphabet)
            shield.reset()
            eval_env = PreemptiveShieldWrapper(env, shield)
            env = model._wrap_env(eval_env)
            model.env = env
            shield_policy(model, eval_env=eval_env, force_eval=True)
        elif os.path.exists(safe_padding_path):
            safe_padding = SafePadding(ltl_formula=ltl_formula,
                                       player1_alphabet=player1_alphabet,
                                       player2_alphabet=player2_alphabet,
                                       output_alphabet=output_alphabet,
                                       evaluate_output=evaluate_output)
            safe_padding.load_pickle(safe_padding_path)
            safe_padding.reset()
            eval_env = SafePaddingWrapper(env, safe_padding)
            env = model._wrap_env(eval_env)
            model.env = env
            shield_policy(model, eval_env=eval_env, force_eval=True)

    return list(map(lambda _: evaluate(eval_env, model, max_episode_steps), range(num_repetition)))


def evaluate_list_repeat(env: GymEnv, model_list: List[PPO], num_repetition: int,
                         max_episode_steps: int) -> List[Result]:
    """Evaluate the model with the same environment
    """
    return sum(list(map(lambda model: evaluate_repeat(env, model, num_repetition, max_episode_steps), model_list)), [])


def print_results(result_list: List[Result]):
    success_count = sum(map(is_success, result_list))
    crash_count = sum(map(is_crash, result_list))

    print(f'success count: {success_count} / {len(result_list)}')
    print(f'success rate: {success_count / len(result_list)}')
    print(f'crash count: {crash_count} / {len(result_list)}')
    print(f'crash rate: {crash_count / len(result_list)}')
