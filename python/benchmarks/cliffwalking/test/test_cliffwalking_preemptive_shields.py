import sys

sys.path.append('../../../')

from benchmarks.cliffwalking.cliffwalking_preemptive_wrappers import CliffWalkingDynamicPreemptiveShieldWrapper
from benchmarks.cliffwalking.cliffwalking_specifications import dont_fall_formula
from benchmarks.cliffwalking.cliffwalking_callbacks import CliffWalkingShieldingCallback
from src.wrappers.shield_policy import shield_policy
from test.base_tests import Py4JTestCase
import gym
from stable_baselines3 import PPO


class TestCliffWalkingPreemptiveShields(Py4JTestCase):
    def test_cliffwalking_preemptive_dynamic_shield_depth0(self) -> None:
        game = 'CliffWalkingExt-v0'
        env = CliffWalkingDynamicPreemptiveShieldWrapper(env=gym.make(game), ltl_formula=dont_fall_formula(),
                                                         gateway=self.gateway, min_depth=0, max_shield_life=20)
        model = PPO('MlpPolicy', env)
        shield_policy(model)
        model.learn(total_timesteps=int(5e3), callback=CliffWalkingShieldingCallback())

    def test_cliffwalking_preemptive_dynamic_shield_depth1(self) -> None:
        game = 'CliffWalkingExt-v0'
        env = CliffWalkingDynamicPreemptiveShieldWrapper(env=gym.make(game), ltl_formula=dont_fall_formula(),
                                                         gateway=self.gateway, min_depth=1, max_shield_life=20)
        model = PPO('MlpPolicy', env)
        shield_policy(model)
        model.learn(total_timesteps=int(5e3), callback=CliffWalkingShieldingCallback())
