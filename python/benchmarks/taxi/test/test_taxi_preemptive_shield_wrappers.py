import sys
import unittest

sys.path.append('../../../')

from benchmarks.taxi.taxi_preemptive_wrappers import TaxiDynamicPreemptiveShieldWrapper
from benchmarks.taxi.taxi_specifications import safe_and_proactive_formula
from benchmarks.taxi.taxi_callbacks import TaxiShieldingCallback
from src.wrappers.shield_policy import shield_policy
from test.base_tests import Py4JTestCase
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from benchmarks.common.train import train


class TestTaxiPreemptiveShieldWrappers(Py4JTestCase):
    def test_taxi_preemptive_dynamic_shield_depth0(self) -> None:
        game = 'TaxiFixStart-v3'
        env = TaxiDynamicPreemptiveShieldWrapper(env=gym.make(game), ltl_formula=safe_and_proactive_formula(),
                                                 gateway=self.gateway, min_depth=0, max_shield_life=20)
        model = PPO('MlpPolicy', env)
        shield_policy(model)
        model.learn(total_timesteps=int(5e3), callback=TaxiShieldingCallback())

    def test_taxi_preemptive_dynamic_shield_depth1(self) -> None:
        game = 'TaxiFixStart-v3'
        env = TaxiDynamicPreemptiveShieldWrapper(env=gym.make(game), ltl_formula=safe_and_proactive_formula(),
                                                 gateway=self.gateway, min_depth=1, max_shield_life=20)
        model = PPO('MlpPolicy', env)
        shield_policy(model)
        model.learn(total_timesteps=int(5e3), callback=TaxiShieldingCallback())
