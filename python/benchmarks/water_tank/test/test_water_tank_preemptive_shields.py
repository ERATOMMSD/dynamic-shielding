import sys

sys.path.append('../../../')

from benchmarks.water_tank.water_tank_preemptive_wrappers import WaterTankDynamicPreemptiveShieldWrapper
from benchmarks.water_tank.water_tank_callbacks import WaterTankShieldingCallback
from benchmarks.water_tank.water_tank_specifications import safety_formula
from src.wrappers.shield_policy import shield_policy
from test.base_tests import Py4JTestCase
import gym
from stable_baselines3 import PPO


class TestWaterTankPreemptiveShields(Py4JTestCase):
    def test_water_tank_preemptive_dynamic_shield_depth0(self) -> None:
        game = 'WaterTank-c100-i50-v0'
        env = WaterTankDynamicPreemptiveShieldWrapper(env=gym.make(game), ltl_formula=safety_formula(), gateway=self.gateway, min_depth=0, max_shield_life=20)
        model = PPO('MlpPolicy', env)
        shield_policy(model)
        model.learn(total_timesteps=int(5e3), callback=WaterTankShieldingCallback())

    def test_water_tank_preemptive_dynamic_shield_depth1(self) -> None:
        game = 'WaterTank-c100-i50-v0'
        env = WaterTankDynamicPreemptiveShieldWrapper(env=gym.make(game), ltl_formula=safety_formula(), gateway=self.gateway, min_depth=1, max_shield_life=20)
        model = PPO('MlpPolicy', env)
        shield_policy(model)
        model.learn(total_timesteps=int(5e3), callback=WaterTankShieldingCallback())
