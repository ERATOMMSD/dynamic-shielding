import random
import unittest

import gym
from pyvirtualdisplay import Display

display = Display(visible=False, size=(400, 300))
display.start()
from benchmarks.sidewalk.deterministic_side_walk import DeterministicSideWalk


class TestDeterministicSideWalk(unittest.TestCase):
    def setUp(self) -> None:
        self.env = gym.make('deterministic_side_walk-v0')

    def test_make(self):
        self.assertIsInstance(self.env, DeterministicSideWalk)

    def test_determinism(self):
        actions = [random.randint(0, 3) for _ in range(20)]
        observations = [self.env.reset()]
        for action in actions:
            obs, _, _, _ = self.env.step(action)
            observations.append(list(obs))
        self.assertTrue((observations[0] == self.env.reset()).all())
        for i in range(len(actions)):
            obs, _, _, _ = self.env.step(actions[i])
            self.assertTrue((observations[i + 1] == obs).all())


if __name__ == '__main__':
    unittest.main()
