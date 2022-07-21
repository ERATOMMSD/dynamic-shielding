from unittest import TestCase

from benchmarks.grid_world.grid_world_env import GridWorldEnv, AbstractGridWorldEnv, str_to_int


class TestGridWorldEnv(TestCase):
    env: AbstractGridWorldEnv

    def setUp(self) -> None:
        self.env = GridWorldEnv()

    def test_initialState(self):
        self.assertEqual(2543, self.env.mdp.getInitialState())

    def test_state(self):
        self.assertEqual(self.env.mdp.getInitialState(), self.env.state)

    def test_stateSize(self):
        self.assertEqual(6642, len(self.env.mdp.getStates()))

    def test_step(self):
        delay_discount = -5e-06
        expected_io = [
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 1, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("DOWN", 0, False),
            ("DOWN", 0, False),
            ("DOWN", 0, False),
            ("DOWN", 0, False),
            ("DOWN", 0, False),
            ("DOWN", 0, False),
            ("DOWN", 0, False),
            ("DOWN", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 0, False),
            ("RIGHT", 1, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("UP", 1, False),
            ("DOWN", 0, False),
            ("LEFT", 0, False),
            ("LEFT", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 0, False),
            ("UP", 1, True),
        ]
        for action, expected_reward_without_delay, expected_done in expected_io:
            _, new_reward, done, _ = self.env.step(str_to_int(action))
            if expected_done == done:  # no crash
                self.assertEqual(expected_reward_without_delay + delay_discount, new_reward)
            else:  # crash
                self.assertTrue(done)
                self.assertEqual(-1, new_reward)
                break
