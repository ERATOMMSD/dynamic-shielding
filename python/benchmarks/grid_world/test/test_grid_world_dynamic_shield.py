import unittest
from pathlib import Path

from benchmarks.grid_world.grid_world_arena import AgentActions, NoiseActions, make_two_robots_grid_world_mdp, \
    encode_environment_action
from benchmarks.grid_world.grid_world_dynamic_shield import GridWorldDynamicShield
from benchmarks.grid_world.grid_world_env import GridWorldEnvOneColor
from benchmarks.grid_world.grid_world_specifications import no_wall
from src.model.mdp import MDP
from src.shields.abstract_shield import UnsafeStateError
from test.base_tests import Py4JTestCase


class TestGridWorldDynamicShield(Py4JTestCase):
    def test_postposed_no_wall(self):
        ltl_formula: str = no_wall
        self.mdp: MDP = make_two_robots_grid_world_mdp(no_noise_for_wall=True)
        self.state = self.mdp.getInitialState()
        self.dynamic_shield = GridWorldDynamicShield(ltl_formula, self.gateway)
        self.dynamic_shield.reconstructShield()
        self.dynamic_shield.reset()

        # Initially, all the actions are allowed
        self.assertEqual({0, 1, 2, 3, 4}, set(self.dynamic_shield.preemptive()))
        self.assertIsNotNone(self.dynamic_shield.reactive_system)
        self.assertIsNotNone(self.dynamic_shield.reactive_system_state)
        self.assertTrue(self.dynamic_shield.consistent)
        self.assertTrue(self.dynamic_shield.consistent_from_latest_construction)
        self.move(0)
        self.assertEqual({0, 1, 2, 3, 4}, set(self.dynamic_shield.preemptive()))
        self.assertIsNotNone(self.dynamic_shield.reactive_system)
        self.assertIsNotNone(self.dynamic_shield.reactive_system_state)
        self.assertFalse(self.dynamic_shield.consistent)
        self.assertFalse(self.dynamic_shield.consistent_from_latest_construction)
        self.move(1)
        self.assertIsNotNone(self.dynamic_shield.reactive_system)
        self.assertIsNotNone(self.dynamic_shield.reactive_system_state)
        self.assertFalse(self.dynamic_shield.consistent)
        self.assertFalse(self.dynamic_shield.consistent_from_latest_construction)

        # Now, no action is allowed because we are at unsafe state
        with self.assertRaises(UnsafeStateError):
            self.dynamic_shield.preemptive()
        self.move(2)

        # reset the current state and try all the input of length 1
        for action in range(1, 5):
            self.reset()
            self.move(action)
        self.reset()
        self.assertEqual({0, 1, 4}, set(self.dynamic_shield.preemptive()))

    def move(self, player_int_action: int) -> None:
        enemy_action: int = encode_environment_action(AgentActions.STAY, NoiseActions.NO_NOISE)
        _, target = self.mdp.getSuccessorWithP2Action(self.state, player_int_action, enemy_action)
        self.dynamic_shield.move(player_int_action, enemy_action,
                                 self.mdp.getOutput(self.state, player_int_action, enemy_action))
        self.dynamic_shield.reconstructShield()
        self.state = target

    def reset(self) -> None:
        self.state = self.mdp.getInitialState()
        self.dynamic_shield.reset()


class TestGridWorldDynamicShieldWithGym(Py4JTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.gym: GridWorldEnvOneColor = GridWorldEnvOneColor()

    def test_postposed_int(self):
        ltl_formula: str = no_wall
        self.dynamic_shield = GridWorldDynamicShield(ltl_formula, self.gateway)
        self.dynamic_shield.reset()
        # Go DOWN
        self.assertEqual(int(AgentActions.DOWN), self.dynamic_shield.postposed(int(AgentActions.DOWN)))
        self.gym.step(int(AgentActions.DOWN))
        # We die here due to the WALL
        self.assertTrue(self.gym.done)
        action1, action2, output = self.gym.get_latest_actions()
        self.dynamic_shield.move(action1, action2, output)
        self.assertIsNotNone(self.dynamic_shield.reactive_system)
        self.assertIsNotNone(self.dynamic_shield.reactive_system_state)
        self.assertFalse(self.dynamic_shield.consistent)
        self.assertFalse(self.dynamic_shield.consistent_from_latest_construction)

        # We reset the game
        self.dynamic_shield.reset()
        self.gym.reset()
        # Going DOWN is unsafe
        self.assertNotEqual(int(AgentActions.DOWN), self.dynamic_shield.postposed(int(AgentActions.DOWN)))

    def test_load_pickle(self):
        ltl_formula: str = no_wall
        self.dynamic_shield = GridWorldDynamicShield(ltl_formula, self.gateway, concurrent_reconstruction=True)
        len_win_set_before = len(self.dynamic_shield.win_set)
        cd = Path(__file__).parent.absolute()
        self.dynamic_shield.load_transition_cover(str(cd / 'test.pickle'))
        self.dynamic_shield.reset()
        self.assertNotEqual(len_win_set_before, len(self.dynamic_shield.win_set))
        self.assertTrue(self.dynamic_shield.concurrent_reconstruction)
        self.assertTrue(self.dynamic_shield.consistent_from_latest_construction)


class TestGridWorldDynamicShieldNoLearning(unittest.TestCase):
    def test_alphabet_mapper_reverse_alphabet_mapper(self):
        for combined_action in range(50):
            player1_action, player2_action = GridWorldDynamicShield.alphabet_mapper(combined_action)
            self.assertEqual(combined_action,
                             GridWorldDynamicShield.reverse_alphabet_mapper(player1_action, player2_action))

    def test_reverse_alphabet_mapper_alphabet_mapper(self):
        for player1_action in AgentActions:
            for player2_action in AgentActions:
                for noise_action in NoiseActions:
                    player2_action_int: int = encode_environment_action(player2_action, noise_action)
                    self.assertEqual((int(player1_action), player2_action_int),
                                     GridWorldDynamicShield.alphabet_mapper(
                                         GridWorldDynamicShield.reverse_alphabet_mapper(int(player1_action),
                                                                                        player2_action_int)))


if __name__ == '__main__':
    unittest.main()
