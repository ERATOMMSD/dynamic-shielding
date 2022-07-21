import unittest
import random
import sys

from benchmarks.water_tank.test.water_tank_shield_debugger import ShieldedDebuggerWaterTank

sys.path.append('../../../')
from benchmarks.water_tank.water_tank_dynamic_shield import WaterTankDynamicShield
from benchmarks.water_tank.water_tank_arena import AgentActions, EnvironmentActions
from benchmarks.water_tank.water_tank_specifications import safety_formula
from test.base_tests import Py4JTestCase
from src.exceptions.shielding_exceptions import UnsafeStateError, InvalidMoveError, InvalidMoveWithShieldError


class DynamicallyShieldedWaterTank(ShieldedDebuggerWaterTank):
    def __init__(self, initial_level, capacity, formula, gateway, min_depth=0):
        shield = WaterTankDynamicShield(formula, gateway, min_depth)
        super().__init__(initial_level, capacity, shield)


class TestWaterTankDynamicShield(Py4JTestCase):
    def test_evaluate_losing_mini_tank(self) -> None:
        # Capacity 3, Initial Level 1
        shielded_arena = DynamicallyShieldedWaterTank(1, 3, safety_formula(), self.gateway)
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

        # After noticing that the tank becomes FULL it raises UnsafeStateError
        with self.assertRaises(UnsafeStateError):
            shielded_arena.explore_all()

    def test_evaluate_short_exploration(self) -> None:
        # Capacity 8, WL 3
        shielded_arena = DynamicallyShieldedWaterTank(3, 8, safety_formula(), self.gateway)
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

        shielded_arena.explore_all()

        # no full/empty output was observed --> no changes in the shield
        shielded_arena.reconstruct_shield()
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

    def test_evaluate_short_exploration_almost_empty(self) -> None:
        # Capacity 8, WL 1
        shielded_arena = DynamicallyShieldedWaterTank(1, 8, safety_formula(), self.gateway)
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

        # After noticing that the tank becomes EMPTY it raises UnsafeStateError
        with self.assertRaises(UnsafeStateError):
            shielded_arena.explore_all()

    def test_evaluate_short_exploration_almost_full(self) -> None:
        # Capacity 8, WL 1
        shielded_arena = DynamicallyShieldedWaterTank(7, 8, safety_formula(), self.gateway)
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

        # After noticing that the tank becomes FULL it raises UnsafeStateError
        with self.assertRaises(UnsafeStateError):
            shielded_arena.explore_all()

    @unittest.skip("This test does not seem to be applicable anymore.")
    def test_evaluate_impossible_almost_full(self) -> None:
        # Capacity 4, WL 3
        shielded_arena = DynamicallyShieldedWaterTank(3, 4, safety_formula(), self.gateway)
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

        shielded_arena.explore_all(unsafe_states=True)

        # The dynamic shield observes that opening the thank may lead to full tank
        shielded_arena.reconstruct_shield()

        # The agent executes the only available action
        self.assertEqual({AgentActions.CLOSE.value}, set(shielded_arena.preemptive()))
        # It is still unsafe to open because it may make the tank full
        shielded_arena.explore_all([(AgentActions.CLOSE, EnvironmentActions.NORMAL)], unsafe_states=True)
        shielded_arena.reconstruct_shield()

        # Same as above
        # It is unsafe to close because water level is already 1
        self.assertEqual({AgentActions.CLOSE.value}, set(shielded_arena.preemptive()))
        shielded_arena.explore_all([(AgentActions.CLOSE, EnvironmentActions.NORMAL),
                                    (AgentActions.CLOSE, EnvironmentActions.NORMAL)],
                                   unsafe_states=True)

        # The dynamic shield discovers that the tank may become empty if we keep the valve closed
        shielded_arena.reconstruct_shield()
        self.assertFalse(shielded_arena.shield_enabled)

    @unittest.skip("This test does not seem to be applicable anymore.")
    def test_evaluate_short_exploration_missing_transition(self) -> None:
        # Capacity 4, WL 3
        shielded_arena = DynamicallyShieldedWaterTank(3, 4, safety_formula(), self.gateway)
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

        shielded_arena.explore_all(unsafe_states=True)

        # The dynamic shield observes that opening the thank may lead to full tank
        shielded_arena.reconstruct_shield()

        # The agent executes the only available action
        self.assertEqual({AgentActions.CLOSE.value}, set(shielded_arena.preemptive()))
        shielded_arena.explore_all([(AgentActions.CLOSE, EnvironmentActions.NORMAL)], unsafe_states=True)
        shielded_arena.reconstruct_shield()

        # Same as above
        self.assertEqual({AgentActions.CLOSE.value}, set(shielded_arena.preemptive()))
        for p1_action, p2_action in [(AgentActions.OPEN, EnvironmentActions.NONE),
                                     (AgentActions.OPEN, EnvironmentActions.NORMAL),
                                     (AgentActions.OPEN, EnvironmentActions.HIGH),
                                     (AgentActions.CLOSE, EnvironmentActions.NONE)]:
            shielded_arena.moves([(AgentActions.CLOSE, EnvironmentActions.NORMAL),
                                  (AgentActions.CLOSE, EnvironmentActions.NORMAL)])
            shielded_arena.move(p1_action.value, p2_action.value)
            shielded_arena.reset()
        # Transition (AgentActions.CLOSE, EnvironmentActions.NORMAL) remains unexplored.
        # The dynamic shield DOES NOT discover that the tank may become empty if we keep the valve closed
        shielded_arena.reconstruct_shield()
        # Thus, it believes that closing the tank is a good option
        self.assertEqual({AgentActions.CLOSE.value}, set(shielded_arena.preemptive()))

        with self.assertRaises(UnsafeStateError):
            shielded_arena.moves([(AgentActions.CLOSE, EnvironmentActions.NORMAL),
                                  (AgentActions.CLOSE, EnvironmentActions.NORMAL),
                                  (AgentActions.CLOSE, EnvironmentActions.NORMAL)])

        shielded_arena.reset()
        self.assertFalse(shielded_arena.shield_enabled)

    @unittest.skip("Execution is too long")
    def test_mdp_explore_all_length_7_and_random_simulation(self):
        shielded_arena = DynamicallyShieldedWaterTank(5, 10, safety_formula(), self.gateway)
        random.seed(1)
        # Perform an exploration of all traces of length
        shielded_arena.explore_all_traces(max_length=7)

        self.assertTrue(shielded_arena.current_is_winning())
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

        # Intends to violate the specification by opening when is not possible
        shielded_arena.reset()
        shielded_arena.move(AgentActions.OPEN.value)
        shielded_arena.move(AgentActions.OPEN.value)
        shielded_arena.move(AgentActions.CLOSE.value)
        shielded_arena.move(AgentActions.CLOSE.value)
        self.assertEqual({AgentActions.CLOSE.value}, set(shielded_arena.preemptive()))
        with self.assertRaises(InvalidMoveWithShieldError):
            shielded_arena.move(AgentActions.OPEN.value)

        # Intends to violate the specification by closing when is not possible
        shielded_arena.reset()
        shielded_arena.move(AgentActions.CLOSE.value)
        shielded_arena.move(AgentActions.OPEN.value)
        shielded_arena.move(AgentActions.OPEN.value)
        self.assertEqual({AgentActions.OPEN.value}, set(shielded_arena.preemptive()))
        with self.assertRaises(InvalidMoveWithShieldError):
            shielded_arena.move(AgentActions.CLOSE.value)

        # Do a random exploration of 2000 trials of 200 steps on the already trained shield
        lengths, _, _, _, _ = shielded_arena.random_exploration(2000, 200)
        shielded_arena.reset()
        self.assertTrue(lengths[-1] == 200)
        self.assertTrue(shielded_arena.current_is_winning())
        self.assertEqual({AgentActions.CLOSE.value, AgentActions.OPEN.value}, set(shielded_arena.preemptive()))

if __name__ == '__main__':
    unittest.main()
