import unittest
import itertools

from benchmarks.water_tank.water_tank_arena import Observations, WaterTankInputOutputManager, \
    WaterTankPropBuilder, AgentActions, EnvironmentActions


class TestWaterTankArena(unittest.TestCase):

    def test_evaluate_output_open(self) -> None:
        io_manager = WaterTankInputOutputManager()
        prop_builder = WaterTankPropBuilder(capacity=100)
        output_id = prop_builder.build(100, AgentActions.OPEN)
        valuation = io_manager.evaluate_output(output_id)
        self.assertTrue(valuation(Observations.OPEN.name))
        self.assertFalse(valuation(Observations.EMPTY.name))
        self.assertTrue(valuation(Observations.FULL.name))

    def test_evaluate_output_closed_and_empty(self) -> None:
        io_manager = WaterTankInputOutputManager()
        prop_builder = WaterTankPropBuilder(capacity=100)
        output_id = prop_builder.build(0, AgentActions.CLOSE)
        valuation = io_manager.evaluate_output(output_id)
        self.assertFalse(valuation(Observations.OPEN.name))
        self.assertTrue(valuation(Observations.EMPTY.name))
        self.assertFalse(valuation(Observations.FULL.name))

    def test_evaluate_output_closed_and_normal(self) -> None:
        io_manager = WaterTankInputOutputManager()
        prop_builder = WaterTankPropBuilder(capacity=100)
        output_id = prop_builder.build(20, AgentActions.CLOSE)
        valuation = io_manager.evaluate_output(output_id)
        self.assertFalse(valuation(Observations.OPEN.name))
        self.assertFalse(valuation(Observations.EMPTY.name))
        self.assertFalse(valuation(Observations.FULL.name))

    def test_evaluate_action_mapping(self):
        io_manager = WaterTankInputOutputManager()
        for agent_action, env_action in itertools.product(AgentActions, EnvironmentActions):
            input = io_manager.reverse_alphabet_mapper(agent_action.value, env_action.value)
            agent_value, env_value = io_manager.alphabet_mapper(input)
            self.assertEqual(agent_value, agent_action.value)
            self.assertEqual(env_value, env_action.value)
