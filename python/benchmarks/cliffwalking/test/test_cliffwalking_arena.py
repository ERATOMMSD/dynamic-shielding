import unittest

from benchmarks.cliffwalking.cliffwalking_alphabet import CliffWalkingInputOutputManager, Observations, AgentActions, \
    EnvironmentActions
from benchmarks.cliffwalking.cliffwalking_arena import cliffwalking_reactive_system
from benchmarks.cliffwalking.cliffwalking_env import CliffWalkingExt


class TestCliffWalkingArena(unittest.TestCase):

    def test_right_is_cliff(self):
        io_manager = CliffWalkingInputOutputManager()
        rs = cliffwalking_reactive_system()
        output_id = rs.getOutput(rs.getInitialState(), AgentActions.RIGHT.value, EnvironmentActions.IDLE.value)
        successor = rs.getSuccessor(rs.getInitialState(), AgentActions.RIGHT.value, EnvironmentActions.IDLE.value)
        valuation = io_manager.evaluate_output(output_id)
        self.assertTrue(valuation(Observations.CLIFF.name))
        self.assertEqual(successor, rs.getInitialState())
        self.assertFalse(valuation(Observations.GOAL.name))

    def test_up_is_safe(self):
        io_manager = CliffWalkingInputOutputManager()
        rs = cliffwalking_reactive_system()
        output_id = rs.getOutput(rs.getInitialState(), AgentActions.UP.value, EnvironmentActions.IDLE.value)
        successor = rs.getSuccessor(rs.getInitialState(), AgentActions.UP.value, EnvironmentActions.IDLE.value)
        self.assertNotEqual(successor, rs.getInitialState())
        valuation = io_manager.evaluate_output(output_id)
        self.assertFalse(valuation(Observations.CLIFF.name))

    def test_three_step_is_cliff(self):
        io_manager = CliffWalkingInputOutputManager()
        rs = cliffwalking_reactive_system()
        # Move UP
        output_id = rs.getOutput(rs.getInitialState(), AgentActions.UP.value, EnvironmentActions.IDLE.value)
        state_1 = rs.getSuccessor(rs.getInitialState(), AgentActions.UP.value, EnvironmentActions.IDLE.value)
        valuation = io_manager.evaluate_output(output_id)
        self.assertFalse(valuation(Observations.CLIFF.name))
        self.assertNotEqual(state_1, rs.getInitialState())
        # Move RIGHT
        output_id = rs.getOutput(state_1, AgentActions.RIGHT.value, EnvironmentActions.IDLE.value)
        state_2 = rs.getSuccessor(state_1, AgentActions.RIGHT.value, EnvironmentActions.IDLE.value)
        valuation = io_manager.evaluate_output(output_id)
        self.assertFalse(valuation(Observations.CLIFF.name))
        self.assertNotEqual(state_2, state_1)
        # Move DOWN
        output_id = rs.getOutput(state_2, AgentActions.DOWN.value, EnvironmentActions.IDLE.value)
        state_3 = rs.getSuccessor(state_2, AgentActions.DOWN.value, EnvironmentActions.IDLE.value)
        valuation = io_manager.evaluate_output(output_id)
        self.assertTrue(valuation(Observations.CLIFF.name))
        self.assertNotEqual(state_2, state_3)

    def testCliffIsAlwaysCliff(self):
        env = CliffWalkingExt()
        self.assertTrue(env.is_cliff(env.s, AgentActions.RIGHT.value))
        row, col = env.decode(env.s)
        self.assertEqual(0, col)
        self.assertEqual(3, row)
        env.step(AgentActions.UP.value)
        row, col = env.decode(env.s)
        self.assertEqual(0, col)
        self.assertEqual(2, row)
        for i in range(1, 11):
            env.step(AgentActions.RIGHT.value)
            row, col = env.decode(env.s)
            self.assertEqual(i, col)
            self.assertEqual(2, row)
            self.assertTrue(env.is_cliff(env.s, AgentActions.DOWN.value))
        env.step(AgentActions.RIGHT.value)
        row, col = env.decode(env.s)
        self.assertEqual(11, col)
        self.assertEqual(2, row)
        self.assertTrue(env.is_goal(env.s, AgentActions.DOWN.value))
        env.step(AgentActions.DOWN.value)
        row, col = env.decode(env.s)
        self.assertEqual(11, col)
        self.assertEqual(3, row)
        self.assertTrue(env.is_cliff(env.s, AgentActions.LEFT.value))
