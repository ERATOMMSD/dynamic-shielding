import unittest
from benchmarks.grid_world.grid_world_specifications import safe1, safe2, safe, safe1_with_no_crash_duration
import itertools
from src.logic import ltl_to_dfa_spot
from src.model import DFA


class TestGridWorldSpecifications(unittest.TestCase):

    def test_safe1(self):
        safe1_dfa: DFA = ltl_to_dfa_spot(safe1)
        # The DFA has only one state (the safe state) + one sink state
        self.assertEqual(1, len(safe1_dfa.getStates()))
        self.assertEqual(1, safe1_dfa.getInitialState())
        self.assertTrue(safe1_dfa.isSafe(1))
        self.assertFalse(safe1_dfa.isSafe(2))

        for is_wall, is_crash in itertools.product(range(0, 2), range(0, 2)):
            def valuation(variable: str) -> bool:
                if variable == 'NO_CRASH':
                    return not is_crash
                elif variable == 'WALL':
                    return is_wall
                else:
                    self.fail('unexpected variable: ' + variable)

            if is_wall or is_crash:
                # go to the sink state
                self.assertEqual(2, safe1_dfa.getSuccessor(1, valuation))
            else:
                self.assertEqual(1, safe1_dfa.getSuccessor(1, valuation))
            # stay in the sink state
            self.assertEqual(2, safe1_dfa.getSuccessor(2, valuation))

    def test_safe2(self):
        safe2_dfa: DFA = ltl_to_dfa_spot(safe2)
        # The DFA has two states (the safe state) + one sink state
        self.assertEqual(2, len(safe2_dfa.getStates()))
        self.assertEqual(1, safe2_dfa.getInitialState())
        self.assertTrue(safe2_dfa.isSafe(1))
        self.assertTrue(safe2_dfa.isSafe(2))
        self.assertFalse(safe2_dfa.isSafe(3))

        for is_bomb in range(0, 2):
            def valuation(variable: str) -> bool:
                if variable == 'BOMB':
                    return bool(is_bomb)
                else:
                    self.fail('unexpected variable: ' + variable)

            if is_bomb:
                # increment the state
                self.assertEqual(2, safe2_dfa.getSuccessor(1, valuation))
                self.assertEqual(3, safe2_dfa.getSuccessor(2, valuation))
            else:
                # go back to the initial state
                self.assertEqual(1, safe2_dfa.getSuccessor(1, valuation))
                self.assertEqual(1, safe2_dfa.getSuccessor(2, valuation))
            # stay in the sink state
            self.assertEqual(3, safe2_dfa.getSuccessor(3, valuation))

    def test_safe1_with_no_crash_duration(self):
        with self.assertRaises(IndexError):
            # the duration must be positive
            safe1_with_no_crash_duration(0)
        with self.assertRaises(IndexError):
            # the duration must be positive
            safe1_with_no_crash_duration(-1)
        self.assertEqual('((CrashPositions.NO_CRASH)) & (G (!ArenaPropositions.WALL))', safe1_with_no_crash_duration(1))
        self.assertEqual('((CrashPositions.NO_CRASH) & (X(CrashPositions.NO_CRASH))) & (G (!ArenaPropositions.WALL))',
                         safe1_with_no_crash_duration(2))
        self.assertEqual(
            '((CrashPositions.NO_CRASH) & (X(CrashPositions.NO_CRASH)) & (X(X(CrashPositions.NO_CRASH)))) & (G (!ArenaPropositions.WALL))',
            safe1_with_no_crash_duration(3))

    def test_safe(self):
        safe_dfa: DFA = ltl_to_dfa_spot(safe)
        # The DFA has two states (the safe state) + one sink state
        self.assertEqual(2, len(safe_dfa.getStates()))
        self.assertEqual(2, safe_dfa.getInitialState())
        self.assertTrue(safe_dfa.isSafe(1))
        self.assertTrue(safe_dfa.isSafe(2))
        self.assertFalse(safe_dfa.isSafe(3))

        for is_wall, is_crash, is_bomb in itertools.product(range(0, 2), range(0, 2), range(0, 2)):
            def valuation(variable: str) -> bool:
                if variable == 'NO_CRASH':
                    return not is_crash
                elif variable == 'WALL':
                    return is_wall
                elif variable == 'BOMB':
                    return is_bomb
                else:
                    self.fail('unexpected variable: ' + variable)

            if is_wall or is_crash:
                # go to the sink state
                self.assertEqual(3, safe_dfa.getSuccessor(1, valuation))
                self.assertEqual(3, safe_dfa.getSuccessor(2, valuation))
            elif is_bomb:
                # increment the state (Note: 2 is the initial state and 1 is the one bomb state)
                self.assertEqual(1, safe_dfa.getSuccessor(2, valuation))
                self.assertEqual(3, safe_dfa.getSuccessor(1, valuation))
            else:
                # go back to the initial state
                self.assertEqual(2, safe_dfa.getSuccessor(1, valuation))
                self.assertEqual(2, safe_dfa.getSuccessor(2, valuation))
            # stay in the sink state
            self.assertEqual(3, safe_dfa.getSuccessor(3, valuation))


if __name__ == '__main__':
    unittest.main()
