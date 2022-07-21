import unittest
from src.logic import ltl_to_dfa_spot
from src.model import DFA
from benchmarks.cliffwalking.cliffwalking_specifications import dont_fall_formula


class TestTaxiSpecifications(unittest.TestCase):

    def test_initial_state_is_safe(self):
        safety = dont_fall_formula()
        dfa: DFA = ltl_to_dfa_spot(safety)
        self.assertEqual(1, len(dfa.getStates()))
        self.assertTrue(dfa.isSafe(dfa.getInitialState()))


if __name__ == '__main__':
    unittest.main()
