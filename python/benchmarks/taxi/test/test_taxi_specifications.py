import unittest
from src.logic import ltl_to_dfa_spot
from src.model import DFA
from benchmarks.taxi.taxi_specifications import drive_safely, safe_and_proactive_formula


class TestTaxiSpecifications(unittest.TestCase):

    def test_initial_state_is_safe(self):
        safety = drive_safely()
        dfa: DFA = ltl_to_dfa_spot(safety)
        self.assertEqual(1, len(dfa.getStates()))
        self.assertTrue(dfa.isSafe(dfa.getInitialState()))

    def test_initial_state_is_safe_proactive_formula(self):
        safety = safe_and_proactive_formula()
        dfa: DFA = ltl_to_dfa_spot(safety)
        self.assertEqual(4, len(dfa.getStates()))
        self.assertTrue(dfa.isSafe(dfa.getInitialState()))


if __name__ == '__main__':
    unittest.main()
