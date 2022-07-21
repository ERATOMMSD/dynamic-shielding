import unittest
from src.logic import ltl_to_dfa_spot
from src.model import DFA
from benchmarks.water_tank.water_tank_specifications import safety_formula


def closed_valuation(proposition: str) -> bool:
    return {"EMPTY": False, "FULL": False, "OPEN": False}[proposition]


def open_valuation(proposition: str) -> bool:
    return {"EMPTY": False, "FULL": False, "OPEN": True}[proposition]


class TestWaterTankSpecifications(unittest.TestCase):

    def test_initial_state_is_safe(self):
        safety = safety_formula()
        dfa: DFA = ltl_to_dfa_spot(safety)
        self.assertEqual(7, len(dfa.getStates()))
        self.assertTrue(dfa.isSafe(dfa.getInitialState()))
        self.assertNotEqual(dfa.getSuccessor(dfa.getInitialState(), open_valuation), dfa.getSinkState())
        self.assertNotEqual(dfa.getSuccessor(dfa.getInitialState(), closed_valuation), dfa.getSinkState())

    def test_opening_disabled_after_switching_to_close(self):
        safety = safety_formula()
        dfa: DFA = ltl_to_dfa_spot(safety)
        first_open = dfa.getSuccessor(dfa.getInitialState(), open_valuation)

        # Opening the valve when its closed locks the valve for three ticks
        first_close = dfa.getSuccessor(first_open, closed_valuation)
        self.assertEqual(dfa.getSuccessor(first_close, open_valuation), dfa.getSinkState())
        second_close = dfa.getSuccessor(first_close, closed_valuation)
        self.assertEqual(dfa.getSuccessor(second_close, open_valuation), dfa.getSinkState())
        third_close = dfa.getSuccessor(second_close, closed_valuation)

        # After three ticks, closing should be allowed
        self.assertNotEqual(dfa.getSuccessor(third_close, open_valuation), dfa.getSinkState())
        self.assertEqual(dfa.getSuccessor(third_close, closed_valuation), third_close)

    def test_closing_disabled_after_switching_to_open(self):
        safety = safety_formula()
        dfa: DFA = ltl_to_dfa_spot(safety)
        first_closed = dfa.getSuccessor(dfa.getInitialState(), closed_valuation)

        # Closing the valve when its open locks the valve for three ticks
        first_open = dfa.getSuccessor(first_closed, open_valuation)
        self.assertEqual(dfa.getSuccessor(first_open, closed_valuation), dfa.getSinkState())
        second_open = dfa.getSuccessor(first_open, open_valuation)
        self.assertEqual(dfa.getSuccessor(second_open, closed_valuation), dfa.getSinkState())
        third_open = dfa.getSuccessor(second_open, open_valuation)

        # After three ticks, opening should be allowed
        self.assertNotEqual(dfa.getSuccessor(third_open, closed_valuation), dfa.getSinkState())
        self.assertEqual(dfa.getSuccessor(third_open, open_valuation), third_open)


if __name__ == '__main__':
    unittest.main()
