import unittest
import sys
import pathlib
from typing import Dict, Tuple

from src.logic.ltl2dfa_translator import ltl_to_dfa_spot, is_safety, spotGuardToDict


class TestLTLtoDFATranslation(unittest.TestCase):
    def test_translate(self):
        dfa = ltl_to_dfa_spot("!F(r & X(y))")
        self.assertEqual(True, True)

        self.assertIn('r', dfa.alphabet)
        self.assertIn('y', dfa.alphabet)

        self.assertEqual(dfa.initialState, 1)
        self.assertEqual(dfa.safeStates, [1,2])
        for k, v in dfa.transitions[1].items():
            print(k, type(k))
#        print(type())
        # TODO: Check correct transitions

    def test_is_safety(self):
        self.assertTrue(is_safety("!F(red & X(yellow))"))
        self.assertFalse(is_safety("red U yellow"))

    def test_guard_expr_parsing(self):
        param_list = [
            ("y",       [{'y': True}]),
            ("~y",      [{'y': False}]),
            ("a & ~b",  [{'a': True, 'b': False}]),
            ("~y | !r",  [{'y': False}, {'r': False}]),   # also tests whether ! is accepted
            ("a & ~b",  [{'a': True, 'b': False}]),
            ("~a & (b | ~c)",  [{'a': False, 'b': True}, {'a': False, 'c': False}]),
            ("~a & ~(b & ~c)",  [{'a': False, 'b': False}, {'a': False, 'c': True}]),
            ("a => b",  [{'a': False}, {'b': True}]),
            ("1", [{}])
        ]

        for formula, result in param_list:
            with self.subTest(formula=formula, result=result):
                adjusted = spotGuardToDict(formula)
                self.assertEqual(len(result), len(adjusted), f"Expected {len(result)} disjuncts, but received {len(adjusted)}")
                for conj in result:
                    self.assertTrue(conj in adjusted, f"Expected {conj} to be in the list of conjuncts, but it isn't.")

if __name__ == '__main__':
    unittest.main()
