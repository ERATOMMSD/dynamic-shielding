import unittest
from typing import Callable, List, Tuple

from src.shields import PTADynamicShield


def evaluate_output(s: str) -> Callable[[str], bool]:
    if s == '0':
        return lambda _: False
    else:
        return lambda _: True


class TestPTADynamicShield(unittest.TestCase):
    def test_tickShielding_Gp(self):
        ltl_formula = '[] (p)'
        min_depth_expected_size: List[Tuple[int, int]] = [(1, 9), (2, 14), (3, 23), (4, 30), (5, 30), (6, 30),
                                                          (9999999, 30)]
        nonempty_player2_alphabet = ['a', 'b']
        for player2_alphabet in [nonempty_player2_alphabet, ['a']]:  # player2_alphabet can be incomplete
            for min_depth, expected_size in min_depth_expected_size:
                dynamic_shield = PTADynamicShield(ltl_formula, ['a', 'b'], player2_alphabet, ['0', '1'],
                                                  evaluate_output,
                                                  min_depth=min_depth, no_merging=False)

                # Initially, all the actions are allowed
                self.assertEqual({'a', 'b'}, set(dynamic_shield.preemptive()))
                # add observations to show safe actions
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('b', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('b', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('b', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('b', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.move('a', 'b', '1')
                dynamic_shield.reset()

                # add observations to show unsafe actions
                dynamic_shield.move('b', 'b', '1')
                dynamic_shield.move('a', 'a', '0')
                dynamic_shield.reset()
                dynamic_shield.move('b', 'b', '1')
                dynamic_shield.move('a', 'b', '0')
                dynamic_shield.reset()
                dynamic_shield.move('b', 'b', '1')
                dynamic_shield.move('b', 'a', '0')
                dynamic_shield.reset()
                dynamic_shield.move('b', 'b', '1')
                dynamic_shield.move('b', 'b', '0')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('b', 'a', '0')
                dynamic_shield.reset()
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('a', 'a', '1')
                dynamic_shield.move('b', 'b', '0')
                dynamic_shield.reset()

                self.assertEqual(expected_size, len(dynamic_shield.safety_game.getStates()))
                # Then, only a is allowed at state 1
                self.assertEqual(1, dynamic_shield.state)
                self.assertIn(dynamic_shield.state, dynamic_shield.win_set)
                self.assertEqual(['a'], dynamic_shield.preemptive())
                # move to another safe state
                dynamic_shield.move('a', 'a', '1')
                self.assertIn(dynamic_shield.state, dynamic_shield.win_set)
                self.assertEqual(['a', 'b'], dynamic_shield.preemptive())
                dynamic_shield.move('a', 'a', '1')
                self.assertIn(dynamic_shield.state, dynamic_shield.win_set)
                # Only a is allowed here
                self.assertEqual(['a'], dynamic_shield.preemptive())
                # player2_alphabet should be constructed as expected
                self.assertEqual(nonempty_player2_alphabet, dynamic_shield.safety_game.getPlayer2Alphabet())


if __name__ == '__main__':
    unittest.main()
