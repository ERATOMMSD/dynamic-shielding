import itertools
import random
import unittest
from typing import Callable, Dict, Tuple, List

from src.exceptions.shielding_exceptions import UnsafeStateError
from src.shields import DynamicShield
from test.base_tests import Py4JTestCase


def alphabet_mapper(mealy_action: int):
    if mealy_action in {1, 2}:
        return mealy_action, 1
    elif mealy_action == 3:
        return 1, 2
    elif mealy_action == 4:
        return 2, 2
    else:
        raise ValueError


def evaluate_output(s: int) -> Callable[[str], bool]:
    if s == 9:
        return lambda _: False
    else:
        return lambda _: True


def reverse_alphabet_mapper(player1_action: int, player2_action: int) -> int:
    reverse_dict: Dict[Tuple[int, int], int] = {
        (1, 1): 1,
        (2, 1): 2,
        (1, 2): 3,
        (2, 2): 4,
    }
    return reverse_dict[player1_action, player2_action]


class TestDynamicShield(Py4JTestCase):
    def test_tickShielding_Gp(self):
        ltl_formula = 'G(p)'

        id_str: Callable[[str], str] = lambda x: x

        dynamic_shield = DynamicShield(ltl_formula, self.gateway, 1, 4, alphabet_mapper, evaluate_output,
                                       reverse_alphabet_mapper, id_str)
        dynamic_shield.reconstructShield()

        # Initially, all the actions are allowed
        self.assertEqual({1, 2}, set(dynamic_shield.preemptive()))
        # add observations to show safe actions
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(2, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(2, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(2, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(2, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()

        # add observations to show unsafe actions
        dynamic_shield.move(2, 2, 10)
        dynamic_shield.move(1, 1, 9)
        dynamic_shield.reset()
        dynamic_shield.move(2, 2, 10)
        dynamic_shield.move(2, 1, 9)
        dynamic_shield.reset()
        dynamic_shield.move(2, 2, 10)
        dynamic_shield.move(1, 2, 9)
        dynamic_shield.reset()
        dynamic_shield.move(2, 2, 10)
        dynamic_shield.move(2, 2, 9)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(2, 1, 9)
        dynamic_shield.reset()
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.move(2, 2, 9)
        dynamic_shield.reset()

        # reconstruct the shield
        dynamic_shield.reconstructShield()
        dynamic_shield.reset()

        # Then, only a is allowed at state 1
        self.assertEqual(1, dynamic_shield.state)
        self.assertTrue(dynamic_shield.state in dynamic_shield.win_set)
        self.assertEqual([1], dynamic_shield.preemptive())
        # move to another safe state
        dynamic_shield.move(1, 1, 10)
        # self.assertTrue(dynamic_shield.safety_game.isSafe(dynamic_shield.state))
        # Only b is allowed here
        # self.assertEqual([2], dynamic_shield.preemptive())

    def test_fault_only_once(self):
        ltl_formula = 'G(p)'

        id_int: Callable[[int], int] = lambda x: x

        for _ in range(10):
            for length in range(1, 30):
                dynamic_shield = DynamicShield(ltl_formula, self.gateway, 1, 4, alphabet_mapper, evaluate_output,
                                               reverse_alphabet_mapper, id_int)
                dynamic_shield.reconstructShield()

                # Initially, all the actions are allowed
                self.assertEqual({1, 2}, set(dynamic_shield.preemptive()))
                random_actions: List[int] = []
                for i in range(length):
                    if random.random() < 0.5:
                        random_actions.append(1)
                    else:
                        random_actions.append(2)
                    if i < length - 1:
                        dynamic_shield.move(random_actions[i], 1, 10)
                    else:
                        dynamic_shield.move(random_actions[i], 1, 9)
                dynamic_shield.reset()

                for i in range(length - 1):
                    self.assertEqual(random_actions[i], dynamic_shield.postposed(random_actions[i]),
                                     f'Not Equal for {random_actions} at {i}')
                    dynamic_shield.move(random_actions[i], 1, 10)
                self.assertEqual(random_actions[-1], dynamic_shield.postposed(random_actions[-1]))

    def test_multiple_spec(self):
        ltl_formula = 'G(p)'
        ltl_formulas = ['G(p)', 'p && X(p)']

        id_int: Callable[[int], int] = lambda x: x

        single_dynamic_shield = DynamicShield(ltl_formula, self.gateway, 1, 4, alphabet_mapper, evaluate_output,
                                              reverse_alphabet_mapper, id_int)
        multiple_dynamic_shield = DynamicShield(ltl_formulas, self.gateway, 1, 4, alphabet_mapper, evaluate_output,
                                                reverse_alphabet_mapper, id_int)
        single_dynamic_shield.reconstructShield()
        multiple_dynamic_shield.reconstructShield()

        def move_both(player1_action, player2_action, output):
            single_dynamic_shield.move(player1_action, player2_action, output)
            multiple_dynamic_shield.move(player1_action, player2_action, output)

        def reset_both():
            single_dynamic_shield.reset()
            multiple_dynamic_shield.reset()

        # Initially, all the actions are allowed
        self.assertEqual({1, 2}, set(single_dynamic_shield.preemptive()))
        self.assertEqual({1, 2}, set(multiple_dynamic_shield.preemptive()))
        # add observations to show safe actions
        move_both(1, 1, 10)
        reset_both()
        move_both(2, 1, 10)
        reset_both()
        move_both(1, 2, 10)
        reset_both()
        move_both(2, 2, 10)
        reset_both()
        self.assertEqual({1, 2}, set(single_dynamic_shield.preemptive()))
        self.assertEqual({1, 2}, set(multiple_dynamic_shield.preemptive()))

        reset_both()

        # add observations to show unsafe actions
        move_both(2, 2, 10)
        move_both(1, 1, 9)
        reset_both()
        move_both(2, 2, 10)
        move_both(2, 1, 9)
        reset_both()
        move_both(2, 2, 10)
        move_both(1, 2, 9)
        reset_both()
        move_both(2, 2, 10)
        move_both(2, 2, 9)
        reset_both()
        move_both(1, 1, 10)
        move_both(1, 1, 10)
        move_both(2, 1, 9)
        reset_both()
        move_both(1, 1, 10)
        move_both(1, 1, 10)
        move_both(2, 2, 9)
        reset_both()

        # Then, only a is allowed at state 1
        self.assertEqual(1, single_dynamic_shield.state)
        self.assertEqual(1, multiple_dynamic_shield.state)

        self.assertIn(single_dynamic_shield.state, single_dynamic_shield.win_set)
        self.assertIn(multiple_dynamic_shield.state, multiple_dynamic_shield.win_set)

        self.assertEqual([1], single_dynamic_shield.preemptive())
        self.assertEqual([1], multiple_dynamic_shield.preemptive())

        # move to another safe state
        single_dynamic_shield.move(1, 1, 10)
        multiple_dynamic_shield.move(1, 1, 10)

        self.assertTrue(single_dynamic_shield.safety_game.isSafe(single_dynamic_shield.state))
        self.assertTrue(multiple_dynamic_shield.safety_game.isSafe(multiple_dynamic_shield.state))
        single_dynamic_shield.move(2, 1, 9)
        multiple_dynamic_shield.move(2, 1, 9)

        reset_both()
        move_both(1, 1, 10)
        move_both(1, 2, 10)
        move_both(1, 1, 9)
        reset_both()
        move_both(1, 2, 10)
        move_both(1, 2, 10)
        move_both(1, 1, 9)
        reset_both()
        move_both(1, 1, 10)
        move_both(1, 1, 10)
        move_both(1, 1, 9)
        reset_both()
        move_both(1, 2, 10)
        move_both(1, 1, 10)
        move_both(1, 1, 9)
        reset_both()

        # Here, only multiple_dynamic_shield works fine since the specification in single_dynamic_shield is too strong.
        with self.assertRaises(UnsafeStateError):
            single_dynamic_shield.preemptive()
        self.assertEqual([1], multiple_dynamic_shield.preemptive())

    def test_alphabet(self):
        ltl_formula = 'G(p)'
        id_int: Callable[[int], int] = lambda x: x

        class AlphabetMapper(Callable[[int], Tuple[int, int]]):
            def __init__(self, _player1_alphabet_size: int):
                self.player1_alphabet_size = _player1_alphabet_size

            def __call__(self, mealy_action: int) -> Tuple[int, int]:
                return (mealy_action - 1) % self.player1_alphabet_size + 1, \
                       (mealy_action - 1) // self.player1_alphabet_size + 1

        for player1_alphabet_size, player2_alphabet_size in itertools.product(range(2, 5), range(2, 5)):
            dynamic_shield = DynamicShield(ltl_formula, self.gateway, 1, player1_alphabet_size * player2_alphabet_size,
                                           AlphabetMapper(player1_alphabet_size), evaluate_output,
                                           reverse_alphabet_mapper, id_int)

            self.assertEqual(len(dynamic_shield.learner.alphabet),
                             len(dynamic_shield.safety_game.getPlayer1Alphabet()) * len(
                                 dynamic_shield.safety_game.getPlayer2Alphabet()))

    def test_not_use_deviating_shield(self):
        ltl_formula = 'G(p)'
        id_int: Callable[[int], int] = lambda x: x

        class AlphabetMapper(Callable[[int], Tuple[int, int]]):
            def __init__(self, _player1_alphabet_size: int):
                self.player1_alphabet_size = _player1_alphabet_size

            def __call__(self, mealy_action: int) -> Tuple[int, int]:
                return (mealy_action - 1) % self.player1_alphabet_size + 1, \
                       (mealy_action - 1) // self.player1_alphabet_size + 1

        player1_alphabet_size = 2
        player2_alphabet_size = 1
        dynamic_shield = DynamicShield(ltl_formula, self.gateway, 1, player1_alphabet_size * player2_alphabet_size,
                                       AlphabetMapper(player1_alphabet_size), evaluate_output,
                                       reverse_alphabet_mapper, id_int)
        dynamic_shield_stop_deviation = DynamicShield(ltl_formula, self.gateway, 1,
                                                      player1_alphabet_size * player2_alphabet_size,
                                                      AlphabetMapper(player1_alphabet_size), evaluate_output,
                                                      reverse_alphabet_mapper, id_int, not_use_deviating_shield=True)

        def reconstruct():
            dynamic_shield.reconstructShield()
            dynamic_shield_stop_deviation.reconstructShield()

        def move(player1_input, player2_input, output):
            dynamic_shield.move(player1_input, player2_input, output)
            dynamic_shield_stop_deviation.move(player1_input, player2_input, output)

        def reset():
            dynamic_shield.reset()
            dynamic_shield_stop_deviation.reset()

        reconstruct()

        # feed training data and reconstruct
        move(1, 1, 0)
        reset()
        move(2, 1, 9)
        reset()
        reconstruct()

        # Try unexplored input
        self.assertEqual(1, dynamic_shield.postposed(1))
        self.assertEqual(1, dynamic_shield_stop_deviation.postposed(1))
        move(1, 1, 0)
        self.assertEqual(1, dynamic_shield.postposed(1))
        self.assertEqual(1, dynamic_shield_stop_deviation.postposed(1))
        move(1, 1, 1)  # deviation!!
        self.assertFalse(dynamic_shield.consistent)
        self.assertFalse(dynamic_shield_stop_deviation.consistent)
        self.assertEqual(1, dynamic_shield.postposed(2))
        self.assertEqual(2, dynamic_shield_stop_deviation.postposed(2))

    def test_pickle(self):
        ltl_formula = 'G(p)'

        id_str: Callable[[str], str] = lambda x: x

        dynamic_shield = DynamicShield(ltl_formula, self.gateway, 1, 4, alphabet_mapper, evaluate_output,
                                       reverse_alphabet_mapper, id_str)
        dynamic_shield.reconstructShield()

        # Initially, all the actions are allowed
        self.assertEqual({1, 2}, set(dynamic_shield.preemptive()))
        # add observations to show safe actions
        dynamic_shield.move(1, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(2, 1, 10)
        dynamic_shield.reset()
        dynamic_shield.move(1, 2, 10)
        dynamic_shield.reset()

        dynamic_shield._save_transition_cover('cover.pickle')
        new_dynamic_shield = DynamicShield(ltl_formula, self.gateway, 1, 4, alphabet_mapper, evaluate_output,
                                           reverse_alphabet_mapper, id_str)
        new_dynamic_shield.load_transition_cover('cover.pickle')

if __name__ == '__main__':
    unittest.main()
