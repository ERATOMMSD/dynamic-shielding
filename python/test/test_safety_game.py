import unittest
from typing import Dict, Tuple, Callable

from src.model import DFA, SafetyGame, ReactiveSystem


class TestSafetyGame(unittest.TestCase):
    def test_getPredecessorAfterAddTransition(self):
        safety_game = SafetyGame([1, 2], [0])
        safety_game.setInitialState(1)
        safety_game.add_transition(1, 1, 0, 2)
        safety_game.add_transition(1, 2, 0, 1)
        safety_game.add_transition(2, 1, 0, 2)
        safety_game.add_transition(2, 2, 0, 1)
        self.assertEqual(set(), safety_game.getPredecessors(1, 1, 0))
        self.assertEqual({1, 2}, safety_game.getPredecessors(1, 2, 0))
        self.assertEqual({1, 2}, safety_game.getPredecessors(2, 1, 0))
        self.assertEqual(set(), safety_game.getPredecessors(2, 2, 0))

    def test_getPredecessorAfterSetTransitions(self):
        safety_game = SafetyGame([1, 2], [0])
        safety_game.setInitialState(1)
        transitions: Dict[Tuple[int, int, int], int] = {(1, 1, 0): 2,
                                                        (1, 2, 0): 1,
                                                        (2, 1, 0): 2,
                                                        (2, 2, 0): 1}
        safety_game.setTransitions(transitions)
        self.assertEqual(set(), safety_game.getPredecessors(1, 1, 0))
        self.assertEqual({1, 2}, safety_game.getPredecessors(1, 2, 0))
        self.assertEqual({1, 2}, safety_game.getPredecessors(2, 1, 0))
        self.assertEqual(set(), safety_game.getPredecessors(2, 2, 0))

    def test_fromReactiveSystemAndDFA(self):
        reactiveSystem = ReactiveSystem([1, 2], [0],
                                        [0b00, 0b01, 0b10, 0b11])
        reactiveSystem.addTransition(1, 1, 0, 0b11, 2)
        reactiveSystem.addTransition(1, 2, 0, 0b01, 3)
        reactiveSystem.addTransition(2, 1, 0, 0b00, 1)
        reactiveSystem.addTransition(2, 2, 0, 0b10, 3)
        reactiveSystem.addTransition(3, 1, 0, 0b11, 1)
        reactiveSystem.addTransition(3, 2, 0, 0b00, 2)

        dfa = DFA(['p', 'q'])
        dfa.addTransition(1, {'p': True}, 1)
        dfa.addTransition(1, {'p': False, 'q': False}, 1)
        dfa.addTransition(1, {'p': False, 'q': True}, 2)
        dfa.addTransition(2, {'p': True}, 2)
        dfa.addTransition(2, {'p': False}, 1)
        dfa.addSafeState(1)

        def evaluate_output(output: int) -> Callable[[str], bool]:
            return lambda ap: output // 2 == 1 if ap == 'p' else output % 2 == 1

        safety_game = SafetyGame.fromReactiveSystemAndDFA(reactiveSystem, dfa,
                                                          evaluate_output)

        self.assertEqual(7, len(safety_game.getStates()))
        self.assertEqual(1, safety_game.getInitialState())
        self.assertEqual([1, 2], safety_game.getPlayer1Alphabet())
        self.assertEqual([0], safety_game.getPlayer2Alphabet())
        # test transitions
        self.assertEqual(safety_game.state_mapper[(2, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(1, 1)], 1, 0))
        self.assertEqual(safety_game.state_mapper[(3, 2)],
                         safety_game.getSuccessor(safety_game.state_mapper[(1, 1)], 2, 0))
        self.assertEqual(safety_game.state_mapper[(2, 2)],
                         safety_game.getSuccessor(safety_game.state_mapper[(1, 2)], 1, 0))
        self.assertEqual(safety_game.state_mapper[(3, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(1, 2)], 2, 0))

        self.assertEqual(safety_game.state_mapper[(1, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(2, 1)], 1, 0))
        self.assertEqual(safety_game.state_mapper[(3, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(2, 1)], 2, 0))
        self.assertEqual(safety_game.state_mapper[(1, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(2, 2)], 1, 0))
        self.assertEqual(safety_game.state_mapper[(3, 2)],
                         safety_game.getSuccessor(safety_game.state_mapper[(2, 2)], 2, 0))

        self.assertEqual(safety_game.state_mapper[(1, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(3, 1)], 1, 0))
        self.assertEqual(safety_game.state_mapper[(2, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(3, 1)], 2, 0))
        self.assertEqual(safety_game.state_mapper[(1, 2)],
                         safety_game.getSuccessor(safety_game.state_mapper[(3, 2)], 1, 0))
        self.assertEqual(safety_game.state_mapper[(2, 1)],
                         safety_game.getSuccessor(safety_game.state_mapper[(3, 2)], 2, 0))


if __name__ == '__main__':
    unittest.main()
