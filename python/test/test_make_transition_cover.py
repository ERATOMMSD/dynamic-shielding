import unittest
from typing import List, Tuple

from test.base_tests import Py4JTestCase
from src.logic import PassiveLearning
from src.logic.make_transition_cover import make_transition_cover


class TestMakeTransitionCover(Py4JTestCase):
    def test_make_transition_cover(self):
        learner = PassiveLearning(self.gateway, 1, 2)
        training_data = [([1, 1], 0), ([1, 2], 0xb), ([2, 1], 1), ([2, 2], 0xb),
                         ([2, 1, 1], 1), ([2, 1, 2], 0xb), ([2, 2, 1], 0), ([2, 2, 2], 0xb), ]
        for (inputWord, outputChar) in training_data:
            learner.addSample(inputWord, outputChar)
        mealy = learner.computeMealy()

        transition_cover: List[Tuple[List[int], int]] = make_transition_cover(mealy)

        def fst(tpl):
            return tpl[0]

        self.assertIn([1], list(map(fst, transition_cover)))
        self.assertIn([2], list(map(fst, transition_cover)))
        self.assertIn([2, 1], list(map(fst, transition_cover)))
        self.assertIn([2, 2], list(map(fst, transition_cover)))


if __name__ == '__main__':
    unittest.main()
