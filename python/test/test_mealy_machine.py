from src.logic import PassiveLearning
from test.base_tests import Py4JTestCase


class TestMealyMachine(Py4JTestCase):
    def test_states(self):
        learner = PassiveLearning(self.gateway, 1, 2)
        training_data = [([1], "0"), ([2], "b"), ([1, 1], "0"), ([1, 2], "b"), ([2, 1], "1"), ([2, 2], "b"),
                         ([2, 1, 1], "1"), ([2, 1, 2], "b"), ([2, 2, 1], "0"), ([2, 2, 2], "b"), ]
        for (inputWord, outputChar) in training_data:
            learner.addSample(inputWord, outputChar)
        mealy = learner.computeMealy()
        self.assertListEqual(list(range(len(mealy.getStates()))), list(mealy.getStates()))
