import unittest

from src.logic import PassiveLearning
from src.logic.reduce_training_data import ReduceTrainingData
from test.base_tests import Py4JTestCase


class TestReduceTrainingData(Py4JTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.learner = PassiveLearning(self.gateway, 0, 1)
        self.training_data = [([0], 0), ([1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0, 0], 1),
                              ([0, 0, 1], 0), ([0, 1, 0], 0), ([0, 1, 1], 0), ([1, 0, 0], 0), ([1, 0, 1], 0),
                              ([1, 1, 0], 0), ([1, 1, 1], 1)]
        for (input_word, output_char) in self.training_data:
            self.learner.addSample(input_word, output_char)
        self.mealy = self.learner.computeMealy()
        self.reducer = ReduceTrainingData(self.mealy)
        for (input_word, output_char) in self.training_data:
            state = self.mealy.getInitialState()
            for action in input_word[:-1]:
                state = self.mealy.getSuccessor(state, action)
            self.assertEqual(output_char, self.mealy.getOutput(state, input_word[-1]))

    def test_filter_redundant_samples(self):
        reduced_training_data = list(self.reducer.filter_redundant_samples(self.training_data))
        new_learner = PassiveLearning(self.gateway, 0, 1)
        for (input_word, output_char) in reduced_training_data:
            new_learner.addSample(input_word, output_char)
        new_mealy = new_learner.computeMealy()
        for (input_word, output_char) in self.training_data:
            state = new_mealy.getInitialState()
            for action in input_word[:-1]:
                state = new_mealy.getSuccessor(state, action)
            self.assertEqual(output_char, new_mealy.getOutput(state, input_word[-1]))

    def test_make_separation_sequences(self):
        separation_sequences = self.reducer.make_separation_sequences()
        self.assertListEqual([[[0], [1]], [[0], [1]], [[0]]], separation_sequences)

    def test_make_initial_partition(self):
        initial_partition, initial_separation_sequences = self.reducer.make_initial_partition(self.mealy)
        self.assertListEqual([[[0], [1]], [[0], [1]], [[0]]], initial_separation_sequences)


if __name__ == '__main__':
    unittest.main()
