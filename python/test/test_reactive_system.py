import os.path as osp
import pickle

from src.logic import PassiveLearning
from src.model import ReactiveSystem
from test.base_tests import Py4JTestCase


class TestReactiveSystem(Py4JTestCase):

    @staticmethod
    def generate_simpleReactiveSystem():
        reactive_system = ReactiveSystem([1, 3], [2], [10, 11])
        reactive_system.addTransition(0, 1, 2, 11, 1)
        reactive_system.addTransition(1, 1, 2, 11, 0)
        reactive_system.addTransition(0, 3, 2, 10, 0)
        reactive_system.addTransition(1, 3, 2, 10, 1)
        return reactive_system

    def test_simpleReactiveSystem(self):
        reactive_system = self.generate_simpleReactiveSystem()

        self.assertEqual(str(reactive_system), "{0: {'(1,2) / 11': 1, '(3,2) / 10': 0},"
                                               " 1: {'(1,2) / 11': 0, '(3,2) / 10': 1}}")
        self.assertEqual(set(reactive_system.getStates()), {0, 1})

    def test_save_reactive_system(self):
        reactive_system = self.generate_simpleReactiveSystem()
        with open('simple_reactive_system.pickle', mode='wb') as f:
            pickle.dump(reactive_system, f)

    def test_load_reactive_system(self):
        pickle_path = osp.join(osp.dirname(__file__), 'simple_reactive_system.pickle')
        with open(pickle_path, mode='rb') as f:
            reactive_system = pickle.load(f)
            self.assertEqual(str(reactive_system), "{0: {'(1,2) / 11': 1, '(3,2) / 10': 0},"
                                                   " 1: {'(1,2) / 11': 0, '(3,2) / 10': 1}}")
            self.assertEqual(set(reactive_system.getStates()), {0, 1})

    def test_mealyMachineToReactiveSystem(self):

        def alphabet_mapper(x: int) -> (int, int):
            return x * 100, x * 1000

        learner = PassiveLearning(self.gateway, 1, 2)
        training_data = [([1, 1], 0), ([1, 2], 0xb), ([1, 1], 0), ([1, 2], 0xb), ([2, 1], 1), ([2, 2], 0xb),
                         ([2, 1, 1], 1), ([2, 1, 2], 0xb), ([2, 2, 1], 0), ([2, 2, 2], 0xb), ]
        for (inputWord, outputChar) in training_data:
            learner.addSample(inputWord, outputChar)
        mealy = learner.computeMealy()
        reactive_system = ReactiveSystem.fromMealyMachine(mealy, alphabet_mapper)

        self.assertEqual({100, 200}, set(reactive_system.getPlayer1Alphabet()))
        self.assertEqual({1000, 2000}, set(reactive_system.getPlayer2Alphabet()))
        self.assertEqual(set(reactive_system.getStates()), set(mealy.getStates()))
        self.assertEqual(reactive_system.getInitialState(), mealy.getInitialState())
        for state in mealy.getStates():
            for action in mealy.getInputAlphabet():
                action_p1, action_p2 = alphabet_mapper(action)
                self.assertEqual(reactive_system.getSuccessor(state, action_p1, action_p2),
                                 mealy.getSuccessor(state, action))
                self.assertEqual(reactive_system.getOutput(state, action_p1, action_p2), mealy.getOutput(state, action))
