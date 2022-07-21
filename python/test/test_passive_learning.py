import string
import unittest

from src.logic import PassiveLearning
from test.base_tests import Py4JTestCase


def remove_blanks(s):
    return s.translate({ord(c): None for c in string.whitespace})


class TestPassiveLearning(Py4JTestCase):
    def test_init_str(self):
        learner = PassiveLearning(self.gateway, 'a', 'c')
        self.assertEqual(list(learner.alphabet), [ord('a'), ord('b'), ord('c')])

    def test_init_int(self):
        learner = PassiveLearning(self.gateway, 0, 10)
        self.assertEqual(list(learner.alphabet), list(range(0, 11)))

    def test_addSamples(self):
        learner = PassiveLearning(self.gateway, 'a', 'b')
        training_data = [("a", "0"), ("b", "b"), ("aa", "0"), ("ab", "b"), ("ba", "1"), ("bb", "b"), ("baa", "1"),
                         ("bab", "b"), ("bba", "0"), ("bbb", "b"), ]
        for (inputWord, outputChar) in training_data:
            learner.addSample(inputWord, outputChar)
        expected_dot = """digraph g {

	s0 [shape="circle" label="0"];
	s1 [shape="circle" label="1"];
	s0 -> s0 [label="97 / 48"];
	s0 -> s1 [label="98 / 98"];
	s1 -> s1 [label="97 / 49"];
	s1 -> s0 [label="98 / 98"];

__start0 [label="" shape="none" width="0" height="0"];
__start0 -> s0;
}
"""
        self.assertEqual(remove_blanks(learner.computeMealy().getDot()), remove_blanks(expected_dot))

    def test_addSamples_int(self):
        learner = PassiveLearning(self.gateway, 1, 2)
        training_data = [([1], "0"), ([2], "b"), ([1, 1], "0"), ([1, 2], "b"), ([2, 1], "1"), ([2, 2], "b"),
                         ([2, 1, 1], "1"), ([2, 1, 2], "b"), ([2, 2, 1], "0"), ([2, 2, 2], "b"), ]
        for (inputWord, outputChar) in training_data:
            learner.addSample(inputWord, outputChar)
        expected_dot = """digraph g {

    	s0 [shape="circle" label="0"];
    	s1 [shape="circle" label="1"];
    	s0 -> s0 [label="1 / 48"];
    	s0 -> s1 [label="2 / 98"];
    	s1 -> s1 [label="1 / 49"];
    	s1 -> s0 [label="2 / 98"];

    __start0 [label="" shape="none" width="0" height="0"];
    __start0 -> s0;
    }
    """
        self.assertEqual(remove_blanks(expected_dot), remove_blanks(learner.computeMealy().getDot()))

    def test_addSamples_int_zero(self):
        learner = PassiveLearning(self.gateway, 1, 2, min_depth=1)
        learner.min_depth = 0
        training_data = [([1], "0"), ([2], "b"), ([1, 1], "0"), ([1, 2], "b"), ([2, 1], "1"), ([2, 2], "b"),
                         ([2, 1, 1], "1"), ([2, 1, 2], "b"), ([2, 2, 1], "0"), ([2, 2, 2], "b"), ]
        for (inputWord, outputChar) in training_data:
            learner.addSample(inputWord, outputChar)
        expected_dot = """digraph g {

    	s0 [shape="circle" label="0"];
    	s1 [shape="circle" label="1"];
    	s0 -> s0 [label="1 / 48"];
    	s0 -> s1 [label="2 / 98"];
    	s1 -> s1 [label="1 / 49"];
    	s1 -> s0 [label="2 / 98"];

    __start0 [label="" shape="none" width="0" height="0"];
    __start0 -> s0;
    }
    """
        self.assertEqual(remove_blanks(expected_dot), remove_blanks(learner.computeMealy().getDot()))

    def test_addSamplesForStrongRPNI(self):
        expected_state_size = [3, 4, 5, 5, 9, 15, 15, 15, 15, 15]
        for min_depth in range(10):
            learner = PassiveLearning(self.gateway, 'a', 'b', min_depth)
            training_data = [("abbab", "0"),
                             ("baaba", "1"),
                             ("aabaa", "0"),
                             ("ababb", "1"),
                             ("aabbb", "0"),
                             ("abaab", "1")]
            for (input_word, output_char) in training_data:
                learner.addSample(input_word, output_char)
            self.assertEqual(expected_state_size[min_depth], len(learner.computeMealy().getStates()))

    def test_addSamples_insufficient(self):
        learner = PassiveLearning(self.gateway, 'a', 'b')
        training_data = [("a", "0")]
        for (input_word, output_char) in training_data:
            learner.addSample(input_word, output_char)
        expected_dot = """digraph g {
    
        s0 [shape="circle" label="0"];
        s0 -> s0 [label="97 / 48"];
    
    __start0 [label="" shape="none" width="0" height="0"];
    __start0 -> s0;
    }"""
        self.assertEqual(remove_blanks(learner.computeMealy().getDot()), remove_blanks(expected_dot))

    def test_getSamples_unimplemented(self):
        learner = PassiveLearning(self.gateway, 'a', 'b')
        training_data = [("a", "0"), ("b", "b"), ("aa", "0"), ("ab", "b"), ("ba", "1"), ("bb", "b"), ("baa", "1"),
                         ("bab", "b"), ("bba", "0"), ("bbb", "b"), ]
        learner.addSamples(training_data)
        learner.computeMealy()
        with self.assertRaises(NotImplementedError):
            learner.getSamples()

    def test_getSamples(self):
        learner = PassiveLearning(self.gateway, 0, 1, min_depth=5)
        training_data = [([0], 0), ([1], 2), ([0, 0], 0), ([0, 1], 2), ([1, 0], 1), ([1, 1], 2), ([1, 0, 0], 1),
                         ([1, 0, 1], 2), ([1, 1, 0], 0), ([1, 1, 1], 2), ]
        learner.addSamples(training_data)
        learner.computeMealy()
        java_samples = learner.getSamples()
        self.assertEqual(len(training_data), len(java_samples))
        for elem in training_data:
            self.assertIn(elem, java_samples)


if __name__ == '__main__':
    unittest.main()
