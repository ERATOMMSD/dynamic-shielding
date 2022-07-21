import unittest
from src.model import PTA


class TestPTA(unittest.TestCase):
    def test_init(self):
        player1_alphabet = ['a', 'b', 'c']
        player2_alphabet = ['*']
        output_alphabet = ['0', '1']
        learner = PTA(player1_alphabet, player2_alphabet, output_alphabet)
        self.assertEqual(learner.player1Alphabet, player1_alphabet)
        self.assertEqual(learner.player2Alphabet, player2_alphabet)
        self.assertEqual(learner.outputAlphabet, output_alphabet)

    def test_addSample(self):
        player1_alphabet = ['a', 'b', 'c']
        player2_alphabet = ['*']
        output_alphabet = ['0', '1']
        learner = PTA(player1_alphabet, player2_alphabet, output_alphabet)
        training_data = [("a", "0"), ("b", "b"), ("aa", "0"), ("ab", "b"), ("ba", "1"), ("bb", "b"), ("baa", "1"),
                         ("bab", "b"), ("bba", "0"), ("bbb", "b"), ]
        for player1_actions, output in training_data:
            input_word = list(map(lambda x: (x, '*'), player1_actions))
            learner.addSample(input_word, output)

        # test the output
        for player1_actions, output in training_data:
            state: int = learner.getInitialState()
            for player1_action in player1_actions[:-1]:
                state = learner.getSuccessor(state, player1_action, '*')
            self.assertEqual(learner.getOutput(state, player1_actions[-1], '*'), output)


if __name__ == '__main__':
    unittest.main()
