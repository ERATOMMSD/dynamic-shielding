import unittest

from benchmarks.grid_world.grid_world_arena import make_two_robots_grid_world_mdp
from src.logic import BlueFringeRPNI
from src.model import ReactiveSystem


class TestBlueFringeRPNI(unittest.TestCase):
    def test_init(self):
        player1_alphabet = ['a', 'b', 'c']
        player2_alphabet = ['*']
        output_alphabet = ['0', '1']
        learner = BlueFringeRPNI(player1_alphabet, player2_alphabet, output_alphabet)
        self.assertEqual(learner.pta.player1Alphabet, player1_alphabet)
        self.assertEqual(learner.pta.player2Alphabet, player2_alphabet)
        self.assertEqual(learner.pta.outputAlphabet, output_alphabet)

    def test_addSample(self):
        player1_alphabet = ['a', 'b', 'c']
        player2_alphabet = ['*']
        output_alphabet = ['0', '1']
        learner = BlueFringeRPNI(player1_alphabet, player2_alphabet, output_alphabet)
        training_data = [("a", "0"), ("b", "b"), ("aa", "0"), ("ab", "b"), ("ba", "1"), ("bb", "b"), ("baa", "1"),
                         ("bab", "b"), ("bba", "0"), ("bbb", "b"), ]
        for player1_actions, output_action in training_data:
            input_word = list(map(lambda x: (x, '*'), player1_actions))
            learner.addSample(input_word, output_action)

        # compute the reactive system
        reactive_system: ReactiveSystem = learner.compute_model(3)
        self.assertEqual(len(reactive_system.getStates()), 11)
        # test the output_action
        for player1_actions, output_action in training_data:
            state: int = reactive_system.getInitialState()
            for player1_action in player1_actions[:-1]:
                state = reactive_system.getSuccessor(state, player1_action, '*')
            self.assertEqual(reactive_system.getOutput(state, player1_actions[-1], '*'), output_action)

    def test_addSample_mergeable_depth1(self):
        player1_alphabet = ['a', 'b', 'c']
        player2_alphabet = ['*']
        output_alphabet = ['0', '1']
        depth_and_expected_state_size = [(0, 2), (1, 2), (2, 2), (3, 11), (4, 11)]

        learner = BlueFringeRPNI(player1_alphabet, player2_alphabet, output_alphabet)
        training_data = [("a", "1"), ("b", "1"), ("aa", "0"), ("ab", "0"), ("ba", "0"), ("bb", "0"), ("baa", "1"),
                         ("bab", "1"), ("bba", "1"), ("bbb", "1"), ]
        for player1_actions, output_action in training_data:
            input_word = list(map(lambda x: (x, '*'), player1_actions))
            learner.addSample(input_word, output_action)

        for depth, state_size in depth_and_expected_state_size:
            # compute the reactive system
            reactive_system: ReactiveSystem = learner.compute_model(depth)
            self.assertEqual(state_size, len(reactive_system.getStates()))
            # test the output_action
            for player1_actions, output_action in training_data:
                state: int = reactive_system.getInitialState()
                for player1_action in player1_actions[:-1]:
                    state = reactive_system.getSuccessor(state, player1_action, '*')
                self.assertEqual(output_action, reactive_system.getOutput(state, player1_actions[-1], '*'))

    # Test case to debug an actual execution of grid world
    def test_addSample_grid_world(self):
        mdp = make_two_robots_grid_world_mdp()
        learner = BlueFringeRPNI(mdp.player1Alphabet, mdp.player2Alphabet, mdp.outputAlphabet)

        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.GREEN_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.GREEN_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.BLUE_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.BLUE_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.BLUE_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.GREEN_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.BLUE_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.BLUE_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.BLUE_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.LEFT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.BOMB_CrashPropositions.CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.RIGHT', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.NOTHING_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)
        input_word = [('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.UP', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.UP'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.STAY'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.RIGHT'),
                      ('AgentActions.STAY', 'NoiseActions.NO_NOISE_AgentActions.DOWN'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT'),
                      ('AgentActions.DOWN', 'NoiseActions.NO_NOISE_AgentActions.LEFT')]
        output_action = "ArenaPropositions.WALL_CrashPropositions.NO_CRASH"
        learner.addSample(input_word, output_action)
        learner.compute_model(min_depth=2)


if __name__ == '__main__':
    unittest.main()
