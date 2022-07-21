import itertools
from unittest import TestCase

import gym

from benchmarks.grid_world.grid_world_arena import evaluate_output
from benchmarks.grid_world.grid_world_arena2 import make_grid_world_mdp, CrashPropositions, ArenaPropositions, \
    decode_int_output, AgentActions, decode_mdp_int_output
from benchmarks.grid_world.grid_world_dynamic_shield import GridWorldDynamicShield


class TestGridWorldArena2(TestCase):
    def assertIntermediateReward(self, done, reward):
        if done:
            self.assertEqual(-1.0, reward)
        else:
            self.assertEqual(-0.5 / self.max_step, reward)

    def setUp(self) -> None:
        self.env = gym.make('grid_world2-v1')
        self.max_step = self.env.MAX_STEP
        self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_reward(self):
        for _ in range(3):
            _, reward, done, _ = self.env.step(int(AgentActions.RIGHT))
            self.assertFalse(done)
            self.assertIntermediateReward(done, reward)
            self.assertIn(ArenaPropositions.NOTHING, decode_mdp_int_output(self.env.latest_output))
        _, reward, done, _ = self.env.step(int(AgentActions.RIGHT))
        self.assertIntermediateReward(done, reward)
        self.assertIn(ArenaPropositions.NOTHING, decode_mdp_int_output(self.env.latest_output))
        if done:
            return

        for _ in range(2):
            _, reward, done, _ = self.env.step(int(AgentActions.UP))
            self.assertIntermediateReward(done, reward)
            self.assertIn(ArenaPropositions.NOTHING, decode_mdp_int_output(self.env.latest_output))
            if done:
                return

        _, reward, done, _ = self.env.step(int(AgentActions.LEFT))
        self.assertIntermediateReward(done, reward)
        self.assertIn(ArenaPropositions.NOTHING, decode_mdp_int_output(self.env.latest_output))
        if done:
            return

        _, reward, done, _ = self.env.step(int(AgentActions.LEFT))
        self.assertIn(ArenaPropositions.GOAL, decode_mdp_int_output(self.env.latest_output))
        self.assertTrue(done)
        if CrashPropositions.CRASH in decode_mdp_int_output(self.env.latest_output):
            self.assertEqual(0, reward)
        else:
            self.assertEqual(1.0 - 0.5 / self.max_step, reward)

    def test_make_grid_world_mdp(self):
        mdp = make_grid_world_mdp()
        for src, player1_action, player2_action in itertools.product(mdp.getStates(),
                                                                     mdp.getPlayer1Alphabet(),
                                                                     mdp.getPlayer2Alphabet()):
            try:
                output = mdp.getOutput(src, player1_action, player2_action)
            except KeyError:
                continue
            self.assertEqual(int, type(output))
            direct_valuation = evaluate_output(self.env.translate_output(output), decode_int_output)
            valuation_via_mealy = GridWorldDynamicShield.evaluate_output(
                GridWorldDynamicShield.reverse_output_mapper(output))
            # Try all the propositions
            for crash_proposition in CrashPropositions:
                self.assertEqual(direct_valuation(crash_proposition), valuation_via_mealy(crash_proposition))
            for arena_proposition in ArenaPropositions:
                self.assertEqual(direct_valuation(arena_proposition), valuation_via_mealy(arena_proposition))
