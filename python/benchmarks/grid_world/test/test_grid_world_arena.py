from unittest import TestCase

from benchmarks.grid_world.grid_world_arena import encode_int_output, decode_int_output, ArenaPropositions, \
    CrashPropositions


class TestGridWorldArena(TestCase):
    def test_decode_encode_int_output(self):
        for int_output in range(len(ArenaPropositions) * len(CrashPropositions)):
            arena_proposition, crash_proposition = decode_int_output(int_output)
            self.assertEqual(int_output, encode_int_output(arena_proposition, crash_proposition))

    def test_encode_decode_int_output(self):
        for arena_proposition in ArenaPropositions:
            for crash_proposition in CrashPropositions:
                self.assertEqual((arena_proposition, crash_proposition),
                                 decode_int_output(encode_int_output(arena_proposition, crash_proposition)))
