import unittest
from typing import Callable

from src.model import ReactiveSystem
from src.shields import StaticShield


class TestStaticShield(unittest.TestCase):
    def makeSimpleSystem(self) -> ReactiveSystem:
        reactive_system = ReactiveSystem([1, 2], [1, 2], [10, 11])
        # make initial state
        reactive_system.setInitialState(1)
        # make transitions
        reactive_system.addTransition(1, 1, 1, 11, 2)
        reactive_system.addTransition(1, 1, 2, 11, 1)
        reactive_system.addTransition(1, 2, 1, 11, 1)
        reactive_system.addTransition(1, 2, 2, 11, 3)

        reactive_system.addTransition(2, 1, 1, 11, 4)
        reactive_system.addTransition(2, 1, 2, 11, 2)
        reactive_system.addTransition(2, 2, 1, 11, 2)
        reactive_system.addTransition(2, 2, 2, 11, 1)

        reactive_system.addTransition(3, 1, 1, 10, 4)
        reactive_system.addTransition(3, 1, 2, 10, 2)
        reactive_system.addTransition(3, 2, 1, 10, 2)
        reactive_system.addTransition(3, 2, 2, 10, 1)

        reactive_system.addTransition(4, 1, 1, 10, 2)
        reactive_system.addTransition(4, 1, 2, 10, 1)
        reactive_system.addTransition(4, 2, 1, 10, 1)
        reactive_system.addTransition(4, 2, 2, 10, 3)
        return reactive_system

    def test_tickShielding_Gp(self):
        ltl_formula = 'G(p)'
        reactive_system = self.makeSimpleSystem()

        def evaluate_output(s: int) -> Callable[[str], bool]:
            if s == 10:
                return lambda _: False
            else:
                return lambda _: True

        static_shield = StaticShield(ltl_formula, reactive_system, evaluate_output)

        # 1 is allowed at state 1
        self.assertEqual(1, static_shield.state)
        self.assertEqual(1, static_shield.tick(1))
        # go to state 2
        static_shield.move(1, 1, 11)
        self.assertEqual(2, static_shield.state)
        # 1 is NOT allowed at state 2
        self.assertEqual(2, static_shield.tick(1))
        # stay at state 2
        static_shield.move(2, 1, 11)
        self.assertEqual(2, static_shield.state)
        # 2 is allowed at state 2
        self.assertEqual(2, static_shield.tick(2))
        # go back to state 1
        static_shield.move(2, 2, 11)
        self.assertEqual(1, static_shield.state)
        # 2 is NOT allowed at state 1
        self.assertEqual(1, static_shield.tick(2))

    def test_tickShielding_GnegpXp(self):
        ltl_formula = 'G(!p => X(p))'
        reactive_system = self.makeSimpleSystem()

        def evaluate_output(s: int) -> Callable[[str], bool]:
            if s == 10:
                return lambda _: False
            else:
                return lambda _: True

        static_shield = StaticShield(ltl_formula, reactive_system, evaluate_output)

        # 1 is allowed at state 1
        self.assertEqual(1, static_shield.state)
        self.assertEqual(1, static_shield.tick(1))
        # go to state 2
        static_shield.move(1, 1, 11)
        self.assertEqual(2, static_shield.state)
        # 1 is allowed at the second state
        self.assertEqual(1, static_shield.tick(1))
        # go to the initial state with one !p
        static_shield.move(1, 1, 11)
        self.assertEqual(4, static_shield.state)
        # 2 is allowed at this state
        self.assertEqual(1, static_shield.tick(2))

    def test_reset_Gp(self):
        ltl_formula = 'G(p)'
        reactive_system = self.makeSimpleSystem()

        def evaluate_output(s: int) -> Callable[[str], bool]:
            if s == 10:
                return lambda _: False
            else:
                return lambda _: True

        static_shield = StaticShield(ltl_formula, reactive_system, evaluate_output)

        self.assertEqual(1, static_shield.state)
        # go to state 2
        static_shield.move(1, 1, 11)
        self.assertEqual(2, static_shield.state)
        # reset and go back to state 1
        static_shield.reset()
        self.assertEqual(1, static_shield.state)
        # reseting at state 1 does not change anything
        static_shield.reset()
        self.assertEqual(1, static_shield.state)

    def test_reset_GnegpXp(self):
        ltl_formula = 'G(!p => X(p))'
        reactive_system = self.makeSimpleSystem()

        def evaluate_output(s: int) -> Callable[[str], bool]:
            if s == 10:
                return lambda _: False
            else:
                return lambda _: True

        static_shield = StaticShield(ltl_formula, reactive_system, evaluate_output)

        self.assertEqual(1, static_shield.state)
        # go to state 2
        static_shield.move(1, 1, 11)
        self.assertEqual(2, static_shield.state)
        # go to state 4
        static_shield.move(1, 1, 11)
        self.assertEqual(4, static_shield.state)
        # reset and go back to state 1
        static_shield.reset()
        self.assertEqual(1, static_shield.state)

    def test_preemptiveShielding_Gp(self):
        ltl_formula = 'G(p)'
        reactive_system = self.makeSimpleSystem()

        def evaluate_output(s: int) -> Callable[[str], bool]:
            if s == 10:
                return lambda _: False
            else:
                return lambda _: True

        static_shield = StaticShield(ltl_formula, reactive_system, evaluate_output)

        # only 1 is allowed at state 1
        self.assertEqual(1, static_shield.state)
        self.assertEqual([1], static_shield.preemptive())
        # go to state 2
        static_shield.move(1, 1, 11)
        self.assertEqual(2, static_shield.state)
        # only 2 is allowed at state 2
        self.assertEqual([2], static_shield.preemptive())

    def test_preemptiveShielding_GnegpXp(self):
        ltl_formula = 'G(!p => X(p))'
        reactive_system = self.makeSimpleSystem()

        def evaluate_output(s: int) -> Callable[[str], bool]:
            if s == 10:
                return lambda _: False
            else:
                return lambda _: True

        static_shield = StaticShield(ltl_formula, reactive_system, evaluate_output)

        # Both 1 and 2 are allowed at state 1
        self.assertEqual(1, static_shield.state)
        self.assertEqual([1, 2], static_shield.preemptive())
        # go to state 2
        static_shield.move(1, 1, 11)
        self.assertEqual(2, static_shield.state)
        # Both 1 and 2 are allowed at state 2
        self.assertEqual([1, 2], static_shield.preemptive())
        # go to state 4
        static_shield.move(1, 1, 11)
        self.assertEqual(4, static_shield.state)
        # only 1 is allowed at this state
        self.assertEqual([1], static_shield.preemptive())


if __name__ == '__main__':
    unittest.main()
