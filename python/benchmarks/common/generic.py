import os.path as osp
from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Callable, Tuple, List

from py4j.java_gateway import JavaGateway

from src.model import ReactiveSystem


class ShieldKind(Flag):
    Preemptive = auto()
    Postposed = auto()


JAVA_ROOT = osp.join(osp.dirname(__file__), '../../../java/')
JAR_PATH = f'{JAVA_ROOT}/target/learnlib-py4j-example-1.0-SNAPSHOT.jar'


def launch_gateway(die_on_exit: bool = True) -> JavaGateway:
    return JavaGateway.launch_gateway(jarpath=JAR_PATH, die_on_exit=die_on_exit)


def list_names(enum):
    return [enum(i).name for i in range(len(enum))]


def list_ids(enum):
    return list(range(len(enum)))


def int_to_bool_list(num: int, size: int) -> List[bool]:
    bin_string = format(num, f'0{size}b')
    return [x == '1' for x in bin_string]


def bool_list_to_int(lst: List[bool]) -> int:
    return int('0b' + ''.join(['1' if x else '0' for x in lst]), 2)


class AbstractInputOutputManager(ABC):
    """
    Generic class to define a default mapping between
    - agent_action and p2_actions to input (integer)
    - list of observations to output (integer)
    """

    def __init__(self, p1_actions, p2_actions, observations):
        """"
            p1_actions: Enum: the actions of player 1
            p2_actions: Enum: the actions of player 2
            observations: Enum: the observations of the environment
        """
        self.p1_actions = p1_actions
        self.p2_actions = p2_actions
        self.observations = observations

    def get_first_input(self):
        """"
        By default the initial alphabet is zero
        Returns:
            the initial input alphabet
        """
        return 0

    def get_last_input(self):
        """
        Returns:
            the last input alphabet
        """
        return (len(self.p1_actions) * len(self.p2_actions)) - 1

    def alphabet_mapper(self, input: int) -> Tuple[int, int]:
        """
        Args:
            input: int: input
        Returns:
            a pair p1_action, p2_action
        """
        assert self.get_first_input() <= input <= self.get_last_input()
        return input % len(self.p1_actions), input // len(self.p1_actions)

    def reverse_alphabet_mapper(self, player1_action: int, player2_action: int) -> int:
        """
        Args:
            p1_action: int: an action of player 1
            p2_action: int: an action of player 2
        Returns:
            the input value that maps to this pair of actions
        """
        assert 0 <= player1_action < len(self.p1_actions)
        assert 0 <= player2_action < len(self.p2_actions)
        return player1_action + player2_action * len(self.p1_actions)

    def get_p1_actions(self):
        """
        Returns:
            the list of valid actions of player 1
        """
        return list_ids(self.p1_actions)

    def get_p2_actions(self):
        """
        Returns:
            the list of valid actions of player 2
        """
        return list_ids(self.p2_actions)

    @abstractmethod
    def get_outputs(self) -> List[int]:
        """
        Returns:
            the list of valid outputs
        """
        raise NotImplementedError

    @abstractmethod
    def reverse_output_mapper(self, output: int) -> int:
        """
        Args:
            output: int: output value in the system.
        Returns:
            the output value in the environment model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_output(self, output: int) -> Callable[[str], bool]:
        """
        Only one observation is enabled at a time.
        Args:
            output: int: output value
        Returns:
            a callable mapping observations to boolean values
        """
        raise NotImplementedError


class SimpleIOManager(AbstractInputOutputManager):

    def evaluate_output(self, output: int) -> Callable[[str], bool]:
        """
        Only one observation is enabled at a time.
        Args:
            output: int: output value
        Returns:
            a callable mapping observations to boolean values
        """
        assert 0 <= output < len(self.observations)
        return lambda out: self.observations(output).name == out

    def get_outputs(self) -> List[int]:
        """
        Returns:
            the list of valid outputs
        """
        return list_ids(self.observations)

    def reverse_output_mapper(self, output: int) -> int:
        """
        Args:
            output: int: output value
        Returns:
            same output value (assuming that the output values of the environment are same)
        """
        assert 0 <= output < len(self.observations)
        return output


class BooleanIOManager(AbstractInputOutputManager):

    def get_output_id_from_valuation(self, valuation) -> int:
        """"
        Args:
            valuation: List[bool]: a list of values True/False for each observation
        Returns:
            the integer value that represents that valuation using binary representation
        """
        assert len(valuation) == len(self.observations)
        return bool_list_to_int(valuation)

    def evaluate_output(self, output: int) -> Callable[[str], bool]:
        """
        Args:
            output: int: output value
        Returns:
            a callable mapping observations to boolean values
        """
        assert 0 <= output < 2 ** len(self.observations)
        return lambda out: dict(zip(list_names(self.observations), int_to_bool_list(output, len(self.observations))))[
            out]

    def observation_list_from_output(self, output: int) -> List[str]:
        """
        Args:
            output: int: output value
        Returns:
            a list observations as strings
        """
        valuation = self.evaluate_output(output)
        obs = [o for o in list_names(self.observations) if valuation(o)]
        return obs

    def get_outputs(self) -> List[int]:
        """
        Returns:
            the list of valid outputs
        """
        return list(range(2 ** len(self.observations)))

    def reverse_output_mapper(self, output: int) -> int:
        """
        Args:
            output: int: output value
        Returns:
            same output value (assuming that the output values of the environment are same)
        """
        assert 0 <= output < 2 ** len(self.observations)
        return output

    def alphabet_mapper(self, input: int) -> Tuple[int, int]:
        """
        Args:
            input: int: input
        Returns:
            a pair p1_action, p2_action
        """
        assert self.get_first_input() <= input <= self.get_last_input()
        return input % len(self.p1_actions), input // len(self.p1_actions)

    def reverse_alphabet_mapper(self, player1_action: int, player2_action: int) -> int:
        """
        Args:
            p1_action: int: an action of player 1
            p2_action: int: an action of player 2
        Returns:
            the input value that maps to this pair of actions
        """
        assert 0 <= player1_action < len(self.p1_actions)
        assert 0 <= player2_action < len(self.p2_actions)
        return player1_action + player2_action * len(self.p1_actions)

    def get_outputs(self) -> List[int]:
        """
        Returns:
            the list of valid outputs
        """
        return list(range(2 ** len(self.observations)))

    def get_p1_actions(self):
        """
        Returns:
            the list of valid actions of player 1
        """
        return list_ids(self.p1_actions)

    def get_p2_actions(self):
        """
        Returns:
            the list of valid actions of player 2
        """
        return list_ids(self.p2_actions)


def reactive_system_from_deterministic_environment(env, io_manager: AbstractInputOutputManager):

    reactive_system = ReactiveSystem(io_manager.get_p1_actions(),
                                     io_manager.get_p2_actions(),
                                     io_manager.get_outputs())

    for source in env.P:
        if hasattr(env, 'get_p2_action'):
            p2_action = env.get_p2_action(source).value
        else:
            assert len(io_manager.get_p2_actions()) == 1
            p2_action = io_manager.get_p2_actions()[0]
        for p1_action in io_manager.get_p1_actions():
            # we get new state from pos 1 (prob, new_state, reward, done)
            target = env.P[source][p1_action][0][1]
            output = env.output(source, p1_action, target)
            reactive_system.addTransition(source, p1_action, p2_action, output, target)
    reactive_system.setInitialState(env.s)
    return reactive_system
