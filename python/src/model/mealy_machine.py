from typing import List

from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JJavaError

from src.exceptions.shielding_exceptions import UnknownOutputError

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "06 August 2020"


class MealyMachine:
    """
    The Mealy Machine class wrapping the CompactMealy in automatalib (https://learnlib.de/projects/automatalib/) in Java. Mealy machine is constructed by PassiveLearning.computeMealy.
    Note:
        This class assumes that the LearnLib JVM gateway is running.
    """

    def __init__(self, gateway: JavaGateway, mealy) -> None:
        """
        The constructor
        Args:
            gateway: JavaGateway : The java gateway of py4j
            mealy: The raw Mealy machine of LearnLib. 
        Note:
            We can construct gateway by the following.
            gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
        """
        self.gateway = gateway
        self.mealy = mealy
        self.inputAlphabet: List[int] = list(self.mealy.getInputAlphabet())

    def getStates(self) -> List[int]:
        """
        Returns the states of the Mealy machine
        Returns:
            The list of the states represented by integers
        """
        return self.mealy.getStates()

    def getInitialState(self) -> int:
        """
        Returns the initial state of the Mealy machine
        Returns:
            The initial state represented by an integer
        """
        return self.mealy.getInitialState()

    def getStateId(self, state: int) -> int:
        """
        Translate the integer-valued state to the ID of LearnLib.
        Args:
            state: int : integer-valued state
        Returns:
            ID of LearnLib representing the state.
        """
        return self.mealy.getStateId(state)

    def getInputAlphabet(self) -> List[int]:
        """
        Returns the input alpabet
        Returns:
            The list of strings representing the input alphabet
        """
        return self.inputAlphabet

    def getCharId(self, c: int) -> int:
        """
        Translate the character in the input alphabet to the ID integer.
        Args:
            c: int : integer in the input alphabet
        Return:
            ID of LearnLib representing the character.
        """
        return self.inputAlphabet.index(c)

    def getOutputAlphabet(self) -> List[int]:
        """
        Returns the output alphabet
        Returns:
            The list of strings representing the output alphabet
        """
        output: List[int] = []
        for state in self.getStates():
            for input_action in self.getInputAlphabet():
                try:
                    output.append(self.getOutput(state, input_action))
                except UnknownOutputError:
                    pass  # The output of a transition might be unknown.

        return list(set(output))

    def getSuccessor(self, src: int, c: int) -> int:
        """
        Returns the next state
        Args:
            src: int : the source state
            c: src : the input character for the transition
        Returns:
            The next state after the transition
        """
        transition = self.mealy.getTransition(self.getStateId(src), self.getCharId(c))
        return self.mealy.getSuccessor(transition)

    def getOutput(self, src: int, c: int) -> int:
        """
        Returns the output of the transition
        Args:
            src: int : the source state
            c: src : the input character for the transition
        Returns:
            The output character of the transition
        """
        try:
            transition = self.mealy.getTransition(self.getStateId(src), self.getCharId(c))
            output = self.mealy.getTransitionOutput(transition)
            if output is None:
                raise UnknownOutputError
        except Py4JJavaError:
            raise UnknownOutputError
        return output

    def getDot(self) -> str:
        """
        Return:
            the DOT representation of the Mealy machine
        """
        # Construct a buffer that we will use to print results on the Python side of our setup
        string_writer = self.gateway.jvm.java.io.StringWriter()

        # Serialize the hypothesis to the DOT format and write it to the string_writer
        self.gateway.jvm.net.automatalib.serialization.dot.GraphDOT.write(self.mealy,
                                                                          self.mealy.getInputAlphabet(),
                                                                          string_writer)
        return string_writer.toString()
