import threading
from copy import deepcopy
from logging import getLogger
from typing import List, Tuple, Union

from py4j.java_gateway import JavaGateway

from src.model import MealyMachine

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.2"
__date__ = "11 August 2020"

LOGGER = getLogger(__name__)


class PassiveLearning:
    min_depth: int
    skip_mealy_size: int

    def __init__(self, gateway: JavaGateway, alphabet_start: Union[str, int], alphabet_end: Union[str, int],
                 min_depth: int = 0, skip_mealy_size: int = 0) -> None:
        """
        The class for passive Mealy machine learning using LearnLib (https://learnlib.de/projects/automatalib/).

        :param gateway: JavaGateway : The java gateway of py4j
        :param alphabet_start: Union[str, int] : The beginning character of the input alphabet of the Mealy machine
        :param alphabet_end: Union[str, int] : The end character of the input alphabet of the Mealy machine
        :param skip_mealy_size: int : We do not merge the states if the Mealy machine is smaller than this
        :param min_depth: int : We do not merge the states if there is not common children of at least this depth

        .. NOTE::
            This class assumes that the LearnLib JVM gateway is running. We can construct gateway by the following.
            gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
        .. NOTE::
            The constructed alphabet includes both alphabet_start and alphabet_end. For example, if alphabet_start = 1
            and alphabet_end = 3, the constructed alphabet is [1, 2, 3].
        """
        self.gateway = gateway
        self.alphabet = self._construct_alphabet(alphabet_start, alphabet_end)
        self.__min_depth = min_depth
        self.skip_mealy_size = skip_mealy_size
        # Lock for the mutual exclusion in the access to Java
        self.lock = threading.Lock()
        # List of the samples that is not added to the learner due to the lock
        self.sample_pool = []
        if self.min_depth <= 0:
            if self.min_depth < 0:
                LOGGER.warning(f"negative min_depth is given. We let min_depth = 0: min_depth = {self.min_depth}")
            self.learner = self.gateway.jvm.de.learnlib.algorithms.rpni.BlueFringeRPNIMealy(self.alphabet)
        else:
            self.learner = self.gateway.jvm.org.group_mmm.StrongBlueFringeRPNIMealy(self.alphabet, self.min_depth,
                                                                                    self.skip_mealy_size)

    def reset(self, alphabet_start: Union[str, int], alphabet_end: Union[str, int]) -> None:
        """
        Reset the learning information

        :param alphabet_start: Union[str, int] : The beginning character of the input alphabet of the Mealy machine
        :param alphabet_end: Union[str, int] : The end character of the input alphabet of the Mealy machine
        """
        self.alphabet = self._construct_alphabet(alphabet_start, alphabet_end)
        if self.min_depth <= 0:
            if self.min_depth < 0:
                LOGGER.warning(f"negative min_depth is given. We let min_depth = 0: min_depth = {self.min_depth}")
            self.learner = self.gateway.jvm.de.learnlib.algorithms.rpni.BlueFringeRPNIMealy(self.alphabet)
        else:
            self.learner = self.gateway.jvm.org.group_mmm.StrongBlueFringeRPNIMealy(self.alphabet, self.min_depth,
                                                                                    self.skip_mealy_size)

    def _construct_alphabet(self, alphabet_start: Union[str, int], alphabet_end: Union[str, int]):
        assert type(alphabet_start) == type(alphabet_end), 'Inconsistent start and end type of the alphabet'
        assert type(alphabet_start) == str or type(alphabet_start) == int, 'Alphabet type must be str or int'
        if type(alphabet_start) == str:
            alphabet_start: int = ord(alphabet_start)
            alphabet_end: int = ord(alphabet_end)
        return self.gateway.jvm.net.automatalib.words.impl.Alphabets.integers(alphabet_start, alphabet_end)

    def addSample(self, input_word: Union[str, List[int]], output_char: Union[str, int]) -> None:
        """
        Add a training pair

        :param input_word: str : an input word
        :param output_char: str : the expected output word for input_word
        """
        self.lock.acquire()
        self.sample_pool.append((input_word, output_char))
        self.lock.release()

    def addSamples(self, training_data: Union[List[Tuple[str, str]], List[Tuple[List[int], int]]]) -> None:
        """
        Add training data

        :param training_data: Union[List[Tuple[str, str]], List[Tuple[List[int], List[int]]]] : an training data.
          A list of pairs of input and output words
        """
        for (inputWord, outputChar) in training_data:
            self.addSample(inputWord, outputChar)

    def getSamples(self) -> List[Tuple[List[int], int]]:
        if self.min_depth <= 0:
            raise NotImplementedError('getSamples is implemented only for StrongBlueFringeRPNIMealy')
        return [(list(java_pair.getFirst()), java_pair.getSecond()[0]) for java_pair in self.learner.getSamples()]

    def computeMealy(self) -> MealyMachine:
        """
        Constructs a Mealy machine from the current training data
        Warning: Given the same input, the computed mealy machine might be different, because the construction of the
        mealy machine is done in parallel, which may produce different results. To make the result deterministic, the
        parameter deterministic shall be True. However, this may impact the performance of the algorithm.
        For more info check: https://github.com/LearnLib/learnlib/blob/develop/algorithms/passive/rpni/src/main/java/de/learnlib/algorithms/rpni/AbstractBlueFringeRPNI.java
        Returns:
            The constructed Mealy machine
        """
        self.lock.acquire()
        LOGGER.debug('started computeModel by LearnLib')
        sample_pool = deepcopy(self.sample_pool)
        self.sample_pool.clear()
        self.lock.release()
        for input_sample, output_sample in sample_pool:
            java_input_word = self.gateway.jvm.net.automatalib.words.Word.epsilon()
            if type(input_sample) == str:
                for elem in input_sample:
                    java_input_word = java_input_word.append(ord(elem))
            else:
                for elem in input_sample:
                    java_input_word = java_input_word.append(int(elem))
            if type(output_sample) == str:
                java_output: int = ord(output_sample)
            else:
                java_output: int = int(output_sample)
            self.learner.addSample(java_input_word,
                                   self.gateway.jvm.net.automatalib.words.Word.fromLetter(java_output))
        mealy = MealyMachine(self.gateway, self.learner.computeModel())
        return mealy

    @property
    def min_depth(self) -> int:
        return self.__min_depth

    @min_depth.setter
    def min_depth(self, min_depth: int):
        if self.__min_depth != min_depth:
            self.__min_depth = min_depth
            self.learner.setMin_depth(min_depth)
