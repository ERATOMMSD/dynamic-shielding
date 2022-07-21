from copy import deepcopy
from logging import getLogger
from typing import Set, List, Dict, Tuple, Optional, Generator

from py4j.protocol import Py4JJavaError
from tryalgo.partition_refinement import PartitionRefinement

from src.exceptions.shielding_exceptions import UnknownOutputError
from src.model import MealyMachine

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "14 September 2021"

LOGGER = getLogger(__name__)


class ReduceTrainingData:
    """
    Reduce training data using the separating suffixes
    """
    mealy: MealyMachine
    partition: PartitionRefinement
    reversed_transitions: Dict[Tuple[int, int], Set[int]]

    def __init__(self, mealy: MealyMachine):
        self.mealy = mealy
        # This initialization works because list(range(len(mealy.getStates())) == list(mealy.getStates())
        self.partition = PartitionRefinement(len(self.mealy.getStates()))
        self.reversed_transitions = dict()
        for source in self.mealy.getStates():
            for action in self.mealy.getInputAlphabet():
                try:
                    target = self.mealy.getSuccessor(source, action)
                    if (target, action) in self.reversed_transitions:
                        self.reversed_transitions[target, action].add(source)
                    else:
                        self.reversed_transitions[target, action] = {source}
                except (ValueError, Py4JJavaError, UnknownOutputError):
                    pass

    def filter_redundant_samples(self, samples: List[Tuple[List[int], int]]) -> \
            Generator[Tuple[List[int], int], None, None]:
        separation_sequences = self.make_separation_sequences()
        enhanced_separation_sequences = deepcopy(separation_sequences)
        for source in self.mealy.getStates():
            for action in self.mealy.getInputAlphabet():
                try:
                    target = self.mealy.getSuccessor(source, action)
                    enhanced_separation_sequences[source] += [[action] + separation_sequence for separation_sequence in
                                                              separation_sequences[target]]
                except (ValueError, Py4JJavaError, UnknownOutputError):
                    pass
        unvisited_successors: List[Set[int]] = [set() for _ in self.mealy.getStates()]
        for source in range(len(self.mealy.getStates())):
            for action in self.mealy.getInputAlphabet():
                try:
                    self.mealy.getSuccessor(source, action)
                    unvisited_successors[source].add(action)
                except (ValueError, Py4JJavaError, UnknownOutputError):
                    pass
        for (input_word, output_char) in samples:
            state = self.mealy.getInitialState()
            for i in range(len(input_word)):
                if input_word[i:] in enhanced_separation_sequences[state]:
                    enhanced_separation_sequences[state].remove(input_word[i:])
                    for action in input_word[i:-1]:
                        state = self.mealy.getSuccessor(state, action)
                    unvisited_successors[state].discard(input_word[-1])
                    yield input_word, output_char
                    break
                if i == len(input_word) - 1:
                    if input_word[-1] in unvisited_successors[state]:
                        unvisited_successors[state].remove(input_word[-1])
                        yield input_word, output_char
                try:
                    state = self.mealy.getSuccessor(state, input_word[i])
                except (ValueError, Py4JJavaError, UnknownOutputError):
                    pass

    def make_separation_sequences(self) -> List[List[List[int]]]:
        """
        Compute the separation sequences with a variant of Hopcroft's algorithm
        """
        queue: List[Tuple[int, Set[int]]] = []
        initial_partition, separating_sequences = self.make_initial_partition(self.mealy)
        for pivot in initial_partition:
            self.partition.refine(pivot)
            queue += [(action, pivot) for action in self.mealy.inputAlphabet]
        while len(queue) > 0:
            action, pivot = queue.pop()
            predecessors = self.reversed_transitions_set(pivot, action)
            prev_partition_list = [set(part_list) for part_list in self.partition.tolist()]
            self.partition.refine(predecessors)
            partition_list = [set(part_list) for part_list in self.partition.tolist()]
            if len(partition_list) != len(prev_partition_list):
                # refined
                for predecessor in predecessors:
                    previous_partition = \
                        next((partition for partition in prev_partition_list if predecessor in partition), None)
                    current_partition = \
                        next((partition for partition in partition_list if predecessor in partition), None)
                    if len(previous_partition) != len(current_partition):
                        # state is refined
                        for state in pivot:
                            for separating_sequence in separating_sequences[state]:
                                separating_sequences[predecessor].append([action] + separating_sequence)
                        for action in self.mealy.getInputAlphabet():
                            if (action, current_partition) not in queue:
                                queue.append((action, current_partition))

        def get_unique_list(seq):
            seen = []
            return [x for x in seq if x not in seen and not seen.append(x)]

        return [get_unique_list(separating_sequence) for separating_sequence in separating_sequences]

    def reversed_transitions_set(self, targets: Set[int], action: int) -> Set[int]:
        sources: Set[int] = set()
        for target in targets:
            if (target, action) in self.reversed_transitions:
                sources.union(self.reversed_transitions[target, action])
        return sources

    @staticmethod
    def make_initial_partition(mealy: MealyMachine) -> Tuple[List[Set[int]], List[List[List[int]]]]:
        """
        Construct the initial partition based on the labeling function
        """
        remaining_states: List[int] = list(mealy.getStates())
        result: List[Set[int]] = []
        separating_actions: List[Set[int]] = [set() for _ in mealy.getStates()]

        def find_inconsistent_label(s1: int, s2: int) -> Optional[int]:
            for action in mealy.getInputAlphabet():
                try:
                    if mealy.getOutput(s1, action) != mealy.getOutput(s2, action):
                        return action
                except (ValueError, Py4JJavaError, UnknownOutputError):
                    pass
            return None

        while len(remaining_states) > 0:
            state = remaining_states.pop()
            found = False
            for i in range(len(result)):
                inconsistent_label = find_inconsistent_label(state, next(iter(result[i])))
                if inconsistent_label is None:
                    result[i].add(state)
                    found = True
                    break
                separating_actions[state].add(inconsistent_label)
                for another_state in result[i]:
                    separating_actions[another_state].add(inconsistent_label)
            if not found:
                result.append({state})
        separating_sequences: List[List[List[int]]] = \
            [[[action] for action in action_set] for action_set in separating_actions]
        return result, separating_sequences
