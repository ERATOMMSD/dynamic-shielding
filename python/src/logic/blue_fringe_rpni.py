from copy import deepcopy
from logging import getLogger
from typing import List, Tuple, Optional, Set, Dict

from src.model import PTA
from src.model import ReactiveSystem

LOGGER = getLogger(__name__)


class CheckMergeability:
    def __init__(self, reactive_system: ReactiveSystem) -> None:
        self._reactive_system = reactive_system
        self._compatibility_memo: Dict[Tuple[int, int], bool] = dict()
        self._mergeable_depth_memo: Dict[Tuple[int, int], Optional[int]] = dict()

    def output_compatible(self, red_state: int, blue_state: int) -> bool:
        if red_state == blue_state:
            return True
        if (red_state, blue_state) in self._compatibility_memo:
            return self._compatibility_memo[red_state, blue_state]
        if red_state not in self._reactive_system.output or blue_state not in self._reactive_system.output:
            self._compatibility_memo[red_state, blue_state] = True
            return True
        result = all(self._reactive_system.output[red_state][action] == self._reactive_system.output[blue_state][action]
                     for action in
                     self._reactive_system.output[red_state].keys() & self._reactive_system.output[blue_state].keys())
        self._compatibility_memo[red_state, blue_state] = result
        return result

    def mergeable_depth(self, red_state: int, blue_state: int,
                        visited_pair: Optional[Set[Tuple[int, int]]] = None) -> Optional[int]:
        """
            Decide if blue_state is mergeable to red_state and returns the depth of the matching if mergeable
            Args
                _reactive_system:
                red_state:
                blue_state:
                visited_pair:
            :returns
                None if blue_state is not mergeable to red_state
                otherwise, returns the depth of the matched subgraph
        """
        if visited_pair is None:
            visited_pair = set()
        if (red_state, blue_state) in self._mergeable_depth_memo:
            return self._mergeable_depth_memo[red_state, blue_state]
        if red_state == blue_state or (red_state, blue_state) in visited_pair:
            self._mergeable_depth_memo[red_state, blue_state] = 1
            return 1
        visited_pair.add((red_state, blue_state))
        visited_pair.add((blue_state, red_state))
        # Returns None if we cannot merge red_state and blue_state
        if not self.output_compatible(red_state, blue_state):
            self._mergeable_depth_memo[red_state, blue_state] = None
            return None
        if red_state not in self._reactive_system.transitions or blue_state not in self._reactive_system.transitions:
            self._mergeable_depth_memo[red_state, blue_state] = 1
            return 1
        max_depth: int = 1
        for action in self._reactive_system.transitions[red_state].keys() & \
                      self._reactive_system.transitions[blue_state].keys():
            depth: Optional[int]
            depth = self.mergeable_depth(self._reactive_system.transitions[red_state][action],
                                         self._reactive_system.transitions[blue_state][action],
                                         visited_pair)
            if depth is None:
                self._mergeable_depth_memo[red_state, blue_state] = None
                return None
            max_depth = max(max_depth, depth + 1)
        self._mergeable_depth_memo[red_state, blue_state] = max_depth
        return max_depth


def _merge(reactive_system: ReactiveSystem, red_state: int, blue_state: int,
           visited_blue_states=None) -> None:
    """
        Merge blue_state to red_state in _reactive_system
        Note: we do not check if blue_state is mergeable to red_state. please check by _mergeable_depth beforehand
    """
    if visited_blue_states is None:
        visited_blue_states = set()
    # mergeable_depth = _mergeable_depth(_reactive_system, red_state, blue_state)
    # if mergeable_depth is None or red_state == blue_state:
    #    return
    # merge the outputs
    if blue_state in reactive_system.output:
        for action in reactive_system.output[blue_state]:
            if red_state in reactive_system.output:
                reactive_system.output[red_state][action] = reactive_system.output[blue_state][action]
            else:
                reactive_system.output[red_state] = {action: reactive_system.output[blue_state][action]}
        del reactive_system.output[blue_state]

    # merge the transitions
    if blue_state in reactive_system.transitions:
        if blue_state not in visited_blue_states:
            visited_blue_states.add(blue_state)
            last_available_actions = set()
            while last_available_actions != set(reactive_system.transitions[blue_state].keys()):
                last_available_actions = set(reactive_system.transitions[blue_state].keys())
                for action in last_available_actions:
                    if red_state not in reactive_system.transitions:
                        reactive_system.transitions[red_state] = dict()
                    if action in reactive_system.transitions[red_state]:
                        _merge(reactive_system, reactive_system.transitions[red_state][action],
                               reactive_system.transitions[blue_state][action], visited_blue_states)
                    else:
                        reactive_system.transitions[red_state][action] = reactive_system.transitions[blue_state][action]
            del reactive_system.transitions[blue_state]
    for source in reactive_system.transitions:
        for action in reactive_system.transitions[source]:
            if reactive_system.transitions[source][action] == blue_state:
                reactive_system.transitions[source][action] = red_state


class BlueFringeRPNI:
    def __init__(self, player1_alphabet: List[int], player2_alphabet: List[int], output_alphabet: List[int]) -> None:
        self.pta: PTA = PTA(player1_alphabet, player2_alphabet, output_alphabet)

    def addSample(self, input_word: List[Tuple[int, int]], output_action: int) -> None:
        self.pta.addSample(input_word, output_action)

    def compute_model(self, min_depth: int = 0) -> ReactiveSystem:
        reactive_system: ReactiveSystem = deepcopy(self.pta)
        blue_states: List[int] = reactive_system.getStates()
        red_states: List[int] = []
        while len(blue_states) > 0:
            # pick from the root
            blue_state: int = blue_states.pop(0)
            mergeable_states: List[Tuple[int, int]] = []
            mergeability_checker = CheckMergeability(reactive_system)
            # Perhaps this exhaustive comparison is slow
            for red_state in red_states:
                depth = mergeability_checker.mergeable_depth(red_state, blue_state)
                if depth is not None and depth >= min_depth:
                    mergeable_states.append((depth, red_state))
                    break  # Now, we do not try exhaustively
            if len(mergeable_states) > 0:
                _, red_state = sorted(mergeable_states, reverse=True)[0]
                visited_blue_states = set()
                _merge(reactive_system, red_state, blue_state, visited_blue_states)
                blue_states = [blue_state for blue_state in blue_states if blue_state not in visited_blue_states]
            else:
                red_states.append(blue_state)
                continue

        return reactive_system
