from copy import deepcopy
from logging import getLogger
from typing import Set, List, Dict, Tuple

from py4j.protocol import Py4JJavaError

from src.exceptions.shielding_exceptions import UnknownOutputError
from src.model import MealyMachine

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "21 July 2021"

LOGGER = getLogger(__name__)


def make_transition_cover(mealy: MealyMachine) -> List[Tuple[List[int], int]]:
    """
    Function to construct a set of inputs/output to cover the transitions of the given Mealy machine.
    """
    accessors: Dict[int, List[int]] = {}
    result: List[Tuple[List[int], int]] = []
    visited_states: Set[int] = set()
    new_states: List[int] = [mealy.getInitialState()]
    accessors[mealy.getInitialState()] = []
    while len(new_states) > 0:
        new_state = new_states.pop()
        if new_state in visited_states:
            continue
        visited_states.add(new_state)
        accessor: List[int] = accessors[new_state]
        for action in mealy.getInputAlphabet():
            try:
                tmp = deepcopy(accessor)
                tmp.append(action)
                output = mealy.getOutput(new_state, action)
                result.append((deepcopy(tmp), output))
                successor = mealy.getSuccessor(new_state, action)
                if successor not in visited_states:
                    accessors[successor] = tmp
                    new_states.append(successor)
            except (Py4JJavaError, UnknownOutputError):
                pass
    return result
