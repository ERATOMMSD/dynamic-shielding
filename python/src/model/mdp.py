from typing import List, Tuple, Dict

from src.model import ReactiveSystem

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "12 October 2020"


class MDP(ReactiveSystem):
    """
    The class for MDP with actions of player 1 and 2.
    Note: When we refactor the code, we can consider an implementation as a ReactiveSystem with probability.
    """

    def __init__(self, player1_alphabet: List[int], player2_alphabet: List[int], output_alphabet: List[int]) -> None:
        self.transition_probability: Dict[int, Dict[Tuple[int, int], float]] = {}
        super(MDP, self).__init__(player1_alphabet, player2_alphabet, output_alphabet)

    def addProbabilisticTransition(self, source: int, player1_action: int, player2_action: int, output: int,
                                   probability: float, target: int) -> None:
        """
        Adds a transition
        Args:
            source: int : the source state
            player1_action: int : the action of player 1
            player2_action: int : the action of player 2
            output: int output of the transition
            probability: the probability of the transition
            target: int : the target state
        """
        if source not in self.output:
            self.output[source], self.transitions[source], self.transition_probability[source] = {}, {}, {}
        assert output in self.outputAlphabet
        self.output[source][player1_action, player2_action] = output
        self.transitions[source][player1_action, player2_action] = target
        self.transition_probability[source][player1_action, player2_action] = probability

    def getTransitionProbability(self) -> Dict[int, Dict[Tuple[int, int], float]]:
        return self.transition_probability

    def getProbabilisticSuccessor(self, src: int, player1_action: int) -> List[Tuple[int, float, int]]:
        """
        Returns the next state
        Args:
            src: int : the source state
            player1_action: str : the action of player 1
        Returns:
            The list of tuples (player2_action, probability, nextState)
        """
        return list(map(lambda tpl: (tpl[0], self.transition_probability[src][player1_action, tpl[0]], tpl[1]),
                        ((p2Act, nextState) for ((p1Act, p2Act), nextState)
                         in self.transitions[src].items()
                         if p1Act == player1_action))) if src in self.transitions else []

    def getSuccessorWithP2Action(self, src: int, player1_action: int, player2_action: int) -> Tuple[float, int]:
        """
        Returns the next state
        Args:
            src: int : the source state
            player1_action: int : the action of player 1
            player2_action: int : the action of player 2
        Returns:
            The next state after the transition with the transition probability
        """
        return (self.transition_probability[src][player1_action, player2_action],
                self.transitions[src][player1_action, player2_action])

    def getOutput(self, src: int, player1_action: int, player2_action: int) -> int:
        """
        Returns the output of the transition
        Args:
            src: int : the source state
            player1_action: src : the action of player 1
            player2_action: src : the action of player 2
        Returns:
            The output character
        """
        return self.output[src][player1_action, player2_action]

    def __str__(self):
        r = {}
        for source in self.getStates():
            r[source] = {}
            if source in self.transitions:
                for ((action_1, action_2), (probability, target)) in self.transitions[source].items():
                    output = self.getOutput(source, action_1, action_2)
                    r[source]['({},{}, {}) / {}'.format(action_1, action_2, probability, output)] = target
        return str(r)

    def to_dot(self) -> str:
        dot = 'digraph MDP {\n'
        for src, transitions in self.transitions.items():
            for (player1_action, player2_action), (probability, tgt) in transitions.items():
                output = self.getOutput(src, player1_action, player2_action)
                dot += '    {} -> {} [label="{}, {}, {} / {}"];\n'.format(str(src), str(tgt),
                                                                          player1_action, player2_action,
                                                                          probability, output)
        dot += '}\n'
        return dot
