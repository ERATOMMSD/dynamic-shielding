import math
from typing import List, Tuple, Callable, Union

from py4j.java_gateway import JavaGateway

from src.model import ReactiveSystem
from src.shields import DynamicShield
from src.shields.abstract_dynamic_shield import ShieldLifeType, UpdateShield


class AdaptiveDynamicShield(DynamicShield):
    """
    Dynamic shield with adaptive min_depth
    """

    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway,
                 alphabet_start: int, alphabet_end: int,
                 alphabet_mapper: Callable[[int], Tuple[int, int]],
                 evaluate_output: Callable[[int], Callable[[str], bool]],
                 reverse_alphabet_mapper: Callable[[int, int], int],
                 reverse_output_mapper: Callable[[int], int],
                 max_episode_length: int,
                 update_shield: UpdateShield = UpdateShield.RESET,
                 shield_life_type: ShieldLifeType = ShieldLifeType.EPISODES,
                 concurrent_reconstruction=False, max_shield_life=100,
                 not_use_deviating_shield=False, skip_mealy_size: int = 0,
                 factor: float = 1.0, discard_min_duration: int = 20,
                 max_min_depth: int = 10):
        """
        :param ltl_formula: str : the LTL formula for the shielded specification
        :param gateway: JavaGateway : The java gateway of py4j
        :param alphabet_start: int : The beginning character of the input alphabet of the Mealy machine
        :param alphabet_end: int : The end character of the input alphabet of the Mealy machine
        :param alphabet_mapper: Callable[[str],(str, str)] : a callable that defines how to split the alphabet
        :param evaluate_output: Callable[[str], Callable[[str], bool]] : The function to evaluate the output of the
         reactive system
        :param reverse_alphabet_mapper: Callable[[str, str], str] : a function that combine the split action to
          reconstruct an action in Mealy machine
        :param reverse_output_mapper: Callable[[str], str] : a function maps the output in ReactiveSystem to the output
         of MealyMachine
        :param update_shield: UpdateShield: specify where to reconstruct the shield
        :param shield_life_type: ShieldLifeType: determines if the shield_life is measured in episodes or steps
        :param max_shield_life: int: The number of the maximum episodes/steps to refresh the learned shield.
         This is used only when concurrent_reconstruction = True
        :param not_use_deviating_shield: bool: Do not use the shield if the system behavior is not the same as the
         learned reactive system until `reset` is called.
        :param skip_mealy_size: int : We do not merge the states if the Mealy machine is smaller than this
        :param factor: We should increase this when the proposition is the same in most of the positions in the arena.
        :param discard_min_duration:

        .. NOTE::
            The constructed alphabet includes both alphabet_start and alphabet_end. For example, if alphabet_start = 1
            and alphabet_end = 3, the constructed alphabet is [1, 2, 3].
        """
        self.discard_min_duration = discard_min_duration
        self.factor = factor
        self.max_episode_length = max_episode_length
        self.max_min_depth = max_min_depth
        self.episode_lengths: List[int] = []
        self.smallest_min_depth: int = self.max_min_depth
        super(AdaptiveDynamicShield, self).__init__(ltl_formula, gateway, alphabet_start, alphabet_end, alphabet_mapper,
                                                    evaluate_output, reverse_alphabet_mapper, reverse_output_mapper,
                                                    update_shield, shield_life_type, 1, concurrent_reconstruction,
                                                    max_shield_life, not_use_deviating_shield, skip_mealy_size)

    def compute_min_depth(self) -> int:
        mean_episode_length = sum(self.episode_lengths) / len(self.episode_lengths)
        if self.max_episode_length < mean_episode_length:
            raise RuntimeError('mean_episode_length is greater than max_episode_length. This is strange')
        if mean_episode_length == 0:
            return self.max_min_depth
        else:
            return min(math.ceil(self.factor * (self.max_episode_length - mean_episode_length) / mean_episode_length),
                       self.max_min_depth)

    def reconstruct_reactive_system(self) -> ReactiveSystem:
        if len(self.episode_lengths) == 0:
            self.learner.min_depth = self.max_min_depth
        elif len(self.episode_lengths) < self.discard_min_duration:
            self.learner.min_depth = self.compute_min_depth()
        else:
            # We take the min so that we do not increase min_depth after learning a reasonable controller
            self.learner.min_depth = min(self.compute_min_depth(), self.smallest_min_depth)
            self.smallest_min_depth = self.learner.min_depth
        return super(AdaptiveDynamicShield, self).reconstruct_reactive_system()

    def reset(self) -> None:
        if len(self.history) > 0:
            self.episode_lengths.append(len(self.history))
        super(AdaptiveDynamicShield, self).reset()
