import logging
from copy import deepcopy
from logging import getLogger

from src.shields import AbstractShield
from src.shields.abstract_dynamic_shield import AbstractDynamicShield
from src.wrappers.shield_wrappers import AbstractShieldWrapper

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', level=logging.WARN)
LOGGER = getLogger(__name__)


class EvaluationShield(AbstractShield):
    """
    Shield only for the evaluation. The shield is obtained from the training environment.
    """

    def __init__(self, training_env: AbstractShieldWrapper):
        """
        Initializes the evaluation shield.
        :param training_env: The environment used for training.
        """
        self.training_env = training_env
        super().__init__(safety_game=training_env.shield.safety_game,
                         win_set=training_env.shield.win_set,
                         win_strategy=training_env.shield.win_strategy)

    def reset(self) -> None:
        """
        Reset the evaluation shield.
        """
        LOGGER.info("Reset evaluation shield.")
        self.safety_game = deepcopy(self.training_env.shield.safety_game)
        self.win_set = deepcopy(self.training_env.shield.win_set)
        self.win_strategy = deepcopy(self.training_env.shield.win_strategy)
        self.state = self.safety_game.getInitialState()

    def save_pickle(self, pickle_filename: str) -> None:
        """
        Save the reactive system as a pickle file.
        :param pickle_filename: The filename of the pickle file to save the reactive system.
        """
        LOGGER.info('Save reactive system as pickle file.')
        if isinstance(self.training_env.shield, AbstractDynamicShield):
            self.training_env.shield.save_pickle(pickle_filename)
