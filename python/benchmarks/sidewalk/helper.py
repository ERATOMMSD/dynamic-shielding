import argparse
import logging
import os
import os.path as osp
import shutil
import time

# Constants
TASK_STRING: str = 'RangedDeterministicSidewalk-DomainRand-v0'
EVAL_TASK_STRING: str = TASK_STRING
NUM_TIMESTEP: int = 100 * 1000
LEARNING_RATE: float = 0.00005
GAMMA: float = 0.99
MIN_DEPTH: int = 1
# Note: MAX_SHIELD_LIFE should not be too large. The total number of episodes of sidewalk is about 1,000--2,000.
MAX_SHIELD_LIFE = 75
# Note: In sidewalk, the output is almost always the same.
#       Therefore, min_depth must be a bit larger than the other benchmarks.
FACTOR: float = 1.0


def configure_logger(log_path: str) -> None:
    """
    Configure the logger
    :param log_path: the path to the directory to save the logs
    """
    if osp.isdir(log_path):
        shutil.move(log_path, f'{log_path[:-1]}-{time.time()}/')
    os.mkdir(log_path)
    logging.basicConfig(format='%(asctime)s %(module)s[%(lineno)d] [%(levelname)s]: %(message)s',
                        filename=osp.join(log_path, 'logger.log'), level=logging.DEBUG)
    # Disable the logging of py4j
    py4j_logger = logging.getLogger("py4j.java_gateway")
    py4j_logger.setLevel(logging.INFO)


def get_default_parser() -> argparse.ArgumentParser:
    """
    Define the parser of the arguments
    """
    parser = argparse.ArgumentParser(description='evaluate dynamic shielding with side_walk benchmarks')
    parser.add_argument('--shield', type=str, default='all',
                        help='the shields to use [all (default) | dynamic | no]')
    parser.add_argument('--shield_kind', type=str, default='preemptive',
                        help='the shields to use [all | preemptive (default) | postposed]')
    return parser
