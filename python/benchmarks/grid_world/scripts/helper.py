import argparse
import logging
import os
import os.path as osp
import shutil
import time

TASK_STRING: str = 'grid_world2-v1'
MAX_SHIELD_LIFE = 200
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


def get_default_parser():
    parser = argparse.ArgumentParser(description='Run the experiment on grid world')
    parser.add_argument('--task_string', type=str, default='grid_world2-v1',
                        help='the string repesentation of the task')
    parser.add_argument('--max_step', type=int, default=100,
                        help='the maximum step in one episode')
    parser.add_argument('--total_steps', type=int, default=int(1e5),
                        help='the number of steps in this experiment')
    parser.add_argument('--num', type=int, default=3,
                        help='the number to distinguish this execution')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the learning rate')
    parser.add_argument('--gamma', type=float, default=0.8, # gamma=0.9 or 0.7 may also be an option for grid_world3-v2
                        help='gamma')
    return parser
