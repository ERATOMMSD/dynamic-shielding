import argparse
import logging
import os
import os.path as osp
import shutil
import time

TASK_STRING = 'discrete_car_racing-v4'
LEARNING_RATE = 3e-4
GAMMA = 0.99
TOTAL_STEPS = int(200 * 1000)  # 200k
MAX_SHIELD_LIFE = 100
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
    # logger.configure(log_path)
    # Disable the logging of py4j
    py4j_logger = logging.getLogger("py4j.java_gateway")
    py4j_logger.setLevel(logging.INFO)


def get_default_parser():
    parser = argparse.ArgumentParser(description='Run the experiment on discrete car racing')
    parser.add_argument('--task_string', type=str, default=TASK_STRING,
                        help='the string repesentation of the task')
    parser.add_argument('--total_steps', type=int, default=TOTAL_STEPS,
                        help='the number of steps in this experiment')
    parser.add_argument('--num', type=int, default=3,
                        help='the number to distinguish this execution')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='the learning rate')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help='gamma')
    return parser
