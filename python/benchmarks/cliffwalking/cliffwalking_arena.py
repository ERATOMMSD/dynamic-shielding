from benchmarks.cliffwalking.cliffwalking_env import CliffWalkingExt
from benchmarks.cliffwalking.cliffwalking_alphabet import CliffWalkingInputOutputManager
from benchmarks.common.generic import reactive_system_from_deterministic_environment

__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "12 April 2021"


def cliffwalking_reactive_system(env=None):
    if not env:
        env = CliffWalkingExt()

    io_manager = CliffWalkingInputOutputManager()

    reactive_system = reactive_system_from_deterministic_environment(env, io_manager)

    return reactive_system





