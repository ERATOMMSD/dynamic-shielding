import gym
from benchmarks.taxi.taxi_alphabet import TaxiInputOutputManager
from benchmarks.taxi.taxi_ext import TaxiExt
from benchmarks.common.generic import reactive_system_from_deterministic_environment

__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "08 April 2021"


def taxi_reactive_system(env=None):
    if not env:
        env = TaxiExt()

    io_manager = TaxiInputOutputManager()

    reactive_system = reactive_system_from_deterministic_environment(env.unwrapped, io_manager)

    return reactive_system




