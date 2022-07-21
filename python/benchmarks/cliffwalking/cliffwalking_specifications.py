__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "12 April 2021"

from benchmarks.cliffwalking.cliffwalking_alphabet import Observations

"""
The shielded specifications in the taxi benchmark.
"""

dont_fall: str = f'G (!{Observations.CLIFF.name})'


def dont_fall_formula():
    return f'({dont_fall})'
