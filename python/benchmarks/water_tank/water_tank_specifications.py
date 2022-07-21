__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "05 November 2020"

from benchmarks.water_tank.water_tank_arena import Observations

"""
The shielded specifications in the water tank benchmark.
"""

# The following safe specification is from arxiv paper
not_empty: str = f'G (!{Observations.EMPTY})'
not_full: str = f'G (!{Observations.FULL})'
keep_open: str = f'G (({Observations.OPEN} ∧ X !{Observations.OPEN}) → XX !{Observations.OPEN} ∧ XXX !{Observations.OPEN})'
keep_closed: str = f'G ((!{Observations.OPEN} ∧ X {Observations.OPEN}) → XX {Observations.OPEN} ∧ XXX {Observations.OPEN})'


def safety_formula():
    return f'({not_empty}) ∧ ({not_full}) ∧ ({keep_open}) ∧ ({keep_closed})'
