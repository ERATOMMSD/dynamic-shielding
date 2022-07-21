__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "08 April 2021"

from benchmarks.taxi.taxi_alphabet import Observations

"""
The shielded specifications in the taxi benchmark.
"""

dont_crash: str = f'G (!{Observations.SCRATCH.name} ∧ !{Observations.BROKEN.name})'
use_door_carefully: str = f'G (!{Observations.WRONG_PICKUP.name} ∧ !{Observations.WRONG_DROPOFF.name})'
pick_at_pickup: str = f'G ({Observations.ARRIVED_PICKUP.name} → X {Observations.SUCCESSFUL_PICKUP.name})'
drop_at_dropoff: str = f'G ({Observations.ARRIVED_DROPOFF.name} → X {Observations.SUCCESSFUL_DROPOFF.name})'


def drive_safely():
    return f'({dont_crash}) ∧ ({use_door_carefully})'


def safe_and_proactive_formula():
    return f'({dont_crash}) ∧ ({use_door_carefully}) ∧  ({pick_at_pickup}) ∧ ({drop_at_dropoff})'


def proactive_formula():
    return f'({pick_at_pickup}) ∧ ({drop_at_dropoff})'
