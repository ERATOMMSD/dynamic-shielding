__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "21 October 2020"

from typing import List

"""
The shielded specifications in the grid_world benchmark.
"""

# The following safe1 specification is from AAAI'18 paper
no_crash: str = 'G (CrashPositions.NO_CRASH)'
no_wall: str = 'G (!ArenaPropositions.WALL)'
safe1: str = f'({no_crash}) & ({no_wall})'


# The following is the relaxed no_crash specification since global no_crash is too difficult in noisy situation
def no_crash_duration(duration: int) -> str:
    """
    Args:
        duration: int : the duration with no crash. We assume that duration is positive.
    """
    if duration <= 0:
        raise IndexError(f'duration must be positive (duration: {duration})')

    def add_next(original_formula: str) -> str:
        return f'X({original_formula})'

    formulas: List[str] = []
    for i in range(duration):
        formula = 'CrashPositions.NO_CRASH'
        for j in range(i):
            formula = add_next(formula)
        formulas.append(f'({formula})')
    return ' & '.join(formulas)


def safe1_with_no_crash_duration(duration: int) -> str:
    no_crash_with_duration = no_crash_duration(duration)
    return f'({no_crash_with_duration}) & ({no_wall})'


# The following safe2 specification is only in the arXiv version
safe2: str = 'G(ArenaPropositions.BOMB -> (X(!ArenaPropositions.BOMB)))'

# Both safety
safe: str = f'({safe1}) & ({safe2})'


def safe_with_no_crash_duration(duration: int) -> str:
    return f'({safe1_with_no_crash_duration(duration)}) & ({safe2})'
