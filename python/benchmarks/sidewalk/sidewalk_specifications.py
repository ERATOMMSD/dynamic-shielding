__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "19 September 2021"

from typing import List

"""
The shielded specifications in the sidewalk benchmark.
"""

no_crash: str = 'G (! ArenaPropositions.CRASH)'
no_consecutive_wall: str = 'G (ArenaPropositions.NONE | X(ArenaPropositions.NONE))'
safe: str = f'{no_crash} & {no_consecutive_wall}'


# The following is the relaxed no_crash specification since global no_crash may be too difficult
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
        formula = '! ArenaPropositions.CRASH'
        for j in range(i):
            formula = add_next(formula)
        formulas.append(f'({formula})')
    return ' & '.join(formulas)


SPECIFICATIONS: List[str] = [no_crash, no_crash_duration(5), no_crash_duration(3), no_crash_duration(1)]
