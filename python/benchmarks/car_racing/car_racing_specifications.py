from typing import List

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "3 August 2021"

"""
The shielded specifications in the car_racing benchmark.
"""

# We should no enter the grass area
no_grass: str = 'G (ArenaProposition.NORMAL)'
# We should no enter the grass area
no_consecutive_grass: str = 'G (ArenaProposition.NORMAL || X(ArenaProposition.NORMAL))'


# The following is the relaxed no_grass specification
def no_grass_duration(duration: int) -> str:
    """
    Args:
        duration: int : the duration with no grass. We assume that duration is positive.
    """
    if duration <= 0:
        raise IndexError(f'duration must be positive (duration: {duration})')

    def add_next(original_formula: str) -> str:
        return f'X({original_formula})'

    formulas: List[str] = []
    for i in range(duration):
        formula = 'ArenaProposition.NORMAL'
        for j in range(i):
            formula = add_next(formula)
        formulas.append(f'({formula})')
    return ' & '.join(formulas)


# The following is the relaxed no_consecutive_grass specification
def no_consecutive_grass_duration(duration: int) -> str:
    """
    Args:
        duration: int : the duration with no grass. We assume that duration is positive.
    """
    if duration <= 0:
        raise IndexError(f'duration must be positive (duration: {duration})')

    def add_next(original_formula: str) -> str:
        return f'X({original_formula})'

    formulas: List[str] = []
    for i in range(duration):
        formula = '(ArenaProposition.NORMAL || X(ArenaProposition.NORMAL))'
        for j in range(i):
            formula = add_next(formula)
        formulas.append(f'({formula})')
    return ' & '.join(formulas)
