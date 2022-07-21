import sys
import argparse

from typing import Union, Type

sys.path.append('../../')
from benchmarks.common.train import run
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper, PostposedShieldWrapper
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper

from benchmarks.taxi.taxi_specifications import drive_safely, safe_and_proactive_formula, proactive_formula
from benchmarks.taxi.taxi_callbacks import TaxiShieldingCallback
from benchmarks.taxi.taxi_preemptive_wrappers import TaxiDynamicPreemptiveShieldWrapper, \
    TaxiAdaptiveDynamicPreemptiveShieldWrapper, TaxiSafePaddingWrapper
from benchmarks.taxi.taxi_postposed_wrappers import TaxiDynamicPostposedShieldWrapper


def setup_parser() -> argparse.ArgumentParser:
    """
        Define the parser of the arguments
    """
    parser = argparse.ArgumentParser(description='evaluates shielding with taxi benchmarks')

    parser.add_argument('--steps', type=int, default=int(2e5),
                        help='number of steps that each environment is run.')
    parser.add_argument('--shield', type=str, default='pre-adaptive',
                        help='the shield to be used [pre-adaptive (default) | pre-dynamic | safe-padding |'
                             'post-dynamic | no]')
    parser.add_argument('--specification', type=str, default='safety',
                        help='the formula used by the shields [safety (default) | proactive | safe-proactive]')
    parser.add_argument('--game', type=str, default='fix',
                        help='the taxi game [fix (default) | center]')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--shield-life', type=int, default=100,
                        help='frequency of shield reconstruction in terms of episodes.')
    parser.add_argument('--depths', nargs='+', default=[1, 3, 5, 7],
                        help='a list of min-depths for dynamic shield only (usage: --depths 0 1 3)')
    parser.add_argument('--penalties', nargs='+', default=[0.0, 1.0, 10.0, 100.0],
                        help='a list of penalties for no shield only (usage: --penalties 0.0 1.0 100.0)')
    return parser


def get_shield(shield) -> Union[None, Type[PreemptiveShieldWrapper],
                                Type[SafePaddingWrapper], Type[PostposedShieldWrapper]]:
    if shield == 'pre-dynamic':
        return TaxiDynamicPreemptiveShieldWrapper
    elif shield == 'pre-adaptive':
        return TaxiAdaptiveDynamicPreemptiveShieldWrapper
    elif shield == 'safe-padding':
        return TaxiSafePaddingWrapper
    elif shield == 'post-dynamic':
        return TaxiDynamicPostposedShieldWrapper
    elif shield == 'no':
        return None
    else:
        raise RuntimeError(f'Unknown shields {shield}')


def get_specification(specification) -> str:
    if specification == 'safety':
        return drive_safely()
    elif specification == 'proactive':
        return safe_and_proactive_formula()
    elif specification == 'safe-proactive':
        return proactive_formula()
    else:
        raise RuntimeError(f'Unknown specification {specification}')


def get_game(game) -> str:
    if game == 'fix':
        return 'TaxiFixStart-v3'
    elif game == 'center':
        return 'TaxiStartCenter-v3'
    else:
        raise RuntimeError(f'Unknown shields {game}')


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    steps = args.steps
    learning_rate = args.learning_rate
    shield = get_shield(args.shield)
    game = get_game(args.game)
    depths = list(map(int, args.depths))
    shield_life = args.shield_life
    penalties = list(map(float, args.penalties))
    specification = get_specification(args.specification)

    run(shield=shield,
        ltl_formula=specification,
        depths=depths,
        game=game,
        callback=TaxiShieldingCallback,
        penalties=penalties,
        total_steps=steps,
        learning_rate=learning_rate,
        shield_life=shield_life)

    sys.exit(0)
