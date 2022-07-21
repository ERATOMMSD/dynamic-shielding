from typing import Optional

from py4j.java_gateway import JavaGateway

from benchmarks.sidewalk.sidewalk_dynamic_shield import SidewalkDynamicShield
from benchmarks.sidewalk.sidewalk_safe_padding import SidewalkSafePadding
from benchmarks.sidewalk.sidewalk_specifications import SPECIFICATIONS
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper


def identity(x):
    return x


class SidewalkPreDynamicShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, min_depth: int = 0, punish=False, max_shield_life: int = 100,
                 pickle_filename: Optional[str] = None):
        if hasattr(env, 'num_seeds'):
            num_seeds = env.num_seeds
        else:
            num_seeds = 1
        self.shield = SidewalkDynamicShield(SPECIFICATIONS, gateway, min_depth=min_depth,
                                            max_shield_life=max_shield_life, num_seeds=num_seeds)
        if pickle_filename is not None:
            self.shield.load_transition_cover(pickle_filename)
        super().__init__(env=env, shield=self.shield, punish=punish)


class SidewalkSafePaddingWrapper(SafePaddingWrapper):
    def __init__(self, env):
        if hasattr(env, 'num_seeds'):
            num_seeds = env.num_seeds
        else:
            num_seeds = 1
        self.safe_padding = SidewalkSafePadding(SPECIFICATIONS, num_seeds=num_seeds)
        super().__init__(env=env, safe_padding=self.safe_padding)
