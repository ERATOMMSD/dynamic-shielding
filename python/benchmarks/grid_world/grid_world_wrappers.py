from typing import Optional

from py4j.java_gateway import JavaGateway

import benchmarks.grid_world.deterministic_grid_world_dynamic_shield as ds
from benchmarks.grid_world.grid_world_safe_padding import GridWorldSafePadding
from benchmarks.grid_world.grid_world_specifications import safe, safe_with_no_crash_duration, safe2, no_wall
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import PostposedShieldWrapper
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper


def identity(x):
    return x


class GridWorldPostDynamicShieldWrapper(PostposedShieldWrapper):
    """
        Dynamic shielding for grid_world2 and grid_world3 (the current benchmarks)
    """

    def __init__(self, env, gateway: JavaGateway, min_depth: int = 0, punish=False,
                 concurrent_reconstruction: bool = True, max_shield_life: int = 100,
                 pickle_filename: Optional[str] = None, skip_mealy_size: int = 5000):
        shield = ds.GridWorldDynamicShield(
            [safe, safe_with_no_crash_duration(5), safe_with_no_crash_duration(3), safe_with_no_crash_duration(1),
             safe2, no_wall], gateway,
            min_depth=min_depth,
            concurrent_reconstruction=concurrent_reconstruction,
            max_shield_life=max_shield_life, skip_mealy_size=skip_mealy_size)
        if pickle_filename is not None:
            self.shield.load_transition_cover(pickle_filename)
        super().__init__(env=env, shield=shield, punish=punish)


class GridWorldPreDynamicShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, min_depth: int = 0, punish=False,
                 concurrent_reconstruction: bool = True, max_shield_life: int = 100, skip_mealy_size: int = 5000,
                 pickle_filename: Optional[str] = None):
        self.shield = ds.GridWorldDynamicShield(
            [safe, safe_with_no_crash_duration(5), safe_with_no_crash_duration(3), safe_with_no_crash_duration(1),
             safe2, no_wall], gateway,
            min_depth=min_depth,
            concurrent_reconstruction=concurrent_reconstruction,
            max_shield_life=max_shield_life, skip_mealy_size=skip_mealy_size)
        if pickle_filename is not None:
            self.shield.load_transition_cover(pickle_filename)
        super().__init__(env=env, shield=self.shield, punish=punish)


class GridWorldSafePaddingWrapper(SafePaddingWrapper):
    def __init__(self, env):
        self.safe_padding = GridWorldSafePadding(
            ltl_formula=[safe, safe_with_no_crash_duration(5), safe_with_no_crash_duration(3),
                         safe_with_no_crash_duration(1),
                         safe2, no_wall])
        super().__init__(env=env, safe_padding=self.safe_padding)
