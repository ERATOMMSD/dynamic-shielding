from py4j.java_gateway import JavaGateway

from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper
from benchmarks.water_tank.water_tank_dynamic_shield import WaterTankDynamicShield, WaterTankAdaptiveDynamicShield, \
    WaterTankSafePadding


class WaterTankDynamicPreemptiveShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, ltl_formula: str, min_depth: int = 0, punish=False,
                 max_shield_life=10, no_pdb=True):
        shield = WaterTankDynamicShield(ltl_formula=ltl_formula,
                                        gateway=gateway,
                                        min_depth=min_depth,
                                        max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield, punish=punish, no_pdb=no_pdb)


class WaterTankAdaptiveDynamicPreemptiveShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, ltl_formula: str, punish=False, max_shield_life=100, no_pdb=True):
        shield = WaterTankAdaptiveDynamicShield(ltl_formula=ltl_formula, gateway=gateway,
                                                max_episode_length=env.unwrapped.spec.max_episode_steps,
                                                max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield, punish=punish, no_pdb=no_pdb)


class WaterTankSafePaddingWrapper(SafePaddingWrapper):
    def __init__(self, env, ltl_formula: str):
        safe_padding = WaterTankSafePadding(ltl_formula=ltl_formula)
        super().__init__(env=env, safe_padding=safe_padding)
