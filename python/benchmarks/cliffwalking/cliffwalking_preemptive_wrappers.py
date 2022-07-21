from py4j.java_gateway import JavaGateway

from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper
from benchmarks.cliffwalking.cliffwalking_dynamic_shield import CliffWalkingDynamicShield, \
    CliffWalkingAdaptiveDynamicShield, CliffWalkingSafePadding
from benchmarks.cliffwalking.cliffwalking_alphabet import CliffWalkingInputOutputManager
from benchmarks.cliffwalking.cliffwalking_arena import cliffwalking_reactive_system


class CliffWalkingDynamicPreemptiveShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, ltl_formula: str, min_depth: int = 0, punish=False,
                 max_shield_life=100, no_pdb=True):
        shield = CliffWalkingDynamicShield(ltl_formula=ltl_formula, gateway=gateway, min_depth=min_depth,
                                           max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield, punish=punish, no_pdb=no_pdb)


class CliffWalkingAdaptiveDynamicPreemptiveShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, ltl_formula: str, punish=False, max_shield_life=100, no_pdb=True):
        shield = CliffWalkingAdaptiveDynamicShield(ltl_formula=ltl_formula, gateway=gateway,
                                                   max_episode_length=env.unwrapped.spec.max_episode_steps,
                                                   max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield, punish=punish, no_pdb=no_pdb)


class CliffWalkingSafePaddingWrapper(SafePaddingWrapper):
    def __init__(self, env, ltl_formula: str):
        safe_padding = CliffWalkingSafePadding(ltl_formula=ltl_formula)
        super().__init__(env=env, safe_padding=safe_padding)
