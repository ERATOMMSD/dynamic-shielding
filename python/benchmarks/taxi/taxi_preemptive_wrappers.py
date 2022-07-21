from py4j.java_gateway import JavaGateway

from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper
from benchmarks.taxi.taxi_alphabet import TaxiInputOutputManager
from benchmarks.taxi.taxi_dynamic_shield import TaxiDynamicShield, TaxiAdaptiveDynamicShield, TaxiSafePadding
from benchmarks.taxi.taxi_arena import taxi_reactive_system


# Dynamic Shields
class TaxiDynamicPreemptiveShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, ltl_formula: str,  min_depth: int = 0, punish=False, max_shield_life=100, no_pdb=True):
        shield = TaxiDynamicShield(ltl_formula=ltl_formula, gateway=gateway, min_depth=min_depth, max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield,  punish=punish, no_pdb=no_pdb)


class TaxiAdaptiveDynamicPreemptiveShieldWrapper(PreemptiveShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, ltl_formula: str, punish=False, max_shield_life=100, no_pdb=True):
        shield = TaxiAdaptiveDynamicShield(ltl_formula=ltl_formula, gateway=gateway,
                                           max_episode_length=env.unwrapped.spec.max_episode_steps,
                                           max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield, punish=punish, no_pdb=no_pdb)


# Self-padding shields
class TaxiSafePaddingWrapper(SafePaddingWrapper):
    def __init__(self, env, ltl_formula: str):
        safe_padding = TaxiSafePadding(ltl_formula=ltl_formula)
        super().__init__(env=env, safe_padding=safe_padding)
