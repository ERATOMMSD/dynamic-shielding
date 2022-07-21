from py4j.java_gateway import JavaGateway

from src.wrappers.shield_wrappers import PostposedShieldWrapper
from taxi_dynamic_shield import TaxiDynamicShield


# Dynamic Shield
class TaxiDynamicPostposedShieldWrapper(PostposedShieldWrapper):
    def __init__(self, env, ltl_formula: str,  gateway: JavaGateway, min_depth: int = 0, punish=False, max_shield_life=100):
        shield = TaxiDynamicShield(ltl_formula=ltl_formula, gateway=gateway, min_depth=min_depth, max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield,  punish=punish)
