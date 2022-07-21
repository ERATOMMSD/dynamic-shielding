from py4j.java_gateway import JavaGateway
from src.wrappers.shield_wrappers import PostposedShieldWrapper
from cliffwalking_dynamic_shield import CliffWalkingDynamicShield
from cliffwalking_alphabet import CliffWalkingInputOutputManager
from cliffwalking_arena import cliffwalking_reactive_system


class CliffWalkingDynamicPostposedShieldWrapper(PostposedShieldWrapper):
    def __init__(self, env, gateway: JavaGateway, ltl_formula: str, min_depth: int = 0, max_shield_life=100, punish=False):
        shield = CliffWalkingDynamicShield(ltl_formula=ltl_formula, gateway=gateway, min_depth=min_depth, max_shield_life=max_shield_life)
        super().__init__(env=env, shield=shield, punish=punish)
